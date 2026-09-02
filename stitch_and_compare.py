#!/usr/bin/env python3
"""Stitch virtually-stained patches back into whole FOVs and score them against real H&E.

Geometry is read from the filenames rather than assumed. Patches are named
``<fov>_x<X>_y<Y>.png``, where X and Y are coordinates **in the original image**.

For the 250711_slides set, template-matching a patch back into its source FOV
(correlation 0.98-0.99, position exact) shows the extraction was: take a
**204x204 tile** at stride 204 -- adjacent, *not* overlapping -- and upscale it
2.51x to 512x512. So stitching means resizing each patch back down to the tile
size and laying the tiles side by side; the 4x4 grid then covers the top-left
816x816 of the FOV.

The tile size is inferred from the smallest coordinate stride, so this adapts to
other patch sets. Use ``--no_rescale`` for the other convention, where patches are
native-resolution crops rather than upscaled tiles.

Patches that overlap are feather-blended, in either convention. Abutting tiles are
what produce the block grid in the first place -- each is inferred alone, so two
independent sets of edge effects meet at a hard line. Cutting overlapping windows
with ``make_overlap_patches.py``, inferring those, and stitching them here with
``--tile_size 204`` cross-fades the collision away.

TPAF and real H&E are registered and share a size per FOV, so with
``--reference_dir`` the stitched result is compared against the matching region of
the real H&E, pixel for pixel.

Usage
-----
Stitch only::

    python stitch_and_compare.py \\
        --patch_dir  results/vH_denoised \\
        --output_dir results/stitched_vH

Stitch, align to the real H&E, and score::

    python stitch_and_compare.py \\
        --patch_dir  results/vHE_patches \\
        --output_dir results/stitched_vHE \\
        --reference_dir datasets/test_data/250711_slides/00_og_real_HE_full \\
        --compare

Sanity check the stitcher itself by round-tripping the real H&E patches: the
result should match the reference almost exactly (SSIM > 0.95). Anything lower
means the geometry or blending is wrong, not the model::

    python stitch_and_compare.py \\
        --patch_dir  datasets/test_data/250711_slides/01_og_real_HE_patches \\
        --output_dir results/stitched_selftest \\
        --reference_dir datasets/test_data/250711_slides/00_og_real_HE_full \\
        --compare

Stitch overlapping windows, where the stride is no longer the tile size::

    python stitch_and_compare.py \\
        --patch_dir  results/vHE_overlap_corrected_masked \\
        --output_dir results/stitched_vHE_overlap \\
        --tile_size 204 --strip _nuc_enhanced_fake_B \\
        --reference_dir datasets/test_data/250711_slides/00_og_real_HE_full \\
        --compare

Score the seams with ``measure_seam.py``.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
_COORD = re.compile(r"^(?P<fov>.*)_x(?P<x>\d+)_y(?P<y>\d+)$")


def tukey_window(h: int, w: int, taper: int) -> np.ndarray:
    """Flat-topped 2-D window with cosine edges, for seamless overlap blending.

    A plain Hann window would taper to zero at the canvas border, where only one
    patch contributes and there is nothing to blend with. Tukey stays at 1.0
    across the interior and only ramps over *taper* px, so border pixels keep
    full weight while overlaps cross-fade.
    """
    def ramp(n: int) -> np.ndarray:
        w1 = np.ones(n, dtype=np.float32)
        t = max(1, min(taper, n // 2))
        edge = 0.5 * (1 - np.cos(np.linspace(0, np.pi, t + 2)[1:-1]))
        w1[:t], w1[-t:] = edge, edge[::-1]
        return w1
    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def group_patches(patch_dir: Path, strip: str) -> dict[str, list[tuple[int, int, Path]]]:
    """Map FOV name → [(x, y, path)], parsed from ``*_x<X>_y<Y>`` filenames."""
    groups: dict[str, list] = collections.defaultdict(list)
    skipped = 0
    for p in sorted(patch_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        stem = p.stem.replace(strip, "") if strip else p.stem
        m = _COORD.match(stem)
        if not m:
            skipped += 1
            continue
        groups[m.group("fov")].append((int(m.group("x")), int(m.group("y")), p))
    if skipped:
        print(f"  [warn] {skipped} files had no _x<X>_y<Y> suffix and were skipped",
              file=sys.stderr)
    return groups


def infer_tile(items: list[tuple[int, int, Path]]) -> int:
    """Smallest positive coordinate stride = the tile size in original pixels."""
    strides = []
    for axis in (0, 1):
        vals = sorted({it[axis] for it in items})
        strides += [b - a for a, b in zip(vals, vals[1:]) if b > a]
    return min(strides) if strides else 0


def stitch(items: list[tuple[int, int, Path]], taper: int | None,
           tile: int | None, level: bool = False) -> np.ndarray:
    """Lay patches onto a canvas in original-image coordinates.

    With *tile* set, each patch is resized to ``tile`` px (the upscaled-tile
    convention); without it, patches are placed at native size.

    Either way, whether patches are feather-blended is decided by the geometry
    rather than by the mode: if the coordinate stride is smaller than a patch's
    footprint they overlap and get a Tukey cross-fade, otherwise they abut and are
    laid down flat. Overlapping upscaled tiles are the point of
    ``make_overlap_patches.py`` -- pass ``--tile_size`` explicitly for those, since
    the stride no longer equals the tile.
    """
    first = np.array(Image.open(items[0][2]))
    bands = 1 if first.ndim == 2 else first.shape[2]
    ph = pw = tile if tile else first.shape[0]
    if not tile:
        ph, pw = first.shape[:2]

    if level and tile:
        loaded = {}
        for x, y, path in items:
            im = Image.open(path)
            if im.size != (pw, ph):
                im = im.resize((pw, ph), Image.BICUBIC)
            loaded[(x, y)] = np.array(im).astype(np.float32)
        loaded = level_tiles(loaded, tile)

    ch = max(y for _, y, _ in items) + ph
    cw = max(x for x, _, _ in items) + pw
    acc = np.zeros((ch, cw, bands), np.float32)
    wsum = np.zeros((ch, cw, 1), np.float32)
    # Abutting patches need no cross-fade; overlapping ones do. Default the ramp to
    # the full overlap, which is the widest cross-fade the geometry allows.
    step = infer_tile(items)
    overlap = min(ph, pw) - step if 0 < step < min(ph, pw) else 0
    win = (tukey_window(ph, pw, taper if taper is not None else overlap)[:, :, None]
           if overlap else np.ones((ph, pw, 1), np.float32))

    for x, y, path in items:
        if level and tile:
            a = loaded[(x, y)]
        else:
            im = Image.open(path)
            if tile and im.size != (pw, ph):
                im = im.resize((pw, ph), Image.BICUBIC)
            a = np.array(im).astype(np.float32)
        if a.ndim == 2:
            a = a[:, :, None]
        acc[y:y + ph, x:x + pw] += a * win
        wsum[y:y + ph, x:x + pw] += win

    out = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
    return out[:, :, 0] if bands == 1 else out


def level_tiles(tiles: dict[tuple[int, int], np.ndarray], tile: int
                ) -> dict[tuple[int, int], np.ndarray]:
    """Remove per-tile brightness offsets by making shared borders agree.

    Each patch is inferred independently, and InstanceNorm normalises every patch
    by its own statistics, so neighbouring tiles come out at slightly different
    overall levels. Pasted together that shows up as a visible grid of blocks --
    a stitching artifact, not a model one.

    Adjacent tiles abut, so the last row/column of one and the first of the next
    sample almost the same tissue and should match. This solves least squares for
    a per-tile, per-channel offset minimising those border mismatches, pinned to
    zero mean so overall brightness is unchanged.

    Measured effect is small. Seam ratio from ``measure_seam.py`` (mean cross-border
    jump / mean jump 8 px inside, so 1.0 means no seam) over the seven
    250711_slides FOVs:

        no levelling                3.35
        + offset                    3.19   <- this function
        + log-space gain                   (tested, worse)
        overlapping windows         1.26
        real H&E                    1.01

    So the block grid is *not* mainly a per-tile constant offset or gain: it varies
    within each tile, which points at edge effects in the generator (padding,
    truncated receptive field) plus InstanceNorm statistics that depend on tile
    content. Cutting **overlapping** windows with ``make_overlap_patches.py``,
    inferring those, and feather-blending them here removes 89% of the excess over
    the real-H&E floor and is the actual fix. Keep this only as a cosmetic touch-up
    for patch sets that cannot be re-inferred.
    """
    keys = sorted(tiles)
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    if n < 2:
        return tiles
    bands = 1 if tiles[keys[0]].ndim == 2 else tiles[keys[0]].shape[2]
    out = {k: v.astype(np.float32).copy() for k, v in tiles.items()}

    for b in range(bands):
        rows, rhs = [], []
        for (x, y) in keys:
            cur = tiles[(x, y)]
            cur_b = cur if cur.ndim == 2 else cur[:, :, b]
            for dx, dy in ((tile, 0), (0, tile)):
                nb = tiles.get((x + dx, y + dy))
                if nb is None:
                    continue
                nb_b = nb if nb.ndim == 2 else nb[:, :, b]
                if dx:                      # right neighbour: compare columns
                    diff = float(nb_b[:, 0].mean() - cur_b[:, -1].mean())
                else:                       # below: compare rows
                    diff = float(nb_b[0, :].mean() - cur_b[-1, :].mean())
                r = np.zeros(n, np.float64)
                r[idx[(x + dx, y + dy)]] = 1.0
                r[idx[(x, y)]] = -1.0
                rows.append(r)
                rhs.append(-diff)           # want offset_nb - offset_cur = -diff
        if not rows:
            continue
        anchor = np.ones((1, n))            # pin the mean offset to zero
        A = np.vstack(rows + [anchor])
        y_vec = np.array(rhs + [0.0])
        offsets, *_ = np.linalg.lstsq(A, y_vec, rcond=None)
        for k in keys:
            if out[k].ndim == 2:
                out[k] += offsets[idx[k]]
            else:
                out[k][:, :, b] += offsets[idx[k]]

    return {k: np.clip(v, 0, 255) for k, v in out.items()}


def find_reference(ref_dir: Path, fov: str, rules: list[tuple[str, str]]) -> Path | None:
    """Locate the reference image for *fov*, applying ``old=new`` name rules."""
    names = {p.stem: p for p in ref_dir.iterdir()
             if p.is_file() and p.suffix.lower() in _EXT}
    for old, new in rules:
        cand = fov.replace(old, new)
        if cand in names:
            return names[cand]
    if fov in names:
        return names[fov]
    # Last resort: match on the shared FOV id (e.g. "Line-1_0002").
    core = re.search(r"Line-?\d+[_-]\d+", fov)
    if core:
        for stem, path in names.items():
            if core.group(0) in stem:
                return path
    return None


def score(virtual: np.ndarray, real: np.ndarray) -> dict:
    """Structural and colour agreement between an aligned virtual/real pair."""
    from skimage.metrics import structural_similarity as ssim
    v = virtual if virtual.ndim == 2 else virtual.mean(2)
    r = real if real.ndim == 2 else real.mean(2)
    out = {"ssim": float(ssim(v, r, data_range=255))}
    if virtual.ndim == 3 and real.ndim == 3:
        tis_v, tis_r = virtual.mean(2) < 235, real.mean(2) < 235
        if tis_v.sum() > 500 and tis_r.sum() > 500:
            mv, mr = virtual[tis_v].mean(0), real[tis_r].mean(0)
            out["virtual_RmB"] = float(mv[0] - mv[2])
            out["real_RmB"] = float(mr[0] - mr[2])
            G, R, B = virtual[..., 1], virtual[..., 0], virtual[..., 2]
            out["green_pct"] = 100.0 * float(((G > R) & (G > B) & tis_v).sum()) / float(tis_v.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stitch virtual-stain patches into FOVs and compare with real H&E.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--patch_dir", required=True, help="Directory of *_x<X>_y<Y> patches.")
    ap.add_argument("--output_dir", required=True, help="Where stitched FOVs are written.")
    ap.add_argument("--strip", default="",
                    help="Substring removed from filenames before parsing coordinates "
                         "(e.g. '_fake_B').")
    ap.add_argument("--taper", type=int, default=None,
                    help="Cosine blend width, in the same units the patches are laid "
                         "down in. Ignored when patches abut. Default: the full "
                         "overlap width.")
    ap.add_argument("--tile_size", type=int, default=None,
                    help="Original-image tile size each patch represents. Default: "
                         "inferred from the coordinate stride, which is only right "
                         "for abutting patches (204 here) -- pass it explicitly for "
                         "overlapping ones.")
    ap.add_argument("--level_tiles", action="store_true",
                    help="Remove per-tile brightness offsets so the block grid "
                         "from independent per-patch inference disappears.")
    ap.add_argument("--no_rescale", action="store_true",
                    help="Treat patches as native-resolution overlapping crops "
                         "instead of upscaled tiles.")
    ap.add_argument("--reference_dir", default=None,
                    help="Real H&E full images. Enables resize-back so the stitched "
                         "result aligns with the reference.")
    ap.add_argument("--name_rule", action="append", default=None, metavar="OLD=NEW",
                    help="Filename rewrite used to find the reference "
                         "(default: '_AF=_HE_reg'). Repeatable.")
    ap.add_argument("--compare", action="store_true",
                    help="Write a side-by-side figure and print SSIM / colour scores.")
    args = ap.parse_args()

    rules = [tuple(r.split("=", 1)) for r in (args.name_rule or ["_AF=_HE_reg"])]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.reference_dir) if args.reference_dir else None

    groups = group_patches(Path(args.patch_dir), args.strip)
    if not groups:
        sys.exit(f"No *_x<X>_y<Y> patches found in {args.patch_dir}")
    print(f"{len(groups)} FOV(s), {sum(len(v) for v in groups.values())} patches, "
          f"taper={args.taper if args.taper is not None else 'auto'}")

    rows = []
    for fov, items in sorted(groups.items()):
        tile = None if args.no_rescale else (args.tile_size or infer_tile(items))
        img = stitch(items, args.taper, tile, args.level_tiles)
        ref_img = None
        if ref_dir is not None:
            ref_path = find_reference(ref_dir, fov, rules)
            if ref_path is None:
                print(f"  [warn] no reference for {fov[:50]}", file=sys.stderr)
            else:
                ref_full = np.array(Image.open(ref_path).convert(
                    "L" if img.ndim == 2 else "RGB"))
                # The grid covers only part of the FOV; compare like for like.
                ref_img = ref_full[:img.shape[0], :img.shape[1]]
                if ref_img.shape[:2] != img.shape[:2]:
                    img = np.array(Image.fromarray(img).resize(
                        (ref_img.shape[1], ref_img.shape[0]), Image.BICUBIC))

        Image.fromarray(img).save(out_dir / f"{fov}.png")
        note = f"{len(items):2d} patches"
        note += f" (tile {tile}px)" if tile else " (native)"
        note += f" -> {img.shape[1]}x{img.shape[0]}"

        if args.compare and ref_img is not None:
            s = score(img, ref_img)
            rows.append((fov, s))
            note += f"   SSIM={s['ssim']:.3f}"
            if "green_pct" in s:
                note += f"  R-B {s['virtual_RmB']:+.1f} (real {s['real_RmB']:+.1f})" \
                        f"  green {s['green_pct']:.2f}%"
            pair = np.concatenate(
                [img if img.ndim == 3 else np.stack([img] * 3, -1),
                 ref_img if ref_img.ndim == 3 else np.stack([ref_img] * 3, -1)], axis=1)
            (out_dir / "side_by_side").mkdir(exist_ok=True)
            Image.fromarray(pair).save(out_dir / "side_by_side" / f"{fov}.png")
        print(f"  {fov[:54]:<54} {note}")

    print(f"\nstitched -> {out_dir}")
    if rows:
        ss = [r[1]["ssim"] for r in rows]
        print(f"side-by-side (virtual | real) -> {out_dir / 'side_by_side'}")
        print(f"\nSSIM mean {np.mean(ss):.3f}   min {np.min(ss):.3f}   max {np.max(ss):.3f}")
        if "green_pct" in rows[0][1]:
            print(f"green-dominant pixels: mean "
                  f"{np.mean([r[1]['green_pct'] for r in rows]):.2f} %")


if __name__ == "__main__":
    main()
