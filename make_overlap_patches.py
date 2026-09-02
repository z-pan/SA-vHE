#!/usr/bin/env python3
"""Cut a test FOV into **overlapping** model-scale windows so the stitcher can feather them.

Why
---
The 250711_slides patches are adjacent, not overlapping: a 204x204 tile at stride
204, upscaled 2.51x to 512. Every tile is inferred on its own, so each one carries
its own edge effects (padding, truncated receptive field) and its own InstanceNorm
statistics. Where two tiles abut, those two independent errors collide with nothing
in between, which is the block grid visible in the stitched FOV. Per-tile offset or
gain levelling barely dents it (seam ratio 3.19 -> 2.92 against 1.17 for real H&E)
because the mismatch is not constant within a tile.

The fix is to stop letting tiles meet at a hard line: infer on windows that
overlap, then cross-fade them. This script produces those windows.

Geometry
--------
Same convention as the existing patches, verified against them exactly (BICUBIC,
zero difference): crop a ``--tile`` px square from the original image and upscale
it to ``--patch`` px. Only the stride changes, from ``tile`` to ``--stride``.
Coordinates in the filename stay in **original** pixels, so the existing stitcher
reads them unchanged -- pass it ``--tile_size 204``, since it can no longer infer
the tile from the coordinate stride.

The covered region is taken from an existing abutting patch set
(``--grid_ref_dir``) so the output covers exactly the same area as the baseline and
the two stitches are directly comparable.

Nuclear masks
-------------
Masks were segmented per 512 px patch, so there is no full-FOV mask to crop from.
This pastes the per-patch masks onto a model-scale canvas and cuts windows out of
that. Windows aligned with the old grid get a byte-identical mask; windows
straddling a boundary inherit whatever mismatch the two independent segmentations
left there. Check ``--report`` output: if the input windows themselves show a seam,
re-segment the FOV canvas instead of pasting.

The nuclei enhancement reproduces ``enhance_gray_TPAF_nuclei_with_instance_mask.py``
exactly -- binarise, 11x11 Gaussian, subtract 158, add -- but computes the blur on
the canvas, so it has no per-patch blur edges.

Usage
-----
Model input windows (gray TPAF, nuclei-enhanced) plus the matching binary masks::

    python make_overlap_patches.py \\
        --full_dir       datasets/test_data/250711_slides/01_og_TPAF_gray_full \\
        --grid_ref_dir   datasets/test_data/250711_slides/02_og_TPAF_gray_patches \\
        --mask_patch_dir datasets/test_data/250711_slides/03_og_TPAF_gray_patches_nuc_mask \\
        --out_dir        datasets/test_data/250711_slides/07_overlap_gray_nuc_hi_patches \\
        --mask_out_dir   datasets/test_data/250711_slides/07_overlap_nuc_mask_bin \\
        --enhance --stride 102

Overlapping real-H&E windows, for round-tripping the stitcher (no mask, no
enhancement) -- stitching these back should return the reference at SSIM > 0.95::

    python make_overlap_patches.py \\
        --full_dir     datasets/test_data/250711_slides/00_og_real_HE_full \\
        --grid_ref_dir datasets/test_data/250711_slides/01_og_real_HE_patches \\
        --out_dir      datasets/test_data/250711_slides/07_overlap_real_HE_patches \\
        --stride 102
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
_COORD = re.compile(r"^(?P<fov>.*)_x(?P<x>\d+)_y(?P<y>\d+)$")


def grid_geometry(ref_dir: Path) -> dict[str, tuple[int, int, int]]:
    """Per FOV: (tile, region_w, region_h) read off an abutting patch set."""
    coords: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for p in sorted(ref_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        m = _COORD.match(p.stem)
        if m:
            coords[m.group("fov")].append((int(m.group("x")), int(m.group("y"))))

    out = {}
    for fov, items in coords.items():
        strides = []
        for axis in (0, 1):
            vals = sorted({it[axis] for it in items})
            strides += [b - a for a, b in zip(vals, vals[1:]) if b > a]
        tile = min(strides) if strides else 0
        out[fov] = (tile,
                    max(x for x, _ in items) + tile,
                    max(y for _, y in items) + tile)
    return out


def mask_canvas(mask_dir: Path, fov: str, tile: int, patch: int,
                region_w: int, region_h: int) -> np.ndarray:
    """Binary nuclear mask for the whole FOV at model scale, pasted from patches."""
    scale = patch // tile if patch % tile == 0 else patch / tile
    canvas = np.zeros((int(round(region_h * patch / tile)),
                       int(round(region_w * patch / tile))), np.uint8)
    n = 0
    for p in sorted(mask_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        m = _COORD.match(p.stem)
        if not m or m.group("fov") != fov:
            continue
        x, y = int(m.group("x")), int(m.group("y"))
        mx, my = round(x * patch / tile), round(y * patch / tile)
        arr = np.array(Image.open(p))
        canvas[my:my + arr.shape[0], mx:mx + arr.shape[1]] = np.where(arr > 0, 255, 0)
        n += 1
    if n == 0:
        raise RuntimeError(f"no mask patches matched FOV {fov}")
    del scale
    return canvas


def enhance(gray: np.ndarray, mask_win: np.ndarray) -> tuple[np.ndarray, int]:
    """Additive nuclei boost, identical to the per-patch script (uint8 wrap and all)."""
    soft = cv2.GaussianBlur(mask_win, (11, 11), 0)
    soft = np.where(soft > 158, soft - 158, 0).astype(np.uint8)
    over = int((gray.astype(np.int32) + soft.astype(np.int32) > 255).sum())
    return (gray + soft), over


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cut overlapping model-scale windows from full test FOVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--full_dir", required=True, help="Full-size source images, one per FOV.")
    ap.add_argument("--out_dir", required=True, help="Where the windows are written.")
    ap.add_argument("--grid_ref_dir", default=None,
                    help="Existing abutting patch set; supplies tile size and covered "
                         "region so output matches the baseline area.")
    ap.add_argument("--tile", type=int, default=None,
                    help="Original-image crop size. Default: inferred from --grid_ref_dir.")
    ap.add_argument("--region", type=int, default=None,
                    help="Square region to cover, in original px. Default: from --grid_ref_dir.")
    ap.add_argument("--stride", type=int, default=102,
                    help="Step between windows in original px. Half the tile = 50%% overlap.")
    ap.add_argument("--patch", type=int, default=512, help="Output window size in px.")
    ap.add_argument("--mask_patch_dir", default=None,
                    help="Per-patch nuclear masks, pasted into a FOV canvas and re-cut.")
    ap.add_argument("--mask_out_dir", default=None,
                    help="Where the per-window binary masks are written.")
    ap.add_argument("--enhance", action="store_true",
                    help="Apply the additive nuclei boost (needs --mask_patch_dir).")
    ap.add_argument("--suffix", default=None,
                    help="Appended to window names. Default '_nuc_enhanced' with "
                         "--enhance, empty otherwise.")
    args = ap.parse_args()

    if args.enhance and not args.mask_patch_dir:
        sys.exit("--enhance needs --mask_patch_dir")
    suffix = args.suffix if args.suffix is not None else ("_nuc_enhanced" if args.enhance else "")

    full_dir = Path(args.full_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_out = Path(args.mask_out_dir) if args.mask_out_dir else None
    if mask_out:
        mask_out.mkdir(parents=True, exist_ok=True)

    geom = grid_geometry(Path(args.grid_ref_dir)) if args.grid_ref_dir else {}
    fulls = {p.stem: p for p in sorted(full_dir.iterdir())
             if p.is_file() and p.suffix.lower() in _EXT}
    if not fulls:
        sys.exit(f"No images in {full_dir}")

    total = overflow = 0
    for fov, path in fulls.items():
        tile, region_w, region_h = geom.get(fov, (args.tile, args.region, args.region))
        tile = args.tile or tile
        if args.region:
            region_w = region_h = args.region
        if not tile or not region_w:
            print(f"  [warn] no geometry for {fov[:50]}; give --tile/--region", file=sys.stderr)
            continue
        if (args.stride * args.patch) % tile:
            sys.exit(f"stride {args.stride} does not map to whole model pixels "
                     f"(stride*patch/tile must be an integer)")

        img = Image.open(path)
        canvas = (mask_canvas(Path(args.mask_patch_dir), fov, tile, args.patch,
                              region_w, region_h)
                  if args.mask_patch_dir else None)

        xs = list(range(0, region_w - tile + 1, args.stride))
        ys = list(range(0, region_h - tile + 1, args.stride))
        for y in ys:
            for x in xs:
                win = np.array(img.crop((x, y, x + tile, y + tile))
                                  .resize((args.patch, args.patch), Image.BICUBIC))
                name = f"{fov}_x{x}_y{y}"
                if canvas is not None:
                    mx, my = x * args.patch // tile, y * args.patch // tile
                    mwin = canvas[my:my + args.patch, mx:mx + args.patch]
                    if mask_out:
                        Image.fromarray(mwin).save(mask_out / f"{name}.png")
                    if args.enhance:
                        win, over = enhance(win, mwin)
                        overflow += over
                Image.fromarray(win).save(out_dir / f"{name}{suffix}.png")
                total += 1
        print(f"  {fov[:54]:<54} tile {tile} region {region_w}x{region_h} "
              f"-> {len(xs)}x{len(ys)} = {len(xs) * len(ys)} windows")

    print(f"\n{total} windows (stride {args.stride}, tile {args.tile or 'auto'}, "
          f"{args.patch}px) -> {out_dir}")
    if mask_out:
        print(f"masks -> {mask_out}")
    if overflow:
        print(f"[warn] {overflow} px overflowed uint8 in the additive boost and wrapped "
              f"dark; the per-patch script has the same behaviour", file=sys.stderr)


if __name__ == "__main__":
    main()
