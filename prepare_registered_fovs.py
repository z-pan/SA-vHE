#!/usr/bin/env python3
"""Turn the raw registered TPAF/H&E pairs into a tiled-ready test set, reproducibly.

Why
---
``250711_slides`` was built by hand: someone opened each 1024x1024 registered pair
in Fiji, dragged a rectangle that missed the black warp border, cropped, and did
that for 7 of the 177 pairs. There is no script, no manifest, and no record of why
those 7. That does not scale to the full slide, and it cannot be re-run.

This script replaces both manual steps with a stated rule.

The crop rule
-------------
``notebook_AF_HE_registration.ipynb`` runs with ``crop_border(..., border=0)``, so
the warp leaves a black border inside ``*_HE_reg.tif`` (page 0). The crop must
exclude it. Downstream everything is cut on a ``--tile`` px grid, so a crop whose
size is not a whole number of tiles just throws away its remainder. Hence:

    the largest ``tile``-aligned rectangle containing zero border pixels,
    ties broken by the most tissue.

"Largest" is in tiles (ny, nx), preferring the bigger area and then the squarer
shape, so the choice never depends on scan order. Both images get the same
rectangle. Sizes come out as exact multiples of ``--tile``, which lets
``make_overlap_patches.py`` read the region straight off the image.

Selection
---------
Every FOV is measured and written to ``manifest.csv`` whether or not it is kept;
``--min_tissue`` / ``--min_grid`` only decide the ``kept`` column and which images
get written. Re-running with different thresholds does not need a re-measure --
pass ``--manifest_only`` first, look at the numbers, then commit.

Registration accuracy is deliberately *not* a criterion. The metrics this dataset
feeds (per-compartment R-B, green fraction, seam ratio) compare distributions, not
pixels, and an NMI shift-peak test over a 19-FOV sample put 17 peaks exactly at
(0,0) with the other two flat to within 4e-4. What does vary, a lot, is how much
tissue is in frame -- 0.10 to 0.78 across the same sample.

Usage
-----
::

    python prepare_registered_fovs.py --reg_dir <.../registered> \\
        --out_root datasets/test_data/260815_slides_full --manifest_only

    python prepare_registered_fovs.py --reg_dir <.../registered> \\
        --out_root datasets/test_data/260815_slides_full --min_tissue 0.35
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HE_SUFFIX = "_HE_reg.tif"
AF_SUFFIX = "_AF.tif"


def integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row/column, so any rect is 4 lookups."""
    return np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rect_sum(ii: np.ndarray, y: np.ndarray, x: np.ndarray, h: int, w: int) -> np.ndarray:
    return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]


def best_crop(border: np.ndarray, tissue: np.ndarray, tile: int
              ) -> tuple[int, int, int, int] | None:
    """Largest tile-aligned border-free rectangle, most tissue among equals.

    Returns (y, x, ny, nx) in pixels/tiles, or None if not even one tile fits.
    """
    H, W = border.shape
    ib, it = integral(border), integral(tissue)

    # Prefer area, then squareness: a 5x3 and a 3x5 rank the same, a 4x4 beats both.
    shapes = sorted(((ny, nx) for ny in range(1, H // tile + 1)
                     for nx in range(1, W // tile + 1)),
                    key=lambda s: (-s[0] * s[1], abs(s[0] - s[1])))

    for ny, nx in shapes:
        h, w = ny * tile, nx * tile
        ys, xs = np.meshgrid(np.arange(H - h + 1), np.arange(W - w + 1), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
        clean = rect_sum(ib, ys, xs, h, w) == 0
        if not clean.any():
            continue
        ys, xs = ys[clean], xs[clean]
        best = int(np.argmax(rect_sum(it, ys, xs, h, w)))
        return int(ys[best]), int(xs[best]), ny, nx
    return None


def focus_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- low means the FOV is out of focus or empty."""
    g = gray.astype(np.float32)
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    return float(lap.var())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crop registered TPAF/H&E pairs to a tile-aligned border-free rect.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--reg_dir", required=True, help="Folder of *_AF.tif / *_HE_reg.tif.")
    ap.add_argument("--out_root", required=True, help="Dataset root to create.")
    ap.add_argument("--tile", type=int, default=204, help="Grid pitch in original px.")
    ap.add_argument("--min_grid", type=int, default=4,
                    help="Reject a FOV whose crop is smaller than this many tiles per axis.")
    ap.add_argument("--min_tissue", type=float, default=0.35,
                    help="Reject a FOV with less than this tissue fraction in the crop.")
    ap.add_argument("--tissue_thresh", type=int, default=200,
                    help="H&E grey level below which a pixel counts as tissue.")
    ap.add_argument("--manifest_only", action="store_true",
                    help="Measure and write manifest.csv, write no images.")
    args = ap.parse_args()

    reg = Path(args.reg_dir)
    out = Path(args.out_root)
    bases = sorted(p.name[: -len(HE_SUFFIX)] for p in reg.iterdir()
                   if p.name.endswith(HE_SUFFIX))
    if not bases:
        sys.exit(f"No {HE_SUFFIX} in {reg}")

    dirs = {k: out / k for k in ("00_og_real_HE_full", "00_og_TPAF_RGB_full",
                                 "01_og_TPAF_gray_full")}
    if not args.manifest_only:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, base in enumerate(bases, 1):
        he_path, af_path = reg / (base + HE_SUFFIX), reg / (base + AF_SUFFIX)
        if not af_path.exists():
            print(f"  [warn] no AF for {base}", file=sys.stderr)
            continue

        he_im = Image.open(he_path)          # page 0 is the registered H&E
        he = np.array(he_im)
        af_im = Image.open(af_path)
        af = np.array(af_im)
        if he.shape[:2] != af.shape[:2]:
            print(f"  [warn] size mismatch {base}: {he.shape} vs {af.shape}", file=sys.stderr)
            continue

        border = he.sum(2) == 0
        he_gray = np.array(he_im.convert("L"))
        tissue_px = (he_gray < args.tissue_thresh) & ~border

        crop = best_crop(border, tissue_px, args.tile)
        if crop is None:
            rows.append(dict(fov=base, kept=0, reason="no tile fits",
                             y="", x="", ny=0, nx=0, h=0, w=0, tissue="",
                             af_mean="", af_p99="", focus="",
                             border_frac=round(float(border.mean()), 4)))
            continue
        y, x, ny, nx = crop
        h, w = ny * args.tile, nx * args.tile
        sl = (slice(y, y + h), slice(x, x + w))

        af_gray = np.array(af_im.convert("L"))
        tissue = float(tissue_px[sl].mean())
        row = dict(fov=base, kept=0, reason="", y=y, x=x, ny=ny, nx=nx, h=h, w=w,
                   tissue=round(tissue, 4),
                   af_mean=round(float(af_gray[sl].mean()), 2),
                   af_p99=int(np.percentile(af_gray[sl], 99)),
                   focus=round(focus_score(af_gray[sl]), 1),
                   border_frac=round(float(border.mean()), 4))

        if min(ny, nx) < args.min_grid:
            row["reason"] = f"grid {ny}x{nx} < {args.min_grid}"
        elif tissue < args.min_tissue:
            row["reason"] = f"tissue {tissue:.3f} < {args.min_tissue}"
        else:
            row["kept"] = 1

        if row["kept"] and not args.manifest_only:
            Image.fromarray(he[sl]).save(dirs["00_og_real_HE_full"] / f"{base}_HE_reg.tif")
            Image.fromarray(af[sl]).save(dirs["00_og_TPAF_RGB_full"] / f"{base}_AF.tif")
            Image.fromarray(af_gray[sl]).save(dirs["01_og_TPAF_gray_full"] / f"{base}_AF.tif")
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(bases)}")

    man = out / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    kept = [r for r in rows if r["kept"]]
    windows = sum(((r["w"] - args.tile) // (args.tile // 2) + 1)
                  * ((r["h"] - args.tile) // (args.tile // 2) + 1) for r in kept)
    print(f"\n{len(kept)}/{len(rows)} FOVs kept -> {windows} windows at 50% overlap")
    from collections import Counter
    for reason, n in Counter(r["reason"] for r in rows if not r["kept"]).most_common():
        print(f"  dropped {n:>3}  {reason}")
    for g, n in sorted(Counter(f'{r["ny"]}x{r["nx"]}' for r in kept).items()):
        print(f"  grid {g}: {n} FOVs")
    print(f"manifest -> {man}")


if __name__ == "__main__":
    main()
