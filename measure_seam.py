#!/usr/bin/env python3
"""Score the block grid left in a stitched FOV by per-tile inference.

The number
----------
Seam ratio = mean pixel-to-pixel jump *across* a tile boundary, divided by the
mean jump at the same lines shifted ``--offset`` px into the tiles. Dividing by a
nearby in-tile jump normalises away how busy the tissue is, so the ratio is
comparable across images and stains:

    1.0   the boundary is no sharper than ordinary tissue detail -- no seam
    >1    a visible discontinuity at the tile grid

Measured on the seven 250711_slides FOVs over the same 816 px region (period 204,
offset 8):

    real H&E, never tiled                       1.01   <- the floor
    stitched vHE, colour-corrected              3.35
    stitched vHE, corrected + tile levelling    3.19

Real H&E lands on 1.01, so the metric has essentially no bias of its own and 1.0 is
the target. An earlier note in ``stitch_and_compare.level_tiles`` quotes a 1.17
floor with seam ratios on a slightly different scale; those came from an ad-hoc
variant of this measurement and are not comparable to the numbers above. Score
everything with this script before comparing.

Only boundaries strictly inside the image count, and the comparison region is
clipped with ``--region`` so images of different sizes (816 px stitched output vs
1010x848 full reference) are scored over the same area.

Usage
-----
Compare the baseline, the overlap-blended result, and the real-H&E floor::

    python measure_seam.py \\
        results/stitched_vHE_leveled \\
        results/stitched_vHE_overlap \\
        datasets/test_data/250711_slides/00_og_real_HE_full \\
        --period 204 --region 816

``--period`` is the tile grid being tested. With overlapping windows also check the
window stride (``--period 102``): blending removes the old boundaries but must not
introduce new ones at the new positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _line_jump(gray: np.ndarray, axis: int, index: int) -> float:
    """Mean |difference| between the two pixel lines meeting at *index*."""
    if axis == 0:                                    # horizontal boundary
        return float(np.abs(gray[index] - gray[index - 1]).mean())
    return float(np.abs(gray[:, index] - gray[:, index - 1]).mean())


def seam_ratio(img: np.ndarray, period: int, offset: int) -> tuple[float, float, float]:
    """(ratio, mean border jump, mean in-tile jump) over every interior boundary."""
    gray = img.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(2)
    border, inside = [], []
    for axis, n in enumerate(gray.shape[:2]):
        for b in range(period, n, period):
            if not offset < b < n - offset:
                continue
            border.append(_line_jump(gray, axis, b))
            inside += [_line_jump(gray, axis, b - offset),
                       _line_jump(gray, axis, b + offset)]
    if not border:
        return float("nan"), float("nan"), float("nan")
    mb, mi = float(np.mean(border)), float(np.mean(inside))
    return (mb / mi if mi > 1e-9 else float("nan")), mb, mi


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure the tile-grid seam left in stitched FOVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("dirs", nargs="+", help="Directories of stitched FOVs to score.")
    ap.add_argument("--period", type=int, default=204,
                    help="Tile grid spacing in px, i.e. where seams would sit.")
    ap.add_argument("--offset", type=int, default=8,
                    help="Distance into the tile for the reference jump.")
    ap.add_argument("--region", type=int, default=None,
                    help="Crop every image to this square before scoring, so "
                         "different-sized images are compared over the same area.")
    ap.add_argument("--per_image", action="store_true", help="Also print each FOV.")
    args = ap.parse_args()

    print(f"seam ratio at period {args.period} px, reference jump {args.offset} px "
          f"inside" + (f", region {args.region}px" if args.region else ""))
    print(f"\n{'directory':<44} {'n':>3} {'ratio':>7} {'border':>8} {'inside':>8}")

    for d in args.dirs:
        paths = sorted(p for p in Path(d).iterdir()
                       if p.is_file() and p.suffix.lower() in _EXT)
        rows = []
        for p in paths:
            img = np.array(Image.open(p).convert("RGB"))
            if args.region:
                img = img[:args.region, :args.region]
            r, mb, mi = seam_ratio(img, args.period, args.offset)
            if np.isfinite(r):
                rows.append((p.stem, r, mb, mi))
        if not rows:
            print(f"{Path(d).name:<44}   -  no scorable images")
            continue
        print(f"{Path(d).name:<44} {len(rows):>3} "
              f"{np.mean([r[1] for r in rows]):>7.2f} "
              f"{np.mean([r[2] for r in rows]):>8.2f} "
              f"{np.mean([r[3] for r in rows]):>8.2f}")
        if args.per_image:
            for stem, r, mb, mi in rows:
                print(f"    {stem[-46:]:<46} {r:>7.2f} {mb:>8.2f} {mi:>8.2f}")


if __name__ == "__main__":
    main()
