#!/usr/bin/env python3
"""Crop the same strip across a tile boundary from several stitched sets, side by side.

``measure_seam.py`` says whether the block grid is gone; this shows it. For each
FOV it takes a window centred on one tile boundary and stacks the same window from
every directory given, so the seam -- or its absence -- is on one screen.

The boundary line is where a reader's eye should find nothing. Put the real H&E
last as the reference for what "nothing" looks like.

Usage
-----
    python qc_seam_montage.py \\
        results/stitched_vHE_leveled \\
        results/stitched_vHE_overlap \\
        datasets/test_data/250711_slides/00_og_real_HE_full \\
        --out_dir results/qc_seam --boundary 408 --size 320
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _index(d: Path) -> dict[str, Path]:
    """Map a short FOV key to a path, so differently-suffixed sets line up."""
    out = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in _EXT:
            key = p.stem.replace("_HE_reg", "").replace("_AF", "")
            out[key] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side crops across a tile boundary.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("dirs", nargs="+", help="Stitched-FOV directories, in display order.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--boundary", type=int, default=408, help="Tile boundary to centre on.")
    ap.add_argument("--size", type=int, default=320, help="Crop side length in px.")
    ap.add_argument("--mark", action="store_true",
                    help="Draw a hairline where the boundary is, for orientation.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = [(Path(d).name, _index(Path(d))) for d in args.dirs]
    keys = sorted(set.intersection(*(set(s[1]) for s in sets)))
    if not keys:
        raise SystemExit("no FOV keys shared by all directories")

    half = args.size // 2
    x0, y0 = args.boundary - half, args.boundary - half
    for key in keys:
        tiles = []
        for _, idx in sets:
            a = np.array(Image.open(idx[key]).convert("RGB"))
            tiles.append(a[y0:y0 + args.size, x0:x0 + args.size])
        strip = Image.fromarray(np.concatenate(tiles, axis=1))
        if args.mark:
            draw = ImageDraw.Draw(strip)
            for i in range(len(tiles)):
                x = i * args.size + half
                draw.line([(x, 0), (x, 6)], fill=(255, 255, 0), width=1)
                draw.line([(x, args.size - 6), (x, args.size)], fill=(255, 255, 0), width=1)
        strip.save(out_dir / f"{key}.png")

    print(f"{len(keys)} montages ({' | '.join(n for n, _ in sets)}) "
          f"at x=y={args.boundary} -> {out_dir}")


if __name__ == "__main__":
    main()
