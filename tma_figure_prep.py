#!/usr/bin/env python3
"""Prepare per-core figure material: TPAF mosaic, matching real H&E, vHE input tiles.

Produces, for each core, into <out>/<well>/:
  mosaic.tif       stitched TPAF, ~2670 px (1.66 mm at 0.621 um/px)
  real_HE.tif      real H&E warped into the mosaic's frame, same size and scale
  vhe_in/          204 px crops upscaled to 512 with BICUBIC, mode 'L'
  meta.json        the fitted similarity and the tile grid

The vHE tiles follow the geometry UTOM was trained on -- crop 204, resize to 512
BICUBIC -- because the model reads nuclei at that apparent size. Mode 'L' matters
too: unaligned_dataset does no .convert(), so the file's own mode decides the channel
count and a 3-channel file fails at the first conv.

Real H&E comes through the same similarity fit used for the pair dataset (disc centre,
then a bounded mask search over rotation, scale and translation), which lands within
about 39 um. That is far finer than an attention map's patch, so it is enough for a
figure; it is not pixel-level, and the panels should not be read as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tma_align import SCALE, align, crop_he_for_tile  # noqa: E402
from tma_build_pairs import he_disc  # noqa: E402
from tma_stitch import build_mosaic, load_tiles, solve_positions  # noqa: E402

TILE, PATCH = 204, 512


def g(a):
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tpaf_dir', required=True)
    ap.add_argument('--he_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wells', required=True)
    args = ap.parse_args()

    for well in [w.strip() for w in args.wells.split(',')]:
        d = None
        for name in sorted(os.listdir(args.tpaf_dir)):
            if name.endswith(f'{well}.tif.frames') and not name.lower().startswith('not used'):
                d = os.path.join(args.tpaf_dir, name)
        he_p = os.path.join(args.he_dir, f'{well}.tif')
        if d is None or not os.path.exists(he_p):
            print(f'  [skip] {well}'); continue

        tiles = load_tiles(d, well)
        if len(tiles) != 9:
            print(f'  [skip] {well}: {len(tiles)} tiles'); continue
        pos = solve_positions(tiles)
        mos, (mh, mw) = build_mosaic(tiles, pos)

        he = tiff.imread(he_p)
        disc = he_disc(g(he), expect_r=max(mh, mw) * SCALE / 2.0)
        cy, cx, rr = disc
        y0 = cy - mh * SCALE / 2.0
        x0 = cx - mw * SCALE / 2.0
        a = align(mos, g(he), y0, x0)
        if a['score'] > 0.12:
            y0, x0 = a['y'], a['x']
        # the whole mosaic is one big "tile" at origin, so the same warp gives the
        # matching H&E in the mosaic's own frame
        real = crop_he_for_tile(he[..., :3], (0, 0), (y0, x0), a['angle'], a['scale'],
                                (mh, mw), max(mh, mw), upsample=1.0)[:mh, :mw]

        od = os.path.join(args.out, well)
        os.makedirs(os.path.join(od, 'vhe_in'), exist_ok=True)
        tiff.imwrite(os.path.join(od, 'mosaic.tif'),
                     np.clip(mos, 0, 65535).astype(np.uint16), compression='zlib')
        tiff.imwrite(os.path.join(od, 'real_HE.tif'), real, compression='zlib')

        gray8 = cv2.normalize(mos, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        im = Image.fromarray(gray8, mode='L')
        n = 0
        for yy in range(0, mh - TILE + 1, TILE):
            for xx in range(0, mw - TILE + 1, TILE):
                im.crop((xx, yy, xx + TILE, yy + TILE)) \
                  .resize((PATCH, PATCH), Image.BICUBIC) \
                  .save(os.path.join(od, 'vhe_in', f'{well}_x{xx}_y{yy}.png'))
                n += 1
        json.dump(dict(well=well, mosaic_h=mh, mosaic_w=mw, tile=TILE, patch=PATCH,
                       n_tiles=n, angle=a['angle'], scale=a['scale'],
                       he_y=a['y'], he_x=a['x'], align_score=a['score'],
                       disc_r=rr, um_per_px_tpaf=0.621),
                  open(os.path.join(od, 'meta.json'), 'w'), indent=2)
        print(f'  {well}: mosaic {mw}x{mh}  align {a["score"]:.2f} '
              f'rot {a["angle"]:+.2f}  {n} vHE tiles -> {od}', flush=True)


if __name__ == '__main__':
    main()
