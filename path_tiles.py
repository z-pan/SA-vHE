#!/usr/bin/env python3
"""Cut an H&E image into tiles for pathology-model scoring, keeping their coordinates.

The point of scoring is to pick regions worth putting in a figure, so every tile has
to be traceable back to where it came from. The index this writes is what turns a
per-tile score into a map over the slide.

Tiles are cut at a physical size, not a pixel size, so an H&E slide at 0.242 um/px and a
TPAF mosaic at 0.621 um/px give tiles covering the same amount of tissue -- otherwise
scores are not comparable between them.

Scales in this project. Four numbers, three instruments -- none of them interchangeable,
and the TIFF metadata is a placeholder in every case, so these were settled by
measurement and by asking how the files were produced:

    0.621   TPAF mosaic. Authoritative: stated in the per-FOV data_description export
            ("1024, 0.0 - 636.396 [um], 0.621 [um/pixel]").
    0.5     H&E whole-slide TIFs (WSI_*.tif). A 20x slide-scanner acquisition, same
            scanner for all slides in this set. Independently supported: matching 13
            separate H&E FOV captures into the 240817 montage over candidate scales,
            10 peaked at 0.4942 and 2 at 0.5022, none at the Hamamatsu values.
    0.242   H&E per-FOV capture (BioHD-C20 microscope camera, 40x). This is the number
            behind the 1/0.39 = 2.564 TPAF:H&E ratio in
            notebook_AF_HE_registration.ipynb -- 0.621/0.242 = 2.566.
    0.4429  H&E TMA. A different scanner (Hamamatsu, read from the ndpi XResolution).

The whole-slide TIF and the per-FOV captures are two independent acquisitions of the
same slide at different magnifications, not a crop or a downsample of one another. That
is why template-matching a FOV into the montage gives no sharp peak.

The trap: 2.564 is TPAF versus the 40x FOV captures. Against the 20x whole-slide scan
the ratio is only 0.621/0.5 = 1.24, which is why the two look like similar
magnifications by eye. Applying 2.564 to the montage puts its scale at 0.242 and makes
every tile twice its intended physical size, with nothing to signal the error.

At 0.5 um/px a 256 um tile is exactly 512 px, so --out_px 512 needs no resampling.

Blank tiles are dropped: a whole-slide image is mostly glass, and scoring it wastes
most of the run.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np
import tifffile as tiff


def load_any(path):
    """memmap when the file allows it, so a 1 GB slide costs nothing to open.

    PNG falls through to cv2, which is how the stitched vHE mosaics in results/ are
    stored; those are small enough that reading them whole costs nothing.
    """
    if os.path.splitext(path)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f'cannot read {path}')
        return im[..., ::-1] if im.ndim == 3 else im  # cv2 is BGR, the rest here is RGB
    try:
        return tiff.memmap(path)
    except Exception:
        return tiff.imread(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--um_per_px', type=float, required=True,
                    help='0.5 H&E whole-slide scan (20x), 0.242 H&E per-FOV capture '
                         '(40x), 0.4429 H&E TMA, 0.621 TPAF mosaic. See the module '
                         'docstring: the whole-slide scan and the FOV captures are '
                         'separate acquisitions, and using the FOV scale for a slide '
                         'doubles the physical tile size with nothing to signal it.')
    ap.add_argument('--tile_um', type=float, default=256.0,
                    help='Physical tile size. 256 um at 20x is the usual MIL patch.')
    ap.add_argument('--overlap', type=float, default=0.0)
    ap.add_argument('--out_px', type=int, default=512, help='Saved tile size.')
    ap.add_argument('--min_tissue', type=float, default=0.25)
    ap.add_argument('--tissue_thresh', type=int, default=215,
                    help='Grey level below which a pixel counts as tissue. 215 suits most slides here, but check it per sample: 240729 is stained so palely that 215 keeps only 6.8 percent of the slide and discards genuine pale myxoid stroma -- exactly the ECM-rich tissue TPAF is meant to show. It needs 240 (23.9 percent). Otsu and triangle thresholding were both tried as automatic replacements and each worked on only half the samples: triangle overshoots to 80-92 percent where the glass background is clipped at 255, Otsu undershoots where tissue covers little of the slide.')
    ap.add_argument('--invert_tissue', action='store_true',
                    help='For TPAF, where tissue is bright rather than dark.')
    ap.add_argument('--jpeg_quality', type=int, default=0,
                    help='Save tiles as JPEG at this quality instead of PNG. At 50 '
                         'percent overlap a slide runs to thousands of tiles and PNG '
                         'makes the upload the bottleneck; CONCH resizes to 448 '
                         'anyway, so q95 costs nothing that matters. 0 keeps PNG.')
    ap.add_argument('--max_tiles', type=int, default=0)
    ap.add_argument('--thumb_ds', type=int, default=16)
    args = ap.parse_args()

    # Store the resolved absolute path: a mistyped one can still open on Windows but
    # is then written to source.txt and breaks anything that reads it back.
    args.image = os.path.abspath(args.image)
    os.makedirs(os.path.join(args.out, 'tiles'), exist_ok=True)
    img = load_any(args.image)
    H, W = img.shape[:2]
    tile = int(round(args.tile_um / args.um_per_px))
    step = max(1, int(round(tile * (1 - args.overlap))))
    print(f'{os.path.basename(args.image)} {W}x{H} @ {args.um_per_px} um/px', flush=True)
    print(f'tile {tile} px = {args.tile_um} um, step {step} px', flush=True)

    thumb = np.asarray(img[::args.thumb_ds, ::args.thumb_ds])
    tg = cv2.cvtColor(thumb[..., :3], cv2.COLOR_RGB2GRAY) if thumb.ndim == 3 else thumb
    cv2.imwrite(os.path.join(args.out, 'thumbnail.png'),
                thumb[..., ::-1] if thumb.ndim == 3 else thumb)

    rows = []
    n_blank = 0
    for y in range(0, H - tile + 1, step):
        for x in range(0, W - tile + 1, step):
            ty, tx = y // args.thumb_ds, x // args.thumb_ds
            th = max(1, tile // args.thumb_ds)
            patch = tg[ty:ty + th, tx:tx + th]
            frac = float((patch > 20).mean() if args.invert_tissue
                         else (patch < args.tissue_thresh).mean())
            if frac < args.min_tissue:
                n_blank += 1
                continue
            rows.append(dict(tile_id=f'y{y}_x{x}', y=y, x=x, size=tile,
                             tissue=round(frac, 3)))
    if args.max_tiles and len(rows) > args.max_tiles:
        rows = sorted(rows, key=lambda r: -r['tissue'])[:args.max_tiles]
        rows.sort(key=lambda r: (r['y'], r['x']))
        print(f'capped to the {args.max_tiles} tiles with most tissue', flush=True)
    print(f'{len(rows)} tiles kept, {n_blank} dropped as blank', flush=True)

    ext = '.jpg' if args.jpeg_quality else '.png'
    for i, r in enumerate(rows, 1):
        a = np.asarray(img[r['y']:r['y'] + tile, r['x']:r['x'] + tile])
        if a.ndim == 2:
            a = cv2.cvtColor(cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                             cv2.COLOR_GRAY2RGB)
        else:
            a = a[..., :3]
        a = cv2.resize(a, (args.out_px, args.out_px), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.out, 'tiles', r['tile_id'] + ext), a[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if ext == '.jpg' else [])
        if i % 200 == 0:
            print(f'  {i}/{len(rows)}', flush=True)

    with open(os.path.join(args.out, 'index.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['tile_id', 'y', 'x', 'size', 'tissue'])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, 'source.txt'), 'w', encoding='utf-8') as fh:
        fh.write(f'image={args.image}\nW={W}\nH={H}\num_per_px={args.um_per_px}\n'
                 f'tile_px={tile}\ntile_um={args.tile_um}\nstep_px={step}\n'
                 f'thumb_ds={args.thumb_ds}\ntile_ext={ext}\n'
                 f'overlap={args.overlap}\njpeg_quality={args.jpeg_quality}\n'
                 # The threshold is per sample -- 240729 needs 240 where the other
                 # five take 215 -- and path_locate.py derives the tissue bounding
                 # box it maps onto the TPAF grid with the same test. Leaving it
                 # unrecorded meant that tool silently fell back to its own default
                 # and boxed a different piece of tissue than the one that was
                 # tiled. Nothing errors; the only visible symptom is a frac outside
                 # 0..1 for a tile near the edge, and only if you look.
                 f'tissue_thresh={args.tissue_thresh}\n'
                 f'min_tissue={args.min_tissue}\n'
                 f'invert_tissue={int(bool(args.invert_tissue))}\n')
    print(f'-> {args.out}/tiles, index.csv, thumbnail.png')


if __name__ == '__main__':
    main()
