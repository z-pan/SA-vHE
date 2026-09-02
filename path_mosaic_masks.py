#!/usr/bin/env python3
"""Cellpose-SAM nuclei mask over a whole stitched TPAF mosaic.

Why a separate script. path_vhe_masks.py walks the 148-region manifest and each patch
fits in memory and on the card. A slide mosaic is 13371x12719 -- 170 Mpx, which will
not go through a 4 GB GPU in one piece -- so it has to be tiled here, and the tiling
has to not introduce edges of its own.

Everything about the model call is copied from path_vhe_masks.py rather than re-chosen:

    model            cpsam_20260228_gray, fine-tuned on 1-channel TPAF
    input            cv2 BGR2GRAY of the mosaic -- the SAME grey the staining pipeline
                     feeds the generator, so the mask lines up with what gets stained
    flow_threshold   0.85   (not the library default 0.4; it moves the nuclei count a
                            lot, and 0.85 is what every mask in this project used)
    cellprob         0.0
    diameter         30     for TPAF at 0.621 um/px

Only a binary mask is kept, matching _vhe/masks and _vhe/masks_he, so overlapping tiles
merge by OR and a nucleus cut by a tile edge is still whole in the union. Instance
labels would need stitching across the seam; nothing downstream uses them --
enhance_nuclei() blurs the binary mask and adds it to the input.

Background tiles are skipped. Roughly half a stitched slide is empty, and the model has
no reason to be asked about it.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

MODEL = 'cpsam_20260228_gray'
FLOW = 0.85
CELLPROB = 0.0
DIAMETER = 30


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if im is None:
        raise SystemExit(f'cannot decode {path}')
    return im


def imwrite_u(path, im):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], im)
    if not ok:
        raise SystemExit(f'cannot encode {path}')
    buf.tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='Stitched TPAF mosaic.')
    ap.add_argument('--out', required=True, help='Output PNG, binary mask.')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--flow_threshold', type=float, default=FLOW)
    ap.add_argument('--cellprob_threshold', type=float, default=CELLPROB)
    ap.add_argument('--diameter', type=float, default=DIAMETER)
    ap.add_argument('--tile', type=int, default=1024)
    ap.add_argument('--overlap', type=int, default=128,
                    help='Tiles overlap by this much and merge by OR, so a nucleus '
                         'sitting on a tile edge is complete in at least one of them.')
    ap.add_argument('--min_signal', type=int, default=6,
                    help='Skip tiles whose grey never exceeds this.')
    args = ap.parse_args()

    import torch
    from cellpose import models

    src = imread_u(args.image)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    H, W = gray.shape[:2]

    gpu = torch.cuda.is_available()
    mpath = os.path.expanduser(os.path.join('~', '.cellpose', 'models', args.model))
    model = models.CellposeModel(gpu=gpu, pretrained_model=mpath)
    print(f'{args.model}  gpu={gpu}  flow={args.flow_threshold} '
          f'cellprob={args.cellprob_threshold} diameter={args.diameter}')
    print(f'{os.path.basename(args.image)}  {W}x{H}', flush=True)

    step = args.tile - args.overlap
    ys = list(range(0, max(1, H - args.tile) + 1, step))
    xs = list(range(0, max(1, W - args.tile) + 1, step))
    if ys[-1] != max(0, H - args.tile):
        ys.append(max(0, H - args.tile))
    if xs[-1] != max(0, W - args.tile):
        xs.append(max(0, W - args.tile))
    coords = [(y, x) for y in ys for x in xs]
    live = [(y, x) for y, x in coords
            if gray[y:y + args.tile, x:x + args.tile].max() > args.min_signal]
    print(f'{len(coords)} tiles, {len(coords) - len(live)} background, '
          f'{len(live)} to run', flush=True)

    out = np.zeros((H, W), np.uint8)
    t0 = time.time()
    total = 0
    for i, (y, x) in enumerate(live, 1):
        sub = gray[y:y + args.tile, x:x + args.tile]
        lab, _, _ = model.eval(sub, diameter=args.diameter,
                               flow_threshold=args.flow_threshold,
                               cellprob_threshold=args.cellprob_threshold)
        total += int(lab.max())
        np.maximum(out[y:y + sub.shape[0], x:x + sub.shape[1]],
                   (lab > 0).astype(np.uint8) * 255,
                   out=out[y:y + sub.shape[0], x:x + sub.shape[1]])
        if i % 20 == 0 or i == len(live):
            el = time.time() - t0
            print(f'  {i}/{len(live)}  {el:.0f}s  '
                  f'eta {el / i * (len(live) - i) / 60:.1f} min', flush=True)

    imwrite_u(args.out, out)
    tis = gray > args.min_signal
    print(f'\n{total} nuclei summed over tiles (double-counted in the overlaps)')
    print(f'mask covers {100 * float((out > 0).mean()):.2f}% of the mosaic, '
          f'{100 * float((out > 0)[tis].mean()):.2f}% of tissue')
    print(f'-> {args.out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
