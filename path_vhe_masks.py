#!/usr/bin/env python3
"""Nuclei masks for the manifest patches, with Cellpose-SAM.

Needed twice over, which is why it is worth doing once and caching: the nuclei-enhanced
model input adds a blurred mask to the grayscale, and the partitioned colour correction
scales haematoxylin differently inside and outside nuclei. Both read the same file.

Parameters are the ones tuned for this checkpoint, not the library defaults, and the
difference is not cosmetic -- flow_threshold moves the number of detected nuclei a lot,
and the library default of 0.4 has silently produced the wrong count in this project
before. They are printed on every run so a result can be traced to them:

    model               cpsam_20260228_gray, fine-tuned on 1-channel TPAF
    flow_threshold      0.85
    cellprob_threshold  0.0
    diameter            30      for raw TPAF at 0.621 um/px

Input is single-channel uint8, matching how the checkpoint was trained. Output is a
0/255 uint8 mask the same size as the patch, which is what enhance_nuclei expects to
blur and threshold at 158.

Run it in the environment that has Cellpose-SAM and a CUDA torch:

    C:/Users/zpanp/anaconda3/envs/cellpose_clean/python.exe path_vhe_masks.py
"""

from __future__ import annotations

import argparse
import csv
import io
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
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--out', default='results/path_screen/survey/_vhe/masks')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--flow_threshold', type=float, default=FLOW)
    ap.add_argument('--cellprob_threshold', type=float, default=CELLPROB)
    ap.add_argument('--diameter', type=float, default=DIAMETER)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--redo', action='store_true',
                    help='Recompute masks that already exist. Off by default so an '
                         'interrupted run continues instead of starting over.')
    args = ap.parse_args()

    from cellpose import models
    import torch

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    patches = {}
    for r in rows:
        patches.setdefault(r['patch_path'], r)
    items = sorted(patches.items())
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    gpu = torch.cuda.is_available()
    mpath = os.path.expanduser(os.path.join('~', '.cellpose', 'models', args.model))
    model = models.CellposeModel(gpu=gpu, pretrained_model=mpath)
    print(f'{args.model}  gpu={gpu}')
    print(f'flow_threshold={args.flow_threshold}  '
          f'cellprob_threshold={args.cellprob_threshold}  diameter={args.diameter}')
    print(f'{len(items)} patches\n')

    t0 = time.time()
    counts = []
    for i, (path, r) in enumerate(items, 1):
        name = os.path.splitext(r['stage_name'])[0]
        dst = os.path.join(args.out, name + '.png')
        if os.path.exists(dst) and not args.redo:
            continue
        gray = cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2GRAY)
        lab, _, _ = model.eval(gray, diameter=args.diameter,
                               flow_threshold=args.flow_threshold,
                               cellprob_threshold=args.cellprob_threshold)
        n = int(lab.max())
        counts.append((name, n, gray.size))
        imwrite_u(dst, ((lab > 0).astype(np.uint8) * 255))
        if i % 10 == 0 or i == len(items):
            el = time.time() - t0
            print(f'  {i}/{len(items)}  {el:.0f}s  '
                  f'eta {el / max(1, i) * (len(items) - i) / 60:.0f} min', flush=True)

    print(f'\n{len(counts)} masks in {time.time() - t0:.0f}s -> {args.out}')
    if counts:
        per = [n for _, n, _ in counts]
        dens = [1e6 * n / s for _, n, s in counts]
        print(f'  nuclei per patch: median {int(np.median(per))}, '
              f'range {min(per)}-{max(per)}')
        print(f'  density per Mpx : median {np.median(dens):.0f}, '
              f'range {min(dens):.0f}-{max(dens):.0f}')
        zero = [n for n, c, _ in counts if c == 0]
        if zero:
            print(f'  {len(zero)} patches with no nuclei found: {zero[:5]}')


if __name__ == '__main__':
    main()
