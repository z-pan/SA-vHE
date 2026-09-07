#!/usr/bin/env python3
"""Sweep the Cellpose-SAM settings until the nucleus mask covers what nuclei cover.

Why
---
The partitioned colour correction splits the image into nucleus, dense stroma and pale
stroma, and the nucleus compartment is this mask. Measured on 40 regions of nuc_flat:

    compartment share of tissue          nucleus 0.048   dense 0.574   pale 0.377
    where the nucleus-dark pixels land           0.104         0.882         0.014

So h_nuc governs a tenth of the dark material and h_dense governs seven eighths, and
every "nucleus-aware" setting is really a stroma setting. That is why the dE fit
converged on a near-uniform reduction (0.53 / 0.46 / 0.51), and why suppressing the
stroma compartment removes the spurious nuclei together with the real ones that fall
outside the mask. The mask has to cover the nuclei before a compartment-wise correction
can separate anything.

The objective
-------------
Not a nucleus count -- counts from different tools disagree threefold here (Cellpose
2459/mm2 on real H&E, HoVer-Net 1263, TPAF 1274). Area, matched per region against the
same region's real H&E:

    TPAF mask coverage of the crop      median 0.0277
    real H&E nuclear coverage           median 0.0872      ratio 0.32, r = 0.75

The correlation says the mask tracks real nuclear content; the ratio says it finds a
third of it. And the shortfall is not uniform -- 64 of 148 regions come in under a
quarter of real, several at exactly zero, and five of the six worst belong to 240729,
the pale slide that needed its own tissue threshold at tiling time. So the sample has to
span slides, or it measures whichever slide happens to come first in the manifest.

Nucleus count is printed but not optimised: it moves with `diameter` for reasons that
have nothing to do with how much nuclear material is present -- one nucleus found at
twice the size, or two nuclei merged, both leave the area unchanged.

    python path_vhe_masksweep.py --n 12                       # the SWEEP list
    python path_vhe_masksweep.py --n 12 --cellprob -3 --diameter 10   # one setting
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time

import cv2
import numpy as np

SUR = 'results/path_screen/survey/_vhe'
MODEL = 'cpsam_20260228_gray'
TPAF_MPP = 0.621

# (flow, cellprob, diameter). cellprob is the one that moves coverage: at the current
# 0.0 the model reports a third of the nuclear area, and each step down admits more of
# the low-contrast nuclei that the pale slides are full of. flow stays at 0.85, already
# far more permissive than the library's 0.4, so that only one thing changes at a time.
SWEEP = [
    (0.85, 0.0, 30),      # current settings, the reference row
    (0.85, -3.0, 14),
    (0.85, -3.0, 10),
    (0.85, -4.5, 10),
]


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if im is None:
        raise SystemExit('cannot decode ' + path)
    return im


def stratified(rows, n):
    """One region per slide in rotation.

    The manifest is ordered by slide, so rows[:n] is a sample of one slide, and the
    failure being measured is slide-dependent.
    """
    by = {}
    for r in rows:
        by.setdefault(r['sample'], []).append(r)
    out, i = [], 0
    while len(out) < n:
        before = len(out)
        for s in sorted(by):
            if i < len(by[s]):
                out.append(by[s][i])
                if len(out) >= n:
                    break
        if len(out) == before:
            break
        i += 1
    return out, len(by)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--cp', default='results/path_screen/survey/_downstream/cp',
                    help='Where the real H&E label images live.')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--flow', type=float, default=None)
    ap.add_argument('--cellprob', type=float, default=None)
    ap.add_argument('--diameter', type=float, default=None)
    args = ap.parse_args()

    from cellpose import models
    import torch

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    seen, uniq = set(), []
    for r in rows:
        if r['patch_path'] not in seen:
            seen.add(r['patch_path'])
            uniq.append(r)
    items, n_slides = stratified(uniq, args.n)

    gpu = torch.cuda.is_available()
    mpath = os.path.expanduser(os.path.join('~', '.cellpose', 'models', args.model))
    model = models.CellposeModel(gpu=gpu, pretrained_model=mpath)

    # Cache the grayscale input, the crop box and the real H&E coverage once; the sweep
    # only changes segmentation settings, and re-reading a 1024x1024 TIF is wasteful.
    cache = []
    for r in items:
        hp = os.path.join(args.cp, 'real_HE', r['id'] + '.png')
        if not (os.path.exists(r['patch_path']) and os.path.exists(hp)):
            continue
        gray = cv2.cvtColor(imread_u(r['patch_path']), cv2.COLOR_BGR2GRAY)
        box = tuple(int(r[k]) for k in ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
        cache.append((r['id'], gray, box, float((imread_u(hp) > 0).mean())))
    print('%s  gpu=%s  %d regions from %d slides' % (args.model, gpu, len(cache),
                                                     n_slides))
    print('real H&E coverage of these regions: median %.4f'
          % np.median([c[3] for c in cache]))
    print('objective: ratio -> 1.00, zeros -> 0\n')

    combos = SWEEP
    if args.cellprob is not None or args.diameter is not None:
        combos = [(args.flow if args.flow is not None else 0.85,
                   args.cellprob if args.cellprob is not None else 0.0,
                   args.diameter if args.diameter is not None else 30)]

    print('%6s%10s%9s%9s%11s%11s%9s%7s'
          % ('flow', 'cellprob', 'diam px', 'diam um', 'mask cov', 'real cov',
             'ratio', 'zeros'))
    for flow, cp, diam in combos:
        t0 = time.time()
        mk, he, ratio, zeros = [], [], [], 0
        for rid, gray, (x, y, w, h), hecov in cache:
            lab, _, _ = model.eval(gray, diameter=diam, flow_threshold=flow,
                                   cellprob_threshold=cp)
            c = float((lab[y:y + h, x:x + w] > 0).mean())
            mk.append(c)
            he.append(hecov)
            if hecov > 0:
                ratio.append(c / hecov)
            if c < 1e-4:
                zeros += 1
        print('%6.2f%10.1f%9d%9.1f%11.4f%11.4f%9.2f%7d   %.0fs'
              % (flow, cp, diam, diam * TPAF_MPP, np.median(mk), np.median(he),
                 np.median(ratio), zeros, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
