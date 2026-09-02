#!/usr/bin/env python3
"""Measure the residual TPAF/H&E offset of pairs that are already on disk.

{well}_T00N_HE_reg.tif is the same field at TPAF scale with --redundancy on each
side, so the 1024 px AF tile should sit at exactly (red, red) inside it. Correlating
the two tissue masks and comparing the peak against that expected origin gives the
placement error per pair, in TPAF pixels, without needing any ground truth.

Reported in TPAF px; multiply by 0.621 for um, or by 1.402 for H&E px.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import tifffile as tiff

NAME = re.compile(r'^(?P<well>\d{1,2}[A-H])_T(?P<tile>\d{3})_AF\.tif$')


def af_mask(a, ds):
    if a.ndim == 3:
        a = a[..., :2].mean(2)
    a = cv2.resize(a.astype(np.float32), None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 2)
    return (a > max(np.percentile(a, 55), a.max() * 0.07)).astype(np.float32)


def he_mask(h, ds):
    g = cv2.cvtColor(h[..., :3], cv2.COLOR_RGB2GRAY) if h.ndim == 3 else h
    g = cv2.resize(g.astype(np.float32), None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    return (cv2.GaussianBlur(g, (0, 0), 2) < 205).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--redundancy', type=float, default=0.05)
    ap.add_argument('--ds', type=int, default=4)
    ap.add_argument('--out', default=None, help='CSV of per-pair offsets. Default <dir>/offsets.csv')
    args = ap.parse_args()

    red = int(round(1024 * args.redundancy))
    pairs = []
    for f in sorted(os.listdir(args.dir)):
        m = NAME.match(f)
        if m:
            pairs.append((m.group('well'), int(m.group('tile'))))
    print(f'{len(pairs)} pairs, expected AF origin ({red},{red}) in HE_reg', flush=True)

    rows = []
    for n, (well, tile) in enumerate(pairs, 1):
        b = os.path.join(args.dir, f'{well}_T{tile:03d}')
        try:
            am = af_mask(tiff.imread(b + '_AF.tif'), args.ds)
            hm = he_mask(tiff.imread(b + '_HE_reg.tif'), args.ds)
        except Exception as e:
            print(f'  [skip] {well}_T{tile:03d}: {e}'); continue
        if am.shape[0] >= hm.shape[0] or am.shape[1] >= hm.shape[1]:
            continue
        r = cv2.matchTemplate(hm, am, cv2.TM_CCOEFF_NORMED)
        _, score, _, mx = cv2.minMaxLoc(r)
        dy = mx[1] * args.ds - red
        dx = mx[0] * args.ds - red
        rows.append(dict(well=well, tile=tile, dy=dy, dx=dx,
                         dist=round(float(np.hypot(dy, dx)), 1), ncc=round(float(score), 3)))
        if n % 100 == 0:
            print(f'  {n}/{len(pairs)}', flush=True)

    out = args.out or os.path.join(args.dir, 'offsets.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    d = np.array([r['dist'] for r in rows])
    nc = np.array([r['ncc'] for r in rows])
    print(f'\nper-pair offset (TPAF px):  median {np.median(d):.0f}  '
          f'p90 {np.percentile(d,90):.0f}  max {d.max():.0f}')
    print(f'  = um:                     median {np.median(d)*0.621:.0f}  '
          f'p90 {np.percentile(d,90)*0.621:.0f}')
    print(f'mask ncc: median {np.median(nc):.2f}\n')

    # a core-level shift is a systematic placement error; scatter within a core is noise
    per = defaultdict(list)
    for r in rows:
        per[r['well']].append(r)
    core = []
    for w, rs in per.items():
        med = np.hypot(np.median([r['dy'] for r in rs]), np.median([r['dx'] for r in rs]))
        core.append((med, w, len(rs), float(np.median([r['ncc'] for r in rs]))))
    core.sort(reverse=True)
    print(f'{"worst cores":<14}{"median shift":>14}{"pairs":>7}{"ncc":>7}')
    for med, w, n_, nc_ in core[:15]:
        print(f'{w:<14}{med:>11.0f}px{n_:>7}{nc_:>7.2f}')
    cm = np.array([c[0] for c in core])
    print(f'\ncore-level shift: median {np.median(cm):.0f}px  '
          f'>100px {(cm>100).sum()}/{len(cm)} cores  >200px {(cm>200).sum()}')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
