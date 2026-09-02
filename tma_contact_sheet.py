#!/usr/bin/env python3
"""Contact sheets of TPAF/H&E pairs, for eyeballing a whole run at once.

The build writes only a sampled QC panel, which is too thin to spot systematic
misplacement. This lays every core (or every pair) out side by side as PNG, sorted
worst-first when asked, so failures surface immediately rather than being hunted for.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import cv2
import numpy as np
import tifffile as tiff

PANEL = 260


def norm(a):
    if a.ndim == 3:
        a = a[..., :2].mean(2) if a.shape[2] >= 3 else a[..., 0]
    return cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def panel(d, well, tile, af_mean, tissue):
    b = os.path.join(d, f'{well}_T{tile:03d}')
    af = cv2.cvtColor(cv2.resize(norm(tiff.imread(b + '_AF.tif')), (PANEL, PANEL)),
                      cv2.COLOR_GRAY2RGB)
    he = tiff.imread(b + '_HE_reg.tif')[..., :3]
    he = cv2.resize(he, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    p = np.concatenate([af, he], axis=1)
    cv2.rectangle(p, (0, 0), (p.shape[1] - 1, p.shape[0] - 1), (200, 200, 200), 1)
    cv2.putText(p, f'{well}_T{tile:03d}', (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(p, f'af{af_mean:.0f} ti{tissue:.2f}', (5, p.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default=None, help='Default: <dir>/_sheets')
    ap.add_argument('--mode', choices=('per_core', 'all'), default='per_core',
                    help='per_core: one representative pair per core.')
    ap.add_argument('--sort', choices=('well', 'af', 'tissue'), default='well',
                    help='af/tissue sort ascending, i.e. the doubtful ones first.')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--per_sheet', type=int, default=24)
    args = ap.parse_args()

    out = args.out or os.path.join(args.dir, '_sheets')
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(args.dir, 'pairs.csv'), encoding='utf-8')))
    for r in rows:
        r['tile'] = int(r['tile']); r['af_mean'] = float(r['af_mean']); r['tissue'] = float(r['tissue'])

    if args.mode == 'per_core':
        best = {}
        for r in rows:                      # the strongest tile represents its core
            k = r['well']
            if k not in best or r['af_mean'] * r['tissue'] > best[k]['af_mean'] * best[k]['tissue']:
                best[k] = r
        rows = list(best.values())

    if args.sort == 'well':
        rows.sort(key=lambda r: (int(r['well'][:-1]), r['well'][-1], r['tile']))
    else:
        rows.sort(key=lambda r: r['af_mean'] if args.sort == 'af' else r['tissue'])

    print(f'{len(rows)} panels, {args.cols} cols, {args.per_sheet} per sheet, sorted by {args.sort}')
    n = 0
    for s in range(0, len(rows), args.per_sheet):
        chunk = rows[s:s + args.per_sheet]
        panels = []
        for r in chunk:
            try:
                panels.append(panel(args.dir, r['well'], r['tile'], r['af_mean'], r['tissue']))
            except Exception as e:
                print(f'  [skip] {r["well"]}_T{r["tile"]:03d}: {e}')
        if not panels:
            continue
        ph, pw = panels[0].shape[:2]
        nrow = (len(panels) + args.cols - 1) // args.cols
        sheet = np.full((nrow * ph, args.cols * pw, 3), 255, np.uint8)
        for i, p in enumerate(panels):
            r_, c_ = divmod(i, args.cols)
            sheet[r_ * ph:(r_ + 1) * ph, c_ * pw:(c_ + 1) * pw] = p
        f = os.path.join(out, f'sheet_{args.mode}_{args.sort}_{n:02d}.png')
        cv2.imwrite(f, sheet[..., ::-1])
        print(f'  {os.path.basename(f)}  {len(panels)} panels')
        n += 1
    print(f'-> {out}')


if __name__ == '__main__':
    main()
