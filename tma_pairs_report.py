#!/usr/bin/env python3
"""Backfill pairs.csv for pairs written before it was incremental, then stratify.

The first 17 cores were built when pairs.csv was only written at the end of a run,
and every one of those runs was killed first, so 149 pairs have files but no row.
tissue and af_mean -- the two columns used to select a training subset -- are
recoverable from the saved images; he_y/he_x are not, and are left blank.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import cv2
import numpy as np
import tifffile as tiff

FIELDS = ['well', 'tile', 'cell', 'he_y', 'he_x', 'side', 'tissue', 'af_mean',
          'disc_r', 'mosaic_h', 'mosaic_w']
NAME = re.compile(r'^(?P<well>\d{1,2}[A-H])_T(?P<tile>\d{3})_AF\.tif$')
LAYOUT = {9: '00', 8: '01', 7: '02', 4: '10', 5: '11', 6: '12', 3: '20', 2: '21', 1: '22'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--af_min', type=float, default=8.0,
                    help='TPAF tile mean below which the pair is too dim to be useful.')
    ap.add_argument('--tissue_min', type=float, default=0.25)
    args = ap.parse_args()

    p = os.path.join(args.dir, 'pairs.csv')
    rows = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            rows[(r['well'], int(r['tile']))] = r

    on_disk = []
    for f in sorted(os.listdir(args.dir)):
        m = NAME.match(f)
        if m:
            on_disk.append((m.group('well'), int(m.group('tile'))))

    missing = [k for k in on_disk if k not in rows]
    print(f'{len(on_disk)} pairs on disk, {len(rows)} in csv, {len(missing)} to backfill')
    for n, (well, tile) in enumerate(missing, 1):
        base = os.path.join(args.dir, f'{well}_T{tile:03d}')
        af = tiff.imread(base + '_AF.tif')
        he = tiff.imread(base + '_HE.tif')
        hg = cv2.cvtColor(he[..., :3], cv2.COLOR_RGB2GRAY) if he.ndim == 3 else he
        rows[(well, tile)] = dict(
            well=well, tile=tile, cell=LAYOUT.get(tile, ''), he_y='', he_x='',
            side=he.shape[0], tissue=round(float((hg < 205).mean()), 3),
            af_mean=round(float(af[..., :2].mean() if af.ndim == 3 else af.mean()), 2),
            disc_r='', mosaic_h='', mosaic_w='')
        if n % 40 == 0:
            print(f'  {n}/{len(missing)}', flush=True)

    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows, key=lambda k: (int(k[0][:-1]), k[0][-1], k[1])):
            w.writerow({f: rows[k].get(f, '') for f in FIELDS})
    print(f'pairs.csv now has {len(rows)} rows\n')

    af = np.array([float(r['af_mean']) for r in rows.values()])
    ti = np.array([float(r['tissue']) for r in rows.values()])
    wells = sorted({r['well'] for r in rows.values()})
    print(f'cores {len(wells)}   pairs {len(rows)}')
    print(f'af_mean  min {af.min():.1f}  median {np.median(af):.1f}  max {af.max():.1f}')
    print(f'tissue   min {ti.min():.2f}  median {np.median(ti):.2f}  max {ti.max():.2f}\n')

    print(f'{"subset":<34}{"pairs":>7}{"cores":>7}')
    for label, sel in (
            ('all', np.ones(len(af), bool)),
            (f'af_mean >= {args.af_min}', af >= args.af_min),
            (f'tissue >= {args.tissue_min}', ti >= args.tissue_min),
            (f'both', (af >= args.af_min) & (ti >= args.tissue_min))):
        ws = {r['well'] for r, s in zip(rows.values(), sel) if s}
        print(f'{label:<34}{int(sel.sum()):>7}{len(ws):>7}')

    print('\nper-core pair count:')
    from collections import Counter
    c = Counter(r['well'] for r in rows.values())
    for n, k in sorted(Counter(c.values()).items()):
        print(f'  {n} pairs: {k} cores')


if __name__ == '__main__':
    main()
