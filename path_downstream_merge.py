#!/usr/bin/env python3
"""Bring the two segmentation tools together and score them on grading criteria.

Cellpose ran locally over seven sources including the TPAF itself; HoVer-Net ran on
Colab over the six H&E-like ones. TPAF has no HoVer-Net column because HoVer-Net is
an H&E model -- but it is the column that matters most for reading the rest, since
it says how many nuclei the field contains independently of any stain.

What is compared, and why those things. Ovarian serous carcinoma is graded on a
two-tier system whose operative criterion for nuclear atypia is a ratio: grade 3 is
>=3:1 variation in nuclear size across the field, grade 2 under 3:1, grade 1 uniform
round-to-oval nuclei. Alongside sit hyperchromasia, irregular contours and crowding.
So size *variation* leads and median size trails; a virtual stain that reproduces
the median size while flattening the variation has lost the thing being graded on.

Two conventions, both learned the hard way in this project:

  Agreement, not correlation. A stain rendering every nucleus 40% too large still
  correlates perfectly. Lin's CCC and the Bland-Altman bias catch a systematic
  offset; Pearson r does not, and is shown only so the gap is visible.

  No split-half reference. Measuring two halves of one region and comparing them was
  used here as a stand-in for how well a measure agrees with itself. It is not that: it
  measures how spatially uniform the tissue is at that scale -- both halves of a tumour
  region are tumour -- so a method scored as a percentage of it was being scored against
  tissue homogeneity. Dropped 2026-09-06; path_downstream_stats.py no longer writes the
  rows either.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

ROOT = 'results/path_screen/survey/_downstream'
# Ordered by how directly each feeds the grading criteria; see module docstring.
TIER1 = ['density', 'size_ratio', 'solidity', 'circularity', 'hema_od']
TIER2 = ['hema_sd', 'size_cv', 'area_ratio']
TIER3 = ['area_um2', 'eqdiam_um', 'ecc', 'nn_um']


def ccc(x, y):
    """Lin's concordance: correlation penalised by any shift in mean or spread."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, 0
    c = 2 * np.cov(x, y)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    bias = float(np.mean(y - x) / max(abs(np.mean(x)), 1e-9))
    return c, float(np.corrcoef(x, y)[0, 1]), bias, len(x)


def load_cellpose(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    by = {}
    for r in rows:
        by[(r['id'], r['source'])] = r
    return by


def load_hovernet(path, crop_scale=None, min_prob=0.0):
    """Per-region counts and type mix from the per-nucleus rows.

    `type` is a class name string in tiatoolbox 2.x. Nuclei the model declined to
    type come back as Background; they are counted in the total and excluded from
    the type mix, and their share is reported because it differs between sources
    and is itself a signal about how confident the model is on synthetic images.
    """
    area = {}
    if crop_scale and os.path.exists(crop_scale):
        for r in csv.DictReader(open(crop_scale, encoding='utf-8')):
            area[(r['source'], r['id'])] = (int(r['out_w']) * int(r['out_h'])
                                            * float(r['out_mpp']) ** 2 / 1e6)
    n = collections.Counter()
    typ = collections.defaultdict(collections.Counter)
    ar = collections.defaultdict(list)
    probs = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        k = (r['source'], r['id'])
        p = float(r['prob'] or 0)
        probs[r['source']].append(p)
        if p < min_prob:
            continue
        n[k] += 1
        typ[k][r['type_name']] += 1
        try:
            ar[k].append(float(r['area_px']))
        except ValueError:
            pass
    out = {}
    for k in n:
        a = area.get(k)
        px = np.array(ar[k], float) if ar[k] else np.array([np.nan])
        # area_px is polygon area at 0.25 um/px
        um2 = px * 0.25 ** 2
        eq = 2 * np.sqrt(um2 / np.pi)
        out[k] = dict(n=n[k], density=(n[k] / a) if a else np.nan,
                      area_um2=float(np.nanmedian(um2)),
                      size_ratio=(float(np.nanpercentile(eq, 90)
                                        / max(np.nanpercentile(eq, 10), 1e-9))
                                  if len(eq) >= 10 else np.nan),
                      types=typ[k])
    return out, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--cp', default=ROOT + '/cp_morphology.csv')
    ap.add_argument('--hv', default=ROOT + '/hv_hovernet_fast_pannuke.csv')
    ap.add_argument('--min_prob', type=float, default=0.0,
                    help='Drop HoVer-Net nuclei below this type confidence.')
    args = ap.parse_args()

    cp = load_cellpose(args.cp)
    ids = sorted({k[0] for k in cp if k[1] == 'real_HE'})
    srcs = sorted({k[1] for k in cp} - {'real_HE'})
    num = lambda r, k: (float(r[k]) if r and r.get(k) not in (None, '', 'nan')
                        else np.nan)

    print('=' * 78)
    print('CELLPOSE  (%d regions)' % len(ids))
    print('=' * 78)
    for tier, keys in (('TIER 1  grading criteria', TIER1),
                       ('TIER 2  supporting', TIER2),
                       ('TIER 3  not graded on', TIER3)):
        print('\n%s' % tier)
        print('%-16s' % '' + ''.join('%16s' % k[:15] for k in keys))
        for s in srcs:
            cells = []
            for k in keys:
                x = np.array([num(cp.get((i, 'real_HE')), k) for i in ids
                              if (i, s) in cp])
                y = np.array([num(cp.get((i, s)), k) for i in ids if (i, s) in cp])
                c, _, bias, _ = ccc(x, y)
                cells.append('%5.2f %+5.0f%%' % (c, 100 * bias))
            print('%-16s' % s + ''.join('%16s' % v for v in cells))
    print('\n  cell = CCC and Bland-Altman bias against real H&E')

    if not os.path.exists(args.hv):
        print('\n(no HoVer-Net csv at %s yet)' % args.hv)
        return
    hv, probs = load_hovernet(args.hv, os.path.join(args.root, 'crop_scale.csv'),
                              args.min_prob)
    hsrc = sorted({k[0] for k in hv})
    print('\n' + '=' * 78)
    # The model name comes from the csv filename (hv_hovernet_fast_<model>.csv);
    # two models were run and mislabelling one as the other would be easy to miss.
    tag = os.path.basename(args.hv)[len('hv_hovernet_fast_'):-len('.csv')]         if os.path.basename(args.hv).startswith('hv_hovernet_fast_') else '?'
    print('HOVER-NET  (%s, %d sources)' % (tag, len(hsrc)))
    print('=' * 78)
    print('\n%-14s%10s%10s%12s%12s%10s' % ('source', 'density', 'size_ratio',
                                           'area_um2', 'untyped %', 'prob p50'))
    for s in hsrc:
        rows = [hv[(s, i)] for i in ids if (s, i) in hv]
        if not rows:
            continue
        # 1.x wrote the class as an integer mapped to a lowercase name, 2.x writes
        # the name itself capitalised. Match on either.
        unt = np.mean([sum(v for k, v in r['types'].items()
                           if k.lower() == 'background')
                       / max(sum(r['types'].values()), 1) for r in rows])
        print('%-14s%10.0f%10.2f%12.1f%12.1f%10.3f'
              % (s, np.nanmedian([r['density'] for r in rows]),
                 np.nanmedian([r['size_ratio'] for r in rows]),
                 np.nanmedian([r['area_um2'] for r in rows]),
                 100 * unt, np.median(probs[s]) if probs[s] else np.nan))

    print('\nagreement with real H&E, per region')
    print('%-14s%22s%22s' % ('source', 'density CCC/r/bias', 'size_ratio CCC/r/bias'))
    for s in hsrc:
        if s == 'real_HE':
            continue
        line = []
        for k in ('density', 'size_ratio'):
            common = [i for i in ids if (s, i) in hv and ('real_HE', i) in hv]
            x = np.array([hv[('real_HE', i)][k] for i in common])
            y = np.array([hv[(s, i)][k] for i in common])
            c, r_, b, nn = ccc(x, y)
            line.append('%5.2f %5.2f %+5.0f%%' % (c, r_, 100 * b))
        print('%-14s%22s%22s' % (s, line[0], line[1]))

    print('\ntype mix, %% of typed nuclei')
    names = sorted({t for s in hsrc for i in ids if (s, i) in hv
                    for t in hv[(s, i)]['types'] if t.lower() != 'background'})
    print('%-14s' % 'source' + ''.join('%14s' % n[:13] for n in names))
    for s in hsrc:
        tot = collections.Counter()
        for i in ids:
            if (s, i) in hv:
                tot.update({k: v for k, v in hv[(s, i)]['types'].items()
                            if k.lower() != 'background'})
        n_all = sum(tot.values()) or 1
        print('%-14s' % s + ''.join('%14.1f' % (100 * tot[n] / n_all) for n in names))


if __name__ == '__main__':
    main()
