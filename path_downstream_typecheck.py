#!/usr/bin/env python3
"""Is HoVer-Net's nuclear *type* call trustworthy enough to report at all?

The density comparison has a floor: two halves of one real H&E region. The type
call has no such split available, but it has something nearly as good -- two
independently trained HoVer-Net heads (PanNuke and MoNuSAC) ran over byte-identical
crops at the same 0.25 um/px, so their instance centroids are directly comparable.
If two H&E-trained models disagree about what a nucleus is *on real H&E*, then a
disagreement on virtual H&E says nothing about the virtual stain.

The taxonomies differ, so only one axis is comparable: epithelial/tumour vs
everything else. PanNuke splits it in two (Neoplastic, Non-Neoplastic Epithelial)
where MoNuSAC has one class (Epithelial); the union is the fair comparison.

Reported:
  match rate    -- fraction of PanNuke instances with a mutual nearest MoNuSAC
                   partner within 2 um. Bounds everything below: unmatched
                   instances cannot disagree about type, they disagree about
                   existence.
  kappa         -- chance-corrected agreement on the epithelial axis, per source.
  epi fraction  -- per-region epithelial share under each model, and the CCC
                   between the two models. This is the number that would go in a
                   paper; the CCC on real H&E is its ceiling.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

ROOT = 'results/path_screen/survey/_downstream'
# The union that makes PanNuke's two-class split comparable to MoNuSAC's one.
EPI = {'neoplastic', 'non-neoplastic epithelial', 'epithelial'}
BG = 'background'
TOL_UM = 2.0
MPP = 0.25


def load(path, min_prob=0.0):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if float(r['prob'] or 0) < min_prob:
            continue
        d[(r['source'], r['id'])].append(
            (float(r['cx']), float(r['cy']), r['type_name']))
    return d


def mutual_nn(a, b, tol):
    """Pairs that pick each other. A one-sided nearest neighbour would happily
    map three crowded PanNuke instances onto one MoNuSAC instance and count all
    three as agreeing; requiring the choice to be mutual makes it a matching."""
    if not a or not b:
        return []
    pa = np.array([[x, y] for x, y, _ in a])
    pb = np.array([[x, y] for x, y, _ in b])
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    ia = d.argmin(1)
    ib = d.argmin(0)
    return [(i, j) for i, j in enumerate(ia)
            if ib[j] == i and d[i, j] <= tol]


def ccc(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan
    return 2 * np.cov(x, y)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def kappa(t11, t10, t01, t00):
    n = t11 + t10 + t01 + t00
    if not n:
        return np.nan
    po = (t11 + t00) / n
    pe = ((t11 + t10) * (t11 + t01) + (t01 + t00) * (t10 + t00)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def epi_frac(rows):
    typed = [t for _, _, t in rows if t.lower() != BG]
    if not typed:
        return np.nan
    return sum(t.lower() in EPI for t in typed) / len(typed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--min_prob', type=float, default=0.0)
    a = ap.parse_args()
    P = load(os.path.join(a.root, 'hv_hovernet_fast_pannuke.csv'), a.min_prob)
    M = load(os.path.join(a.root, 'hv_hovernet_fast_monusac.csv'), a.min_prob)
    srcs = sorted({k[0] for k in P} & {k[0] for k in M})
    tol = TOL_UM / MPP

    print('=' * 76)
    print('PanNuke vs MoNuSAC on identical crops   (min_prob %.2f, tol %.0f um)'
          % (a.min_prob, TOL_UM))
    print('=' * 76)
    print('\n%-14s%10s%10s%12s%10s%10s%9s'
          % ('source', 'n_pan', 'n_mon', 'match rate', 'epi pan', 'epi mon', 'kappa'))
    conf = collections.Counter()
    per_region = collections.defaultdict(lambda: ([], []))
    for s in srcs:
        ids = sorted({k[1] for k in P if k[0] == s} & {k[1] for k in M if k[0] == s})
        np_, nm, nmatch = 0, 0, 0
        t = [0, 0, 0, 0]
        for i in ids:
            ra, rb = P[(s, i)], M[(s, i)]
            np_ += len(ra)
            nm += len(rb)
            fa, fb = epi_frac(ra), epi_frac(rb)
            per_region[s][0].append(fa)
            per_region[s][1].append(fb)
            for x, y in mutual_nn(ra, rb, tol):
                ta, tb = ra[x][2], rb[y][2]
                if ta.lower() == BG or tb.lower() == BG:
                    continue
                nmatch += 1
                ea, eb = ta.lower() in EPI, tb.lower() in EPI
                t[0 if (ea and eb) else 1 if (ea and not eb)
                  else 2 if (eb and not ea) else 3] += 1
                if s == 'real_HE':
                    conf[(ta, tb)] += 1
        x = np.array(per_region[s][0])
        y = np.array(per_region[s][1])
        print('%-14s%10d%10d%11.0f%%%10.2f%10.2f%9.2f'
              % (s, np_, nm, 100 * nmatch / max(np_, 1),
                 np.nanmean(x), np.nanmean(y), kappa(*t)))

    print('\nper-region epithelial fraction, PanNuke vs MoNuSAC (CCC)')
    for s in srcs:
        print('  %-14s%6.2f' % (s, ccc(np.array(per_region[s][0]),
                                       np.array(per_region[s][1]))))
    print('\n  real_HE is the ceiling: it is what the two models manage on the')
    print('  images neither of them has any reason to find strange.')

    print('\nreal_HE confusion, matched pairs   rows PanNuke / cols MoNuSAC')
    rn = sorted({k[0] for k in conf})
    cn = sorted({k[1] for k in conf})
    print('%-28s' % '' + ''.join('%12s' % c[:11] for c in cn))
    for r in rn:
        tot = sum(conf[(r, c)] for c in cn) or 1
        print('%-28s' % r + ''.join('%11.0f%%' % (100 * conf[(r, c)] / tot)
                                    for c in cn))

    print('\nCCC of per-region epithelial fraction vs real_HE, within each model')
    print('%-14s%12s%12s' % ('source', 'PanNuke', 'MoNuSAC'))
    for s in srcs:
        if s == 'real_HE':
            continue
        ids = sorted({k[1] for k in P if k[0] == s} & {k[1] for k in P
                                                       if k[0] == 'real_HE'})
        out = []
        for D in (P, M):
            xx = np.array([epi_frac(D[('real_HE', i)]) for i in ids
                           if (s, i) in D and ('real_HE', i) in D])
            yy = np.array([epi_frac(D[(s, i)]) for i in ids
                           if (s, i) in D and ('real_HE', i) in D])
            out.append(ccc(xx, yy))
        print('%-14s%12.2f%12.2f' % (s, out[0], out[1]))


if __name__ == '__main__':
    main()
