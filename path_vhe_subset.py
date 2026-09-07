#!/usr/bin/env python3
"""Which regions carry nuclear information in TPAF, and so admit nucleus-level metrics.

Nuclei do not autofluoresce. They are visible only as dark holes against NADH, FAD and
collagen, so where that surrounding signal is weak or where another signal covers the
nuclear area, there is nothing for a segmenter to find -- and H&E processing moves
tissue morphology besides, so in some regions nuclear shape and arrangement genuinely
differ between the two modalities. Counting nuclei there measures the modality, not the
virtual stain, and forcing the mask to reach real H&E's nuclear coverage would mean
inventing nuclei.

So nucleus-level metrics get a subset, and this defines it.

The criterion uses TPAF only
----------------------------
Selecting on how well the mask already agrees with real H&E would keep exactly the
regions where the method works and call the result a finding. The question is instead
whether the input carries the information, which is a property of the TPAF image:

    hole_depth    at nucleus scale, how far the dark pixels sit below their own local
                  background, relative to it. Blur at ~2x a nucleus diameter to get the
                  background a nucleus sits in, take the pixels below it, and report the
                  p75 of the relative shortfall over tissue. High where nuclei punch
                  visible holes; low where the tissue is flat or the signal is weak.

The threshold is set once, on this statistic's own distribution, before looking at any
agreement number. What agreement does inside and outside the subset is then a
consequence to be reported, not an input to the choice.

    python path_vhe_subset.py
    python path_vhe_subset.py --pct 40 --out results/.../nuc_subset.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os

import cv2
import numpy as np

SUR = 'results/path_screen/survey/_vhe'
CP = 'results/path_screen/survey/_downstream/cp'
TPAF_MPP = 0.621
NUC_UM = 6.0          # equivalent diameter of the median real nucleus, HoVer-Net
DARK = 12             # TPAF background is near black; below this is not tissue


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def hole_depth(gray):
    """p75 of how far below its local background a dark tissue pixel sits.

    Blurring at 2x a nucleus diameter gives the background the nucleus is a hole in;
    at 1x the nucleus would blur into its own background and the depth would vanish.
    Relative, not absolute, so a bright slide and a dim one are comparable -- absolute
    depth would just re-measure overall brightness.
    """
    g = gray.astype(np.float32)
    tis = g > DARK
    if tis.sum() < 5000:
        return np.nan, 0.0
    sigma = 2 * NUC_UM / TPAF_MPP / 2.355        # FWHM of 2 nucleus diameters
    bg = cv2.GaussianBlur(g, (0, 0), sigma)
    rel = np.zeros_like(g)
    below = (g < bg) & tis
    rel[below] = (bg[below] - g[below]) / np.maximum(bg[below], 1.0)
    return float(np.percentile(rel[tis], 75)), float(tis.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--masks', default=SUR + '/masks_v2')
    ap.add_argument('--pct', type=float, default=50.0,
                    help='Keep regions above this percentile of hole_depth.')
    ap.add_argument('--out', default=SUR + '/nuc_subset.csv')
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    recs = []
    for r in rows:
        name = os.path.splitext(r['stage_name'])[0]
        hp = os.path.join(CP, 'real_HE', r['id'] + '.png')
        mp = os.path.join(args.masks, name + '.png')
        if not (os.path.exists(r['patch_path']) and os.path.exists(hp)
                and os.path.exists(mp)):
            continue
        x, y, w, h = (int(r[k]) for k in ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
        g = cv2.cvtColor(imread_u(r['patch_path']), cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        d, tf = hole_depth(g)
        if not np.isfinite(d):
            continue
        m = float((imread_u(mp, cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w] > 127).mean())
        he = float((imread_u(hp) > 0).mean())
        recs.append(dict(id=r['id'], sample=r['sample'], stage=name,
                         hole_depth=d, tissue_frac=tf, mask_cov=m, he_cov=he,
                         ratio=(m / he if he > 0 else np.nan)))

    d = np.array([x['hole_depth'] for x in recs])
    thr = float(np.percentile(d, args.pct))
    keep = d >= thr
    for x, k in zip(recs, keep):
        x['keep'] = int(k)
    ratio = np.array([x['ratio'] for x in recs])
    fin = np.isfinite(ratio)

    print('%d regions; hole_depth threshold %.4f (p%.0f)' % (len(recs), thr, args.pct))
    print('kept %d, excluded %d\n' % (int(keep.sum()), int((~keep).sum())))
    print('%-14s%8s%14s%14s%12s' % ('', 'n', 'hole_depth', 'mask/real', 'real cov'))
    for lab, sel in (('kept', keep & fin), ('excluded', ~keep & fin)):
        print('%-14s%8d%14.4f%14.2f%12.4f'
              % (lab, int(sel.sum()), np.median(d[sel]), np.median(ratio[sel]),
                 np.median([x['he_cov'] for x, s in zip(recs, sel) if s])))

    print('\n每片子保留比例')
    by = collections.defaultdict(lambda: [0, 0])
    for x in recs:
        by[x['sample']][0] += 1
        by[x['sample']][1] += x['keep']
    print('%-16s%8s%8s%8s' % ('slide', 'n', 'kept', '%'))
    for s in sorted(by):
        n, k = by[s]
        print('%-16s%8d%8d%7.0f%%' % (s, n, k, 100 * k / n))

    with io.open(args.out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'sample', 'stage', 'keep',
                                           'hole_depth', 'tissue_frac', 'mask_cov',
                                           'he_cov', 'ratio'])
        w.writeheader()
        for x in sorted(recs, key=lambda z: -z['hole_depth']):
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in x.items()})
    print('\n-> ' + os.path.abspath(args.out))


if __name__ == '__main__':
    main()
