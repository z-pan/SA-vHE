#!/usr/bin/env python3
"""Nuclei morphology per region, weighted toward what ovarian cancer grading uses.

The metrics are chosen from the WHO 2020 two-tier system for ovarian serous carcinoma,
not from what is easy to compute. That system grades on nuclear atypia, and its
operative criterion is a RATIO: grade 3 (high-grade) is >=3:1 variation in nuclear size
across the field, grade 2 is <3:1, grade 1 is uniform round-to-oval nuclei. Alongside
it: hyperchromasia, pleomorphic and irregular nuclear contours, crowding and loss of
polarity, prominent nucleoli.

So the headline quantity is size VARIATION within a field, not median size. An earlier
version of this script reported medians throughout, which measures something the
grading system does not use -- and reported that every virtual variant failed to track
the median while its floor was 0.76, a finding about a quantity no pathologist grades
on.

    tier 1   size_ratio_p90_p10   the 3:1 rule
             solidity, circularity   contour irregularity
             density                 crowding
             hema_od                 hyperchromasia
    tier 2   hema_sd                 chromatin coarseness, nucleoli
    tier 3   median area / eqdiam    only meaningful inside the ratio
             eccentricity, nn_um     redundant or non-standard

Nucleus-to-cytoplasm ratio is a grading feature and is NOT here: it needs cytoplasm
segmentation, which none of these models provide.

Everything is reported in physical units. The label images are at their source's own
resolution -- 0.621 um/px for TPAF and every virtual variant, 0.44-0.77 and varying
region to region for real H&E -- so a count in pixels or an area in pixels would not be
the same quantity on the two sides.

Two things this is careful about.

Agreement, not correlation. A virtual stain that renders every nucleus at 1.4x the real
size still correlates perfectly with the truth. Concordance (Lin's CCC) and the
Bland-Altman bias are what catch a systematic offset; Pearson r does not, and is
reported only so the gap between the two is visible.

A floor. Adjacent halves of the same real H&E region differ from each other by some
amount that owes nothing to virtual staining. Until that is measured, "vHE is within
8% of real" means nothing -- the same mistake the colour work made, where dE 0.97
turned out to sit at the 45th percentile of the estimator's own noise.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

ROOT = 'results/path_screen/survey/_downstream'
SUR = 'results/path_screen/survey/_vhe'
TPAF_MPP = 0.621


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def props(lab, mpp, half=None, rgb=None):
    """Per-nucleus morphology in microns. `half` restricts to x < / >= the midline.

    `rgb` enables the stain-dependent features: haematoxylin optical density inside
    each nucleus (hyperchromasia) and its spread (chromatin coarseness).
    """
    if half is not None:
        w = lab.shape[1]
        keep = np.zeros_like(lab, bool)
        if half == 0:
            keep[:, :w // 2] = True
        else:
            keep[:, w // 2:] = True
        lab = np.where(keep, lab, 0)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    if ids.size == 0:
        return None
    area, ecc, sol, cen, circ, hod, hsd = [], [], [], [], [], [], []
    hem = None
    if rgb is not None:
        from skimage.color import rgb2hed
        hem = rgb2hed(rgb.astype(np.float32) / 255.0)[..., 0]
    for i in ids:
        m = (lab == i).astype(np.uint8)
        a = int(m.sum())
        if a < 4:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area.append(a * mpp * mpp)
        per = cv2.arcLength(c, True)
        circ.append(4 * np.pi * cv2.contourArea(c) / (per * per) if per > 1e-6 else np.nan)
        if hem is not None:
            v = hem[m > 0]
            hod.append(float(v.mean())); hsd.append(float(v.std()))
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        sol.append(cv2.contourArea(c) / ha if ha > 1e-6 else np.nan)
        if len(c) >= 5:
            (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
            a_, b_ = max(ax1, ax2), min(ax1, ax2)
            ecc.append(np.sqrt(1 - (b_ / a_) ** 2) if a_ > 1e-6 else np.nan)
        else:
            ecc.append(np.nan)
        M = cv2.moments(m)
        if M['m00'] > 0:
            cen.append((M['m01'] / M['m00'] * mpp, M['m10'] / M['m00'] * mpp))
    if not area:
        return None
    area = np.array(area)
    cen = np.array(cen)
    nn = np.nan
    if len(cen) > 1:
        d = np.hypot(cen[:, None, 0] - cen[None, :, 0], cen[:, None, 1] - cen[None, :, 1])
        np.fill_diagonal(d, np.inf)
        nn = float(np.median(d.min(1)))
    mm2 = lab.shape[0] * lab.shape[1] * mpp * mpp / 1e6
    if half is not None:
        mm2 /= 2
    eq = 2 * np.sqrt(area / np.pi)
    # The grading criterion is a ratio across the field. p90/p10 rather than max/min so
    # one blown-up segmentation error cannot decide the grade.
    ratio = (float(np.percentile(eq, 90) / max(np.percentile(eq, 10), 1e-6))
             if len(eq) >= 10 else np.nan)
    ratio_a = (float(np.percentile(area, 90) / max(np.percentile(area, 10), 1e-6))
               if len(area) >= 10 else np.nan)
    return dict(n=len(area), density=len(area) / mm2,
                size_ratio=ratio, area_ratio=ratio_a,
                size_cv=float(np.std(eq) / max(np.mean(eq), 1e-9)),
                solidity=float(np.nanmedian(sol)),
                circularity=float(np.nanmedian(circ)),
                hema_od=float(np.nanmedian(hod)) if hod else np.nan,
                hema_sd=float(np.nanmedian(hsd)) if hsd else np.nan,
                area_um2=float(np.median(area)), eqdiam_um=float(np.median(eq)),
                ecc=float(np.nanmedian(ecc)), nn_um=nn,
                area_p90=float(np.percentile(area, 90)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default=ROOT + '/cp_morphology.csv')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding='utf-8-sig')))
    if args.limit:
        rows = rows[:args.limit]
    mpp_he = {}
    for r in rows:
        he = imread_u(r['he_path'])
        if he is not None:
            mpp_he[r['id']] = float(r['crop_um']) / he.shape[1]

    # Where each source's pixels live, so the stain-dependent features can be read
    # from the same image the labels came from.
    img_of = {}
    for r in rows:
        name = os.path.splitext(r['stage_name'])[0]
        x, y = int(r['crop_x']), int(r['crop_y'])
        w, h = int(r['crop_w']), int(r['crop_h'])
        img_of[('real_HE', r['id'])] = (r['he_path'], None)
        img_of[('TPAF', r['id'])] = (r['patch_path'], (x, y, w, h))
        for v in os.listdir(os.path.join(SUR, 'stained')):
            img_of[(v, r['id'])] = (os.path.join(SUR, 'stained', v, name + '.png'),
                                    (x, y, w, h))

    def source_rgb(src, rid):
        e = img_of.get((src, rid))
        if not e or not os.path.exists(e[0]):
            return None
        im = imread_u(e[0])
        if im is None:
            return None
        if e[1]:
            x, y, w, h = e[1]
            im = im[y:y + h, x:x + w]
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else None

    srcs = sorted(os.listdir(os.path.join(args.root, 'cp')))
    cols = ['n', 'density', 'size_ratio', 'area_ratio', 'size_cv', 'solidity',
            'circularity', 'hema_od', 'hema_sd', 'area_um2', 'eqdiam_um', 'ecc',
            'nn_um', 'area_p90']
    out = []
    for src in srcs:
        d = os.path.join(args.root, 'cp', src)
        for f in sorted(os.listdir(d)):
            rid = f[:-4]
            lab = imread_u(os.path.join(d, f))
            if lab is None:
                continue
            mpp = mpp_he.get(rid, TPAF_MPP) if src == 'real_HE' else TPAF_MPP
            rgb = source_rgb(src, rid)
            p = props(lab, mpp, rgb=rgb)
            if p:
                out.append(dict(source=src, id=rid, half='', mpp=round(mpp, 4), **p))
            if src == 'real_HE':
                # split-half floor: two halves of the same region, same stain, same
                # scanner, differing only in which tissue is on each side
                for h in (0, 1):
                    ph = props(lab, mpp, half=h, rgb=rgb)
                    if ph:
                        out.append(dict(source='real_HE_half', id=rid, half=str(h),
                                        mpp=round(mpp, 4), **ph))
        print(f'{src}: done', flush=True)

    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'id', 'half', 'mpp'] + cols)
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f'{len(out)} rows -> {args.out}')


if __name__ == '__main__':
    main()
