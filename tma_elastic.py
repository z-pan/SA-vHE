#!/usr/bin/env python3
"""Elastic refinement of TPAF/H&E pairs, with a held-out check that it is real.

Justified because the two modalities are the *same physical section* -- TPAF imaged
unstained, then stained and rescanned -- so residual non-rigid difference comes from
coverslipping, tissue swelling and the objective-vs-scanner distortion difference.
On serial sections this would be fabricating correspondence instead, and must not run.

Guards, because a deformation field can always drive a similarity metric down without
improving true correspondence:

  coarse grid     6x6 B-spline control points over ~1126 px. A core is 1.66 mm; a fine
                  grid would model noise.
  capped          fitted displacement above --max_um is rejected and the pair keeps its
                  rigid alignment. A large "correction" means the rigid stage is wrong,
                  and elastic must not paper over that.
  held out        the fit sees only coarse tissue structure (Gaussian sigma 8). Scoring
                  is done on high-pass detail the fit never saw -- if only the fitted
                  band improves, that is overfitting, and the pair is rejected.

The last one matters beyond registration quality: if these pairs are used to *evaluate*
virtual staining, warping H&E onto TPAF bakes in the very correspondence the model is
supposed to produce. Every accepted deformation is recorded in elastic.csv so that
downstream work can see how much each pair was moved.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import cv2
import numpy as np
import SimpleITK as sitk
import tifffile as tiff

TPAF_UM = 0.621
NAME = re.compile(r'^(?P<well>\d{1,2}[A-H])_T(?P<tile>\d{3})_AF\.tif$')


def tissue_map(a, invert, sigma):
    """Continuous tissue-high map. Binary masks make a degenerate metric."""
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY) if invert else a[..., :2].mean(2)
    a = a.astype(np.float32)
    if invert:
        a = 255.0 - a
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def highpass(a, invert, sigma=8, keep=2):
    """Detail the fit never saw: band below the fitting scale."""
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY) if invert else a[..., :2].mean(2)
    a = a.astype(np.float32)
    if invert:
        a = 255.0 - a
    fine = cv2.GaussianBlur(a, (0, 0), keep)
    return fine - cv2.GaussianBlur(a, (0, 0), sigma)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def register(fixed_np, moving_np, grid=6, iters=120):
    fixed = sitk.GetImageFromArray(fixed_np)
    moving = sitk.GetImageFromArray(moving_np)
    tx = sitk.BSplineTransformInitializer(fixed, [grid, grid], order=3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=1234)
    R.SetInterpolator(sitk.sitkLinear)
    # a plain LBFGSB run on a correlation metric happily folds the field; bound the
    # step and stop early instead of letting it chase texture
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iters,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=True)
    R.SetShrinkFactorsPerLevel([4, 2])
    R.SetSmoothingSigmasPerLevel([2, 1])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    R.Execute(fixed, moving)
    return tx


def jacobian_stats(tx, shape):
    """Fold detection. A physical deformation keeps det(J) > 0 and near 1; swirls and
    folds drive it to zero or negative. The held-out detail metric does NOT catch this
    -- swirl artefacts live in exactly that band and inflate the score -- so geometry
    has to be checked separately."""
    f = sitk.TransformToDisplacementField(
        tx, sitk.sitkVectorFloat64, shape[::-1], [0.0, 0.0], [1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0])
    j = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(f))
    return float(j.min()), float(np.percentile(j, 1)), float(np.percentile(j, 99))


def displacement_stats(tx, shape):
    f = sitk.TransformToDisplacementField(
        tx, sitk.sitkVectorFloat64, shape[::-1], [0.0, 0.0], [1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0])
    d = sitk.GetArrayFromImage(f)
    mag = np.hypot(d[..., 0], d[..., 1])
    return float(np.median(mag)), float(np.percentile(mag, 95)), float(mag.max())


def apply_tx(img, tx):
    ref = sitk.GetImageFromArray(img[..., 0] if img.ndim == 3 else img)
    if img.ndim == 2:
        return sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(img), ref, tx,
                                                    sitk.sitkLinear, 255.0))
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        m = sitk.GetImageFromArray(img[..., c].astype(np.float32))
        out[..., c] = np.clip(sitk.GetArrayFromImage(
            sitk.Resample(m, ref, tx, sitk.sitkLinear, 255.0)), 0, 255).astype(img.dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='Rigidly aligned pairs (v3).')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_um', type=float, default=50.0, help='Reject above this p95 displacement.')
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--min_jac', type=float, default=0.35,
                    help='Reject if det(J) dips below this anywhere: the field has folded.')
    ap.add_argument('--fit_sigma', type=float, default=8.0)
    ap.add_argument('--min_gain', type=float, default=0.005,
                    help='Required improvement in held-out detail NCC.')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--wells', default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    want = {w.strip() for w in args.wells.split(',')} if args.wells else None
    pairs = []
    for f in sorted(os.listdir(args.dir)):
        m = NAME.match(f)
        if m and (want is None or m.group('well') in want):
            pairs.append((m.group('well'), int(m.group('tile'))))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f'{len(pairs)} pairs; cap {args.max_um} um = {args.max_um/TPAF_UM:.0f} TPAF px', flush=True)

    csv_p = os.path.join(args.out_dir, 'elastic.csv')
    new = not (os.path.exists(csv_p) and os.path.getsize(csv_p) > 0)
    fh = open(csv_p, 'w' if new else 'a', newline='', encoding='utf-8')
    wr = csv.DictWriter(fh, fieldnames=['well', 'tile', 'accepted', 'reason',
                                        'disp_median_um', 'disp_p95_um', 'disp_max_um',
                                        'fit_ncc_before', 'fit_ncc_after',
                                        'held_ncc_before', 'held_ncc_after', 'held_gain',
                                        'jac_min', 'jac_p01', 'jac_p99'])
    if new:
        wr.writeheader(); fh.flush()

    n_ok = n_cap = n_over = n_fold = 0
    for i, (well, tile) in enumerate(pairs, 1):
        b = os.path.join(args.dir, f'{well}_T{tile:03d}')
        af = tiff.imread(b + '_AF.tif')
        he = tiff.imread(b + '_HE_reg.tif')
        if he.shape[:2] != af.shape[:2]:
            he = cv2.resize(he, (af.shape[1], af.shape[0]), interpolation=cv2.INTER_AREA)

        fx = tissue_map(af, invert=False, sigma=args.fit_sigma)
        mv = tissue_map(he, invert=True, sigma=args.fit_sigma)
        hf = highpass(af, invert=False, sigma=args.fit_sigma)
        hm_before = highpass(he, invert=True, sigma=args.fit_sigma)

        fit_before = ncc(fx, mv)
        held_before = ncc(hf, hm_before)
        try:
            tx = register(fx, mv, grid=args.grid)
        except Exception as e:
            print(f'  [err] {well}_T{tile:03d}: {e}', flush=True); continue
        med, p95, mx = (v * TPAF_UM for v in displacement_stats(tx, af.shape[:2]))
        jmin, j01, j99 = jacobian_stats(tx, af.shape[:2])

        warped = apply_tx(he, tx)
        fit_after = ncc(fx, tissue_map(warped, invert=True, sigma=args.fit_sigma))
        held_after = ncc(hf, highpass(warped, invert=True, sigma=args.fit_sigma))
        gain = held_after - held_before

        reason = ''
        if jmin < args.min_jac:
            reason = f'jacobian min {jmin:.2f} < {args.min_jac} (folding)'
            n_fold += 1
        elif p95 > args.max_um:
            reason = f'disp p95 {p95:.0f}um > {args.max_um:.0f}'
            n_cap += 1
        elif gain < args.min_gain:
            reason = f'held-out gain {gain:+.4f} < {args.min_gain}'
            n_over += 1
        accepted = reason == ''
        if accepted:
            tiff.imwrite(os.path.join(args.out_dir, f'{well}_T{tile:03d}_HE_ela.tif'),
                         warped, compression='zlib')
            n_ok += 1

        wr.writerow(dict(well=well, tile=tile, accepted=int(accepted), reason=reason,
                         disp_median_um=round(med, 1), disp_p95_um=round(p95, 1),
                         disp_max_um=round(mx, 1),
                         fit_ncc_before=round(fit_before, 4), fit_ncc_after=round(fit_after, 4),
                         held_ncc_before=round(held_before, 4), held_ncc_after=round(held_after, 4),
                         held_gain=round(gain, 4), jac_min=round(jmin, 3),
                         jac_p01=round(j01, 3), jac_p99=round(j99, 3)))
        fh.flush()
        if i % 10 == 0 or i <= 5:
            print(f'  {i}/{len(pairs)} {well}_T{tile:03d} disp p95 {p95:>5.1f}um  '
                  f'fit {fit_before:.3f}->{fit_after:.3f}  held {held_before:.3f}->{held_after:.3f}'
                  f'  J[{jmin:.2f},{j99:.2f}]  {"OK" if accepted else reason}', flush=True)
    fh.close()
    print(f'\naccepted {n_ok}   rejected: {n_cap} over cap, {n_over} no held-out gain')
    print(f'-> {args.out_dir}')


if __name__ == '__main__':
    main()
