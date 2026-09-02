#!/usr/bin/env python3
"""Grid-search the partitioned stain correction against real H&E compartment statistics.

The multipliers in stain_color_correction.correct_masked (H x1.5 inside nuclei, x0.5
outside) were tuned on a different set and, it turns out, against a different definition
of "nucleus": with nuclei segmented on the real H&E here, real nuclear R-B measures -2.1
and cytoplasmic +9.5, where the values those multipliers were fitted to were -43.1 and
-2.1. Multipliers are only meaningful relative to the mask that defines the compartments,
so they have to be refitted rather than carried over.

What is being fitted, and to what

  target        real H&E, per compartment: nuclear and cytoplasmic R-B, haematoxylin and
                eosin concentration. Four numbers per compartment, eight in all.
  free          h_in, h_out, e_in, e_out -- haematoxylin and eosin scaling inside and
                outside the blurred nuclear mask. e_in is fixed at 1.0 in the original
                function; it is freed here because the eosin split is half of what
                distinguishes the compartments in real H&E.
  also free     whether the global stat-match runs first. It changes what the multipliers
                are correcting, so fitting them without deciding that is fitting two
                things at once.

Errors are divided by the spread of that statistic across the real regions before being
summed. R-B lives in grey levels and H in concentration units around 0.01; added raw,
the colour terms would decide everything and the stain terms would be noise.

The point of the search is not only the best setting. It is whether the best setting is
good enough: if the optimum still sits far from real H&E, no post-processing parameter
will fix it and the shortfall is in the generator, not in the correction.

    python path_vhe_tune.py
    python path_vhe_tune.py --apply gray_tuned    # write the winner out
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import os
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import hed2rgb, rgb2hed

# The D of HED is the third stain vector, the one neither H nor E accounts for. Real
# H&E has almost none of it. The correction never scales D directly -- only H and E are
# multiplied -- but it moves anyway, because hed2rgb is clipped to [0,1] and a setting
# that drives a channel out of range comes back with the residue in D. That shows up in
# the image as an orange-brown cast over the cytoplasm.
#
# It was left out of the first objective, and the search promptly exploited it: the
# winning setting scored well on all six colour and stain terms while quietly pushing D
# up, so the numbers improved and the pictures got worse. A term that is not in the
# objective is a term the search is free to spend.
STATS = ['nuc_rb', 'cyt_rb', 'nuc_H', 'cyt_H', 'nuc_E', 'cyt_E', 'nuc_D', 'cyt_D']


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


def region_stats(rgb, is_nuc, tissue):
    """R-B, H and E means for each compartment, on tissue pixels only."""
    hed = rgb2hed(rgb)
    out = {}
    for name, sel in (('nuc', tissue & is_nuc), ('cyt', tissue & ~is_nuc)):
        if sel.sum() < 200:
            out[name + '_rb'] = out[name + '_H'] = np.nan
            out[name + '_E'] = out[name + '_D'] = np.nan
            continue
        px = rgb[sel]
        out[name + '_rb'] = float((px[..., 0] - px[..., 2]).mean() * 255)
        out[name + '_H'] = float(hed[sel][:, 0].mean())
        out[name + '_E'] = float(hed[sel][:, 1].mean())
        out[name + '_D'] = float(hed[sel][:, 2].mean())
    return out


def sample_region(rgb, mask, blur, n_px, rng, white_thresh=0.92):
    """Pixels kept for the search: HED, mask weight, and which compartment.

    Sampled rather than whole because the search evaluates hundreds of settings and
    hed2rgb is the cost. A few thousand pixels per region fixes each mean to well under
    the differences being resolved, and the sample is drawn once so every setting is
    scored on identical pixels.
    """
    weight = gaussian_filter(mask.astype(np.float32), blur)
    tissue = rgb.mean(2) < white_thresh
    idx = np.flatnonzero(tissue)
    if idx.size == 0:
        return None
    if idx.size > n_px:
        idx = rng.choice(idx, n_px, replace=False)
    flat = rgb.reshape(-1, 3)[idx]
    hed = rgb2hed(flat.reshape(-1, 1, 3)).reshape(-1, 3)
    return hed, weight.reshape(-1)[idx], mask.reshape(-1)[idx] > 0.5


def apply_masked(hed, w, h_in, h_out, e_in, e_out):
    out = hed.copy()
    out[:, 0] *= h_out + (h_in - h_out) * w
    out[:, 1] *= e_out + (e_in - e_out) * w
    return np.clip(hed2rgb(out.reshape(-1, 1, 3)).reshape(-1, 3), 0.0, 1.0)


def score_params(regions, h_in, h_out, e_in, e_out):
    """Pooled compartment means over all regions for one setting."""
    acc = {k: [] for k in STATS}
    for hed, w, is_nuc in regions:
        rgb = apply_masked(hed, w, h_in, h_out, e_in, e_out)
        hed2 = rgb2hed(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        for name, sel in (('nuc', is_nuc), ('cyt', ~is_nuc)):
            if sel.sum() < 50:
                continue
            acc[name + '_rb'].append((rgb[sel, 0] - rgb[sel, 2]).mean() * 255)
            acc[name + '_H'].append(hed2[sel, 0].mean())
            acc[name + '_E'].append(hed2[sel, 1].mean())
            acc[name + '_D'].append(hed2[sel, 2].mean())
    return {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--stained', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--masks', default='results/path_screen/survey/_vhe/masks')
    ap.add_argument('--masks_he', default='results/path_screen/survey/_vhe/masks_he')
    ap.add_argument('--base', action='append', default=None,
                    help='Stained variants to search over. Defaults to gray and '
                         'gray_globalonly, i.e. with and without the global step.')
    ap.add_argument('--n_px', type=int, default=6000, help='Pixels sampled per region.')
    ap.add_argument('--regions', type=int, default=0, help='Use only the first N.')
    ap.add_argument('--blur', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--apply', default=None,
                    help='After the search, write every patch corrected with the best '
                         'setting to stained/<this name>.')
    args = ap.parse_args()

    bases = args.base or ['gray', 'gray_globalonly']
    rows = [r for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig'))
            if r['he_path']]
    if args.regions:
        rows = rows[:args.regions]
    rng = np.random.default_rng(args.seed)

    # --- the target, from real H&E, with its spread for weighting
    print(f'reading {len(rows)} real H&E regions for the target', flush=True)
    real = []
    for r in rows:
        bgr = imread_u(r['he_path'])
        m = imread_u(os.path.join(args.masks_he, r['id'] + '.png'),
                     cv2.IMREAD_GRAYSCALE) > 127
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        real.append(region_stats(rgb, m, rgb.mean(2) < 0.92))
    target = {k: float(np.nanmean([d[k] for d in real])) for k in STATS}
    spread = {k: float(np.nanstd([d[k] for d in real])) for k in STATS}
    print(f'{"":<10}' + ''.join(f'{k:>10}' for k in STATS))
    print(f'{"real":<10}' + ''.join(f'{target[k]:>10.4f}' for k in STATS))
    print(f'{"spread":<10}' + ''.join(f'{spread[k]:>10.4f}' for k in STATS))
    print()

    def err(got):
        return float(np.mean([abs(got[k] - target[k]) / (spread[k] + 1e-9)
                              for k in STATS]))

    best = None
    for base in bases:
        print(f'sampling {base}', flush=True)
        regions = []
        for r in rows:
            name = os.path.splitext(r['stage_name'])[0]
            p = os.path.join(args.stained, base, name + '.png')
            if not os.path.exists(p):
                continue
            x, y = int(r['crop_x']), int(r['crop_y'])
            w, h = int(r['crop_w']), int(r['crop_h'])
            rgb = cv2.cvtColor(imread_u(p)[y:y + h, x:x + w],
                               cv2.COLOR_BGR2RGB).astype(np.float64) / 255
            m = imread_u(os.path.join(args.masks, name + '.png'),
                         cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w] > 127
            s = sample_region(rgb, m, args.blur, args.n_px, rng)
            if s:
                regions.append(s)
        base_got = score_params(regions, 1.0, 1.0, 1.0, 1.0)
        print(f'  {len(regions)} regions, {sum(len(r[0]) for r in regions)} px; '
              f'uncorrected error {err(base_got):.3f}', flush=True)

        # Wide enough that the optimum is interior. A first pass with h_in from 1.0
        # and h_out from 0.2 put every one of the best five on a grid edge, which is
        # the search reporting the boundary rather than a minimum.
        # Wide enough that the optimum is interior. Two passes were needed: the first
        # ran h_in from 1.0 and h_out from 0.2 and put every one of the best five on a
        # grid edge, which is the search reporting the boundary rather than a minimum;
        # the second still pinned h_in at its 3.0 ceiling. An optimum on an edge is not
        # an optimum, it is the largest value that was allowed.
        grid = dict(h_in=[0.6, 1.0, 1.4, 1.8, 2.4, 3.0, 3.6, 4.4, 5.2, 6.0, 7.0],
                    h_out=[0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0],
                    e_in=[0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3],
                    e_out=[0.2, 0.35, 0.5, 0.7, 1.0, 1.3])
        combos = list(itertools.product(*grid.values()))
        t0 = time.time()
        results = []
        for i, (hi, ho, ei, eo) in enumerate(combos, 1):
            got = score_params(regions, hi, ho, ei, eo)
            results.append((err(got), hi, ho, ei, eo, got))
            if i % 100 == 0:
                print(f'    {i}/{len(combos)}  {time.time() - t0:.0f}s', flush=True)
        results.sort(key=lambda t: t[0])
        print(f'  best five for {base}:')
        print(f'    {"err":>7}{"h_in":>7}{"h_out":>7}{"e_in":>7}{"e_out":>7}'
              + ''.join(f'{k:>10}' for k in STATS))
        for e, hi, ho, ei, eo, got in results[:5]:
            print(f'    {e:>7.3f}{hi:>7.1f}{ho:>7.1f}{ei:>7.1f}{eo:>7.1f}'
                  + ''.join(f'{got[k]:>10.4f}' for k in STATS))
        if best is None or results[0][0] < best[0]:
            best = (results[0][0], base) + results[0][1:]
        print()

    e, base, hi, ho, ei, eo, got = best
    print('=' * 78)
    print(f'best overall: {base}  h_in {hi}  h_out {ho}  e_in {ei}  e_out {eo}')
    print(f'  normalised error {e:.3f} (0 would be exact; 1 means off by one '
          'between-region standard deviation on average)')
    print(f'{"":<10}' + ''.join(f'{k:>10}' for k in STATS))
    print(f'{"real":<10}' + ''.join(f'{target[k]:>10.4f}' for k in STATS))
    print(f'{"tuned":<10}' + ''.join(f'{got[k]:>10.4f}' for k in STATS))
    print(f'{"diff/sd":<10}'
          + ''.join(f'{(got[k] - target[k]) / (spread[k] + 1e-9):>10.2f}' for k in STATS))

    if args.apply:
        out_dir = os.path.join(args.stained, args.apply)
        os.makedirs(out_dir, exist_ok=True)
        seen = {}
        for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')):
            seen.setdefault(os.path.splitext(r['stage_name'])[0], r)
        for name in sorted(seen):
            p = os.path.join(args.stained, base, name + '.png')
            mp = os.path.join(args.masks, name + '.png')
            if not (os.path.exists(p) and os.path.exists(mp)):
                continue
            rgb = cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
            m = imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127
            w = gaussian_filter(m.astype(np.float32), args.blur)
            hed = rgb2hed(rgb)
            hed[..., 0] *= ho + (hi - ho) * w
            hed[..., 1] *= eo + (ei - eo) * w
            tis = rgb.mean(2) < 0.92
            out = np.where(tis[..., None], hed, rgb2hed(rgb))
            im = (np.clip(hed2rgb(out), 0, 1) * 255).round().astype(np.uint8)
            imwrite_u(os.path.join(out_dir, name + '.png'),
                      cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f'\n-> {out_dir}  (from {base})')


if __name__ == '__main__':
    main()
