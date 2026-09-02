#!/usr/bin/env python3
"""Three-compartment stain correction, fitted on perceptual colour difference.

Why this exists
---------------
The two-compartment fit reached CIELAB dE 0.9 in nuclei and 1.2 in cytoplasm -- below
the threshold of perception -- while the pale stroma went to 8.1, worse than doing
nothing at all (3.6). The mechanism is visible in b*: real pale stroma sits at -14,
the corrected image at -2, twelve units toward yellow, with a* down from 28 to 19.
Less red and more yellow is the tan cast that shows in the images.

Two separate causes, and both are addressed here.

The objective could not see it. "Cytoplasm" was every non-nuclear tissue pixel, and the
dense eosinophilic areas outnumber the pale stroma, so the pale error averaged away and
the search was free to spend it. Pale stroma is now its own compartment with its own
term.

The transform could not fix it. With a binary partition, one h_out governs both dense
and pale stroma, and they want opposite things: dense needs haematoxylin suppressed,
pale does not. No setting satisfies both, so the search had to sacrifice one. The
partition is now three-way -- nucleus, dense stroma, pale stroma -- and haematoxylin
and eosin get their own multiplier in each, six parameters against six compartment
constraints.

The split between dense and pale is made on the haematoxylin concentration of the
*input*, once, and held fixed. Recomputing it from the corrected image at every setting
would make the compartments move with the parameters and the objective meaningless.

The target is CIELAB dE2000 per compartment rather than R-B, which was invented for
this project. dE is calibrated to perception: 1 is the threshold of a just-noticeable
difference, and the previous objective scored a 12-unit b* error as a success.

    python path_vhe_tune3.py
    python path_vhe_tune3.py --apply gray_tuned3
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from skimage.color import deltaE_ciede2000, hed2rgb, rgb2hed, rgb2lab

COMP = ['nuc', 'dense', 'pale']
PNAMES = ['h_nuc', 'h_dense', 'h_pale', 'e_nuc', 'e_dense', 'e_pale']
# Wide. The two-compartment fit hit its h_in ceiling twice and its h_out floor
# once, and an optimum on a bound is not an optimum, it is the largest or
# smallest value that was allowed. h_pale went to 0.20 on the first run here too.
BOUNDS = [(0.02, 8.0)] * 3 + [(0.02, 4.0)] * 3


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


def partition(rgb, mask, blur, pale_pct, soft):
    """Soft weights for the three compartments, plus a hard label for scoring.

    The hard labels answer "what is this pixel" for the statistics; the soft weights
    say how much of each multiplier a pixel receives, so the transform cross-fades
    across boundaries instead of leaving edges at them.
    """
    tissue = rgb.mean(2) < 0.92
    if tissue.sum() < 200:
        return None
    h = rgb2hed(rgb)[..., 0]
    thr = float(np.percentile(h[tissue], pale_pct))
    w_n = np.clip(gaussian_filter(mask.astype(np.float32), blur), 0, 1)
    # sigmoid on haematoxylin: 1 where pale, 0 where dense
    s = 1.0 / (1.0 + np.exp((h - thr) / max(soft, 1e-6)))
    w_p = (1 - w_n) * s
    w_d = (1 - w_n) * (1 - s)
    hard = np.where(mask > 0.5, 0, np.where(h < thr, 2, 1))
    return tissue, w_n, w_d, w_p, hard


def sample(rgb, mask, blur, pale_pct, soft, n_px, rng):
    p = partition(rgb, mask, blur, pale_pct, soft)
    if p is None:
        return None
    tissue, w_n, w_d, w_p, hard = p
    idx = np.flatnonzero(tissue)
    if idx.size > n_px:
        idx = rng.choice(idx, n_px, replace=False)
    hed = rgb2hed(rgb.reshape(-1, 3)[idx].reshape(-1, 1, 3)).reshape(-1, 3)
    return (hed, w_n.reshape(-1)[idx], w_d.reshape(-1)[idx],
            w_p.reshape(-1)[idx], hard.reshape(-1)[idx])


def apply_params(hed, wn, wd, wp, p):
    hn, hd, hp, en, ed, ep = p
    out = hed.copy()
    out[:, 0] *= wn * hn + wd * hd + wp * hp
    out[:, 1] *= wn * en + wd * ed + wp * ep
    return np.clip(hed2rgb(out.reshape(-1, 1, 3)).reshape(-1, 3), 0.0, 1.0)


def lab_means(regions, p):
    acc = {i: [] for i in range(3)}
    for hed, wn, wd, wp, hard in regions:
        rgb = apply_params(hed, wn, wd, wp, p)
        lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        for i in range(3):
            sel = hard == i
            if sel.sum() > 50:
                acc[i].append(lab[sel].mean(0))
    return {COMP[i]: (np.mean(acc[i], axis=0) if acc[i] else np.full(3, np.nan))
            for i in range(3)}


def dE(a, b):
    return float(deltaE_ciede2000(np.asarray(a).reshape(1, 1, 3),
                                  np.asarray(b).reshape(1, 1, 3))[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--stained', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--masks', default='results/path_screen/survey/_vhe/masks')
    ap.add_argument('--masks_he', default='results/path_screen/survey/_vhe/masks_he')
    ap.add_argument('--base', action='append', default=None)
    ap.add_argument('--n_px', type=int, default=4000)
    ap.add_argument('--regions', type=int, default=0)
    ap.add_argument('--blur', type=float, default=2.0)
    ap.add_argument('--pale_pct', type=float, default=40.0,
                    help='Haematoxylin percentile below which tissue counts as pale '
                         'stroma. Applied the same way to the real and the virtual '
                         'image, so the compartments mean the same thing on both.')
    ap.add_argument('--soft', type=float, default=0.004,
                    help='Width of the sigmoid across that threshold, in haematoxylin '
                         'units.')
    ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--maxiter', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--apply', default=None)
    args = ap.parse_args()

    bases = args.base or ['gray_globalonly', 'gray']
    rows = [r for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig'))
            if r['he_path']]
    if args.regions:
        rows = rows[:args.regions]
    rng = np.random.default_rng(args.seed)

    print(f'target from {len(rows)} real H&E regions', flush=True)
    real_acc = {i: [] for i in range(3)}
    for r in rows:
        rgb = cv2.cvtColor(imread_u(r['he_path']), cv2.COLOR_BGR2RGB) / 255.0
        m = imread_u(os.path.join(args.masks_he, r['id'] + '.png'),
                     cv2.IMREAD_GRAYSCALE) > 127
        p = partition(rgb, m, args.blur, args.pale_pct, args.soft)
        if p is None:
            continue
        lab = rgb2lab(rgb)
        for i in range(3):
            sel = p[0] & (p[4] == i)
            if sel.sum() > 200:
                real_acc[i].append(lab[sel].mean(0))
    target = {COMP[i]: np.mean(real_acc[i], axis=0) for i in range(3)}
    print(f'{"":<8}{"L*":>8}{"a*":>8}{"b*":>8}   n regions')
    for i, c in enumerate(COMP):
        print(f'{c:<8}' + ''.join(f'{v:>8.1f}' for v in target[c])
              + f'   {len(real_acc[i])}')
    print()

    best_all = None
    for base in bases:
        print(f'sampling {base}', flush=True)
        regions = []
        for r in rows:
            name = os.path.splitext(r['stage_name'])[0]
            sp = os.path.join(args.stained, base, name + '.png')
            if not os.path.exists(sp):
                continue
            x, y = int(r['crop_x']), int(r['crop_y'])
            w, h = int(r['crop_w']), int(r['crop_h'])
            rgb = cv2.cvtColor(imread_u(sp)[y:y + h, x:x + w],
                               cv2.COLOR_BGR2RGB) / 255.0
            m = imread_u(os.path.join(args.masks, name + '.png'),
                         cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w] > 127
            s = sample(rgb, m, args.blur, args.pale_pct, args.soft, args.n_px, rng)
            if s:
                regions.append(s)

        def obj(p):
            p = np.clip(p, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
            got = lab_means(regions, p)
            return float(np.mean([dE(target[c], got[c]) for c in COMP]))

        e0 = obj(np.ones(6))
        print(f'  {len(regions)} regions; uncorrected mean dE {e0:.2f}', flush=True)
        t0 = time.time()
        best = None
        starts = [np.ones(6)] + [rng.uniform([b[0] for b in BOUNDS],
                                             [min(b[1], 3.0) for b in BOUNDS])
                                 for _ in range(args.restarts - 1)]
        for k, x0 in enumerate(starts, 1):
            res = minimize(obj, x0, method='Nelder-Mead',
                           options=dict(maxiter=args.maxiter, xatol=1e-3, fatol=1e-3))
            p = np.clip(res.x, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
            v = obj(p)
            print(f'    start {k}: dE {v:.2f}  ' +
                  ' '.join(f'{n}={x:.2f}' for n, x in zip(PNAMES, p)), flush=True)
            if best is None or v < best[0]:
                best = (v, p)
        got = lab_means(regions, best[1])
        per = {c: dE(target[c], got[c]) for c in COMP}
        print(f'  best {base}: mean dE {best[0]:.2f}  ('
              + ', '.join(f'{c} {per[c]:.1f}' for c in COMP) + f')  {time.time()-t0:.0f}s')
        for c in COMP:
            print(f'    {c:<6} real ' + ''.join(f'{v:>7.1f}' for v in target[c])
                  + '   tuned ' + ''.join(f'{v:>7.1f}' for v in got[c]))
        if best_all is None or best[0] < best_all[0]:
            best_all = (best[0], base, best[1], per, got)
        print()

    e, base, p, per, got = best_all
    print('=' * 78)
    print(f'best: {base}  ' + '  '.join(f'{n} {x:.2f}' for n, x in zip(PNAMES, p)))
    print(f'  mean dE {e:.2f}   ' + '   '.join(f'{c} {per[c]:.2f}' for c in COMP))
    print('  dE below 1 is imperceptible, below 2 barely perceptible.')

    if args.apply:
        out_dir = os.path.join(args.stained, args.apply)
        os.makedirs(out_dir, exist_ok=True)
        seen = {}
        for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')):
            seen.setdefault(os.path.splitext(r['stage_name'])[0], r)
        hn, hd, hp, en, ed, ep = p
        for name in sorted(seen):
            sp = os.path.join(args.stained, base, name + '.png')
            mp = os.path.join(args.masks, name + '.png')
            if not (os.path.exists(sp) and os.path.exists(mp)):
                continue
            rgb = cv2.cvtColor(imread_u(sp), cv2.COLOR_BGR2RGB) / 255.0
            m = imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127
            part = partition(rgb, m, args.blur, args.pale_pct, args.soft)
            if part is None:
                continue
            tissue, wn, wd, wp, _ = part
            hed = rgb2hed(rgb)
            out = hed.copy()
            out[..., 0] *= wn * hn + wd * hd + wp * hp
            out[..., 1] *= wn * en + wd * ed + wp * ep
            out = np.where(tissue[..., None], out, hed)
            im = (np.clip(hed2rgb(out), 0, 1) * 255).round().astype(np.uint8)
            imwrite_u(os.path.join(out_dir, name + '.png'),
                      cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f'\n-> {out_dir}  (from {base})')


if __name__ == '__main__':
    main()
