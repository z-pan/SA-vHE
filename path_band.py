#!/usr/bin/env python3
"""Measure how far two real H&E slides sit from each other, in the same units the
virtual stain is scored in.

Why this exists. The virtual stain reaches CIELAB dE2000 0.97 against the real H&E,
averaged over three compartments. On its own that number supports no claim. The
textbook reading -- dE below 1 is imperceptible -- is calibrated on uniform colour
patches viewed side by side under standard illumination, not on tissue, and it says
nothing about whether 0.97 is easy or hard to reach in this material. Two real slides
from this same laboratory also differ; until that difference is measured, there is no
scale to put 0.97 on. The reference haematoxylin concentration across five samples here
already ranges 0.0055 to 0.0223, a factor of four inside one lab.

So: run the identical estimator over pairs of real slides, and report where the virtual
value falls in that distribution.

The estimator has to match exactly, or the comparison is meaningless. From
path_vhe_tune3.py:

    nucleus mask    base cpsam, diameter 20, flow_threshold 0.4, cellprob 0.0
                    Recovered by matching the stored masks_he: IoU 1.000 on four
                    regions, exactly. The setting was never written down; the masks
                    themselves are the record.
    compartments    nucleus / dense stroma / pale stroma, split on the haematoxylin
                    percentile of the image being measured (pale_pct 40)
    aggregation     per-tile compartment mean Lab, averaged over tiles, then ONE
                    dE2000 between the two averaged Labs. Not the mean of per-tile
                    dE. This matters: the pooled form averages away all per-tile
                    variation, which is why it is so easy to make small.

Two distances are reported, and the second is what makes the first readable:

    between-slide   slide A pooled vs slide B pooled. Contains staining batch, section,
                    patient, and tissue composition.
    within-slide    a slide's tiles split into two random halves, half vs half.
                    Contains tissue composition and local staining variation but no
                    batch or section effect. This is the floor -- no method can be
                    expected to beat the distance a slide has to itself.

The gap between them is the slide effect. If the virtual value sits below the
within-slide floor, the correction is fitting the target rather than reproducing it.

Caveat that must survive into the write-up: the virtual-vs-real pairs are the same
tissue, these pairs are not. Measuring each compartment separately absorbs most of the
composition difference -- nucleus colour is compared with nucleus colour -- but not all
of it, since compartments are not internally homogeneous. The within-slide figure
bounds how much of the between-slide distance that residual can explain.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2hed, rgb2lab

COMP = ['nuc', 'dense', 'pale']


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread returns None on a path with non-ASCII characters on Windows, without
    raising. Several slide directories here are named in Chinese."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def partition(rgb, mask, blur, pale_pct, soft):
    """Identical to path_vhe_tune3.partition -- kept in sync by hand, deliberately not
    imported, so that editing the tuner cannot silently move the band."""
    tissue = rgb.mean(2) < 0.92
    if tissue.sum() < 200:
        return None
    h = rgb2hed(rgb)[..., 0]
    thr = float(np.percentile(h[tissue], pale_pct))
    w_n = np.clip(gaussian_filter(mask.astype(np.float32), blur), 0, 1)
    s = 1.0 / (1.0 + np.exp((h - thr) / max(soft, 1e-6)))
    hard = np.where(mask > 0.5, 0, np.where(h < thr, 2, 1))
    return tissue, hard


def dE(a, b):
    return float(deltaE_ciede2000(np.asarray(a).reshape(1, 1, 3),
                                  np.asarray(b).reshape(1, 1, 3))[0, 0])


def pooled(records, idx=None):
    """Average the per-tile compartment means, then that is the slide's colour.

    A tile contributes to a compartment only if it had enough of it; a tile that is all
    stroma should not vote on nucleus colour with 12 pixels.
    """
    sel = records if idx is None else [records[i] for i in idx]
    out = {}
    for i, c in enumerate(COMP):
        v = [r[c] for r in sel if r[c] is not None]
        out[c] = np.mean(v, axis=0) if v else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey', default='results/path_screen/survey')
    ap.add_argument('--slides', default='',
                    help='Comma-separated slide names; default is every directory '
                         'under --survey that has a tiles/ subdirectory. Names ending '
                         '_pt1/_pt2 are merged into one slide -- they are two files '
                         'from one section, not two sections.')
    ap.add_argument('--n_tiles', type=int, default=80,
                    help='Tiles sampled per slide. The estimator pools them, so this '
                         'trades run time against how tightly the slide colour is '
                         'estimated, not against how many pairs there are.')
    ap.add_argument('--min_px', type=int, default=200,
                    help='Pixels a compartment needs in a tile before that tile votes '
                         'on its colour. Matches the tuner (>200).')
    ap.add_argument('--blur', type=float, default=2.0)
    ap.add_argument('--pale_pct', type=float, default=40.0)
    ap.add_argument('--soft', type=float, default=0.004)
    ap.add_argument('--diameter', type=float, default=20.0)
    ap.add_argument('--flow_threshold', type=float, default=0.4)
    ap.add_argument('--cellprob_threshold', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/path_screen/survey/_band')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.slides:
        names = [s.strip() for s in args.slides.split(',') if s.strip()]
    else:
        names = sorted(d for d in os.listdir(args.survey)
                       if os.path.isdir(os.path.join(args.survey, d, 'tiles')))

    # _pt1/_pt2 are one physical section scanned in two files. Treating them as two
    # slides would put a pair into the band whose distance contains no section effect
    # at all, which is exactly the quantity the band is supposed to measure.
    groups = {}
    for n in names:
        key = n.rsplit('_pt', 1)[0] if '_pt' in n else n
        groups.setdefault(key, []).append(n)
    print(f'{len(groups)} slides from {len(names)} tile directories')
    for k, v in sorted(groups.items()):
        if len(v) > 1:
            print(f'  {k} <- {", ".join(v)}')

    from cellpose import models
    import torch
    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    print(f'cpsam base, gpu={gpu}, diameter={args.diameter}, '
          f'flow={args.flow_threshold}, cellprob={args.cellprob_threshold}')

    per_tile = []
    slide_recs = {}
    for slide, dirs in sorted(groups.items()):
        pool = []
        for d in dirs:
            tdir = os.path.join(args.survey, d, 'tiles')
            pool += [os.path.join(tdir, f) for f in sorted(os.listdir(tdir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        take = pool if len(pool) <= args.n_tiles else \
            [pool[i] for i in rng.choice(len(pool), args.n_tiles, replace=False)]
        recs = []
        for k, p in enumerate(take):
            im = imread_u(p)
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
            lab_msk, _, _ = model.eval(
                (rgb * 255).astype(np.uint8), diameter=args.diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold)
            pr = partition(rgb, lab_msk > 0, args.blur, args.pale_pct, args.soft)
            if pr is None:
                continue
            tissue, hard = pr
            lab = rgb2lab(rgb)
            rec = {'slide': slide, 'tile': os.path.basename(p)}
            for i, c in enumerate(COMP):
                sel = tissue & (hard == i)
                rec[c] = lab[sel].mean(0) if sel.sum() > args.min_px else None
                rec[c + '_px'] = int(sel.sum())
            recs.append(rec)
            per_tile.append(rec)
            if (k + 1) % 20 == 0:
                print(f'  {slide} {k + 1}/{len(take)}', flush=True)
        slide_recs[slide] = recs
        pl = pooled(recs)
        print(f'{slide:<12} {len(recs):>3} tiles  ' + '  '.join(
            f'{c}=' + ('--' if pl[c] is None else
                       '(' + ','.join(f'{v:.1f}' for v in pl[c]) + ')')
            for c in COMP), flush=True)

    with open(os.path.join(args.out, 'tile_lab.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['slide', 'tile'] + [f'{c}_{k}' for c in COMP for k in 'Lab']
                   + [f'{c}_px' for c in COMP])
        for r in per_tile:
            row = [r['slide'], r['tile']]
            for c in COMP:
                row += ['' if r[c] is None else f'{v:.4f}' for v in
                        (r[c] if r[c] is not None else [None] * 3)]
            row += [r[c + '_px'] for c in COMP]
            w.writerow(row)

    slides = sorted(slide_recs)
    pooled_by = {s: pooled(slide_recs[s]) for s in slides}

    between = []
    for i in range(len(slides)):
        for j in range(i + 1, len(slides)):
            a, b = pooled_by[slides[i]], pooled_by[slides[j]]
            d = {c: (dE(a[c], b[c]) if a[c] is not None and b[c] is not None
                     else np.nan) for c in COMP}
            d['mean'] = float(np.nanmean([d[c] for c in COMP]))
            between.append({'a': slides[i], 'b': slides[j], **d})

    # Split-half, repeated: the floor. Not a bootstrap -- the halves are disjoint, so
    # this is the distance a slide has to itself under the same estimator.
    within = []
    for s in slides:
        recs = slide_recs[s]
        if len(recs) < 8:
            continue
        for rep in range(20):
            perm = rng.permutation(len(recs))
            h = len(recs) // 2
            a, b = pooled(recs, perm[:h]), pooled(recs, perm[h:2 * h])
            d = {c: (dE(a[c], b[c]) if a[c] is not None and b[c] is not None
                     else np.nan) for c in COMP}
            d['mean'] = float(np.nanmean([d[c] for c in COMP]))
            within.append({'slide': s, 'rep': rep, **d})

    for name, rowset, keys in (('between_slide.csv', between, ['a', 'b']),
                               ('within_slide.csv', within, ['slide', 'rep'])):
        with open(os.path.join(args.out, name), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(keys + COMP + ['mean'])
            for r in rowset:
                w.writerow([r[k] for k in keys]
                           + [f'{r[c]:.4f}' for c in COMP] + [f'{r["mean"]:.4f}'])

    def quant(rowset, c):
        v = np.array([r[c] for r in rowset], float)
        v = v[~np.isnan(v)]
        return np.percentile(v, [5, 25, 50, 75, 95]) if v.size else np.full(5, np.nan)

    print(f'\n{len(slides)} slides -> {len(between)} between-slide pairs, '
          f'{len(within)} split-half draws\n')
    print(f'{"":<8}{"p5":>8}{"p25":>8}{"p50":>8}{"p75":>8}{"p95":>8}')
    for label, rowset in (('BETWEEN', between), ('WITHIN', within)):
        print(label)
        for c in COMP + ['mean']:
            print(f'  {c:<6}' + ''.join(f'{v:>8.2f}' for v in quant(rowset, c)))
    print(f'\nwrote {args.out}/tile_lab.csv, between_slide.csv, within_slide.csv')


if __name__ == '__main__':
    main()
