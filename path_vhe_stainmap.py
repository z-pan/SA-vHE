#!/usr/bin/env python3
"""How well does the virtual stain reproduce the purple clumps a pathologist reads?

Why this and not the nucleus metrics
------------------------------------
Nucleus-level agreement can only be asked where TPAF carries nuclear information.
Nuclei do not autofluoresce; they are visible as dark holes against NADH, FAD and
collagen, so where the surrounding signal is weak they are not there to be found, and
H&E processing moves tissue morphology besides. Those regions are not the ones to
report nucleus counts on, and forcing the mask to match real H&E's nuclear coverage
there would mean inventing nuclei.

They still have to look right, because what separates lesional from non-lesional tissue
at low power is a pattern of distinctly purple clumps, and that is what a pathologist
reads first, what a MIL model sees, and what a tumour/stroma segmenter keys on. None of
it depends on resolving a single nucleus.

What is measured
----------------
Haematoxylin concentration by colour deconvolution, pooled to a coarse grid at
COARSE_UM per pixel -- roughly what a 4x objective resolves, and the scale a clump
lives at. Then, on that grid:

    purple_frac   share of tissue above the level real H&E calls purple
    clump_med     median connected-component size of that region, in coarse pixels
    clump_max     largest one, which is what makes a field read as lesional
    h_p50, h_p90  the concentration distribution itself

Distribution and morphology, never a pixel-to-pixel comparison: the virtual and real
images are the same field but not registered to each other, so any per-pixel agreement
would be measuring registration.

The purple level is calibrated once on real H&E -- the concentration at the percentile
that leaves REF_FRAC of its tissue above -- and the same absolute level is then applied
to every version, so a version cannot score well by shifting its own threshold.

    python path_vhe_stainmap.py
    python path_vhe_stainmap.py --versions nuc_flat --versions nuc_flat_final
"""

from __future__ import annotations

import argparse
import csv
import io
import os

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2hed

SUR = 'results/path_screen/survey/_vhe'
TPAF_MPP = 0.621
COARSE_UM = 16.0     # a low-power resolving distance; a clump is several of these
WHITE = 0.92         # same white threshold the staining and correction pipelines use
REF_FRAC = 0.25      # share of real H&E tissue defined as purple; sets the level


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def coarse_h(rgb, mpp):
    """Haematoxylin per coarse cell, plus which cells are tissue.

    Averaging concentration over the cell, not colour: haematoxylin is what stains,
    and averaging RGB first would let a pale cell and a dark one cancel into a mid
    grey that has no interpretation as a stain amount.
    """
    f = mpp / COARSE_UM
    hed = rgb2hed(rgb)
    h = hed[..., 0].astype(np.float32)
    tis = (rgb.mean(2) < WHITE).astype(np.float32)
    hw = (max(1, int(round(rgb.shape[0] * f))), max(1, int(round(rgb.shape[1] * f))))
    hs = cv2.resize(h * tis, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    ts = cv2.resize(tis, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    ok = ts > 0.5
    out = np.zeros_like(hs)
    out[ok] = hs[ok] / ts[ok]
    return out, ok


def stats(h, ok, thr):
    if ok.sum() < 16:
        return None
    v = h[ok]
    pur = (h > thr) & ok
    lab, n = ndimage.label(pur)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    return dict(purple_frac=float(pur.sum() / ok.sum()),
                clump_med=float(np.median(sizes)) if n else 0.0,
                clump_max=float(sizes.max()) if n else 0.0,
                n_clump=float(n) / max(ok.sum(), 1) * 100,
                h_p50=float(np.percentile(v, 50)),
                h_p90=float(np.percentile(v, 90)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--stained', default=SUR + '/stained')
    ap.add_argument('--versions', action='append', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    if args.limit:
        rows = rows[:args.limit]
    versions = args.versions or sorted(
        d for d in os.listdir(args.stained)
        if os.path.isdir(os.path.join(args.stained, d)))

    # Calibrate the purple level on real H&E, pooling every region's coarse cells.
    pool = []
    real_cache = {}
    for r in rows:
        im = imread_u(r['he_path'])
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
        mpp = float(r['crop_um']) / im.shape[1]
        h, ok = coarse_h(rgb, mpp)
        real_cache[r['id']] = (h, ok)
        pool.append(h[ok])
    allh = np.concatenate(pool)
    THR = float(np.percentile(allh, 100 * (1 - REF_FRAC)))
    print('%d real H&E regions; purple level H > %.4f (top %.0f%% of its tissue)'
          % (len(real_cache), THR, 100 * REF_FRAC))
    print('coarse grid %.0f um/px\n' % COARSE_UM)

    KEYS = ['purple_frac', 'clump_med', 'clump_max', 'n_clump', 'h_p50', 'h_p90']

    def summarise(per):
        return {k: float(np.median([p[k] for p in per])) for k in KEYS}

    real = summarise([stats(h, ok, THR) for h, ok in real_cache.values()
                      if stats(h, ok, THR)])

    print('%-22s' % 'version' + ''.join('%12s' % k for k in KEYS) + '%10s' % 'dist')
    print('%-22s' % 'real_HE' + ''.join('%12.3f' % real[k] for k in KEYS))
    out = []
    for v in versions:
        per = []
        for r in rows:
            p = os.path.join(args.stained, v,
                             os.path.splitext(r['stage_name'])[0] + '.png')
            if not os.path.exists(p):
                continue
            im = imread_u(p)
            if im is None:
                continue
            x, y, w, hh = (int(r[k]) for k in ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
            rgb = cv2.cvtColor(im[y:y + hh, x:x + w], cv2.COLOR_BGR2RGB) / 255.0
            h, ok = coarse_h(rgb, TPAF_MPP)
            s = stats(h, ok, THR)
            if s:
                per.append(s)
        if not per:
            continue
        g = summarise(per)
        # One number to rank on: mean relative departure over the four shape measures.
        # h_p50 and h_p90 are left out of it because the colour corrections were fitted
        # to match concentration, so they would be scoring their own objective.
        d = float(np.mean([abs(g[k] / real[k] - 1) if real[k] else np.nan
                           for k in ('purple_frac', 'clump_med', 'clump_max',
                                     'n_clump')]))
        out.append((d, v, g))
    for d, v, g in sorted(out):
        print('%-22s' % v + ''.join('%12.3f' % g[k] for k in KEYS) + '%10.2f' % d)
    print('\n  dist = mean |version/real - 1| over purple_frac, clump_med, clump_max,')
    print('  n_clump. Lower is closer to real H&E. Concentration columns are shown but')
    print('  not scored: the colour corrections were fitted to match them.')


if __name__ == '__main__':
    main()
