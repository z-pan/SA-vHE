#!/usr/bin/env python3
"""Correct nucleus and stroma haematoxylin separately, and sweep the ratio.

Why this exists
---------------
path_vhe_tune3.py fits six multipliers -- haematoxylin and eosin in each of nucleus,
dense stroma and pale stroma -- against CIELAB dE2000 per compartment. On nuc_flat the
fit lands at

    h_nuc 0.53   h_dense 0.46   h_pale 0.51

which is a uniform haematoxylin reduction wearing a three-compartment costume. The
nucleus-to-stroma ratio comes out at 1.15, essentially unchanged from the generator's
own output, because a uniform reduction already satisfies the colour objective.

That objective is not the one that matters downstream. Measured on 148 regions with
HoVer-Net:

                        count vs TPAF   nuclei as % of field   epithelial share
    real H&E                     -1%                    3.5%              0.632
    nuc_flat  (no correction)  +230%                   13.8%              0.637
    nuc_flat_final  (dE fit)     -8%                    2.7%              0.379

The uncorrected image has four times as much of the field covered in nucleus-dark
material at a normal nucleus size (32.8 um2 against 28.0) -- so the generator is
rendering nuclei where there are none, rather than fragmenting real ones. Lowering
haematoxylin everywhere pushes the spurious ones under the detection threshold and the
real ones with them, which is why the epithelial share collapses: PanNuke reads the
survivors as connective or dead. gray, which has no nucleus enhancement at all, behaves
the same way, so this is the generator's doing and not the enhancement's.

What is needed is the one thing the dE fit had no reason to do: raise the nucleus
haematoxylin relative to the stroma. Two knobs on top of the fitted parameters --
`--h_nuc_mul` scales h_nuc, `--h_stroma_mul` scales h_dense and h_pale together -- and
a sweep over the pair.

    python path_vhe_nucsep.py --base nuc_flat --sweep
    python path_vhe_nucsep.py --base nuc_flat --h_nuc_mul 1.4 --h_stroma_mul 0.4 \
                              --out nuc_flat_sep

The compartment partition, the HED round trip and the tissue mask are imported from
path_vhe_tune3 rather than reimplemented, so a sweep result and a dE fit are the same
transform at different parameters and remain comparable.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time

import cv2
import numpy as np
from skimage.color import hed2rgb, rgb2hed

from path_vhe_tune3 import imread_u, imwrite_u, partition

SUR = 'results/path_screen/survey/_vhe'
# The dE2000 optimum for nuc_flat, from path_vhe_tune3.py --base nuc_flat --regions 60.
# Kept as the starting point so the sweep is expressed as a departure from the best
# purely-colorimetric correction rather than from nothing.
FITTED = dict(h_nuc=0.53, h_dense=0.46, h_pale=0.51,
              e_nuc=0.40, e_dense=0.40, e_pale=1.00)
# Nucleus multiplier x stroma multiplier. 1.0/1.0 reproduces the dE fit. The nucleus
# side goes up because nuc_flat_final's nuclei came out 29% lighter than real H&E; the
# stroma side goes down because that is where the spurious nuclei are.
SWEEP = [(1.0, 0.5), (1.4, 0.5), (1.4, 0.3), (1.8, 0.3)]


def apply_one(rgb, mask, p, blur, pale_pct, soft):
    part = partition(rgb, mask, blur, pale_pct, soft)
    if part is None:
        return None
    tissue, wn, wd, wp, _ = part
    hed = rgb2hed(rgb)
    out = hed.copy()
    out[..., 0] *= wn * p['h_nuc'] + wd * p['h_dense'] + wp * p['h_pale']
    out[..., 1] *= wn * p['e_nuc'] + wd * p['e_dense'] + wp * p['e_pale']
    # Leave the background alone: outside tissue there is no stain to rescale, and
    # multiplying near-zero concentrations only amplifies sensor noise into colour.
    out = np.where(tissue[..., None], out, hed)
    return (np.clip(hed2rgb(out), 0, 1) * 255).round().astype(np.uint8)


def run(base, params, out_name, names, args):
    out_dir = os.path.join(args.stained, out_name)
    os.makedirs(out_dir, exist_ok=True)
    t0, n, skipped = time.time(), 0, 0
    for name in names:
        sp = os.path.join(args.stained, base, name + '.png')
        mp = os.path.join(args.masks, name + '.png')
        if not (os.path.exists(sp) and os.path.exists(mp)):
            skipped += 1
            continue
        dst = os.path.join(out_dir, name + '.png')
        if os.path.exists(dst) and not args.redo:
            n += 1
            continue
        rgb = cv2.cvtColor(imread_u(sp), cv2.COLOR_BGR2RGB) / 255.0
        m = imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127
        im = apply_one(rgb, m, params, args.blur, args.pale_pct, args.soft)
        if im is None:
            skipped += 1
            continue
        imwrite_u(dst, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        n += 1
    print('  -> %-24s %3d images, %2d skipped, %.0fs'
          % (out_name, n, skipped, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--stained', default=SUR + '/stained')
    ap.add_argument('--masks', default=SUR + '/masks')
    ap.add_argument('--base', default='nuc_flat')
    ap.add_argument('--h_nuc_mul', type=float, default=1.0)
    ap.add_argument('--h_stroma_mul', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sweep', action='store_true',
                    help='Run the pairs in SWEEP instead of a single setting.')
    ap.add_argument('--redo', action='store_true')
    # These three define the partition and must match path_vhe_tune3's defaults, or
    # the fitted parameters are being applied to different compartments than they
    # were fitted on.
    ap.add_argument('--blur', type=float, default=2.0)
    ap.add_argument('--pale_pct', type=float, default=40.0)
    ap.add_argument('--soft', type=float, default=0.004)
    args = ap.parse_args()

    seen = {}
    for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')):
        seen.setdefault(os.path.splitext(r['stage_name'])[0], r)
    names = sorted(seen)
    print('base %s, %d patches' % (args.base, len(names)))
    print('fitted: ' + '  '.join('%s %.2f' % kv for kv in FITTED.items()))

    combos = SWEEP if args.sweep else [(args.h_nuc_mul, args.h_stroma_mul)]
    for hn_mul, hs_mul in combos:
        p = dict(FITTED)
        p['h_nuc'] *= hn_mul
        p['h_dense'] *= hs_mul
        p['h_pale'] *= hs_mul
        name = args.out if (args.out and not args.sweep) else (
            '%s_sep%02d%02d' % (args.base, round(hn_mul * 10), round(hs_mul * 10)))
        print('%s: h_nuc %.2f  h_dense %.2f  h_pale %.2f   (ratio nuc/dense %.2f)'
              % (name, p['h_nuc'], p['h_dense'], p['h_pale'],
                 p['h_nuc'] / p['h_dense']))
        run(args.base, p, name, names, args)


if __name__ == '__main__':
    main()
