#!/usr/bin/env python3
"""Blend two stained variants per compartment, taking each where it is measurably better.

The global stat-match fixes nuclei and dense stroma and damages the pale stroma. Fitted
three-compartment corrections on each base give, in CIELAB dE against real H&E:

                       nuc   dense    pale
    gray_globalonly    0.8     1.5     4.5
    gray               4.8     5.1     2.7

The advantages are complementary and, unlike the parameter settings that were being
compared before, they are separable in space: one is better in a region the other is
worse in, and the region is identifiable. So rather than choosing, take each where it
wins -- pale stroma from the unmatched image, everything else from the matched one.

This is only legitimate because the two are pixel-aligned: same patch, same geometry,
same generator output, differing by a colour transform. Blending images that were not
aligned would be inventing tissue.

The weights are the same soft partition the correction uses, so the seam between the
two sources is a gradient rather than an edge.

    python path_vhe_blend.py --a gray_globalonly --b gray --out gray_blend
"""

from __future__ import annotations

import argparse
import csv
import io
import os

import cv2
import numpy as np

from path_vhe_tune3 import imread_u, imwrite_u, partition


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--stained', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--masks', default='results/path_screen/survey/_vhe/masks')
    ap.add_argument('--a', default='gray_globalonly', help='Used for nuclei and dense.')
    ap.add_argument('--b', default='gray', help='Used for the pale compartment.')
    ap.add_argument('--out', default='gray_blend')
    ap.add_argument('--blur', type=float, default=2.0)
    ap.add_argument('--pale_pct', type=float, default=40.0)
    ap.add_argument('--soft', type=float, default=0.004)
    args = ap.parse_args()

    seen = {}
    for r in csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')):
        seen.setdefault(os.path.splitext(r['stage_name'])[0], r)
    out_dir = os.path.join(args.stained, args.out)
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for name in sorted(seen):
        pa = os.path.join(args.stained, args.a, name + '.png')
        pb = os.path.join(args.stained, args.b, name + '.png')
        mp = os.path.join(args.masks, name + '.png')
        if not all(os.path.exists(p) for p in (pa, pb, mp)):
            continue
        A = cv2.cvtColor(imread_u(pa), cv2.COLOR_BGR2RGB) / 255.0
        B = cv2.cvtColor(imread_u(pb), cv2.COLOR_BGR2RGB) / 255.0
        if A.shape != B.shape:
            print(f'  {name}: shapes differ, skipped')
            continue
        m = imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127
        # Partition from A, so the compartments are the ones the fit was measured on.
        part = partition(A, m, args.blur, args.pale_pct, args.soft)
        if part is None:
            continue
        tissue, w_n, w_d, w_p = part[0], part[1], part[2], part[3]
        w = w_p[..., None]                      # 1 in pale, 0 in nuclei and dense
        mix = np.where(tissue[..., None], A * (1 - w) + B * w, A)
        im = (np.clip(mix, 0, 1) * 255).round().astype(np.uint8)
        imwrite_u(os.path.join(out_dir, name + '.png'),
                  cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        n += 1
    print(f'{n} blended: {args.a} everywhere, {args.b} in the pale compartment')
    print(f'  -> {out_dir}')


if __name__ == '__main__':
    main()
