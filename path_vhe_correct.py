#!/usr/bin/env python3
"""Colour-correct the stained patches against whole-slide real H&E statistics.

Two corrections, applied in that order, because they fix different things:

  global    match this sample's virtual stain concentrations to the sample's real H&E.
            Sets the overall level -- how much haematoxylin and eosin there is.
  masked    scale haematoxylin up inside nuclei and down outside, on the blurred
            Cellpose mask. Fixes the split between compartments, which no global
            scaling can: raising haematoxylin darkens the nuclei toward the reference
            but drags the cytoplasm further violet, because the generator put
            haematoxylin where there should be none. The grid search over global (H, E)
            scalings bottomed out at a combined error of 16.8; splitting by the mask
            reached 2.7.

The reference is drawn from the slide, not from the 148 hand-picked crops. Those crops
were chosen for pathological content -- they are deliberately the densest, most
cellular, most stromal regions on the slide -- so their colour distribution is not the
slide's. Correcting to it would tune the staining to the sample the regions were picked
from and call the agreement a result.

Reference statistics are per sample. Slide-to-slide staining genuinely differs here
(240729 is pale enough to have needed its own tissue threshold at tiling time), and
pooling would correct every slide toward an average none of them has.

What this does to the evaluation
--------------------------------
After the global step, the colour metrics against real H&E are no longer an independent
test: the correction optimises the very quantity that would be measured, so agreement is
guaranteed rather than observed. What stays independent is everything the correction
cannot touch -- nucleus size and density, spatial arrangement, and whether the residual
error depends on tissue content. Those are what the comparison should rest on from here.

    python path_vhe_correct.py                 # both steps, both variants
    python path_vhe_correct.py --no_global     # masked only, the 2026-08-13 recipe
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

import stain_color_correction as scc


def correct_he_only(rgb, src, ref, white_thresh, strength, match_d=True):
    """Stat-match the HED channels; match_d=False leaves DAB alone.

    Leaving DAB alone was tried, on the theory that it is a residue neither stain
    accounts for and that matching it was what put a yellow-brown cast on the output.
    Both halves of that were wrong, and measurably so. Real H&E in this set carries
    substantial DAB: 0.0169 in nuclei and 0.0114 in cytoplasm, against a between-region
    spread of 0.0068 and 0.0045. Not matching it leaves the virtual image at 0.0029 and
    0.0017 -- more than two standard deviations low -- and drags R-B down with it, to
    -16.8 and -4.8 where real H&E sits at -2.1 and +9.5. The overall fit went from 0.228
    to 0.915.

    With the standard H&E stain matrix the DAB vector is not orthogonal to what real
    H&E actually contains; part of the pink is expressed through it. So it is matched,
    and the flag is kept only to make that a recorded decision rather than an
    assumption.
    """
    from skimage.color import hed2rgb, rgb2hed
    src_mean, src_std = src
    ref_mean, ref_std = ref
    hed = rgb2hed(rgb)
    tissue = rgb.mean(2) < white_thresh
    scale = np.where(src_std > 1e-9, ref_std / src_std, 1.0)
    shifted = (hed - src_mean) * scale + ref_mean
    shifted = hed + strength * (shifted - hed)
    if not match_d:
        shifted[..., 2] = hed[..., 2]
    out = np.where(tissue[..., None], shifted, hed)
    return np.clip(hed2rgb(out), 0.0, 1.0)

SAMPLES = ['240703', '240720', '240729', '240817', '240828_pt1', '240828_pt2']


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--stained', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--masks', default='results/path_screen/survey/_vhe/masks')
    ap.add_argument('--survey', default='results/path_screen/survey',
                    help='Holds {sample}/tiles, the whole-slide H&E the reference '
                         'statistics are drawn from.')
    ap.add_argument('--variant', action='append', default=None)
    ap.add_argument('--ref_tiles', type=int, default=300,
                    help='Whole-slide H&E tiles sampled per sample for the reference. '
                         'They are 512 px at 0.5 um/px and already tissue-filtered, so '
                         '300 is some 20 million tissue pixels -- the estimate is flat '
                         'well below this.')
    ap.add_argument('--seed', type=int, default=0,
                    help='Which tiles are sampled. Fixed so the reference, and every '
                         'corrected image, is reproducible.')
    ap.add_argument('--no_global', action='store_true',
                    help='Skip the stat-matching and do only the masked correction.')
    ap.add_argument('--no_masked', action='store_true',
                    help='Skip the masked step and do only the global stat-matching.')
    ap.add_argument('--suffix', default='_corrected',
                    help='Appended to the variant name for the output directory, so '
                         'the combinations can sit side by side and be compared rather '
                         'than one overwriting the next.')
    ap.add_argument('--strength', type=float, default=1.0)
    ap.add_argument('--no_match_d', action='store_true',
                    help='Leave the DAB channel unmatched. Measured to be clearly '
                         'worse -- real H&E here carries DAB at 0.0169 / 0.0114 and '
                         'skipping the match lands at 0.0029 / 0.0017, two standard '
                         'deviations low, taking R-B with it.')
    ap.add_argument('--white_thresh', type=float, default=0.92)
    ap.add_argument('--h_in', type=float, default=1.5)
    ap.add_argument('--h_out', type=float, default=0.5)
    ap.add_argument('--e_out', type=float, default=1.0)
    ap.add_argument('--mask_blur', type=float, default=2.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    patches = {}
    for r in rows:
        patches.setdefault(r['patch_path'], r)
    variants = args.variant or sorted(
        d for d in os.listdir(args.stained)
        if os.path.isdir(os.path.join(args.stained, d)) and '_' not in d[6:])

    by_sample = {}
    for r in patches.values():
        by_sample.setdefault(r['sample'], []).append(r)

    print(f'variants: {", ".join(variants)}')
    print(f'masked: H {args.h_in} in nuclei / {args.h_out} outside, E {args.e_out} '
          f'outside, blur {args.mask_blur}')
    print('global channels: ' + ('H and E only' if args.no_match_d else 'H, E, D'))
    print('global: ' + ('skipped' if args.no_global
                        else f'per-sample match to {args.ref_tiles} whole-slide tiles'))
    print()

    refs = {}
    if not args.no_global:
        print(f'{"sample":<12}{"tiles":>7}{"ref H":>10}{"ref E":>10}'
              f'{"src H":>10}{"src E":>10}')
        for sample in sorted(by_sample):
            tdir = Path(args.survey) / sample / 'tiles'
            tiles = sorted(p for p in tdir.iterdir() if p.suffix.lower()
                           in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            rnd = random.Random(args.seed)
            pick = rnd.sample(tiles, min(args.ref_tiles, len(tiles)))
            refs[sample] = scc.stain_stats(pick, args.white_thresh)
            print(f'{sample:<12}{len(pick):>7}{refs[sample][0][0]:>10.4f}'
                  f'{refs[sample][0][1]:>10.4f}', end='')
            src_paths = [Path(args.stained) / variants[0] /
                         (os.path.splitext(r['stage_name'])[0] + '.png')
                         for r in by_sample[sample]]
            src_paths = [p for p in src_paths if p.exists()]
            s = scc.stain_stats(src_paths, args.white_thresh)
            print(f'{s[0][0]:>10.4f}{s[0][1]:>10.4f}')
        print()

    t0 = time.time()
    n = 0
    for v in variants:
        out_dir = os.path.join(args.stained, v + args.suffix)
        os.makedirs(out_dir, exist_ok=True)
        for sample in sorted(by_sample):
            group = by_sample[sample]
            src_stat = None
            if not args.no_global:
                paths = [Path(args.stained) / v /
                         (os.path.splitext(r['stage_name'])[0] + '.png')
                         for r in group]
                src_stat = scc.stain_stats([p for p in paths if p.exists()],
                                           args.white_thresh)
            for r in group:
                name = os.path.splitext(r['stage_name'])[0]
                sp = os.path.join(args.stained, v, name + '.png')
                mp = os.path.join(args.masks, name + '.png')
                if not os.path.exists(sp):
                    continue
                rgb = cv2.cvtColor(imread_u(sp), cv2.COLOR_BGR2RGB)
                rgb = rgb.astype(np.float32) / 255.0
                if src_stat is not None:
                    rgb = correct_he_only(rgb, src_stat, refs[sample],
                                          args.white_thresh, args.strength,
                                          not args.no_match_d).astype(np.float32)
                if os.path.exists(mp) and not args.no_masked:
                    m = imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127
                    rgb = scc.correct_masked(rgb, m, args.white_thresh, args.h_in,
                                             args.h_out, args.e_out, args.mask_blur)
                elif not args.no_masked:
                    print(f'  {name}: no mask, global only')
                out = (np.asarray(rgb) * 255).round().clip(0, 255).astype(np.uint8)
                imwrite_u(os.path.join(out_dir, name + '.png'),
                          cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                n += 1
            print(f'  {v}/{sample}: {len(group)} patches  '
                  f'{time.time() - t0:.0f}s', flush=True)
        print(f'  -> {out_dir}')

    print(f'\n{n} images corrected in {time.time() - t0:.0f}s')
    print('Colour metrics against real H&E are no longer an independent test after the')
    print('global step -- it fits exactly what they measure. Judge on nucleus size and')
    print('density, spatial arrangement, and whether the residual tracks tissue content.')


if __name__ == '__main__':
    main()
