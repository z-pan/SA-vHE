#!/usr/bin/env python3
"""Run the SA-CUT baseline over the same patches, into the same layout.

SA-CUT is the closest competing method and the only one with a trained checkpoint
(SA-CUT/checkpoints/E1_sa_cut_full). Getting it into stained/<name>.png at 1024x1024
means every downstream step -- crop coordinates, Cellpose, the HoVer-Net crops, the
stain-distribution map -- reads it without modification.

Three things have to match how it was trained, and each is easy to get wrong:

  Scale. SA-CUT's patch extractor cuts at level-0 with no resampling, so it was trained
  on 512 px native TPAF at 0.621 um/px -- a 318 um field. UTOM is the opposite: it reads
  204 px native upsampled to 512, so its model sees the tissue 2.51x larger. Feeding
  SA-CUT the UTOM input would make it look bad for a reason that has nothing to do with
  the method. Windows here are 512 px native.

  Normalisation. Clip to the 1st and 99th percentile of the patch, then scale to [0, 1],
  which is what data/patch_extractor.py does. Not min-max, which the SA-CUT project
  explicitly forbids: one hot pixel would compress everything else.

  Masks. The generator takes two channels, TPAF and nuclear mask, so the mask is not
  optional and its distribution matters. Training used cpsam_20260228_gray at
  diameter 30 / flow 0.85 / cellprob 0.0, which is exactly what produced _vhe/masks --
  not masks_v2, whose retuned settings would present the generator with a mask
  distribution it never saw.

Windows overlap by half and are feathered back together, matching how path_vhe_stain.py
assembles UTOM's output, so seam handling is not a difference between the two.

    python path_vhe_sacut.py --stage prep     # write .npy windows + masks
    python path_vhe_sacut.py --stage infer    # call SA-CUT's scripts/test.py
    python path_vhe_sacut.py --stage stitch   # feather back to 1024x1024 PNGs
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys

import cv2
import numpy as np

SUR = 'results/path_screen/survey/_vhe'
SACUT = r'C:\Users\zpanp\projects\SA-CUT'
CKPT = SACUT + r'\checkpoints\E1_sa_cut_full\epoch_0399.pth'
WORK = 'results/path_screen/survey/_sacut'
WIN, STRIDE = 512, 256        # native px; half-overlap, as UTOM's tiling uses
P_LO, P_HI = 1.0, 99.0


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def imwrite_u(path, im):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], im)
    if not ok:
        raise SystemExit('cannot encode ' + path)
    buf.tofile(path)


def norm_tpaf(g):
    """Percentile clip then scale, per data/patch_extractor.py.

    Computed over the whole 1024 patch rather than per window: a window that happens to
    be all background would otherwise stretch its noise across the full range and be
    handed to the generator as though it were tissue.
    """
    lo, hi = (float(v) for v in np.percentile(g, [P_LO, P_HI]))
    if hi <= lo:
        hi = lo + 1.0
    # float32 explicitly: np.percentile returns float64 and would promote the result,
    # doubling every file on disk for no precision the generator can use.
    return np.clip((g.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def windows(h, w):
    ys = list(range(0, max(h - WIN, 0) + 1, STRIDE))
    xs = list(range(0, max(w - WIN, 0) + 1, STRIDE))
    if ys[-1] + WIN < h:
        ys.append(h - WIN)
    if xs[-1] + WIN < w:
        xs.append(w - WIN)
    return [(y, x) for y in ys for x in xs]


def feather(n, edge=64):
    """Raised-cosine ramp on each side, flat in the middle."""
    w = np.ones(n, np.float32)
    r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
    w[:edge] = r
    w[-edge:] = r[::-1]
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--masks', default=SUR + '/masks',
                    help='Training-time masks. Do not point this at masks_v2.')
    ap.add_argument('--work', default=WORK)
    ap.add_argument('--out', default=SUR + '/stained/sacut')
    ap.add_argument('--stage', choices=('prep', 'infer', 'stitch', 'all'),
                    default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    patches = {}
    for r in rows:
        patches.setdefault(r['patch_path'], r)
    items = sorted(patches.items())
    if args.limit:
        items = items[:args.limit]
    tp_dir = os.path.join(args.work, 'tpaf')
    mk_dir = os.path.join(args.work, 'masks')
    gen_dir = os.path.join(args.work, 'gen')

    if args.stage in ('prep', 'all'):
        os.makedirs(tp_dir, exist_ok=True)
        os.makedirs(mk_dir, exist_ok=True)
        n = 0
        for path, r in items:
            name = os.path.splitext(r['stage_name'])[0]
            g = cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2GRAY)
            mp = os.path.join(args.masks, name + '.png')
            if not os.path.exists(mp):
                print('  no mask, skipping ' + name)
                continue
            m = (imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            t = norm_tpaf(g)
            for y, x in windows(*g.shape[:2]):
                stem = '%s_y%07d_x%07d' % (name, y, x)
                np.save(os.path.join(tp_dir, stem + '.npy'),
                        t[y:y + WIN, x:x + WIN])
                np.save(os.path.join(mk_dir, stem + '.npy'),
                        m[y:y + WIN, x:x + WIN])
                n += 1
        print('%d windows -> %s' % (n, tp_dir))

    if args.stage in ('infer', 'all'):
        os.makedirs(gen_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(SACUT, 'scripts', 'test.py'),
               '--checkpoint', CKPT,
               '--input_dir', os.path.abspath(tp_dir),
               '--output_dir', os.path.abspath(gen_dir),
               '--mask_dir', os.path.abspath(mk_dir),
               '--patch_size', str(WIN),
               '--batch_size', str(args.batch_size)]
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=SACUT, check=True)

    if args.stage in ('stitch', 'all'):
        os.makedirs(args.out, exist_ok=True)
        wy = feather(WIN)
        wgt2 = np.outer(wy, wy)[..., None]
        done = 0
        for path, r in items:
            name = os.path.splitext(r['stage_name'])[0]
            g = cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2GRAY)
            h, w = g.shape[:2]
            acc = np.zeros((h, w, 3), np.float32)
            den = np.zeros((h, w, 1), np.float32)
            miss = 0
            for y, x in windows(h, w):
                stem = '%s_y%07d_x%07d' % (name, y, x)
                p = os.path.join(gen_dir, stem + '.png')
                if not os.path.exists(p):
                    miss += 1
                    continue
                im = imread_u(p)
                if im is None:
                    miss += 1
                    continue
                acc[y:y + WIN, x:x + WIN] += im.astype(np.float32) * wgt2
                den[y:y + WIN, x:x + WIN] += wgt2
            if miss:
                print('  %s: %d windows missing' % (name, miss))
                continue
            out = np.where(den > 0, acc / np.maximum(den, 1e-6), 255.0)
            imwrite_u(os.path.join(args.out, name + '.png'),
                      out.round().clip(0, 255).astype(np.uint8))
            done += 1
        print('%d images -> %s' % (done, args.out))


if __name__ == '__main__':
    main()
