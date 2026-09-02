#!/usr/bin/env python3
"""Nuclei segmentation on every source, with the scale handled per image.

The question this answers is the one Ch5 turns on: do tools built for real H&E find,
in a virtual stain, the nuclei that the TPAF says are there?

Sources
    TPAF        the source modality, segmented with the TPAF fine-tuned model. This is
                the reference for what is present, independent of any staining.
    real_HE     the same fields in real H&E.
    gray ...    the virtual variants.

Scale. Every stage of this project has been bitten by two numbers that were each
correct in their own frame and not the same thing: the mosaic grey, the mask settings,
the brightness estimator, the ring width, and the resolution HoVer-Net expects. So
nothing here is assumed.

    Cellpose      run at native resolution with `diameter` converted per image from one
                  physical size. The stored masks_he used diameter=20 px, but the real
                  H&E crops are all saved at 512 px whatever their extent, so their
                  scale runs 0.44-0.77 um/px -- that fixed 20 px meant anywhere from
                  8.8 to 15.4 um depending on the region. DIAM_UM fixes the physical
                  size instead and converts.
    HoVer-Net     resampled to 0.25 um/px, its training resolution. Left at 0.621 it
                  found 110-626 nuclei/mm2 where the same fields at 0.25 give 467-1742.
                  Nothing errors; the count is simply low, and low by an amount that
                  depends on the image's own scale.

The TPAF side keeps the project's own convention (cpsam_20260228_gray, flow 0.85,
diameter 30) rather than DIAM_UM, because that is what produced the masks every earlier
result was built on and changing it would make this incomparable with them.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import cv2
import numpy as np

SUR = 'results/path_screen/survey/_vhe'
DIAM_UM = 12.0          # physical nucleus diameter fed to base cpsam
TPAF_MPP = 0.621
TPAF_MODEL, TPAF_FLOW, TPAF_DIAM = 'cpsam_20260228_gray', 0.85, 30.0
VHE_VARIANTS = ('gray', 'gray_final', 'nuc_hi_final', 'nuc_flat', 'nuc_signed')


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def imwrite_u(path, im):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], im)
    if not ok:
        raise SystemExit(f'cannot encode {path}')
    buf.tofile(path)


def sources(rows):
    """(source, region_id, image, um_per_px) for everything to be segmented."""
    for r in rows:
        name = os.path.splitext(r['stage_name'])[0]
        x, y = int(r['crop_x']), int(r['crop_y'])
        w, h = int(r['crop_w']), int(r['crop_h'])
        he = imread_u(r['he_path'])
        if he is not None:
            yield 'real_HE', r['id'], he, float(r['crop_um']) / he.shape[1]
        tp = imread_u(r['patch_path'])
        if tp is not None:
            yield 'TPAF', r['id'], tp[y:y + h, x:x + w], TPAF_MPP
        for v in VHE_VARIANTS:
            p = os.path.join(SUR, 'stained', v, name + '.png')
            if os.path.exists(p):
                im = imread_u(p)
                if im is not None:
                    yield v, r['id'], im[y:y + h, x:x + w], TPAF_MPP


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--out', default='results/path_screen/survey/_downstream')
    ap.add_argument('--tool', choices=('cellpose', 'crops'), default='cellpose')
    ap.add_argument('--source', action='append', default=None,
                    help='Restrict to these sources; default is all.')
    ap.add_argument('--diam_um', type=float, default=DIAM_UM)
    ap.add_argument('--hv_mpp', type=float, default=0.25,
                    help='Resolution the HoVer-Net crops are written at.')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding='utf-8-sig')))
    if args.limit:
        rows = rows[:args.limit]
    want = set(args.source) if args.source else None

    if args.tool == 'crops':
        # Write the HoVer-Net inputs, resampled to its training resolution, plus a
        # scale table so the segmentation can be brought back to physical units.
        n = 0
        with open(os.path.join(args.out, 'crop_scale.csv'), 'w', newline='',
                  encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['source', 'id', 'src_mpp', 'out_mpp', 'out_w', 'out_h'])
            for src, rid, im, mpp in sources(rows):
                if src == 'TPAF' or (want and src not in want):
                    continue
                d = os.path.join(args.out, 'crops_hv', src)
                os.makedirs(d, exist_ok=True)
                f = mpp / args.hv_mpp
                sc = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                imwrite_u(os.path.join(d, rid + '.png'), sc)
                w.writerow([src, rid, f'{mpp:.4f}', args.hv_mpp, sc.shape[1], sc.shape[0]])
                n += 1
        print(f'{n} crops for HoVer-Net at {args.hv_mpp} um/px -> {args.out}/crops_hv')
        return

    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    base = models.CellposeModel(gpu=gpu)
    mp = os.path.expanduser(os.path.join('~', '.cellpose', 'models', TPAF_MODEL))
    tpaf = models.CellposeModel(gpu=gpu, pretrained_model=mp)
    print(f'cellpose gpu={gpu}; H&E-like: base cpsam, diameter {args.diam_um} um '
          f'converted per image; TPAF: {TPAF_MODEL} flow {TPAF_FLOW} '
          f'diameter {TPAF_DIAM} px (project convention)', flush=True)

    t0 = time.time()
    done = 0
    for src, rid, im, mpp in sources(rows):
        if want and src not in want:
            continue
        d = os.path.join(args.out, 'cp', src)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, rid + '.png')
        if os.path.exists(dst) and not args.redo:
            continue
        if src == 'TPAF':
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            lab, _, _ = tpaf.eval(g, diameter=TPAF_DIAM, flow_threshold=TPAF_FLOW,
                                  cellprob_threshold=0.0)
        else:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im
            lab, _, _ = base.eval(rgb, diameter=args.diam_um / mpp,
                                  flow_threshold=0.4, cellprob_threshold=0.0)
        # 16-bit label image: instance identity is needed for morphology, and these
        # rarely exceed a few thousand nuclei per region.
        if lab.max() > 65535:
            raise SystemExit(f'{src}/{rid}: {lab.max()} instances exceeds uint16')
        imwrite_u(dst, lab.astype(np.uint16))
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f'  {done} done  {el:.0f}s  ({el / done:.1f}s each)', flush=True)
    print(f'{done} segmentations in {time.time() - t0:.0f}s -> {args.out}/cp')


if __name__ == '__main__':
    main()
