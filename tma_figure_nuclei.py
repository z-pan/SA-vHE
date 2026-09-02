#!/usr/bin/env python3
"""Nuclear segmentation of the TPAF mosaics, for the figure's mask panel.

Run under the cellpose_clean environment -- cellpose is not installed in neuroclear.

Parameters are this project's, not the library defaults: flow_threshold 0.85 rather
than 0.4, and diameter 30 because the mosaic is at the native 0.621 um/px. The 78 px
figure recorded elsewhere belongs to the 204->512 upscaled patches, where nuclei
appear 2.51x larger.

Writes per core:
  nuc_labels.tif   uint16 instance labels
  nuc_overlay.png  outlines over the TPAF, for eyeballing
  nuc.json         count and area statistics
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import tifffile as tiff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='TMA_figures root.')
    ap.add_argument('--wells', required=True)
    ap.add_argument('--model', default=os.path.expanduser('~/.cellpose/models/cpsam_20260228_gray'))
    ap.add_argument('--diameter', type=float, default=30.0)
    ap.add_argument('--flow_threshold', type=float, default=0.85)
    ap.add_argument('--cellprob_threshold', type=float, default=0.0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from cellpose import models
    import torch
    gpu = args.device == 'cuda' and torch.cuda.is_available()
    print(f'cellpose model {os.path.basename(args.model)}  gpu={gpu}', flush=True)
    model = models.CellposeModel(gpu=gpu, pretrained_model=args.model)

    for well in [w.strip() for w in args.wells.split(',')]:
        od = os.path.join(args.dir, well)
        mp = os.path.join(od, 'mosaic.tif')
        if not os.path.exists(mp):
            print(f'  [skip] {well}'); continue
        mos = tiff.imread(mp).astype(np.float32)
        img = cv2.normalize(mos, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        print(f'  {well}: {img.shape} segmenting...', flush=True)
        out = model.eval(img, diameter=args.diameter,
                         flow_threshold=args.flow_threshold,
                         cellprob_threshold=args.cellprob_threshold)
        masks = out[0]
        n = int(masks.max())
        tiff.imwrite(os.path.join(od, 'nuc_labels.tif'),
                     masks.astype(np.uint16), compression='zlib')

        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        edges = cv2.morphologyEx((masks > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8))
        rgb[edges > 0] = (255, 60, 60)
        cv2.imwrite(os.path.join(od, 'nuc_overlay.png'),
                    cv2.resize(rgb, None, fx=0.4, fy=0.4)[..., ::-1])

        areas = np.bincount(masks.ravel())[1:] if n else np.array([0])
        px_um2 = 0.621 ** 2
        json.dump(dict(well=well, n_nuclei=n,
                       median_area_um2=float(np.median(areas) * px_um2) if n else 0,
                       density_per_mm2=float(n / (masks.size * px_um2 / 1e6)),
                       diameter=args.diameter, flow_threshold=args.flow_threshold),
                  open(os.path.join(od, 'nuc.json'), 'w'), indent=2)
        print(f'    {n} nuclei, median area '
              f'{np.median(areas)*px_um2:.0f} um^2, '
              f'{n/(masks.size*px_um2/1e6):.0f} /mm^2', flush=True)


if __name__ == '__main__':
    main()
