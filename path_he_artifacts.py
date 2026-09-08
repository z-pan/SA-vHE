#!/usr/bin/env python3
"""How much of a real H&E slide carries staining and mounting artefacts.

Why this is worth measuring
---------------------------
Ch5 argues that virtual staining makes TPAF usable in an H&E workflow, and part of
"usable" is that it skips the failure modes of physical staining. Real slides here carry
black inclusions -- debris, mounting-medium specks -- that no tissue can produce, and
the virtual stain cannot produce them either: they have no source in the TPAF and the
generator never saw one. That is a claim about a whole slide, so it has to be measured
on slide tiles, not on the 148 hand-picked evaluation regions, which are clean (their
tissue never goes below V=165, and the artefacts do not appear in them at all).

What counts as an artefact
--------------------------
Dark *and* achromatic. Haematoxylin is violet and eosin is pink, so tissue that is dark
is also saturated; debris is neutral. Calibrated against tiles inspected by eye:

    V < 80 and S < 100      black debris     378 - 3242 px per tile
                            dark epithelium, red blood cells, out-of-focus tissue
                                             0 - 4 px per tile

Components below MIN_PX are dropped: a handful of scattered pixels is sensor noise,
and the debris in these slides is tens to thousands of pixels in one blob.

Out-of-focus regions and dense red blood cells are what a naive "darkest pixels" or
"highest frequency" rule finds instead, and both are tissue.

    python path_he_artifacts.py
    python path_he_artifacts.py --vhe          # also check the virtual stain
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import os

import cv2
import numpy as np
from scipy import ndimage

SUR = 'results/path_screen/survey'
SLIDES = ('240703', '240720', '240729', '240817', '240828_pt1')
V_MAX, S_MAX = 80, 100      # see docstring
MIN_PX = 30                 # a blob, not a few noisy pixels
WHITE = 235                 # V above this is slide background, not tissue


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def measure(bgr):
    """Artefact pixel count, blob count, and tissue pixel count for one image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    tissue = v < WHITE
    if tissue.sum() < 1000:
        return None
    m = (v < V_MAX) & (s < S_MAX)
    lab, n = ndimage.label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = sizes >= MIN_PX
        m = keep[lab]
        n = int(keep.sum())
    return int(m.sum()), n, int(tissue.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sur', default=SUR)
    ap.add_argument('--vhe', action='store_true',
                    help='Also measure the virtual stain over the 148 regions.')
    ap.add_argument('--out', default=SUR + '/_he_artifacts.csv')
    args = ap.parse_args()

    print('真实 H&E 整张切片的 tile   （%d < V, %d < S, 连通域 >= %d px）'
          % (V_MAX, S_MAX, MIN_PX))
    print('%-14s%8s%10s%12s%14s%12s'
          % ('slide', 'tiles', '含杂质', '占 tile 比', '杂质占组织', '最大单块 px'))
    rows = []
    tot = collections.Counter()
    for sl in SLIDES:
        files = sorted(glob.glob(os.path.join(args.sur, sl, 'tiles', '*')))
        hit = 0
        px = 0
        tis = 0
        biggest = 0
        for p in files:
            im = imread_u(p)
            if im is None:
                continue
            r = measure(im)
            if r is None:
                continue
            a, nblob, t = r
            px += a
            tis += t
            if a:
                hit += 1
                biggest = max(biggest, a)
                rows.append(dict(slide=sl, tile=os.path.basename(p),
                                 artifact_px=a, blobs=nblob, tissue_px=t))
        tot['tiles'] += len(files)
        tot['hit'] += hit
        tot['px'] += px
        tot['tis'] += tis
        print('%-14s%8d%10d%11.1f%%%13.4f%%%12d'
              % (sl, len(files), hit, 100 * hit / max(len(files), 1),
                 100 * px / max(tis, 1), biggest))
    print('%-14s%8d%10d%11.1f%%%13.4f%%'
          % ('合计', tot['tiles'], tot['hit'],
             100 * tot['hit'] / max(tot['tiles'], 1),
             100 * tot['px'] / max(tot['tis'], 1)))

    if args.vhe:
        man = os.path.join(args.sur, '_vhe', 'vhe_manifest.csv')
        mrows = list(csv.DictReader(io.open(man, encoding='utf-8-sig')))
        stained = os.path.join(args.sur, '_vhe', 'stained')
        versions = ['real_HE'] + [v for v in ('nuc_flat_final', 'gray_final',
                                              'cyclegan_final', 'sacut')
                                  if os.path.isdir(os.path.join(stained, v))]
        print('\n148 个评估区域   （评估用的区域是否干净）')
        print('%-18s%10s%12s%14s' % ('source', '区域数', '含杂质', '杂质占组织'))
        for v in versions:
            hit = px = tis = n = 0
            for r in mrows:
                if v == 'real_HE':
                    im = imread_u(r['he_path'])
                else:
                    name = os.path.splitext(r['stage_name'])[0]
                    p = os.path.join(stained, v, name + '.png')
                    if not os.path.exists(p):
                        continue
                    im = imread_u(p)
                    if im is not None:
                        x, y, w, h = (int(r[k]) for k in
                                      ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
                        im = im[y:y + h, x:x + w]
                if im is None:
                    continue
                res = measure(im)
                if res is None:
                    continue
                a, _, t = res
                n += 1
                px += a
                tis += t
                hit += bool(a)
            print('%-18s%10d%12d%13.4f%%'
                  % (v, n, hit, 100 * px / max(tis, 1)))

    if rows:
        with io.open(args.out, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['slide', 'tile', 'artifact_px',
                                               'blobs', 'tissue_px'])
            w.writeheader()
            for r in sorted(rows, key=lambda z: -z['artifact_px']):
                w.writerow(r)
        print('\n%d 张含杂质的 tile -> %s' % (len(rows), os.path.abspath(args.out)))


if __name__ == '__main__':
    main()
