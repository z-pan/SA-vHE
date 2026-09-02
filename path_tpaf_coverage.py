#!/usr/bin/env python3
"""Where on each slide the TPAF acquisition actually reaches, from the verified links.

Tier B of the evaluation compares distributions rather than pairs, and a distribution
comparison is only about staining if both sides look at the same tissue. TPAF does not
cover the whole slide -- on 240828_pt1 it reaches 3.07 x 2.86 mm of an 8.6 x 10.2 mm
specimen -- so sampling real H&E from the whole slide and virtual H&E from the TPAF
would put a difference in tissue composition into every metric and call it staining.

The covered region is taken from the 148 hand-verified TPAF-to-H&E correspondences, not
from matching the stitched TPAF silhouette against the slide. That was tried: over every
rotation and mirror the best silhouette IoU was 0.317 where a true match would be 0.7 to
0.9, because the montage does not span the same tissue as the WSI. The manual links are
the only correspondence in this dataset that is known to be right.

What is produced is a conservative region: the union of the linked tiles, dilated by
--dilate_um and closed. Conservative in the sense that TPAF certainly reaches the linked
tiles, and probably reaches somewhat further; it is a lower bound on coverage, not an
estimate of its boundary. That asymmetry matters for how it should be used -- sampling
inside it is safe, and any claim about what lies outside it is not supported.

The linked tiles are themselves a biased sample: they were picked for pathological
content. So the region they span is used, but the *tiles* inside it are drawn afresh
from the full tile index, which restores an unbiased sample within a defensible boundary.

    python path_tpaf_coverage.py
    python path_tpaf_coverage.py --write   # record it in each sample's view.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re

import cv2
import numpy as np


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if im is None:
        raise SystemExit(f'cannot decode {path}')
    return im


def read_source(d):
    kv = {}
    for line in io.open(os.path.join(d, 'source.txt'), encoding='utf-8'):
        k, _, v = line.strip().partition('=')
        if k:
            kv[k] = v
    return kv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey', default='results/path_screen/survey')
    ap.add_argument('--links',
                    default='results/path_screen/survey/_candidates/tpaf_links.csv')
    ap.add_argument('--out', default='results/path_screen/survey/_tierB')
    ap.add_argument('--dilate_um', type=float, default=400.0,
                    help='How far past the linked tiles the region is taken to extend. '
                         'A TPAF FOV is 636 um across and the links name its centre '
                         'region, so a few hundred um is the width of one frame, not a '
                         'guess about unseen tissue.')
    ap.add_argument('--close_um', type=float, default=800.0,
                    help='Morphological closing, to join links that are separated only '
                         'because no candidate happened to be picked between them.')
    ap.add_argument('--write', action='store_true',
                    help="Append the coverage to each sample's candidates/view.txt.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    links = list(csv.DictReader(io.open(args.links, encoding='utf-8-sig')))
    by_sample = {}
    for r in links:
        if r['tpaf_path']:
            by_sample.setdefault(r['sample'], set()).add(r['tile_id'])

    print(f'{"sample":<12}{"links":>7}{"tiles in":>10}{"of":>7}{"%":>7}'
          f'{"cover mm2":>11}{"tissue mm2":>12}{"%":>7}')
    rows = []
    for sample in sorted(by_sample):
        d = os.path.join(args.survey, sample)
        src = read_source(d)
        ds = int(src['thumb_ds'])
        tile = int(src['tile_px'])
        um = float(src['um_per_px'])
        thumb = imread_u(os.path.join(d, 'thumbnail.png'))
        H, W = thumb.shape[:2]

        cov = np.zeros((H, W), np.uint8)
        for tid in by_sample[sample]:
            m = re.match(r'y(\d+)_x(\d+)', tid)
            y, x = int(m.group(1)) // ds, int(m.group(2)) // ds
            s = max(1, tile // ds)
            cov[y:y + s, x:x + s] = 255
        dil = max(1, int(round(args.dilate_um / um / ds)))
        clo = max(1, int(round(args.close_um / um / ds)))
        cov = cv2.morphologyEx(cov, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clo, clo)))
        cov = cv2.dilate(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))

        thr = int(src.get('tissue_thresh', 215))
        g = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        tissue = ((g < thr).astype(np.uint8)) * 255
        tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Coverage only counts where there is tissue: dilating past the specimen edge
        # onto glass would inflate the area and let the sampler ask for tiles that do
        # not exist.
        cov = cv2.bitwise_and(cov, tissue)

        idx = list(csv.DictReader(io.open(os.path.join(d, 'index.csv'),
                                          encoding='utf-8')))
        inside = []
        for r in idx:
            cy = (int(r['y']) + tile / 2) / ds
            cx = (int(r['x']) + tile / 2) / ds
            if 0 <= int(cy) < H and 0 <= int(cx) < W and cov[int(cy), int(cx)]:
                inside.append(r['tile_id'])
        px_mm2 = (ds * um / 1000) ** 2
        cov_mm2 = float((cov > 0).sum()) * px_mm2
        tis_mm2 = float((tissue > 0).sum()) * px_mm2
        print(f'{sample:<12}{len(by_sample[sample]):>7}{len(inside):>10}{len(idx):>7}'
              f'{100 * len(inside) / len(idx):>7.1f}{cov_mm2:>11.1f}{tis_mm2:>12.1f}'
              f'{100 * cov_mm2 / tis_mm2:>7.1f}')
        rows.append(dict(sample=sample, n_links=len(by_sample[sample]),
                         n_tiles_inside=len(inside), n_tiles_total=len(idx),
                         cover_mm2=round(cov_mm2, 2), tissue_mm2=round(tis_mm2, 2),
                         cover_pct=round(100 * cov_mm2 / tis_mm2, 1)))

        cv2.imwrite(os.path.join(args.out, f'coverage_{sample}.png'), cov)
        vis = thumb.copy()
        vis[cov > 0] = (0.55 * vis[cov > 0] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, f'coverage_{sample}_overlay.png'), vis)
        with io.open(os.path.join(args.out, f'tiles_{sample}.txt'), 'w',
                     encoding='utf-8') as fh:
            fh.write('\n'.join(sorted(inside)) + '\n')
        if args.write:
            vp = os.path.join(d, 'candidates', 'view.txt')
            prev = io.open(vp, encoding='utf-8').read() if os.path.exists(vp) else ''
            prev = '\n'.join(l for l in prev.split('\n')
                             if not l.startswith('tpaf_cover'))
            with io.open(vp, 'w', encoding='utf-8') as fh:
                fh.write(prev.rstrip('\n') + ('\n' if prev.strip() else ''))
                fh.write(f'tpaf_cover_mm2={cov_mm2:.2f}\n')
                fh.write(f'tpaf_cover_pct_of_tissue={100 * cov_mm2 / tis_mm2:.1f}\n')
                fh.write(f'tpaf_cover_tiles={len(inside)}/{len(idx)}\n')
                fh.write('tpaf_cover_source=union of verified links, dilated '
                         f'{args.dilate_um:g} um, closed {args.close_um:g} um\n')

    with io.open(os.path.join(args.out, 'coverage.csv'), 'w', newline='',
                 encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot_in = sum(r['n_tiles_inside'] for r in rows)
    print(f'\n{tot_in} real H&E tiles inside the covered regions -> {args.out}')
    print('Overlays written; check that the green area sits where TPAF was acquired')
    print('before any of it is used as a sampling frame.')


if __name__ == '__main__':
    main()
