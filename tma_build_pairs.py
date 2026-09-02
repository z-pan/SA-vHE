#!/usr/bin/env python3
"""Build TPAF/H&E pairs for every TMA core: one pair per 1024 px TPAF tile.

Pipeline
--------
1. stitch the core's nine tiles into a mosaic          (tma_stitch)
2. place the mosaic on the H&E core                    (disc centre, scale pinned)
3. for each tile, crop the H&E region it covers        (+ --redundancy margin)

Geometry is measured, not fitted: TPAF 0.621 um/px and H&E 0.4429 um/px give a
1.4021 ratio, cross-checked against core diameter to 0.5%. That leaves translation,
and since the mosaic is a rectangle driven to cover the core, its centre is the core
centre -- so only the H&E disc has to be segmented, where tissue against white is
unambiguous.

Accuracy
--------
About 100 px (45 um) at H&E scale, i.e. 7% of a tile, judged against the operator's
own {well}_same_FOV crops. Mask-IoU search, disc centring and disc centring plus a
local NCC refine all land within 104 px median of those crops, which is the level of
disagreement of the hand crops themselves rather than of any one method -- on 15C,
disc placement lines up an internal cleft in both modalities while sitting 110 px
from the hand crop. Good enough for unpaired training, colour and texture statistics,
and qualitative comparison; not for pixel-wise metrics. Per-tile SIFT/RANSAC refine
(as in notebook_AF_HE_registration.ipynb Step 2) is the upgrade path.

Output per tile, following the existing 240817 convention:
  {well}_T00N_AF.tif      1024x1024 TPAF tile as acquired
  {well}_T00N_HE.tif      H&E crop at native 0.4429 um/px, with redundancy
  {well}_T00N_HE_reg.tif  the same crop resampled to TPAF scale
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np
import tifffile as tiff

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tma_stitch import LAYOUT, build_mosaic, load_tiles, solve_positions, tile_number  # noqa: E402
from tma_align import align as similarity_align, crop_he_for_tile  # noqa: E402

SCALE = 0.621 / 0.4429
DS = 8


def g(a):
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def he_disc(he_gray, expect_r=None):
    """Core centre and radius in full-resolution H&E px, found on a 1/8 image.

    The largest connected component is the core -- unless pale staining splits the
    core into pieces, in which case it is one piece and the centre is badly off. The
    mosaic was driven to cover the core, so its half-width says what the radius should
    be; if the component comes back well under that, fall back to the extent of all
    tissue. Without this, cores like 3B and 3D report r~1070 against a normal
    1850-2050 and yield 0 usable pairs.
    """
    hs = cv2.resize(he_gray, None, fx=1 / DS, fy=1 / DS, interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(hs.astype(np.float32), (0, 0), 2)
    k = cv2.morphologyEx((b < 225).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(k)
    if n < 2:
        return None

    def circle_of(mask):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        pts = np.vstack(cs) if len(cs) > 1 else cs[0]
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        return cy, cx, r

    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    got = circle_of(lab == i)
    if got is None:
        return None
    if expect_r is not None and got[2] * DS < 0.75 * expect_r:
        # drop specks, then take the extent of everything that is left
        keep = np.zeros_like(k)
        big = max(stats[1:, cv2.CC_STAT_AREA].max(), 1)
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] > 0.02 * big:
                keep |= (lab == j).astype(np.uint8)
        alt = circle_of(keep)
        if alt is not None and alt[2] > got[2]:
            got = alt
    return got[0] * DS, got[1] * DS, got[2] * DS



def refine_placement(mos, he_gray, y0, x0, rad=380, ds=4):
    """Bounded mask search around the disc-centre guess. Returns (y, x, score).

    The disc centre assumes the mosaic was aimed at the geometric centre of the core,
    but it was aimed at tissue, and on some cores that is 150-300 H&E px away -- past
    what the redundancy margin absorbs. Per-tile matching is too noisy to fix it (mask
    ncc 0.15-0.42), so the whole mosaic is the template: nine tiles' worth of structure
    in one correlation.

    Unbounded, this search wanders off on dim cores; hence rad, which keeps it inside
    the neighbourhood the disc centre already established.
    """
    mm = cv2.GaussianBlur(mos.astype(np.float32), (0, 0), 12)
    mm = (mm > max(np.percentile(mm, 45), mm.max() * 0.06)).astype(np.float32)
    mm = cv2.resize(mm, None, fx=SCALE / ds, fy=SCALE / ds, interpolation=cv2.INTER_AREA)

    hm = cv2.GaussianBlur(he_gray.astype(np.float32), (0, 0), 12)
    hm = (hm < 212).astype(np.float32)

    th, tw = mm.shape
    wy0 = max(0, int(y0 - rad)); wx0 = max(0, int(x0 - rad))
    wy1 = min(hm.shape[0], int(y0 + th * ds + rad)); wx1 = min(hm.shape[1], int(x0 + tw * ds + rad))
    win = cv2.resize(hm[wy0:wy1, wx0:wx1], None, fx=1 / ds, fy=1 / ds,
                     interpolation=cv2.INTER_AREA)
    if win.shape[0] <= th or win.shape[1] <= tw:
        return y0, x0, -1.0
    r = cv2.matchTemplate(win, mm, cv2.TM_CCOEFF_NORMED)
    _, sc, _, mx = cv2.minMaxLoc(r)
    return wy0 + mx[1] * ds, wx0 + mx[0] * ds, float(sc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tpaf_dir', required=True)
    ap.add_argument('--he_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--wells', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--redundancy', type=float, default=0.05,
                    help='H&E margin as a fraction of the tile, to absorb the ~7% offset.')
    ap.add_argument('--min_tissue', type=float, default=0.10,
                    help='Drop a pair whose H&E crop has less tissue than this. '
                         'Corner tiles of a square mosaic over a round core are mostly blank.')
    ap.add_argument('--save_mosaic', action='store_true')
    ap.add_argument('--qc_every', type=int, default=10)
    ap.add_argument('--force', action='store_true', help='Redo cores that already have output.')
    ap.add_argument('--no_refine', action='store_true', help='Disc centre only, no refinement.')
    ap.add_argument('--refine_rad', type=int, default=380,
                    help='Refinement search radius in H&E px around the disc centre.')
    ap.add_argument('--max_angle', type=float, default=5.0,
                    help='Rotation search bound in degrees.')
    ap.add_argument('--lock_angle', type=float, default=None,
                    help='Pin rotation (deg) to a slide-level constant; fit translation only.')
    ap.add_argument('--lock_scale', type=float, default=None,
                    help='Pin the scale correction likewise.')
    ap.add_argument('--refine_min', type=float, default=0.12,
                    help='Keep the refined position only above this mask correlation.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    done_files = set(os.listdir(args.out_dir))
    qc = os.path.join(args.out_dir, '_qc')
    os.makedirs(qc, exist_ok=True)

    dirs = {}
    for d in sorted(os.listdir(args.tpaf_dir)):
        m = re.search(r'(?<![0-9A-Za-z])(\d{1,2}[A-H])\.tif\.frames$', d)
        if m and not d.lower().startswith('not used'):
            dirs[m.group(1)] = os.path.join(args.tpaf_dir, d)
    if args.wells:
        want = [w.strip() for w in args.wells.split(',')]
        dirs = {w: dirs[w] for w in want if w in dirs}
    print(f'{len(dirs)} cores', flush=True)

    # written per core, not at the end: long runs get interrupted and the metadata
    # for everything already on disk would be lost
    csv_p = os.path.join(args.out_dir, 'pairs.csv')
    csv_new = not (os.path.exists(csv_p) and os.path.getsize(csv_p) > 0)
    FIELDS = ['well', 'tile', 'cell', 'he_y', 'he_x', 'side', 'tissue', 'af_mean',
              'disc_r', 'mosaic_h', 'mosaic_w']
    csv_fh = open(csv_p, 'w' if csv_new else 'a', newline='', encoding='utf-8')
    csv_wr = csv.DictWriter(csv_fh, fieldnames=FIELDS)
    if csv_new:
        csv_wr.writeheader(); csv_fh.flush()

    rows, n_pair, n_drop = [], 0, 0
    for i, (well, d) in enumerate(sorted(dirs.items(), key=lambda kv: (int(kv[0][:-1]), kv[0][-1]))):
        if args.limit and i >= args.limit:
            break
        he_p = os.path.join(args.he_dir, f'{well}.tif')
        if not os.path.exists(he_p):
            print(f'  [skip] {well}: no H&E core', flush=True); continue
        # resumable: a full run takes hours and gets interrupted, so skip finished cores
        if not args.force and any(f.startswith(f'{well}_T') and f.endswith('_HE_reg.tif')
                                  for f in done_files):
            n_have = sum(1 for f in done_files
                         if f.startswith(f'{well}_T') and f.endswith('_AF.tif'))
            print(f'  {well:>4} already done ({n_have} pairs), skipping', flush=True)
            continue
        tiles = load_tiles(d)
        if len(tiles) != 9:
            print(f'  [skip] {well}: {len(tiles)} tiles', flush=True); continue
        raw = {}
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.tif', '.tiff')):
                continue
            k = tile_number(f, well)
            if k is not None:
                raw[k] = tiff.imread(os.path.join(d, f))

        pos = solve_positions(tiles)
        mos, (mh, mw) = build_mosaic(tiles, pos)
        oy = min(p[0] for p in pos.values())
        ox = min(p[1] for p in pos.values())
        if args.save_mosaic:
            tiff.imwrite(os.path.join(args.out_dir, f'{well}_mosaic.tif'),
                         np.clip(mos, 0, 65535).astype(np.uint16))

        he = tiff.imread(he_p)
        disc = he_disc(g(he), expect_r=max(mh, mw) * SCALE / 2.0)
        if disc is None:
            print(f'  [skip] {well}: no disc', flush=True); continue
        cy, cx, rr = disc
        my0 = cy - mh * SCALE / 2.0
        mx0 = cx - mw * SCALE / 2.0
        angle, fit_scale = 0.0, 1.0
        if not args.no_refine:
            a = similarity_align(mos, g(he), my0, mx0, rad=args.refine_rad,
                                 max_angle=args.max_angle,
                                 lock_angle=args.lock_angle, lock_scale=args.lock_scale)
            shift = float(np.hypot(a['y'] - my0, a['x'] - mx0))
            rsc = a['score']
            if rsc > args.refine_min:
                my0, mx0 = a['y'], a['x']
                angle, fit_scale = a['angle'], a['scale']
            else:
                shift = 0.0
        else:
            rsc, shift = '', 0.0

        red = int(round(1024 * args.redundancy))
        written = []
        side_he = int(round((1024 + 2 * red) * SCALE))
        kept = 0
        side_tpaf = 1024 + 2 * red
        for k in sorted(tiles):
            ty, tx = pos[k][0] - oy, pos[k][1] - ox        # tile origin in mosaic px
            # rotation is in the model now, so an axis-aligned slice no longer matches
            # the tile: resample H&E through the inverse similarity instead
            crop = crop_he_for_tile(he[..., :3], (ty - red, tx - red), (my0, mx0),
                                    angle, fit_scale, (mh, mw), side_tpaf,
                                    upsample=SCALE)
            hy2 = int(round(my0 + (ty - red) * SCALE))
            hx2 = int(round(mx0 + (tx - red) * SCALE))
            tissue = float((g(crop) < 205).mean())
            if tissue < args.min_tissue:
                n_drop += 1; continue
            base = f'{well}_T{k:03d}'
            # keep the tile exactly as acquired: both emission channels (618 and 553 nm)
            # matter downstream, the grey average exists only to drive registration
            tiff.imwrite(os.path.join(args.out_dir, f'{base}_AF.tif'), raw[k],
                         compression='zlib')
            tiff.imwrite(os.path.join(args.out_dir, f'{base}_HE.tif'), crop, compression='zlib')
            reg = crop_he_for_tile(he[..., :3], (ty - red, tx - red), (my0, mx0),
                                   angle, fit_scale, (mh, mw), side_tpaf, upsample=1.0)
            tiff.imwrite(os.path.join(args.out_dir, f'{base}_HE_reg.tif'), reg, compression='zlib')
            rows.append(dict(well=well, tile=k, cell=f'{LAYOUT[k][0]}{LAYOUT[k][1]}',
                             he_y=hy2, he_x=hx2, side=side_he, tissue=round(tissue, 3),
                             af_mean=round(float(tiles[k].mean()), 2),
                             disc_r=round(rr), mosaic_h=mh, mosaic_w=mw))
            kept += 1; n_pair += 1
            written.append(k)

        if written:
            csv_wr.writerows([r for r in rows if r['well'] == well]); csv_fh.flush()
        print(f'  {well:>4} mosaic {mw}x{mh} disc r{rr:.0f} refine {rsc if rsc=="" else f"{rsc:.2f}"}'
              f' shift {shift:.0f}px rot {angle:+.2f} sc {fit_scale:.4f}'
              f' -> {kept}/9 pairs', flush=True)
        if i % args.qc_every == 0 and written:
            k0 = written[len(written) // 2]      # a tile that survived, not a fixed one
            b = f'{well}_T{k0:03d}'
            af = cv2.normalize(tiles[k0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            p = np.concatenate([
                cv2.cvtColor(cv2.resize(af, (360, 360)), cv2.COLOR_GRAY2RGB),
                cv2.resize(tiff.imread(os.path.join(args.out_dir, f'{b}_HE_reg.tif')), (360, 360)),
                cv2.resize(he[..., :3], (360, 360), interpolation=cv2.INTER_AREA)], axis=1)
            cv2.putText(p, b, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(qc, f'{b}.png'), p[..., ::-1])

    csv_fh.close()
    ncore = len({r['well'] for r in rows})
    total = sum(1 for f in os.listdir(args.out_dir) if f.endswith('_AF.tif'))
    print(f'\nthis run: {n_pair} pairs from {ncore} cores ({n_drop} dropped: blank or off-edge)')
    print(f'on disk : {total} pairs -> {args.out_dir}')


if __name__ == '__main__':
    main()
