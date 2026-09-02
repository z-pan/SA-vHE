#!/usr/bin/env python3
"""Mark the regions a scoring run picked on the slide, and guess their TPAF FOV.

path_report.py answers "which tiles score high". This answers "where is that on the
slide, and which TPAF FOV do I re-image to get its virtual H&E" -- the step between
picking a region and generating vHE for it on demand.

Three of the four things it reports are exact; the fourth is a guess, and is labelled
as one.

Exact
-----
box position   index.csv holds each tile's (y, x) in full-resolution pixels; dividing
               by thumb_ds from source.txt puts it on the thumbnail.
physical size  tile_px * um_per_px, both from source.txt.
tissue-relative position
               A slide is mostly glass, so a fraction of the canvas would be
               meaningless. The thumbnail is thresholded to a tissue mask and the
               fractions are taken inside its bounding box.

A guess
-------
TPAF line / FOV index. There is no usable transform between WSI coordinates and the
TPAF grid in this dataset: the Fiji TileConfiguration_new.registered.txt is degenerate
(all 179 entries at 0,0), the un-registered one holds a nominal 0%-overlap grid while
the measured step is 754 px (~26% overlap), the per-FOV .txt exports carry no stage
coordinates, and matching an H&E FOV into the montage gives no sharp peak. So the
estimate assumes the TPAF grid spans the same tissue bounding box as the WSI, which is
plausible and unverified.

That assumption is correctable. Confirm one region by hand, pass it back with --anchor,
and the mapping is refit: one anchor fixes the offset, two or more fit an independent
scale and offset per axis. The tool gets better as it is used.

    python path_locate.py --tiles results/path_screen/240817_HE_slide \\
        --scores conch_slide.csv --out results/path_screen/240817_picks --top 3

    # after checking one by hand:
    python path_locate.py ... --anchor y12695_x7406=12,3
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys

import cv2
import numpy as np
import tifffile as tiff

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_structures import (ALL_ENTRIES, ECM_KEYS, HGSOC_KEYS,  # noqa: E402
                             KEYS, PROBE_KEYS)

# ALL_ENTRIES, not STRUCTURES, so a --probes run can label its columns too.
ZH = {s['key']: s['zh'] for s in ALL_ENTRIES}

# TPAF acquisition grid for 240817HOC240827-4, counted from the per-FOV .txt exports in
# slide_TPAF .../data_description. Odd lines from 7 on are acquired right-to-left, so
# their FOV index runs backwards relative to slide x.
TPAF_LINE_FOVS = {1: 4, 2: 7, 3: 7, 4: 8, 5: 8, 6: 8, 7: 7, 8: 9, 9: 8, 10: 7, 11: 6,
                  12: 7, 13: 7, 14: 6, 15: 6, 16: 5, 17: 6, 18: 8, 19: 7, 20: 6, 21: 5,
                  22: 5, 23: 3}
TPAF_REV_LINES = {7, 9, 11, 13, 15, 17, 19, 21, 23}



def tile_path(tiles_dir, tid):
    """Tiles are PNG or JPEG depending on how they were cut; the id does not say
    which, and a set may legitimately mix the two."""
    for ext in ('.png', '.jpg'):
        p = os.path.join(tiles_dir, 'tiles', tid + ext)
        if os.path.exists(p):
            return p
    raise SystemExit(f'no tile image for {tid} in {tiles_dir}/tiles')

def read_source(d):
    kv = {}
    for line in open(os.path.join(d, 'source.txt'), encoding='utf-8'):
        k, _, v = line.strip().partition('=')
        kv[k] = v
    return kv


def tissue_bbox(thumb, thresh=215):
    """Bounding box of the tissue, so fractions are relative to the specimen rather
    than to a canvas that is mostly glass."""
    g = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY) if thumb.ndim == 3 else thumb
    m = (g < thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(m)
    if not len(ys):
        raise SystemExit('no tissue found in thumbnail -- wrong tissue_thresh?')
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def fit_axis(anchor_pairs):
    """frac -> grid coordinate, as scale * f + offset.

    0 anchors  identity over the full range (the naive assumption)
    1 anchor   offset only, scale kept at 1 -- one point cannot separate the two
    2+         least squares on both
    """
    if not anchor_pairs:
        return 1.0, 0.0
    if len(anchor_pairs) == 1:
        f, g = anchor_pairs[0]
        return 1.0, g - f
    F = np.array([p[0] for p in anchor_pairs])
    G = np.array([p[1] for p in anchor_pairs])
    A = np.vstack([F, np.ones_like(F)]).T
    (s, o), *_ = np.linalg.lstsq(A, G, rcond=None)
    return float(s), float(o)


def estimate_tpaf(fy, fx, fit_y, fit_x):
    """Tissue-relative position -> TPAF line and FOV index, plus neighbours.

    The line comes from the vertical fraction across all 23 lines; the FOV index from
    the horizontal fraction across however many FOVs that line holds, reversed on the
    serpentine lines.
    """
    lines = sorted(TPAF_LINE_FOVS)
    sy, oy = fit_y
    gy = np.clip(sy * fy + oy, 0.0, 1.0)
    li = int(round(gy * (len(lines) - 1)))
    line = lines[li]

    n = TPAF_LINE_FOVS[line]
    sx, ox = fit_x
    gx = np.clip(sx * fx + ox, 0.0, 1.0)
    idx = int(round(gx * (n - 1)))
    if line in TPAF_REV_LINES:
        idx = (n - 1) - idx

    alt = []
    for dl in (-1, 0, 1):
        j = li + dl
        if 0 <= j < len(lines):
            alt.append(f'Line-{lines[j]:02d}')
    return line, idx, n, line in TPAF_REV_LINES, alt


def rot_cw(deg, w, h):
    """Rotation matrix for `deg` clockwise about the centre of a w x h image, with the
    canvas grown so nothing is cut off. cv2 measures angles anticlockwise, hence -deg.
    """
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return M, nw, nh


def apply_M(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return (pts @ M[:, :2].T) + M[:, 2]


def rotated_crop(src, cy, cx, side, deg, out_px):
    """A `side`-pixel square of the source, seen rotated `deg` clockwise, centred on
    (cy, cx) in full-resolution pixels.

    Read from the source rather than from the saved tile: rotating the saved tile
    leaves the corners empty, because a square rotated inside its own bounds does not
    cover them. Reading a window wide enough to survive the rotation -- side*sqrt(2)
    -- and cropping back to `side` afterwards gives a full frame. The region is the
    same either way; only the sampling changes, so the pick this belongs to is
    unaffected.

    White outside the slide, matching the glass, so an edge region does not come back
    with black wedges that read as tissue.
    """
    H, W = src.shape[:2]
    half = int(np.ceil(side * np.sqrt(2) / 2)) + 2
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    win = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, y1), min(W, x1)
    if sy1 > sy0 and sx1 > sx0:
        patch = np.asarray(src[sy0:sy1, sx0:sx1])
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        else:
            patch = patch[..., :3][..., ::-1]  # source is RGB, cv2 writes BGR
        win[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = patch
    wh, ww = win.shape[:2]
    M = cv2.getRotationMatrix2D((ww / 2, wh / 2), -deg, 1.0)
    rot = cv2.warpAffine(win, M, (ww, wh), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    a = (wh - side) // 2
    b = (ww - side) // 2
    return cv2.resize(rot[a:a + side, b:b + side], (out_px, out_px),
                      interpolation=cv2.INTER_AREA)


def draw_marks(img, marks, k=1, ox=0, oy=0, font=0.8, thick=2):
    """Stamp the numbered outlines onto `img`, which may be a scaled crop of the
    canvas they were computed on.

    Red is the per-structure pick set, green the second tier added by --extra; the two
    answer different questions and a figure pool should not silently mix them.

    Drawn rather than baked in, so the region map and the zoomed-in ROI map can be
    rendered at their own resolutions. Labelling once on the full canvas and then
    upscaling a crop of it magnifies the text along with everything else, and a 3 mm
    fragment blown up to 900 px ends up with numbers larger than the tissue they point
    at.
    """
    for n, quad, colour in marks:
        q = (np.asarray(quad) - (ox, oy)) * k
        q = q.round().astype(np.int32)
        cv2.polylines(img, [q], True, colour, thick)
        lab = str(n)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        lx = min(max(int(q[:, 0].min()), 0), img.shape[1] - tw - 8)
        ly = int(q[:, 1].min()) - 4
        if ly - th - 8 < 0:
            ly = int(q[:, 1].max()) + th + 8
        cv2.rectangle(img, (lx, ly - th - 6), (lx + tw + 8, ly + 4), colour, -1)
        cv2.putText(img, lab, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    font, (255, 255, 255), thick)


def parse_anchor(s):
    m = re.match(r'^(?P<tid>[^=]+)=(?P<line>\d+),(?P<idx>\d+)$', s.strip())
    if not m:
        raise SystemExit(f'bad --anchor {s!r}; expected tile_id=LINE,FOVINDEX')
    return m['tid'], int(m['line']), int(m['idx'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', type=int, default=3, help='Per structure.')
    ap.add_argument('--top_nonecm', type=int, default=None,
                    help='Separate quota for the non-ECM structures, which are there '
                         'to show what else the slide holds rather than to be compared '
                         'across samples. --top 3 --top_nonecm 1 is the survey set: '
                         'three regions for each of the five ECM structures, one for '
                         'each of the other thirteen. Defaults to --top.')
    ap.add_argument('--only_ecm', action='store_true')
    ap.add_argument('--min_sep', type=float, default=2.0,
                    help='Minimum centre-to-centre spacing between picks for the SAME '
                         'structure, in tile widths. Without it, a structure with one '
                         'strong locus spends its whole quota on neighbouring tiles. '
                         '0 disables. Spacing is not enforced across structures -- two '
                         'structures pointing at one region is a real finding.')
    ap.add_argument('--tile_ids', default=None,
                    help='Comma-separated tile_ids, instead of taking the top scorers.')
    ap.add_argument('--anchor', action='append', default=[],
                    help='tile_id=LINE,FOVINDEX for a correspondence you verified by '
                         'hand. Repeatable; 2+ fit scale as well as offset.')
    ap.add_argument('--tissue_thresh', type=int, default=None,
                    help='Defaults to whatever source.txt records for this tile '
                         'set, so the box is drawn round the tissue that was '
                         'actually tiled rather than round a differently '
                         'thresholded version of it. Tile sets cut before that '
                         'field existed fall back to 215 with a warning.')
    ap.add_argument('--extra', type=int, default=0,
                    help='After the per-structure picks, add up to this many more '
                         'regions as a second tier, drawn green. Taken round robin '
                         'across HGSOC_KEYS by z, so the extra set spreads over the '
                         'structures instead of filling up with whichever one the '
                         'slide happens to score highest on.')
    ap.add_argument('--extra_z', type=float, default=2.0,
                    help='A second-tier region has to reach this z on at least one '
                         'HGSOC structure. The cap is what should bind, not this: if '
                         'fewer regions clear it than --extra asks for, fewer are '
                         'returned and the shortfall is printed, because padding the '
                         'list with unremarkable tissue is worse than a short list.')
    ap.add_argument('--roi', default=None,
                    help='x0,y0,x1,y1 in full-resolution pixels. Only tiles whose '
                         'centre falls inside are eligible, and the z scores are '
                         'then taken over that subset -- the question becomes "strongest '
                         'example in this piece of tissue", which is the right one when '
                         'the other modality only covers part of the slide. 240828_pt1 '
                         'has TPAF for one fragment only: 7360,1152,13504,6880.')
    ap.add_argument('--rotate', type=float, default=0.0,
                    help='Degrees clockwise to turn the OUTPUT by, so the map and the '
                         'crops match another modality. Display only: the tiling, the '
                         'scores, the pick numbers and every coordinate written to '
                         'selection.csv stay in the original frame. 240729 needs 135 '
                         'to sit the way its TPAF montage does.')
    ap.add_argument('--crop_px', type=int, default=0,
                    help='Also save each picked tile at this size.')
    ap.add_argument('--probes', action='store_true',
                    help='Report the PROBES candidate-caption columns instead of the '
                         'eighteen real ones. Needs a CSV scored with --probes.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # The crop filenames carry a number that is assigned per run -- it depends on which
    # tiles this run picked and in what order -- while selection.csv and
    # selection_map.png are single files that just get overwritten. Rerun into a
    # directory that already holds crops and the two disagree: the map says 4 and the
    # leftover pick04_*.png from the previous run is a different tile. Nothing errors,
    # and the only way to notice is to look at a crop and not recognise it.
    stale = [f for f in os.listdir(args.out) if f.startswith('pick') and f.endswith('.png')]
    for f in stale:
        os.remove(os.path.join(args.out, f))
    if stale:
        print(f'cleared {len(stale)} crop(s) from a previous run', flush=True)
    src = read_source(args.tiles)
    # A tile set cut at a different threshold than the one used here is boxed around
    # tissue it does not contain. 240729 is the case that matters -- it was tiled at
    # 240 because its staining is pale, and reading it at the old 215 default shrank
    # the box by 0.6 x 0.8 mm and pushed picks near the edge outside it, which is what
    # a frac of 1.044 in an earlier selection.csv was.
    if args.tissue_thresh is None:
        if 'tissue_thresh' in src:
            args.tissue_thresh = int(src['tissue_thresh'])
            print(f'tissue_thresh {args.tissue_thresh} (from source.txt)', flush=True)
        else:
            args.tissue_thresh = 215
            print('WARNING: source.txt records no tissue_thresh; this tile set '
                  'predates the field, so 215 is a guess.', flush=True)
            print('         If it was cut at another value then the tissue box, and '
                  'every frac', flush=True)
            print('         and TPAF estimate taken from it, is wrong. Recover the '
                  'real value', flush=True)
            print('         with path_thresh_recover.py.', flush=True)
    ds = int(src['thumb_ds']); tile_px = int(src['tile_px']); um = float(src['um_per_px'])
    thumb = cv2.imread(os.path.join(args.tiles, 'thumbnail.png'))
    idx = {r['tile_id']: r for r in
           csv.DictReader(open(os.path.join(args.tiles, 'index.csv'), encoding='utf-8'))}
    rows = list(csv.DictReader(open(args.scores, encoding='utf-8')))
    rows = [r for r in rows if r['tile_id'] in idx]
    if not rows:
        raise SystemExit('no scored tile is in index.csv -- wrong --tiles directory?')

    # Restricting before the z scores are taken, not after, is the point of --roi. A
    # slide-wide z answers "how unusual is this tile for the slide"; inside a fragment
    # that the other modality actually covers, the useful question is "how unusual is
    # it for the fragment", and the two disagree whenever the fragment is not typical
    # of the slide. Filtering here makes every number downstream refer to the subset.
    roi = None
    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(','))
            assert len(roi) == 4
        except Exception:
            raise SystemExit(f'bad --roi {args.roi!r}; expected x0,y0,x1,y1 in pixels')
        rx0, ry0, rx1, ry1 = roi
        n_all = len(rows)
        rows = [r for r in rows
                if rx0 <= int(idx[r['tile_id']]['x']) + tile_px / 2 <= rx1
                and ry0 <= int(idx[r['tile_id']]['y']) + tile_px / 2 <= ry1]
        if not rows:
            raise SystemExit(f'--roi {args.roi} contains no scored tile')
        print(f'roi {rx0},{ry0},{rx1},{ry1} px '
              f'({(rx1-rx0)*um/1000:.2f} x {(ry1-ry0)*um/1000:.2f} mm): '
              f'{len(rows)}/{n_all} tiles; z is over these {len(rows)}', flush=True)

    x0, y0, x1, y1 = tissue_bbox(thumb, args.tissue_thresh)
    bw, bh = x1 - x0, y1 - y0
    print(f'tissue bbox on thumbnail: x {x0}..{x1}, y {y0}..{y1}  '
          f'({bw*ds*um/1000:.1f} x {bh*ds*um/1000:.1f} mm)', flush=True)

    def frac(tid):
        r = idx[tid]
        cy = (int(r['y']) + tile_px / 2) / ds
        cx = (int(r['x']) + tile_px / 2) / ds
        return (cy - y0) / bh, (cx - x0) / bw

    # anchors are given as verified (tile_id -> line, fov index); turn them into
    # fraction -> normalised grid coordinate pairs
    ay, ax = [], []
    for tid, line, fidx in map(parse_anchor, args.anchor):
        if tid not in idx:
            raise SystemExit(f'--anchor tile_id {tid} not in index.csv')
        fy, fx = frac(tid)
        lines = sorted(TPAF_LINE_FOVS)
        if line not in TPAF_LINE_FOVS:
            raise SystemExit(f'--anchor line {line} outside 1..23')
        ay.append((fy, lines.index(line) / (len(lines) - 1)))
        n = TPAF_LINE_FOVS[line]
        g = fidx if n == 1 else fidx / (n - 1)
        ax.append((fx, (1 - g) if line in TPAF_REV_LINES else g))
    fit_y, fit_x = fit_axis(ay), fit_axis(ax)
    if args.anchor:
        print(f'{len(args.anchor)} anchor(s): y = {fit_y[0]:.3f}f + {fit_y[1]:+.3f}, '
              f'x = {fit_x[0]:.3f}f + {fit_x[1]:+.3f}', flush=True)
    else:
        print('no anchors -- TPAF estimate assumes the grid spans the same tissue bbox',
              flush=True)

    wanted = PROBE_KEYS if args.probes else (ECM_KEYS if args.only_ecm else KEYS)
    keys = [k for k in wanted if k in rows[0]]
    if args.probes and not keys:
        raise SystemExit('--probes but the CSV has no probe columns; rescore with '
                         'path_colab_score.py --probes')
    picks, extra_tids = [], set()
    if args.tile_ids:
        for t in [s.strip() for s in args.tile_ids.split(',') if s.strip()]:
            if t not in idx:
                raise SystemExit(f'--tile_ids {t} not in index.csv')
            picks.append((t, '(manual)', float('nan')))
    else:
        sep2 = (args.min_sep * tile_px) ** 2
        n_ecm = args.top
        n_other = args.top if args.top_nonecm is None else args.top_nonecm
        for k in keys:
            want = n_ecm if k in ECM_KEYS else n_other
            v = np.array([float(r[k]) for r in rows])
            z = (v - v.mean()) / (v.std() + 1e-9)
            taken = []
            for i in np.argsort(-v):
                if len(taken) >= want:
                    break
                rr = idx[rows[i]['tile_id']]
                cy = int(rr['y']) + tile_px / 2
                cx = int(rr['x']) + tile_px / 2
                if any((cy - py) ** 2 + (cx - px) ** 2 < sep2 for py, px in taken):
                    continue
                taken.append((cy, cx))
                picks.append((rows[i]['tile_id'], k, float(z[i])))
            # Say so rather than quietly returning fewer regions than asked for.
            if len(taken) < want:
                print(f'  {k}: only {len(taken)}/{want} regions at least '
                      f'{args.min_sep:g} tiles apart', flush=True)

        if args.extra:
            # A second tier, for filling a figure pool rather than answering "where is
            # the most X on this slide". Two things shape it.
            #
            # Round robin across the structures, not one ranked list: taking the next
            # 25 by raw z fills up with whichever structure the slide scores highest
            # on, and a pool of 25 papillary tiles is not more useful than the three
            # already picked. Cycling gives roughly two of each.
            #
            # A floor on z rather than a quota alone. The instruction is to add
            # regions worth adding, so if fewer clear the bar than asked for, fewer
            # come back and the shortfall is printed. Padding with unremarkable tissue
            # would make the tier look complete while making it worthless.
            centres = {}
            for tid, _, _ in picks:
                rr = idx[tid]
                centres[tid] = (int(rr['y']) + tile_px / 2, int(rr['x']) + tile_px / 2)
            pts = list(centres.values())
            zs = {}
            for k in HGSOC_KEYS:
                if k not in rows[0]:
                    continue
                v = np.array([float(r[k]) for r in rows])
                zs[k] = (v - v.mean()) / (v.std() + 1e-9)
            queues = {k: list(np.argsort(-z)) for k, z in zs.items()}
            order = [k for k in HGSOC_KEYS if k in queues]
            n_extra = 0
            first_extra = len(picks)
            while n_extra < args.extra and any(queues[k] for k in order):
                progressed = False
                for k in order:
                    if n_extra >= args.extra:
                        break
                    while queues[k]:
                        i = queues[k].pop(0)
                        if zs[k][i] < args.extra_z:
                            queues[k] = []
                            break
                        tid = rows[i]['tile_id']
                        if tid in centres:
                            continue
                        rr = idx[tid]
                        cy = int(rr['y']) + tile_px / 2
                        cx = int(rr['x']) + tile_px / 2
                        if any((cy - py) ** 2 + (cx - px) ** 2 < sep2 for py, px in pts):
                            continue
                        centres[tid] = (cy, cx)
                        pts.append((cy, cx))
                        picks.append((tid, k, float(zs[k][i])))
                        n_extra += 1
                        progressed = True
                        break
                if not progressed:
                    break
            extra_tids = {t for t, _, _ in picks[first_extra:]}
            print(f'extra tier: {n_extra}/{args.extra} regions at z >= {args.extra_z:g} '
                  f'on an HGSOC structure', flush=True)
            if n_extra < args.extra:
                print(f'  {args.extra - n_extra} short -- nothing else clears the bar; '
                      'lower --extra_z only if you have looked and disagree', flush=True)

    # Rotation is applied to what is drawn and saved, never to what is measured, so
    # everything downstream of selection.csv keeps working in the original frame.
    src_img = None
    if args.rotate and args.crop_px:
        ipath = src.get('image', '')
        if os.path.exists(ipath):
            src_img = tiff.memmap(ipath) if ipath.lower().endswith(
                ('.tif', '.tiff')) else cv2.imread(ipath)[..., ::-1]
        else:
            print('WARNING: source image not found, so the crops are rotations of the',
                  flush=True)
            print('         saved tiles and have empty corners:', ipath, flush=True)

    seen, out_rows, marks = {}, [], []
    if args.rotate:
        Mrot, rw, rh = rot_cw(args.rotate, thumb.shape[1], thumb.shape[0])
        canvas = cv2.warpAffine(thumb, Mrot, (rw, rh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        bb = apply_M(Mrot, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        cv2.polylines(canvas, [bb.round().astype(np.int32)], True, (0, 160, 0), 1)
        print(f'output rotated {args.rotate:g} deg clockwise '
              f'({thumb.shape[1]}x{thumb.shape[0]} -> {rw}x{rh}); '
              'coordinates in selection.csv are unrotated', flush=True)
    else:
        Mrot = None
        canvas = thumb.copy()
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 160, 0), 1)
    plain = canvas.copy()
    for tid, k, z in picks:
        r = idx[tid]
        fy, fx = frac(tid)
        line, fidx, n, rev, alt = estimate_tpaf(fy, fx, fit_y, fit_x)
        if tid not in seen:
            seen[tid] = len(seen) + 1
            ty, tx = int(r['y']) // ds, int(r['x']) // ds
            s = max(2, tile_px // ds)
            q = [(tx, ty), (tx + s, ty), (tx + s, ty + s), (tx, ty + s)]
            # once the map is turned the square is no longer axis-aligned, so the four
            # transformed corners are drawn rather than a bounding box, which would
            # overstate the region by up to 41 percent
            marks.append((seen[tid], apply_M(Mrot, q) if Mrot is not None else q,
                          (0, 170, 0) if tid in extra_tids else (0, 0, 255)))
        out_rows.append(dict(
            n=seen[tid], tier=2 if tid in extra_tids else 1,
            structure=k, zh=ZH.get(k, ''), ecm=int(k in ECM_KEYS),
            tile_id=tid, z=('' if z != z else round(z, 2)),
            y_px=r['y'], x_px=r['x'],
            y_mm=round(int(r['y']) * um / 1000, 2), x_mm=round(int(r['x']) * um / 1000, 2),
            frac_y=round(fy, 3), frac_x=round(fx, 3),
            tpaf_line_est=f'Line-{line:02d}', tpaf_fov_est=f'{fidx:04d}',
            tpaf_line_alt=' '.join(alt), tpaf_rev=int(rev), tpaf_line_n_fov=n))
        if args.crop_px:
            dst = os.path.join(args.out, f'pick{seen[tid]:02d}_{k}_{tid}.png')
            if src_img is not None:
                cv2.imwrite(dst, rotated_crop(
                    src_img, int(r['y']) + tile_px / 2, int(r['x']) + tile_px / 2,
                    tile_px, args.rotate, args.crop_px))
            else:
                im = cv2.imread(tile_path(args.tiles, tid))
                if im is not None:
                    if args.rotate:
                        M2, w2, h2 = rot_cw(args.rotate, im.shape[1], im.shape[0])
                        im = cv2.warpAffine(im, M2, (w2, h2), flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(255, 255, 255))
                    cv2.imwrite(dst, cv2.resize(im, (args.crop_px, args.crop_px)))

    if roi is not None:
        rq = [(roi[0] / ds, roi[1] / ds), (roi[2] / ds, roi[1] / ds),
              (roi[2] / ds, roi[3] / ds), (roi[0] / ds, roi[3] / ds)]
        if Mrot is not None:
            rq = apply_M(Mrot, rq)
        rq = np.asarray(rq)
        cv2.polylines(canvas, [rq.round().astype(np.int32)], True, (200, 80, 0), 2)
        # A second map cropped to the ROI. On a whole slide a 3 mm fragment is a
        # thumbnail-sized corner, so the crop is enlarged -- and the marks are drawn
        # after the enlargement, at their normal size, rather than being magnified
        # along with the pixels.
        m = 12
        cx0, cy0 = max(0, int(rq[:, 0].min()) - m), max(0, int(rq[:, 1].min()) - m)
        cx1 = min(canvas.shape[1], int(rq[:, 0].max()) + m)
        cy1 = min(canvas.shape[0], int(rq[:, 1].max()) + m)
        crop = plain[cy0:cy1, cx0:cx1]
        if crop.size:
            k = max(1, int(np.ceil(1400 / max(crop.shape[:2]))))
            big = cv2.resize(crop, (crop.shape[1] * k, crop.shape[0] * k),
                             interpolation=cv2.INTER_LANCZOS4)
            cv2.polylines(big, [((rq - (cx0, cy0)) * k).round().astype(np.int32)],
                          True, (200, 80, 0), 2)
            draw_marks(big, marks, k=k, ox=cx0, oy=cy0)
            cv2.imwrite(os.path.join(args.out, 'selection_map_roi.png'), big)
    draw_marks(canvas, marks)
    cv2.imwrite(os.path.join(args.out, 'selection_map.png'), canvas)
    # Leave a marker beside the rotated outputs. A rotated crop is not otherwise
    # distinguishable from an unrotated one, and these sit next to five other
    # samples that are not turned; anything reading this directory later needs to
    # know, and so does anyone comparing two samples by eye.
    # Leave a marker beside the outputs describing how they were made. A rotated crop
    # is not distinguishable from an unrotated one, and a restricted pick set is not
    # distinguishable from a slide-wide one; both sit next to samples that had neither.
    # Anything reading this directory later needs to know, and so does anyone comparing
    # two samples by eye.
    vpath = os.path.join(args.out, 'view.txt')
    for old in ('rotation.txt',):
        if os.path.exists(os.path.join(args.out, old)):
            os.remove(os.path.join(args.out, old))
    if args.rotate or roi is not None:
        nl = chr(10)
        with io.open(vpath, 'w', encoding='utf-8') as fh:
            if args.rotate:
                fh.write(f'rotate_cw={args.rotate:g}' + nl)
            if roi is not None:
                fh.write('roi=' + ','.join(str(v) for v in roi) + nl)
                fh.write(f'roi_tiles={len(rows)}' + nl)
            fh.write('applies_to=selection_map.png, pick*.png' + nl)
            fh.write('applies_to_coordinates=no' + nl)
    elif os.path.exists(vpath):
        os.remove(vpath)
    # utf-8-sig so Excel on a Chinese Windows does not read the zh column as GBK
    with open(os.path.join(args.out, 'selection.csv'), 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    print(f'\n{len(seen)} distinct regions from {len(picks)} picks')
    for r in out_rows[:24]:
        print(f"  #{r['n']:>2} {r['structure']:<18}{r['tile_id']:<16}"
              # labelled, because the tile_id reads y-then-x and this line reads
              # x-then-y, and an unlabelled pair invites reading one as the other
              f"x{r['x_mm']:>6.2f} y{r['y_mm']:>6.2f} mm   "
              f"~{r['tpaf_line_est']}_{r['tpaf_fov_est']}"
              f"{'  [rev]' if r['tpaf_rev'] else ''}")
    if len(out_rows) > 24:
        print(f'  ... {len(out_rows)-24} more in selection.csv')
    print(f'\n-> {args.out}: selection_map.png, selection.csv')
    print('tpaf_line_est / tpaf_fov_est are ESTIMATES. Verify one, then pass it back '
          'as --anchor tile_id=LINE,FOVINDEX to refit.')


if __name__ == '__main__':
    main()
