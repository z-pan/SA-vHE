#!/usr/bin/env python3
"""Stitch each TMA core's 3x3 TPAF mosaic from its nine tiles.

Layout (verified by locating every tile inside 251218_selected_TMA/*_combine.tif at
ncc 0.88-0.98, on three cores):

    T009 T008 T007          serpentine: bottom row right-to-left (T001..T003),
    T004 T005 T006          middle row left-to-right (T004..T006),
    T003 T002 T001          top row right-to-left (T007..T009)

Nominal step is ~815 px for a 1024 px tile, i.e. ~20% overlap. Stage repeatability
leaves +-30 px of jitter and a slight shear (about -16 px of y per +815 px of x, ~1.1
degrees), so tiles are placed by pairwise correlation rather than on a fixed grid.

The frames folders also hold the operator's own by-products -- 'enhanced/' subdirs,
Fiji 'img_t1_z1_c1' outputs, _T00N.png copies. Only files matching _T\\d{3}\\.tif are
read, so those are ignored.
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

TILE_RE = re.compile(r'_T(\d{3})\.tif$', re.I)
# tile number -> (row, col), row 0 at top
LAYOUT = {9: (0, 0), 8: (0, 1), 7: (0, 2),
          4: (1, 0), 5: (1, 1), 6: (1, 2),
          3: (2, 0), 2: (2, 1), 1: (2, 2)}
NOMINAL = 815


# Two cores were acquired in two goes and are numbered accordingly. Resolved by
# overlap agreement, not by assumption: the wrong assignment scores 0.04-0.10 against
# 0.57 for the right one, where an ordinary core scores 0.66 (15C).
#   6D  : "6D_0001.tif" is the missing ninth tile, top-left  (T009 slot); T001-T008 stand.
#   10H : "10H_0001.tif" is the FIRST tile, bottom-right (T001 slot), and the eight
#         "10H_0002_T00N" from the restart are really T00(N+1).
# Filenames on disk are left alone; this only changes how they are read.
IRREGULAR = {
    '6D': [(re.compile(r'6D_0001\.tif$', re.I), lambda n: 9)],
    '10H': [(re.compile(r'10H_0001\.tif$', re.I), lambda n: 1),
            (re.compile(r'10H_0002_T(\d{3})\.tif$', re.I), lambda n: n + 1)],
}


def tile_number(fname, well=None):
    """Grid tile number for a file, or None if it is not a tile."""
    for pat, fn in IRREGULAR.get(well or '', []):
        m = pat.search(fname)
        if m:
            return fn(int(m.group(1)) if m.groups() else 0)
    m = TILE_RE.search(fname)
    return int(m.group(1)) if m else None


def load_tiles(frames_dir, well=None):
    if well is None:
        m = re.search(r'(?<![0-9A-Za-z])(\d{1,2}[A-H])\.tif\.frames$', str(frames_dir))
        well = m.group(1) if m else None
    out = {}
    for f in sorted(os.listdir(frames_dir)):
        if not f.lower().endswith(('.tif', '.tiff')):
            continue
        k = tile_number(f, well)
        if k is None:
            continue
        a = tiff.imread(os.path.join(frames_dir, f)).astype(np.float32)
        out[k] = a[..., :2].mean(2) if a.ndim == 3 else a
    return out


def pair_shift(a, b, axis, nominal=NOMINAL, search=90):
    """Offset of b relative to a. axis 0 = b below a, axis 1 = b right of a.

    Correlates a's trailing overlap strip against b's leading strip, so the search is
    a small 2-D window around the nominal step instead of the whole tile.
    """
    n = a.shape[axis] - nominal            # overlap thickness, ~209 px
    if axis == 1:
        strip_a = a[:, -n:]
        strip_b = b[:, :n + 2 * search] if b.shape[1] >= n + 2 * search else b
    else:
        strip_a = a[-n:, :]
        strip_b = b[:n + 2 * search, :] if b.shape[0] >= n + 2 * search else b
    # trim the template so it fits inside the target with room to slide
    if axis == 1:
        tpl = strip_a[search:-search or None, :]
    else:
        tpl = strip_a[:, search:-search or None]
    if tpl.shape[0] >= strip_b.shape[0] or tpl.shape[1] >= strip_b.shape[1]:
        return (nominal, 0) if axis == 1 else (0, nominal), 0.0
    r = cv2.matchTemplate(strip_b.astype(np.float32), tpl.astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
    _, score, _, mx = cv2.minMaxLoc(r)
    py, px = mx[1], mx[0]
    if axis == 1:
        dy = search - py
        dx = nominal - px
        return (dy, dx), float(score)
    dy = nominal - py
    dx = search - px
    return (dy, dx), float(score)


def solve_positions(tiles, min_score=0.15):
    """Least-squares tile origins from pairwise shifts, anchored at tile (0,0)."""
    idx = {LAYOUT[k]: k for k in tiles if k in LAYOUT}
    cells = sorted(idx)
    order = {c: i for i, c in enumerate(cells)}
    rows_A, rows_b, weights = [], [], []
    for (r, c) in cells:
        for axis, (r2, c2) in ((1, (r, c + 1)), (0, (r + 1, c))):
            if (r2, c2) not in order:
                continue
            (dy, dx), s = pair_shift(tiles[idx[(r, c)]], tiles[idx[(r2, c2)]], axis)
            if s < min_score:                       # fall back to the nominal grid
                dy, dx = (0, NOMINAL) if axis == 1 else (NOMINAL, 0)
                s = min_score
            for k, d in ((0, dy), (1, dx)):
                row = np.zeros(len(cells))
                row[order[(r2, c2)]] = 1
                row[order[(r, c)]] = -1
                rows_A.append(np.concatenate([row, [k]]))
                rows_b.append(d)
                weights.append(s)
    pos = {}
    for k in (0, 1):
        sel = [i for i, r in enumerate(rows_A) if r[-1] == k]
        A = np.array([rows_A[i][:-1] for i in sel])
        bb = np.array([rows_b[i] for i in sel])
        w = np.sqrt(np.array([weights[i] for i in sel]))
        A = np.vstack([A * w[:, None], np.eye(1, len(cells))])   # anchor cell 0 at 0
        bb = np.concatenate([bb * w, [0.0]])
        sol = np.linalg.lstsq(A, bb, rcond=None)[0]
        for c, i in order.items():
            pos.setdefault(c, [0, 0])[k] = float(sol[i])
    return {idx[c]: (int(round(v[0])), int(round(v[1]))) for c, v in pos.items()}


def build_mosaic(tiles, pos):
    ys = [p[0] for p in pos.values()]
    xs = [p[1] for p in pos.values()]
    oy, ox = min(ys), min(xs)
    h = max(p[0] - oy + tiles[k].shape[0] for k, p in pos.items())
    w = max(p[1] - ox + tiles[k].shape[1] for k, p in pos.items())
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float64)
    for k, (y, x) in pos.items():
        t = tiles[k]
        y -= oy; x -= ox
        acc[y:y + t.shape[0], x:x + t.shape[1]] += t
        cnt[y:y + t.shape[0], x:x + t.shape[1]] += 1
    return np.divide(acc, np.maximum(cnt, 1)).astype(np.float32), (h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tpaf_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--verify_dir', default=None,
                    help='251218_selected_TMA, to score the stitch against *_combine.tif.')
    ap.add_argument('--wells', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = {}
    for d in sorted(os.listdir(args.tpaf_dir)):
        m = re.search(r'(?<![0-9A-Za-z])(\d{1,2}[A-H])\.tif\.frames$', d)
        if m and not d.lower().startswith('not used'):
            dirs[m.group(1)] = os.path.join(args.tpaf_dir, d)
    if args.wells:
        want = [w.strip() for w in args.wells.split(',')]
        dirs = {w: dirs[w] for w in want if w in dirs}
    print(f'{len(dirs)} cores', flush=True)

    rows = []
    for i, (well, d) in enumerate(sorted(dirs.items(), key=lambda kv: (int(kv[0][:-1]), kv[0][-1]))):
        if args.limit and i >= args.limit:
            break
        tiles = load_tiles(d)
        if len(tiles) != 9:
            print(f'  [warn] {well}: {len(tiles)} tiles, skipping', flush=True)
            rows.append(dict(well=well, tiles=len(tiles), h=0, w=0, verify_ncc=''))
            continue
        pos = solve_positions(tiles)
        mos, (h, w) = build_mosaic(tiles, pos)
        tiff.imwrite(os.path.join(args.out_dir, f'{well}_mosaic.tif'),
                     np.clip(mos, 0, 65535).astype(np.uint16))

        vn = ''
        if args.verify_dir:
            p = os.path.join(args.verify_dir, f'{well}_combine.tif')
            if os.path.exists(p):
                ref = tiff.imread(p).astype(np.float32)
                ref = ref[:2].mean(0) if ref.ndim == 3 else ref
                a = cv2.resize(mos, (ref.shape[1], ref.shape[0]))
                a = (a - a.mean()) / (a.std() + 1e-9)
                b = (ref - ref.mean()) / (ref.std() + 1e-9)
                vn = round(float((a * b).mean()), 4)
        rows.append(dict(well=well, tiles=len(tiles), h=h, w=w, verify_ncc=vn))
        print(f'  {well:>4} 9 tiles -> {w}x{h}' + (f'  vs combine r={vn}' if vn != '' else ''),
              flush=True)

    with open(os.path.join(args.out_dir, 'stitch.csv'), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    ok = [r for r in rows if r['tiles'] == 9]
    print(f'\n{len(ok)}/{len(rows)} stitched -> {args.out_dir}')
    vs = [r['verify_ncc'] for r in ok if r['verify_ncc'] != '']
    if vs:
        print(f'verify r: min {min(vs):.3f} median {sorted(vs)[len(vs)//2]:.3f} max {max(vs):.3f}')


if __name__ == '__main__':
    main()
