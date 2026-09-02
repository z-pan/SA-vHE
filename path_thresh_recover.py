#!/usr/bin/env python3
"""Recover the --tissue_thresh a tile set was cut with, and write it into source.txt.

Tile sets cut before source.txt carried the field do not say which threshold produced
them, and path_locate.py / path_report.py then fall back to 215 and box a different
piece of tissue than the one that was tiled. Nothing errors.

The value is recoverable rather than guessable. path_tiles.py makes the keep/drop
decision on the saved thumbnail, not on the source image, and index.csv stores the
resulting tissue fraction per tile at three decimals -- so replaying the test at a
candidate threshold reproduces the tile set exactly when the candidate is right, and
visibly fails when it is not. That is a measurement of what happened, not an inference
about it, and it needs neither the slide nor a GPU.

Reported as a match only when the whole tile_id set and every stored fraction agree.
Counts alone would not do: two thresholds can keep the same number of tiles while
disagreeing about which ones.

    python path_thresh_recover.py results/path_screen/survey/*        # report
    python path_thresh_recover.py results/path_screen/survey/* --write # and record it
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2


def read_source(d):
    kv = {}
    for line in open(os.path.join(d, 'source.txt'), encoding='utf-8'):
        k, _, v = line.strip().partition('=')
        if k:
            kv[k] = v
    return kv


def replay(d, thresh, min_tissue):
    """path_tiles.py's tiling loop, re-run against the saved thumbnail."""
    src = read_source(d)
    ds = int(src['thumb_ds'])
    tile = int(src['tile_px'])
    step = int(src['step_px'])
    W, H = int(src['W']), int(src['H'])
    tg = cv2.cvtColor(cv2.imread(os.path.join(d, 'thumbnail.png')), cv2.COLOR_BGR2GRAY)
    th = max(1, tile // ds)
    out = {}
    for y in range(0, H - tile + 1, step):
        for x in range(0, W - tile + 1, step):
            patch = tg[y // ds:y // ds + th, x // ds:x // ds + th]
            frac = float((patch < thresh).mean())
            if frac >= min_tissue:
                out[f'y{y}_x{x}'] = round(frac, 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='Tile-set directories made by path_tiles.py.')
    ap.add_argument('--min_tissue', type=float, default=0.25,
                    help="path_tiles.py's default; a set cut with another value will "
                         'not reproduce at any threshold, which is itself informative.')
    ap.add_argument('--range', default='150,255,1', help='lo,hi,step to search.')
    ap.add_argument('--write', action='store_true',
                    help='Append the recovered value to source.txt. Only ever writes '
                         'an exactly reproducing value, and never overwrites one that '
                         'is already recorded.')
    args = ap.parse_args()

    lo, hi, stp = (int(v) for v in args.range.split(','))
    for d in args.dirs:
        if not os.path.exists(os.path.join(d, 'source.txt')):
            continue
        name = os.path.basename(os.path.normpath(d))
        src = read_source(d)
        have = {r['tile_id']: float(r['tissue']) for r in
                csv.DictReader(open(os.path.join(d, 'index.csv'), encoding='utf-8'))}
        if 'tissue_thresh' in src:
            print(f'{name:<14} already records tissue_thresh={src["tissue_thresh"]}')
            continue
        hits = [t for t in range(lo, hi + 1, stp) if replay(d, t, args.min_tissue) == have]
        if not hits:
            print(f'{name:<14} NO threshold in {lo}..{hi} reproduces the {len(have)} '
                  'tiles -- --min_tissue or the thumbnail must differ too')
            continue
        # Thresholds between two grey levels present in the image act identically, so a
        # run of consecutive hits is expected; any of them replays the set exactly.
        lo_h, hi_h = hits[0], hits[-1]
        val = lo_h if len(hits) == 1 else (lo_h + hi_h) // 2
        span = '' if len(hits) == 1 else f'  (equivalent over {lo_h}..{hi_h})'
        print(f'{name:<14} tissue_thresh = {val}   {len(have)} tiles reproduced '
              f'exactly{span}')
        if args.write:
            with open(os.path.join(d, 'source.txt'), 'a', encoding='utf-8') as fh:
                fh.write(f'tissue_thresh={val}\n')
                fh.write(f'min_tissue={args.min_tissue}\n')
                fh.write('tissue_thresh_source=recovered by path_thresh_recover.py '
                         '(exact replay of index.csv)\n')
            print(f'{"":<14} -> written to {d}/source.txt')


if __name__ == '__main__':
    main()
