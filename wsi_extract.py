#!/usr/bin/env python3
"""Pull a region out of a whole-slide TIFF at native resolution, and find where a
previously exported crop came from.

Why not ImageJ
--------------
Nothing is wrong with the ImageJ crops in this project -- the one checked
(WSI_240703HOC240717-3_2_TPAF_match.tif) is pixel-identical to its source, ncc
1.000000, max difference 0 -- but it is rotated 180 degrees relative to the slide,
which silently breaks any later registration, and ImageJ has to materialise the whole
image to do it. These slide TIFFs are flat single-page uncompressed files, so a memmap
plus a slice reads only the requested rows: no decode, no full-image allocation.

The pyramid worry is real for other formats. .ndpi, .svs and pyramidal OME-TIFF hold
several resolutions in one file, and a reader that opens "the image" may hand back a
reduced level. This always uses level 0, and says so when a file has more than one.

Modes
-----
extract   write a region, optionally rotated/flipped, optionally downsampled
locate    find where a given image sits inside the slide, including 90-degree
          rotations and flips -- which is how the 180-degree turn above was found

Examples::

    python wsi_extract.py extract --wsi slide.tif --y 6208 --x 1960 \\
        --h 16008 --w 13920 --rotate 180 --out region.tif

    python wsi_extract.py locate --wsi slide.tif --ref crop.tif
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import tifffile as tiff


def open_level0(path):
    """(array-like, note) for the highest-resolution level, memmapped when possible."""
    with tiff.TiffFile(path) as tf:
        s = tf.series[0]
        levels = len(s.levels) if hasattr(s, 'levels') else 1
        shape = s.levels[0].shape if levels > 1 else s.shape
        tiled = tf.pages[0].is_tiled
        comp = tf.pages[0].compression
    note = f'{shape} levels={levels} tiled={tiled} compression={comp}'
    if levels > 1:
        note += '  [pyramid: using level 0]'
    try:
        arr = tiff.memmap(path)                     # flat, uncompressed: no decode
        if arr.shape[:2] != tuple(shape[:2]):
            raise ValueError('memmap is not level 0')
        return arr, note + '  (memmap)'
    except Exception:
        return tiff.imread(path, level=0) if levels > 1 else tiff.imread(path), \
            note + '  (decoded into RAM)'


def orient(a, rotate=0, flip=None):
    if rotate:
        a = cv2.rotate(a, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate % 360])
    if flip == 'h':
        a = a[:, ::-1]
    elif flip == 'v':
        a = a[::-1]
    return np.ascontiguousarray(a)


def gray_thumb(arr, ds):
    a = np.asarray(arr[::ds, ::ds])
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) \
        if a.ndim == 3 else a.astype(np.float32)


def cmd_extract(args):
    arr, note = open_level0(args.wsi)
    H, W = arr.shape[:2]
    print(f'{os.path.basename(args.wsi)}: {note}')
    y = args.y or 0
    x = args.x or 0
    h = args.h or (H - y)
    w = args.w or (W - x)
    if y < 0 or x < 0 or y + h > H or x + w > W:
        sys.exit(f'region y{y}..{y+h} x{x}..{x+w} is outside {H}x{W}')
    print(f'region y {y}..{y+h}  x {x}..{x+w}  ({w}x{h})', flush=True)

    out = np.asarray(arr[y:y + h, x:x + w])
    if args.rotate or args.flip:
        out = orient(out, args.rotate, args.flip)
        print(f'oriented: rotate {args.rotate} flip {args.flip}')
    if args.downsample > 1:
        out = cv2.resize(out, None, fx=1 / args.downsample, fy=1 / args.downsample,
                         interpolation=cv2.INTER_AREA)
        print(f'downsampled by {args.downsample} -> {out.shape[1]}x{out.shape[0]}'
              '  [resolution IS reduced, by request]')
    tiff.imwrite(args.out, out, compression=args.compression or None)
    print(f'-> {args.out}  {out.shape[1]}x{out.shape[0]}  '
          f'{os.path.getsize(args.out)/2**20:.0f} MB')


def cmd_locate(args):
    arr, note = open_level0(args.wsi)
    print(f'{os.path.basename(args.wsi)}: {note}')
    ref, rnote = open_level0(args.ref)
    print(f'{os.path.basename(args.ref)}: {rnote}')

    ds = args.ds
    big = gray_thumb(arr, ds)
    small0 = gray_thumb(ref, ds)
    variants = {0: small0,
                90: cv2.rotate(small0, cv2.ROTATE_90_CLOCKWISE),
                180: cv2.rotate(small0, cv2.ROTATE_180),
                270: cv2.rotate(small0, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    if args.try_flips:
        variants['flipH'] = np.ascontiguousarray(small0[:, ::-1])
        variants['flipV'] = np.ascontiguousarray(small0[::-1])

    best = (-1.0, None, 1.0, 0, 0)
    print(f'\n{"orient":<8}{"scale":>7}{"ncc":>9}   offset (full-res)')
    for name, v in variants.items():
        for s in args.scales:
            t = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else v
            if t.shape[0] >= big.shape[0] or t.shape[1] >= big.shape[1]:
                continue
            r = cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED)
            _, sc, _, mx = cv2.minMaxLoc(r)
            if sc >= args.report_above:
                print(f'{str(name):<8}{s:>7.2f}{sc:>9.4f}   (y={mx[1]*ds}, x={mx[0]*ds})')
            if sc > best[0]:
                best = (sc, name, s, mx[1] * ds, mx[0] * ds)
    sc, name, s, oy, ox = best
    print(f'\nbest: ncc {sc:.4f}  orient {name}  scale {s:.2f}  offset (y={oy}, x={ox})')
    if sc < 0.5:
        print('[warn] weak match -- the reference may not come from this slide')
        return

    # refine on a centre patch at full resolution, which also proves whether the
    # reference is a lossless crop or was resampled on the way out
    rh, rw = ref.shape[:2]
    P = min(512, rh // 2, rw // 2)
    patch = np.asarray(ref[rh // 2 - P // 2:rh // 2 + P // 2, rw // 2 - P // 2:rw // 2 + P // 2])
    inv = {0: 0, 90: 270, 180: 180, 270: 90}.get(name, 0)
    patch_slide = orient(patch, inv) if isinstance(name, int) else patch
    if name == 180:
        ey = oy + (rh - (rh // 2 + P // 2)); ex = ox + (rw - (rw // 2 + P // 2))
    elif name == 0:
        ey = oy + rh // 2 - P // 2; ex = ox + rw // 2 - P // 2
    else:
        print('(full-res check implemented for orient 0 and 180 only)')
        return
    M = args.refine
    sub = np.asarray(arr[max(0, ey - M):ey + P + M, max(0, ex - M):ex + P + M])
    r = cv2.matchTemplate(cv2.cvtColor(sub[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32),
                          cv2.cvtColor(patch_slide[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
    _, sc2, _, mx = cv2.minMaxLoc(r)
    fy = max(0, ey - M) + mx[1]; fx = max(0, ex - M) + mx[0]
    win = np.asarray(arr[fy:fy + P, fx:fx + P])
    d = np.abs(win.astype(np.int32) - patch_slide.astype(np.int32))
    print(f'full-res check: ncc {sc2:.6f}  max diff {d.max()}  identical px '
          f'{100*(d.sum(-1) == 0).mean():.2f}%')
    print('=> lossless crop' if d.max() == 0 else
          '=> NOT pixel-identical: the reference was resampled or edited')
    top = fy - (ey - oy); left = fx - (ex - ox)
    print(f'\nre-extract with:\n  python wsi_extract.py extract --wsi "{args.wsi}" '
          f'--y {top} --x {left} --h {rh} --w {rw} '
          f'{"--rotate " + str(name) if isinstance(name, int) and name else ""}'
          f' --out <file>')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    e = sub.add_parser('extract', help='write a region at native resolution')
    e.add_argument('--wsi', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--y', type=int, default=0)
    e.add_argument('--x', type=int, default=0)
    e.add_argument('--h', type=int, default=None, help='Default: to the bottom edge.')
    e.add_argument('--w', type=int, default=None, help='Default: to the right edge.')
    e.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    e.add_argument('--flip', choices=['h', 'v'], default=None)
    e.add_argument('--downsample', type=int, default=1,
                   help='Only for previews -- this does lose resolution.')
    e.add_argument('--compression', default='zlib')
    e.set_defaults(func=cmd_extract)

    l = sub.add_parser('locate', help='find where an exported crop came from')
    l.add_argument('--wsi', required=True)
    l.add_argument('--ref', required=True)
    l.add_argument('--ds', type=int, default=8, help='Downsample for the coarse search.')
    l.add_argument('--scales', type=float, nargs='+',
                   default=[0.5, 0.75, 0.9, 1.0, 1.1, 1.25],
                   help='Scale hypotheses; 1.0 winning means a straight crop.')
    l.add_argument('--try_flips', action='store_true')
    l.add_argument('--refine', type=int, default=400)
    l.add_argument('--report_above', type=float, default=0.4)
    l.set_defaults(func=cmd_locate)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
