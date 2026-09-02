#!/usr/bin/env python3
"""Turn per-tile scores into a map over the slide and a shortlist of figure regions.

Three outputs, because a CSV of scores does not tell you where to look:

  heatmap_<key>.png   the score painted back onto the slide thumbnail, per structure
  top_<key>.png       the highest-scoring tiles for that structure, side by side
  summary.csv         per-structure statistics and the coordinates of the best tiles

Scores are z-normalised per structure before mapping. CONCH similarities sit in a
narrow band whose absolute level differs between prompts, so raw values would make one
structure look bright everywhere and another dark everywhere; what matters is where a
structure stands out relative to the rest of this slide.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import tifffile as tiff

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_structures import ALL_ENTRIES, ECM_KEYS, KEYS, PROBE_KEYS  # noqa: E402

# ALL_ENTRIES, not STRUCTURES, so a --probes run can label its columns too.
ZH = {s['key']: s['zh'] for s in ALL_ENTRIES}



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


def tissue_mask(thumb, thresh=215):
    """Heat only belongs where tissue is. Painting it over glass suggests a score was
    measured there, and none was."""
    g = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY) if thumb.ndim == 3 else thumb
    m = (g < thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def tile_lattice(rows, ds, tile):
    """Sorted tile origins on the thumbnail, and the sampling step between them.

    Tiles sit on a regular lattice (optionally overlapping), so their scores can be
    lifted onto a small dense array and resampled, rather than stamped as squares.
    """
    ys = sorted({int(r['y']) for r in rows})
    xs = sorted({int(r['x']) for r in rows})
    step = min([b - a for a, b in zip(ys, ys[1:])] +
               [b - a for a, b in zip(xs, xs[1:])] or [tile])
    return ys, xs, step


def interpolate_field(rows, z, shape, ds, tile, ys, xs, step, smooth=True):
    """Score field on the thumbnail, plus a mask of where it was actually sampled.

    Without smooth=True this is the original behaviour: every tile stamped as a flat
    square, which makes the sampling grid look like a property of the tissue. With it,
    tile centres become a small lattice that is resampled with cubic interpolation --
    the field then varies continuously and the visible edges are the tissue's, not the
    grid's. Either way `sampled` marks the area the tiles covered, so nothing is drawn
    where nothing was measured.
    """
    H, W = shape
    h = max(1, tile // ds)
    sampled = np.zeros(shape, bool)
    for r in rows:
        y, x = int(r['y']) // ds, int(r['x']) // ds
        sampled[y:y + h, x:x + h] = True

    if not smooth:
        heat = np.zeros(shape, np.float32)
        cnt = np.zeros(shape, np.float32)
        for r, zz in zip(rows, z):
            y, x = int(r['y']) // ds, int(r['x']) // ds
            heat[y:y + h, x:x + h] += zz
            cnt[y:y + h, x:x + h] += 1
        return heat / np.maximum(cnt, 1), sampled

    yi = {y: i for i, y in enumerate(ys)}
    xi = {x: i for i, x in enumerate(xs)}
    grid = np.full((len(ys), len(xs)), np.nan, np.float32)
    for r, zz in zip(rows, z):
        grid[yi[int(r['y'])], xi[int(r['x'])]] = zz

    # Blank tiles leave holes. Filling them by nearest neighbour keeps the cubic
    # resample from ringing across them; they are masked out again afterwards.
    hole = np.isnan(grid)
    if hole.any():
        filled = grid.copy()
        filled[hole] = 0
        _, lbl = cv2.distanceTransformWithLabels(
            hole.astype(np.uint8), cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)
        src = np.zeros(lbl.max() + 1, np.float32)
        src[lbl[~hole]] = grid[~hole]
        filled[hole] = src[lbl[hole]]
        grid = filled

    # Place each tile centre at its own position, so the resample lands where the
    # samples actually are rather than being stretched to the image corners.
    cy0 = (ys[0] + tile / 2) / ds
    cx0 = (xs[0] + tile / 2) / ds
    big = cv2.resize(grid, (len(xs) * 8, len(ys) * 8), interpolation=cv2.INTER_CUBIC)
    heat = np.zeros(shape, np.float32)
    sy, sx = step / ds, step / ds
    yy = np.clip(((np.arange(H) - cy0) / sy) * 8 + 4, 0, big.shape[0] - 1)
    xx = np.clip(((np.arange(W) - cx0) / sx) * 8 + 4, 0, big.shape[1] - 1)
    heat = cv2.remap(big, np.tile(xx, (H, 1)).astype(np.float32),
                     np.tile(yy[:, None], (1, W)).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return heat, sampled


def draw_colorbar(img, label, lo=-2.0, hi=2.0):
    """Legend for the map, scaled so it looks the same at any output resolution.

    Everything is sized from the image height against a 1800 px reference. Without that
    the bar keeps its 28 percent share of a --hires canvas and becomes a 2000 px ribbon
    next to unchanged 10 px text.

    The wording is deliberately "match", not "amount": a high score says this region
    resembles the named structure more than the rest of this slide does, which is not
    the same as the structure being present -- adipose scored a confident top tile on a
    slide with no fat in it.
    """
    H, W = img.shape[:2]
    sc = max(0.6, H / 1800)
    bh = int(H * 0.26)
    w = int(22 * sc)
    pad = int(14 * sc)
    fs, fs_s = 0.42 * sc, 0.38 * sc
    th = max(1, int(round(sc)))
    y0 = H - bh - pad - int(46 * sc)
    x0 = pad

    ramp = np.linspace(255, 0, bh).astype(np.uint8).reshape(-1, 1)
    img[y0:y0 + bh, x0:x0 + w] = cv2.applyColorMap(np.repeat(ramp, w, axis=1),
                                                   cv2.COLORMAP_INFERNO)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + bh), (60, 60, 60), th)
    for frac, txt in ((0.0, f'{hi:+.0f}'), (0.5, ' 0'), (1.0, f'{lo:+.0f}')):
        yy = int(y0 + frac * bh)
        cv2.line(img, (x0 + w, yy), (x0 + w + int(5 * sc), yy), (60, 60, 60), th)
        cv2.putText(img, txt, (x0 + w + int(8 * sc), yy + int(5 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), th)
    # Naming the structure on the bar means a heatmap read on its own, out of the
    # folder, still says what it is a match FOR. Two lines because 'strongest match for normal_epithelium' is wider than the margin at 1x.
    cv2.putText(img, 'strongest match for', (x0 - int(3 * sc), y0 - int(26 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
    cv2.putText(img, label, (x0 - int(3 * sc), y0 - int(9 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 1.05, (0, 0, 160), th + 1)
    cv2.putText(img, 'weakest', (x0 - int(3 * sc), y0 + bh + int(20 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th)
    cv2.putText(img, 'relative to the rest of this slide',
                (x0 - int(3 * sc), y0 + bh + int(38 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, fs_s, (110, 110, 110), th)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--tissue_thresh', type=int, default=None,
                    help='Defaults to whatever source.txt records for this tile '
                         'set, so the box is drawn round the tissue that was '
                         'actually tiled rather than round a differently '
                         'thresholded version of it. Tile sets cut before that '
                         'field existed fall back to 215 with a warning.')
    ap.add_argument('--panel_px', type=int, default=512,
                    help='Tile size in the top_*.png sheets. Tiles are cut at 512 px, so the old 256 threw away half the detail for no reason.')
    ap.add_argument('--image', default=None,
                    help='Source image for --hires, when the path recorded in source.txt no longer resolves.')
    ap.add_argument('--hires', type=int, default=0,
                    help='Re-read the source image at this downsample to draw the heatmap on. The thumbnail saved at tiling time uses thumb_ds=16, which is only 1024 px across for a 8 mm slide; 4 or 8 gives a far sharper map.')
    ap.add_argument('--blocky', action='store_true',
                    help='Stamp each tile as a flat square, the original behaviour. The default interpolates between tile centres instead, which stops the sampling grid from reading as a property of the tissue.')
    ap.add_argument('--only_ecm', action='store_true',
                    help='Only the ECM structures, where TPAF has a physical reason to differ.')
    ap.add_argument('--probes', action='store_true',
                    help='Report the PROBES candidate-caption columns instead of the '
                         'eighteen real ones. Needs a CSV scored with --probes.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
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
    tile = int(src['tile_px'])
    um = float(src['um_per_px'])
    ds = int(src['thumb_ds'])
    if args.hires:
        # The thumbnail written at tiling time is thumb_ds=16, i.e. ~1000 px
        # across for a whole slide, which is why the maps look soft. Re-reading
        # the source at a finer step costs a few seconds and nothing else; the
        # tile coordinates are in full-resolution pixels either way.
        ipath = args.image or src['image']
        if not os.path.exists(ipath):
            raise SystemExit(
                f'--hires needs the source image; {ipath!r} (from source.txt) does'
                ' not exist. Pass --image explicitly.')
        img = tiff.memmap(ipath)
        thumb = np.asarray(img[::args.hires, ::args.hires])[..., :3][..., ::-1]
        thumb = np.ascontiguousarray(thumb)
        ds = args.hires
        print(f'hires thumbnail {thumb.shape[1]}x{thumb.shape[0]} (ds={ds})', flush=True)
    else:
        thumb = cv2.imread(os.path.join(args.tiles, 'thumbnail.png'))
    rows = list(csv.DictReader(open(args.scores, encoding='utf-8')))

    # The scoring CSV carries tile_id and scores only; coordinates live in index.csv.
    # Join them here, and drop scored tiles the index does not know about so a mismatched
    # --tiles/--scores pair fails loudly on the count rather than silently mapping wrong.
    coords = {r['tile_id']: r for r in
              csv.DictReader(open(os.path.join(args.tiles, 'index.csv'), encoding='utf-8'))}
    n_in = len(rows)
    rows = [dict(r, y=coords[r['tile_id']]['y'], x=coords[r['tile_id']]['x'])
            for r in rows if r['tile_id'] in coords]
    if not rows:
        raise SystemExit('no scored tile appears in index.csv -- wrong --tiles directory?')
    if len(rows) < n_in:
        print(f'warning: {n_in - len(rows)} scored tiles not in index.csv, dropped', flush=True)

    wanted = PROBE_KEYS if args.probes else (ECM_KEYS if args.only_ecm else KEYS)
    keys = [k for k in wanted if k in rows[0]]
    if args.probes and not keys:
        raise SystemExit('--probes but the CSV has no probe columns; rescore with '
                         'path_colab_score.py --probes')
    print(f'{len(rows)} tiles, {len(keys)} structures, tile {tile} px '
          f'= {tile*um:.0f} um', flush=True)

    # The scores are sampled on the tile grid, so the honest picture is blocky. Two
    # things are still worth fixing: heat was being painted onto glass, where nothing
    # was ever measured, and a grid of hard squares reads as if the boundaries meant
    # something. Interpolating between tile centres and clipping to the tissue mask
    # fixes both -- but the smoothness is presentation, not resolution, so the caption
    # states the sampling step.
    tmask = tissue_mask(thumb, args.tissue_thresh)
    gy, gx, gstep = tile_lattice(rows, ds, tile)

    summary = []
    for k in keys:
        v = np.array([float(r[k]) for r in rows])
        z = (v - v.mean()) / (v.std() + 1e-9)
        heat, sampled = interpolate_field(rows, z, thumb.shape[:2], ds, tile,
                                          gy, gx, gstep, smooth=not args.blocky)
        norm = np.clip((heat + 2) / 4, 0, 1)
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        show = sampled & tmask
        col[~show] = 255
        blend = cv2.addWeighted(thumb, 0.45, col, 0.55, 0)
        blend[~show] = thumb[~show]
        draw_colorbar(blend, k)
        # On a dark bar: the top of a slide thumbnail is glass, so white-on-white was
        # invisible. English only -- Hershey fonts have no CJK glyphs, and the Chinese
        # name is carried in summary.csv instead.
        tsc = max(0.6, blend.shape[0] / 1800)
        cv2.rectangle(blend, (0, 0), (blend.shape[1], int(34 * tsc)), (0, 0, 0), -1)
        cv2.putText(blend, f'{k}   sampled every {gstep*um:.0f} um'
                    f"{'' if args.blocky else '  (interpolated)'}", (int(8*tsc), int(25*tsc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * tsc, (255, 255, 255),
                    max(1, int(round(2 * tsc))))
        cv2.imwrite(os.path.join(args.out, f'heatmap_{k}.png'), blend)

        order = np.argsort(-v)[:args.top]
        panels = []
        for i in order:
            r = rows[i]
            p = tile_path(args.tiles, r['tile_id'])
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.resize(im, (args.panel_px, args.panel_px),
                            interpolation=cv2.INTER_AREA)
            sc = args.panel_px / 256
            cv2.putText(im, f"z{z[i]:+.1f}", (int(6*sc), int(22*sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55*sc, (0, 0, 255), max(1,int(2*sc)))
            cv2.putText(im, r['tile_id'], (int(6*sc), int(250*sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42*sc, (0, 0, 255), max(1,int(1*sc)))
            panels.append(im)
        if panels:
            cols = min(4, len(panels))
            rws = (len(panels) + cols - 1) // cols
            sheet = np.full((rws * args.panel_px, cols * args.panel_px, 3), 255, np.uint8)
            for i, p in enumerate(panels):
                rr, cc = divmod(i, cols)
                P = args.panel_px
                sheet[rr * P:(rr + 1) * P, cc * P:(cc + 1) * P] = p
            cv2.imwrite(os.path.join(args.out, f'top_{k}.png'), sheet)

        best = [rows[i] for i in order[:3]]
        summary.append(dict(structure=k, zh=ZH[k], ecm=int(k in ECM_KEYS),
                            mean=round(float(v.mean()), 4), std=round(float(v.std()), 4),
                            top_z=round(float(z[order[0]]), 2),
                            top_tiles=' '.join(f"{b['tile_id']}" for b in best)))
        print(f'  {k:<20}{ZH[k]:<10} top z {z[order[0]]:+.1f}  best {best[0]["tile_id"]}',
              flush=True)

    # utf-8-sig: Excel on a Chinese Windows opens a plain UTF-8 CSV as GBK and turns
    # the zh column into mojibake. The BOM is what makes it pick UTF-8.
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f'\n-> {args.out}: heatmap_*.png, top_*.png, summary.csv')
    print('top_tiles coordinates are y_x in full-resolution pixels of the source image')


if __name__ == '__main__':
    main()
