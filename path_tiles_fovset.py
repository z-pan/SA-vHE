#!/usr/bin/env python3
"""Cut a set of same-size FOV images into tiles, optionally as a matched pair.

path_tiles.py cuts one big image. The virtual-staining outputs are not one big image:
they are 7 separate 816x816 FOVs, one per registered field. Tiling a montage of them
would emit tiles straddling two FOVs, which are not real fields of view.

So the grid is generated per FOV, and the coordinates written to index.csv are
positions on a virtual canvas that lays the FOVs out side by side. That keeps
path_report.py working unchanged -- it maps scores back onto the canvas thumbnail and
gets one heatmap showing every FOV at once.

Paired mode
-----------
With --b_dir, the same tile list is cut from a second modality. The tissue filter runs
once, on --filter_on, and the surviving positions are used for both, so tile N of one
set is the same field as tile N of the other. Filtering each side separately would give
two different tile sets and silently destroy the comparison.

Pairing is only meaningful where the two modalities are actually registered, so each
pair is scored by normalised cross-correlation after high-pass filtering and the result
is written to index.csv as paired_ncc / paired_ok. Tiles from a badly registered FOV
are still emitted -- they are valid for scoring one modality on its own -- but they
must not be read as a matched comparison.

    python path_tiles_fovset.py --a_dir results/stitched_vHE_overlap --a_suffix _AF \\
        --b_dir results/_st --b_suffix _HE_reg --filter_on b \\
        --um_per_px 0.621 --tile_um 256 --overlap 0.5 --out results/path_screen/240817_vHE
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import cv2
import numpy as np

PAIR_NCC_MIN = 0.10  # below this a pair is noise; a registered pair here scores 0.15-0.52


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f'cannot read {path}')
    return im[..., ::-1]


def highpass_norm(rgb, sigma=12):
    """Structure only. Raw grayscale correlation is dominated by the stain and
    illumination level, which differ between a generated and a real H&E by
    construction; the high pass leaves the tissue architecture the match is about."""
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return (g - g.mean()) / (g.std() + 1e-9)


def collect(d, suffix):
    out = {}
    for n in sorted(os.listdir(d)):
        if not n.lower().endswith('.png') or suffix + '.png' not in n:
            continue
        out[n[:-(len(suffix) + 4)]] = os.path.join(d, n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a_dir', required=True)
    ap.add_argument('--a_suffix', default='')
    ap.add_argument('--a_name', default='vHE')
    ap.add_argument('--b_dir', default=None)
    ap.add_argument('--b_suffix', default='')
    ap.add_argument('--b_name', default='realHE')
    ap.add_argument('--filter_on', choices=['a', 'b'], default='a',
                    help='Which modality the tissue filter reads. Use the real one.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--um_per_px', type=float, required=True)
    ap.add_argument('--tile_um', type=float, default=256.0)
    ap.add_argument('--overlap', type=float, default=0.5,
                    help='An 816 px FOV holds only one 412 px tile without overlap.')
    ap.add_argument('--out_px', type=int, default=512)
    ap.add_argument('--min_tissue', type=float, default=0.25)
    ap.add_argument('--tissue_thresh', type=int, default=215)
    ap.add_argument('--cols', type=int, default=4, help='Canvas layout.')
    ap.add_argument('--thumb_ds', type=int, default=4)
    args = ap.parse_args()

    A = collect(args.a_dir, args.a_suffix)
    B = collect(args.b_dir, args.b_suffix) if args.b_dir else {}
    keys = sorted(set(A) & set(B)) if B else sorted(A)
    if not keys:
        raise SystemExit('no FOV matched between the two directories')
    for k in sorted(set(A) ^ set(B)) if B else []:
        print(f'skipped unpaired FOV: {k}', flush=True)
    if args.filter_on == 'b' and not B:
        raise SystemExit('--filter_on b needs --b_dir')

    sets = {args.a_name: A}
    if B:
        sets[args.b_name] = B
    for name in sets:
        os.makedirs(os.path.join(args.out, name, 'tiles'), exist_ok=True)

    h0, w0 = read_rgb(A[keys[0]]).shape[:2]
    tile = int(round(args.tile_um / args.um_per_px))
    step = max(1, int(round(tile * (1 - args.overlap))))
    if tile > min(h0, w0):
        raise SystemExit(f'tile {tile} px > FOV {w0}x{h0} px')
    cols = min(args.cols, len(keys))
    rows_n = (len(keys) + cols - 1) // cols
    print(f'{len(keys)} FOV of {w0}x{h0} @ {args.um_per_px} um/px', flush=True)
    print(f'tile {tile} px = {args.tile_um} um, step {step} px, '
          f'canvas {cols*w0}x{rows_n*h0}', flush=True)

    canvas = {n: np.full((rows_n * h0, cols * w0, 3), 255, np.uint8) for n in sets}
    rows, dropped = [], 0
    for fi, k in enumerate(keys):
        oy, ox = (fi // cols) * h0, (fi % cols) * w0
        ims = {n: read_rgb(s[k]) for n, s in sets.items()}
        if any(im.shape[:2] != (h0, w0) for im in ims.values()):
            raise SystemExit(f'{k}: FOVs are not all the same size')
        for n, im in ims.items():
            canvas[n][oy:oy + h0, ox:ox + w0] = im

        ncc, ok = '', ''
        if B:
            ncc = float((highpass_norm(ims[args.a_name])
                         * highpass_norm(ims[args.b_name])).mean())
            ok = int(ncc >= PAIR_NCC_MIN)
            print(f'  {k[-11:]:<12} pair ncc {ncc:+.3f}  '
                  f'{"paired" if ok else "NOT REGISTERED -- do not compare"}', flush=True)
            ncc = round(ncc, 4)

        ref = ims[args.b_name if args.filter_on == 'b' else args.a_name]
        gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        for y in range(0, h0 - tile + 1, step):
            for x in range(0, w0 - tile + 1, step):
                frac = float((gray[y:y + tile, x:x + tile] < args.tissue_thresh).mean())
                if frac < args.min_tissue:
                    dropped += 1
                    continue
                tid = f'{re.sub(r"[^A-Za-z0-9]+", "-", k).strip("-")[-11:]}__y{y}_x{x}'
                rows.append(dict(tile_id=tid, y=oy + y, x=ox + x, size=tile,
                                 tissue=round(frac, 3), fov=k,
                                 fov_y=y, fov_x=x, paired_ncc=ncc, paired_ok=ok))
                for n, im in ims.items():
                    cv2.imwrite(
                        os.path.join(args.out, n, 'tiles', tid + '.png'),
                        cv2.resize(im[y:y + tile, x:x + tile], (args.out_px, args.out_px),
                                   interpolation=cv2.INTER_AREA)[..., ::-1])
    print(f'{len(rows)} tiles per modality, {dropped} dropped as blank', flush=True)
    if B:
        bad = sorted({r['fov'] for r in rows if not r['paired_ok']})
        n_bad = sum(1 for r in rows if not r['paired_ok'])
        if bad:
            print(f'WARNING {n_bad}/{len(rows)} tiles come from {len(bad)} FOV whose two '
                  f'modalities are not registered; valid per modality, not as a pair',
                  flush=True)

    for n in sets:
        d = os.path.join(args.out, n)
        cv2.imwrite(os.path.join(d, 'thumbnail.png'),
                    canvas[n][::args.thumb_ds, ::args.thumb_ds, ::-1])
        with open(os.path.join(d, 'index.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(d, 'source.txt'), 'w', encoding='utf-8') as fh:
            fh.write(f'image={os.path.abspath(args.a_dir if n == args.a_name else args.b_dir)}\n'
                     f'W={cols*w0}\nH={rows_n*h0}\num_per_px={args.um_per_px}\n'
                     f'tile_px={tile}\ntile_um={args.tile_um}\nstep_px={step}\n'
                     f'thumb_ds={args.thumb_ds}\nn_fov={len(keys)}\ncanvas_cols={cols}\n'
                     f'fov_px={w0}\nmodality={n}\n')
        print(f'-> {d}: {len(rows)} tiles, index.csv, thumbnail.png')


if __name__ == '__main__':
    main()
