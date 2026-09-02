#!/usr/bin/env python3
"""Cut the stained patches at the recovered boxes and set them beside the real H&E.

The cut happens here and not before staining, which is the whole point of the manifest:
a stained crop and the same rectangle taken out of a stained patch are different images,
because the generator's normalisation and receptive field depend on what it was handed.

What can honestly be compared, and what cannot
----------------------------------------------
The two sides are the same tissue but not the same pixels. They are separate
acquisitions on separate instruments, at 0.621 and 0.5 um/px, and the regions were
matched by eye to "almost the same area" -- the crops run 201-525 um against the H&E
candidate's 256 um. Nothing here is registered.

So SSIM, PSNR and any other pixel-correspondence metric are not computed. On unregistered
pairs they measure the misalignment, and a number that mostly reflects how well the crop
was placed by hand would be read as a statement about staining quality. What is computed
instead is distribution-level: colour, stain concentration after deconvolution, and
tissue coverage -- quantities that survive the two sides showing overlapping rather than
identical fields.

Statistics are taken on raw values, pooled across the set, never normalised per image.
Per-image standardisation would remove exactly the difference being measured: the mean
offset between virtual and real staining is the thing under test, and subtracting each
image's own mean guarantees it comes out at zero.

For viewing, both sides are resampled to a common um/px and cut to the smaller of the
two physical extents, so what sits side by side covers the same amount of tissue at the
same scale. That resampling is for the figures only; the metrics use the native pixels.

    python path_vhe_collect.py --variant gray
    python path_vhe_collect.py --variant nuc_hi --sheet
"""

from __future__ import annotations

import argparse
import csv
import io
import os

import cv2
import numpy as np
from skimage.color import rgb2hed

TPAF_UM = 0.621
HE_UM = 0.5


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if im is None:
        raise SystemExit(f'cannot decode {path}')
    return im


def imwrite_u(path, im):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], im)
    if not ok:
        raise SystemExit(f'cannot encode {path}')
    buf.tofile(path)


def stats(bgr, white_thresh=0.92):
    """Colour and stain concentration over tissue only.

    Background is excluded because the two sides carry different amounts of it -- a
    crop with more glass would otherwise look paler in every channel, which says
    nothing about the staining.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tissue = rgb.mean(axis=2) < white_thresh
    frac = float(tissue.mean())
    if frac < 0.01:
        return dict(tissue_frac=frac)
    hed = rgb2hed(rgb)[tissue]
    px = rgb[tissue]
    return dict(
        tissue_frac=round(frac, 4),
        R=round(float(px[:, 0].mean() * 255), 1),
        G=round(float(px[:, 1].mean() * 255), 1),
        B=round(float(px[:, 2].mean() * 255), 1),
        RB_gap=round(float((px[:, 0].mean() - px[:, 2].mean()) * 255), 1),
        H=round(float(hed[:, 0].mean()), 4),
        E=round(float(hed[:, 1].mean()), 4),
        HE_ratio=round(float(hed[:, 0].mean() / (hed[:, 1].mean() + 1e-9)), 3),
    )


def to_common(im, um_per_px, out_um_per_px, extent_um):
    """Centre-cut to `extent_um` of tissue, then resample to `out_um_per_px`."""
    side = int(round(extent_um / um_per_px))
    h, w = im.shape[:2]
    side = min(side, h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cut = im[y0:y0 + side, x0:x0 + side]
    out = int(round(extent_um / out_um_per_px))
    interp = cv2.INTER_AREA if out < side else cv2.INTER_CUBIC
    return cv2.resize(cut, (out, out), interpolation=interp)


def label(im, text, colour):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1,
                cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--stained', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--variant', action='append', default=None,
                    help='Subdirectory of --stained to collect. Repeatable; defaults to '
                         'every one present, so the variants land in one table and can '
                         'be compared against each other and against the real H&E.')
    ap.add_argument('--out', default='results/path_screen/survey/_vhe/compare')
    ap.add_argument('--view_um', type=float, default=0.5,
                    help='um/px the side-by-side images are resampled to.')
    ap.add_argument('--panel', action='store_true', help='Write one panel per region.')
    ap.add_argument('--sheet', action='store_true',
                    help='Also write contact sheets, 6 regions per page.')
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    variants = args.variant or sorted(
        d for d in os.listdir(args.stained)
        if os.path.isdir(os.path.join(args.stained, d)))
    if not variants:
        raise SystemExit(f'no stained variants in {args.stained}')
    os.makedirs(args.out, exist_ok=True)
    for v in variants:
        os.makedirs(os.path.join(args.out, v), exist_ok=True)
    print(f'variants: {", ".join(variants)}\n')

    out_rows, panels, missing = [], [], set()
    for r in sorted(rows, key=lambda r: (r['sample'], int(r['n']))):
        name = os.path.splitext(r['stage_name'])[0]
        x, y = int(r['crop_x']), int(r['crop_y'])
        w, h = int(r['crop_w']), int(r['crop_h'])
        he = imread_u(r['he_path']) if r['he_path'] else None
        he_um = (he.shape[0] * HE_UM) if he is not None else 0.0
        vhe_um = w * TPAF_UM
        extent = min(he_um, vhe_um) if he is not None else vhe_um

        cells = []
        if he is not None:
            s = stats(he)
            out_rows.append(dict(id=r['id'], sample=r['sample'], n=r['n'],
                                 variant='real_HE', extent_um=round(extent, 1),
                                 px=he.shape[0], um_per_px=HE_UM, **s))
            cells.append(label(to_common(he, HE_UM, args.view_um, extent),
                               f'{r["id"]} real H&E', (80, 80, 255)))
        for v in variants:
            sp = os.path.join(args.stained, v, name + '.png')
            if not os.path.exists(sp):
                missing.add((v, name))
                continue
            full = imread_u(sp)
            if full.shape[0] < y + h or full.shape[1] < x + w:
                raise SystemExit(f'{r["id"]}: stained {v} is {full.shape[1]}x'
                                 f'{full.shape[0]}, box needs {x + w}x{y + h}')
            crop = full[y:y + h, x:x + w]
            imwrite_u(os.path.join(args.out, v, f'{r["id"]}.png'), crop)
            s = stats(crop)
            out_rows.append(dict(id=r['id'], sample=r['sample'], n=r['n'],
                                 variant=v, extent_um=round(extent, 1),
                                 px=w, um_per_px=TPAF_UM, **s))
            cells.append(label(to_common(crop, TPAF_UM, args.view_um, extent),
                               f'vHE {v}', (120, 255, 120)))
        if cells and (args.panel or args.sheet):
            hgt = max(c.shape[0] for c in cells)
            cells = [cv2.copyMakeBorder(c, 0, hgt - c.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
                     for c in cells]
            gap = np.full((hgt, 8, 3), 255, np.uint8)
            row = cells[0]
            for c in cells[1:]:
                row = np.hstack([row, gap, c])
            if args.panel:
                imwrite_u(os.path.join(args.out, f'panel_{r["id"]}.png'), row)
            panels.append(row)

    if args.sheet and panels:
        W = max(p.shape[1] for p in panels)
        for i in range(0, len(panels), 6):
            grp = [cv2.copyMakeBorder(p, 0, 8, 0, W - p.shape[1],
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
                   for p in panels[i:i + 6]]
            imwrite_u(os.path.join(args.out, f'sheet_{i // 6 + 1:02d}.png'),
                      np.vstack(grp))
        print(f'{(len(panels) + 5) // 6} contact sheets')

    cols = ['id', 'sample', 'n', 'variant', 'extent_um', 'px', 'um_per_px',
            'tissue_frac', 'R', 'G', 'B', 'RB_gap', 'H', 'E', 'HE_ratio']
    cpath = os.path.join(args.out, 'compare_metrics.csv')
    with io.open(cpath, 'w', newline='', encoding='utf-8-sig') as fh:
        w_ = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w_.writeheader()
        w_.writerows(out_rows)

    print(f'{len(out_rows)} rows -> {cpath}')
    if missing:
        print(f'{len(missing)} stained patches missing; e.g. {sorted(missing)[:3]}')

    # Pooled, on raw values. Per-image normalisation would zero out the mean offset
    # that is the thing being measured.
    print()
    print(f'{"variant":<12}{"n":>5}{"R":>8}{"G":>8}{"B":>8}{"R-B":>8}'
          f'{"H":>9}{"E":>9}{"H/E":>8}')
    for v in ['real_HE'] + variants:
        sub = [r for r in out_rows if r['variant'] == v and 'R' in r]
        if not sub:
            continue
        g = lambda k: np.mean([r[k] for r in sub])
        print(f'{v:<12}{len(sub):>5}{g("R"):>8.1f}{g("G"):>8.1f}{g("B"):>8.1f}'
              f'{g("RB_gap"):>8.1f}{g("H"):>9.4f}{g("E"):>9.4f}{g("HE_ratio"):>8.3f}')
    print()
    print('No SSIM or PSNR: the two sides are unregistered acquisitions of overlapping,')
    print('not identical, fields, so a pixel-correspondence score would mostly measure')
    print('how the crop was placed by hand.')


if __name__ == '__main__':
    main()
