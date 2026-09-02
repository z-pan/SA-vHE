#!/usr/bin/env python3
"""Measure the offsets between the TPAF FOVs a candidate region spans, and stitch them.

Twenty of the 148 linked regions straddle a FOV boundary, so their TPAF side is two or
four 1024 px frames that have to be put together before the region can go in a figure.
The offsets are measured from the images, not taken from the acquisition grid: the
per-FOV exports carry no stage coordinates, the nominal grid in
TileConfiguration_new.txt disagrees with the measured step by about 26 percent, and the
registered one is degenerate (all entries at 0,0). Phase correlation on the overlap is
the only thing here that knows where the frames actually sit.

What it reports per pair, and why each number is worth reading:

  dy, dx    the shift, in pixels, that puts B into A's frame.
  overlap   how many pixels wide the shared strip is. An acquisition with ~26 percent
            overlap should land near 270 on the axis of travel; far from that and the
            two frames are probably not neighbours.
  ncc       correlation over the shared strip only. This is the one that decides
            whether the fit is real. TPAF frames are mostly black, and two mostly-black
            frames correlate well at almost any offset, so a peak in phase correlation
            is not by itself evidence -- the strip has to agree.

Nothing is stitched unless the fit passes; a bad pair is reported and skipped rather
than blended into a plausible-looking wrong image.

    python path_tpaf_stitch.py                 # measure and report only
    python path_tpaf_stitch.py --write         # also write the stitched images
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import itertools
import os

import cv2
import numpy as np


def load(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f'cannot read {path}')
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im[..., :3]


def measure(a, b):
    """Candidate shifts (dy, dx) putting b into a's frame, best first.

    Windowed, because an unwindowed transform of a square image correlates its own
    borders and reports zero.

    Phase correlation is periodic: it cannot tell +790 from -234, and for neighbouring
    FOVs the wrapped value is the small one, which looks like a near-perfect overlap
    and is completely wrong. So every combination of the raw peak and the peak plus or
    minus a frame is returned, and the caller settles it on the overlap strip -- the
    unwrapping is a measurement, not an assumption about the acquisition step.
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Two frames in this set are cropped short (847 and 773 px wide) -- the acquisition
    # was cut at the edge of the tissue. Zero-pad both to a common size so the transform
    # is defined; the padding is dark like the background and the strip NCC, which is
    # computed on the real pixels, is what decides anything.
    h = max(ga.shape[0], gb.shape[0])
    w = max(ga.shape[1], gb.shape[1])
    pa = np.zeros((h, w), np.float32); pa[:ga.shape[0], :ga.shape[1]] = ga
    pb = np.zeros((h, w), np.float32); pb[:gb.shape[0], :gb.shape[1]] = gb
    ga, gb = pa, pb
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ga * win, gb * win)
    dy0, dx0 = -sy, -sx
    out = []
    for dy in {int(round(dy0)), int(round(dy0 + h)), int(round(dy0 - h))}:
        for dx in {int(round(dx0)), int(round(dx0 + w)), int(round(dx0 - w))}:
            if abs(dy) < h and abs(dx) < w:
                out.append((dy, dx))
    return out


def overlap_ncc(a, b, dy, dx):
    """Correlation over just the shared strip, and its size."""
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]
    ay0, by0 = max(0, dy), max(0, -dy)
    ax0, bx0 = max(0, dx), max(0, -dx)
    hh = min(ah - ay0, bh - by0)
    ww = min(aw - ax0, bw - bx0)
    if hh < 32 or ww < 32:
        return 0.0, 0, 0
    pa = cv2.cvtColor(a[ay0:ay0 + hh, ax0:ax0 + ww], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pb = cv2.cvtColor(b[by0:by0 + hh, bx0:bx0 + ww], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pa -= pa.mean()
    pb -= pb.mean()
    d = np.linalg.norm(pa) * np.linalg.norm(pb)
    return (float((pa * pb).sum() / d) if d else 0.0), ww, hh


def refine(a, b, dy, dx, r=24):
    """Local search around the phase-correlation peak, maximising the strip NCC.

    Phase correlation finds the periodic best fit; what matters for stitching is that
    the shared tissue lines up, and the two differ by a few pixels often enough to be
    visible as a doubled edge in the seam.
    """
    best = (overlap_ncc(a, b, dy, dx)[0], dy, dx)
    for ddy in range(-r, r + 1, 4):
        for ddx in range(-r, r + 1, 4):
            n = overlap_ncc(a, b, dy + ddy, dx + ddx)[0]
            if n > best[0]:
                best = (n, dy + ddy, dx + ddx)
    _, by, bx = best
    for ddy in range(-3, 4):
        for ddx in range(-3, 4):
            n = overlap_ncc(a, b, by + ddy, bx + ddx)[0]
            if n > best[0]:
                best = (n, by + ddy, bx + ddx)
    return best[1], best[2], best[0]


def place(imgs, names, min_ncc, min_ov, log):
    """Lay the frames out in one canvas by growing from the first that fits.

    Pairwise offsets are measured for every pair, then frames are added one at a time
    to whichever placed frame they agree with best. Chaining like this keeps a weak
    pair from anchoring the layout when a strong one exists -- in a 2x2 block the
    diagonal pair barely overlaps, and forcing the layout through it would be the worst
    available choice.
    """
    n = len(imgs)
    fits = {}
    for i, j in itertools.combinations(range(n), 2):
        best = None
        for cy, cx in measure(imgs[i], imgs[j]):
            ry, rx, rn = refine(imgs[i], imgs[j], cy, cx)
            # Prefer the branch that actually agrees on its strip. A tie on correlation
            # is broken toward the smaller overlap, because two frames sharing 90
            # percent of their area is not a pair of neighbours -- it is the wrapped
            # solution wearing the right number.
            ov = overlap_ncc(imgs[i], imgs[j], ry, rx)[1:]
            # A fit whose shared strip is a sliver is not a neighbour relation however
            # well it correlates: 30 px of mostly-background edge will agree with
            # anything, and stitching on it puts the frames in the wrong place with no
            # visible seam to give it away. Ranked below any fit with a real strip.
            key = (min(ov) >= min_ov, round(rn, 3), -(ov[0] * ov[1]))
            if best is None or key > best[0]:
                best = (key, ry, rx, rn, ov)
        _, dy, dx, ncc, (ov_w, ov_h) = best
        good = min(ov_w, ov_h) >= min_ov
        fits[(i, j)] = (dy, dx, ncc, ov_w, ov_h, good)
        fits[(j, i)] = (-dy, -dx, ncc, ov_w, ov_h, good)
        log.append(f'    {names[i][-22:]} -> {names[j][-22:]}  '
                   f'dy{dy:+5d} dx{dx:+5d}  ncc {ncc:.3f}  overlap {ov_w}x{ov_h}'
                   + ('' if good else '  (strip too thin)'))

    pos = {0: (0, 0)}
    while len(pos) < n:
        best = None
        for j in range(n):
            if j in pos:
                continue
            for i in pos:
                dy, dx, ncc, ow, oh, good = fits[(i, j)]
                key = (good, ncc)
                if best is None or key > best[0]:
                    best = (key, i, j, dy, dx, ncc, good)
        (_, ncc), i, j, dy, dx, ncc, good = best
        if not good or ncc < min_ncc:
            why = 'strip too thin' if not good else f'ncc {ncc:.3f} < {min_ncc}'
            log.append(f'    REJECT {names[j][-22:]}: {why}')
            return None, fits
        pos[j] = (pos[i][0] + dy, pos[i][1] + dx)
    return pos, fits


def compose(imgs, pos, feather=64):
    """Lay the frames on one canvas, blending only where they actually overlap.

    Lossless where it can be. A pixel covered by a single frame is copied straight
    across, byte for byte -- running it through the weighted sum instead costs a grey
    level to float rounding, which is silly for the 80-odd percent of the canvas that
    has nothing to blend. Only genuinely shared pixels are averaged, on a linear ramp
    so the seam is a gradient rather than a step.
    """
    ys = [p[0] for p in pos.values()]
    xs = [p[1] for p in pos.values()]
    y0, x0 = min(ys), min(xs)
    H = max(pos[i][0] + imgs[i].shape[0] for i in pos) - y0
    W = max(pos[i][1] + imgs[i].shape[1] for i in pos) - x0

    cover = np.zeros((H, W), np.uint8)
    for i, (dy, dx) in pos.items():
        h, w = imgs[i].shape[:2]
        cover[dy - y0:dy - y0 + h, dx - x0:dx - x0 + w] += 1

    out = np.zeros((H, W, 3), np.uint8)
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    for i, (dy, dx) in pos.items():
        h, w = imgs[i].shape[:2]
        yy, xx = dy - y0, dx - x0
        sl = (slice(yy, yy + h), slice(xx, xx + w))
        single = cover[sl] == 1
        out[sl][single] = imgs[i][single]
        ramp = np.ones((h, w), np.float32)
        f = max(1, min(feather, h // 2, w // 2))
        lin = np.linspace(0, 1, f, dtype=np.float32)
        ramp[:f, :] *= lin[:, None]
        ramp[-f:, :] *= lin[::-1, None]
        ramp[:, :f] *= lin[None, :]
        ramp[:, -f:] *= lin[None, ::-1]
        ramp = ramp[..., None] + 1e-3
        acc[sl] += imgs[i].astype(np.float32) * ramp
        wsum[sl] += ramp
    shared = cover > 1
    out[shared] = (acc[shared] / np.maximum(wsum[shared], 1e-6)).round().clip(0, 255)
    return out


def read_layout(path):
    """A layout.txt, hand-edited or machine-written -> [(filename, y, x)].

    Tab-separated so a filename with spaces -- every one of them here -- survives,
    with a whitespace fallback for a file that has been through an editor that
    helpfully turned tabs into spaces.
    """
    out = []
    for raw in io.open(path, encoding='utf-8'):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) < 3:
            parts = [p.strip() for p in line.rsplit(None, 2)]
        if len(parts) < 3 or not parts[1].startswith('y='):
            continue
        out.append((parts[0], int(parts[1][2:]), int(parts[2][2:])))
    return out


def write_layout(path, rid, tile_id, entries, note=''):
    with io.open(path, 'w', encoding='utf-8') as fh:
        fh.write(f'region={rid}' + chr(10))
        fh.write(f'tile_id={tile_id}' + chr(10))
        fh.write('um_per_px=0.621' + chr(10))
        fh.write(f'frames={len(entries)}' + chr(10))
        for line in (note.split(chr(10)) if note else []):
            fh.write(f'# {line}' + chr(10))
        for name, y, x in entries:
            fh.write(name + chr(9) + f'y={y}' + chr(9) + f'x={x}' + chr(10))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--links', default='results/path_screen/survey/_candidates/tpaf_links.csv')
    ap.add_argument('--out', default='results/path_screen/survey/_candidates/stitched')
    ap.add_argument('--write', action='store_true',
                    help='Write the stitched images. Without it nothing is created and '
                         'the run is just the measurement report.')
    ap.add_argument('--min_ncc', type=float, default=0.30,
                    help='Overlap correlation a pair must reach to be stitched. Frames '
                         'that are largely background correlate well anywhere, so this '
                         'rejects rather than confirms: passing it means the fit is not '
                         'obviously wrong, not that it is right. Look at the seams.')
    ap.add_argument('--min_overlap', type=int, default=120,
                    help='Shortest side the shared strip must have. A 26 percent '
                         'overlap of a 1024 px frame is about 270; anything under this '
                         'is an edge touching an edge, which correlates well and places '
                         'the frames wrongly with no seam to show it.')
    ap.add_argument('--feather', type=int, default=64,
                    help='Blend width, in pixels, inside the overlap. 0 butts the '
                         'frames together with a hard edge. Either way a pixel only '
                         'one frame covers is copied byte for byte.')
    ap.add_argument('--place', action='store_true',
                    help='Skip measuring and compose from the layout.txt files in '
                         '--out instead, so offsets found by hand elsewhere can be used. '
                         'Every region that has one is composed; the rest are skipped.')
    ap.add_argument('--templates', action='store_true',
                    help='Write a layout.txt for each region that could not be measured, '
                         'with all frames at 0,0 as a starting point to edit.')
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.links, encoding='utf-8-sig')))
    by = collections.OrderedDict()
    for r in rows:
        if r['tpaf_path']:
            by.setdefault(r['id'], []).append(r)
    multi = {k: v for k, v in by.items() if len(v) > 1}

    if args.place:
        # Offsets settled elsewhere -- Fiji, an image editor, by eye -- and written back
        # into the layout file. Composing them here rather than exporting from that tool
        # keeps every stitched region on the same footing: same blending, same lossless
        # single-cover copy, same recorded provenance.
        os.makedirs(args.out, exist_ok=True)
        done = 0
        for rid, group in multi.items():
            lp = os.path.join(args.out, f'{rid}_layout.txt')
            if not os.path.exists(lp):
                continue
            ent = read_layout(lp)
            paths = {r['tpaf_file']: r['tpaf_path'] for r in group}
            unknown = [n for n, _, _ in ent if n not in paths]
            if unknown:
                print(f'{rid}: layout names a frame not linked to this region: '
                      f'{unknown[0]}')
                continue
            if len(ent) != len(group):
                print(f'{rid}: layout has {len(ent)} frames, the region has '
                      f'{len(group)} -- skipped')
                continue
            if len(ent) > 1 and all(y == 0 and x == 0 for _, y, x in ent):
                # An untouched template. Composing it stacks every frame on the same
                # spot and writes a plausible-looking single-frame image, which is
                # worse than writing nothing at all.
                print(f'{rid}: template not filled in yet -- skipped')
                continue
            imgs = [load(paths[n]) for n, _, _ in ent]
            pos = {i: (y, x) for i, (_, y, x) in enumerate(ent)}
            im = compose(imgs, pos, args.feather)
            op = os.path.join(args.out, f'{rid}_stitched.png')
            cv2.imwrite(op, im)
            print(f'{rid}  {len(imgs)} frames -> {im.shape[1]}x{im.shape[0]} px '
                  f'= {im.shape[1] * 0.621 / 1000:.2f}x{im.shape[0] * 0.621 / 1000:.2f} mm'
                  f'  -> {op}')
            done += 1
        print()
        print(f'{done} composed from layout files')
        return

    if args.write:
        os.makedirs(args.out, exist_ok=True)

    print(f'\n{len(multi)} regions span more than one FOV '
          f'({sum(len(v) for v in multi.values())} frames total)\n')
    ok = fail = 0
    for rid, group in multi.items():
        names = [r['tpaf_file'] for r in group]
        imgs = [load(r['tpaf_path']) for r in group]
        log = []
        pos, fits = place(imgs, names, args.min_ncc, args.min_overlap, log)
        r0 = group[0]
        head = (f'{rid}  {len(imgs)} frames  tier{r0["tier"]}  {r0["tile_id"]}  '
                f'{r0["structures"]}')
        print(head)
        for line in log:
            print(line)
        if pos is None:
            fail += 1
            if args.templates:
                os.makedirs(args.out, exist_ok=True)
                tp = os.path.join(args.out, f'{rid}_layout.txt')
                if os.path.exists(tp):
                    print(f'  -> not stitched; {os.path.basename(tp)} exists, left alone')
                else:
                    write_layout(
                        tp, rid, r0['tile_id'], [(n, 0, 0) for n in names],
                        'Automatic alignment failed -- fill the offsets in by hand.'
                        + chr(10) +
                        'y is down and x is right, relative to the first frame.'
                        + chr(10) +
                        'Neighbouring FOVs on this set sit about 750 px apart '
                        '(26 percent overlap).'
                        + chr(10) +
                        'Then: python path_tpaf_stitch.py --place')
                    print(f'  -> not stitched; template at {tp}')
            else:
                print('  -> not stitched')
            print()
            continue
        H = (max(pos[i][0] + imgs[i].shape[0] for i in pos)
             - min(p[0] for p in pos.values()))
        W = (max(pos[i][1] + imgs[i].shape[1] for i in pos)
             - min(p[1] for p in pos.values()))
        layout = ' '.join(f'{names[i][-14:]}@({pos[i][0]},{pos[i][1]})'
                          for i in sorted(pos))
        print(f'  canvas {W}x{H} px = {W * 0.621 / 1000:.2f}x{H * 0.621 / 1000:.2f} mm')
        print(f'  {layout}')
        ok += 1
        if args.write:
            im = compose(imgs, pos, args.feather)
            p = os.path.join(args.out, f'{rid}_stitched.png')
            cv2.imwrite(p, im)
            with io.open(os.path.join(args.out, f'{rid}_layout.txt'), 'w',
                         encoding='utf-8') as fh:
                fh.write(f'region={rid}\ntile_id={r0["tile_id"]}\n'
                         f'um_per_px=0.621\nframes={len(imgs)}\n')
                for i in sorted(pos):
                    fh.write(f'{names[i]}\ty={pos[i][0]}\tx={pos[i][1]}\n')
            print(f'  -> {p}')
        print()

    print(f'{ok} stitched, {fail} rejected')
    if not args.write:
        print('measurement only; pass --write to produce the images')


if __name__ == '__main__':
    main()
