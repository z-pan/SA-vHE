#!/usr/bin/env python3
"""Stitch a slide's TPAF FOVs into one mosaic, discovering the layout from the images.

Unlike the TMA, a slide acquisition has no known grid: the files are just
``..._0001.tif`` upward. The layout is recovered rather than assumed.

Measured on 240720 before writing this: consecutive indices overlap by about 27%
(step ~750 px for a 1024 px tile) with dy ~ +-16, and the sign of dx flips between
runs -- a serpentine raster. Rows change where a pair shows a large dy instead.

Three stages, because a chain of consecutive links alone is not rigid: error
accumulates along it and nothing ties one row to the next.

  1. link every (i, i+1) by full-range normalised cross-correlation
  2. solve provisional positions, then test pairs that land within ~1.2 tiles of each
     other -- these are the cross-row neighbours the acquisition order never visits
  3. re-solve with both sets

Correlation is normalised by the actual overlap count per shift, not by image area,
or every pair would be biased toward reporting a small shift.

Low-texture FOVs give a flat correlation surface and no usable link; they are left to
be positioned by their confident neighbours through the least-squares fit, the same
fallback the TMA stitcher uses.
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

IDX_RE = re.compile(r'_(\d{4})\.tif$', re.I)


def series(folder, wavelength=None):
    """Files of the dominant acquisition series, in acquisition order."""
    fs = [f for f in os.listdir(folder) if f.lower().endswith('.tif') and IDX_RE.search(f)]
    if not fs:
        return []
    waves = {}
    for f in fs:
        m = re.search(r'(\d{3}nm)', f)
        waves.setdefault(m.group(1) if m else '?', []).append(f)
    if wavelength is None:
        wavelength = max(waves, key=lambda k: len(waves[k]))
    keep = waves[wavelength]
    return sorted(keep, key=lambda f: int(IDX_RE.search(f).group(1))), wavelength


def load(path, ds, highpass=True):
    """Downsampled FOV, vignetting removed.

    Each FOV carries its own illumination falloff, and correlating raw intensity makes
    that falloff compete with the tissue. Subtracting a wide Gaussian fixes it and is
    worth a lot: on the first 40 FOVs of 240720 it takes accepted links from 30/39 to
    38/39 and more than doubles the median peak-to-runner-up ratio, 1.80 -> 4.37.
    Gradient magnitude, with or without the high pass, lands in between (33/39).
    """
    a = tiff.imread(path).astype(np.float32)
    a = a[..., :2].mean(2) if a.ndim == 3 else a
    a = cv2.resize(a, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    return a - cv2.GaussianBlur(a, (0, 0), 12.0 / ds * 4) if highpass else a


def _corr_map(a, b, min_overlap=0.10):
    """Correlation over all shifts. a and b may differ in size (whole row strips do)."""
    ha, wa = a.shape
    hb, wb = b.shape
    N = 1 << int(np.ceil(np.log2(max(ha + hb, wa + wb))))
    az = np.zeros((N, N), np.float32); bz = np.zeros((N, N), np.float32)
    ma = np.zeros((N, N), np.float32); mb = np.zeros((N, N), np.float32)
    az[:ha, :wa] = (a - a.mean()) / (a.std() + 1e-9)
    bz[:hb, :wb] = (b - b.mean()) / (b.std() + 1e-9)
    ma[:ha, :wa] = 1.0; mb[:hb, :wb] = 1.0
    num = np.fft.irfft2(np.fft.rfft2(bz) * np.conj(np.fft.rfft2(az)), s=(N, N))
    cnt = np.fft.irfft2(np.fft.rfft2(mb) * np.conj(np.fft.rfft2(ma)), s=(N, N))
    corr = num / np.maximum(cnt, 1.0)          # per overlapping pixel, unbiased by area
    corr[cnt < min_overlap * min(ha * wa, hb * wb)] = 0.0
    return corr, N


def _peak(corr, N, exclude_zero=0):
    """Best shift and its margin over the runner-up, ignoring a zero-shift blob.

    Per-overlap-pixel normalisation peaks spuriously at zero displacement: that is
    where the overlap is largest and the estimate steadiest, so two unrelated FOVs of
    similar texture score well there. On 240720 that produced a (0, 0) link at ratio
    1.41 -- just over the old threshold -- which pinned two tiles on top of each other
    and dragged the whole layout in. Raster neighbours are never coincident, so the
    zero blob is simply removed.
    """
    c = corr.copy()
    if exclude_zero:
        r = exclude_zero
        c[:r, :r] = 0; c[:r, -r:] = 0; c[-r:, :r] = 0; c[-r:, -r:] = 0
    p = np.unravel_index(np.argmax(c), c.shape)
    pk = float(c[p])
    dy = p[0] - N if p[0] > N // 2 else p[0]
    dx = p[1] - N if p[1] > N // 2 else p[1]
    c2 = c.copy()
    c2[max(0, p[0] - 10):p[0] + 10, max(0, p[1] - 10):p[1] + 10] = 0
    return dy, dx, pk, float(pk / (c2.max() + 1e-9))


def xcorr(a, b, min_overlap=0.10, exclude_zero=0):
    corr, N = _corr_map(a, b, min_overlap)
    return _peak(corr, N, exclude_zero)


def xcorr_near(a, b, cands, rad, min_overlap=0.10, exclude_zero=0):
    """Best shift within `rad` of any expected candidate, and its margin elsewhere.

    Used to recover row transitions the free search misses: once the raster step is
    known, looking only where a neighbour can physically be turns a weak peak into a
    decisive one.
    """
    corr, N = _corr_map(a, b, min_overlap)
    if exclude_zero:                 # the same spurious zero-shift blob, which would
        r = exclude_zero             # otherwise dominate the "outside" and crush every
        corr[:r, :r] = 0             # ratio below 1 -- it is why the first version of
        corr[:r, -r:] = 0            # this recovered 0 of 4 row changes
        corr[-r:, :r] = 0
        corr[-r:, -r:] = 0
    mask = np.zeros_like(corr, bool)
    yy, xx = np.ogrid[:N, :N]
    for cy, cx in cands:
        for oy in (0, N):
            for ox in (0, N):
                mask |= ((yy - (cy + oy)) ** 2 + (xx - (cx + ox)) ** 2) <= rad * rad
    inside = np.where(mask, corr, -np.inf)
    p = np.unravel_index(np.argmax(inside), inside.shape)
    pk = float(corr[p])
    dy = p[0] - N if p[0] > N // 2 else p[0]
    dx = p[1] - N if p[1] > N // 2 else p[1]
    out = corr[~mask]
    return dy, dx, pk, float(pk / (out.max() + 1e-9) if out.size else 1.0)


def solve(n, links, anchor=0):
    """Weighted least squares for tile origins from (i, j, dy, dx, weight) links."""
    pos = np.zeros((n, 2))
    for axis in (0, 1):
        rows, rhs, wts = [], [], []
        for i, j, dy, dx, w in links:
            r = np.zeros(n); r[j] = 1; r[i] = -1
            rows.append(r); rhs.append(dy if axis == 0 else dx); wts.append(w)
        r = np.zeros(n); r[anchor] = 1
        rows.append(r); rhs.append(0.0); wts.append(10.0)
        A = np.array(rows) * np.sqrt(np.array(wts))[:, None]
        b = np.array(rhs) * np.sqrt(np.array(wts))
        pos[:, axis] = np.linalg.lstsq(A, b, rcond=None)[0]
    return pos


def flat_field(paths, scale):
    """Illumination profile as the per-pixel median across FOVs.

    Every FOV shares the same vignetting, so the median over many of them is the
    illumination and not the tissue. Without dividing it out, blending overlapping
    tiles averages bright centres against dark edges and the mosaic interior washes
    out -- worse where more tiles overlap, which is exactly where coverage is best.
    """
    sub = paths[::max(1, len(paths) // 40)][:40]
    st = []
    for p in sub:
        a = tiff.imread(p).astype(np.float32)
        a = a[..., :2].mean(2) if a.ndim == 3 else a
        st.append(cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    prof = np.median(np.stack(st), 0)
    prof = cv2.GaussianBlur(prof, (0, 0), max(prof.shape) / 12.0)
    return prof / max(float(np.median(prof)), 1e-6)


def _components(n, links):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i, j, *_ in links:
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: min(g))


def link_rows(imgs, links, extra, n, tile, args):
    """Connect row segments by correlating the strips they render to."""
    comps = _components(n, links + extra)
    if len(comps) < 2:
        return links, extra
    print(f'{len(comps)} disconnected segments: '
          f'{[len(c) for c in comps][:12]}{"..." if len(comps) > 12 else ""}', flush=True)
    ds = args.ds
    t = imgs[0].shape[0]

    def render(idx, local):
        ys = [local[i][0] for i in idx]; xs = [local[i][1] for i in idx]
        h = int(max(y for y in ys) - min(ys)) + t
        w = int(max(x for x in xs) - min(xs)) + t
        acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        for i in idx:
            y = int(local[i][0] - min(ys)); x = int(local[i][1] - min(xs))
            acc[y:y + t, x:x + t] += imgs[i]; cnt[y:y + t, x:x + t] += 1
        return acc / np.maximum(cnt, 1), (min(ys), min(xs))

    local = {}
    for c in comps:
        sub = [(c.index(i), c.index(j), dy / ds, dx / ds, w)
               for i, j, dy, dx, w in links + extra if i in c and j in c]
        pl = solve(len(c), sub) if sub else np.zeros((len(c), 2))
        for k, i in enumerate(c):
            local[i] = pl[k]

    strips = [render(c, local) for c in comps]
    added = 0
    step = args._step / ds
    for a in range(len(comps) - 1):
        for b in range(a + 1, min(a + 3, len(comps))):
            ia, ib = comps[a], comps[b]
            base_y = strips[b][1][0] - strips[a][1][0]
            base_x = strips[b][1][1] - strips[a][1][1]
            # Constrain dy, not dx. Rows sit about one raster step apart vertically --
            # measured 780/736/732 against a 754 within-row step -- but they start at
            # different x and are of different lengths, so dx has to come from the data.
            # Constraining both rejected everything; constraining neither let dx wander
            # thousands of pixels, since two long horizontal strips barely fix the
            # offset along their own direction.
            corr, N = _corr_map(strips[a][0], strips[b][0])
            ez = int(0.25 * step)
            corr[:ez, :ez] = 0; corr[:ez, -ez:] = 0
            corr[-ez:, :ez] = 0; corr[-ez:, -ez:] = 0
            yy = np.arange(N)
            ay = np.where(yy > N // 2, yy - N, yy) - base_y
            ok = (np.abs(np.abs(ay) - step) < 0.45 * step)
            band = np.where(ok[:, None], corr, -np.inf)
            p_ = np.unravel_index(np.argmax(band), band.shape)
            pk = float(corr[p_])
            outside = corr[~ok[:, None] & np.ones_like(corr, bool)]
            r = float(pk / (outside.max() + 1e-9)) if outside.size else 1.0
            dy = p_[0] - N if p_[0] > N // 2 else p_[0]
            dx = p_[1] - N if p_[1] > N // 2 else p_[1]
            if r < args.min_ratio_row:
                if args.verbose:
                    print(f'    segment {a}<->{b}: dy={dy*ds:.0f} dx={dx*ds:.0f} '
                          f'r={r:.2f} rejected', flush=True)
                continue
            oy, ox = dy + base_y, dx + base_x
            gy = (oy - local[ib[0]][0] + local[ia[0]][0]) * ds
            gx = (ox - local[ib[0]][1] + local[ia[0]][1]) * ds
            extra.append((ia[0], ib[0], gy, gx, r))
            added += 1
            print(f'  segment {a}<->{b}: ({gy:.0f},{gx:.0f}) r={r:.2f}', flush=True)
    print(f'row-strip links added: {added}', flush=True)
    return links, extra


def blend(args, fs, pos, tile):
    s = args.out_scale
    H = int(np.ceil((pos[:, 0].max() + tile) * s)) + 2
    W = int(np.ceil((pos[:, 1].max() + tile) * s)) + 2
    print(f'mosaic {W}x{H} at scale {s}', flush=True)
    paths = [os.path.join(args.fov_dir, f) for f in fs]
    prof = None if args.no_flatfield else flat_field(paths, s)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for i, p in enumerate(paths):
        a = tiff.imread(p).astype(np.float32)
        a = a[..., :2].mean(2) if a.ndim == 3 else a
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if prof is not None:
            a = a / np.maximum(prof[:a.shape[0], :a.shape[1]], 0.15)
        y, x = int(round(pos[i, 0] * s)), int(round(pos[i, 1] * s))
        acc[y:y + a.shape[0], x:x + a.shape[1]] += a
        cnt[y:y + a.shape[0], x:x + a.shape[1]] += 1
    mos = acc / np.maximum(cnt, 1)
    tiff.imwrite(args.out + '_mosaic.tif', np.clip(mos, 0, 65535).astype(np.uint16),
                 compression='zlib')
    lo, hi = np.percentile(mos[cnt > 0], (1, 99.5)) if (cnt > 0).any() else (0, 1)
    cv2.imwrite(args.out + '_mosaic.png',
                np.clip((mos - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))
    print(f'covered {100*float((cnt>0).mean()):.0f}% of the bounding box; '
          f'{int((cnt>1).sum())/max(int((cnt>0).sum()),1)*100:.0f}% of it overlapped')
    print(f'-> {args.out}_mosaic.tif / .png / _pos.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fov_dir', required=True)
    ap.add_argument('--out', required=True, help='Output prefix (writes _mosaic.tif, _pos.csv).')
    ap.add_argument('--wavelength', default=None)
    ap.add_argument('--ds', type=int, default=4, help='Downsample for link search.')
    ap.add_argument('--out_scale', type=float, default=0.25, help='Mosaic output scale.')
    ap.add_argument('--min_ratio', type=float, default=1.4,
                    help='Peak / runner-up needed to trust a link.')
    ap.add_argument('--min_ratio_near', type=float, default=1.15,
                    help='Looser bar for the constrained re-test, where the search is '
                         'already restricted to physically possible neighbours.')
    ap.add_argument('--rows', default=None,
                    help='Tiles per row, comma separated, e.g. "6,10,11,15,14,14". '
                         'Row lengths differ and rows start at different x, so this '
                         'only says where rows break -- the offsets still come from '
                         'the overlap.')
    ap.add_argument('--min_ratio_row', type=float, default=1.3,
                    help='Bar for a row-strip link.')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--from_pos', default=None,
                    help='Reuse a solved _pos.csv and only re-blend.')
    ap.add_argument('--no_flatfield', action='store_true')
    args = ap.parse_args()

    fs, wl = series(args.fov_dir, args.wavelength)
    if args.limit:
        fs = fs[:args.limit]
    n = len(fs)
    print(f'{n} FOVs, series {wl}', flush=True)
    if n < 2:
        sys.exit('need at least two FOVs')

    if args.from_pos:
        rows = list(csv.DictReader(open(args.from_pos, encoding='utf-8')))
        fs = [r['file'] for r in rows]
        n = len(fs)
        pos = np.array([[float(r['y']), float(r['x'])] for r in rows])
        tile = tiff.imread(os.path.join(args.fov_dir, fs[0])).shape[0]
        print(f'reusing {args.from_pos}: {n} FOVs, tile {tile}', flush=True)
        blend(args, fs, pos, tile)
        return

    imgs = [load(os.path.join(args.fov_dir, f), args.ds) for f in fs]
    tile = imgs[0].shape[0] * args.ds
    print(f'tile {tile} px, searching links at 1/{args.ds}', flush=True)

    # 1. free search, with the zero blob excluded
    ez = int(0.25 * tile / args.ds)
    raw = []
    for i in range(n - 1):
        raw.append(xcorr(imgs[i], imgs[i + 1], exclude_zero=ez))
        if (i + 1) % 50 == 0:
            print(f'  sequential {i+1}/{n-1}', flush=True)

    # 2. learn the raster step from the confident links, then keep only links that a
    #    serpentine can actually produce: a step sideways, or a step to the next row
    conf = [(dy * args.ds, dx * args.ds) for dy, dx, pk, r in raw if r >= args.min_ratio]
    step = float(np.median([max(abs(dy), abs(dx)) for dy, dx in conf])) if conf else 750.0
    print(f'raster step {step:.0f} px ({100*(1-step/tile):.0f}% overlap)', flush=True)

    def kind(dy, dx, tol=0.15):
        """Serpentine: a pair is a step along a row or a step to the next row, never
        in between. Row-internal links measure dy ~ +-16 against a 754 px step, so a
        loose tolerance is not needed -- and is actively harmful. At 0.4*step the pair
        27->28 of 240720, measured (244, 864) and really a row change, passed as
        horizontal and pinned two rows to the same height."""
        h = 0.6 * step < abs(dx) < 1.4 * step and abs(dy) < tol * step
        v = 0.6 * step < abs(dy) < 1.4 * step and abs(dx) < tol * step
        return 'h' if h else ('v' if v else None)

    breaks = set()
    if args.rows:
        k = 0
        for c in [int(v) for v in args.rows.split(',') if v.strip()]:
            k += c
            breaks.add(k - 1)                  # link (k-1 -> k) crosses a row boundary
        print(f'row breaks given after tiles: {sorted(breaks)}', flush=True)

    links, recheck = [], []
    for i, (dy, dx, pk, r) in enumerate(raw):
        if i in breaks:                        # never treat a row change as within-row
            recheck.append(i)
            continue
        dy, dx = dy * args.ds, dx * args.ds
        if r >= args.min_ratio and kind(dy, dx):
            links.append((i, i + 1, dy, dx, r))
        else:
            recheck.append(i)
    print(f'sequential: {len(links)} accepted, {len(recheck)} to re-test', flush=True)

    # 3. whatever is left is a row change: search only where the next row can be.
    #    Constraining the search this tightly makes a weak peak usable, so the bar can
    #    be lower than for the free search without inviting false links.
    cands = [(int(round(sy * step / args.ds)), int(round(sx * step / args.ds)))
             for sy, sx in ((1, 0), (-1, 0))]
    rad = int(0.45 * step / args.ds)
    rec = 0
    for i in recheck:
        dy, dx, pk, r = xcorr_near(imgs[i], imgs[i + 1], cands, rad, exclude_zero=ez)
        dy, dx = dy * args.ds, dx * args.ds
        if r >= args.min_ratio_near and kind(dy, dx, tol=0.30):
            links.append((i, i + 1, dy, dx, max(r, 1.0)))
            rec += 1
        elif args.verbose:
            print(f'    [{i}->{i+1}] row-change search: ({dy},{dx}) r={r:.2f} rejected',
                  flush=True)
    print(f'row changes recovered: {rec}/{len(recheck)}; '
          f'{len(links)}/{n-1} sequential links', flush=True)

    # Join the row segments before anything else. A single FOV pair at a row change
    # carries little signal -- four of them on 240720 sat at ratio 0.4-1.1, which no
    # threshold separates from noise -- but two rows overlap along their whole length,
    # so their strips correlate decisively.
    #
    # Order matters: run this after the cross-link stage instead and it finds nothing to
    # do, because bad cross-links have already connected the graph horizontally and
    # stacked rows 3-5 of 240720 on top of each other.
    args._step = step
    links, row_links = link_rows(imgs, links, [], n, tile, args)
    links = links + row_links

    pos = solve(n, links) if links else np.zeros((n, 2))

    # cross-row neighbours: pairs the acquisition order never puts next to each other
    extra, tested = [], 0
    for i in range(n):
        for j in range(i + 2, n):
            d = pos[j] - pos[i]
            if abs(d[0]) < 1.2 * tile and abs(d[1]) < 1.2 * tile:
                dy, dx, pk, ratio = xcorr(imgs[i], imgs[j], exclude_zero=ez)
                tested += 1
                dy, dx = dy * args.ds, dx * args.ds
                if ratio >= args.min_ratio and kind(dy, dx):
                    extra.append((i, j, dy, dx, ratio))
    print(f'cross links: {len(extra)} kept of {tested} tested', flush=True)

    all_links = links + extra
    pos = solve(n, all_links)
    pos -= pos.min(0)

    with open(args.out + '_pos.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['index', 'file', 'y', 'x'])
        for i, f in enumerate(fs):
            w.writerow([i, f, round(float(pos[i, 0]), 1), round(float(pos[i, 1]), 1)])

    blend(args, fs, pos, tile)


if __name__ == '__main__':
    main()
