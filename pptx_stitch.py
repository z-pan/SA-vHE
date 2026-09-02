#!/usr/bin/env python3
"""Turn a hand-made PowerPoint tile layout into a native-resolution TPAF mosaic.

Why this and not the PPT export
-------------------------------
Exporting the slide as an image resamples and usually recompresses it. Virtual
staining needs the TPAF at its own pixel scale -- UTOM reads nuclei at the size the
204->512 BICUBIC crop gives them -- so a resampled mosaic degrades the output and is
not reproducible. A .pptx is a zip of XML plus the embedded media, and on the file
checked here (TPAF_stitch.pptx, 177 tiles) every embedded image is byte-identical to
its source TIF. So the deck can supply the layout while the pixels come from the
originals.

That split is also what the automatic stitcher was missing. Within a row it recovers
tile offsets confidently (65/69 links on 240720, dx +-750, dy +-16, peak/runner-up
2.3-9.6), but row changes are weak and ambiguous, and a wrong guess there collapses
the layout. A person arranging tiles by eye settles the topology in minutes; pairwise
correlation then refines each position to the pixel.

Usage::

    python pptx_stitch.py --pptx TPAF_stitch.pptx --fov_dir <folder of source TIFs> \\
        --out mosaic --refine
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
import tifffile as tiff

NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def read_layout(pptx, slide=1):
    """[(media_path, x_emu, y_emu, cx_emu, cy_emu)] for one slide."""
    z = zipfile.ZipFile(pptx)
    rels = {r.get('Id'): 'ppt/' + r.get('Target').replace('../', '')
            for r in ET.fromstring(z.read(f'ppt/slides/_rels/slide{slide}.xml.rels'))}
    root = ET.fromstring(z.read(f'ppt/slides/slide{slide}.xml'))
    out = []
    for p in root.findall('.//p:pic', NS):
        blip = p.find('.//a:blip', NS)
        xfrm = p.find('.//a:xfrm', NS)
        if blip is None or xfrm is None:
            continue
        off, ext = xfrm.find('a:off', NS), xfrm.find('a:ext', NS)
        if off is None or ext is None:
            continue
        out.append((rels[blip.get(REL + 'embed')], int(off.get('x')), int(off.get('y')),
                    int(ext.get('cx')), int(ext.get('cy'))))
    return z, out


def match_sources(z, layout, fov_dir):
    """media path -> source filename, by content. PowerPoint drops the original name."""
    by_size = {}
    for f in os.listdir(fov_dir):
        if f.lower().endswith(('.tif', '.tiff')):
            by_size.setdefault(os.path.getsize(os.path.join(fov_dir, f)), []).append(f)
    digest = {}
    out = {}
    for media, *_ in layout:
        if media in out:
            continue
        b = z.read(media)
        cands = by_size.get(len(b), [])
        h = hashlib.md5(b).hexdigest()
        hit = None
        for f in cands:
            p = os.path.join(fov_dir, f)
            if p not in digest:
                digest[p] = hashlib.md5(open(p, 'rb').read()).hexdigest()
            if digest[p] == h:
                hit = f
                break
        out[media] = hit
    return out


def load_gray(path, highpass=True, sigma=12.0):
    a = tiff.imread(path).astype(np.float32)
    a = a[..., :2].mean(2) if a.ndim == 3 else a
    return a - cv2.GaussianBlur(a, (0, 0), sigma) if highpass else a


def refine(pos, imgs, tile, rad=120, ds=2, min_score=0.25):
    """Nudge each tile by correlating it against overlapping neighbours.

    The deck fixes the arrangement; this fixes the placement. Bounded by `rad`, so a
    weak pair cannot move a tile far -- the layout is already right, only imprecise.
    """
    n = len(pos)
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            dy, dx = pos[j] - pos[i]
            if abs(dy) >= tile * 0.95 or abs(dx) >= tile * 0.95:
                continue                      # no usable overlap
            a = cv2.resize(imgs[i], None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
            b = cv2.resize(imgs[j], None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
            # matchTemplate over the shared window. The brute nested grid this
            # replaces evaluated (2*rad/ds/2+1)^2 correlations per pair -- ~3700 on
            # the default radius, times ~500 pairs. Same answer, seconds not hours.
            oy, ox = int(round(dy / ds)), int(round(dx / ds))
            r = rad // ds
            h, w = a.shape
            ay0, by0 = max(0, oy), max(0, -oy)
            ax0, bx0 = max(0, ox), max(0, -ox)
            hh, ww = h - abs(oy), w - abs(ox)
            if hh < r * 2 + 8 or ww < r * 2 + 8:
                continue
            A = a[ay0:ay0 + hh, ax0:ax0 + ww]
            Bq = b[by0 + r:by0 + hh - r, bx0 + r:bx0 + ww - r]
            if A.std() < 1e-6 or Bq.std() < 1e-6:
                continue
            cc = cv2.matchTemplate(A, Bq, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(cc)
            best = (float(mx), (oy + loc[1] - r) * ds, (ox + loc[0] - r) * ds)
            if best[0] >= min_score:
                links.append((i, j, best[1], best[2], best[0]))
    print(f'  refine: {len(links)} overlapping pairs correlated', flush=True)
    if not links:
        return pos
    new = np.zeros_like(pos, dtype=float)
    for axis in (0, 1):
        rows, rhs, wts = [], [], []
        for i, j, dy, dx, w in links:
            r = np.zeros(n); r[j] = 1; r[i] = -1
            rows.append(r); rhs.append(dy if axis == 0 else dx); wts.append(w)
        for i in range(n):                     # keep the deck's absolute placement
            r = np.zeros(n); r[i] = 1
            rows.append(r); rhs.append(float(pos[i, axis])); wts.append(0.02)
        A = np.array(rows) * np.sqrt(np.array(wts))[:, None]
        b = np.array(rhs) * np.sqrt(np.array(wts))
        new[:, axis] = np.linalg.lstsq(A, b, rcond=None)[0]
    shift = np.hypot(*(new - pos).T)
    print(f'  refine: moved median {np.median(shift):.0f} px, max {shift.max():.0f} px',
          flush=True)
    return new


def pipeline_gray(a):
    """The grey the staining pipeline will actually see.

    path_vhe_stain.py reads the mosaic with cv2 and calls COLOR_BGR2GRAY, i.e.
    0.299R + 0.587G + 0.114B, and the reference level of 45 was measured that way on
    the accepted patches. Measuring here with (R+G)/2 instead -- the weighting used
    for the display mosaic -- normalises to a level in a different space and lands the
    input somewhere the model was not calibrated for.
    """
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def vignette_profile(loader, n, tile, sample=60, thresh=8.0, sigma=60.0):
    """Shape of the illumination falloff inside a FOV, with the level divided out.

    The original --no_flatfield path took a median over raw tiles, so the profile
    carried the brightest tiles' level as well as the falloff; dividing by it pushed
    values to 649 on a uint8 source. Normalising each tile by its own tissue median
    first leaves only the shape, and the correction then spans 0.79-1.30.

    Why it matters even with nearest-blending: placements are irregular, so a seam can
    put a pixel 150 px from one tile's centre against a pixel 450 px from another's.
    Measured on 240703 the profile runs 1.26 at the centre to 0.77 at the rim, 1.63x,
    and the generator turns a 1.6x input difference into roughly 33% more haematoxylin.
    No per-tile scalar can reach this: the correction depends on where in the FOV the
    pixel sat, not on which FOV it came from.
    """
    st = []
    step = max(1, n // sample)
    for i in range(0, n, step):
        a = loader(i)
        g = a[..., :2].mean(2) if a.ndim == 3 else a
        m = g > thresh
        if m.sum() < 2000:
            continue
        med = float(np.median(g[m]))
        if med > 2:
            st.append(g / med)
        if len(st) >= sample:
            break
    if len(st) < 8:
        print('  devignette: too few tiles to estimate a profile; skipped', flush=True)
        return None
    prof = np.median(np.stack(st), 0)
    prof = cv2.GaussianBlur(prof, (0, 0), sigma)
    prof /= max(float(np.median(prof)), 1e-6)
    t = prof.shape[0]
    yy, xx = np.mgrid[0:t, 0:t]
    r = np.maximum(np.abs(yy - t / 2), np.abs(xx - t / 2))
    c, e = float(prof[r < 100].mean()), float(prof[r >= 400].mean())
    print(f'  devignette: profile from {len(st)} tiles, centre {c:.2f} / rim {e:.2f} '
          f'= {c / max(e, 1e-6):.2f}x, correction {1/prof.max():.2f}..{1/prof.min():.2f}',
          flush=True)
    return prof.astype(np.float32)


def solve_gains(loader, layout, pos, tile, nch, min_px=5000, thresh=8.0,
                ridge=0.05):
    """One brightness gain per tile per channel, fitted on the overlaps.

    Measured on 240703: neighbouring tiles disagree in brightness by 37% at the median
    and 97% at p90, and the gains span 0.25 to 3.72 across the slide -- fifteenfold.
    That is the seam. Solving one scalar per tile per channel takes the disagreement
    to 5% median / 15% p90, so the gain accounts for about six sevenths of it; the
    rest correlates weakly with radius in the FOV (r = -0.28), i.e. residual
    vignetting, which is left alone here.

    The ratio is taken between medians of the shared tissue pixels, not by regressing
    one tile on the other. Least squares on two noisy measurements of the same signal
    attenuates the slope -- the first attempt returned a median slope of 0.5 with a +28
    intercept, which cannot hold for a relation that is symmetric in i and j.

    A gain is a multiply. It cannot move an edge, blur, or invent texture, so applying
    it costs nothing in resolution or structural fidelity.
    """
    n = len(layout)
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            dy, dx = pos[j] - pos[i]
            if abs(dy) >= tile - 200 or abs(dx) >= tile - 200:
                continue
            dyi, dxi = int(round(dy)), int(round(dx))
            ay0, by0 = max(0, dyi), max(0, -dyi)
            ax0, bx0 = max(0, dxi), max(0, -dxi)
            hh, ww = tile - abs(dyi), tile - abs(dxi)
            A, Bm = loader(i), loader(j)
            A = A[ay0:ay0 + hh, ax0:ax0 + ww]
            Bm = Bm[by0:by0 + hh, bx0:bx0 + ww]
            fa = A[..., :2].mean(2) if A.ndim == 3 else A
            fb = Bm[..., :2].mean(2) if Bm.ndim == 3 else Bm
            m = (fa > thresh) & (fb > thresh)
            if m.sum() < min_px:
                continue
            for c in range(nch):
                a = float(np.median(A[..., c][m] if A.ndim == 3 else A[m]))
                b = float(np.median(Bm[..., c][m] if Bm.ndim == 3 else Bm[m]))
                if a > 2 and b > 2:
                    links.append((c, i, j, np.log(b / a)))
    if not links:
        print('  gain: no usable overlaps; leaving tiles as acquired', flush=True)
        return np.ones((n, nch), np.float32)
    gains = np.ones((n, nch), np.float64)
    for c in range(nch):
        sub = [l for l in links if l[0] == c]
        if len(sub) < 2:
            continue
        # Ridge, not just a mean anchor. Links constrain only differences, so on a
        # sparsely connected graph the solution drifts along chains: without this the
        # 240703 gains spanned 56x while the tiles themselves span 3.9x (p5-p95) and
        # 9x end to end. Pulling each log-gain toward zero bounds the drift without
        # flattening the real differences, the same trick the position refine uses.
        m = len(sub)
        A = np.zeros((m + n, n))
        y = np.zeros(m + n)
        for k, (_, i, j, lr) in enumerate(sub):
            # After correction the two tiles must agree: g_i*a == g_j*b, so
            # log g_j - log g_i = log(a/b) = -log(b/a). Writing +log(b/a) here
            # solves for gains that double the discrepancy instead of removing it,
            # while the residual it reports still falls -- the objective is
            # self-consistent, only the sign of what it means is wrong. The
            # corr(log gain, -log brightness) check below is what catches it.
            A[k, j] = 1; A[k, i] = -1; y[k] = -lr
        for i in range(n):
            A[m + i, i] = ridge
        g, *_ = np.linalg.lstsq(A, y, rcond=None)
        g -= g.mean()
        before = np.abs(y[:m]); after = np.abs(y[:m] - A[:m] @ g)
        print(f'  gain ch{c}: pairwise |log ratio| p50 {np.percentile(before,50):.3f}'
              f' -> {np.percentile(after,50):.3f}, p90 {np.percentile(before,90):.3f}'
              f' -> {np.percentile(after,90):.3f}  '
              f'(gain {np.exp(g.min()):.2f}..{np.exp(g.max()):.2f})', flush=True)
        gains[:, c] = np.exp(g)
        try:
            base = np.array([np.median(loader(i)[..., c][
                (loader(i)[..., :2].mean(2) if loader(i).ndim == 3 else loader(i)) > thresh])
                for i in range(n)], float)
            ok = np.isfinite(base) & (base > 0)
            if ok.sum() > 5:
                print(f'          check: corr(log gain, -log tile brightness) = '
                      f'{np.corrcoef(g[ok], -np.log(base[ok]))[0,1]:+.2f} '
                      f'(should be strongly positive)', flush=True)
        except Exception:
            pass
    return gains.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pptx', required=True)
    ap.add_argument('--fov_dir', required=True, help='Folder with the source TIFs.')
    ap.add_argument('--out', required=True, help='Prefix for _mosaic.tif/.png/_pos.csv.')
    ap.add_argument('--slide', type=int, default=1)
    ap.add_argument('--refine', action='store_true')
    ap.add_argument('--refine_rad', type=int, default=120)
    ap.add_argument('--out_scale', type=float, default=0.25)
    ap.add_argument('--no_flatfield', action='store_true')
    ap.add_argument('--rgb', action='store_true',
                    help='Keep the original channels and write uint8 without '
                         'compression, so the mosaic can be fed to the staining '
                         'pipeline. The default single-channel output averages the '
                         'two detector channels, which (a) throws away what the '
                         '3-channel arms need and (b) is not the grey the pipeline '
                         'uses -- path_vhe_stain.py takes cv2 BGR2GRAY, 0.299R + '
                         '0.587G, not (R+G)/2. Implies --no_flatfield unless '
                         '--flatfield_anyway: dividing by the profile pushes values '
                         'past 255 (measured max 649 on 240703) and no other stage '
                         'of this pipeline flat-fields.')
    ap.add_argument('--devignette', action='store_true',
                    help='Divide out the illumination falloff inside each FOV. '
                         'Measured 1.63x centre to rim on 240703; because '
                         '--blend nearest hands out irregular territories, that '
                         'falloff lands on both sides of a seam at different radii and '
                         'no per-tile scalar can remove it.')
    ap.add_argument('--norm_target', type=float, default=0.0,
                    help='Tissue median every FOV is scaled to, in grey levels. The '
                         'generator is not scale invariant, so this has a right '
                         'answer: 45, the tissue median of the 139 accepted '
                         '148-region patches (p10 28, p90 66). Normalising to the '
                         "slide's own median instead put the mosaic at 22 and the "
                         'staining came out uniformly pale. 0 = use the slide median.')
    ap.add_argument('--norm_tile', choices=('none', 'median'), default='none',
                    help='"median" scales every FOV so its tissue median matches the '
                         'slide median, before placement. Unlike --equalize this needs '
                         'no pairwise graph, so it cannot drift along chains -- the '
                         'graph solve produced gains spanning 21x against a true tile '
                         'spread of 9x. It is the blunter instrument: it also removes '
                         'whatever real autofluorescence differences exist between '
                         'FOVs. That trade is worth making here because the generator '
                         'is strongly sensitive to input level -- 2.67x on the input '
                         'moves output haematoxylin concentration by 81%% -- so an '
                         'uncorrected acquisition drift becomes a staining difference '
                         'across every seam, which is exactly the artefact a MIL model '
                         'would learn instead of biology.')
    ap.add_argument('--equalize', action='store_true',
                    help='Fit one brightness gain per tile per channel on the '
                         'overlaps and apply it. LEAVE THIS OFF -- kept only so the '
                         'measurement below is reproducible. The neighbouring tiles '
                         'do disagree in brightness (97%% at p90 on 240703), but the '
                         'disagreement is edge-localised: 28%% where the shared region '
                         'sits 300-400 px from the tile centres, 47%% at 400-512 px, '
                         'r=+0.37 with radius. That is vignetting, not a whole-tile '
                         'gain -- and because ~50%% overlap puts every shared region '
                         'near the edges, no measurement of a whole-tile gain is even '
                         'identifiable from this data. Meanwhile --blend nearest hands '
                         'each pixel to the tile whose centre is closest, so only the '
                         'central ~500x500 cell of each tile is ever used, inside the '
                         'radius where vignetting is mild. Fitting a scalar on the '
                         'edges and multiplying it into those centres imports an error '
                         'that was not there. Measured on the uncorrected nearest '
                         'mosaic: brightness across a seam steps 76%% over 20 px, '
                         'against 70%% for the same span of ordinary tissue away from '
                         'any seam -- the seam is already at tissue-noise level, so '
                         'there is nothing left to correct. If the residual ever does '
                         'need attention, the right model is a shared vignetting '
                         'field, not a per-tile scalar.')
    ap.add_argument('--gain_ridge', type=float, default=0.05,
                    help='Pull each tile log-gain toward zero. Links constrain only '
                         'differences, so an unregularised solve drifts along chains: '
                         'at ridge 0 the 240703 gains spanned 56x against a true tile '
                         'spread of 3.9x (p5-p95). Raise it if the gains still look '
                         'wider than the tiles themselves.')
    ap.add_argument('--clip_pct', type=float, default=0.1,
                    help='After equalising, scale every gain by one constant chosen so '
                         'this percent of tissue pixels clips at the uint8 ceiling. '
                         'One global constant, so the relative correction between '
                         'tiles -- the part that removes the seam -- is unchanged. '
                         'Lower is safer but darker; 0.1 keeps the dynamic range.')
    ap.add_argument('--drop_unmatched', action='store_true',
                    help='Leave out tiles whose bytes match no source TIF. Such a '
                         'tile was pasted into the deck rather than inserted from '
                         'file, so PowerPoint re-encoded it: on 240906 the one such '
                         'tile is RGBA where every other is RGB, and is displayed at '
                         'a different EMU size, meaning it would also be placed at '
                         'the wrong pixel scale. Neither its pixels nor its geometry '
                         'are the acquisition. One tile of 72 is not worth the '
                         'silent error.')
    ap.add_argument('--flatfield_anyway', action='store_true',
                    help='Apply flat-field even in --rgb mode. Off by default; see '
                         '--rgb.')
    ap.add_argument('--blend', choices=('mean', 'feather', 'nearest'),
                    default='feather',
                    help='How overlapping tiles combine. "mean" is a plain average '
                         'and doubles every edge the placement gets wrong -- the '
                         'hand-made deck positions are off by a median 13.6 px on '
                         '240703 (p90 32.5), which at 0.621 um/px is 8.4 um, larger '
                         'than a nucleus, so the ghosting is nuclei printed twice. '
                         '"feather" weights each tile down toward its own edges, '
                         'which hides the seam but does NOT fix misplacement: run '
                         '--refine as well. "nearest" gives every output pixel to '
                         'exactly one tile -- the one whose centre is closest -- so '
                         'no two copies are ever summed and ghosting is impossible '
                         'by construction, at the price of an intensity step at the '
                         'boundary. Measurement says to prefer it for quantitative '
                         'work: the tiles are not deforming (block spread 0.65 px, '
                         'rotation 0.056 deg, scale 0.20%% -- all under a pixel or '
                         'two, against a 5-10 px nucleus), so the residual is a '
                         'placement problem, and averaging turns a placement problem '
                         'into a doubled nucleus over the 51%% of the image that '
                         'overlaps. A seam is a thin line at a known location; '
                         '_seam.png records it so those pixels can be excluded.')
    args = ap.parse_args()
    if args.rgb and not args.flatfield_anyway:
        args.no_flatfield = True

    z, layout = read_layout(args.pptx, args.slide)
    print(f'{len(layout)} tiles on slide {args.slide}', flush=True)
    src = match_sources(z, layout, args.fov_dir)
    missing = [m for m, *_ in layout if src.get(m) is None]
    if missing:
        print(f'[warn] {len(missing)} tiles not matched to a source file; '
              f'their pixels come from the deck instead', flush=True)

    cx = {c for _, _, _, c, _ in layout}
    if len(cx) > 1:
        print(f'[warn] tiles are displayed at {len(cx)} different sizes {sorted(cx)[:4]}; '
              f'assuming the most common one', flush=True)
    disp = max(cx, key=lambda v: sum(1 for _, _, _, c, _ in layout if c == v))

    first = src[layout[0][0]]
    tile = tiff.imread(os.path.join(args.fov_dir, first)).shape[0] if first else \
        read_media_bytes(z.read(layout[0][0])).shape[0]
    px_per_emu = tile / disp
    print(f'tile {tile} px shown at {disp} EMU -> {px_per_emu:.6f} px/EMU', flush=True)

    pos = np.array([[y * px_per_emu, x * px_per_emu] for _, x, y, _, _ in layout])
    pos -= pos.min(0)

    paths = []
    for media, *_ in layout:
        f = src.get(media)
        paths.append(os.path.join(args.fov_dir, f) if f else None)

    if args.drop_unmatched and any(p is None for p in paths):
        keep = [i for i, p in enumerate(paths) if p is not None]
        print(f'dropping {len(paths) - len(keep)} unmatched tile(s); '
              f'{len(keep)} remain', flush=True)
        layout = [layout[i] for i in keep]
        paths = [paths[i] for i in keep]
        pos = pos[keep]
    imgs = [load_gray(p) if p else load_gray_bytes(z.read(m))
            for (m, *_), p in zip(layout, paths)]

    n_embed = sum(1 for p in paths if p is None)
    if n_embed:
        print(f'[warn] {n_embed}/{len(paths)} tiles have no matching source TIF and '
              f'come from the deck embedded copy -- those pixels are whatever '
              f'PowerPoint stored, not the original acquisition', flush=True)

    if args.refine:
        pos = refine(pos, imgs, tile, rad=args.refine_rad)
        pos -= pos.min(0)

    with open(args.out + '_pos.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['index', 'media', 'source', 'y', 'x'])
        for i, ((m, *_), p) in enumerate(zip(layout, paths)):
            w.writerow([i, m, os.path.basename(p) if p else '', round(pos[i, 0], 1),
                        round(pos[i, 1], 1)])

    s = args.out_scale
    H = int(np.ceil((pos[:, 0].max() + tile) * s)) + 2
    W = int(np.ceil((pos[:, 1].max() + tile) * s)) + 2
    print(f'mosaic {W}x{H} at scale {s} (full res {int((pos[:,1].max()+tile))}x'
          f'{int((pos[:,0].max()+tile))})', flush=True)

    prof = None
    if not args.no_flatfield:
        st = []
        for p, (m, *_) in list(zip(paths, layout))[::max(1, len(layout) // 40)][:40]:
            a = tiff.imread(p) if p else read_media_bytes(z.read(m))
            a = a.astype(np.float32)
            a = a[..., :2].mean(2) if a.ndim == 3 else a
            st.append(cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        prof = np.median(np.stack(st), 0)
        prof = cv2.GaussianBlur(prof, (0, 0), max(prof.shape) / 12.0)
        prof /= max(float(np.median(prof)), 1e-6)

    def tile_weight(h, w, edge=64):
        """Ramp to zero at the tile border, so overlapping tiles cross-fade.

        A plain average treats a tile's outermost pixel as equal evidence to a
        neighbour's centre; where the placement is off, both copies print at full
        strength and the result is a doubled edge. Feathering does not correct the
        placement -- only --refine does -- it stops a misplaced border from carrying
        the same weight as a well-supported centre.
        """
        ry = np.minimum(np.arange(h), h - 1 - np.arange(h)).astype(np.float32)
        rx = np.minimum(np.arange(w), w - 1 - np.arange(w)).astype(np.float32)
        return (np.minimum(ry / max(edge, 1), 1.0)[:, None]
                * np.minimum(rx / max(edge, 1), 1.0)[None, :] + 1e-3)

    def raw_tile(i):
        return (tiff.imread(paths[i]) if paths[i]
                else read_media_bytes(z.read(layout[i][0]))).astype(np.float32)

    nch = 0
    if args.rgb:
        a0 = tiff.imread(paths[0]) if paths[0] else read_media_bytes(z.read(layout[0][0]))
        nch = a0.shape[2] if a0.ndim == 3 else 1
        print(f'rgb mode: {nch} channels, uint8, uncompressed, '
              f'flat-field {"on" if prof is not None else "off"}, blend={args.blend}',
              flush=True)

    vig = None
    if args.devignette:
        _vc = {}
        def _vload(i):
            if i not in _vc:
                if len(_vc) > 40:
                    _vc.clear()
                _vc[i] = raw_tile(i)
            return _vc[i]
        vig = vignette_profile(_vload, len(layout), tile)

    def corrected(i):
        a = raw_tile(i)
        if vig is not None:
            v = vig[:a.shape[0], :a.shape[1]]
            a = a / (v[..., None] if a.ndim == 3 else v)
        return a

    gains = None
    if args.norm_tile == 'median':
        med = []
        for i in range(len(layout)):
            a = corrected(i)
            f = pipeline_gray(a)
            m = f > 8
            med.append(float(np.median(f[m])) if m.sum() > 2000 else np.nan)
        med = np.array(med, float)
        ok = np.isfinite(med) & (med > 0)
        target = args.norm_target if args.norm_target > 0 else float(np.median(med[ok]))
        gains = np.ones((len(layout), max(nch, 1)), np.float32)
        gains[ok] = (target / med[ok])[:, None]
        print(f'  norm_tile median: target {target:.1f}, tile medians '
              f'{np.nanmin(med):.0f}..{np.nanmax(med):.0f}, gains '
              f'{gains[ok].min():.2f}..{gains[ok].max():.2f}', flush=True)
        smp = []
        rng0 = np.random.default_rng(0)
        for i in range(len(layout)):
            a = corrected(i)
            v = (a.reshape(-1, a.shape[2]) if a.ndim == 3 else a.reshape(-1, 1))
            idx = rng0.choice(len(v), min(len(v), 20000), replace=False)
            smp.append(v[idx] * gains[i])
        smp = np.concatenate(smp)
        ref = smp[smp.max(1) > 8]
        clip = 100.0 * float((ref > 255).mean())
        if args.norm_target > 0:
            # An explicit target is the level the model expects. Rescaling the whole
            # slide to avoid clipping would move it off that level again -- which is
            # exactly what darkening by 0.478 did last time -- so clip instead and say
            # how much.
            print(f'  norm_tile: held at target {target:.0f}, {clip:.2f}% of tissue '
                  f'samples clip', flush=True)
        else:
            k = min(1.0, 250.0 / max(float(np.percentile(ref, 100.0 - args.clip_pct)), 1e-6))
            gains = gains * k
            print(f'  norm_tile: headroom scale {k:.3f} '
                  f'(final {gains.min():.2f}..{gains.max():.2f})', flush=True)

    elif args.equalize:
        _gc = {}
        def _load(i):
            if i not in _gc:
                if len(_gc) > 40:
                    _gc.clear()
                _gc[i] = raw_tile(i)
            return _gc[i]
        gains = solve_gains(_load, layout, pos, tile, max(nch, 1),
                            ridge=args.gain_ridge)
        # One global constant on top, so the tiles that need a large gain do not
        # clip against the uint8 ceiling. Scaling every gain by the same k leaves the
        # relative correction -- the thing that removes the seam -- untouched.
        #
        # Pick k from the actual pixel distribution, not from a percentile of
        # per-tile percentiles: the latter is set by whichever tile needed the biggest
        # gain and darkened the whole slide fivefold on 240703, throwing away dynamic
        # range to protect a handful of pixels.
        smp = []
        rng0 = np.random.default_rng(0)
        for i in range(len(layout)):
            a = raw_tile(i)
            v = (a.reshape(-1, a.shape[2]) if a.ndim == 3 else a.reshape(-1, 1))
            idx = rng0.choice(len(v), min(len(v), 20000), replace=False)
            smp.append(v[idx] * gains[i])
        smp = np.concatenate(smp)
        tis = smp[smp.max(1) > 8] if len(smp) else smp
        ref = tis if len(tis) > 1000 else smp
        k = 250.0 / max(float(np.percentile(ref, 100.0 - args.clip_pct)), 1e-6)
        gains = gains * k
        clipped = 100.0 * float((ref * k > 255).mean())
        print(f'  gain: global scale {k:.3f} (range {gains.min():.3f}..{gains.max():.3f}), '
              f'{clipped:.2f}% of tissue samples clip', flush=True)

    shape = (H, W, nch) if nch > 1 else (H, W)
    acc = np.zeros(shape, np.float32)
    cnt = np.zeros((H, W), np.float32)
    best = np.full((H, W), np.inf, np.float32)      # nearest-mode: distance to owner
    owner = np.full((H, W), -1, np.int32)
    for i, (p, (m, *_)) in enumerate(zip(paths, layout)):
        a = corrected(i) if (vig is not None) else (
            tiff.imread(p) if p else read_media_bytes(z.read(m))).astype(np.float32)
        a = a.astype(np.float32)
        if not args.rgb:
            a = a[..., :2].mean(2) if a.ndim == 3 else a
        elif nch > 1:
            if a.ndim == 2:
                a = np.repeat(a[..., None], nch, 2)
            elif a.shape[2] != nch:
                # An RGBA tile among RGB ones. Alpha is not acquisition data.
                a = (a[..., :nch] if a.shape[2] > nch else
                     np.pad(a, ((0, 0), (0, 0), (0, nch - a.shape[2]))))
        if s != 1.0:
            a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if prof is not None:
            pr = prof[:a.shape[0], :a.shape[1]]
            a = a / np.maximum(pr[..., None] if a.ndim == 3 else pr, 0.15)
        if gains is not None:
            a = a * (gains[i] if a.ndim == 3 else gains[i, 0])
        y, x = int(round(pos[i, 0] * s)), int(round(pos[i, 1] * s))
        sl = (slice(y, y + a.shape[0]), slice(x, x + a.shape[1]))
        if args.blend == 'nearest':
            hh, ww = a.shape[:2]
            dy = np.abs(np.arange(hh) - (hh - 1) / 2).astype(np.float32)
            dx = np.abs(np.arange(ww) - (ww - 1) / 2).astype(np.float32)
            dist = np.maximum(dy[:, None], dx[None, :])   # Chebyshev: square cells
            take = dist < best[sl]
            best[sl] = np.where(take, dist, best[sl])
            acc[sl] = np.where(take[..., None] if acc.ndim == 3 else take,
                               a, acc[sl])
            cnt[sl] = np.where(take, 1.0, cnt[sl])
            owner[sl] = np.where(take, i, owner[sl])
        else:
            wgt = (tile_weight(a.shape[0], a.shape[1]) if args.blend == 'feather'
                   else np.ones(a.shape[:2], np.float32))
            acc[sl] += a * (wgt[..., None] if a.ndim == 3 else wgt)
            cnt[sl] += wgt
    if args.blend == 'nearest':
        mos = acc
        # a seam is where the owning tile changes: detect it on the ownership map
        own_edge = cv2.morphologyEx(owner.astype(np.float32), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8)) > 0
        cv2.imwrite(args.out + '_seam.png', (own_edge * 255).astype(np.uint8))
        print(f'  seam map -> {args.out}_seam.png '
              f'({100*float(own_edge.mean()):.2f}% of pixels lie on a tile boundary)',
              flush=True)
    else:
        mos = acc / np.maximum(cnt[..., None] if acc.ndim == 3 else cnt, 1e-6)

    if args.norm_target > 0:
        # Per-FOV normalisation sets each FOV's own median, but the mosaic only keeps
        # each FOV's central cell, so the assembled median lands somewhere else -- 34
        # against a target of 45 on 240703. The level that matters is the one the
        # generator sees, which is the mosaic's, so finish on it.
        gm = pipeline_gray(mos)
        tm = gm > 8
        if tm.sum() > 10000:
            cur = float(np.median(gm[tm]))
            adj = args.norm_target / max(cur, 1e-6)
            mos = mos * adj
            gm2 = pipeline_gray(mos)
            print(f'  norm_target: mosaic median {cur:.1f} -> '
                  f'{float(np.median(gm2[tm])):.1f} (x{adj:.2f}), '
                  f'{100*float((mos[tm] > 255).mean() if mos.ndim == 2 else (mos.max(2)[tm] > 255).mean()):.2f}% clip',
                  flush=True)

    if args.rgb:
        # uint8 and uncompressed: same dtype and channel layout as the source FOVs, so
        # cv2.imread + COLOR_BGR2GRAY in path_vhe_stain.py yields exactly the grey the
        # 148-region run used. Writing uint16 or averaging channels here would move the
        # input distribution without any error being raised anywhere downstream.
        tiff.imwrite(args.out + '_mosaic.tif',
                     np.clip(mos, 0, 255).astype(np.uint8),
                     photometric='rgb' if nch == 3 else 'minisblack')
    else:
        tiff.imwrite(args.out + '_mosaic.tif', np.clip(mos, 0, 65535).astype(np.uint16),
                     compression='zlib')
    view = mos.mean(2) if mos.ndim == 3 else mos
    lo, hi = np.percentile(view[cnt > 0], (1, 99.5)) if (cnt > 0).any() else (0, 1)
    cv2.imwrite(args.out + '_mosaic.png',
                np.clip((view - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))
    print(f'covered {100*float((cnt>0).mean()):.0f}% of the bounding box; '
          f'{int((cnt>1).sum())/max(int((cnt>0).sum()),1)*100:.0f}% overlapped')
    print(f'-> {args.out}_mosaic.tif / .png / _pos.csv')


def read_media_bytes(b):
    """Embedded media are normally byte-identical TIFs, but a deck can also hold a
    PNG -- one tile in the 240906 deck was pasted rather than inserted from file.
    tifffile raises TiffFileError on those, so fall back to cv2.

    A PNG tile is NOT the original pixels: PowerPoint re-encoded it, and it may have
    been resampled. Those tiles break the native-resolution guarantee this script
    exists to keep, so main() counts and reports them.
    """
    try:
        return tiff.imread(io.BytesIO(b))
    except Exception:
        a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise
        return a[..., ::-1] if a.ndim == 3 else a


def load_gray_bytes(b):
    a = read_media_bytes(b).astype(np.float32)
    a = a[..., :2].mean(2) if a.ndim == 3 else a
    return a - cv2.GaussianBlur(a, (0, 0), 12.0)


if __name__ == '__main__':
    main()
