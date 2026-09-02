#!/usr/bin/env python3
"""Similarity (rotation + scale + translation) alignment of a TPAF mosaic to its H&E core.

Why beyond translation
----------------------
Translation-only placement leaves 60-93 H&E px of systematic offset. Two reasons to
expect the rest is not translational:
  * the mosaic carries its own ~1.1 degree shear from stage motion (measured: within a
    row, y drifts about -16 px per +815 px of x), so the TPAF frame is slightly rotated
    against the slide;
  * the 1.4021 scale is calibrated to 0.5%, and 0.5% of a 3743 px footprint is 19 px,
    which matters at the edges even though it is negligible at the centre.

Search
------
Coarse at 1/8 over rotation x scale, then translation refined at 1/4 around the winner.
Bounded throughout -- an unbounded mask search walks off on dim cores, which is how the
first attempt at this failed.

Masks include interior white space (lumens, clefts, adipose), not just the outer
contour: a TMA core is nearly circular, and two circles' outlines constrain rotation
and interior placement hardly at all. What actually discriminates in this data is
internal structure -- the diagonal cleft that identifies core 15C, for instance.
"""

from __future__ import annotations

import cv2
import numpy as np

SCALE = 0.621 / 0.4429


def mosaic_mask(mos):
    b = cv2.GaussianBlur(mos.astype(np.float32), (0, 0), 12)
    return (b > max(np.percentile(b, 45), b.max() * 0.06)).astype(np.float32)


def he_mask(he_gray):
    b = cv2.GaussianBlur(he_gray.astype(np.float32), (0, 0), 12)
    return (b < 212).astype(np.float32)


def _warp_mask(m, angle, scale, ds):
    """Mosaic mask at H&E scale/ds, rotated about its own centre."""
    s = SCALE * scale / ds
    small = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if abs(angle) < 1e-6:
        return small
    h, w = small.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(small, M, (nw, nh), flags=cv2.INTER_NEAREST)


def _search(mm, hm, y0, x0, rad, ds, angles, scales):
    """Best (score, angle, scale, y, x) with the template origin in full-res H&E px."""
    best = (-2.0, 0.0, 1.0, y0, x0)
    base_h = mm.shape[0] * SCALE
    base_w = mm.shape[1] * SCALE
    wy0 = max(0, int(y0 - rad)); wx0 = max(0, int(x0 - rad))
    wy1 = min(hm.shape[0], int(y0 + base_h + rad))
    wx1 = min(hm.shape[1], int(x0 + base_w + rad))
    win = cv2.resize(hm[wy0:wy1, wx0:wx1], None, fx=1 / ds, fy=1 / ds,
                     interpolation=cv2.INTER_AREA)
    for a in angles:
        for sc in scales:
            t = _warp_mask(mm, a, sc, ds)
            if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, v, _, mx = cv2.minMaxLoc(r)
            if v > best[0]:
                # rotation grows the bounding box; report the origin of the *unrotated*
                # footprint so downstream geometry stays in mosaic coordinates
                t0 = _warp_mask(mm, 0.0, sc, ds)
                oy = (t.shape[0] - t0.shape[0]) / 2.0
                ox = (t.shape[1] - t0.shape[1]) / 2.0
                best = (float(v), float(a), float(sc),
                        wy0 + (mx[1] + oy) * ds, wx0 + (mx[0] + ox) * ds)
    return best


def align(mos, he_gray, y0, x0, rad=380, max_angle=5.0, max_scale=0.012,
          lock_angle=None, lock_scale=None):
    """Returns dict with angle (deg), scale, y, x (mosaic origin in H&E px), score.

    lock_angle / lock_scale pin the similarity to slide-level constants and fit only
    translation. One slide sits at one angle on each instrument, so rotation is a
    property of the slide, not of a core; fitting it per core gives 116 free
    parameters for one physical quantity. Measured: as the confidence threshold rises
    the per-core estimates converge (std 2.43 -> 1.20 deg between all cores and those
    scoring >= 0.45), which is what a real global value contaminated by noise looks
    like. A near-circular core constrains rotation weakly, so the low-confidence
    estimates wander -- eight of them ran to the +-5 deg search bound.
    """
    mm = mosaic_mask(mos)
    hm = he_mask(he_gray)

    if lock_angle is not None and lock_scale is not None:
        best = _search(mm, hm, y0, x0, rad, 8, [lock_angle], [lock_scale])
        fine = _search(mm, hm, best[3], best[4], 96, 4, [lock_angle], [lock_scale])
        sc, a, s, y, x = fine if fine[0] > best[0] else best
        return dict(angle=a, scale=s, y=y, x=x, score=sc)

    coarse = _search(mm, hm, y0, x0, rad, 8,
                     np.arange(-max_angle, max_angle + 0.01, 1.0),
                     1.0 + np.linspace(-max_scale, max_scale, 5))
    _, a0, s0, cy, cx = coarse
    fine = _search(mm, hm, cy, cx, 96, 4,
                   np.arange(a0 - 0.75, a0 + 0.76, 0.25),
                   [s0 - max_scale / 4, s0, s0 + max_scale / 4])
    sc, a, s, y, x = fine if fine[0] > coarse[0] else coarse
    return dict(angle=a, scale=s, y=y, x=x, score=sc)


def crop_he_for_tile(he, tile_yx, origin_yx, angle, scale, mosaic_shape, side_tpaf,
                     upsample=1.0):
    """H&E for one tile, warped into the tile's own frame at TPAF scale.

    With rotation in the model an axis-aligned slice no longer corresponds to the
    tile, so the H&E is resampled through the inverse of the fitted similarity. The
    result is directly comparable to the AF tile pixel for pixel.

    tile_yx is the tile origin inside the mosaic in TPAF px, already offset by the
    redundancy margin; side_tpaf is the output side in TPAF px.
    """
    s_ = SCALE * scale
    mh, mw = mosaic_shape
    R = cv2.getRotationMatrix2D((mw * s_ / 2.0, mh * s_ / 2.0), angle, 1.0)
    A = R[:, :2] * (s_ / upsample)
    b = R[:, :2] @ np.array([tile_yx[1] * s_, tile_yx[0] * s_])         + R[:, 2] + np.array([origin_yx[1], origin_yx[0]])
    M = np.hstack([A, b.reshape(2, 1)])
    side = int(round(side_tpaf * upsample))
    return cv2.warpAffine(he, M, (side, side),
                          flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
