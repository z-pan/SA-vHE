#!/usr/bin/env python3
"""Side-by-side page for judging the retuned nucleus mask by eye.

The numbers say the old mask found 0.0277 of the crop against real H&E's 0.0872, with
64 of 148 regions under a quarter and several at exactly zero. They do not say whether
the new mask found the nuclei or merely found more things, which is what this page is
for.

Each FOV gets the whole 1024x1024 patch with the candidate crop outlined, so the mask
can be judged against tissue the region selection did not choose as well as tissue it
did, and then the crop itself at full resolution beside the real H&E of the same field.
Contours rather than filled overlay: a filled mask hides the nucleus it claims, and the
question here is whether the outline sits on a nucleus.

FOVs are chosen to span the failure, not to flatter it -- the ones where the old mask
already worked, the ones where it returned nothing, and the middle.

    python path_vhe_maskpage.py
    python path_vhe_maskpage.py --fov 240703_24 --fov 240729_11
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os

import cv2
import numpy as np

SUR = 'results/path_screen/survey/_vhe'
CP = 'results/path_screen/survey/_downstream/cp'
OLD, NEW = 'masks', 'masks_v2'
# Green for the old mask, magenta for the new: distinguishable in the common forms of
# colour blindness, and neither is a colour the TPAF grayscale or H&E can produce.
C_OLD, C_NEW = (0, 220, 0), (230, 0, 200)


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def b64(im, quality=88, maxw=1400):
    if im.shape[1] > maxw:
        f = maxw / im.shape[1]
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise SystemExit('encode failed')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def outline(bgr, mask, colour, thick=1):
    """Draw mask boundaries. cv2.findContours on a 0/1 image gives one contour per
    connected component, which is what a nucleus is here."""
    if mask is None or not mask.any():
        return bgr
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cs, -1, colour, thick)
    return bgr


def cov(mask, box=None):
    if mask is None:
        return float('nan')
    if box:
        x, y, w, h = box
        mask = mask[y:y + h, x:x + w]
    return float(mask.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=SUR + '/vhe_manifest.csv')
    ap.add_argument('--old', default=SUR + '/' + OLD)
    ap.add_argument('--new', default=SUR + '/' + NEW)
    ap.add_argument('--out', default='results/path_screen/survey/_vhe/mask_compare.html')
    ap.add_argument('--fov', action='append', default=None,
                    help='Patch names to show; default picks across the failure range.')
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    by_patch = {}
    for r in rows:
        by_patch.setdefault(os.path.splitext(r['stage_name'])[0], []).append(r)

    # Rank by how badly the old mask did against the same region's real H&E, then take
    # from both ends and the middle. Ranking on the old mask alone would pick regions
    # with little tissue rather than regions where the mask failed.
    score = []
    for name, rs in by_patch.items():
        r = rs[0]
        mp = os.path.join(args.old, name + '.png')
        hp = os.path.join(CP, 'real_HE', r['id'] + '.png')
        np_ = os.path.join(args.new, name + '.png')
        if not (os.path.exists(mp) and os.path.exists(hp) and os.path.exists(np_)):
            continue
        box = tuple(int(r[k]) for k in ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
        old = cov(imread_u(mp, cv2.IMREAD_GRAYSCALE) > 127, box)
        he = cov(imread_u(hp) > 0)
        if he <= 0:
            continue
        score.append((old / he, name, r, box, old, he))
    score.sort()
    if args.fov:
        pick = [s for s in score if s[1] in set(args.fov)]
    else:
        k = args.n
        idx = sorted({int(round(i * (len(score) - 1) / max(k - 1, 1)))
                      for i in range(k)})
        pick = [score[i] for i in idx]
    print('%d patches with both masks; showing %d' % (len(score), len(pick)))

    cards = []
    for ratio, name, r, box, old_c, he_c in pick:
        x, y, w, h = box
        g = cv2.cvtColor(imread_u(r['patch_path']), cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        mo = imread_u(os.path.join(args.old, name + '.png'),
                      cv2.IMREAD_GRAYSCALE) > 127
        mn = imread_u(os.path.join(args.new, name + '.png'),
                      cv2.IMREAD_GRAYSCALE) > 127
        new_c = cov(mn, box)

        full = base.copy()
        outline(full, mo, C_OLD, 1)
        outline(full, mn, C_NEW, 1)
        cv2.rectangle(full, (x, y), (x + w, y + h), (0, 200, 255), 3)

        crop = base[y:y + h, x:x + w]
        c_old = outline(crop.copy(), mo[y:y + h, x:x + w], C_OLD, 1)
        c_new = outline(crop.copy(), mn[y:y + h, x:x + w], C_NEW, 1)
        he_img = imread_u(r['he_path'])
        he_lab = imread_u(os.path.join(CP, 'real_HE', r['id'] + '.png'))
        he_vis = he_img.copy()
        if he_lab is not None:
            outline(he_vis, he_lab > 0, (0, 0, 255), 1)

        cards.append(dict(
            name=name, rid=r['id'], sample=r['sample'], ratio=ratio,
            old=old_c, new=new_c, he_cov=he_c,
            full=b64(full), c_old=b64(c_old), c_new=b64(c_new),
            he=b64(he_vis)))
        print('  %-16s old %.4f  new %.4f  real %.4f' % (name, old_c, new_c, he_c))

    html = [HEAD]
    for c in cards:
        html.append(CARD.format(
            name=c['name'], rid=c['rid'], sample=c['sample'],
            old=c['old'], new=c['new'], he_cov=c['he_cov'],
            r_old=c['old'] / c['he_cov'], r_new=c['new'] / c['he_cov'],
            full=c['full'], c_old=c['c_old'], c_new=c['c_new'], he=c['he']))
    html.append('</body>')
    with io.open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print('\n-> ' + os.path.abspath(args.out))


HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Nucleus mask, before and after retuning</title>
<style>
 body{background:#14161a;color:#dfe3e8;font:14px/1.55 -apple-system,"Segoe UI",sans-serif;
      margin:0;padding:28px}
 h1{font-size:20px;margin:0 0 6px} h2{font-size:16px;margin:0}
 .lede{color:#9aa4b0;max-width:70ch;margin:0 0 26px}
 .card{border:1px solid #2a2f38;border-radius:8px;padding:16px;margin:0 0 22px;
       background:#181b21}
 .hdr{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}
 .nums{color:#9aa4b0;font:12px ui-monospace,Consolas,monospace}
 .nums b{color:#dfe3e8;font-weight:600}
 .row{display:flex;gap:12px;flex-wrap:wrap}
 .cell{flex:1 1 260px;min-width:240px}
 .cell img{width:100%;border-radius:4px;display:block;background:#000}
 .cap{font-size:12px;color:#9aa4b0;margin:5px 0 0}
 .full img{max-width:760px}
 .k{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px}
 .old{background:#00dc00} .new{background:#e600c8} .he{background:#ff2222}
 .box{background:#ffc800}
</style>
<h1>细胞核 mask 调整前后对比</h1>
<p class="lede">
 <span class="k old"></span> 旧 mask（cellprob 0.0 / diameter 30 px = 18.6 µm）&nbsp;
 <span class="k new"></span> 新 mask（cellprob −1.2 / diameter 14 px = 8.7 µm）&nbsp;
 <span class="k he"></span> 真实 H&amp;E 上 Cellpose 的分割（参照）&nbsp;
 <span class="k box"></span> candidate 区域
 <br><br>
 画的是轮廓不是填充——填充会把它声称的那个核盖住，而这里要判断的正是轮廓有没有落在核上。
 整张 patch 都给出来，这样在 candidate 区域之外的组织上也能看 mask 的表现。
 覆盖率一栏中的比值是「mask 覆盖 ÷ 同一区域真实 H&amp;E 的核覆盖」，目标是 1.00。
</p>
"""

CARD = """
<div class="card">
 <div class="hdr">
  <h2>{name}</h2>
  <span class="nums">{sample} · {rid}</span>
  <span class="nums">旧 <b>{old:.4f}</b> ({r_old:.2f}×) &nbsp; 新 <b>{new:.4f}</b>
   ({r_new:.2f}×) &nbsp; 真实 H&amp;E <b>{he_cov:.4f}</b></span>
 </div>
 <div class="row full"><div class="cell" style="flex:1 1 100%">
   <img src="{full}"><p class="cap">整张 patch，新旧 mask 叠加</p></div></div>
 <div class="row" style="margin-top:12px">
  <div class="cell"><img src="{c_old}"><p class="cap">candidate 区域 · 旧 mask</p></div>
  <div class="cell"><img src="{c_new}"><p class="cap">candidate 区域 · 新 mask</p></div>
  <div class="cell"><img src="{he}"><p class="cap">同一区域真实 H&amp;E · Cellpose 分割</p></div>
 </div>
</div>
"""


if __name__ == '__main__':
    main()
