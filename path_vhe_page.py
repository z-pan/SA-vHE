#!/usr/bin/env python3
"""One browsable page for the virtual-staining comparison.

Twenty-five contact sheets are a poor way to look at 148 regions: you cannot ask which
ones are worst, or see only the stromal picks, or find the one region a colleague named.
This puts every region on a card with its real H&E beside each virtual variant, and
makes the table sortable by how far the staining actually landed from the real thing.

The deltas shown are the two the metrics say matter here, and they are signed:

  R-B   real H&E runs about +8, both virtual variants about -8. The sign is the whole
        point -- virtual staining comes out blue-violet where real H&E is faintly pink,
        so an absolute value would hide which way it went wrong.
  H/E   haematoxylin to eosin, after colour deconvolution. Real 0.70, virtual 0.50:
        too much eosin relative to haematoxylin, which is a different fault from the
        overall colour cast and can move independently of it.

Sorting by these puts the failures first, which is the order to look in -- the median
card tells you nothing you cannot read off the pooled table.

    python path_vhe_page.py
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import html
import io
import json
import os

import cv2
import numpy as np

TPAF_UM = 0.621
HE_UM = 0.5


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return im


def to_common(im, um_per_px, out_um_per_px, extent_um):
    """Centre-cut to `extent_um` of tissue, then resample to a common um/px.

    Without this the three images on a card are three different magnifications shown at
    one width, and a difference in how large the nuclei look means nothing -- which is
    how a 2.5x magnification error in the staining survived a first look at this page.
    The contact sheets always did this; the page did not, and the page is what gets
    looked at.
    """
    side = int(round(extent_um / um_per_px))
    h, w = im.shape[:2]
    side = max(8, min(side, h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    cut = im[y0:y0 + side, x0:x0 + side]
    out = max(8, int(round(extent_um / out_um_per_px)))
    interp = cv2.INTER_AREA if out < side else cv2.INTER_CUBIC
    return cv2.resize(cut, (out, out), interpolation=interp)


def to_uri(im, px, quality, um_per_px=None, extent_um=None, view_um=0.5):
    """Square thumbnail, scale-matched first when the geometry is known."""
    if im is None:
        return ''
    if um_per_px and extent_um:
        im = to_common(im, um_per_px, view_um, extent_um)
    s = px / max(im.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))),
                    interpolation=interp)
    ok, buf = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode() if ok else ''


PAGE = """<title>虚拟染色对照</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f3f4f2; --surface: #fff; --surface-2: #e9ebe7;
  --ink: #1a1d19; --ink-2: #5a6157; --ink-3: #8b9287;
  --line: #d8dcd4; --accent: #5a6e2f; --accent-ink: #fff;
  --warn: #a4442a; --cool: #2f5a6e;
  --shadow: 0 1px 2px rgba(26,29,25,.07), 0 8px 24px -12px rgba(26,29,25,.18);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14170f; --surface: #1e2218; --surface-2: #2a2f22;
    --ink: #e8ece2; --ink-2: #a6ae9c; --ink-3: #767e6c;
    --line: #333a2a; --accent: #9dbc5a; --accent-ink: #14170f;
    --warn: #e08a6d; --cool: #7fb6cd;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --bg: #14170f; --surface: #1e2218; --surface-2: #2a2f22;
  --ink: #e8ece2; --ink-2: #a6ae9c; --ink-3: #767e6c;
  --line: #333a2a; --accent: #9dbc5a; --accent-ink: #14170f;
  --warn: #e08a6d; --cool: #7fb6cd;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.7);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-size: 15px;
  font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
.mono { font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace; }
header { padding: 26px 24px 0; max-width: 1500px; margin: 0 auto; }
h1 { font-size: 21px; font-weight: 600; margin: 0 0 6px; }
.sub { color: var(--ink-2); font-size: 14px; margin: 0 0 14px; max-width: 70ch; }
table.pool { border-collapse: collapse; font-size: 13px; margin: 0 0 18px;
  font-variant-numeric: tabular-nums; }
table.pool th, table.pool td { padding: 3px 12px 3px 0; text-align: right; }
table.pool th:first-child, table.pool td:first-child { text-align: left; }
table.pool th { color: var(--ink-3); font-weight: 500; font-size: 11px;
  letter-spacing: .07em; text-transform: uppercase; }
table.pool tr.real td { color: var(--accent); font-weight: 500; }
.bar { position: sticky; top: 0; z-index: 9; background: var(--bg);
  border-bottom: 1px solid var(--line); padding: 11px 24px; }
.bar-in { max-width: 1500px; margin: 0 auto; display: flex; flex-wrap: wrap;
  gap: 9px 18px; align-items: center; }
.grp { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.lab { font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 500; }
button, select { font: inherit; font-size: 13px; color: var(--ink);
  background: var(--surface); border: 1px solid var(--line); border-radius: 4px;
  padding: 5px 10px; cursor: pointer; }
button:hover, select:hover { border-color: var(--ink-3); }
button:focus-visible, select:focus-visible { outline: 2px solid var(--accent);
  outline-offset: 2px; }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
  color: var(--accent-ink); }
.count { margin-left: auto; color: var(--ink-2); font-size: 13px; }
main { max-width: 1500px; margin: 0 auto; padding: 18px 24px 60px;
  display: grid; gap: 18px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  box-shadow: var(--shadow); padding: 12px 14px 14px; }
.head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 9px; }
.id { font-size: 14px; font-weight: 500; }
.chips { display: flex; gap: 4px; flex-wrap: wrap; }
.chip { font-size: 11px; padding: 1px 6px; border-radius: 3px;
  background: var(--surface-2); color: var(--ink-2); }
.d { font-size: 12px; font-variant-numeric: tabular-nums; margin-left: auto;
  color: var(--ink-2); }
.d b { font-weight: 500; }
.d .hi { color: var(--warn); font-weight: 600; }
.imgs { display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
figure { margin: 0; }
figure img { width: 100%; display: block; border-radius: 4px;
  background: var(--surface-2); }
figcaption { font-size: 11px; color: var(--ink-3); margin-top: 4px;
  display: flex; justify-content: space-between; gap: 8px; }
.empty { color: var(--ink-3); text-align: center; padding: 40px 0; }
</style>

<header>
  <h1>虚拟染色对照</h1>
  <p class="sub">__SUB__</p>
  __POOL__
</header>

<div class="bar"><div class="bar-in">
  <div class="grp"><span class="lab">样本</span><span id="fSample"></span></div>
  <div class="grp"><span class="lab">结构</span><select id="fStruct"></select></div>
  <div class="grp"><span class="lab">排序</span><select id="fSort">
    <option value="rb">R−B 偏离最大</option>
    <option value="he">H/E 偏离最大</option>
    <option value="id">样本 / 编号</option>
    <option value="extent">对比范围最大</option>
  </select></div>
  <span class="count" id="count"></span>
</div></div>

<main id="grid"></main>

<script>
const R = __DATA__;
const VAR = __VARS__;
const grid = document.getElementById('grid');
const samples = [...new Set(R.map(r => r.sample))];
let fs = new Set(samples), fk = '', sort = 'rb';

const sBox = document.getElementById('fSample');
samples.forEach(s => {
  const b = document.createElement('button');
  b.textContent = s; b.setAttribute('aria-pressed', 'true');
  b.onclick = () => { fs.has(s) ? fs.delete(s) : fs.add(s);
    b.setAttribute('aria-pressed', fs.has(s)); draw(); };
  sBox.appendChild(b);
});
const kSel = document.getElementById('fStruct');
const keys = [...new Set(R.flatMap(r => r.structures))].sort();
kSel.innerHTML = '<option value="">全部</option>' +
  keys.map(k => `<option value="${k}">${k}</option>`).join('');
kSel.onchange = () => { fk = kSel.value; draw(); };
document.getElementById('fSort').onchange = e => { sort = e.target.value; draw(); };

const fmt = (v, d) => v === null ? '--' : (v >= 0 ? '+' : '') + v.toFixed(d);

function draw() {
  let v = R.filter(r => fs.has(r.sample)).filter(r => !fk || r.structures.includes(fk));
  const dev = r => Math.abs(r.d_rb ?? 0);
  const devh = r => Math.abs(r.d_he ?? 0);
  if (sort === 'rb') v.sort((a, b) => dev(b) - dev(a));
  else if (sort === 'he') v.sort((a, b) => devh(b) - devh(a));
  else if (sort === 'extent') v.sort((a, b) => b.extent - a.extent);
  else v.sort((a, b) => a.sample.localeCompare(b.sample) || a.n - b.n);

  document.getElementById('count').textContent = `${v.length} / ${R.length} 个区域`;
  grid.innerHTML = v.length ? '' : '<p class="empty">没有符合筛选的区域</p>';
  for (const r of v) {
    const c = document.createElement('div');
    c.className = 'card';
    const figs = [['real', '真实 H&E', r.img.real]].concat(
      VAR.map(k => [k, 'vHE ' + k, r.img[k]]));
    c.innerHTML = `
      <div class="head">
        <span class="id mono">${r.id}</span>
        <span class="chips">${r.structures.map(s => `<span class="chip">${s}</span>`).join('')}</span>
        <span class="d"><b>R−B</b> 真实 ${fmt(r.rb_real,1)} → vHE ${fmt(r.rb_v,1)}
          <span class="${dev(r) > 12 ? 'hi' : ''}">(Δ${fmt(r.d_rb,1)})</span>
          &nbsp;&nbsp;<b>H/E</b> ${fmt(r.he_real,2)} → ${fmt(r.he_v,2)}
          <span class="${devh(r) > 0.25 ? 'hi' : ''}">(Δ${fmt(r.d_he,2)})</span></span>
      </div>
      <div class="imgs">${figs.map(([k, cap, src]) => `
        <figure><img src="${src}" alt="${r.id} ${cap}" loading="lazy">
          <figcaption><span>${cap}</span><span class="mono">${
            k === 'real' ? r.extent.toFixed(0) + ' um' : ''}</span></figcaption>
        </figure>`).join('')}</div>`;
    grid.appendChild(c);
  }
}
draw();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--compare', default='results/path_screen/survey/_vhe/compare')
    ap.add_argument('--links',
                    default='results/path_screen/survey/_candidates/tpaf_links.csv')
    ap.add_argument('--out',
                    default='results/path_screen/survey/_vhe/compare/vhe_compare.html')
    ap.add_argument('--px', type=int, default=300)
    ap.add_argument('--quality', type=int, default=80)
    ap.add_argument('--view_um', type=float, default=0.5,
                    help='um/px the three images are brought to before they '
                         'are shown side by side.')
    args = ap.parse_args()

    man = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    met = list(csv.DictReader(io.open(os.path.join(args.compare, 'compare_metrics.csv'),
                                      encoding='utf-8-sig')))
    struct = {}
    if os.path.exists(args.links):
        for r in csv.DictReader(io.open(args.links, encoding='utf-8-sig')):
            if r.get('structures'):
                struct[r['id']] = r['structures'].split()
    variants = sorted({r['variant'] for r in met} - {'real_HE'})
    by = collections.defaultdict(dict)
    for r in met:
        by[r['id']][r['variant']] = r

    num = lambda r, k: float(r[k]) if r and r.get(k) not in (None, '') else None
    regions = []
    for m in sorted(man, key=lambda r: (r['sample'], int(r['n']))):
        rid = m['id']
        rows = by.get(rid, {})
        real = rows.get('real_HE')
        first = rows.get(variants[0]) if variants else None
        extent = float(first['extent_um']) if first else 0.0
        img = {'real': to_uri(imread_u(m['he_path']), args.px, args.quality,
                              HE_UM, extent, args.view_um) if m['he_path'] else ''}
        for v in variants:
            p = os.path.join(args.compare, v, rid + '.png')
            img[v] = (to_uri(imread_u(p), args.px, args.quality,
                             TPAF_UM, extent, args.view_um)
                      if os.path.exists(p) else '')
        rb_r, rb_v = num(real, 'RB_gap'), num(first, 'RB_gap')
        he_r, he_v = num(real, 'HE_ratio'), num(first, 'HE_ratio')
        regions.append(dict(
            id=rid, sample=m['sample'], n=int(m['n']),
            structures=struct.get(rid, []),
            extent=extent,
            rb_real=rb_r, rb_v=rb_v,
            d_rb=(rb_v - rb_r) if (rb_r is not None and rb_v is not None) else None,
            he_real=he_r, he_v=he_v,
            d_he=(he_v - he_r) if (he_r is not None and he_v is not None) else None,
            img=img))

    pool = ['<table class="pool"><tr><th>集合均值</th><th>R</th><th>G</th><th>B</th>'
            '<th>R−B</th><th>H</th><th>E</th><th>H/E</th></tr>']
    for v in ['real_HE'] + variants:
        sub = [r for r in met if r['variant'] == v and r.get('R')]
        if not sub:
            continue
        g = lambda k: np.mean([float(r[k]) for r in sub])
        cls = ' class="real"' if v == 'real_HE' else ''
        pool.append(f'<tr{cls}><td>{v}</td><td>{g("R"):.1f}</td><td>{g("G"):.1f}</td>'
                    f'<td>{g("B"):.1f}</td><td>{g("RB_gap"):+.1f}</td>'
                    f'<td>{g("H"):.4f}</td><td>{g("E"):.4f}</td>'
                    f'<td>{g("HE_ratio"):.3f}</td></tr>')
    pool.append('</table>')

    sub = (f'{len(regions)} 个区域，每个是「真实 H&E」对「'
           + '」「'.join('vHE ' + v for v in variants) + '」。'
           '虚拟染色在完整 TPAF patch 上进行，再按反解出的框裁出——顺序反过来结果会不同。'
           '两侧是不同仪器的独立采集且未配准，所以不算 SSIM/PSNR；'
           '并排的三张已裁到同一物理范围并重采样到同一 µm/px——不这样做，'
           '核的大小差异读不出任何意义。指标用原生像素。')

    page = (PAGE.replace('__DATA__', json.dumps(regions, ensure_ascii=False))
                .replace('__VARS__', json.dumps(variants))
                .replace('__POOL__', ''.join(pool))
                .replace('__SUB__', html.escape(sub)))
    io.open(args.out, 'w', encoding='utf-8').write(page)
    print(f'{len(regions)} regions, variants: {", ".join(variants)}')
    print(f'  -> {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
