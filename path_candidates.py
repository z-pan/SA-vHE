#!/usr/bin/env python3
"""Merge the six per-sample selection.csv into one browsable candidate set.

path_locate.py answers "where is this region and which TPAF FOV is it" one sample at a
time. Choosing which regions go in a figure is a cross-sample decision, and it cannot be
made from six separate CSVs and 168 loose PNGs -- so this collapses them into one table
and one page that shows every candidate at once.

Two outputs, for two different jobs:

  master.csv      every pick, one row each, with a stable id. For sorting, for the
                  supplement, and for feeding a chosen tile_id back to path_locate.py
                  --tile_ids or --anchor.
  candidates.html one self-contained page, images embedded. Filter by sample, by
                  structure, by ECM; sort by z; click to select; copy the chosen ids
                  out. No server, no relative paths to break when the file moves.

A region is keyed by (sample, n), not by structure, so a tile several structures all
rank highly appears once carrying all of them. That overlap is content, not duplication
-- a tile four structures agree on is a different kind of candidate from one only
desmoplasia liked -- so it is shown as such rather than flattened away or listed four
times.

Two tiers, kept distinct throughout. Tier 1 is the per-structure pick set, tier 2 the
figure pool added by path_locate.py --extra. They answer different questions and a card
says which it is.

    python path_candidates.py --survey results/path_screen/survey \\
        --out results/path_screen/survey/_candidates
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os

import cv2

import path_structures as ps

SAMPLES = ['240703', '240720', '240729', '240817', '240828_pt1', '240828_pt2']
ZH = {s['key']: s['zh'] for s in ps.ALL_ENTRIES}


def read_source(d):
    kv = {}
    for line in open(os.path.join(d, 'source.txt'), encoding='utf-8'):
        k, _, v = line.strip().partition('=')
        if k:
            kv[k] = v
    return kv


def crop_data_uri(path, px, quality):
    im = cv2.imread(path)
    if im is None:
        return ''
    im = cv2.resize(im, (px, px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def load_links(path):
    """The hand-made correspondences, keyed by region id.

    Absent means TPAF does not reach that region, or nothing there could be matched by
    eye -- see path_tpaf_link.py. Either way it is a finding about the data, not a
    pending task, so a card says "no TPAF" rather than leaving the field blank as if it
    were still being filled in.
    """
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(io.open(path, encoding='utf-8-sig')):
        if r['tpaf_path']:
            out.setdefault(r['id'], []).append(r['tpaf_file'])
        else:
            out.setdefault(r['id'], [])
    return out


def collect(survey, samples, px, quality, links=None):
    links = links or {}
    regions, flat = [], []
    for s in samples:
        d = os.path.join(survey, s)
        cdir = os.path.join(d, 'candidates')
        cpath = os.path.join(cdir, 'selection.csv')
        if not os.path.exists(cpath):
            print(f'  {s}: no candidates/selection.csv -- run path_locate.py first')
            continue
        src = read_source(d)
        um = float(src['um_per_px'])
        rot, roi = 0.0, ''
        vpath = os.path.join(cdir, 'view.txt')
        if os.path.exists(vpath):
            for line in io.open(vpath, encoding='utf-8'):
                if line.startswith('rotate_cw='):
                    rot = float(line.split('=', 1)[1])
                elif line.startswith('roi='):
                    roi = line.split('=', 1)[1].strip()
        rows = list(csv.DictReader(open(cpath, encoding='utf-8-sig')))
        by_n = {}
        for r in rows:
            n = int(r['n'])
            by_n.setdefault(n, []).append(r)
            fov = links.get(f'{s}-{n:02d}')
            flat.append(dict(
                id=f'{s}-{n:02d}', sample=s, **r,
                tpaf_n_fov='' if fov is None else len(fov),
                tpaf_files='' if fov is None else ' | '.join(fov)))
        for n, rs in sorted(by_n.items()):
            r0 = rs[0]
            # The crop is written once per (n, structure) but they are the same image.
            img = ''
            for r in rs:
                p = os.path.join(cdir, f'pick{n:02d}_{r["structure"]}_{r["tile_id"]}.png')
                if os.path.exists(p):
                    img = crop_data_uri(p, px, quality)
                    break
            tier = min(int(r.get('tier', 1) or 1) for r in rs)
            picks = sorted(
                ({'k': r['structure'], 'zh': ZH.get(r['structure'], ''),
                  'ecm': int(r['ecm']), 'z': float(r['z']) if r['z'] else None}
                 for r in rs),
                key=lambda p: -(p['z'] if p['z'] is not None else -9))
            zs = [p['z'] for p in picks if p['z'] is not None]
            regions.append(dict(
                id=f'{s}-{n:02d}', sample=s, n=n, tile_id=r0['tile_id'],
                y_mm=float(r0['y_mm']), x_mm=float(r0['x_mm']),
                y_px=int(r0['y_px']), x_px=int(r0['x_px']),
                frac_y=float(r0['frac_y']), frac_x=float(r0['frac_x']),
                tpaf=links.get(f'{s}-{n:02d}'),
                um=um, rot=rot, roi=roi, tier=tier, picks=picks, zmax=max(zs) if zs else 0.0,
                ecm=int(any(p['ecm'] for p in picks)), img=img))
    return regions, flat


PAGE = """<title>TPAF 配图候选区</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f2f4f5; --surface: #ffffff; --surface-2: #e8ecee;
  --ink: #151c21; --ink-2: #4d5f6a; --ink-3: #7d8f9a;
  --line: #d5dcdf; --accent: #0d6b73; --accent-ink: #ffffff;
  --accent-soft: #d9ebec; --ecm: #8a5a12; --ecm-soft: #f6ead6;
  --shadow: 0 1px 2px rgba(21,28,33,.07), 0 8px 24px -12px rgba(21,28,33,.18);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #10161a; --surface: #192126; --surface-2: #222c32;
    --ink: #e4ecef; --ink-2: #9aabb4; --ink-3: #6f818b;
    --line: #2c383f; --accent: #4fb3bc; --accent-ink: #0b1315;
    --accent-soft: #1c3437; --ecm: #d7a45a; --ecm-soft: #33291a;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --bg: #10161a; --surface: #192126; --surface-2: #222c32;
  --ink: #e4ecef; --ink-2: #9aabb4; --ink-3: #6f818b;
  --line: #2c383f; --accent: #4fb3bc; --accent-ink: #0b1315;
  --accent-soft: #1c3437; --ecm: #d7a45a; --ecm-soft: #33291a;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.7);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.5;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Consolas, monospace; }
header { padding: 28px 24px 0; max-width: 1600px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -.01em; }
.sub { color: var(--ink-2); font-size: 14px; margin: 0 0 20px; max-width: 68ch; }
.bar {
  position: sticky; top: 0; z-index: 20; background: var(--bg);
  border-bottom: 1px solid var(--line); padding: 12px 24px;
  display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
}
.bar-inner { max-width: 1600px; margin: 0 auto; width: 100%;
  display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
.grp { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.lab {
  font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 500;
}
button, select {
  font: inherit; font-size: 13px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 4px; padding: 5px 10px;
  cursor: pointer;
}
button:hover, select:hover { border-color: var(--ink-3); }
button:focus-visible, select:focus-visible, .card:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
button[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: var(--accent-ink);
}
.count { color: var(--ink-2); font-size: 13px; margin-left: auto; }
main {
  max-width: 1600px; margin: 0 auto; padding: 20px 24px 140px;
  display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
}
.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  overflow: hidden; cursor: pointer; text-align: left; padding: 0;
  display: flex; flex-direction: column; box-shadow: var(--shadow);
}
.card[aria-pressed="true"] { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
.card img { width: 100%; aspect-ratio: 1; display: block; background: var(--surface-2); }
.card .body { padding: 9px 10px 10px; display: grid; gap: 7px; }
.idrow { display: flex; align-items: baseline; gap: 8px; }
.id { font-size: 13px; font-weight: 500; letter-spacing: .01em; }
.zmax { margin-left: auto; font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  font-size: 11px; padding: 1px 6px; border-radius: 3px; white-space: nowrap;
  background: var(--surface-2); color: var(--ink-2);
}
.tp { color: var(--ink-3); font-size: 11px; line-height: 1.35; word-break: break-all; }
.tp b { color: var(--ink-2); font-weight: 500; }
.card.linked { border-left: 3px solid var(--accent); }
.t2 {
  font-size: 10px; font-weight: 600; color: #fff; background: #1f8a2e;
  border-radius: 3px; padding: 0 4px; line-height: 15px;
}
.chip.ecm { background: var(--ecm-soft); color: var(--ecm); font-weight: 500; }
.meta {
  font-size: 11.5px; color: var(--ink-3); display: grid; gap: 2px;
  font-variant-numeric: tabular-nums;
}
.meta b { color: var(--ink-2); font-weight: 500; }
.tray {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 30;
  background: var(--surface); border-top: 1px solid var(--line);
  padding: 12px 24px; box-shadow: 0 -8px 24px -16px rgba(0,0,0,.5);
}
.tray-inner { max-width: 1600px; margin: 0 auto; display: flex; gap: 14px; align-items: center; }
.tray textarea {
  flex: 1; font-family: "IBM Plex Mono", monospace; font-size: 12px;
  background: var(--bg); color: var(--ink); border: 1px solid var(--line);
  border-radius: 4px; padding: 7px 9px; height: 46px; resize: vertical;
}
.empty { color: var(--ink-3); padding: 40px 0; grid-column: 1/-1; text-align: center; }
@media (prefers-reduced-motion: no-preference) {
  .card { transition: box-shadow .12s ease, border-color .12s ease; }
}
</style>

<header>
  <h1>TPAF 配图候选区</h1>
  <p class="sub">__SUB__</p>
</header>

<div class="bar"><div class="bar-inner">
  <div class="grp"><span class="lab">样本</span><span id="fSample"></span></div>
  <div class="grp"><span class="lab">结构</span><select id="fStruct"></select></div>
  <div class="grp"><button id="fEcm" aria-pressed="false">仅 ECM</button>
    <button id="fTier" aria-pressed="false">仅新增(绿框)</button>
    <button id="fTpaf" aria-pressed="false">仅已配 TPAF</button></div>
  <div class="grp"><span class="lab">排序</span><select id="fSort">
    <option value="z">z 由高到低</option>
    <option value="npick">被多个结构选中优先</option>
    <option value="sample">样本 / 编号</option>
  </select></div>
  <span class="count" id="count"></span>
</div></div>

<main id="grid"></main>

<div class="tray"><div class="tray-inner">
  <span class="lab">已选 <span id="nsel" class="mono">0</span></span>
  <textarea id="out" readonly placeholder="点卡片选区；这里给出 tile_id，可直接喂给 path_locate.py --tile_ids"></textarea>
  <button id="copy">复制</button>
  <button id="clear">清空</button>
</div></div>

<script>
const R = __DATA__;
// The FOV filenames share a long prefix -- date, wavelength, power, sample, PMT gain --
// and differ only at the end. Showing the whole thing on a card pushes everything else
// out; the tail is what identifies the file, and the full name is in the copy tray and
// in master.csv.
const shortFov = f => { const b = f.replace(/[.]tif$/i, ''); const i = b.search(/(Line-|_[0-9]{4}|[[])/); return i > 0 ? '…' + b.slice(i) : b; };
const sel = new Set();
const grid = document.getElementById('grid');
const samples = [...new Set(R.map(r => r.sample))];
let fs = new Set(samples), fk = '', fecm = false, ft2 = false, ftp = false, sort = 'z';

const sBox = document.getElementById('fSample');
samples.forEach(s => {
  const b = document.createElement('button');
  b.textContent = s; b.setAttribute('aria-pressed', 'true');
  b.onclick = () => {
    if (fs.has(s)) { fs.delete(s); } else { fs.add(s); }
    b.setAttribute('aria-pressed', fs.has(s));
    draw();
  };
  sBox.appendChild(b);
});

const kSel = document.getElementById('fStruct');
const keys = [...new Set(R.flatMap(r => r.picks.map(p => p.k)))].sort();
kSel.innerHTML = '<option value="">全部</option>' +
  keys.map(k => `<option value="${k}">${k}</option>`).join('');
kSel.onchange = () => { fk = kSel.value; draw(); };
document.getElementById('fSort').onchange = e => { sort = e.target.value; draw(); };
const eBtn = document.getElementById('fEcm');
eBtn.onclick = () => { fecm = !fecm; eBtn.setAttribute('aria-pressed', fecm); draw(); };
const tBtn = document.getElementById('fTier');
tBtn.onclick = () => { ft2 = !ft2; tBtn.setAttribute('aria-pressed', ft2); draw(); };
const pBtn = document.getElementById('fTpaf');
pBtn.onclick = () => { ftp = !ftp; pBtn.setAttribute('aria-pressed', ftp); draw(); };

function tray() {
  const chosen = R.filter(r => sel.has(r.id));
  document.getElementById('nsel').textContent = chosen.length;
  document.getElementById('out').value = chosen
    .map(r => `${r.id}\\t${r.tile_id}\\t${r.tpaf && r.tpaf.length ? r.tpaf.join(' | ') : '(no TPAF)'}`).join('\\n');
}
document.getElementById('copy').onclick = () => {
  navigator.clipboard.writeText(document.getElementById('out').value);
};
document.getElementById('clear').onclick = () => { sel.clear(); draw(); };

function draw() {
  let v = R.filter(r => fs.has(r.sample))
           .filter(r => !fecm || r.ecm)
           .filter(r => !ft2 || r.tier === 2)
           .filter(r => !ftp || (r.tpaf && r.tpaf.length))
           .filter(r => !fk || r.picks.some(p => p.k === fk));
  const key = r => fk ? (r.picks.find(p => p.k === fk).z ?? -9) : r.zmax;
  if (sort === 'z') v.sort((a, b) => key(b) - key(a));
  else if (sort === 'npick') v.sort((a, b) => b.picks.length - a.picks.length || key(b) - key(a));
  else v.sort((a, b) => a.sample.localeCompare(b.sample) || a.n - b.n);

  document.getElementById('count').textContent =
    `${v.length} / ${R.length} 个区域`;
  grid.innerHTML = v.length ? '' : '<p class="empty">没有符合当前筛选的区域</p>';
  for (const r of v) {
    const c = document.createElement('button');
    c.className = 'card' + (r.tpaf && r.tpaf.length ? ' linked' : '');
    c.setAttribute('aria-pressed', sel.has(r.id));
    c.onclick = () => {
      if (sel.has(r.id)) { sel.delete(r.id); } else { sel.add(r.id); }
      c.setAttribute('aria-pressed', sel.has(r.id));
      tray();
    };
    c.innerHTML = `
      <img src="${r.img}" alt="${r.id} ${r.picks.map(p => p.k).join(' ')}" loading="lazy">
      <div class="body">
        <div class="idrow">
          <span class="id mono">${r.id}</span>${r.tier === 2 ? '<span class="t2">+</span>' : ''}
          <span class="zmax mono">${r.rot ? '↻' + r.rot + '° ' : ''}${r.roi ? 'ROI ' : ''}z ${key(r).toFixed(2)}</span>
        </div>
        <div class="chips">${r.picks.map(p =>
          `<span class="chip${p.ecm ? ' ecm' : ''}">${p.zh || p.k}${
            p.z != null ? ' ' + p.z.toFixed(1) : ''}</span>`).join('')}</div>
        <div class="meta mono">
          <div>${r.tile_id}</div>
          <div><b>x</b> ${r.x_mm.toFixed(2)} <b>y</b> ${r.y_mm.toFixed(2)} mm</div>
          <div class="tp">${r.tpaf === null ? '<b>TPAF</b> 该样本未配对'
            : r.tpaf.length ? '<b>TPAF</b> ' + r.tpaf.map(shortFov).join('<br>')
            : '<b>TPAF</b> 无对应'}</div>
        </div>
      </div>`;
    grid.appendChild(c);
  }
  tray();
}
draw();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey', default='results/path_screen/survey')
    ap.add_argument('--out', default='results/path_screen/survey/_candidates')
    ap.add_argument('--samples', default=','.join(SAMPLES))
    ap.add_argument('--thumb_px', type=int, default=240,
                    help='Size the crops are embedded at. 240 keeps the page a few MB; '
                         'the 512 px originals stay in each candidates/ directory.')
    ap.add_argument('--quality', type=int, default=78)
    ap.add_argument('--links', default=None,
                    help='tpaf_links.csv from path_tpaf_link.py. Defaults to the one in '
                         '--out. Absent means the page shows no TPAF column at all, '
                         'rather than showing every region as unmatched.')
    args = ap.parse_args()

    samples = [s.strip() for s in args.samples.split(',') if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    links = load_links(args.links or os.path.join(args.out, 'tpaf_links.csv'))
    regions, flat = collect(args.survey, samples, args.thumb_px, args.quality, links)
    if not regions:
        raise SystemExit('nothing collected')

    mpath = os.path.join(args.out, 'master.csv')
    cols = ['id', 'sample', 'n', 'tier', 'structure', 'zh', 'ecm', 'tile_id', 'z',
            'y_px', 'x_px', 'y_mm', 'x_mm', 'frac_y', 'frac_x',
            'tpaf_n_fov', 'tpaf_files']
    with open(mpath, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(flat, key=lambda r: (r['sample'], int(r['n']))))

    n_ecm = sum(r['ecm'] for r in regions)
    n_multi = sum(1 for r in regions if len(r['picks']) > 1)
    n_link = sum(1 for r in regions if r['tpaf'])
    sub = (f'{len(regions)} 个候选区，来自 {len(flat)} 次 top 命中'
           f'（{len(samples)} 样本 × 5 个 ECM 结构 top-3 + 13 个非 ECM 结构 top-1）。'
           f'其中 {n_ecm} 个被至少一个 ECM 结构选中，{n_multi} 个被多个结构同时选中。'
           f'z 是该结构在本样本内的分数标准差数，跨样本不可直接比。'
           f'TPAF 一列是人工逐区目视配对的结果（path_tpaf_link.py 校验），'
           f'不是坐标推算；{n_link} 个区有对应 FOV，空白表示 TPAF 未覆盖或找不到对应。')

    page = (PAGE.replace('__DATA__', json.dumps(regions, ensure_ascii=False))
                .replace('__SUB__', html.escape(sub)))
    hpath = os.path.join(args.out, 'candidates.html')
    io.open(hpath, 'w', encoding='utf-8').write(page)

    print(f'{len(regions)} regions from {len(flat)} picks')
    print(f'  -> {mpath}')
    print(f'  -> {hpath}  ({os.path.getsize(hpath)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
