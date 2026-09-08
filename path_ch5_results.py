#!/usr/bin/env python3
"""Every number Ch5 reports, from one command.

The figures were arrived at across many one-off scripts, and three conventions in them
are easy to get wrong -- I got one wrong once already, and the error was invisible
because the table still looked reasonable:

  The TPAF reference is recomputed for whatever region set is being scored. It is the
  Cellpose count on the raw TPAF, which involves no stain, and the nucleus-level subset
  is denser by construction: 1274 /mm2 over all 148 regions, 1608 over the 74. Scoring
  the subset against the full-set reference put real H&E at +39%.

  Methods are compared on the regions they all produced output for. CycleGAN detects no
  nuclei at all in 17 of 148 and SA-CUT in 14, so a table over each method's own regions
  quietly drops each method's worst fields. The count of missing regions is reported as
  a column in its own right rather than hidden in the denominator.

  Nucleus-level metrics are restricted to regions where TPAF carries nuclear
  information (path_vhe_subset.py); the stain-distribution metric covers all 148,
  because it needs neither a segmenter nor a resolvable nucleus. Two tracks, two
  denominators, and the reason belongs next to the numbers.

METHODS below is the whole configuration. A method whose `source` is not in the data
prints as a placeholder row, so the table has its final shape before the last two runs
finish.

    python path_ch5_results.py
    python path_ch5_results.py --md outputs/ch5_tables.md
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os

import numpy as np

R = 'results/path_screen/survey/_downstream'
SUR = 'results/path_screen/survey/_vhe'
EPI = {'neoplastic', 'non-neoplastic epithelial'}

# (label, source directory, saliency constraint, nucleus mask). Order is the reading
# order of the table: reference first, then the proposed method, then the comparisons
# in increasing distance from it.
METHODS = [
    ('真实 H&E',            'real_HE',        None,  None),
    ('SA-CycleGAN（本文）',  'nuc_flat_final', True,  True),
    ('UTOM',                'gray_final',     True,  False),
    ('CycleGAN',            'cyclegan_final', False, False),
    ('SA-CUT v2',           'sacut_v2_final', None,  True),   # 待训练
    ('CUT',                 'cut_final',      None,  False),  # 待训练
]


def load_hv(path):
    n = collections.Counter()
    typ = collections.defaultdict(collections.Counter)
    px = collections.defaultdict(list)
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        k = (r['source'], r['id'])
        n[k] += 1
        typ[k][r['type_name']] += 1
        try:
            px[r['source']].append(float(r['area_px']) * 0.0625)   # 0.25 um/px
        except ValueError:
            pass
    return n, typ, px


def load_area(path):
    a = {}
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        a[(r['source'], r['id'])] = (int(r['out_w']) * int(r['out_h'])
                                     * float(r['out_mpp']) ** 2 / 1e6)
    return a


def load_tpaf_ref(path):
    """Cellpose density on the raw TPAF, per region. No stain is involved, which is
    why this and not real H&E is the reference the counts are scored against."""
    d = {}
    for r in csv.DictReader(io.open(path, encoding='utf-8')):
        if r['source'] == 'TPAF' and r['density'] not in ('', 'nan'):
            d[r['id']] = float(r['density'])
    return d


def load_subset(path):
    if not os.path.exists(path):
        return None
    return {r['id'] for r in csv.DictReader(io.open(path, encoding='utf-8'))
            if r['keep'] == '1'}


def per_source(n, typ, area, src, ids):
    ks = [k for k in n if k[0] == src and k[1] in ids]
    if not ks:
        return None
    dens = np.median([n[k] / area[k] for k in ks if k in area and area[k]])
    tot = collections.Counter()
    for k in ks:
        tot.update({a: b for a, b in typ[k].items() if a.lower() != 'background'})
    t = sum(tot.values()) or 1
    return dict(density=dens,
                epi=sum(tot[c] for c in tot if c.lower() in EPI) / t,
                neo=tot['Neoplastic'] / t)


def table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    out = ['  '.join(str(h).ljust(w[i]) for i, h in enumerate(head)).rstrip()]
    out.append('  '.join('-' * w[i] for i in range(len(head))))
    for r in rows:
        out.append('  '.join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=R)
    ap.add_argument('--subset', default=SUR + '/nuc_subset.csv')
    ap.add_argument('--md', default=None, help='Also write the tables to this file.')
    args = ap.parse_args()

    n, typ, px = load_hv(os.path.join(args.root, 'hv_hovernet_fast_pannuke.csv'))
    area = load_area(os.path.join(args.root, 'crop_scale.csv'))
    tpaf = load_tpaf_ref(os.path.join(args.root, 'cp_morphology.csv'))
    sub = load_subset(args.subset)
    have = {k[0] for k in n}
    present = [m for m in METHODS if m[1] in have]
    missing = [m for m in METHODS if m[1] not in have]

    all_ids = {k[1] for k in n if k[0] == 'real_HE'}
    common = set.intersection(*[{k[1] for k in n if k[0] == m[1]} for m in present])

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('Ch5 结果  —— %d 个方法有数据，%d 个待训练'
         % (len(present), len(missing)))
    if missing:
        emit('  待训练: ' + '、'.join('%s (%s)' % (m[0], m[1]) for m in missing))
    emit()

    emit('无输出区域数（该方法在多少个区域里检不出任何细胞核，共 %d 个区域）'
         % len(all_ids))
    rows = []
    for label, src, sal, msk in METHODS:
        if src in have:
            rows.append([label, len(all_ids) - len({k[1] for k in n if k[0] == src})])
        else:
            rows.append([label, '待训练'])
    emit(table(rows, ['方法', '无输出']))
    emit()

    for name, ids in (('共同区域', common),
                      ('核层面子集 ∩ 共同区域', (sub & common) if sub else None)):
        if not ids:
            continue
        ref = np.median([v for i, v in tpaf.items() if i in ids])
        emit('%s  n=%d   TPAF 参照 %.0f /mm2  （原始 TPAF 的 Cellpose 计数，不经染色）'
             % (name, len(ids), ref))
        rows = []
        for label, src, sal, msk in METHODS:
            if src not in have:
                rows.append([label, '待训练', '', '', ''])
                continue
            s = per_source(n, typ, area, src, ids)
            rows.append([label, '%.0f' % s['density'],
                         '%+.0f%%' % (100 * (s['density'] / ref - 1)),
                         '%.3f' % s['epi'], '%.3f' % s['neo']])
        emit(table(rows, ['方法', '核计数', 'vs TPAF', '上皮核', 'Neoplastic']))
        emit()

    emit('核面积中位数 µm²  （真实 H&E 为目标值）')
    rows = []
    for label, src, sal, msk in METHODS:
        rows.append([label, '%.1f' % np.median(px[src]) if src in have else '待训练'])
    emit(table(rows, ['方法', 'µm²']))
    emit()

    emit('方法构成')
    rows = [[m[0],
             '—' if m[2] is None else ('✓' if m[2] else '✗'),
             '—' if m[3] is None else ('✓' if m[3] else '✗'),
             '有数据' if m[1] in have else '待训练'] for m in METHODS]
    emit(table(rows, ['方法', 'saliency 约束', '核 mask', '状态']))
    emit()
    emit('染色分布（紫色团块）见 path_vhe_stainmap.py，它覆盖全部 %d 个区域：'
         % len(all_ids))
    emit('  python path_vhe_stainmap.py --versions <每个方法的 stained 目录名>')

    if args.md:
        os.makedirs(os.path.dirname(args.md) or '.', exist_ok=True)
        with io.open(args.md, 'w', encoding='utf-8') as f:
            f.write('```\n' + '\n'.join(lines) + '\n```\n')
        print('\n-> ' + os.path.abspath(args.md))


if __name__ == '__main__':
    main()
