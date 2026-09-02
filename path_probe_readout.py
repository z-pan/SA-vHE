#!/usr/bin/env python3
"""Read out a --probes scoring run and decide a caption.

Runs locally on the CSVs the Colab pass brings back; no model, no GPU. Everything it
prints is a comparison between columns of one CSV, which is the point: the candidates
and the incumbent were scored against the same image embeddings, so the differences
between their columns are caused by the captions and by nothing else.

    python path_probe_readout.py --scores_dir results/path_screen/survey/_probes

expects one conch_probes_{sample}.csv per sample there, and --archive pointing at the
previous run's conch_{sample}.csv so the retired caption can be checked against itself.

Everything is calibrated against the run itself. The first version of this script used
absolute cutoffs -- "confirmed above r=0.9", "redundant above r=0.9", "distinct between
0.2 and 0.6" -- and on the 2026-08-20 bake-off all three were wrong in a way that would
have cost the right answer. CONCH similarity columns all correlate positively: the
median of the 918 pairs among the eighteen real columns is +0.19, so +0.66 is the
97.6th percentile, not a middling number. Judged against a fixed 0.7 it read as "not
confirmed"; judged against the run it is the strongest neighbour out of twenty-one. And
the winning candidate sat at +0.61 with dense_collagen while sharing only 2.5 of its top
10 tiles, where a rejected one sat at +0.74 and shared 4.0 -- correlation could not
separate those, top-tile overlap could.

So the rules here are rank and percentile, never a bare threshold, and the redundancy
test is on the tiles the columns actually rank.

The four sections, in the order they should be believed:

  1. Regression. probe_septa_old carries the caption the archived run used, so its
     column has to come back identical. Without it a candidate could look better only
     because the tiles, preprocessing or weights had moved underneath.
  2. Diagnosis. probe_nests_only is not a candidate. Where it ranks among the retired
     caption's neighbours is the evidence for what that caption was reading.
  3. Comparison. Which candidate lost the defect, and which is a second name for a
     column that already exists.
  4. Top tiles. The only section that can confirm. Steps 1-3 say a statistic is clean;
     they cannot say the tiles contain the structure, and they cannot see a
     sample-specific artefact -- on the 2026-08-20 run only this step showed the winner
     still ranking haemorrhage on 240828_pt1.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np

import path_structures as ps

# The defect: the retired caption tracked these.
EPITHELIAL = ['glandular', 'solid', 'high_atypia', 'slit_like', 'necrosis']
# What a stromal column is at risk of duplicating.
FIBROUS = ['dense_collagen', 'desmoplasia', 'ovarian_stroma']
CONTROL = 'probe_septa_old'
MECHANISM = 'probe_nests_only'

SAMPLES = ['240703', '240720', '240729', '240817', '240828_pt1', '240828_pt2']


def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    if not rows:
        raise SystemExit(f'{path} is empty')
    ids = [r['tile_id'] for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'tile_id'}
    return ids, cols


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def zmax(v):
    return float(((v - v.mean()) / (v.std() + 1e-9)).max())


def mean_over(data, fn):
    return float(np.mean([fn(cols) for _, cols in data.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores_dir', required=True,
                    help='Holds conch_probes_{sample}.csv from the --probes run.')
    ap.add_argument('--archive', default='results/path_screen/survey/_scores',
                    help='Previous run, for the regression check. Pass "" to skip it.')
    ap.add_argument('--samples', default=','.join(SAMPLES))
    ap.add_argument('--candidates', default=None,
                    help='Comma-separated columns to compare. Defaults to every probe '
                         'column plus the live fibrous_septa.')
    ap.add_argument('--against', default='fibrous_septa',
                    help='The live entry the probes are candidates for.')
    ap.add_argument('--top', type=int, default=10,
                    help='Top-N tile set used for the redundancy test.')
    args = ap.parse_args()

    samples = [s.strip() for s in args.samples.split(',') if s.strip()]
    data = {}
    for s in samples:
        p = os.path.join(args.scores_dir, f'conch_probes_{s}.csv')
        if not os.path.exists(p):
            print(f'  missing {p} -- skipped')
            continue
        data[s] = load(p)
    if not data:
        raise SystemExit('no conch_probes_*.csv found')
    have = next(iter(data.values()))[1]

    if args.candidates:
        cands = [c.strip() for c in args.candidates.split(',') if c.strip()]
    else:
        # Every column the CSV carries that is not one of the eighteen, plus the live
        # entry. Taken from the file rather than from PROBES, because a promotion
        # rewrites PROBES and the CSV keeps the names it was scored under: after one,
        # this file's `fibrous_septa` column holds whatever that entry said at scoring
        # time, and the promoted caption is still under its old probe name.
        cands = [args.against] + [k for k in have if k not in ps.KEYS]
    seen = set()
    cands = [c for c in cands if c in have and c not in (CONTROL, MECHANISM)
             and not (c in seen or seen.add(c))]
    if CONTROL not in have or MECHANISM not in have:
        raise SystemExit(f'CSV lacks {CONTROL} / {MECHANISM} -- scored without --probes?')

    print()
    print('=' * 78)
    print(f'1. REGRESSION -- {CONTROL} must reproduce the archived {args.against}')
    print('=' * 78)
    worst = 0.0
    if not args.archive:
        print('  skipped (--archive "")')
    else:
        for s, (ids, cols) in data.items():
            ap_ = os.path.join(args.archive, f'conch_{s}.csv')
            if not os.path.exists(ap_):
                print(f'  {s:<12} no archive at {ap_}')
                continue
            aids, acols = load(ap_)
            if aids != ids:
                print(f'  {s:<12} TILE ORDER DIFFERS ({len(aids)} vs {len(ids)}) -- '
                      'not the same tile set, stop here')
                worst = 9
                continue
            # Before the first promotion the retired caption lived in the archive
            # under the live entry's name; after one, the archive carries its own
            # probe_septa_old and the live column holds the newer text. Comparing
            # against the wrong one of those reports a difference that is a rename,
            # not a regression.
            ref = CONTROL if CONTROL in acols else args.against
            d = float(np.abs(cols[CONTROL] - acols[ref]).max())
            worst = max(worst, d)
            # both are rounded to 4 dp on write, so anything at 1e-4 is the rounding
            print(f'  {s:<12} max|diff| = {d:.5f}   r = '
                  f'{corr(cols[CONTROL], acols[ref]):.4f}  '
                  f'{"ok" if d <= 2e-4 else "<-- DIFFERS"}   vs archived {ref}')
        if worst > 2e-4:
            print()
            print('  The retired caption did not reproduce. Something other than the')
            print('  text changed -- tiles, preprocessing or model weights. Everything')
            print('  below compares two different runs; fix this before reading it.')

    # The yardstick everything else is read against.
    base = np.array([corr(cols[a], cols[b])
                     for _, cols in data.values()
                     for a, b in itertools.combinations(ps.KEYS, 2)])
    others = [k for k in list(ps.KEYS) + ps.PROBE_KEYS if k not in cands + [CONTROL]]

    def neighbours(col):
        r = {k: mean_over(data, lambda c, k=k, col=col: corr(c[col], c[k]))
             for k in others}
        return sorted(r.items(), key=lambda kv: -kv[1])

    print()
    print('=' * 78)
    print('2. DIAGNOSIS -- what was the retired caption reading?')
    print('=' * 78)
    print(f'  baseline: {len(base)} pairwise r among the {len(ps.KEYS)} real columns')
    print(f'    median {np.median(base):+.3f}   p75 {np.percentile(base, 75):+.3f}'
          f'   p95 {np.percentile(base, 95):+.3f}   p99 {np.percentile(base, 99):+.3f}')
    print()
    nb = neighbours(CONTROL)
    rank = [k for k, _ in nb].index(MECHANISM) + 1
    r_mech = dict(nb)[MECHANISM]
    pct = float((base < r_mech).mean() * 100)
    print(f'  {CONTROL} nearest neighbours:')
    for k, v in nb[:5]:
        print(f'    {k:<22}{v:+.3f}' + ('   <-- the mechanism' if k == MECHANISM else ''))
    print()
    print(f'  {MECHANISM} ranks {rank} of {len(nb)} at r={r_mech:+.3f}, '
          f'the {pct:.1f}th percentile of the baseline.')
    if rank == 1 and pct >= 95:
        print('  Confirmed: out of everything scored, a bare "tumour nests" caption is')
        print('  the closest thing to the retired one, and by a margin the baseline')
        print('  says is large. The rest of the caption did not survive pooling.')
    elif rank <= 3 and pct >= 90:
        print('  Supported but not clean: the mechanism is among the nearest')
        print('  neighbours without dominating. Expect a smaller effect from the')
        print('  rewrite than the correlation table suggests.')
    else:
        print('  NOT supported. The retired column was keying on something else, so')
        print('  the stated reason for the rewrite is wrong even if a candidate scores')
        print('  better. Work out what it was reading before committing.')

    print()
    print('=' * 78)
    print('3. COMPARISON')
    print('=' * 78)
    print(f'  {"caption":<22}{"epi":>7}{"topz":>7}{"mean":>8}   nearest neighbours')
    rows = {}
    for k in cands + [CONTROL]:
        epi = mean_over(data, lambda c, k=k: np.mean([corr(c[k], c[e]) for e in EPITHELIAL]))
        tz = mean_over(data, lambda c, k=k: zmax(c[k]))
        mu = mean_over(data, lambda c, k=k: c[k].mean())
        nb = neighbours(k)[:3]
        rows[k] = dict(epi=epi, tz=tz, nb=nb)
        tag = '  (retired)' if k == CONTROL else ''
        print(f'  {k:<22}{epi:+7.2f}{tz:7.2f}{mu:+8.3f}   '
              + '  '.join(f'{n}{v:+.2f}' for n, v in nb) + tag)
    print()
    print('  epi = mean r with ' + '/'.join(EPITHELIAL) + '. Read the neighbours, not')
    print('  just epi: a caption can shed the epithelial terms and still land on the')
    print('  wrong stroma. An epithelial term in the top three is disqualifying.')

    print()
    print(f'  redundancy -- shared tiles in the top {args.top}, out of {args.top}:')
    print(f'  {"caption":<22}' + ''.join(f'{f:>17}' for f in FIBROUS))
    for k in cands + [CONTROL]:
        cells = []
        for f in FIBROUS:
            per = [len(set(np.argsort(-cols[k])[:args.top])
                       & set(np.argsort(-cols[f])[:args.top]))
                   for _, cols in data.values()]
            cells.append(f'{np.mean(per):>13.1f}/{args.top}')
        print(f'  {k:<22}' + ''.join(cells))
    print()
    print('  This is the redundancy test, not the correlation. Two captions that rank')
    print('  the same regions are one column under two names whatever their r says.')

    print()
    print('=' * 78)
    print('4. TOP TILES -- the only section that can confirm anything')
    print('=' * 78)
    for k in cands + [CONTROL]:
        print(f'  {k}')
        for s, (ids, cols) in data.items():
            order = np.argsort(-cols[k])[:3]
            z = (cols[k] - cols[k].mean()) / (cols[k].std() + 1e-9)
            print(f'    {s:<12} ' + '  '.join(f'{ids[i]}(z={z[i]:.2f})' for i in order))
    print()
    print('  Render them with:')
    print('    python path_report.py --tiles results/path_screen/survey/SAMPLE \\')
    print('        --scores SCORES --probes --out OUT      # probe columns')
    print('    python path_report.py ... --only_ecm --out OUT   # the live entry')
    print('  A caption that won section 3 and shows the wrong thing here has not been')
    print('  fixed; it has been broken in a way the statistics cannot see.')
    print()


if __name__ == '__main__':
    main()
