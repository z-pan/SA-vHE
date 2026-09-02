#!/usr/bin/env python3
"""Attach the hand-made candidate -> TPAF FOV correspondences to the candidate set.

The estimated TPAF columns path_locate.py writes are a guess from an unverified
assumption, and on five of the six samples they name files that do not exist -- the
grid constants they use were counted off 240817 alone. These files replace the guess
with a reading of the images, which is the only thing that settles it: TPAF and H&E are
different modalities of a shifted field, so a correspondence is confirmed by looking,
not by transforming coordinates.

Input, one per sample directory in the dataset:

    TPAF_candidates_patch.txt
    12、14：<file>.tif
    13：<file>.tif、<file>.tif        # the region spans several FOVs
    9：（没有匹配）                    # looked, found nothing

Numbers are the pick numbers on that sample's selection_map.png. Several numbers before
the colon means one FOV covers all of them; several filenames after it means the region
straddles a boundary. Both are kept as they are -- a region needing two FOVs is a fact
about the region.

Everything is checked against what is on disk and against selection.csv, because the
two are edited by different hands at different times and nothing else would notice them
drifting apart. In particular the pick numbers are assigned per path_locate.py run: if
that is rerun with different arguments the numbering shifts and a correspondence file
written against the old numbers becomes silently wrong. The paired-up crops this writes
are the check on that -- if a pair does not look like the same tissue, the numbering
moved.

    python path_tpaf_link.py                 # validate, report, write tpaf_links.csv
    python path_tpaf_link.py --pairs         # also write side-by-side H&E/TPAF images
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re

import cv2
import numpy as np

DATASET = ('C:/Users/zpanp/projects/datasets/in_house_OV_TPAF_HE_pairs/TPAF-HE_pairs')

# Which dataset directory holds each survey sample's TPAF, and whether its
# correspondence file numbers refer to that sample. 240828 was tiled as two pieces but
# imaged as one; its file covers pt1, the fragment TPAF actually reaches.
SAMPLE_DIRS = {
    '240703': '240703HOC240717',
    '240720': '240720HOC241119-4',
    '240729': '240729HOC241119-4',
    '240817': '240817HOC240827-4',
    '240828_pt1': '240828HOC241127-4',
}

LINKFILE = 'TPAF_candidates_patch.txt'
NOMATCH = re.compile(r'没有匹配|no match|none', re.I)


def parse(path):
    """-> {pick number: [filenames]}, plus the numbers explicitly marked unmatched."""
    links, unmatched, bad = {}, set(), []
    for lineno, raw in enumerate(io.open(path, encoding='utf-8'), 1):
        line = raw.strip()
        if not line:
            continue
        # full-width and ASCII colon both appear; split on the first either way
        m = re.match(r'^\s*([\d\s、,，]+?)\s*[:：]\s*(.*)$', line)
        if not m:
            # the header line "240817 对应：" has no numbers before the colon
            if not re.search(r'\.tif', line):
                continue
            bad.append((lineno, line))
            continue
        nums = [int(v) for v in re.findall(r'\d+', m.group(1))]
        rest = m.group(2)
        if not nums:
            continue
        # Split on the separators first, then take the filename out of each piece.
        # Matching filenames directly out of the whole line swallows the leading
        # separator into the second and later names -- the paths then do not resolve,
        # and it looks like a missing file rather than a parsing bug.
        files = []
        for piece in re.split(r'[、,，]', rest):
            m2 = re.search(r'([^（()]*?\.tif)', piece)
            if m2:
                f = m2.group(1).strip()
                if f and f not in files:
                    files.append(f)
        if not files:
            if NOMATCH.search(rest):
                unmatched.update(nums)
            else:
                bad.append((lineno, line))
            continue
        for n in nums:
            links.setdefault(n, [])
            for f in files:
                if f not in links[n]:
                    links[n].append(f)
    return links, unmatched, bad


def index_tifs(root):
    """filename -> full path, for every TIF under the sample directory."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                out.setdefault(f, os.path.join(dirpath, f))
    return out


def load_regions(survey, sample):
    p = os.path.join(survey, sample, 'candidates', 'selection.csv')
    rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
    regions = {}
    for r in rows:
        n = int(r['n'])
        regions.setdefault(n, dict(tile_id=r['tile_id'], tier=int(r.get('tier', 1) or 1),
                                   structures=[], y_mm=r['y_mm'], x_mm=r['x_mm']))
        regions[n]['structures'].append(r['structure'])
    return regions


def fit(im, h):
    if im is None:
        return np.full((h, h, 3), 40, np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = im[..., :3]
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s = h / im.shape[0]
    return cv2.resize(im, (max(1, int(im.shape[1] * s)), h))


def label(im, text, colour=(255, 255, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1,
                cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey', default='results/path_screen/survey')
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--out', default='results/path_screen/survey/_candidates')
    ap.add_argument('--pairs', action='store_true',
                    help='Also write one H&E-beside-TPAF image per linked region, into '
                         '{out}/pairs/. This is the only check on whether the pick '
                         'numbers in the text file still mean what they meant when it '
                         'was written -- look at them.')
    ap.add_argument('--pair_px', type=int, default=420)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows, problems = [], []
    print()
    for sample, dirname in SAMPLE_DIRS.items():
        sdir = os.path.join(args.dataset, dirname)
        lpath = os.path.join(sdir, LINKFILE)
        if not os.path.exists(lpath):
            print(f'{sample:<12} no {LINKFILE}')
            continue
        links, unmatched, bad = parse(lpath)
        tifs = index_tifs(sdir)
        regions = load_regions(args.survey, sample)

        missing_file = {n: [f for f in fs if f not in tifs] for n, fs in links.items()}
        missing_file = {n: v for n, v in missing_file.items() if v}
        unknown_n = sorted(n for n in links if n not in regions)
        linked = sorted(n for n in links if n in regions)
        nolink = sorted(set(regions) - set(links) - unmatched)

        n_files = len({f for fs in links.values() for f in fs})
        multi = sum(1 for n in linked if len(links[n]) > 1)
        print(f'{sample:<12} {len(linked)}/{len(regions)} regions linked  '
              f'({n_files} distinct FOVs, {multi} regions span >1)')
        if unmatched:
            print(f'  {len(unmatched)} marked no-match: '
                  + ' '.join(str(n) for n in sorted(unmatched)))
        if nolink:
            print(f'  {len(nolink)} not in the file: '
                  + ' '.join(str(n) for n in nolink))
        for lineno, line in bad:
            problems.append(f'{sample}: line {lineno} not understood -- {line[:70]}')
        if unknown_n:
            problems.append(f'{sample}: numbers not in selection.csv (the pick numbers '
                            f'may have shifted): {unknown_n}')
        for n, fs in sorted(missing_file.items()):
            for f in fs:
                problems.append(f'{sample} #{n}: no such file on disk -- {f}')

        # one row per (region, fov)
        for n in linked:
            r = regions[n]
            for i, f in enumerate(links[n]):
                all_rows.append(dict(
                    id=f'{sample}-{n:02d}', sample=sample, n=n, tier=r['tier'],
                    tile_id=r['tile_id'], y_mm=r['y_mm'], x_mm=r['x_mm'],
                    structures=' '.join(sorted(set(r['structures']))),
                    n_fov=len(links[n]), fov_i=i + 1,
                    tpaf_file=f, tpaf_path=tifs.get(f, ''),
                ))
        for n in sorted(unmatched & set(regions)):
            r = regions[n]
            all_rows.append(dict(
                id=f'{sample}-{n:02d}', sample=sample, n=n, tier=r['tier'],
                tile_id=r['tile_id'], y_mm=r['y_mm'], x_mm=r['x_mm'],
                structures=' '.join(sorted(set(r['structures']))),
                n_fov=0, fov_i=0, tpaf_file='(no match)', tpaf_path=''))

        if args.pairs:
            pdir = os.path.join(args.out, 'pairs', sample)
            os.makedirs(pdir, exist_ok=True)
            for f in os.listdir(pdir):
                os.remove(os.path.join(pdir, f))
            cdir = os.path.join(args.survey, sample, 'candidates')
            for n in linked:
                r = regions[n]
                he = None
                for st in r['structures']:
                    q = os.path.join(cdir, f'pick{n:02d}_{st}_{r["tile_id"]}.png')
                    if os.path.exists(q):
                        he = cv2.imread(q)
                        break
                cells = [label(fit(he, args.pair_px),
                               f'{sample} #{n}  {r["tile_id"]}', (80, 80, 255))]
                for f in links[n]:
                    p = tifs.get(f)
                    tp = cv2.imread(p, cv2.IMREAD_UNCHANGED) if p else None
                    cells.append(label(fit(tp, args.pair_px), f[-34:], (80, 255, 255)))
                gap = np.full((args.pair_px, 8, 3), 255, np.uint8)
                out = cells[0]
                for c in cells[1:]:
                    out = np.hstack([out, gap, c])
                cv2.imwrite(os.path.join(pdir, f'{sample}-{n:02d}.png'), out)
            print(f'  -> {pdir}  ({len(linked)} pair images)')

    cols = ['id', 'sample', 'n', 'tier', 'tile_id', 'y_mm', 'x_mm', 'structures',
            'n_fov', 'fov_i', 'tpaf_file', 'tpaf_path']
    cpath = os.path.join(args.out, 'tpaf_links.csv')
    with io.open(cpath, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(all_rows, key=lambda r: (r['sample'], r['n'], r['fov_i'])))

    print()
    if problems:
        print(f'{len(problems)} problem(s):')
        for p in problems:
            print('  ' + p)
    else:
        print('no problems: every filename resolves, every number is a real pick')
    linked_ids = {r['id'] for r in all_rows if r['tpaf_path']}
    print(f'\n{len(linked_ids)} regions linked, {len(all_rows)} rows -> {cpath}')


if __name__ == '__main__':
    main()
