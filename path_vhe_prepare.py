#!/usr/bin/env python3
"""Inventory the hand-made 拼接FOV folders and recover where each crop sits in its patch.

The order matters and is the reason this exists. Virtual staining has to run on the
whole TPAF patch and the region be cut out of the *result*, not the other way round: a
CycleGAN-family generator is not translation-equivariant, its InstanceNorm statistics
are computed over whatever it is handed, and its receptive field is truncated at the
border. Stain a 448 px crop and you get a different image than the same 448 px taken
out of the stained 1024 px patch -- different normalisation, different edge effects.
So the crop cannot be an input; it has to be a rectangle applied afterwards.

Which means the rectangle has to be known, and it was never written down -- only the
cropped image was saved. It is recoverable exactly: the crop is a literal sub-image, so
template matching finds it at correlation 1.000000 and the recovered box reproduces it
byte for byte. That equality is asserted per crop rather than assumed, because a crop
that had been resized, rotated or adjusted would still match at some position with a
high score, and the box would then be quietly wrong.

Layout it reads, per sample directory:

    拼接FOV/<numbers>/
        <patch>.tif                  one FOV, or candidate_<n>.tif for a stitched pair
        <patch>_crop<n>.tif          the region matching candidate n
        pick<nn>_<structure>_<tile>.png   the real H&E candidate, for reference

Folder names carry the candidate numbers: "13", or "12 & 26" when one patch covers
several candidates.

    python path_vhe_prepare.py                  # inventory, recover boxes, report
    python path_vhe_prepare.py --export stage   # also copy the patches out to stain
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil

import cv2
import numpy as np

DATASET = 'C:/Users/zpanp/projects/datasets/in_house_OV_TPAF_HE_pairs/TPAF-HE_pairs'
FOLDER = '拼接FOV'

SAMPLE_DIRS = {
    '240703': '240703HOC240717',
    '240720': '240720HOC241119-4',
    '240729': '240729HOC241119-4',
    '240817': '240817HOC240827-4',
    '240828_pt1': '240828HOC241127-4',
}

# "_crop12", and one folder where it was typed "_ceop46"; both mean the same thing and
# the typo is not worth a rename in the dataset, which is read-only in spirit.
CROP_RE = re.compile(r'_c[re]op(\d*)\.tiff?$', re.I)
PICK_RE = re.compile(r'^pick(\d+)_', re.I)
CAND_RE = re.compile(r'^candidate[_-]?([\d-]+?)(_crop)?\.tiff?$', re.I)
IMG_EXT = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread cannot open a path with non-ASCII characters on Windows, and the
    folder here is literally named 拼接FOV. It returns None rather than raising, so
    without this every image in the tree silently reads as empty."""
    data = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(data, flags)
    if im is None:
        raise SystemExit(f'cannot decode {path}')
    return im


def locate(parent, crop):
    """Where the crop sits in the parent, and whether it really is that rectangle."""
    p = cv2.cvtColor(parent, cv2.COLOR_BGR2GRAY) if parent.ndim == 3 else parent
    c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if c.shape[0] > p.shape[0] or c.shape[1] > p.shape[1]:
        return None, 0.0, False
    r = cv2.matchTemplate(p, c, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(r)
    h, w = c.shape[:2]
    exact = np.array_equal(parent[y:y + h, x:x + w], crop)
    return (x, y, w, h), float(score), exact


def classify(d, files):
    """-> (parent name, {number: crop name}, {number: pick name}, other images)

    Decided by content, not by filename. The naming has been inconsistent three ways
    already -- "_crop12", one "_ceop46", and a folder where candidate_8-10.tif is the
    patch while candidate_8.tif and candidate_10.tif are the crops, the exact opposite
    of everywhere else. So: the patch is the largest image, and anything whose name
    says it is a crop has to prove it by being a byte-exact sub-image of that patch.

    Source FOVs left beside a stitch are not byte-exact inside it -- the stitch blends
    them -- and their names do not claim to be crops, so they fall through to `other`
    and are ignored, which is what should happen to them.
    """
    imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
    picks = {}
    rest = []
    for f in imgs:
        mp = PICK_RE.match(f)
        if mp:
            picks[str(int(mp.group(1)))] = f
        else:
            rest.append(f)
    if not rest:
        return None, {}, picks, [], {}
    shapes = {}
    for f in rest:
        im = imread_u(os.path.join(d, f))
        shapes[f] = im
    parent = max(rest, key=lambda f: shapes[f].shape[0] * shapes[f].shape[1])

    crops, other, rejected = {}, [], {}
    for f in rest:
        if f == parent:
            continue
        m = CROP_RE.search(f) or CAND_RE.match(f)
        if not m:
            other.append(f)
            continue
        num = (m.group(1) or '').strip('-')
        box, score, exact = locate(shapes[parent], shapes[f])
        if box is None or not exact:
            rejected[f] = (box, score)
            other.append(f)
            continue
        crops.setdefault(num, f)
    return parent, crops, picks, other, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--out', default='results/path_screen/survey/_vhe')
    ap.add_argument('--export', default=None,
                    help='Copy each patch here under an ASCII name, ready for the '
                         'staining run. The dataset names carry spaces, brackets and a '
                         'Chinese folder, which several tools in this repo and most '
                         'shell pipelines mishandle.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.export:
        os.makedirs(args.export, exist_ok=True)

    rows, problems = [], []
    for sample, dirname in SAMPLE_DIRS.items():
        base = os.path.join(args.dataset, dirname, FOLDER)
        if not os.path.isdir(base):
            print(f'{sample:<12} no {FOLDER}')
            continue
        n_ok = n_folder = 0
        for sub in sorted(os.listdir(base)):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            n_folder += 1
            nums = [str(int(v)) for v in re.findall(r'\d+', sub)]
            files = sorted(os.listdir(d))
            parent, crops, picks, others, rejected = classify(d, files)
            if parent is None:
                problems.append(f'{sample}/{sub}: no image that could be the patch')
                continue
            for f, (box, score) in rejected.items():
                problems.append(f'{sample}/{sub}: {f} is named like a crop but is not a '
                                f'byte-exact region of {parent} (best score '
                                f'{score:.4f}) -- it was resized or altered')
            P = imread_u(os.path.join(d, parent))
            for n in nums:
                # a single-crop folder may name the crop without a number
                cname = crops.get(n) or (crops.get('') if len(nums) == 1 else None)
                if cname is None:
                    problems.append(f'{sample}/{sub}: no crop file for candidate {n}')
                    continue
                C = imread_u(os.path.join(d, cname))
                box, score, exact = locate(P, C)
                if box is None:
                    problems.append(f'{sample}/{sub}: crop {cname} is larger than the '
                                    'patch -- wrong parent?')
                    continue
                if not exact:
                    problems.append(f'{sample}/{sub} #{n}: best position ({box[0]},'
                                    f'{box[1]}) score {score:.4f} is NOT byte-identical '
                                    '-- the crop was resized or altered, so the box '
                                    'cannot be trusted')
                    continue
                x, y, w, h = box
                if (x, y) == (0, 0) and (w, h) == (P.shape[1], P.shape[0]):
                    # The crop is the whole patch, so there is no region to cut out of
                    # the stained result and nothing to compare against the 256 um H&E
                    # candidate. Almost certainly the patch was saved twice.
                    problems.append(f'{sample}/{sub} #{n}: {cname} IS the whole patch '
                                    f'({w}x{h}) -- no region was cropped')
                pname = picks.get(n)
                if pname is None:
                    problems.append(f'{sample}/{sub}: no pick png for candidate {n} '
                                    '(the real H&E side is missing)')
                rows.append(dict(
                    id=f'{sample}-{int(n):02d}', sample=sample, n=int(n), folder=sub,
                    patch=parent, patch_w=P.shape[1], patch_h=P.shape[0],
                    stitched=int(bool(CAND_RE.match(parent))),
                    n_source_fov=len(others) if CAND_RE.match(parent) else 1,
                    crop=cname, crop_x=x, crop_y=y, crop_w=w, crop_h=h,
                    crop_um=round(w * 0.621, 1),
                    he_pick=pname or '',
                    patch_path=os.path.join(d, parent),
                    crop_path=os.path.join(d, cname),
                    he_path=os.path.join(d, pname) if pname else '',
                ))
                n_ok += 1
        print(f'{sample:<12} {n_ok} regions from {n_folder} folders')

    # One patch can serve several candidates; stain it once.
    patches = {}
    for r in rows:
        patches.setdefault(r['patch_path'], []).append(r)
    print(f'\n{len(rows)} regions, {len(patches)} distinct patches to stain')
    sizes = sorted({(r['patch_w'], r['patch_h']) for r in rows})
    print(f'  patch sizes: ' + ', '.join(f'{w}x{h}' for w, h in sizes))
    cw = [r['crop_w'] for r in rows]
    print(f'  crop widths: {min(cw)}-{max(cw)} px = '
          f'{min(cw) * 0.621:.0f}-{max(cw) * 0.621:.0f} um')

    if args.export:
        used = {}
        for i, (p, group) in enumerate(sorted(patches.items())):
            g0 = group[0]
            key = f"{g0['sample']}_" + '_'.join(str(r['n']) for r in sorted(
                group, key=lambda r: r['n']))
            key = used.setdefault(p, key)
            dst = os.path.join(args.export, f'{key}.tif')
            shutil.copyfile(p, dst)
            for r in group:
                r['stage_name'] = f'{key}.tif'
        print(f'  -> {len(patches)} patches copied to {args.export}')
    else:
        for r in rows:
            r['stage_name'] = ''

    cols = ['id', 'sample', 'n', 'folder', 'stage_name', 'patch', 'patch_w', 'patch_h',
            'stitched', 'n_source_fov', 'crop', 'crop_x', 'crop_y', 'crop_w', 'crop_h',
            'crop_um', 'he_pick', 'patch_path', 'crop_path', 'he_path']
    mpath = os.path.join(args.out, 'vhe_manifest.csv')
    with io.open(mpath, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['sample'], r['n'])))
    print(f'  -> {mpath}')

    print()
    if problems:
        print(f'{len(problems)} thing(s) to look at:')
        for p in problems:
            print('  ' + p)
    else:
        print('every crop located and byte-identical at its recovered box')


if __name__ == '__main__':
    main()
