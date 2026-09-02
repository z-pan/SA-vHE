#!/usr/bin/env python3
"""HoVer-Net over the crops path_downstream_seg.py --tool crops wrote.

Runs in the neuroclear environment: tiatoolbox 1.5.1 with torch 2.1. The original
hover_net repository pins torch 1.6 and numpy 1.19, so tiatoolbox's own loader is the
way in rather than reviving those.

The crops are already at 0.25 um/px, HoVer-Net PanNuke's training resolution. Left at
0.621 the same fields returned 110-626 nuclei/mm2 against 467-1742 when resampled --
tile mode takes the pixels as given and never mentions magnification.

PanNuke gives a type per nucleus as well as an instance, which is the more interesting
half: whether a virtual stain preserves what an H&E-trained model calls neoplastic
versus inflammatory versus connective is a stronger claim than whether it preserves a
count.

The __main__ guard is load-bearing on Windows. Without it tiatoolbox's workers
re-import this module and the run sits with an empty cache directory forever.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

TYPE_NAMES = {0: 'background', 1: 'neoplastic', 2: 'inflammatory',
              3: 'connective', 4: 'dead', 5: 'epithelial'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='results/path_screen/survey/_downstream')
    ap.add_argument('--model', default='hovernet_fast-pannuke')
    ap.add_argument('--source', action='append', default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--loader_workers', type=int, default=2)
    ap.add_argument('--postproc_workers', type=int, default=4,
                    help='Post-processing, not inference, dominates: with these at 0 a '
                         'single region took 51 s of which about 35 s was watershed. '
                         'The __main__ guard above is what lets them be non-zero on '
                         'Windows at all.')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()

    import joblib
    from tiatoolbox.models import NucleusInstanceSegmentor

    crops = os.path.join(args.root, 'crops_hv')
    srcs = args.source or sorted(os.listdir(crops))
    seg = NucleusInstanceSegmentor(pretrained_model=args.model,
                                   num_loader_workers=args.loader_workers,
                                   num_postproc_workers=args.postproc_workers,
                                   batch_size=args.batch_size,
                                   auto_generate_mask=False)
    print(f'{args.model}; sources: {", ".join(srcs)}', flush=True)

    out_csv = os.path.join(args.root, f'hv_{args.model.replace("-", "_")}.csv')
    seen = set()
    if os.path.exists(out_csv) and not args.redo:
        with open(out_csv, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                seen.add((r['source'], r['id']))
    new = not os.path.exists(out_csv) or args.redo
    fh = open(out_csv, 'w' if new else 'a', newline='', encoding='utf-8')
    w = csv.writer(fh)
    if new:
        w.writerow(['source', 'id', 'inst_id', 'type', 'type_name',
                    'area_px', 'cx', 'cy', 'prob'])

    t0 = time.time()
    for src in srcs:
        d = os.path.join(crops, src)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        if args.limit:
            files = files[:args.limit]
        todo = [f for f in files if (src, f[:-4]) not in seen]
        if not todo:
            print(f'{src}: already done', flush=True)
            continue
        paths = [os.path.join(d, f) for f in todo]
        save = os.path.join(args.root, '_hv_cache', src)
        if os.path.exists(save):
            import shutil
            shutil.rmtree(save)
        res = seg.predict(paths, mode='tile', on_gpu=True, crash_on_exception=True,
                          save_dir=save)
        n = 0
        for p, rr in res:
            rid = os.path.splitext(os.path.basename(p))[0]
            dat = joblib.load(rr + '.dat')
            for k, v in dat.items():
                cy, cx = v['centroid'][1], v['centroid'][0]
                w.writerow([src, rid, k, v['type'],
                            TYPE_NAMES.get(v['type'], str(v['type'])),
                            int(v['box'][2] - v['box'][0]) * int(v['box'][3] - v['box'][1]),
                            round(float(cx), 1), round(float(cy), 1),
                            round(float(v.get('prob', 0) or 0), 3)])
                n += 1
        fh.flush()
        print(f'{src}: {len(todo)} regions, {n} nuclei, {time.time()-t0:.0f}s',
              flush=True)
    fh.close()
    print(f'-> {out_csv}')


if __name__ == '__main__':
    main()
