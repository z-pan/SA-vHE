"""HoVer-Net over the prepared crops. Fetched and exec'd from a Colab cell.

Why this exists as a file rather than a notebook cell. The tiatoolbox API moved
under us twice, and each fix meant reloading the notebook from GitHub, which on
Colab means a new runtime -- reinstalling the package and re-extracting 1.46 GB.
Pulling the logic from a URL instead keeps the runtime and makes a fix two lines:

    !wget -q -O /content/run_hv.py https://raw.githubusercontent.com/z-pan/SA-vHE/main/colab/run_hv.py
    exec(open('/content/run_hv.py').read())

Expects `root` and `OUT` already defined (notebook section 3). Set `SUBSET` before
exec to limit regions per source; set `PROBE` to run the four-region check instead
of the full pass.

Written against tiatoolbox 2.1.3:

    MultiTaskSegmentor(model, batch_size=8, num_workers=0, weights=None,
                       *, device='cpu', verbose=True)
    .run(images, *, patch_mode, save_dir, input_resolutions, output_type, ...)

NucleusInstanceSegmentor is a deprecated wrapper around MultiTaskSegmentor in 2.x
and no longer takes `pretrained_model`, so the parent is used directly. Output is
an AnnotationStore .db, not the joblib .dat of 1.x, and area comes from the polygon
rather than a bounding box.
"""

import csv
import os
import shutil
import time

import cv2
import tiatoolbox
from packaging.version import Version

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

TYPE_NAMES = {0: 'background', 1: 'neoplastic', 2: 'inflammatory',
              3: 'connective', 4: 'dead', 5: 'epithelial'}
HV_MPP = 0.25

assert Version(tiatoolbox.__version__) >= Version('2.0'), \
    'written against the 2.x API; got ' + tiatoolbox.__version__
print('tiatoolbox', tiatoolbox.__version__, '<- record this; all six sources must '
      'go through one version')

_g = globals()
root = _g.get('root', '/content/work')
OUT = _g.get('OUT', '/content/drive/MyDrive/ch5_downstream')
SUBSET = _g.get('SUBSET', 0)
PROBE = _g.get('PROBE', False)
NW = _g.get('NW', min(8, max(1, os.cpu_count() - 2)))
BATCH = _g.get('BATCH', 32)
MODEL = _g.get('MODEL', 'hovernet_fast-pannuke')

seg = MultiTaskSegmentor(model=MODEL, batch_size=BATCH, num_workers=NW,
                         device='cuda', verbose=False)
print('model %s, workers %d, batch %d' % (MODEL, NW, BATCH))


def run_seg(paths, save_dir):
    """patch_mode=False is what 1.x called mode='tile': the image is larger than
    the model's input tile.

    input_resolutions is pinned to baseline. The crops were already resampled to
    0.25 um/px locally, per region -- the real H&E crops are stored at 512x512
    whatever their physical extent, so their scale runs 0.44-0.77 um/px and no
    single factor covers them. Letting the engine scale by mpp on top would scale
    twice, and a PNG carries no mpp for it to read anyway.
    """
    return seg.run(paths, patch_mode=False, save_dir=save_dir, overwrite=True,
                   input_resolutions=[{'units': 'baseline', 'resolution': 1.0}],
                   output_type='annotationstore')


def read_store(db):
    """Each annotation carries a shapely geometry and a properties dict."""
    out = []
    store = SQLiteStore(db)
    for key, ann in store.items():
        p = dict(ann.properties or {})
        g = ann.geometry
        out.append(dict(key=str(key), type=int(p.get('type', 0) or 0),
                        prob=float(p.get('prob', 0) or 0),
                        area_px=float(g.area),
                        cx=float(g.centroid.x), cy=float(g.centroid.y)))
    return out


def _pairs(res, paths):
    return res.items() if isinstance(res, dict) else zip(paths, res)


if PROBE:
    d = os.path.join(root, 'crops_hv', 'real_HE')
    paths = [os.path.join(d, f) for f in sorted(os.listdir(d))[:4]]
    t0 = time.time()
    res = run_seg(paths, '/content/_probe')
    dt = time.time() - t0
    print('run() returned', type(res).__name__)
    for p, db in _pairs(res, paths):
        nuc = read_store(db)
        im = cv2.imread(str(p))
        mm2 = im.shape[0] * im.shape[1] * HV_MPP ** 2 / 1e6
        tc = {}
        for v in nuc:
            k = TYPE_NAMES.get(v['type'], v['type'])
            tc[k] = tc.get(k, 0) + 1
        print('%-16s %5d nuclei  %7.0f /mm2   %s'
              % (os.path.basename(str(p)), len(nuc), len(nuc) / mm2, tc))
    print()
    print('%.1f s/region  ->  888 regions ~ %.1f h'
          % (dt / len(paths), 888 * dt / len(paths) / 3600))
    print('local reference on the same fields: 467-1742 /mm2. If this comes back at '
          'one or two hundred, the resolution is wrong again -- stop here.')
else:
    out_csv = os.path.join(OUT, 'hv_%s.csv' % MODEL.replace('-', '_'))
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            for r in csv.DictReader(fh):
                done.add((r['source'], r['id']))
        print('%d (source, region) already done, skipping them' % len(done))
    new = not os.path.exists(out_csv)
    fh = open(out_csv, 'w' if new else 'a', newline='')
    w = csv.writer(fh)
    if new:
        w.writerow(['source', 'id', 'inst_id', 'type', 'type_name',
                    'area_px', 'cx', 'cy', 'prob'])
    t0 = time.time()
    for src in sorted(os.listdir(os.path.join(root, 'crops_hv'))):
        d = os.path.join(root, 'crops_hv', src)
        files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        if SUBSET:
            files = files[:SUBSET]
        todo = [f for f in files if (src, f[:-4]) not in done]
        if not todo:
            print('%s: done already' % src)
            continue
        save = '/content/_hv/' + src
        if os.path.exists(save):
            shutil.rmtree(save)
        paths = [os.path.join(d, f) for f in todo]
        res = run_seg(paths, save)
        n = 0
        for p, db in _pairs(res, paths):
            rid = os.path.splitext(os.path.basename(str(p)))[0]
            for v in read_store(db):
                w.writerow([src, rid, v['key'], v['type'],
                            TYPE_NAMES.get(v['type'], v['type']),
                            round(v['area_px'], 1), round(v['cx'], 1),
                            round(v['cy'], 1), round(v['prob'], 3)])
                n += 1
        # Flushed per source, so a dropped session loses at most one source and the
        # skip list above picks up where it stopped.
        fh.flush()
        os.fsync(fh.fileno())
        print('%s: %d regions, %d nuclei, %.0fs elapsed'
              % (src, len(todo), n, time.time() - t0), flush=True)
    fh.close()
    print()
    print('-> ' + out_csv)
