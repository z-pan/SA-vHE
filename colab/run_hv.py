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

    Two things a PNG needs spelled out.

    Scale. The crops were already resampled to 0.25 um/px locally, per region --
    the real H&E crops are stored at 512x512 whatever their physical extent, so
    their scale runs 0.44-0.77 um/px and no single factor covers them. A PNG has
    no mpp for the reader to find, so `wsireader_kwargs` supplies it. Declaring
    0.25 and requesting 0.25 leaves the scale at 1.0, which is what we want, and
    unlike `units='baseline'` it also makes the annotation coordinates meaningful.

    Tissue mask. With patch_mode=False the engine tries to build one via
    `wsireader.tissue_mask()`, which needs objective power or mpp and raised
    "MPP is None" on these files. Supplying mpp above would fix that too, but the
    mask is not wanted: these crops are already chosen regions, and letting a
    thresholder drop parts of them would silently remove nuclei from the count on
    some sources more than others.
    """
    kw = dict(patch_mode=False, save_dir=save_dir, overwrite=True,
              output_type='annotationstore', auto_get_mask=False)
    try:
        return seg.run(paths,
                       input_resolutions=[{'units': 'mpp', 'resolution': HV_MPP}],
                       wsireader_kwargs={'mpp': (HV_MPP, HV_MPP), 'power': 40},
                       **kw)
    except (TypeError, ValueError) as e:
        # Fall back to reading the pixels 1:1 with no scale declared at all. Same
        # sampling, less metadata; kept because both routes are documented and the
        # engine's acceptance of wsireader_kwargs varies by reader.
        print('mpp route failed (%s); falling back to baseline' % e)
        return seg.run(paths,
                       input_resolutions=[{'units': 'baseline', 'resolution': 1.0}],
                       **kw)


NAME_TO_CODE = {v.lower(): k for k, v in TYPE_NAMES.items()}
_dumped = [False]


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_store(db):
    """Each annotation carries a shapely geometry and a properties dict.

    Two shapes this had to be taught, both found by looking rather than guessing:

    `run()` hands back one entry per image, and that entry is a *list* of .db paths
    rather than a single path -- SQLiteStore was given the list and refused it.

    `type` is a class *name* string ('Neoplastic'), not the integer 1.x used, so
    int() on it raised. The name is the authoritative field here; the numeric code
    is derived where the name is one we know and left at -1 otherwise, rather than
    forcing a number the store never gave.
    """
    out = []
    paths = [str(x) for x in db] if isinstance(db, (list, tuple)) else [str(db)]
    for path in paths:
        store = SQLiteStore(path)
        for key, ann in store.items():
            p = dict(ann.properties or {})
            if not _dumped[0]:
                _dumped[0] = True
                print('  annotation properties:', {k: type(v).__name__
                                                   for k, v in p.items()})
                print('  first values:', {k: str(v)[:24] for k, v in p.items()})
            t = p.get('type', p.get('class', 0))
            if isinstance(t, str):
                name, code = t, NAME_TO_CODE.get(t.lower(), -1)
            else:
                code = int(_num(t))
                name = TYPE_NAMES.get(code, str(code))
            g = ann.geometry
            out.append(dict(key=str(key), type=code, type_name=name,
                            prob=_num(p.get('prob', p.get('probability',
                                                          p.get('score', 0)))),
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
    _k = next(iter(res)) if isinstance(res, dict) else 0
    _v = res[_k] if isinstance(res, dict) else res[0]
    print('  one entry:', type(_v).__name__,
          ('len %d, first %s' % (len(_v), type(_v[0]).__name__))
          if isinstance(_v, (list, tuple)) else str(_v)[:80])
    for p, db in _pairs(res, paths):
        nuc = read_store(db)
        im = cv2.imread(str(p))
        mm2 = im.shape[0] * im.shape[1] * HV_MPP ** 2 / 1e6
        tc = {}
        for v in nuc:
            tc[v['type_name']] = tc.get(v['type_name'], 0) + 1
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
                w.writerow([src, rid, v['key'], v['type'], v['type_name'],
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
