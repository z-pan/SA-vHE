#!/usr/bin/env python3
"""Generate ch5_hovernet_colab.ipynb.

Written as a generator rather than a hand-edited .ipynb so the cell text stays
readable and diffable; the notebook itself is a build artefact.
"""
import io
import json

CELLS = []


def lines(text):
    """Split for the .ipynb `source` field, which wants the newlines kept.

    A plain split drops them, and Jupyter then joins the list into one line: every
    statement in a cell runs together, and the first cell -- which starts with a
    `!nvidia-smi` shell escape -- comes back as a bash syntax error on what is
    perfectly good code. The newline belongs to every line but the last.
    """
    ls = text.strip('\n').split('\n')
    return [l + '\n' for l in ls[:-1]] + [ls[-1]]


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {},
                  'source': lines(text)})


def code(text):
    CELLS.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': lines(text)})


md("""
# Ch5 下游核分割 — HoVer-Net (PanNuke) on Colab

对 6 个来源 × 148 个配对区跑 HoVer-Net，输出**逐核**的实例 + 5 类分类，回答：
**为真实 H&E 训练的模型，能不能在虚拟染色上找到 TPAF 说存在的那些核。**

来源：`real_HE` · `gray`(基线) · `gray_final`(UTOM 终版) · `nuc_hi_final`(当前 SA-CycleGAN)
· `nuc_flat`(平顶增强) · `nuc_signed`(平顶+压环)

## 为什么放 Colab
本机 RTX 3050 Ti (4 GB) 上 **51 秒/区**，888 区约 12.6 小时。

瓶颈**不是显存，是分水岭后处理**（51 秒里约 35 秒），受限于 CPU 核数。
所以 A100 带来的主要收益其实是**运行时附带的 12 个 vCPU**，而不是显卡本身。

## 三个已踩过的坑，本 notebook 已处理

1. **输入分辨率**。HoVer-Net PanNuke 训练在 **0.25 µm/px**。直接喂 0.621 µm/px 的图
   只检出 110–626 核/mm²，重采样后 467–1742（**2–7 倍**），而且**不报任何错**。
   压缩包里的 crop 已在本地按各自的 µm/px 重采样到 0.25 —— 真实 H&E 侧的 µm/px
   逐区在 0.44–0.77 之间变，所以这一步不能"统一乘一个数"。
   这里再用 `units='baseline'` 让模型 1:1 读像素，不做第二次缩放。
2. **tiatoolbox 2.x 换了整套 API**。`pretrained_model=` → `model=`（位置参数），
   `.predict()` → `.run()`，`mode='tile'` → `patch_mode=False`，
   输出从 joblib `.dat` 换成 AnnotationStore `.db`。1.x 装不上（要求 Python <3.12）。
3. **多进程**。Windows 上必须 `__main__` 保护；Colab 是 Linux fork，无此问题。
""")

md("""
## 1. 检查运行时

需要 GPU：菜单 → 代码执行程序 → 更改运行时类型 → **A100 GPU**。

下一格会打印实际的 vCPU 数和显存 —— **以它为准**，不要假设。
A100 运行时约 12 vCPU / 83 GB；若拿到 T4（约 2 vCPU / 12 GB），把 §4 的
`NW` 降到 2、`BATCH` 降到 8。
""")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import multiprocessing
print('vCPU:', multiprocessing.cpu_count())
!free -g | head -2
""")

md("""
## 2. 安装 tiatoolbox

**不固定版本。** Colab 的 Python 是 3.12+，而 tiatoolbox 1.x 全系要求 `<3.12`，
装不上；可用的只有 2.x。

这不影响可比性：**所有 HoVer-Net 结果都来自这里**，本机那次 1.5.1 只是冒烟测试、
不进结果。要守的是「六个来源用同一个版本跑完」，下一格会打印版本号，记下它。

装完**需要重启会话**（菜单 → 代码执行程序 → 重新启动会话）。重启后从 §3 继续。
""")
code("""
!pip -q install "tiatoolbox>=2.0" 2>&1 | tail -5

# pip 失败不会让 cell 报错。上一版在这里无条件打印「装好了」，于是装失败时
# 报错和成功提示一起出现 —— 谎报成功比报错更危险，因为人会带着错误前提往下走。
import importlib, sys
try:
    m = importlib.import_module('tiatoolbox')
    print()
    print('tiatoolbox', m.__version__, '安装成功')
    print('-> 菜单 → 代码执行程序 → 重新启动会话，然后从 §3 继续')
except Exception as e:
    print()
    print('安装失败:', e)
    print('别继续，把上面 pip 的报错发出来。')
    raise
""")

md("""
## 3. 挂载 Drive 并解压

把本地的 `crops_hv.zip`（1.46 GB）上传到 Drive，路径填进 `ZIP`。

重启会话后变量会清空，**这一格要重跑**；`/content` 下的文件还在，
所以解压那步第二次会很快。
""")
code("""
from google.colab import drive
drive.mount('/content/drive')

ZIP = '/content/drive/MyDrive/ch5_downstream/crops_hv.zip'   # ← 改成你的路径
OUT = '/content/drive/MyDrive/ch5_downstream'                # 结果写回这里

import os, zipfile, time
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(ZIP), '找不到 ' + ZIP
root = '/content/work'
if not os.path.isdir(root + '/crops_hv'):
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(root)
    print('解压完成 %.0fs' % (time.time() - t0))
else:
    print('已解压，跳过')

for s in sorted(os.listdir(root + '/crops_hv')):
    print('  %-14s %d 张' % (s, len(os.listdir(root + '/crops_hv/' + s))))
""")

md("""
## 4. 短程验证 —— 必须先跑这一格，不要跳过

先用 4 个区确认三件事，再开全量：

1. 权重能下载、模型能跑、输出能解析
2. **每区耗时** —— 推算全量时间
3. **核密度是否合理** —— 真实 H&E 在 0.25 µm/px 下应为**数百到两千核/mm²**。
   若只有一两百，说明分辨率环节又出了问题，**停下来查**，不要开全量

本机在同样的图上量到 467–1742 /mm²，可作对照。
""")
code("""
import os, csv, time, cv2, numpy as np, tiatoolbox
from packaging.version import Version

print('tiatoolbox', tiatoolbox.__version__, '<- 记下这个版本号，所有来源必须同一个')
assert Version(tiatoolbox.__version__) >= Version('2.0'), \\
    '本 notebook 按 2.x API 写；1.x 请用旧版 notebook'

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
from tiatoolbox.annotation.storage import SQLiteStore

TYPE_NAMES = {0: 'background', 1: 'neoplastic', 2: 'inflammatory',
              3: 'connective', 4: 'dead', 5: 'epithelial'}
HV_MPP = 0.25
NW = min(8, max(1, os.cpu_count() - 2))
BATCH = 32                       # A100 显存充裕；T4 上改回 8

# 2.1.3 签名: MultiTaskSegmentor(model, batch_size=8, num_workers=0,
#                                weights=None, *, device='cpu', verbose=True)
# NucleusInstanceSegmentor 是它的 deprecated 包装，直接用父类。
seg = MultiTaskSegmentor(model='hovernet_fast-pannuke', batch_size=BATCH,
                         num_workers=NW, device='cuda', verbose=False)
print('workers: %d, batch: %d' % (NW, BATCH))


def run_seg(seg, paths, save_dir):
    \"\"\"patch_mode=False 是 2.x 里 mode='tile' 的对应物 —— 图比模型的输入块大。

    input_resolutions 用 baseline: crop 已经在本地重采样到 0.25 um/px 了,
    再让引擎按 mpp 缩放一次就会缩两次。PNG 也没有 mpp 元数据可读。
    \"\"\"
    return seg.run(paths, patch_mode=False, save_dir=save_dir, overwrite=True,
                   input_resolutions=[{'units': 'baseline', 'resolution': 1.0}],
                   output_type='annotationstore')


def read_store(db):
    \"\"\"AnnotationStore 取代了 1.x 的 joblib .dat。

    每个 annotation 带一个 shapely 几何和一个 properties 字典。面积直接取多边形
    面积，比 1.x 时用 bbox 面积更准。
    \"\"\"
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


d = root + '/crops_hv/real_HE'
probe = [os.path.join(d, f) for f in sorted(os.listdir(d))[:4]]
t0 = time.time()
res = run_seg(seg, probe, '/content/_probe')
dt = time.time() - t0
print()
print('run() 返回类型:', type(res).__name__)

items = res.items() if isinstance(res, dict) else zip(probe, res)
for p, db in items:
    nuc = read_store(db)
    im = cv2.imread(str(p))
    mm2 = im.shape[0] * im.shape[1] * HV_MPP ** 2 / 1e6
    tc = {}
    for v in nuc:
        k = TYPE_NAMES.get(v['type'], v['type'])
        tc[k] = tc.get(k, 0) + 1
    print('%-16s %5d 核  %7.0f /mm2   %s'
          % (os.path.basename(str(p)), len(nuc), len(nuc) / mm2, tc))
print()
print('%.1f s/区  ->  888 区约 %.1f 小时' % (dt / len(probe), 888 * dt / len(probe) / 3600))
print('本机对照: 467-1742 /mm2。若这里只有一两百，先别开全量。')
""")

md("""
## 5. 全量运行（可断点续跑）

Colab 会掉线。本格**每个来源跑完就把 CSV 追加写回 Drive 并 fsync**，
重连后重跑本格会自动跳过已完成的区。

`SUBSET` 设正整数则每个来源只跑前 N 个区；各来源按 id 排序，取到的是同一批区。
""")
code("""
SUBSET = 0          # 0 = 全部 148 区。A100 下应该不需要缩减

import os, csv, time, shutil

out_csv = os.path.join(OUT, 'hv_hovernet_fast_pannuke.csv')
done = set()
if os.path.exists(out_csv):
    with open(out_csv) as fh:
        for r in csv.DictReader(fh):
            done.add((r['source'], r['id']))
    print('已完成 %d 个 (source, region)，将跳过' % len(done))

new = not os.path.exists(out_csv)
fh = open(out_csv, 'w' if new else 'a', newline='')
w = csv.writer(fh)
if new:
    w.writerow(['source', 'id', 'inst_id', 'type', 'type_name',
                'area_px', 'cx', 'cy', 'prob'])

t0 = time.time()
for src in sorted(os.listdir(root + '/crops_hv')):
    d = root + '/crops_hv/' + src
    files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    if SUBSET:
        files = files[:SUBSET]
    todo = [f for f in files if (src, f[:-4]) not in done]
    if not todo:
        print('%s: 已完成' % src)
        continue
    save = '/content/_hv/' + src
    if os.path.exists(save):
        shutil.rmtree(save)
    paths = [os.path.join(d, f) for f in todo]
    res = run_seg(seg, paths, save)
    items = res.items() if isinstance(res, dict) else zip(paths, res)
    n = 0
    for p, db in items:
        rid = os.path.splitext(os.path.basename(str(p)))[0]
        for v in read_store(db):
            w.writerow([src, rid, v['key'], v['type'],
                        TYPE_NAMES.get(v['type'], v['type']),
                        round(v['area_px'], 1), round(v['cx'], 1),
                        round(v['cy'], 1), round(v['prob'], 3)])
            n += 1
    fh.flush()
    os.fsync(fh.fileno())
    print('%s: %d 区, %d 核, 累计 %.0fs' % (src, len(todo), n, time.time() - t0),
          flush=True)
fh.close()
print()
print('-> ' + out_csv)
""")

md("""
## 6. 快速核对 —— 跑完就看，别等回本地才发现问题

三件要看的事：

- **各来源核密度**：`real_HE` 是参照，vHE 各变体应在同一量级
- **type=0 (未分类) 占比**：PanNuke 的 0 类表示模型没给出类型。占比过高说明置信度不足，
  后续 5 类分布对比就得加门槛；而且这个占比在真实与虚拟上可能不同，本身即是信息
- **5 类构成**：vHE 的核类型分布是否与真实 H&E 一致 —— 比"数量对得上"更强的证据
""")
code("""
import pandas as pd, cv2, os

df = pd.read_csv(out_csv)
area = {}
for src, rid in df[['source', 'id']].drop_duplicates().itertuples(index=False):
    p = '%s/crops_hv/%s/%s.png' % (root, src, rid)
    if os.path.exists(p):
        im = cv2.imread(p)
        area[(src, rid)] = im.shape[0] * im.shape[1] * 0.25 ** 2 / 1e6

g = df.groupby(['source', 'id']).size().reset_index(name='n')
g['mm2'] = [area.get((s, i), float('nan')) for s, i in zip(g['source'], g['id'])]
g['dens'] = g['n'] / g['mm2']
print('每区核密度 /mm2 (中位):')
print(g.groupby('source')['dens'].median().round(0).to_string())
print()
print('type=0 (未分类) 占比 %:')
print(df.groupby('source')['type'].apply(lambda s: (s == 0).mean() * 100).round(1).to_string())
print()
print('5 类构成 % (排除 type=0):')
t = df[df.type > 0].groupby(['source', 'type_name']).size().unstack(fill_value=0)
print((100 * t.div(t.sum(1), axis=0)).round(1).to_string())
""")

md("""
## 7.（可选）第二个模型作独立佐证

MoNuSAC 是另一批训练数据、另一套类别。两个模型给出一致结论，比单个强得多 ——
审稿人无法说"你挑了个对你有利的模型"。

跑之前确认 §5 的耗时还有余量。
""")
code("""
RUN_SECOND = False   # 改 True 再运行

if not RUN_SECOND:
    print('跳过')
else:
    seg2 = MultiTaskSegmentor(model='hovernet_fast-monusac', batch_size=BATCH,
                              num_workers=NW, device='cuda', verbose=False)
    out2 = os.path.join(OUT, 'hv_hovernet_fast_monusac.csv')
    done2 = set()
    if os.path.exists(out2):
        with open(out2) as f2:
            for r in csv.DictReader(f2):
                done2.add((r['source'], r['id']))
    new2 = not os.path.exists(out2)
    f2 = open(out2, 'w' if new2 else 'a', newline='')
    w2 = csv.writer(f2)
    if new2:
        w2.writerow(['source', 'id', 'inst_id', 'type', 'type_name',
                     'area_px', 'cx', 'cy', 'prob'])
    for src in sorted(os.listdir(root + '/crops_hv')):
        d = root + '/crops_hv/' + src
        files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        if SUBSET:
            files = files[:SUBSET]
        todo = [f for f in files if (src, f[:-4]) not in done2]
        if not todo:
            continue
        save = '/content/_hv2/' + src
        if os.path.exists(save):
            shutil.rmtree(save)
        paths = [os.path.join(d, f) for f in todo]
        res = run_seg(seg2, paths, save)
        items = res.items() if isinstance(res, dict) else zip(paths, res)
        for p, db in items:
            rid = os.path.splitext(os.path.basename(str(p)))[0]
            for v in read_store(db):
                w2.writerow([src, rid, v['key'], v['type'], str(v['type']),
                             round(v['area_px'], 1), round(v['cx'], 1),
                             round(v['cy'], 1), round(v['prob'], 3)])
        f2.flush()
        os.fsync(f2.fileno())
        print('%s done' % src, flush=True)
    f2.close()
    print('-> ' + out2)
""")

md("""
## 8. 取回结果

`hv_hovernet_fast_pannuke.csv` 在 Drive 的 `OUT` 目录下。下载后放到本地：

```
UTOM-master/results/path_screen/survey/_downstream/hv_hovernet_fast_pannuke.csv
```

本地的 Cellpose 结果（7 个来源，**含 TPAF 原图**）会与它合并出最终对比表。
TPAF 那一列是关键 —— 它给出"这个视野里究竟有多少核"，独立于任何染色，
是判断 vHE 保留了多少、H&E 工具找回了多少的参照。

⚠️ `area_px` 在 2.x 下是**多边形面积**，1.x 是 bbox 面积，两者不可混用。
本次全部来自 2.x，内部一致。
""")

nb = {
    'cells': CELLS,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'A100'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}
with io.open('ch5_hovernet_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote ch5_hovernet_colab.ipynb,', len(CELLS), 'cells')
