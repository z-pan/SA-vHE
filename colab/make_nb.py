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
所以 A100 带来的主要收益其实是**运行时附带的 12 个 vCPU**，而不是显卡本身 ——
换句话说，别指望 A100 把时间按显卡算力等比缩短；真正决定总时长的是 `N_POST`。

## 两个已踩过的坑，本 notebook 已处理
1. **输入分辨率**。HoVer-Net PanNuke 训练在 **0.25 µm/px**。直接喂 0.621 µm/px 的图
   只检出 110–626 核/mm²，重采样后 467–1742（**2–7 倍**），而且**不报任何错**。
   压缩包里的 crop 已在本地按各自的 µm/px 重采样到 0.25，`crop_scale.csv` 记录换算。
   真实 H&E 侧的 µm/px 逐区在 0.44–0.77 之间变，所以这一步不能"统一乘一个数"。
2. **多进程**。Windows 上必须 `__main__` 保护，否则 worker 重复导入、死锁在空缓存目录。
   Colab 是 Linux fork，无此问题，但 worker 数要匹配 vCPU，见 §4。
""")

md("""
## 1. 检查运行时

需要 GPU：菜单 → 代码执行程序 → 更改运行时类型 → **A100 GPU**。

本 notebook 的 worker 数和 batch 已按 A100 运行时（约 12 vCPU / 83 GB）设定。
若实际拿到的是 T4（约 2 vCPU / 12 GB），把 §4 的 `N_POST` 降到 2、`BATCH` 降到 8。

下一格会打印实际的 vCPU 数和显存 —— **以它为准**，不要假设。
""")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import multiprocessing
print('vCPU:', multiprocessing.cpu_count())
!free -g | head -2
""")

md("""
## 2. 安装 tiatoolbox

**不固定版本。** 原先钉在 1.5.1 是为了和本机一致，但 Colab 的 Python 已是 3.12+，
而 1.x 全系要求 `<3.12`，装不上。可用的只有 2.x。

这不影响可比性：**所有 HoVer-Net 结果都来自这里**，本机那次 1.5.1 只是冒烟测试，不进结果。
真正要守的是「所有来源用同一个版本跑完」，下一格会把版本号打出来，记下它。

⚠️ 2.0 有两处破坏性改动（`on_gpu=` → `device=`，`.predict()` → `.run()`），
§4 的适配层两种都试，不用你操心。

装完**需要重启运行时**。重启后从 §3 继续。
""")
code("""
!pip -q install "tiatoolbox>=2.0" 2>&1 | tail -5

# pip 失败不会让 cell 报错，所以在这里自己验一次 —— 上一版无条件打印「装好了」，
# 结果版本解析失败时也照样说成功。
import importlib, subprocess, sys
try:
    m = importlib.import_module('tiatoolbox')
    print()
    print('tiatoolbox', m.__version__, '安装成功')
    print('-> 点「代码执行程序 → 重启会话」，然后从 §3 继续')
except Exception as e:
    print()
    print('安装失败:', e)
    print('别继续。把上面 pip 的报错发出来。')
    sys.exit(1)
""")

md("""
## 3. 挂载 Drive 并解压

把本地的 `crops_hv.zip`（1.46 GB）上传到 Drive，路径填进 `ZIP`。
""")
code("""
from google.colab import drive
drive.mount('/content/drive')

ZIP = '/content/drive/MyDrive/ch5_downstream/crops_hv.zip'   # ← 改成你的路径
OUT = '/content/drive/MyDrive/ch5_downstream'                # 结果写回这里

import os, zipfile, time
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(ZIP), '找不到 ' + ZIP
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    z.extractall('/content/work')
print('解压完成 %.0fs' % (time.time() - t0))

root = '/content/work'
for s in sorted(os.listdir(root + '/crops_hv')):
    print('  %-14s %d 张' % (s, len(os.listdir(root + '/crops_hv/' + s))))
""")

md("""
## 4. 短程验证 —— 必须先跑这一格，不要跳过

先用 4 个区确认三件事，再开全量：

1. 权重能下载、模型能跑
2. **每区耗时** —— 推算全量时间，判断会不会撞上 Colab 会话上限
3. **核密度是否合理** —— 真实 H&E 在 0.25 µm/px 下应为数百到两千核/mm²。
   若只有一两百，说明分辨率又错了，**立刻停下来查**，不要开全量
""")
code("""
import os, csv, time, joblib, cv2, tiatoolbox
from tiatoolbox.models import NucleusInstanceSegmentor

print('tiatoolbox', tiatoolbox.__version__, '<- 记下这个版本号，所有来源必须同一个')

TYPE_NAMES = {0: 'background', 1: 'neoplastic', 2: 'inflammatory',
              3: 'connective', 4: 'dead', 5: 'epithelial'}
HV_MPP = 0.25
N_POST = min(8, max(1, os.cpu_count() - 2))   # 后处理是瓶颈，留 2 核给数据加载
BATCH = 32                                     # A100 显存充裕；T4 上改回 8


def make_seg(model='hovernet_fast-pannuke'):
    \"\"\"2.0 起构造器接受 device=，1.x 不接受；两种都试。\"\"\"
    kw = dict(pretrained_model=model, num_loader_workers=2,
              num_postproc_workers=N_POST, batch_size=BATCH,
              auto_generate_mask=False)
    try:
        return NucleusInstanceSegmentor(device='cuda', **kw)
    except TypeError:
        return NucleusInstanceSegmentor(**kw)


def run_seg(seg, paths, save_dir):
    \"\"\"2.0 把 predict() 改名 run()、on_gpu= 改成 device=。

    两个名字都试，因为 Colab 装到哪个大版本不由我们决定，而调错的表现是
    TypeError 而不是安静的错误结果 —— 这一点比大多数别的坑友好。
    \"\"\"
    for call in (
        lambda: seg.run(paths, mode='tile', device='cuda',
                        crash_on_exception=True, save_dir=save_dir),
        lambda: seg.predict(paths, mode='tile', on_gpu=True,
                            crash_on_exception=True, save_dir=save_dir),
    ):
        try:
            return call()
        except (AttributeError, TypeError):
            continue
    raise RuntimeError('predict/run 两种调用都不被接受，检查 tiatoolbox 版本')


seg = make_seg()
print('后处理 worker: %d, batch: %d' % (N_POST, BATCH))

d = root + '/crops_hv/real_HE'
probe = [os.path.join(d, f) for f in sorted(os.listdir(d))[:4]]
t0 = time.time()
res = run_seg(seg, probe, '/content/_probe')
dt = time.time() - t0

for p, rr in res:
    dat = joblib.load(rr + '.dat')
    im = cv2.imread(p)
    mm2 = im.shape[0] * im.shape[1] * HV_MPP ** 2 / 1e6
    tc = {}
    for v in dat.values():
        k = TYPE_NAMES.get(v['type'], v['type'])
        tc[k] = tc.get(k, 0) + 1
    print('%-16s %5d 核  %7.0f /mm2   %s'
          % (os.path.basename(p), len(dat), len(dat) / mm2, tc))
print()
print('%.1f s/区  ->  888 区约 %.1f 小时' % (dt / len(probe), 888 * dt / len(probe) / 3600))
print()
print('A100 + 12 vCPU 上预计 1-2 小时；若 > 8 小时，在 §5 设 SUBSET 只跑一部分区')
""")

md("""
## 5. 全量运行（可断点续跑）

Colab 会掉线。本格**每个来源跑完就把 CSV 追加写回 Drive 并 fsync**，
重连后重跑本格会自动跳过已完成的区。

`SUBSET` 设正整数则每个来源只跑前 N 个区；各来源按 id 排序，取到的是同一批区。
""")
code("""
SUBSET = 0          # 0 = 全部 148 区。A100 下应该不需要缩减

import os, csv, time, shutil, joblib

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
    res = run_seg(seg, [os.path.join(d, f) for f in todo], save)
    n = 0
    for p, rr in res:
        rid = os.path.splitext(os.path.basename(p))[0]
        for k, v in joblib.load(rr + '.dat').items():
            bb = v['box']
            w.writerow([src, rid, k, v['type'], TYPE_NAMES.get(v['type'], v['type']),
                        int(bb[2] - bb[0]) * int(bb[3] - bb[1]),
                        round(float(v['centroid'][0]), 1),
                        round(float(v['centroid'][1]), 1),
                        round(float(v.get('prob', 0) or 0), 3)])
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
  后续 5 类分布对比就得加置信度门槛；而且这个占比在真实与虚拟上可能不同，本身即是信息
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
    seg2 = make_seg('hovernet_fast-monusac')
    out2 = os.path.join(OUT, 'hv_hovernet_fast_monusac.csv')
    done2 = set()
    if os.path.exists(out2):
        with open(out2) as fh2:
            for r in csv.DictReader(fh2):
                done2.add((r['source'], r['id']))
    new2 = not os.path.exists(out2)
    fh2 = open(out2, 'w' if new2 else 'a', newline='')
    w2 = csv.writer(fh2)
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
        res = run_seg(seg2, [os.path.join(d, f) for f in todo], save)
        for p, rr in res:
            rid = os.path.splitext(os.path.basename(p))[0]
            for k, v in joblib.load(rr + '.dat').items():
                bb = v['box']
                w2.writerow([src, rid, k, v['type'], str(v['type']),
                             int(bb[2] - bb[0]) * int(bb[3] - bb[1]),
                             round(float(v['centroid'][0]), 1),
                             round(float(v['centroid'][1]), 1),
                             round(float(v.get('prob', 0) or 0), 3)])
        fh2.flush()
        os.fsync(fh2.fileno())
        print('%s done' % src, flush=True)
    fh2.close()
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
