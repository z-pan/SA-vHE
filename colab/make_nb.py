#!/usr/bin/env python3
"""Generate ch5_hovernet_colab.ipynb.

Written as a generator rather than a hand-edited .ipynb so the cell text stays
readable and diffable; the notebook itself is a build artefact.

Rewritten 2026-09-05 for the correction-strength round. What changed, and why:

  The run loop is no longer inlined. It lives in colab/run_hv.py, fetched from
  GitHub at run time. tiatoolbox's API moved twice mid-project and each fix meant
  reloading the notebook, which on Colab means a new runtime: reinstall the
  package, re-extract a gigabyte of crops. Fetching from a URL keeps the runtime
  and makes a fix one commit. The notebook had kept a stale inlined copy of that
  loop alongside the fixed script, and it duly failed again on the bug the script
  had already fixed.

  MoNuSAC is gone. It is trained on four organs, none of them ovary, and its four
  classes are built for immune profiling. Its rankings disagreed with PanNuke's
  and there was no reason to prefer them.

  The extraction cell now clears any previous crops_hv first. Without that, a
  runtime that had already extracted a different zip silently re-ran versions that
  were already measured.

  The self-check section is trimmed to the two numbers that decide something:
  nucleus count against the stain-free TPAF reference, and the epithelial share.
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
# Ch5 细胞核分割 —— HoVer-Net (PanNuke) on Colab

对 5 个虚拟染色版本 × 148 个区域跑 HoVer-Net，得到逐个细胞核的位置、面积和类型。

## 这一轮要回答的问题

颜色校正的强度分三档。核染色深度（`hema_od`）相对真实 H&E：

| 档 | 版本 | 核染色深度 |
|---|---|---|
| 不校正 | `gray` `nuc_flat` `nuc_hi` `nuc_signed` `nuc_prof60` | 比真实深 57–87% |
| **中间** | **`globalonly` / `blend`** | **真实的 ±8% 内** |
| `final` | `gray_final` `nuc_flat_final` `nuc_hi_final` | 比真实浅 25–29% |

两端都测过，结论是它们把两个指标推向相反方向：不校正时上皮核比例对得上
（0.60–0.64，真实 0.632）但核检出多 2.3 倍；校正到 `final` 时核计数对得上
（−5~−8%，真实 −1%）但上皮核比例掉到 0.21–0.38。

中间这一档一个都没测过。本轮补上五个，找同时最接近真实 H&E 的那一个。

| 本轮版本 | 核染色深度 vs 真实 |
|---|---|
| `nuc_hi_blend` | −3% |
| `gray_globalonly` | −4% |
| `nuc_flat_blend` | +7% |
| `nuc_hi_globalonly` | −8% |
| `nuc_flat_globalonly` | −8% |

`gray_globalonly` 不做核增强，是对照，用来判断核增强在正确的染色深度下还有没有增益。

## 为什么在 Colab 上跑

HoVer-Net 的后处理是 CPU 密集的分水岭。本地 4 GB 显卡加少量核心跑一个区域要近一分钟，
A100 加十几个核心是 4 秒。

**运行时选 A100**：菜单 → 代码执行程序 → 更改运行时类型 → A100 GPU。
""")

md("""
## 1. 检查运行时
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import multiprocessing
print('CPU 核心', multiprocessing.cpu_count())
""")

md("""
## 2. 安装 tiatoolbox

装完**必须重启会话**：菜单 → 代码执行程序 → 重启会话。不重启的话已经载入的旧版
numpy / torch 还在内存里，后面会报形状或符号错误。

版本要求 `>=2.0`。1.5.1 要求 Python `<3.12`，而 Colab 已经是 3.12+，装不上。
2.x 的 API 与 1.x 不同（`MultiTaskSegmentor` 取代 `NucleusInstanceSegmentor`，
输出是 AnnotationStore 而非 joblib，类型是类别名字符串而非整数），
`run_hv.py` 是照 2.x 写的。
""")

code("""
!pip -q install "tiatoolbox>=2.0" 2>&1 | tail -5

# pip 失败不会让 cell 报错，所以显式验证一次 —— 这里原本无条件打印"安装好了"，
# 装失败也照打，直到下一格才炸。
import importlib
try:
    m = importlib.import_module('tiatoolbox')
    print('tiatoolbox', m.__version__, '—— 现在重启会话，然后从 §3 继续')
except Exception as e:
    raise SystemExit('安装失败: %s' % e)
""")

md("""
## 3. 挂载 Drive 并解压

先把本地的 `crops_hv_mid.zip`（1.31 GB）上传到 Drive 的 `ch5_downstream/`。

`shutil.rmtree` 那一行是必要的：如果这个运行时之前解压过别的 zip，
`/content/work/crops_hv` 已经存在，不清掉就会拿旧数据跑一遍。
这个坑踩过一次 —— 白跑半小时，且跑的是已经测过的版本。
""")

code("""
from google.colab import drive
drive.mount('/content/drive')

ZIP  = '/content/drive/MyDrive/ch5_downstream/crops_hv_mid.zip'
OUT  = '/content/drive/MyDrive/ch5_downstream_v2'    # 结果写回这里
root = '/content/work'

import os, shutil, zipfile, time
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(ZIP), '找不到 ' + ZIP
shutil.rmtree(root + '/crops_hv', ignore_errors=True)   # 见上
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    z.extractall(root)
print('解压完成 %.0fs' % (time.time() - t0))

for s in sorted(os.listdir(root + '/crops_hv')):
    print('  %-22s %d 张' % (s, len(os.listdir(root + '/crops_hv/' + s))))
""")

md("""
**确认上面打印的正好是这五行、各 148 张：**

```
  gray_globalonly        148 张
  nuc_flat_blend         148 张
  nuc_flat_globalonly    148 张
  nuc_hi_blend           148 张
  nuc_hi_globalonly      148 张
```

数量或名字不对就停下，别往下跑 —— 跑错版本要 50 分钟才会发现。
""")

md("""
## 4. 全量运行（可断点续跑）

运行逻辑不写在这一格，而是从 GitHub 取 `colab/run_hv.py` 执行。这样修 bug 只需
改一行推一次，不用重建运行时（重建意味着重装依赖、重新解压 1.3 GB）。

`?t=` 是时间戳，绕开 raw.githubusercontent 的 CDN 缓存（它给过旧文件）。
用 `$_t` 而不是 `{...}`，避免 IPython 的 `!` 行把花括号里的括号当成 shell 语法。

脚本从全局变量取 `root` 和 `OUT`（§3 已定义）。可选变量：`SUBSET` 限制每个版本的
区域数，`CHUNK` 改写回 Drive 的间隔（默认 20 个区域），`MODEL` 换模型。

**断点续跑**靠 `OUT` 下的 CSV：已写入的 (版本, 区域) 会被跳过，重跑本格即可续上。
Colab 在浏览器断开约 90 分钟后回收虚拟机，`/content` 随之消失，所以每 20 个区域
就写一次 Drive 并 fsync，掉线最多损失几分钟。

740 个区域，A100 上约 4 秒/区，**50 分钟左右**。
""")

code("""
import time
_t = int(time.time())
!wget -q --no-cache -O /content/run_hv.py "https://raw.githubusercontent.com/z-pan/SA-vHE/main/colab/run_hv.py?t=$_t"
exec(open('/content/run_hv.py').read())
""")

md("""
## 5. 跑完先看一眼，别等回本地才发现问题

两个参照值：

- **核密度**：TPAF 原图的 Cellpose 计数是 **1274 /mm²**，不经过任何染色。
  HoVer-Net 在真实 H&E 上是 1263，只差 1%，所以它的计数可信。已测版本里，
  不校正的三个是 +201~242%，`final` 档是 −5~−40%。中间档应落在两者之间。
- **上皮核比例**：真实 H&E 是 **0.632**（Neoplastic 0.579 + Non-Neoplastic
  Epithelial 0.053）。不校正的版本 0.60–0.64，`final` 档掉到 0.21–0.38。

同时接近这两个值的版本就是要找的。
""")

code("""
import pandas as pd, os

df = pd.read_csv(out_csv)
here = sorted(os.listdir(root + '/crops_hv'))
df = df[df.source.isin(here)]        # CSV 里还有前几轮的版本，只看本轮的

print('%-22s %5s %8s %10s' % ('version', '区数', '核数', '中位核/区'))
for s, g in df.groupby('source'):
    per = g.groupby('id').size()
    print('%-22s %5d %8d %10.0f' % (s, per.size, len(g), per.median()))

typed = df[df.type_name.str.lower() != 'background']
mix = (typed.groupby(['source', 'type_name']).size()
       / typed.groupby('source').size() * 100).unstack(fill_value=0).round(1)
EPI = [c for c in ('Neoplastic', 'Non-Neoplastic Epithelial') if c in mix.columns]
mix['上皮合计'] = mix[EPI].sum(axis=1)
print()
print('类型构成，占已分类核的百分比。真实 H&E 上皮合计 63.2')
print(mix.to_string())
""")

md("""
## 6. 取回结果

`hv_hovernet_fast_pannuke.csv` 在 Drive 的 `OUT` 目录下，下载到本地 `Downloads/`。
文件里同时含有之前几轮的版本，本地按版本名过滤后合并。

如果最后打印了 `N region(s) skipped:`，把那几行一并发回 —— 那是单个区域失败
（tiatoolbox 在没有检出任何核的区域上会 `KeyError('contours')`）。脚本已改成
单张重试并跳过，不会再拖垮整个版本，但要知道少了哪些区域。
""")

code("""
print(out_csv, '%.1f MB' % (os.path.getsize(out_csv) / 1e6))
from google.colab import files
files.download(out_csv)
""")

nb = {
    'cells': CELLS,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'provenance': [], 'gpuType': 'A100'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}
with io.open('ch5_hovernet_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote ch5_hovernet_colab.ipynb,', len(CELLS), 'cells')
