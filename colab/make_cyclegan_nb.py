#!/usr/bin/env python3
"""Generate ch5_cyclegan_colab.ipynb — the vanilla CycleGAN baseline.

Written as a generator rather than a hand-edited .ipynb so the cell text stays
readable and diffable; the notebook itself is a build artefact.

Why this baseline, and why it is trained rather than downloaded
---------------------------------------------------------------
UTOM is CycleGAN plus a saliency constraint. With SA-CUT dropped, UTOM was the only
comparison method left, and it shares a generator with the proposed method -- the two
differ only by the nucleus-mask input. Setting --lambda_content 0 removes the saliency
term and nothing else, so CycleGAN, UTOM and SA-CycleGAN become a progressive ablation
trained on identical data with identical hyperparameters.

None of the fifteen local checkpoints is saliency-free on this dataset, so it has to be
trained. 512 px with two generators and two discriminators does not fit the 4 GB laptop
GPU, hence Colab.
"""
import io
import json

CELLS = []


def lines(text):
    """Split for the .ipynb `source` field, which wants the newlines kept.

    A plain split drops them and Jupyter joins the list into one line, so every
    statement in a cell runs together and a leading `!` shell escape comes back as a
    bash syntax error on perfectly good code.
    """
    ls = text.strip('\n').split('\n')
    return [l + '\n' for l in ls[:-1]] + [ls[-1]]


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': lines(text)})


def code(text):
    CELLS.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': lines(text)})


md("""
# Ch5 对比方法 —— vanilla CycleGAN 训练

## 这个 baseline 是什么

UTOM = CycleGAN + saliency 约束。该约束在 `models/cycle_gan_model.py` 里是
`content_loss_value`，原先无权重直接加进 `loss_G`，没有办法把它关掉。现在加了
`--lambda_content`，设为 0 即得 **vanilla CycleGAN——同一份代码、同一份数据、
同一组超参，只少这一项**。

于是三个方法构成递进消融：

| | saliency 约束 | 核 mask 辅助 |
|---|---|---|
| **CycleGAN**（本 notebook） | ✗ | ✗ |
| **UTOM** | ✓ | ✗ |
| **SA-CycleGAN**（本文方法） | ✓ | ✓ |

本地 15 个 checkpoint 没有一个是这个数据集上不带 saliency 的，所以必须训练。

## 数据

病例 `23FJP075`（TPAF）/ `23FJD075`（H&E），**与五张评估片子（240703 / 240720 /
240729 / 240817 / 240828_pt1）零重叠**，已核对。

trainA 2910 张 512×512 单通道（其中 2424 张是离线增强副本，原始 486 张），
trainB 2482 张 512×512 RGB。

## 为什么不在本地跑

512 分辨率、两个生成器加两个判别器，4 GB 笔记本显卡放不下。运行时选 **A100**。
""")

md("""
## 1. 检查运行时
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch, multiprocessing
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('CPU 核心', multiprocessing.cpu_count())
""")

md("""
## 2. 取代码

从 GitHub 拉，不用上传——`--lambda_content` 的改动已经推上去了。
""")

code("""
import os, time
_t = int(time.time())
!rm -rf /content/SA-vHE
!git clone -q https://github.com/z-pan/SA-vHE.git /content/SA-vHE
%cd /content/SA-vHE
!git log --oneline -1
!grep -n "lambda_content" models/cycle_gan_model.py
!pip -q install dominate visdom 2>&1 | tail -2
""")

md("""
`grep` 必须打印出两处 `lambda_content`（一处是参数定义，一处在 `backward_G` 里）。
没有就说明拉到的是旧版本。
""")

md("""
## 3. 挂载 Drive 并解压数据

先把本地的 `vanilla_cycleGAN_train.zip`（1.35 GB）上传到 Drive 的 `ch5_cyclegan/`。

checkpoint 直接写到 Drive，不写 `/content`——Colab 掉线后 `/content` 会消失，
而这个训练一定跑不完一次会话。
""")

code("""
from google.colab import drive
drive.mount('/content/drive')

ZIP  = '/content/drive/MyDrive/ch5_cyclegan/vanilla_cycleGAN_train.zip'
CKPT = '/content/drive/MyDrive/ch5_cyclegan/checkpoints'   # 断点写这里
DATA = '/content/SA-vHE/datasets/train_data/vanilla_cycleGAN_train'

import os, shutil, zipfile, time
os.makedirs(CKPT, exist_ok=True)
assert os.path.exists(ZIP), '找不到 ' + ZIP
shutil.rmtree(DATA, ignore_errors=True)
os.makedirs(DATA, exist_ok=True)
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    z.extractall(DATA)
print('解压完成 %.0fs' % (time.time() - t0))
for d in ('trainA', 'trainB'):
    print('  %-8s %d 张' % (d, len(os.listdir(os.path.join(DATA, d)))))
""")

md("""
**确认打印 trainA 2910、trainB 2482。**
""")

md("""
## 4. 短程验证 —— 必须先跑这一格

只跑 1 个 epoch、只取 40 张图，**目的不是训练，是确认三件事**：

1. `--lambda_content 0` 真的把 saliency 项关掉了（日志里 `content` 相关的值应为 0，
   且 `checkpoints/*/saliency_A` 目录不应被创建）
2. 显存放得下 512 分辨率
3. **量出每次迭代要多久**，据此排全量训练的时间表

不要跳过这一格直接跑全量。上一次没有先做短程验证，代价是一次白跑的 A100 会话。
""")

code("""
import time, os
t0 = time.time()
!python train.py \\
  --dataroot {DATA} --name smoke_cyclegan --model cycle_gan \\
  --input_nc 1 --output_nc 3 --load_size 512 --crop_size 512 --batch_size 1 \\
  --lambda_content 0 \\
  --n_epochs 1 --n_epochs_decay 0 --max_dataset_size 40 \\
  --save_epoch_freq 1 --print_freq 10 --display_id 0 \\
  --checkpoints_dir /content/_smoke 2>&1 | tail -30
dt = time.time() - t0
print()
print('短程用时 %.0fs' % dt)
print('saliency 目录（应为空）:',
      [d for d in os.listdir('/content/_smoke/smoke_cyclegan')
       if 'saliency' in d] if os.path.isdir('/content/_smoke/smoke_cyclegan') else 'n/a')
""")

md("""
从上面日志里的 iteration 耗时估算全量：

```
每 epoch 迭代数 = 2910（trainA 张数）
UTOM 的 checkpoint 是 epoch 80，所以对比方法也训到 80 epoch 才公平
总迭代 = 2910 x 80 = 232,800
```

把短程测到的「秒/迭代」乘上去，就知道要几次会话。A100 上若是 0.15 s/迭代，
约 10 小时；Colab 单次会话通常撑不到，所以下一格支持断点续跑。
""")

md("""
## 5. 全量训练（可断点续跑）

`--continue_train` 会读 `--checkpoints_dir` 下的 `latest_net_*.pth` 接着跑，
`--epoch_count` 指定从第几个 epoch 继续。**第一次跑时把 `RESUME` 设为 False，
掉线后改成 True 并把 `START` 改成日志里最后完成的 epoch + 1。**

`--save_epoch_freq 5` 是在「掉线最多损失 5 个 epoch」和「Drive 写入开销」之间取的折中。
""")

code("""
RESUME = False     # 掉线重连后改 True
START  = 1         # RESUME=True 时改成最后完成的 epoch + 1

flags = '--continue_train --epoch_count %d' % START if RESUME else ''
!python train.py \\
  --dataroot {DATA} --name vanilla_cyclegan --model cycle_gan \\
  --input_nc 1 --output_nc 3 --load_size 512 --crop_size 512 --batch_size 1 \\
  --lambda_content 0 \\
  --n_epochs 80 --n_epochs_decay 0 \\
  --save_epoch_freq 5 --print_freq 200 --display_id 0 \\
  --checkpoints_dir {CKPT} {flags}
""")

md("""
## 6. 取回权重

推理只需要 `G_A`（TPAF → H&E 那个方向）。把它下载到本地
`UTOM-master/checkpoints/vanilla_cyclegan/` 下，再用 `path_vhe_stain.py`
生成 148 个区域的 vHE，走与本文方法完全相同的后续流程。
""")

code("""
import os, glob
d = os.path.join(CKPT, 'vanilla_cyclegan')
for f in sorted(glob.glob(os.path.join(d, '*_net_G_A.pth'))):
    print('%-28s %.0f MB' % (os.path.basename(f), os.path.getsize(f) / 1e6))
print()
print('训练参数存档:', os.path.join(d, 'train_opt.txt'))
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
with io.open('ch5_cyclegan_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote ch5_cyclegan_colab.ipynb,', len(CELLS), 'cells')
