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
`saliency 目录（应为空）` 必须打印 `[]`，`real_A shape` 必须是 `[1, 1, 512, 512]`。

日志里的 `Time Taken` 除以迭代数就是每步耗时。每个 epoch 是 2910 次迭代
（trainA 张数），据此可以估算，但**不用为了省时间削减训练**——训练预算不是这项工作
关注的对象，训到收敛为止。
""")

md("""
## 5. 全量训练（可断点续跑）

学习率计划沿用 UTOM 的 `n_epochs 100 + n_epochs_decay 100`：前 100 个 epoch 恒定
2e-4，后 100 个线性衰减到 0。和 UTOM 用同一套计划，这样两者的差别只有 saliency 项。

`--save_epoch_freq 5`：掉线最多损失 5 个 epoch，同时给 §6 留下足够密的 checkpoint
用来判断收敛。checkpoint 直接写 Drive，`/content` 掉线就没了。

**掉线重连后**：把 `RESUME` 设 True，`START` 改成日志里最后完成的 epoch + 1，重跑本格。
""")

code(r"""
RESUME = False     # 掉线重连后改 True
START  = 1         # RESUME=True 时改成最后完成的 epoch + 1

flags = '--continue_train --epoch_count %d' % START if RESUME else ''
!python train.py \
  --dataroot {DATA} --name vanilla_cyclegan --model cycle_gan \
  --input_nc 1 --output_nc 3 --load_size 512 --crop_size 512 --batch_size 1 \
  --lambda_content 0 \
  --n_epochs 100 --n_epochs_decay 100 \
  --save_epoch_freq 5 --print_freq 500 --display_id 0 \
  --checkpoints_dir {CKPT} {flags}
""")

md("""
## 6. 判断收敛 —— 两条线一起看，只看 loss 会看错

对抗损失本身不能作为收敛判据：判别器和生成器互相推着走，loss 平了也可能是判别器赢了
而不是生成质量变好。所以看两样东西。

**第一，cycle 损失。** `loss_cycle_A` / `loss_cycle_B` 衡量 A→B→A 能不能还原回去，
不是对抗项，会真正收敛。它先降后平，平了说明生成器不再学到新的映射。

**第二，同一张图在不同 checkpoint 下的输出。** 收敛的直接证据是输出不再变化。
下一格把几个 epoch 的结果并排画出来。

本格随时可以在训练中途跑（另开一个 cell），不影响训练。
""")

code(r"""
import os, re, matplotlib.pyplot as plt

log = os.path.join(CKPT, 'vanilla_cyclegan', 'loss_log.txt')
keys = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
series = {k: [] for k in keys}
ep = []
for line in open(log):
    m = re.match(r'\(epoch: (\d+), iters: (\d+)', line)
    if not m:
        continue
    ep.append(int(m.group(1)) + int(m.group(2)) / 2910)
    for k in keys:
        v = re.search(k + r': ([\d.]+)', line)
        series[k].append(float(v.group(1)) if v else float('nan'))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k in ('cycle_A', 'cycle_B'):
    ax[0].plot(ep, series[k], lw=1, label=k)
ax[0].set_title('cycle 损失 —— 这条要平')
for k in ('D_A', 'G_A', 'D_B', 'G_B'):
    ax[1].plot(ep, series[k], lw=0.7, alpha=0.8, label=k)
ax[1].set_title('对抗损失 —— 震荡正常，看的是没有一方跑飞')
for a in ax:
    a.set_xlabel('epoch'); a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print('最后 %d 个 epoch 的 cycle_A 均值: %.3f' % (10, sum(
    v for e, v in zip(ep, series['cycle_A']) if e > max(ep) - 10) /
    max(1, sum(1 for e in ep if e > max(ep) - 10))))
""")

md("""
## 7. 同一张图，不同 checkpoint

收敛的直接证据：输出不再随 epoch 变化。挑 4 张 TPAF，用几个已保存的 checkpoint
各生成一次并排看。

若最后两三个 checkpoint 的输出肉眼已无差别，就可以停了。
""")

code("""
import os, glob, torch, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import networks

d = os.path.join(CKPT, 'vanilla_cyclegan')
eps = sorted(int(os.path.basename(f).split('_')[0])
             for f in glob.glob(os.path.join(d, '*_net_G_A.pth'))
             if os.path.basename(f).split('_')[0].isdigit())
show = eps[::max(1, len(eps) // 4)][-4:] if len(eps) > 4 else eps
print('已保存的 epoch:', eps)
print('展示:', show)

srcs = sorted(glob.glob(os.path.join(DATA, 'trainA', '*.png')))[:4]
G = networks.define_G(1, 3, 64, 'resnet_9blocks', 'instance', False,
                      'normal', 0.02, [0])
fig, axes = plt.subplots(len(srcs), len(show) + 1, figsize=(3 * (len(show) + 1),
                                                            3 * len(srcs)))
for r, sp in enumerate(srcs):
    a = np.array(Image.open(sp), dtype=np.float32) / 255.0
    axes[r, 0].imshow(a, cmap='gray'); axes[r, 0].set_ylabel(os.path.basename(sp)[:14],
                                                             fontsize=7)
    if r == 0:
        axes[r, 0].set_title('TPAF', fontsize=9)
    t = torch.from_numpy(a * 2 - 1)[None, None].cuda()
    for c, e in enumerate(show, 1):
        sd = torch.load(os.path.join(d, '%d_net_G_A.pth' % e), map_location='cuda')
        G.module.load_state_dict(sd) if hasattr(G, 'module') else G.load_state_dict(sd)
        G.eval()
        with torch.no_grad():
            out = G(t)[0].cpu().numpy().transpose(1, 2, 0)
        axes[r, c].imshow(np.clip(out * 0.5 + 0.5, 0, 1))
        if r == 0:
            axes[r, c].set_title('epoch %d' % e, fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")

md("""
## 8. 取回权重

推理只需要 `G_A`（TPAF → H&E 方向）。下载收敛处那个 epoch 的
`<epoch>_net_G_A.pth` 到本地 `UTOM-master/checkpoints/vanilla_cyclegan/`，
连同 `train_opt.txt` 一起（后者是训练参数的存档，写论文要用）。

之后本地用 `path_vhe_stain.py` 生成 148 个区域的 vHE，走与本文方法完全相同的
后续流程：颜色校正 → Cellpose → HoVer-Net → 染色分布指标。
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
