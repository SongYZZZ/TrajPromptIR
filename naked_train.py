## naked_train.py — 裸版训练循环：不用 Lightning，看清"模型是怎么训练出来的"
##
## 学习目标：
##   1. 亲手看到训练的本质循环（对应路线里的 6 行伪代码）
##   2. loss 从 ~0.09 一路下降，验证集 PSNR 一路上升
##   3. 训练完的模型真的能复原一张图（虽然只有 300 步，很弱）
##
## 数据：skimage 自带 4 张图 + 随机 128x128 裁剪 + 高斯噪声 σ=25
##       （σ=25 是官方训练配置 denoise_25 之一，不需要下载大数据集）
## 模型：PromptIR(decoder=True)，随机初始化——从"什么都不会"开始学

import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor

from net.model import PromptIR

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ================= 0. 造数据：3 张训练图 + 1 张验证图 =================
# 每张训练样本 = 随机位置裁 128x128 + 随机翻转（数据增强） + 加 σ=25 高斯噪声
def get_rgb_images(names):
    imgs = []
    for name in names:
        try:
            imgs.append(getattr(data, name)())
        except AttributeError:
            pass  # 该版本 skimage 没有这张图就跳过
    return imgs

train_imgs = get_rgb_images(['astronaut', 'rocket', 'chelsea'])
val_img = data.coffee()

SIGMA = 25
def random_patch(img, size=128):
    H, W = img.shape[:2]
    h = random.randint(0, H - size)
    w = random.randint(0, W - size)
    p = img[h:h+size, w:w+size]
    if random.random() < 0.5:
        p = p[:, ::-1].copy()  # .copy() 消除负步长视图，torch 不接受负步长
    return p

def make_pair(img):
    clean = random_patch(img)
    noisy = np.clip(clean.astype(np.float64) + np.random.randn(*clean.shape) * SIGMA,
                    0, 255).astype(np.uint8)
    return (ToTensor()(noisy)[None].to(device),   # degraded: 退化图
            ToTensor()(clean)[None].to(device))   # clean:    答案

# ================= 1. 模型与优化器 =================
net = PromptIR(decoder=True).to(device).train()   # 随机初始化，不加载官方权重
n_params = sum(p.numel() for p in net.parameters())
print('PromptIR 参数量: %.1f M' % (n_params / 1e6))
print('注意：模型本身只是一个 nn.Module，与 Lightning 无关；'
      '官方 train.py 只是把下面这个循环包进了 Lightning 的 training_step')

loss_fn = nn.L1Loss()
optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4)   # 与官方一致的 AdamW + lr

# 固定一个验证 patch（噪声固定），用于公平比较训练前后的表现
val_noisy, val_clean = make_pair(val_img)

# ================= 2. 训练循环（整个深度学习的本质）=================
print()
print('iter | train loss | val PSNR')
for it in range(300):
    degraded, clean = make_pair(random.choice(train_imgs))

    restored = net(degraded)            # ① 前向：退化图 -> 复原图
    loss = loss_fn(restored, clean)     # ② 算差距：|复原图 - 答案| 的均值
    optimizer.zero_grad()               # ③ 清空上一轮的梯度（否则会累加） 
    loss.backward()                     # ④ 反向传播：算出每个参数该往哪个方向改
    optimizer.step()                    # ⑤ 按梯度真正修改参数（lr 控制步长）

    if it % 10 == 0 or it == 299:
        net.eval()
        with torch.no_grad():
            val_out = net(val_noisy)
        net.train()
        val_psnr = psnr(val_clean[0].cpu().numpy().transpose(1, 2, 0),
                        val_out[0].cpu().numpy().transpose(1, 2, 0), data_range=1.0)
        print('%4d |   %.5f    |  %.2f dB' % (it, loss.item(), val_psnr))

# ================= 3. 训练完，拿这个"自己练出来的模型"复原一张真正的图 =================
torch.save(net.state_dict(), 'ckpt/naked_train_300.pt')
print()
print('saved: ckpt/naked_train_300.pt')

net.eval()
astro_noisy = ToTensor()(np.array(Image.open('test/demo/astro_noisy25.png').convert('RGB')))[None].to(device)
with torch.no_grad():
    astro_restored = net(astro_noisy)
out_np = (astro_restored[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(out_np).save('output/demo/astro_naked_train.png')

astro_clean = ToTensor()(np.array(Image.open('test/demo/astro_clean.png').convert('RGB')))[None]
print('裸训练(300步)模型在 astro_noisy25 上的恢复 PSNR: %.2f dB'
      % psnr(astro_clean[0].numpy().transpose(1, 2, 0),
             astro_restored[0].cpu().numpy().transpose(1, 2, 0), data_range=1.0))
print('（对比：官方预训练模型是 33.46 dB —— 差距就是数据量和训练时长的差距）')
