## study_feature_diffusion.py — 最小 Feature Diffusion
## 学习目标：
##   1. 第一次亲手训练一个 diffusion 模块
##   2. 亲眼看到"过程中"：z_T(噪声) -> ... -> z_0(干净特征)，一步步变干净
##   3. 让 z_t 和 t 从抽象符号变成你见过的东西 —— 它们就是你未来
##      Trajectory Prompt Router 的两个输入
##
## 结构（每段都是你已会的知识 + 一点新东西）：
##   1. TinyAE：把 64x64 图压成 32x16x16 特征再还原（熟悉的训练循环）
##   2. 正向扩散 q_sample：z_t = sqrt(a_bar_t)*z0 + sqrt(1-a_bar_t)*eps
##      —— 加噪是【公式】，不需要学习
##   3. DiffusionUNet + TimeEmbedding：预测噪声 eps
##      —— 反向去噪必须【网络来学】（TimeEmbedding 将来 router 复用）
##   4. 训练 DDPM：还是那个 5 行循环！loss = MSE(预测噪声, 真实噪声)
##   5. 采样：从纯噪声 z_T 出发，一步步去噪到 z_0，沿途解码成图
##   6. 保存轨迹 strip：你第一次亲眼看到"恢复过程"
##
## 预期：模型很小（几十万参数）、只练一两分钟，最后采样出的图会模糊——
##       这是正常的。本关的目标是【看到轨迹】，不是画质。

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ================= 0. 数据：还是那几张图，64x64 patch =================
def get_rgb_images(names):
    imgs = []
    for n in names:
        try:
            imgs.append(getattr(data, n)())
        except AttributeError:
            pass
    return imgs

imgs = get_rgb_images(['astronaut', 'rocket', 'chelsea', 'coffee'])

def random_patch(img, size=64):
    H, W = img.shape[:2]
    h = random.randint(0, H - size)
    w = random.randint(0, W - size)
    p = img[h:h + size, w:w + size]
    if random.random() < 0.5:
        p = p[:, ::-1].copy()
    return p

def batch_patches(n):
    xs = [ToTensor()(random_patch(random.choice(imgs))) for _ in range(n)]
    return torch.stack(xs).to(device)   # (n, 3, 64, 64)

# ================= 1. TinyAE：图 <-> 特征 =================
class TinyAE(nn.Module):
    def __init__(self, latent_ch=32):
        super().__init__()
        # 编码器：64x64 -> 32x32 -> 16x16，通道 3 -> 16 -> 32
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, latent_ch, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(latent_ch, latent_ch, 3, 1, 1))
        # 解码器：16x16 -> 32x32 -> 64x64
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(latent_ch, 16, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 3, 3, 1, 1))

    def encode(self, x):
        return self.enc(x)                      # (n, 32, 16, 16) = 特征 z

    def decode(self, z):
        return torch.clamp(self.dec(z), 0, 1)

ae = TinyAE().to(device)
opt_ae = torch.optim.AdamW(ae.parameters(), lr=1e-3)
print('== 1. 训练自动编码器（图 -> 32x16x16 特征 -> 图）==')
for it in range(400):
    x = batch_patches(64)
    xr = ae.decode(ae.encode(x))
    loss = F.l1_loss(xr, x)
    opt_ae.zero_grad()
    loss.backward()
    opt_ae.step()
    if it % 50 == 0 or it == 399:
        print('AE iter %3d | recon L1 %.4f' % (it, loss.item()))

val_x = batch_patches(1)
with torch.no_grad():
    val_rec = ae.decode(ae.encode(val_x))
print('AE 验证 PSNR: %.2f dB（特征空间够用即可）'
      % psnr(val_x[0].cpu().numpy().transpose(1, 2, 0),
             val_rec[0].cpu().numpy().transpose(1, 2, 0), data_range=1.0))
for p in ae.parameters():
    p.requires_grad_(False)   # 冻结 AE，只训练扩散网络
torch.save(ae.state_dict(), 'ckpt/tiny_ae.pt')
print('saved: ckpt/tiny_ae.pt（阶段 6 的 Router 要用它算 z_t）')

# ================= 2. 扩散的"时间表"与正向加噪公式 =================
T = 200
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)   # a_bar_t = a_1*a_2*...*a_t

def q_sample(z0, t, eps=None):
    """正向扩散：z_t = sqrt(a_bar_t)*z0 + sqrt(1-a_bar_t)*eps
       这是【公式】，没有可学习的参数——加噪不需要学，去噪才需要。"""
    if eps is None:
        eps = torch.randn_like(z0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * eps, eps

# ================= 3. 扩散网络：预测噪声 =================
class TimeEmbedding(nn.Module):
    """sinusoidal 时间编码 + MLP。
       注意：这个类将来你的 Trajectory Router 也要用（把 t 变成向量）。"""
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Linear(dim * 2, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * np.log(10000) / half)
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, c, t_dim):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.tp = nn.Linear(t_dim, c)   # 把时间向量投进特征里

    def forward(self, x, temb):
        h = F.gelu(self.c1(x) + self.tp(temb)[..., None, None])
        return x + self.c2(h)

class DiffusionUNet(nn.Module):
    def __init__(self, c=32, hid=64):
        super().__init__()
        self.temb = TimeEmbedding(64)
        self.inp = nn.Conv2d(c, hid, 3, 1, 1)
        self.blocks = nn.ModuleList([ResBlock(hid, 64) for _ in range(4)])
        self.out = nn.Conv2d(hid, c, 3, 1, 1)

    def forward(self, zt, t):
        te = self.temb(t)
        h = self.inp(zt)
        for b in self.blocks:
            h = b(h, te)
        return self.out(h)

unet = DiffusionUNet().to(device)
opt_d = torch.optim.AdamW(unet.parameters(), lr=2e-4)
# 训练完成后保存权重 —— 阶段 6 的 Router 需要这个训练好的 AE 来算 z_t
print('（训练完成后会把 AE / UNet 权重存到 ckpt/tiny_ae.pt 和 ckpt/feature_ddpm.pt）')
print()
print('== 2. 训练扩散网络（预测"这一步加了多少噪声"）==')
print('（还是那 5 行循环！只是 loss 从 |图-图| 变成 |预测噪声-真实噪声|）')
for it in range(1500):
    x = batch_patches(64)
    z0 = ae.encode(x)                                 # 干净特征
    t = torch.randint(0, T, (z0.shape[0],), device=device)
    zt, eps = q_sample(z0, t)                         # 加噪后的特征 + 真实噪声
    pred = unet(zt, t)                                # 网络预测噪声
    loss = F.mse_loss(pred, eps)                      # 预测对了吗？
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    if it % 150 == 0 or it == 1499:
        print('Diff iter %4d | noise-pred MSE %.4f  (起点约 1.0，越低越好)'
              % (it, loss.item()))

torch.save(unet.state_dict(), 'ckpt/feature_ddpm.pt')
print('saved: ckpt/feature_ddpm.pt（采样网络，后面完整集成时用）')

# ================= 4. 采样：从纯噪声一步步去噪到干净特征 =================
print()
print('== 3. 采样：z_T(纯噪声) -> ... -> z_0(干净特征) ==')
@torch.no_grad()
def sample(n=1, save_every=40):
    zt = torch.randn(n, 32, 16, 16, device=device)    # 起点：纯噪声
    frames = [(T - 1, ae.decode(zt)[0])]              # 第一步的画面
    for t in reversed(range(T)):                      # t: 199 -> 0
        tt = torch.full((n,), t, device=device)
        eps = unet(zt, tt)                            # 这一步该去掉的噪声
        a, ab = alphas[t], alpha_bar[t]
        zt = 1 / torch.sqrt(a) * (zt - (1 - a) / torch.sqrt(1 - ab) * eps)
        if t > 0:
            zt = zt + torch.sqrt(betas[t]) * torch.randn_like(zt)
        if t % save_every == 0 or t == 0:
            frames.append((t, ae.decode(zt)[0]))
    return frames

frames = sample()
print('z_t 的形状: (1, 32, 16, 16)  —— 这就是将来喂给 Trajectory Router 的 z_t')
print('t 的取值: 0 ~ %d 的整数  —— 这就是将来喂给 Router 的 t' % (T - 1))

# ================= 5. 保存轨迹 strip：你第一次亲眼看到"过程中" =================
import os
os.makedirs('results', exist_ok=True)
strip = np.concatenate([f[1].cpu().numpy().transpose(1, 2, 0) * 255 for f in frames],
                       axis=1).astype(np.uint8)
img = Image.fromarray(strip)
draw = ImageDraw.Draw(img)
for i, (t, _) in enumerate(frames):
    label = 'start(noise)' if t == T - 1 else ('t=%d' % t)
    draw.text((i * 64 + 4, 4), label, fill=(255, 0, 0))
img.save('results/feature_diffusion_trajectory.png')
print()
print('saved: results/feature_diffusion_trajectory.png  <- 打开它！')
print('从左到右 = 恢复过程：纯噪声 -> ... -> 逐渐变清晰的图')
print('（图会模糊：模型很小只练了 1.5k 步。本关目标是看到轨迹，不是画质）')
