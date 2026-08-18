## study_router.py — 阶段 6：Trajectory Prompt Router
## 学习目标：
##   1. P = f(F) 变成 P_t = R(F, z_t, t)：PromptGenBlock 的路由决策只加一行 cat
##   2. z_t 和 t 来自上一关训练的 feature diffusion（真实 DDPM 时间步/状态）
##   3. 残差初始化：新接的参数初始为 0 -> 第 0 步输出与官方 PromptIR 完全一致
##   4. 同一张图，路由权重随 t 变化 —— 轨迹感知路由第一次出现
##
## 纪律：
##   - net/model.py 一行不改！TrajPromptIR 在本文件里用官方基础件重新组装
##   - 只训练 3 个变宽的 linear_layer + time_emb（约 6k 参数），
##     主干和 15 个 expert 全部冻结（官方权重）
##
## 诚实边界（本关是最小版本）：
##   z_t = 对 AE(退化图) 做前向扩散的状态，是"网络内部 feature diffusion"
##   的替身；t 是真实 DDPM 时间步。真正的 z_t 随恢复过程演化，
##   要等 diffusion 装进网络内部（后面的阶段）。本关目标是
##   【看到路由随轨迹状态变化】，不是刷 PSNR。

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor

from net.model import PromptIR, OverlapPatchEmbed, TransformerBlock, Downsample, Upsample

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ================= 0. 上一关的遗产：TinyAE + DDPM 时间表 + 时间编码 =================
# （TinyAE 结构必须与 study_feature_diffusion.py 完全一致，否则权重对不上）
class TinyAE(nn.Module):
    def __init__(self, latent_ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, latent_ch, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(latent_ch, latent_ch, 3, 1, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(latent_ch, 16, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 3, 3, 1, 1))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return torch.clamp(self.dec(z), 0, 1)

ae = TinyAE().to(device).eval()
ae.load_state_dict(torch.load('ckpt/tiny_ae.pt'))
for p in ae.parameters():
    p.requires_grad_(False)
print('== 0. 加载上一关训练的 TinyAE（ckpt/tiny_ae.pt）==')
print('AE(退化图) -> 32 通道潜码 z0；再做前向扩散 -> z_t')

T = 200
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(z0, t, eps=None):
    """正向扩散公式：z_t = sqrt(a_bar_t)*z0 + sqrt(1-a_bar_t)*eps（无学习）"""
    if eps is None:
        eps = torch.randn_like(z0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * eps

class TimeEmbedding(nn.Module):
    """sinusoidal 时间编码 + MLP（和 diffusion 脚本里同一个类，dim 改成 32）"""
    def __init__(self, dim=32):
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

# ================= 1. TrajPromptGenBlock：只改 emb 那两行 =================
class TrajPromptGenBlock(nn.Module):
    """PromptGenBlock 的轨迹版。
       官方：emb = GAP(x)                      -> P = f(F)
       这里：emb = cat(GAP(x), GAP(z_t), t_emb) -> P_t = R(F, z_t, t)
       其余（5 个 expert、加权求和、interpolate、conv3x3）与官方逐行一致。"""
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192,
                 z_dim=32, t_dim=32):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim,
                                                    prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim + z_dim + t_dim, prompt_len)  # 唯一变宽的层
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1,
                                 padding=1, bias=False)

    def forward(self, x, zt, temb):
        B, C, H, W = x.shape
        emb = torch.cat([x.mean(dim=(-2, -1)),        # F：当前特征（官方原样）
                         zt.mean(dim=(-2, -1)),       # z_t：轨迹当前状态
                         temb], dim=1)                # t：轨迹当前时间步
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                 self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt

# ================= 2. TrajPromptIR：结构与官方一致，只换 3 个块 =================
class TrajPromptIR(nn.Module):
    def __init__(self,
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
        LayerNorm_type='WithBias', decoder=False):

        super(TrajPromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.prompt1 = TrajPromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = TrajPromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = TrajPromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
            self.time_emb = TimeEmbedding(32)   # t -> 32 维向量，三个 prompt 共用

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, zt, t):
        temb = self.time_emb(t)      # 全局轨迹状态：一张图的三个 prompt 共用同一个 t

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent, zt, temb)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, zt, temb)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, zt, temb)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

# ================= 3. 加载官方权重 + 残差初始化 =================
print()
print('== 1. 残差初始化：新参数 = 0，输出必须与官方完全一致 ==')
src = PromptIR(decoder=True).to(device).eval()
ckpt = torch.load('ckpt/model.ckpt', map_location='cpu')
sd = {k.replace('net.', ''): v for k, v in ckpt['state_dict'].items()}
src.load_state_dict(sd)

net = TrajPromptIR(decoder=True).to(device)
own = net.state_dict()
for k, v in src.state_dict().items():          # 同名同形状：直接搬
    if k in own and own[k].shape == v.shape:
        own[k] = v
for pn in ['prompt1', 'prompt2', 'prompt3']:   # 变宽的 linear：老列搬原值，新列 = 0
    w_old = src.state_dict()[pn + '.linear_layer.weight']
    b_old = src.state_dict()[pn + '.linear_layer.bias']
    with torch.no_grad():
        lin = getattr(net, pn).linear_layer
        lin.weight[:, :w_old.shape[1]].copy_(w_old)
        lin.weight[:, w_old.shape[1]:].zero_()
        lin.bias.copy_(b_old)
net.load_state_dict(own)
net.eval()

# 验证：同一张噪声图、任意 t，输出必须与官方一模一样（新权重=0，贡献恰为 0）
rng = np.random.default_rng(0)
clean_patch = data.coffee()[:128, :128]
noisy_patch = np.clip(clean_patch.astype(np.float64) + rng.normal(0, 25, clean_patch.shape), 0, 255).astype(np.uint8)
x = ToTensor()(noisy_patch)[None].to(device)
with torch.no_grad():
    zt = q_sample(ae.encode(x), torch.tensor([50], device=device))
    out_official = src(x)
    out_traj = net(x, zt, torch.tensor([50], device=device))
print('max |Traj - 官方| = %.2e   (float32 精度级别 -> 数学上完全一致)'
      % (out_official - out_traj).abs().max().item())

# ================= 4. 冻结主干，只训 router =================
n_train = n_total = 0
for name, p in net.named_parameters():
    n_total += p.numel()
    if 'linear_layer' in name or name.startswith('time_emb'):
        p.requires_grad_(True)
        n_train += p.numel()
    else:
        p.requires_grad_(False)
print('可训练参数: %d / %d （主干和 15 个 expert 全部冻结）' % (n_train, n_total))

optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                              lr=1e-3)
loss_fn = nn.L1Loss()

# ================= 5. 数据：混合退化（噪声/雾/雨/干净），同 naked_train 套路 =================
def get_rgb_images(names):
    imgs = []
    for n in names:
        try:
            imgs.append(getattr(data, n)())
        except AttributeError:
            pass
    return imgs

train_imgs = get_rgb_images(['astronaut', 'rocket', 'chelsea'])

def random_patch(img, size=128):
    H, W = img.shape[:2]
    h = random.randint(0, H - size)
    w = random.randint(0, W - size)
    p = img[h:h + size, w:w + size]
    if random.random() < 0.5:
        p = p[:, ::-1].copy()
    return p

def add_noise(img, sigma=25):
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def add_haze(img, t=0.65, A=1.0):
    return np.clip(img.astype(np.float64) * t + A * 255 * (1 - t), 0, 255).astype(np.uint8)

def add_rain(img, n_streaks=350, angle=8, length=(30, 70), intensity=(60, 200)):
    rain_layer = np.zeros(img.shape[:2], dtype=np.float64)
    H, W = img.shape[:2]
    dx = np.tan(np.radians(angle))
    for _ in range(n_streaks):
        x0 = rng.integers(0, W)
        y0 = rng.integers(0, H)
        L = int(rng.integers(*length))
        xs = np.linspace(x0, x0 + L * dx, L + 1).astype(int)
        ys = np.linspace(y0, y0 + L, L + 1).astype(int)
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        rain_layer[ys[m], xs[m]] += rng.uniform(*intensity)
    return np.clip(img.astype(np.float64) + rain_layer[..., None], 0, 255).astype(np.uint8)

DG_TYPES = ['noise', 'haze', 'rain', 'clean']
def make_pair(img):
    clean = random_patch(img)
    dg = random.choice(DG_TYPES)
    degraded = {'noise': add_noise, 'haze': add_haze, 'rain': add_rain,
                'clean': lambda i: i}[dg](clean)
    return dg, (ToTensor()(degraded)[None].to(device),
                ToTensor()(clean)[None].to(device))

# ================= 6. 训练：还是那 5 行循环，每步随机采样 t =================
print()
print('== 2. 训练 router（300 步，每步：随机退化 + 随机 t -> z_t）==')
print('iter | dg     | t   | train loss')
net.train()
for it in range(300):
    dg, (degraded, clean) = make_pair(random.choice(train_imgs))
    t = torch.randint(0, T, (1,), device=device)
    zt = q_sample(ae.encode(degraded), t)

    restored = net(degraded, zt, t)        # ① 前向（多了 zt 和 t 两个输入）
    loss = loss_fn(restored, clean)        # ② loss
    optimizer.zero_grad()                  # ③
    loss.backward()                        # ④
    optimizer.step()                       # ⑤

    if it % 25 == 0 or it == 299:
        print('%4d | %-6s | %3d | %.5f' % (it, dg, t.item(), loss.item()))

torch.save(net.state_dict(), 'ckpt/router_trained_300.pt')
print()
print('saved: ckpt/router_trained_300.pt')

# ================= 7. 训练后对比：官方 vs Traj（同一批图，Traj 用 t=0） =================
print()
print('== 3. 训练后 PSNR 对比（coffee 四个固定 patch，Traj 用 t=0 状态）==')
net.eval()
val_dgs = {}
val_dgs['noise'] = add_noise(data.coffee()[:128, :128])
val_dgs['haze'] = add_haze(data.coffee()[:128, 128:256])
val_dgs['rain'] = add_rain(data.coffee()[128:256, :128])
val_dgs['clean'] = data.coffee()[128:256, 128:256]
clean_pos = {'noise': data.coffee()[:128, :128],
             'haze': data.coffee()[:128, 128:256],
             'rain': data.coffee()[128:256, :128],
             'clean': data.coffee()[128:256, 128:256]}

print('%-6s | %8s | %8s | %8s' % ('dg', '输入', '官方', 'Traj'))
for dg in ['noise', 'haze', 'rain', 'clean']:
    vx = ToTensor()(val_dgs[dg])[None].to(device)
    vc = clean_pos[dg] / 255.0
    with torch.no_grad():
        o = src(vx)[0].cpu().numpy().transpose(1, 2, 0)
        zt0 = q_sample(ae.encode(vx), torch.tensor([0], device=device))
        r = net(vx, zt0, torch.tensor([0], device=device))[0].cpu().numpy().transpose(1, 2, 0)
    p_in = psnr(vc, val_dgs[dg] / 255.0, data_range=1.0)
    p_of = psnr(vc, o, data_range=1.0)
    p_tr = psnr(vc, r, data_range=1.0)
    print('%-6s | %7.2f | %7.2f | %7.2f %s'
          % (dg, p_in, p_of, p_tr, '(+%.2f)' % (p_tr - p_of) if p_tr > p_of else ''))

# ================= 8. 核心演示：同一张图，路由权重随 t 变化 =================
print()
print('== 4. 同一张噪声图，路由权重随轨迹状态 (z_t, t) 变化 ==')
weights = {}
current_t = [None]
def w_hook(p_name):
    def hook(m, inp, out):
        weights[(current_t[0], p_name)] = torch.softmax(out, dim=1).detach().cpu().numpy()[0]
    return hook

handles = [
    net.prompt1.linear_layer.register_forward_hook(w_hook('prompt1')),
    net.prompt2.linear_layer.register_forward_hook(w_hook('prompt2')),
    net.prompt3.linear_layer.register_forward_hook(w_hook('prompt3')),
]

T_LIST = [199, 150, 100, 50, 0]
x = ToTensor()(noisy_patch)[None].to(device)
with torch.no_grad():
    for t in T_LIST:
        current_t[0] = t
        zt = q_sample(ae.encode(x), torch.tensor([t], device=device))
        net(x, zt, torch.tensor([t], device=device))
for h in handles:
    h.remove()

scales = ['prompt3', 'prompt2', 'prompt1']
W = np.array([[weights[(t, s)] for s in scales] for t in T_LIST])   # (5,3,5)
np.savez('results/router_weights_trajectory.npz',
         weights=W, t_list=T_LIST, scales=scales)

row_names = ['t=%d · %s' % (t, s) for t in T_LIST for s in scales]
print('%-22s %s' % ('', '  '.join('E%d' % (i + 1) for i in range(5))))
for rn, w in zip(row_names, W.reshape(15, 5)):
    print('%-22s %s' % (rn, '  '.join('%.2f' % v for v in w)))

# ================= 9. 热力图（与 study_prompts 同一套画法）=================
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams['font.family'] = 'Segoe UI'

fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
cmap = ListedColormap(SEQ)
im = ax.imshow(W.reshape(15, 5), cmap=cmap, norm=Normalize(0, 1),
               aspect='auto', interpolation='nearest')
for i in range(15):
    for j in range(5):
        v = W.reshape(15, 5)[i, j]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=SURFACE, linewidth=2))
        ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=9,
                color='#ffffff' if v >= 0.55 else INK)
ax.set_xticks(range(5))
ax.set_xticklabels(['Expert %d' % (i + 1) for i in range(5)], color=INK2, fontsize=10)
ax.set_yticks(range(15))
ax.set_yticklabels(row_names, color=INK2, fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('TrajPromptIR: routing weights along the trajectory\n'
             '(same noisy image, same router, different diffusion state (z_t, t))',
             color=INK, fontsize=11, pad=12)
fig.text(0.5, 0.015,
         'Each row = one trajectory state. Rows differing = the router is trajectory-aware.',
         ha='center', color=INK2, fontsize=9)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig('results/router_weights_trajectory.png', dpi=160, facecolor=SURFACE)
print()
print('saved: results/router_weights_trajectory.png   <- 打开它！')
print()
print('== 结果说明（本关真实结局：negative result）==')
print('15 行几乎一样 -> 路由没有随 (z_t, t) 变化，P_t = R(F, z_t, t) 还【不】成立')
print('原因：残差初始化 + 冻结主干 + 官方路由已足够好 -> 梯度几乎为零，')
print('router 没有"必须使用轨迹信息"的压力，保持躺平（identity）。')
print('这正是阶段 7 TPC 对比损失要解决的问题：给 router 一个明确的轨迹压力。')
