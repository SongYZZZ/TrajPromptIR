## study_prompts.py — Checkpoint 3：拆开 PromptGenBlock
## 回答一个问题：不同退化（噪声/雾/雨）真的选择了不同的 prompt expert 吗？
##
## 做法：
##   1. 同一张干净图，合成三种退化版本（高斯噪声 σ=25 / 大气散射雾 / 雨条纹）
##   2. 用官方预训练 PromptIR 分别推理，hook 抓 prompt1/2/3 的 5 个 expert 权重
##   3. 画 9 行(3退化x3尺度) x 5 列(expert) 的 heatmap，并保存权重数据
##
## 意义：这组权重就是 P = f(F) 的实拍。你未来的 TrajPromptIR 要在这之上
##       再加一个"轨迹"维度：同一张图，权重还要随 z_t 变化。

import numpy as np
import torch
import lightning.pytorch as pl
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from net.model import PromptIR

device = torch.device('cuda')
rng = np.random.default_rng(0)

# ================= 1. 加载官方预训练模型 =================
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = torch.nn.L1Loss()
    def forward(self, x):
        return self.net(x)

model = PromptIRModel.load_from_checkpoint('ckpt/model.ckpt').to(device).eval()
net = model.net

# ================= 2. 合成三种退化 =================
clean = data.astronaut()

def add_noise(img, sigma=25):
    noisy = np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape), 0, 255)
    return noisy.astype(np.uint8)

def add_haze(img, t=0.65, A=1.0):
    # 大气散射模型：I(x) = J(x)*t + A*(1-t)，t 越小雾越浓
    hazy = np.clip(img.astype(np.float64) * t + A * 255 * (1 - t), 0, 255)
    return hazy.astype(np.uint8)

def add_rain(img, n_streaks=350, angle=8, length=(30, 70), intensity=(60, 200)):
    # 雨条纹：轻微倾斜的亮线段
    rain_layer = np.zeros(img.shape[:2], dtype=np.float64)
    H, W = img.shape[:2]
    dx = np.tan(np.radians(angle))  # 每向下 1 像素偏移多少
    for _ in range(n_streaks):
        x0 = rng.integers(0, W)
        y0 = rng.integers(0, H)
        L = int(rng.integers(*length))
        xs = np.linspace(x0, x0 + L * dx, L + 1).astype(int)
        ys = np.linspace(y0, y0 + L, L + 1).astype(int)
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        rain_layer[ys[m], xs[m]] += rng.uniform(*intensity)
    rainy = np.clip(img.astype(np.float64) + rain_layer[..., None], 0, 255)
    return rainy.astype(np.uint8)

degradations = {
    'noise-s25': add_noise(clean),
    'haze':      add_haze(clean),
    'rain':      add_rain(clean),
}

# ================= 3. hook 抓权重 =================
weights = {}   # key: (degradation, prompt_name) -> np.array(5,)
current_dg = [None]  # 每次推理前更新，hook 里读取，避免三次推理互相覆盖
def w_hook(p_name):
    def hook(m, inp, out):
        weights[(current_dg[0], p_name)] = torch.softmax(out, dim=1).detach().cpu().numpy()[0]
    return hook

handles = [
    net.prompt1.linear_layer.register_forward_hook(w_hook('prompt1')),
    net.prompt2.linear_layer.register_forward_hook(w_hook('prompt2')),
    net.prompt3.linear_layer.register_forward_hook(w_hook('prompt3')),
]

restored, psnr_table = {}, {}
with torch.no_grad():
    for dg_name, dg_img in degradations.items():
        current_dg[0] = dg_name
        x = ToTensor()(dg_img)[None].to(device)
        out = net(x)
        restored[dg_name] = out[0].cpu().numpy().transpose(1, 2, 0)
        psnr_table[dg_name] = psnr(
            clean / 255.0, restored[dg_name], data_range=1.0)
for h in handles:
    h.remove()

# 权重按 (退化, 尺度) 排成 9x5 矩阵
scales = ['prompt3', 'prompt2', 'prompt1']
W = np.array([[weights[(dg, s)] for s in scales] for dg in degradations])  # (3,3,5)

# ================= 4. 保存结果 =================
import os
os.makedirs('results', exist_ok=True)
np.savez('results/prompt_weights.npz',
         noise=W[0], haze=W[1], rain=W[2],
         degradations=list(degradations.keys()), scales=scales)

for dg_name in degradations:
    Image.fromarray((restored[dg_name] * 255).astype(np.uint8)).save(
        f'output/demo/astro_{dg_name}_restored.png')

print('== PSNR（官方 all-in-one 模型，同一套权重恢复三种退化）==')
for dg_name in degradations:
    dg_in = degradations[dg_name] / 255.0
    print('%-10s 输入 %.2f dB -> 恢复 %.2f dB'
          % (dg_name, psnr(clean / 255.0, dg_in, data_range=1.0), psnr_table[dg_name]))

print()
print('== prompt expert 权重（行 = 退化x尺度，列 = 5 个 expert）==')
row_names = [f'{dg} · {s}' for dg in degradations for s in scales]
print('%-20s %s' % ('', '  '.join('E%d' % (i + 1) for i in range(5))))
for rn, w in zip(row_names, W.reshape(9, 5)):
    print('%-20s %s' % (rn, '  '.join('%.2f' % v for v in w)))

# ================= 5. 画 heatmap（单色系蓝阶，数值直接标注）=================
# 调色板（dataviz 规范 sequential blue ramp, light surface）
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams['font.family'] = 'Segoe UI'

fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

cmap = ListedColormap(SEQ)
im = ax.imshow(W.reshape(9, 5), cmap=cmap, norm=Normalize(0, 1),
               aspect='auto', interpolation='nearest')

# 2px 表面色间隔（dataviz 规范：相邻填充之间留 2px 表面间隙）
for i in range(9):
    for j in range(5):
        v = W.reshape(9, 5)[i, j]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=SURFACE, linewidth=2))
        ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=9,
                color='#ffffff' if v >= 0.55 else INK)

ax.set_xticks(range(5))
ax.set_xticklabels(['Expert %d' % (i + 1) for i in range(5)],
                   color=INK2, fontsize=10)
ax.set_yticks(range(9))
ax.set_yticklabels(row_names, color=INK2, fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title('PromptIR pretrained: prompt expert weights\n'
             '(3 degradations x 3 decoder scales; prompt3 = coarsest, prompt1 = finest)',
             color=INK, fontsize=11, pad=12)
fig.text(0.5, 0.015,
         'Rows show how one fixed model routes different degradations at different scales.',
         ha='center', color=INK2, fontsize=9)

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig('results/prompt_weights_heatmap.png', dpi=160, facecolor=SURFACE)
print()
print('saved: results/prompt_weights_heatmap.png')
print('saved: results/prompt_weights.npz')
