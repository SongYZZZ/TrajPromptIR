## study_forward.py — 追踪一张图如何经过 PromptIR.forward()
## 学习目标：
##   1. 一张图怎么进入 forward()（tensor 长什么样）
##   2. 数据流经过哪些模块、形状怎么变（U 形编码器-解码器）
##   3. PromptGenBlock 里 5 个 expert 的权重是怎么算出来的
##   4. loss 怎么算、backward() 之后发生了什么
## 不改动 net/model.py 的任何代码，全部用 register_forward_hook 挂载观察。

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from PIL import Image
from torchvision.transforms import ToTensor

from net.model import PromptIR

# 与 demo.py 相同的加载方式（ckpt 是 Lightning 格式，先拿到 .net）
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda')
model = PromptIRModel.load_from_checkpoint('ckpt/model.ckpt').to(device).eval()
net = model.net

# ---------- 第 1 步：一张图怎么变成网络的输入 ----------
degraded = ToTensor()(np.array(Image.open('test/demo/astro_noisy25.png').convert('RGB')))
clean    = ToTensor()(np.array(Image.open('test/demo/astro_clean.png').convert('RGB')))
degraded, clean = degraded[None].to(device), clean[None].to(device)  # 加 batch 维
print('== input tensor ==')
print('degraded shape:', tuple(degraded.shape), '| dtype:', degraded.dtype,
      '| value range: [%.2f, %.2f]' % (float(degraded.min()), float(degraded.max())))
print('（shape = [batch=1, RGB=3, H=512, W=512]，值域 0~1 是官方模型的要求）')

# ---------- 第 2 步：给主干每个子模块挂 hook，打印数据流 ----------
def make_hook(name):
    def hook(m, inp, out):
        print('%-22s %s -> %s' % (name, tuple(inp[0].shape), tuple(out.shape)))
    return hook

handles = [child.register_forward_hook(make_hook(name)) for name, child in net.named_children()]

# ---------- 第 3 步：抓 PromptGenBlock 内部算出的 5 个 expert 权重 ----------
# PromptGenBlock.forward 里：logits = linear_layer(GAP(x))，softmax 后就是权重。
# hook 挂在 linear_layer 上，拿到 logits 自己 softmax。
weights = {}
def w_hook(key):
    def hook(m, inp, out):
        weights[key] = torch.softmax(out, dim=1).detach().cpu().numpy()
    return hook
handles.append(net.prompt1.linear_layer.register_forward_hook(w_hook('prompt1')))
handles.append(net.prompt2.linear_layer.register_forward_hook(w_hook('prompt2')))
handles.append(net.prompt3.linear_layer.register_forward_hook(w_hook('prompt3')))

print()
print('== forward() 数据流（按 forward 代码的执行顺序打印）==')
with torch.no_grad():
    restored = net(degraded)

for h in handles:
    h.remove()

print()
print('== 三个 decoder stage 的 prompt expert 权重（这张 σ=25 噪声图）==')
for k in ['prompt3', 'prompt2', 'prompt1']:
    print('%-8s 5 experts = %s' % (k, np.round(weights[k][0], 3)))

# ---------- 第 4 步：loss 怎么算 ----------
# 注意：这一次必须开着梯度记录。反向传播要求把每一层中间结果都存下来，
# 512x512 整图会把 8GB 显存撑爆（刚才就 OOM 了）——
# 这正是官方训练用 128x128 patch 而不是整张图的原因！
del restored
torch.cuda.empty_cache()
patch_degraded = degraded[:, :, 64:192, 64:192].contiguous()  # 从图中切 128x128
patch_clean    = clean[:, :, 64:192, 64:192].contiguous()
print()
print('== 为什么训练用 128x128 patch ==')
print('backward 需要保存每一层的中间结果，512x512 会 OOM；切成 128x128 就够（和官方 options.py 一致）')
print()

restored_grad = net(patch_degraded)
loss_restored = nn.L1Loss()(restored_grad, patch_clean)
loss_input    = nn.L1Loss()(patch_degraded, patch_clean)
print('== loss（L1 = |预测 - 干净图| 的平均值，越小越好）==')
print('L1(退化图, 干净图)   = %.4f' % loss_input.item())
print('L1(恢复图, 干净图)   = %.4f' % loss_restored.item())

# ---------- 第 5 步：backward() 之后发生了什么 ----------
loss_restored.backward()
g = net.patch_embed.proj.weight.grad  # 第一层 3x3 卷积的权重梯度
print()
print('== backward() 之后 ==')
print('patch_embed 卷积权重的梯度模 = %.6f' % g.norm().item())
print('梯度张量形状 =', tuple(g.shape), '（与权重形状一致：每个参数都得到了自己的梯度）')
print('（注意：backward 只是"算出了该往哪改"，参数还没变；optimizer.step() 才会真正修改）')
