# 【学习注释】本文件 = TrajPromptIR 的全部新增部件（官方 model.py 之外的新代码）。
# 数据流：lq/clean 特征 → 路由器(P_t) + 去噪器(ε̂) + 扩散调度 → 预测干净特征 → 门控融合。
"""Core modules for the first trainable TrajPromptIR integration.

This file intentionally contains only the new research variables:

1. a trajectory-aware prompt router;
2. a conditional feature-space diffusion prior (including the TPC mismatch branch);
3. a conservative gated fusion layer.

The official PromptIR implementation in ``net/model.py`` stays functionally
untouched (study comments only) so the baseline remains reproducible.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create the standard sinusoidal embedding for integer timesteps."""
    # 把整数时间步 t 变成向量（正弦位置编码，扩散模型的通用做法）
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch]")
    if dim < 2:
        raise ValueError("embedding dim must be at least 2")

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Gather a diffusion coefficient and reshape it for BCHW broadcasting."""
    gathered = values.gather(0, timesteps)
    return gathered.to(dtype=reference.dtype).view(-1, 1, 1, 1)


class DiffusionSchedule(nn.Module):
    """Linear DDPM schedule plus deterministic DDIM updates."""
    # 扩散的"数学包"：噪声调度表 + 前向加噪公式 + 去噪(DDIM)公式

    def __init__(
        self,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(
        self,
        clean_feature: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample q(x_t | x_0) and return both x_t and the injected noise."""
        if noise is None:
            noise = torch.randn_like(clean_feature)
        if noise.shape != clean_feature.shape:
            raise ValueError("noise and clean_feature must have identical shapes")

        alpha_bar = _extract(self.alpha_bars, timesteps, clean_feature)
        # 前向加噪公式：z_t = √ᾱ_t · z_0 + √(1−ᾱ_t) · ε
        noisy = alpha_bar.sqrt() * clean_feature + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def predict_clean(
        self,
        noisy_feature: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the x_0 estimate implied by an epsilon prediction."""
        # 从噪声预测 ε̂ 反推干净特征：ẑ_0 = (z_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t
        alpha_bar = _extract(self.alpha_bars, timesteps, noisy_feature)
        return (
            noisy_feature - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt().clamp_min(1e-6)

    def ddim_step(
        self,
        noisy_feature: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one deterministic DDIM step (eta = 0)."""
        # 一步确定性去噪：先估干净特征，再按目标时间步重混噪声（推理用它代替整条马尔可夫链）
        clean = self.predict_clean(noisy_feature, timesteps, predicted_noise)
        if torch.all(previous_timesteps < 0):
            return clean

        safe_previous = previous_timesteps.clamp_min(0)
        previous_alpha_bar = _extract(self.alpha_bars, safe_previous, noisy_feature)
        terminal_mask = (previous_timesteps < 0).view(-1, 1, 1, 1)
        previous_alpha_bar = torch.where(
            terminal_mask,
            torch.ones_like(previous_alpha_bar),
            previous_alpha_bar,
        )
        return (
            previous_alpha_bar.sqrt() * clean
            + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
        )

    def inference_timesteps(self, sample_steps: int, device: torch.device) -> torch.Tensor:
        """Return a descending, approximately uniform DDIM timestep grid."""
        # 推理时把 200 步压成 sample_steps 个等距时间步（默认 4 步）
        if not 1 <= sample_steps <= self.num_steps:
            raise ValueError("sample_steps must be in [1, num_steps]")
        return torch.linspace(
            self.num_steps - 1,
            0,
            sample_steps,
            device=device,
        ).round().long()


class TrajectoryPromptRouter(nn.Module):
    """Route an LQ feature and a diffusion state through a learned prompt bank.

    ``trajectory_aware=False`` keeps the architecture and parameter count fixed,
    but zeros the state and timestep inputs. This is the strict static-prompt
    control used by the later Dynamic-vs-Static ablation.
    """

    def __init__(
        self,
        feature_channels: int,
        prompt_dim: int = 64,
        num_experts: int = 8,
        time_dim: int = 64,
        hidden_dim: int = 192,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2")
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.feature_channels = feature_channels
        self.prompt_dim = prompt_dim
        self.num_experts = num_experts
        self.time_dim = time_dim
        self.top_k = top_k
        self.temperature = temperature

        # 路由输入 = GAP(F) 384 维 + GAP(z_t) 384 维 + 时间编码 64 维 = 832 维
        router_input_dim = feature_channels * 2 + time_dim
        self.input_norm = nn.LayerNorm(router_input_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        # 8 个可学习 prompt"专家"（每个 64 维向量），加权混合出最终 prompt
        self.prompt_bank = nn.Parameter(torch.randn(num_experts, prompt_dim) * 0.02)

    def forward(
        self,
        degradation_feature: torch.Tensor,
        diffusion_state: torch.Tensor,
        timesteps: torch.Tensor,
        trajectory_aware: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if degradation_feature.ndim != 4 or diffusion_state.ndim != 4:
            raise ValueError("router features must have shape [B, C, H, W]")
        if degradation_feature.shape[:2] != diffusion_state.shape[:2]:
            raise ValueError("degradation and diffusion features must share B and C")

        degradation_state = degradation_feature.mean(dim=(-2, -1))  # GAP：退化特征 → 向量
        trajectory_state = diffusion_state.mean(dim=(-2, -1))       # GAP：当前扩散状态 z_t → 向量
        time_state = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.time_dim)
        )                                                           # 时间步 t → 向量

        if not trajectory_aware:
            # 静态对照：把轨迹和时间输入清零——架构、参数完全不变，只是"看不见"轨迹
            trajectory_state = torch.zeros_like(trajectory_state)
            time_state = torch.zeros_like(time_state)

        router_input = torch.cat(
            [degradation_state, trajectory_state, time_state],
            dim=1,
        )                                                           # 三路信息拼成 832 维
        logits = self.router(self.input_norm(router_input)) / self.temperature

        if self.top_k is not None and self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            logits = sparse_logits.scatter(1, top_indices, top_values)

        weights = torch.softmax(logits, dim=1)      # 8 个专家的路由权重
        prompt = weights @ self.prompt_bank         # 加权混合 = P_t = f(F, z_t, t)
        return prompt, weights


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FeatureResidualBlock(nn.Module):
    """Small FiLM-conditioned residual block for bottleneck features."""
    # FiLM 残差块：condition 提供"缩放+平移"，让 prompt/时间信息调制特征

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(feature)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return feature + hidden


class ConditionalFeatureDenoiser(nn.Module):
    """Predict epsilon in PromptIR's low-resolution bottleneck space."""

    def __init__(
        self,
        feature_channels: int,
        prompt_dim: int = 64,
        time_dim: int = 64,
        hidden_channels: int = 96,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self.time_dim = time_dim
        self.noisy_projection = nn.Conv2d(feature_channels, hidden_channels, 1)
        self.lq_projection = nn.Conv2d(feature_channels, hidden_channels, 1)
        # condition = MLP(时间编码 + prompt)：prompt 就是从这里影响去噪结果的
        self.condition_mlp = nn.Sequential(
            nn.Linear(time_dim + prompt_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        # TPC-v2：Prompt Bank 初始量级约 0.02，而时间编码约为 1。旧版直接拼接后，
        # Prompt 很容易被时间条件淹没。这里先归一化 Prompt，再用独立 MLP 生成一条
        # 专用 FiLM 条件残差；这样既保留原条件通路，又确保 Prompt 对去噪器有杠杆。
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.prompt_condition_mlp = nn.Sequential(
            nn.Linear(prompt_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        self.blocks = nn.ModuleList(
            [FeatureResidualBlock(hidden_channels, hidden_channels * 2) for _ in range(num_blocks)]
        )
        self.output_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.output = nn.Conv2d(hidden_channels, feature_channels, 1)

        # 【重要修复】输出卷积用小的随机初始化，而不是零初始化：
        # 零初始化会把上游所有参数的梯度乘成 0（路由器一步都学不到）。
        # 注意：这不影响"训练前输出≡官方"——那个不变量由融合门的零初始化保证。
        # A tiny random output keeps the gradient path through the prompt alive
        # from the very first step: a zero-init output multiplies the whole
        # upstream Jacobian by zero, starving the router. The baseline identity
        # at step 0 is guaranteed by the zero-init fusion gate instead, so this
        # change does not affect the official-output invariant.
        nn.init.normal_(self.output.weight, std=0.02)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_feature: torch.Tensor,
        lq_feature: torch.Tensor,
        prompt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim)
        shared_condition = self.condition_mlp(
            torch.cat([time_embedding, prompt], dim=1)
        )
        prompt_condition = self.prompt_condition_mlp(self.prompt_norm(prompt))
        condition = shared_condition + prompt_condition  # TPC-v2：专用 Prompt 条件残差
        hidden = self.noisy_projection(noisy_feature) + self.lq_projection(lq_feature)  # 噪声状态 + 退化条件
        for block in self.blocks:
            hidden = block(hidden, condition)                                       # 3 个 FiLM 残差块
        return self.output(F.silu(self.output_norm(hidden)))   # 输出 ε̂（预测的噪声）


class TrajectoryFeatureDiffusionPrior(nn.Module):
    """Prompt-conditioned feature DDPM used by both training and inference."""

    def __init__(
        self,
        feature_channels: int,
        num_diffusion_steps: int = 200,
        prompt_dim: int = 64,
        num_experts: int = 8,
        hidden_channels: int = 96,
        num_blocks: int = 3,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.schedule = DiffusionSchedule(num_steps=num_diffusion_steps)
        self.router = TrajectoryPromptRouter(
            feature_channels=feature_channels,
            prompt_dim=prompt_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.denoiser = ConditionalFeatureDenoiser(
            feature_channels=feature_channels,
            prompt_dim=prompt_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
        )

    def forward_train(
        self,
        clean_feature: torch.Tensor,
        lq_feature: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        trajectory_aware: bool = True,
        tpc_enabled: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch = clean_feature.shape[0]
        if clean_feature.shape != lq_feature.shape:
            raise ValueError("clean_feature and lq_feature must have identical shapes")
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.schedule.num_steps,
                (batch,),
                device=clean_feature.device,
            )

        # 训练一步：① 抽时间步 t 给干净特征加噪 → ② 路由器算 P_t → ③ 去噪器预测 ε̂
        noisy_feature, target_noise = self.schedule.q_sample(
            clean_feature,
            timesteps,
            noise,
        )
        prompt, routing_weights = self.router(
            lq_feature,
            noisy_feature,
            timesteps,
            trajectory_aware=trajectory_aware,
        )
        predicted_noise = self.denoiser(
            noisy_feature,
            lq_feature,
            prompt,
            timesteps,
        )
        predicted_clean = self.schedule.predict_clean(
            noisy_feature,
            timesteps,
            predicted_noise,
        )
        tpc_output = None
        if tpc_enabled:
            tpc_output = self._tpc_contrast(
                clean_feature=clean_feature,
                lq_feature=lq_feature,
                noisy_feature=noisy_feature,
                timesteps=timesteps,
                target_noise=target_noise,
                trajectory_aware=trajectory_aware,
            )
        output = {
            "predicted_clean": predicted_clean,
            "predicted_noise": predicted_noise,
            "target_noise": target_noise,
            "noisy_feature": noisy_feature,
            "timesteps": timesteps,
            "prompt": prompt,
            "routing_weights": routing_weights,
        }
        if tpc_output is not None:
            output.update(tpc_output)
        return output

    def _sample_mismatch_timesteps(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return a deterministic, distant timestep for TPC debugging.

        The half-trajectory offset gives every matched timestep one stable
        negative direction.  It is intentionally deterministic so short A/B
        runs can reveal whether trajectory conditioning is learnable before we
        introduce harder distance-aware negative sampling.
        """
        if self.schedule.num_steps < 2:
            raise ValueError("TPC requires at least two diffusion timesteps")
        offset = max(1, self.schedule.num_steps // 2)
        return (timesteps + offset) % self.schedule.num_steps

    def _router_only_grad_denoise(
        self,
        noisy_feature: torch.Tensor,
        lq_feature: torch.Tensor,
        prompt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Denoiser forward whose denoiser gradients cancel out.

        The value equals ``self.denoiser(..., prompt)`` but gradients flow only
        through ``prompt`` towards the router. This keeps the TPC mismatch
        branch from teaching the denoiser to fail on purpose for out-of-stage
        prompts, which would let the router "cheat" without learning
        stage-appropriate prompts.
        """
        # 防作弊写法：数值 = 正常前向，但梯度只经 prompt 回路由器、去噪器收到 0
        full = self.denoiser(noisy_feature, lq_feature, prompt, timesteps)
        reference = self.denoiser(
            noisy_feature,
            lq_feature,
            prompt.detach(),
            timesteps,
        )
        return full + reference.detach() - reference

    @staticmethod
    def _per_sample_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return one MSE value per batch entry for a sample-wise TPC hinge."""
        return (prediction - target).square().flatten(1).mean(dim=1)

    def _tpc_contrast(
        self,
        clean_feature: torch.Tensor,
        lq_feature: torch.Tensor,
        noisy_feature: torch.Tensor,
        timesteps: torch.Tensor,
        target_noise: torch.Tensor,
        trajectory_aware: bool,
    ) -> Dict[str, torch.Tensor]:
        """Compare matched and distant-stage prompts on the same denoising task.

        Both TPC predictions use the router-only gradient path.  Consequently,
        the ordinary diffusion loss trains the denoiser while the TPC ranking
        loss specializes only the router and prompt bank.  The mismatch state
        reuses ``target_noise`` so timestep is the only changed variable.
        """
        # Detaching router inputs keeps TPC router-only even during full-model
        # fine-tuning; the ordinary losses still train the PromptIR backbone.
        matched_prompt, matched_routing_weights = self.router(
            lq_feature.detach(),
            noisy_feature.detach(),
            timesteps,
            trajectory_aware=trajectory_aware,
        )
        matched_prediction = self._router_only_grad_denoise(
            noisy_feature.detach(),
            lq_feature.detach(),
            matched_prompt,
            timesteps,
        )
        matched_loss_per_sample = self._per_sample_mse(
            matched_prediction,
            target_noise,
        )

        # 用相隔半条轨迹的 (z_t', t') 算 P'，再拿 P' 做原任务 (z_t, t)。
        mismatch_timesteps = self._sample_mismatch_timesteps(
            timesteps,
        )
        mismatch_noisy, _ = self.schedule.q_sample(
            clean_feature,
            mismatch_timesteps,
            noise=target_noise,
        )
        mismatch_prompt, mismatch_routing_weights = self.router(
            lq_feature.detach(),
            mismatch_noisy.detach(),
            mismatch_timesteps,
            trajectory_aware=trajectory_aware,
        )
        mismatch_prediction = self._router_only_grad_denoise(
            noisy_feature.detach(),
            lq_feature.detach(),
            mismatch_prompt,
            timesteps,
        )
        mismatch_loss_per_sample = self._per_sample_mse(
            mismatch_prediction,
            target_noise,
        )
        prediction_delta_per_sample = (
            matched_prediction.detach() - mismatch_prediction.detach()
        ).abs().flatten(1).mean(dim=1)
        routing_distance_per_sample = (
            matched_routing_weights - mismatch_routing_weights
        ).abs().mean(dim=1)
        return {
            "tpc_positive_loss_per_sample": matched_loss_per_sample,
            "tpc_mismatch_loss_per_sample": mismatch_loss_per_sample,
            "tpc_prediction_delta_per_sample": prediction_delta_per_sample,
            "tpc_routing_distance_per_sample": routing_distance_per_sample,
            "mismatch_loss": mismatch_loss_per_sample.mean(),
            "mismatch_timesteps": mismatch_timesteps,
            "mismatch_prompt": mismatch_prompt,
            "mismatch_routing_weights": mismatch_routing_weights,
        }

    @torch.no_grad()
    def sample(
        self,
        lq_feature: torch.Tensor,
        sample_steps: int = 4,
        trajectory_aware: bool = True,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if initial_noise is None:
            feature = torch.randn(
                lq_feature.shape,
                device=lq_feature.device,
                dtype=lq_feature.dtype,
                generator=generator,
            )
        else:
            if initial_noise.shape != lq_feature.shape:
                raise ValueError("initial_noise and lq_feature must have identical shapes")
            feature = initial_noise

        # 推理：从纯噪声出发走 sample_steps 步 DDIM；每一步都重新路由（trajectory-aware 的核心）
        timestep_grid = self.schedule.inference_timesteps(sample_steps, lq_feature.device)
        routing_history: List[torch.Tensor] = []

        for index, timestep_value in enumerate(timestep_grid):
            timesteps = timestep_value.expand(lq_feature.shape[0])
            if index + 1 < len(timestep_grid):
                previous = timestep_grid[index + 1].expand_as(timesteps)
            else:
                previous = torch.full_like(timesteps, -1)

            prompt, routing_weights = self.router(
                lq_feature,
                feature,
                timesteps,
                trajectory_aware=trajectory_aware,
            )
            predicted_noise = self.denoiser(feature, lq_feature, prompt, timesteps)
            feature = self.schedule.ddim_step(
                feature,
                timesteps,
                previous,
                predicted_noise,
            )
            routing_history.append(routing_weights)

        return {
            "predicted_clean": feature,
            "routing_weights": torch.stack(routing_history, dim=0),
            "timesteps": timestep_grid,
        }


class GatedPriorFusion(nn.Module):
    """Fuse a diffusion prior into the baseline using a zero-init residual path."""
    # 门控融合：baseline + 门 × 先验。两个卷积零初始化 → 第 0 步输出 ≡ 官方 PromptIR

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.prior_projection = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)

        # At initialization, fused == baseline exactly. This is the engineering
        # invariant used to detect accidental baseline regressions.
        nn.init.zeros_(self.prior_projection.weight)
        if self.prior_projection.bias is not None:
            nn.init.zeros_(self.prior_projection.bias)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        baseline_feature: torch.Tensor,
        prior_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if baseline_feature.shape != prior_feature.shape:
            raise ValueError("baseline_feature and prior_feature must have identical shapes")
        gate = torch.sigmoid(self.gate(torch.cat([baseline_feature, prior_feature], dim=1)))  # 每个通道一个 0~1 的"门"
        residual = gate * self.prior_projection(prior_feature)                                # 先验 × 门 = 决定注入多少"新东西"
        return baseline_feature + residual, gate
