"""Core modules for the first trainable TrajPromptIR integration.

This file intentionally contains only the new research variables:

1. a trajectory-aware prompt router;
2. a conditional feature-space diffusion prior;
3. a conservative gated fusion layer.

The official PromptIR implementation in ``net/model.py`` stays untouched so the
baseline remains reproducible.
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
        noisy = alpha_bar.sqrt() * clean_feature + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def predict_clean(
        self,
        noisy_feature: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the x_0 estimate implied by an epsilon prediction."""
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

        degradation_state = degradation_feature.mean(dim=(-2, -1))
        trajectory_state = diffusion_state.mean(dim=(-2, -1))
        time_state = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.time_dim)
        )

        if not trajectory_aware:
            trajectory_state = torch.zeros_like(trajectory_state)
            time_state = torch.zeros_like(time_state)

        router_input = torch.cat(
            [degradation_state, trajectory_state, time_state],
            dim=1,
        )
        logits = self.router(self.input_norm(router_input)) / self.temperature

        if self.top_k is not None and self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            logits = sparse_logits.scatter(1, top_indices, top_values)

        weights = torch.softmax(logits, dim=1)
        prompt = weights @ self.prompt_bank
        return prompt, weights


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FeatureResidualBlock(nn.Module):
    """Small FiLM-conditioned residual block for bottleneck features."""

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
        self.condition_mlp = nn.Sequential(
            nn.Linear(time_dim + prompt_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        self.blocks = nn.ModuleList(
            [FeatureResidualBlock(hidden_channels, hidden_channels * 2) for _ in range(num_blocks)]
        )
        self.output_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.output = nn.Conv2d(hidden_channels, feature_channels, 1)

        # Start from epsilon_hat = 0. This makes the first optimization steps
        # easy to interpret and prevents a random prior from destabilizing fusion.
        nn.init.zeros_(self.output.weight)
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
        condition = self.condition_mlp(torch.cat([time_embedding, prompt], dim=1))
        hidden = self.noisy_projection(noisy_feature) + self.lq_projection(lq_feature)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(F.silu(self.output_norm(hidden)))


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
        return {
            "predicted_clean": predicted_clean,
            "predicted_noise": predicted_noise,
            "target_noise": target_noise,
            "noisy_feature": noisy_feature,
            "timesteps": timesteps,
            "prompt": prompt,
            "routing_weights": routing_weights,
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
        gate = torch.sigmoid(self.gate(torch.cat([baseline_feature, prior_feature], dim=1)))
        residual = gate * self.prior_projection(prior_feature)
        return baseline_feature + residual, gate
