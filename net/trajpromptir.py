"""Stage-5 TrajPromptIR: feature diffusion prior integrated into PromptIR.

The model keeps the official PromptIR encoder, decoder, and prompt modules. The
only added path is:

    clean bottleneck -> forward diffusion -> epsilon prediction
                                      ^       |
    LQ bottleneck -> trajectory router -------+
                                      |
                              predicted clean prior
                                      |
                         zero-init gated fusion -> PromptIR decoder

During inference the clean image is unavailable, so the prior is generated with
a short deterministic DDIM trajectory (2/4/8/16 steps are the intended ablation).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model import PromptIR
from net.trajpromptir_components import GatedPriorFusion, TrajectoryFeatureDiffusionPrior


class TrajPromptIR(PromptIR):
    """PromptIR plus a trajectory-aware bottleneck diffusion prior."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        decoder: bool = True,
        diffusion_steps: int = 200,
        diffusion_hidden_channels: int = 96,
        diffusion_blocks: int = 3,
        prompt_dim: int = 64,
        prompt_experts: int = 8,
        prompt_top_k: Optional[int] = None,
        feature_scale: float = 4.0,
        detach_clean_feature: bool = True,
    ) -> None:
        if not decoder:
            raise ValueError("TrajPromptIR requires decoder=True")
        if feature_scale <= 0:
            raise ValueError("feature_scale must be positive")
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks,
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            decoder=decoder,
        )

        bottleneck_channels = int(dim * 2**3)
        self.feature_scale = feature_scale
        self.detach_clean_feature = detach_clean_feature
        self.feature_prior = TrajectoryFeatureDiffusionPrior(
            feature_channels=bottleneck_channels,
            num_diffusion_steps=diffusion_steps,
            prompt_dim=prompt_dim,
            num_experts=prompt_experts,
            hidden_channels=diffusion_hidden_channels,
            num_blocks=diffusion_blocks,
            top_k=prompt_top_k,
        )
        self.prior_fusion = GatedPriorFusion(bottleneck_channels)

    def encode_backbone(
        self,
        image: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return PromptIR skip features and its lowest-resolution bottleneck."""
        level1 = self.encoder_level1(self.patch_embed(image))
        level2 = self.encoder_level2(self.down1_2(level1))
        level3 = self.encoder_level3(self.down2_3(level2))
        latent = self.latent(self.down3_4(level3))
        return (level1, level2, level3), latent

    def decode_backbone(
        self,
        input_image: torch.Tensor,
        encoder_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Run the unchanged official PromptIR decoder from a supplied latent."""
        level1, level2, level3 = encoder_features

        prompt3 = self.prompt3(latent)
        latent = self.reduce_noise_level3(
            self.noise_level3(torch.cat([latent, prompt3], dim=1))
        )

        decoded3 = self.up4_3(latent)
        decoded3 = self.reduce_chan_level3(torch.cat([decoded3, level3], dim=1))
        decoded3 = self.decoder_level3(decoded3)
        prompt2 = self.prompt2(decoded3)
        decoded3 = self.reduce_noise_level2(
            self.noise_level2(torch.cat([decoded3, prompt2], dim=1))
        )

        decoded2 = self.up3_2(decoded3)
        decoded2 = self.reduce_chan_level2(torch.cat([decoded2, level2], dim=1))
        decoded2 = self.decoder_level2(decoded2)
        prompt1 = self.prompt1(decoded2)
        decoded2 = self.reduce_noise_level1(
            self.noise_level1(torch.cat([decoded2, prompt1], dim=1))
        )

        decoded1 = self.up2_1(decoded2)
        decoded1 = torch.cat([decoded1, level1], dim=1)
        decoded1 = self.decoder_level1(decoded1)
        decoded1 = self.refinement(decoded1)
        return self.output(decoded1) + input_image

    def forward_baseline(self, input_image: torch.Tensor) -> torch.Tensor:
        """Run the decomposed encoder/decoder without the new prior branch."""
        encoder_features, latent = self.encode_backbone(input_image)
        return self.decode_backbone(input_image, encoder_features, latent)

    def freeze_promptir_backbone(self) -> None:
        """Freeze all official PromptIR parameters while leaving additions trainable."""
        for name, parameter in self.named_parameters():
            is_addition = name.startswith("feature_prior.") or name.startswith("prior_fusion.")
            parameter.requires_grad_(is_addition)

    def unfreeze_promptir_backbone(self) -> None:
        """Enable end-to-end fine-tuning after the add-on path is stable."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def trainable_parameter_groups(
        self,
        addon_lr: float,
        backbone_lr: Optional[float] = None,
    ):
        """Build explicit optimizer groups for staged training."""
        additions = []
        backbone = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("feature_prior.") or name.startswith("prior_fusion."):
                additions.append(parameter)
            else:
                backbone.append(parameter)

        groups = []
        if additions:
            groups.append({"params": additions, "lr": addon_lr, "name": "traj_additions"})
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": backbone_lr if backbone_lr is not None else addon_lr,
                    "name": "promptir_backbone",
                }
            )
        return groups

    def forward(
        self,
        input_image: torch.Tensor,
        clean_image: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        sample_steps: int = 4,
        trajectory_aware: bool = True,
        return_aux: bool = False,
    ):
        """Run one training diffusion step or a short inference trajectory.

        Training mode is selected by providing ``clean_image``. The returned
        auxiliary dictionary contains everything required for ``L_diff`` and the
        future routing visualizations. Without ``clean_image``, a DDIM trajectory
        is sampled from noise using only the LQ bottleneck as condition.
        """
        encoder_features, lq_latent = self.encode_backbone(input_image)
        normalized_lq_latent = lq_latent / self.feature_scale

        if clean_image is not None:
            if clean_image.shape != input_image.shape:
                raise ValueError("clean_image and input_image must have identical shapes")
            if self.detach_clean_feature:
                with torch.no_grad():
                    _, clean_latent = self.encode_backbone(clean_image)
                clean_latent = clean_latent.detach()
            else:
                _, clean_latent = self.encode_backbone(clean_image)

            prior_output = self.feature_prior.forward_train(
                clean_feature=clean_latent / self.feature_scale,
                lq_feature=normalized_lq_latent,
                timesteps=timesteps,
                noise=noise,
                trajectory_aware=trajectory_aware,
            )
        else:
            prior_output = self.feature_prior.sample(
                lq_feature=normalized_lq_latent,
                sample_steps=sample_steps,
                trajectory_aware=trajectory_aware,
            )

        fused_latent, fusion_gate = self.prior_fusion(
            lq_latent,
            prior_output["predicted_clean"] * self.feature_scale,
        )
        restored = self.decode_backbone(input_image, encoder_features, fused_latent)

        if not return_aux:
            return restored

        auxiliary: Dict[str, torch.Tensor] = dict(prior_output)
        auxiliary["predicted_clean_raw"] = (
            prior_output["predicted_clean"] * self.feature_scale
        )
        auxiliary["fusion_gate"] = fusion_gate
        if clean_image is not None:
            auxiliary["diffusion_loss"] = F.mse_loss(
                prior_output["predicted_noise"],
                prior_output["target_noise"],
            )
        return restored, auxiliary


def load_promptir_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Dict[str, Tuple[str, ...]]:
    """Load an official Lightning checkpoint into PromptIR or TrajPromptIR.

    Added TrajPromptIR parameters are expected to be missing when an official
    checkpoint is loaded; missing baseline parameters are treated as an error.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        (key[4:] if key.startswith("net.") else key): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_prefixes = ("feature_prior.", "prior_fusion.")
    missing_baseline = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
    ]
    if missing_baseline:
        raise RuntimeError(
            "checkpoint is missing PromptIR baseline parameters: "
            + ", ".join(missing_baseline[:10])
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint has unexpected parameters: "
            + ", ".join(incompatible.unexpected_keys[:10])
        )
    return {
        "missing_additions": tuple(incompatible.missing_keys),
        "unexpected": tuple(incompatible.unexpected_keys),
    }
