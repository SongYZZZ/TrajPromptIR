"""GPU smoke test for the first integrated TrajPromptIR milestone.

Run this before any long training job. It checks four invariants:

1. zero-init fusion preserves the official PromptIR output exactly;
2. the training forward pass produces finite restoration/diffusion losses;
3. gradients reach the router, denoiser, and fusion path while the backbone is frozen;
4. short DDIM inference returns one routing distribution per trajectory step.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from net.model import PromptIR
from net.trajpromptir import TrajPromptIR, load_promptir_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt/model.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_random_baseline(model, baseline):
    """Make the no-checkpoint path deterministic for architecture-only testing."""
    model_state = model.state_dict()
    baseline_state = baseline.state_dict()
    baseline.load_state_dict({key: model_state[key] for key in baseline_state})


def require_finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError("%s contains NaN or Inf" % name)


def main():
    args = parse_args()
    if args.image_size % 8:
        raise ValueError("image_size must be divisible by 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    seed_everything(args.seed)
    device = torch.device(args.device)

    model = TrajPromptIR(decoder=True).to(device)
    baseline = PromptIR(decoder=True).to(device)
    if os.path.isfile(args.checkpoint):
        load_promptir_checkpoint(model, args.checkpoint)
        load_promptir_checkpoint(baseline, args.checkpoint)
        checkpoint_status = "loaded"
    else:
        copy_random_baseline(model, baseline)
        checkpoint_status = "not found; used identical random baseline weights"

    degraded = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    clean = torch.rand_like(degraded)

    model.eval()
    baseline.eval()
    with torch.no_grad():
        expected = baseline(degraded)
        actual, inference_aux = model(
            degraded,
            sample_steps=args.sample_steps,
            return_aux=True,
        )
    baseline_max_error = (expected - actual).abs().max().item()
    if baseline_max_error > 1e-6:
        raise RuntimeError(
            "zero-init fusion changed the baseline output: max error %.3e"
            % baseline_max_error
        )

    expected_routing_shape = (
        args.sample_steps,
        degraded.shape[0],
        model.feature_prior.router.num_experts,
    )
    if tuple(inference_aux["routing_weights"].shape) != expected_routing_shape:
        raise RuntimeError(
            "routing history has shape %r, expected %r"
            % (tuple(inference_aux["routing_weights"].shape), expected_routing_shape)
        )

    model.freeze_promptir_backbone()
    model.train()
    restored, training_aux = model(degraded, clean_image=clean, return_aux=True)
    restoration_loss = F.l1_loss(restored, clean)
    diffusion_loss = training_aux["diffusion_loss"]
    total_loss = restoration_loss + 0.1 * diffusion_loss
    require_finite("restored", restored)
    require_finite("restoration_loss", restoration_loss)
    require_finite("diffusion_loss", diffusion_loss)

    total_loss.backward()
    addon_gradients = {
        "router": any(
            parameter.grad is not None
            for parameter in model.feature_prior.router.parameters()
            if parameter.requires_grad
        ),
        "denoiser": any(
            parameter.grad is not None
            for parameter in model.feature_prior.denoiser.parameters()
            if parameter.requires_grad
        ),
        "fusion": any(
            parameter.grad is not None
            for parameter in model.prior_fusion.parameters()
            if parameter.requires_grad
        ),
    }
    if not all(addon_gradients.values()):
        raise RuntimeError("missing add-on gradients: %r" % addon_gradients)

    backbone_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith(("feature_prior.", "prior_fusion."))
        and parameter.grad is not None
    ]
    if backbone_gradients:
        raise RuntimeError(
            "frozen backbone unexpectedly received gradients: "
            + ", ".join(backbone_gradients[:5])
        )

    report = {
        "status": "PASS",
        "checkpoint": checkpoint_status,
        "device": str(device),
        "baseline_max_error": baseline_max_error,
        "restoration_loss": restoration_loss.item(),
        "diffusion_loss": diffusion_loss.item(),
        "routing_history_shape": list(inference_aux["routing_weights"].shape),
        "addon_gradients": addon_gradients,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
