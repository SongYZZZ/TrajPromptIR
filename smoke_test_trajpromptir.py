"""GPU smoke test for the first integrated TrajPromptIR milestone.

Run this before any long training job. It checks seven invariants:

1. zero-init fusion preserves the official PromptIR output exactly;
2. the training forward pass produces finite restoration/diffusion/TPC losses;
3. gradients reach the router, denoiser, and fusion path while the backbone is frozen;
4. short DDIM inference returns one routing distribution per trajectory step;
5. the TPC mismatch branch reaches the router without leaking gradients into the denoiser;
6. the complete TPC hinge updates only the router/prompt bank;
7. with trajectory inputs zeroed (static mode), TPC degrades to a harmless constant
   whose gradient vanishes, so Dynamic-vs-Static stays a clean comparison.
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
    parser.add_argument("--lambda_tpc", type=float, default=1.0)
    parser.add_argument("--tpc_margin", type=float, default=0.05)
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
    # 不变量 1：零初始化融合 → 训练前输出与官方 PromptIR 完全一致
    baseline_max_error = (expected - actual).abs().max().item()
    if baseline_max_error > 1e-6:
        raise RuntimeError(
            "zero-init fusion changed the baseline output: max error %.3e"
            % baseline_max_error
        )

    # 不变量 4：DDIM 推理每一步都重新路由，历史形状 = [步数, 批, 专家数]
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
    restored, training_aux = model(
        degraded,
        clean_image=clean,
        return_aux=True,
        tpc_enabled=args.lambda_tpc > 0,
    )
    restoration_loss = F.l1_loss(restored, clean)
    diffusion_loss = training_aux["diffusion_loss"]
    tpc_per_sample = torch.clamp(
        args.tpc_margin
        + training_aux["tpc_positive_loss_per_sample"]
        - training_aux["tpc_mismatch_loss_per_sample"],
        min=0.0,
    )
    tpc_loss = tpc_per_sample.mean()
    total_loss = restoration_loss + 0.1 * diffusion_loss + args.lambda_tpc * tpc_loss
    require_finite("restored", restored)
    require_finite("restoration_loss", restoration_loss)
    require_finite("diffusion_loss", diffusion_loss)
    require_finite("mismatch_loss", training_aux["mismatch_loss"])
    require_finite("tpc_loss", tpc_loss)

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
    # 不变量 3：梯度只到新增部件（router/denoiser/fusion），冻结的主干必须收不到
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

    # Repeat the complete TPC check with the backbone unfrozen: TPC must still
    # remain router-only during the later full fine-tuning stage.
    model.unfreeze_promptir_backbone()
    model.zero_grad(set_to_none=True)
    _, mismatch_aux = model(
        degraded,
        clean_image=clean,
        return_aux=True,
        tpc_enabled=True,
    )
    # 不变量 5：TPC 错配分支梯度只达路由器；去噪器泄漏必须恰好为 0（防作弊）
    mismatch_aux["mismatch_loss"].backward()
    mismatch_reaches_router = any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for parameter in model.feature_prior.router.parameters()
        if parameter.requires_grad
    )
    if not mismatch_reaches_router:
        raise RuntimeError("TPC mismatch branch does not reach the router")
    mismatch_denoiser_leak = [
        name
        for name, parameter in model.feature_prior.denoiser.named_parameters()
        if parameter.grad is not None
        and not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
    ]
    if mismatch_denoiser_leak:
        raise RuntimeError(
            "TPC mismatch branch leaked gradients into the denoiser: "
            + ", ".join(mismatch_denoiser_leak[:5])
        )

    model.zero_grad(set_to_none=True)
    _, tpc_aux = model(
        degraded,
        clean_image=clean,
        return_aux=True,
        tpc_enabled=True,
    )
    router_only_tpc = torch.clamp(
        args.tpc_margin
        + tpc_aux["tpc_positive_loss_per_sample"]
        - tpc_aux["tpc_mismatch_loss_per_sample"],
        min=0.0,
    ).mean()
    # 不变量 6：完整 TPC 正负两支只更新路由器，不更新去噪器或主干。
    router_only_tpc.backward()
    tpc_reaches_router = any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for parameter in model.feature_prior.router.parameters()
        if parameter.requires_grad
    )
    if not tpc_reaches_router:
        raise RuntimeError("complete TPC hinge does not reach the router")
    tpc_denoiser_leak = [
        name
        for name, parameter in model.feature_prior.denoiser.named_parameters()
        if parameter.grad is not None
        and not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
    ]
    if tpc_denoiser_leak:
        raise RuntimeError(
            "complete TPC hinge leaked gradients into the denoiser: "
            + ", ".join(tpc_denoiser_leak[:5])
        )
    tpc_non_router_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("feature_prior.router.")
        and parameter.grad is not None
        and not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
    ]
    if tpc_non_router_gradients:
        raise RuntimeError(
            "complete TPC hinge reached non-router parameters: "
            + ", ".join(tpc_non_router_gradients[:5])
        )

    model.zero_grad(set_to_none=True)
    _, static_aux = model(
        degraded,
        clean_image=clean,
        trajectory_aware=False,
        return_aux=True,
        tpc_enabled=True,
    )
    # 不变量 7：静态模式下正负 Prompt 相同，TPC 恒等于 margin 且梯度为 0。
    static_tpc_loss = torch.clamp(
        args.tpc_margin
        + static_aux["tpc_positive_loss_per_sample"]
        - static_aux["tpc_mismatch_loss_per_sample"],
        min=0.0,
    ).mean()
    if not torch.allclose(
        static_tpc_loss,
        static_tpc_loss.new_full((), args.tpc_margin),
    ):
        raise RuntimeError(
            "static-mode TPC is %.5f, expected the margin %.5f"
            % (static_tpc_loss.item(), args.tpc_margin)
        )
    model.zero_grad(set_to_none=True)
    static_tpc_loss.backward()
    static_router_gradients = [
        name
        for name, parameter in model.feature_prior.router.named_parameters()
        if parameter.grad is not None
        and not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
    ]
    if static_router_gradients:
        raise RuntimeError(
            "static-mode TPC moved the router: " + ", ".join(static_router_gradients[:5])
        )
    static_denoiser_gradients = [
        name
        for name, parameter in model.feature_prior.denoiser.named_parameters()
        if parameter.grad is not None
        and not torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
    ]
    if static_denoiser_gradients:
        raise RuntimeError(
            "static-mode TPC moved the denoiser: "
            + ", ".join(static_denoiser_gradients[:5])
        )

    report = {
        "status": "PASS",
        "checkpoint": checkpoint_status,
        "device": str(device),
        "baseline_max_error": baseline_max_error,
        "restoration_loss": restoration_loss.item(),
        "diffusion_loss": diffusion_loss.item(),
        "mismatch_loss": training_aux["mismatch_loss"].item(),
        "tpc_loss": tpc_loss.item(),
        "routing_history_shape": list(inference_aux["routing_weights"].shape),
        "addon_gradients": addon_gradients,
        "tpc_mismatch_reaches_router": mismatch_reaches_router,
        "tpc_mismatch_denoiser_leak": mismatch_denoiser_leak,
        "complete_tpc_reaches_router": tpc_reaches_router,
        "complete_tpc_denoiser_leak": tpc_denoiser_leak,
        "complete_tpc_non_router_gradients": tpc_non_router_gradients,
        "static_tpc_loss": static_tpc_loss.item(),
        "static_tpc_denoiser_gradients": static_denoiser_gradients,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
