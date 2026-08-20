"""Evaluate PromptIR, TrajPromptIR, and GT-latent oracle variants.

Examples:

    python test_trajpromptir.py --mode traj --checkpoint train_ckpt/run/last.pt \
        --task denoise --data_path test/denoise/bsd68 --sigma 25

    python test_trajpromptir.py --mode oracle_replace \
        --checkpoint train_ckpt/run/last.pt --task derain \
        --data_path test/derain/Rain100L

The oracle modes use the paired clean image only for diagnosis. They are not
deployable inference methods and must never be reported as normal test scores.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from net.model import PromptIR
from net.trajpromptir import TrajPromptIR
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible PSNR/SSIM evaluation for TrajPromptIR"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "traj", "oracle_replace", "oracle_fusion"],
        default="traj",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["denoise", "derain", "dehaze"], required=True)
    parser.add_argument(
        "--data_path",
        required=True,
        help="denoise image directory or paired task root containing input/ and target/",
    )
    parser.add_argument("--sigma", type=int, choices=[15, 25, 50], default=25)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--static_prompt", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", default="eval_output/trajpromptir")
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def with_separator(path):
    return path if path.endswith(("/", "\\")) else path + os.sep


def build_dataset(args):
    data_path = with_separator(args.data_path)
    if args.task == "denoise":
        dataset_args = argparse.Namespace(denoise_path=data_path)
        dataset = DenoiseTestDataset(dataset_args)
        dataset.clean_ids.sort()
        dataset.set_sigma(args.sigma)
    else:
        dataset_args = argparse.Namespace(
            derain_path=data_path,
            dehaze_path=data_path,
        )
        dataset = DerainDehazeDataset(dataset_args, task=args.task)
        dataset.ids.sort()

    if len(dataset) == 0:
        raise ValueError("evaluation dataset is empty: %s" % args.data_path)
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    return dataset


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "net.")):
            key = key.split(".", 1)[1]
        normalized[key] = value
    return normalized


def load_evaluation_checkpoint(model, checkpoint_path, baseline_only=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = normalize_state_dict(state_dict)

    expected = model.state_dict()
    missing = [key for key in expected if key not in state_dict]
    shape_mismatch = [
        key
        for key in expected
        if key in state_dict and expected[key].shape != state_dict[key].shape
    ]
    if missing or shape_mismatch:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:5]))
        if shape_mismatch:
            details.append("shape_mismatch=" + ", ".join(shape_mismatch[:5]))
        raise RuntimeError("incompatible checkpoint: " + "; ".join(details))

    selected = {key: state_dict[key] for key in expected}
    model.load_state_dict(selected, strict=True)
    ignored = [key for key in state_dict if key not in expected]
    if ignored and not baseline_only:
        raise RuntimeError(
            "TrajPromptIR checkpoint has unexpected keys: " + ", ".join(ignored[:5])
        )
    return {
        "global_step": checkpoint.get("global_step") if isinstance(checkpoint, dict) else None,
        "ignored_addition_keys": len(ignored),
    }


def first_name(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_diffusion_noise(seed):
    """Reset only Torch RNGs; NumPy owns deterministic dataset degradation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_model(model, mode, degraded, clean, args):
    if mode == "baseline":
        return model(degraded), {}
    if mode == "traj":
        restored, auxiliary = model(
            degraded,
            sample_steps=args.sample_steps,
            trajectory_aware=not args.static_prompt,
            return_aux=True,
        )
        routing = auxiliary["routing_weights"]
        entropy = -(
            routing * routing.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        diagnostics = {
            "routing_entropy": entropy.detach(),
            "effective_prompts": entropy.exp().detach(),
        }
        diagnostics.update(model.prior_fusion.last_diagnostics)
        return restored, diagnostics

    encoder_features, lq_latent = model.encode_backbone(degraded)
    _, gt_latent = model.encode_backbone(clean)
    if mode == "oracle_replace":
        # Upper bound for the bottleneck location itself: give the decoder the
        # exact GT bottleneck while retaining LQ skip features.
        restored = model.decode_backbone(degraded, encoder_features, gt_latent)
        return restored, {}

    # This lower oracle also tests whether the currently learned Fusion can use
    # a perfect prior. A large replace gain but tiny fusion gain implicates Fusion.
    fused_latent, _ = model.prior_fusion(lq_latent, gt_latent)
    restored = model.decode_backbone(degraded, encoder_features, fused_latent)
    return restored, dict(model.prior_fusion.last_diagnostics)


def main():
    args = parse_args()
    if args.sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    if args.max_images < 0:
        raise ValueError("max_images must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.mode == "baseline":
        model = PromptIR(decoder=True)
    else:
        model = TrajPromptIR(decoder=True)
    checkpoint_info = load_evaluation_checkpoint(
        model,
        args.checkpoint,
        baseline_only=args.mode == "baseline",
    )
    model = model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    image_output_dir = os.path.join(args.output_dir, "images")
    if args.save_images:
        os.makedirs(image_output_dir, exist_ok=True)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    time_meter = AverageMeter()
    diagnostic_meters = {}
    per_image = []

    with torch.inference_mode():
        for index, (name_field, degraded, clean) in enumerate(loader):
            name = first_name(name_field)
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # Each image gets its own deterministic diffusion noise, independent
            # of dataset length and whether output saving is enabled.
            seed_diffusion_noise(args.seed + index)
            synchronize(device)
            start = time.perf_counter()
            restored, diagnostics = run_model(
                model, args.mode, degraded, clean, args
            )
            synchronize(device)
            elapsed = time.perf_counter() - start
            restored = restored.clamp(0.0, 1.0)

            image_psnr, image_ssim, count = compute_psnr_ssim(restored, clean)
            image_psnr = float(image_psnr)
            image_ssim = float(image_ssim)
            psnr_meter.update(image_psnr, count)
            ssim_meter.update(image_ssim, count)
            time_meter.update(elapsed, count)

            image_record = {
                "name": name,
                "psnr": image_psnr,
                "ssim": image_ssim,
                "inference_seconds": elapsed,
            }
            for diagnostic_name, diagnostic_value in diagnostics.items():
                value = float(diagnostic_value.detach().mean().cpu())
                diagnostic_meters.setdefault(
                    diagnostic_name, AverageMeter()
                ).update(value, count)
                image_record[diagnostic_name] = value
            per_image.append(image_record)

            if args.save_images:
                safe_name = os.path.basename(name).replace(os.sep, "_")
                save_image_tensor(
                    restored,
                    os.path.join(image_output_dir, safe_name + ".png"),
                )

    result = {
        "mode": args.mode,
        "task": args.task,
        "sigma": args.sigma if args.task == "denoise" else None,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": checkpoint_info["global_step"],
        "trajectory_aware": (
            not args.static_prompt if args.mode == "traj" else None
        ),
        "sample_steps": args.sample_steps if args.mode == "traj" else None,
        "seed": args.seed,
        "num_images": psnr_meter.count,
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
        "mean_inference_seconds": time_meter.avg,
        "diagnostics": {
            name: meter.avg for name, meter in diagnostic_meters.items()
        },
        "per_image": per_image,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved metrics: %s" % metrics_path)


if __name__ == "__main__":
    main()
