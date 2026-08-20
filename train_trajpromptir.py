"""Single-GPU staged trainer for the first integrated TrajPromptIR model.

Stage A (default) freezes official PromptIR and trains only the router, feature
denoiser, and gated fusion. Stage B optionally fine-tunes the full network with a
smaller backbone learning rate. TPC (trajectory prompt contrast) is enabled by
--lambda_tpc > 0 and adds a hinge loss that requires the matched-stage prompt to
beat the mismatched-stage prompt by --tpc_margin.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from net.trajpromptir import TrajPromptIR, load_promptir_checkpoint
from utils.dataset_utils import PromptTrainDataset


def with_separator(path):
    return path if path.endswith(("/", "\\")) else path + os.sep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt/model.ckpt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output_dir", default="train_ckpt/trajpromptir_stage5")

    parser.add_argument("--data_file_dir", default="data_dir/")
    parser.add_argument("--denoise_dir", default="data/Train/Denoise/")
    parser.add_argument("--derain_dir", default="data/Train/Derain/")
    parser.add_argument("--dehaze_dir", default="data/Train/Dehaze/")
    parser.add_argument(
        "--de_type",
        nargs="+",
        default=["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze"],
    )
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mini_clean_image",
        default=None,
        help="optional clean image for a dependency-light 10-20 sample mini-overfit",
    )
    parser.add_argument("--mini_samples", type=int, default=20)

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epoch limit used only when max_steps <= 0",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="authoritative optimizer-step limit when positive",
    )
    parser.add_argument("--addon_lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--lambda_diff", type=float, default=0.1)
    parser.add_argument(
        "--lambda_tpc",
        type=float,
        default=0.0,
        help="weight of the TPC hinge loss; 0 disables the mismatch branch entirely",
    )
    parser.add_argument(
        "--tpc_margin",
        type=float,
        default=0.05,
        help="required edge of the matched prompt over the mismatched prompt",
    )
    parser.add_argument(
        "--tpc_route_margin",
        type=float,
        default=0.05,
        help="minimum mean-L1 distance between matched and distant routing distributions",
    )
    parser.add_argument(
        "--tpc_route_weight",
        type=float,
        default=1.0,
        help="weight of the routing-separation term inside TPC",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--stage", choices=["addon", "finetune"], default="addon")
    parser.add_argument("--static_prompt", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--no_save", action="store_true", help="run without writing checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args):
    if args.patch_size % 8:
        raise ValueError("patch_size must be divisible by 8")
    if args.lambda_diff < 0:
        raise ValueError("lambda_diff must be non-negative")
    if args.lambda_tpc < 0:
        raise ValueError("lambda_tpc must be non-negative")
    if args.tpc_margin < 0:
        raise ValueError("tpc_margin must be non-negative")
    if args.tpc_route_margin < 0:
        raise ValueError("tpc_route_margin must be non-negative")
    if args.tpc_route_weight < 0:
        raise ValueError("tpc_route_weight must be non-negative")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.mini_samples < 1:
        raise ValueError("mini_samples must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not os.path.isfile(args.checkpoint) and args.resume is None:
        raise FileNotFoundError("official checkpoint not found: %s" % args.checkpoint)

    args.data_file_dir = with_separator(args.data_file_dir)
    args.denoise_dir = with_separator(args.denoise_dir)
    args.derain_dir = with_separator(args.derain_dir)
    args.dehaze_dir = with_separator(args.dehaze_dir)


class MiniSyntheticRestorationDataset(Dataset):
    """Generate repeatable LQ/GT pairs from one clean image for mini-overfit."""

    def __init__(self, image_path, patch_size, length, seed):
        if not os.path.isfile(image_path):
            raise FileNotFoundError("mini clean image not found: %s" % image_path)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        if min(image.shape[:2]) < patch_size:
            raise ValueError("mini clean image is smaller than patch_size")
        self.image = image
        self.patch_size = patch_size
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        height, width = self.image.shape[:2]
        top = int(rng.integers(0, height - self.patch_size + 1))
        left = int(rng.integers(0, width - self.patch_size + 1))
        clean = self.image[
            top : top + self.patch_size,
            left : left + self.patch_size,
        ].copy()
        if rng.random() < 0.5:
            clean = clean[:, ::-1].copy()

        # 每 3 个样本轮换一种退化：0=高斯噪声 1=雾 2=雨（seed 固定，完全可复现）
        degradation_id = index % 3
        if degradation_id == 0:
            sigma = float(rng.choice([15.0, 25.0, 50.0])) / 255.0
            degraded = np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
        elif degradation_id == 1:
            transmission = float(rng.uniform(0.45, 0.8))
            degraded = np.clip(clean * transmission + (1.0 - transmission), 0.0, 1.0)
        else:
            degraded = clean.copy()
            rain = np.zeros(clean.shape[:2], dtype=np.float32)
            for _ in range(max(20, self.patch_size)):
                x = int(rng.integers(0, self.patch_size))
                y = int(rng.integers(0, self.patch_size))
                length = int(rng.integers(5, max(6, self.patch_size // 3)))
                end = min(self.patch_size, y + length)
                rain[y:end, x] += float(rng.uniform(0.15, 0.5))
            degraded = np.clip(degraded + rain[:, :, None], 0.0, 1.0)

        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        degraded_tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).float()
        return (["mini_%03d" % index, degradation_id], degraded_tensor, clean_tensor)


def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        path,
    )


def load_training_checkpoint(path, model, optimizer=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mini_clean_image is not None:
        dataset = MiniSyntheticRestorationDataset(
            args.mini_clean_image,
            patch_size=args.patch_size,
            length=args.mini_samples,
            seed=args.seed,
        )
    else:
        dataset = PromptTrainDataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(
            "training loader is empty; reduce batch_size or provide more samples"
        )

    model = TrajPromptIR(decoder=True).to(device)
    if args.resume is None:
        load_promptir_checkpoint(model, args.checkpoint)

    if args.stage == "addon":
        model.freeze_promptir_backbone()
        parameter_groups = model.trainable_parameter_groups(args.addon_lr)
    else:
        model.unfreeze_promptir_backbone()
        parameter_groups = model.trainable_parameter_groups(
            args.addon_lr,
            backbone_lr=args.backbone_lr,
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)

    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        state = load_training_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)

    print(
        json.dumps(
            {
                "stage": args.stage,
                "trajectory_aware": not args.static_prompt,
                "lambda_tpc": args.lambda_tpc,
                "tpc_margin": args.tpc_margin,
                "tpc_route_margin": args.tpc_route_margin,
                "tpc_route_weight": args.tpc_route_weight,
                "dataset_samples": len(dataset),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "amp": amp_enabled,
                "start_step": global_step,
            },
            indent=2,
        )
    )

    model.train()
    epoch = start_epoch
    while True:
        if args.max_steps > 0:
            if global_step >= args.max_steps:
                break
        elif epoch >= args.epochs:
            break

        for batch in loader:
            _, degraded, clean = batch
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                restored, auxiliary = model(
                    degraded,
                    clean_image=clean,
                    trajectory_aware=not args.static_prompt,
                    return_aux=True,
                    tpc_enabled=args.lambda_tpc > 0,
                )
                # 三个损失各管一件事：
                # ① L1：重建质量（输出图贴近干净图）——所有复原网络的标配
                # ② diff：扩散损失（预测噪声 ε̂ 贴近真实 ε）——训练去噪器+路由器
                # ③ TPC hinge：匹配时间步的 prompt 必须比错配的好 margin 以上
                restoration_loss = F.l1_loss(restored, clean)
                diffusion_loss = auxiliary["diffusion_loss"]
                total_loss = restoration_loss + args.lambda_diff * diffusion_loss
                tpc_loss = None
                if args.lambda_tpc > 0:
                    positive_per_sample = auxiliary["tpc_positive_loss_per_sample"]
                    mismatch_per_sample = auxiliary["tpc_mismatch_loss_per_sample"]
                    tpc_per_sample = torch.clamp(
                        args.tpc_margin + positive_per_sample - mismatch_per_sample,
                        min=0.0,
                    )
                    route_tpc_per_sample = torch.clamp(
                        args.tpc_route_margin
                        - auxiliary["tpc_routing_distance_per_sample"],
                        min=0.0,
                    )
                    route_tpc_loss = route_tpc_per_sample.mean()
                    tpc_loss = (
                        tpc_per_sample.mean()
                        + args.tpc_route_weight * route_tpc_loss
                    )
                    total_loss = total_loss + args.lambda_tpc * tpc_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                weights = auxiliary["routing_weights"].detach()
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
                effective_experts = entropy.exp()
                fusion_diagnostics = model.prior_fusion.last_diagnostics
                gate_mean = fusion_diagnostics["gate_mean"].mean()
                residual_norm = fusion_diagnostics["residual_norm"].mean()
                fusion_ratio = fusion_diagnostics["fusion_ratio"].mean()
                if tpc_loss is not None:
                    positive_loss = positive_per_sample.detach().mean()
                    mismatch_loss = mismatch_per_sample.detach().mean()
                    pos_neg_gap = mismatch_loss - positive_loss
                    active_rate = (tpc_per_sample.detach() > 0).float().mean()
                    route_delta = (
                        weights - auxiliary["mismatch_routing_weights"].detach()
                    ).abs().mean()
                    prompt_delta = (
                        auxiliary["prompt"].detach()
                        - auxiliary["mismatch_prompt"].detach()
                    ).abs().mean()
                    prediction_delta = auxiliary[
                        "tpc_prediction_delta_per_sample"
                    ].detach().mean()
                    print(
                        "step %6d | rec %.5f | diff %.5f | pos %.5f | neg %.5f | gap %+.5f | tpc %.5f | route_tpc %.5f | active %.2f | route_d %.5f | prompt_d %.5f | pred_d %.6f | gate %.4f | res_norm %.6f | fusion_ratio %.6f | H %.3f | eff %.2f | total %.5f | %s"
                        % (
                            global_step,
                            restoration_loss.item(),
                            diffusion_loss.item(),
                            positive_loss.item(),
                            mismatch_loss.item(),
                            pos_neg_gap.item(),
                            tpc_loss.item(),
                            route_tpc_loss.item(),
                            active_rate.item(),
                            route_delta.item(),
                            prompt_delta.item(),
                            prediction_delta.item(),
                            gate_mean.item(),
                            residual_norm.item(),
                            fusion_ratio.item(),
                            entropy.item(),
                            effective_experts.item(),
                            total_loss.item(),
                            time.strftime("%H:%M:%S"),
                        )
                    )
                else:
                    print(
                        "step %6d | rec %.5f | diff %.5f | gate %.4f | res_norm %.6f | fusion_ratio %.6f | H %.3f | eff %.2f | total %.5f | %s"
                        % (
                            global_step,
                            restoration_loss.item(),
                            diffusion_loss.item(),
                            gate_mean.item(),
                            residual_norm.item(),
                            fusion_ratio.item(),
                            entropy.item(),
                            effective_experts.item(),
                            total_loss.item(),
                            time.strftime("%H:%M:%S"),
                        )
                    )

            if not args.no_save and global_step % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.output_dir, "step_%06d.pt" % global_step),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    args,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        epoch += 1

    if args.no_save:
        print("training finished without checkpoint output (--no_save)")
    else:
        final_path = os.path.join(args.output_dir, "last.pt")
        save_checkpoint(
            final_path,
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            args,
        )
        print("saved: %s" % final_path)


if __name__ == "__main__":
    main()
