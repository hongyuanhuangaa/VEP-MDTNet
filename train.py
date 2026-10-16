"""Train VEP-MDTNet or one of its ablation / single-modality variants.

Example:
    python train.py --preset full --data-root data/82lb --out-dir runs

Training protocol of the paper (all defaults below): AdamW (betas 0.95/0.999, weight decay
5e-4), lr 1e-4 with a 10-epoch linear warm-up from 0.1x followed by cosine decay to 0.005x,
batch 8 with 4-step gradient accumulation (effective batch 32), mixed precision on CUDA,
gradient clipping at 1.0, loss 0.3 BCE + 0.7 Dice, 100 epochs without early stopping and
the checkpoint with the highest validation mIoU kept.  Online augmentation: random flips /
90-degree rotations and a synchronized RGB/DEM/mask CutMix whose probability adapts to the
validation mIoU trend within [0.1, 0.4].
"""

from __future__ import annotations

import os

# Thread policy used for all experiments; must be set before numpy/torch are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from vepmdtnet.checkpoint import build, model_inputs
from vepmdtnet.complexity import profile_model_complexity
from vepmdtnet.data import TerraceDataset, load_or_compute_terrain_statistics
from vepmdtnet.losses import MainSegmentationLoss
from vepmdtnet.presets import PRESETS, make_config, preset_switches


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_print(message: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def format_duration(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class WarmupCosineAnnealingV2:
    """Linear warm-up from 0.1x to 1x, then cosine decay to ``min_factor`` (per epoch)."""

    def __init__(self, optimizer, warmup_epochs: int = 10, max_epochs: int = 100,
                 min_factor: float = 0.005) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_factor = min_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / max(1, self.warmup_epochs)
            factor = 0.1 + 0.9 * alpha
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_factor + (1.0 - self.min_factor) * cosine_decay
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[index] * factor
        return self.optimizer.param_groups[0]["lr"]


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    """One AdamW parameter group with every trainable parameter."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("The model has no trainable parameters")
    return torch.optim.AdamW(
        [{"name": "all_parameters", "params": parameters, "lr": learning_rate}],
        weight_decay=weight_decay,
        betas=(0.95, 0.999),
    )


# ============================================================
# Streaming binary metrics (class means over background and terrace)
# ============================================================


def update_confusion(confusion: np.ndarray, prediction: Tensor, target: Tensor) -> None:
    prediction_np = prediction.detach().long().cpu().numpy().reshape(-1)
    target_np = target.detach().long().cpu().numpy().reshape(-1)
    confusion += np.bincount(target_np * 2 + prediction_np, minlength=4).reshape(2, 2)


def metrics_from_confusion(confusion: np.ndarray, prefix: str = "val_") -> Dict[str, float]:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    precision = tp / (tp + fp + 1.0e-6)
    recall = tp / (tp + fn + 1.0e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1.0e-6)
    iou = tp / (tp + fp + fn + 1.0e-6)
    oa = tp.sum() / (confusion.sum() + 1.0e-6)
    return {
        prefix + "Precision": float(precision.mean()),
        prefix + "Recall": float(recall.mean()),
        prefix + "F1": float(f1.mean()),
        prefix + "mIoU": float(iou.mean()),
        prefix + "OA": float(oa),
        prefix + "fg_Precision": float(precision[1]),
        prefix + "fg_Recall": float(recall[1]),
        prefix + "fg_F1": float(f1[1]),
        prefix + "fg_IoU": float(iou[1]),
    }


def segmentation_prediction(logits: Tensor) -> Tensor:
    if logits.size(1) == 2:
        return torch.argmax(logits, dim=1)
    if logits.size(1) == 1:
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    raise ValueError("Binary terrace segmentation expects one or two output channels")


# ============================================================
# Synchronized CutMix with an adaptive probability
# ============================================================


def synchronized_cutmix(
    inputs: Sequence[Tensor],
    target_mask: Tensor,
    probability: float,
    alpha: float,
) -> Tuple[Tuple[Tensor, ...], Tensor, bool]:
    """Paste one rectangle from a shifted batch order into every input and the mask.

    All modalities and the hard segmentation target share the same permutation and
    rectangle, so cross-modal registration is preserved.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("CutMix probability must lie in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("CutMix alpha must be positive")
    first = inputs[0]
    if first.size(0) < 2 or probability == 0.0 or random.random() >= probability:
        return tuple(inputs), target_mask, False
    if any(t.shape[-2:] != target_mask.shape[-2:] for t in inputs):
        raise ValueError("Inputs and target must share the same spatial size for CutMix")

    batch_size = first.size(0)
    height, width = first.shape[-2:]
    lam = float(np.random.beta(alpha, alpha))
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_height = max(1, int(round(height * cut_ratio)))
    cut_width = max(1, int(round(width * cut_ratio)))
    center_y = random.randrange(height)
    center_x = random.randrange(width)
    y1 = max(0, center_y - cut_height // 2)
    y2 = min(height, center_y + (cut_height + 1) // 2)
    x1 = max(0, center_x - cut_width // 2)
    x2 = min(width, center_x + (cut_width + 1) // 2)

    # A non-zero circular shift guarantees that no sample is paired with itself.
    shift = random.randint(1, batch_size - 1)
    permutation = torch.roll(torch.arange(batch_size, device=first.device), shifts=shift)
    mixed = []
    for tensor in (*inputs, target_mask):
        source = tensor.index_select(0, permutation)
        tensor = tensor.clone()
        tensor[..., y1:y2, x1:x2] = source[..., y1:y2, x1:x2]
        mixed.append(tensor)
    return tuple(mixed[:-1]), mixed[-1], True


class AdaptiveCutMixController:
    """Raise / lower next epoch's CutMix probability when validation mIoU rises / falls."""

    def __init__(self, min_probability: float, max_probability: float,
                 initial_probability: float, probability_step: float, miou_delta: float) -> None:
        if not 0.0 <= min_probability <= initial_probability <= max_probability <= 1.0:
            raise ValueError("CutMix probabilities must satisfy 0 <= min <= initial <= max <= 1")
        if probability_step <= 0.0:
            raise ValueError("CutMix probability_step must be positive")
        if miou_delta < 0.0:
            raise ValueError("CutMix miou_delta cannot be negative")
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.probability = initial_probability
        self.probability_step = probability_step
        self.miou_delta = miou_delta
        self.previous_miou: Optional[float] = None

    def update(self, validation_miou: float) -> Dict[str, object]:
        old = self.probability
        delta: Optional[float] = None
        action = "initial"
        if self.previous_miou is not None:
            delta = validation_miou - self.previous_miou
            if delta >= self.miou_delta:
                self.probability = min(self.max_probability, self.probability + self.probability_step)
                action = "increase" if self.probability > old else "at_max"
            elif delta <= -self.miou_delta:
                self.probability = max(self.min_probability, self.probability - self.probability_step)
                action = "decrease" if self.probability < old else "at_min"
            else:
                action = "hold"
        self.previous_miou = validation_miou
        return {"old_probability": old, "new_probability": self.probability,
                "miou_delta": delta, "action": action}


# ============================================================
# Training and validation
# ============================================================


def train_one_epoch(model, dataloader, optimizer, device, modality, accum_steps, max_grad_norm,
                    criterion, scaler, amp_enabled, cutmix_probability, cutmix_alpha,
                    epoch_idx) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    cutmix_batches = 0
    total_steps = len(dataloader)
    final_group_size = total_steps % accum_steps

    progress = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1} [Train]", unit="batch")
    for step, batch in enumerate(progress, start=1):
        inputs = model_inputs(batch, modality, device)
        target_mask = batch["mask"].to(device, non_blocking=True)
        inputs, target_mask, applied = synchronized_cutmix(
            inputs, target_mask, probability=cutmix_probability, alpha=cutmix_alpha)
        cutmix_batches += int(applied)

        with autocast(enabled=amp_enabled):
            logits = model(*inputs)
            loss, _ = criterion(logits, target_mask)
            # The last, possibly shorter accumulation group is averaged over its own size.
            in_final_partial_group = final_group_size > 0 and step > total_steps - final_group_size
            divisor = final_group_size if in_final_partial_group else accum_steps
            backward_loss = loss / divisor

        scaler.scale(backward_loss).backward()
        if step % accum_steps == 0 or step == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        current = float(loss.detach())
        running_loss += current
        progress.set_postfix({"Loss": f"{current:.4f}", "CutMix": f"{cutmix_batches}/{step}"})
    return running_loss / max(1, total_steps)


def evaluate(model, dataloader, device, modality, criterion, amp_enabled):
    model.eval()
    running_loss = 0.0
    samples = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            inputs = model_inputs(batch, modality, device)
            target_mask = batch["mask"].to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                logits = model(*inputs)
                loss, _ = criterion(logits, target_mask)
            n = inputs[0].size(0)
            running_loss += float(loss) * n
            samples += n
            update_confusion(confusion, segmentation_prediction(logits), target_mask[:, 0])
    return running_loss / max(1, samples), metrics_from_confusion(confusion, prefix="val_")


def save_curves(history: Dict[str, List[float]], out_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="val loss")
    axes[1].plot(history["val_mIoU"], label="val mIoU")
    axes[1].plot(history["val_F1"], label="val F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, default="full",
                        help="model variant (see vepmdtnet/presets.py)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="folder with train/ and val/ (and dem_stats.json)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", default=None, help="run name (default: the preset)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr-factor", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cutmix", action="store_true")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-range", type=float, nargs=2, default=(0.10, 0.40))
    parser.add_argument("--cutmix-initial", type=float, default=0.20)
    parser.add_argument("--cutmix-step", type=float, default=0.02)
    parser.add_argument("--cutmix-miou-delta", type=float, default=0.002)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="do not initialise the ResNet-50 encoders with ImageNet weights")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    amp_enabled = device.type == "cuda" and not args.no_amp
    input_size = tuple(args.input_size)

    model_config, modality = make_config(args.preset, pretrained=not args.no_pretrained,
                                         input_size=input_size)
    run_dir = args.out_dir / (args.name or args.preset)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.txt"
    stamp = datetime.now().isoformat(timespec="seconds")
    log_print(f"\n===== {stamp} | preset={args.preset} | modality={modality}", log_file)
    log_print(f"Arguments: {vars(args)}", log_file)
    log_print(f"Model config: {asdict(model_config)}", log_file)

    terrain_statistics = (
        load_or_compute_terrain_statistics(args.data_root) if modality != "rgb" else None
    )
    dataset_options = dict(dataset_root=args.data_root, terrain_statistics=terrain_statistics,
                           modality=modality, input_size=input_size,
                           image_channels=model_config.image_channels, image_scale=255.0)
    train_dataset = TerraceDataset(split="train", augment=True, **dataset_options)
    val_dataset = TerraceDataset(split="val", augment=False, **dataset_options)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)
    if len(train_loader) == 0:
        raise ValueError("Fewer training tiles than one batch; lower --batch-size")
    print(f"Train / val tiles: {len(train_dataset)} / {len(val_dataset)}; "
          f"effective batch {args.batch_size * args.accum_steps}")
    print(f"Preset '{args.preset}': {preset_switches(args.preset)}")

    model = build(model_config, modality).to(device)
    input_channels = {"rgbdem": (3, 1), "rgb": (3,), "dem": (1,)}[modality]
    complexity = profile_model_complexity(model, input_channels, input_size, device)
    message = (f"Params={complexity['total_parameters'] / 1e6:.3f} M, "
               f"GFLOPs={complexity['estimated_gflops']:.3f} (batch 1, {input_size[0]}x{input_size[1]})")
    print(message)
    log_print(message, log_file)

    criterion = MainSegmentationLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = WarmupCosineAnnealingV2(optimizer, args.warmup_epochs, args.epochs, args.min_lr_factor)
    scaler = GradScaler(enabled=amp_enabled)
    use_cutmix = not args.no_cutmix
    cutmix = AdaptiveCutMixController(args.cutmix_range[0], args.cutmix_range[1],
                                      args.cutmix_initial, args.cutmix_step, args.cutmix_miou_delta)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": [], "cutmix_p": []}
    best_val_miou = -float("inf")
    best_path = run_dir / "best.pt"
    start = time.perf_counter()
    for epoch in range(args.epochs):
        lr = scheduler.step(epoch)
        cutmix_probability = cutmix.probability if use_cutmix else 0.0
        train_loss = train_one_epoch(model, train_loader, optimizer, device, modality,
                                     args.accum_steps, args.max_grad_norm, criterion, scaler,
                                     amp_enabled, cutmix_probability, args.cutmix_alpha, epoch)
        val_loss, val_metrics = evaluate(model, val_loader, device, modality, criterion, amp_enabled)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        history["cutmix_p"].append(cutmix_probability)
        for key, value in val_metrics.items():
            history.setdefault(key, []).append(value)
        line = (f"[Epoch {epoch + 1:03d}] LR={lr:.8f} TrainLoss={train_loss:.4f} "
                f"ValLoss={val_loss:.4f} CutMixP={cutmix_probability:.2f} "
                f"mIoU={val_metrics['val_mIoU']:.4f} F1={val_metrics['val_F1']:.4f} "
                f"FgIoU={val_metrics['val_fg_IoU']:.4f} OA={val_metrics['val_OA']:.4f}")
        tqdm.write(line)
        log_print(line, log_file)
        update = cutmix.update(val_metrics["val_mIoU"])
        if use_cutmix:
            log_print(f"  CutMix {update['action']}: {update['old_probability']:.2f}"
                      f"->{update['new_probability']:.2f}", log_file)

        if val_metrics["val_mIoU"] > best_val_miou:
            best_val_miou = val_metrics["val_mIoU"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "model_config": asdict(model_config),
                "input_modality": modality,
                "preset": args.preset,
                "experiment_config": preset_switches(args.preset),
                "terrain_statistics": terrain_statistics,
                "dataset_root": str(args.data_root),
                "selection_metric": "val_mIoU",
                "best_val_miou": best_val_miou,
                "val_loss": val_loss,
                "metrics": val_metrics,
                "model_complexity": complexity,
                "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "elapsed_training_seconds_at_checkpoint": time.perf_counter() - start,
            }, best_path)
            message = f"  saved {best_path} (val mIoU={best_val_miou:.6f})"
            tqdm.write(message)
            log_print(message, log_file)

    elapsed = time.perf_counter() - start
    message = f"Training time {format_duration(elapsed)}; best val mIoU={best_val_miou:.6f}"
    print(message)
    log_print(message, log_file)
    (run_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    save_curves(history, run_dir, f"{args.preset} ({modality})")


if __name__ == "__main__":
    main()
