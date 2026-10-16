from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def foreground_logits(logits: Tensor) -> Tensor:
    if logits.size(1) == 1:
        return logits
    if logits.size(1) == 2:
        return logits[:, 1:2] - logits[:, 0:1]
    raise ValueError("Binary terrace segmentation expects one or two output channels")


def dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits).flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)).mean()


class MainSegmentationLoss(nn.Module):
    """Single final segmentation loss: 0.3 BCE + 0.7 Dice (paper Eq. loss)."""

    def __init__(self, bce_weight: float = 0.3, dice_weight: float = 0.7) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self, segmentation_logits: Tensor, target_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        logits = foreground_logits(segmentation_logits)
        target = target_mask.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = dice_loss(logits, target)
        total = self.bce_weight * bce + self.dice_weight * dice
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(
                f"Non-finite main segmentation loss: bce={float(bce.detach())}, "
                f"dice={float(dice.detach())}"
            )
        return total, {
            "loss_total": total.detach(),
            "loss_seg": total.detach(),
            "loss_bce": bce.detach(),
            "loss_dice": dice.detach(),
        }

