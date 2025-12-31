from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    mode: str = "inv_sqrt",
    beta: float = 0.999,
) -> torch.Tensor:
    """
    mode:
        - inv: 1 / freq
        - inv_sqrt: 1 / sqrt(freq) (gentler)
        - effective: class-balanced weight using effective number, beta in (0,1)
    """
    counts = torch.bincount(targets, minlength=num_classes).float().clamp_min(1.0)
    if mode == "inv":
        weights = 1.0 / counts
    elif mode == "inv_sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "effective":
        # w_c = (1 - beta) / (1 - beta^n_c)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    weights = weights * (num_classes / weights.sum())  # normalize mean=1
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_factor = torch.pow(1.0 - probs, self.gamma)
        loss = F.nll_loss(focal_factor * log_probs, targets, weight=self.weight, reduction=self.reduction)
        return loss


def get_loss_fn(name: str, class_weights: Optional[torch.Tensor] = None, gamma: float = 2.0) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "wce":
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted CE.")
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=class_weights)
    raise ValueError(f"Unsupported loss: {name}")
