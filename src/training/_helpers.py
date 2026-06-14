"""Shared building blocks for the single-fold trainers.

``single_fold`` (image) and ``single_fold_meta`` (image + metadata) were almost
entirely copy-pasted. These helpers hold the parts that were genuinely identical
— validation metrics and optimizer/scheduler construction — so there's one
definition to fix and the trainers shrink to their actually-different logic
(metadata batching, the two-phase meta schedule).
"""
from __future__ import annotations

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import AUROC, F1Score, Recall


def compute_val_metrics(all_probs: torch.Tensor, all_true: torch.Tensor,
                        num_classes: int) -> dict[str, float]:
    """Macro F1 / AUROC / sensitivity from accumulated probabilities + labels.

    AUROC is wrapped because it raises when a class is absent from a val fold.
    """
    preds = all_probs.argmax(dim=1)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")(
        preds, all_true).item()
    try:
        auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")(
            all_probs, all_true).item()
    except Exception:
        auroc = float("nan")
    sens = Recall(task="multiclass", num_classes=num_classes, average="macro",
                  zero_division=0)(preds, all_true).item()
    return {"f1_macro": f1, "auroc_macro": auroc, "sensitivity_macro": sens}


def build_optimizer(params, optim_cfg: dict, base_lr: float):
    """AdamW (default) or SGD from an ``optimizer`` config block."""
    wd = optim_cfg.get("weight_decay", 1e-4)
    if optim_cfg.get("type", "AdamW").lower() == "sgd":
        return SGD(params, lr=base_lr, weight_decay=wd,
                   momentum=optim_cfg.get("momentum", 0.9))
    return AdamW(params, lr=base_lr, weight_decay=wd)


def build_scheduler(optimizer, sched_cfg: dict, t_max: int):
    """StepLR or CosineAnnealingLR (default) from a ``scheduler`` config block."""
    if sched_cfg.get("type", "cosineannealinglr").lower() == "steplr":
        return StepLR(optimizer, step_size=sched_cfg.get("step_size", 10),
                      gamma=sched_cfg.get("gamma", 0.1))
    return CosineAnnealingLR(optimizer, T_max=(t_max if t_max > 0 else 1),
                             eta_min=sched_cfg.get("min_lr", 0.0))
