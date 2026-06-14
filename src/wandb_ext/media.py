"""W&B media logging: data examples, augmentation previews, and GradCAM.

These give a run a visual record of *what the model sees*: a batch after CPU
augmentation, a few raw examples per class, and an augmentation-variety panel
(one image transformed several times). Reuses ``build_transform`` and
``denormalize_image`` so the previews match the real training pipeline exactly.

Everything no-ops when there is no active W&B run, so disabled/offline runs and
unit tests skip it for free.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from src.data.datasets import FlatDataset
from src.data.transforms import build_transform
from src.utils.tb_logger import denormalize_image

logger = logging.getLogger(__name__)

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def _active_wandb():
    try:
        import wandb
    except ImportError:
        return None
    return wandb if wandb.run is not None else None


def _denorm_grid(tensors, cpu_aug, nrow):
    imgs = torch.stack(tensors) if isinstance(tensors, list) else tensors
    imgs = denormalize_image(
        imgs.cpu(),
        cpu_aug.get("norm_mean", _DEFAULT_MEAN),
        cpu_aug.get("norm_std", _DEFAULT_STD),
    )
    return to_pil_image(make_grid(imgs, nrow=nrow))


def log_dataset_media(df, root, label2idx, cpu_aug, *, image_loader="pil",
                      n_aug=16, per_class=3, variety_imgs=4, variety_n=5) -> None:
    """Log augmented samples, per-class examples, and an augmentation-variety grid."""
    wandb = _active_wandb()
    if wandb is None:
        return
    try:
        tf_train = build_transform(cpu_aug, train=True)
        tf_val = build_transform(cpu_aug, train=False)
        idx2label = {v: k for k, v in label2idx.items()}

        # 1) Augmented training samples — what the model actually trains on.
        ds_train = FlatDataset(df=df, root=root, label2idx=label2idx,
                               tf=tf_train, image_loader=image_loader)
        n = min(n_aug, len(ds_train))
        aug = [ds_train[i][0] for i in range(n)]
        wandb.log({"media/train_augmented": wandb.Image(
            _denorm_grid(aug, cpu_aug, nrow=int(np.ceil(np.sqrt(n)))),
            caption="Training batch after CPU augmentation")})

        # 2) A few (deterministic) examples per class.
        ds_val = FlatDataset(df=df, root=root, label2idx=label2idx,
                             tf=tf_val, image_loader=image_loader)
        by_class: dict[int, list[int]] = {}
        for i, (_, lbl) in enumerate(ds_val.samples):
            by_class.setdefault(lbl, []).append(i)
        ex_tensors = [ds_val[i][0] for lbl in sorted(by_class)
                      for i in by_class[lbl][:per_class]]
        if ex_tensors:
            rows = ", ".join(idx2label[l] for l in sorted(by_class))
            wandb.log({"media/class_examples": wandb.Image(
                _denorm_grid(ex_tensors, cpu_aug, nrow=per_class),
                caption=f"{per_class} examples per class (rows: {rows})")})

        # 3) Augmentation variety — the same images transformed several times.
        variety = []
        for path, _ in ds_train.samples[:variety_imgs]:
            img = Image.open(str(path)).convert("RGB")
            variety.extend(tf_train(img) for _ in range(variety_n))
        if variety:
            wandb.log({"media/augmentation_variety": wandb.Image(
                _denorm_grid(variety, cpu_aug, nrow=variety_n),
                caption=f"{variety_imgs} images x {variety_n} augmentations")})

        logger.info("Logged dataset/augmentation media to W&B.")
    except Exception as e:  # visuals are best-effort; never break training
        logger.warning(f"W&B media logging skipped: {e}")
