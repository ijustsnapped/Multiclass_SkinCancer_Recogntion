"""Hydra entrypoint that drives the legacy single-fold trainer.

Usage:
    python -m src.train                                  # default config
    python -m src.train experiment=b3_ldam               # a preset
    python -m src.train model=efficientnet_b3 optim=sgd loss=ldam
    python -m src.train run.fold_id=2                    # validate on fold 2
    python -m src.train run.all_folds=true               # cross-validate all folds
    python -m src.train wandb.enable=false               # no W&B

Composition (conf/) yields the exact nested dict the legacy ``train_one_fold``
consumes; W&B logging is layered on by mirroring TensorBoard scalars.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Legacy trainers: reuse their fold loops + path helper verbatim.
from src.training import single_fold as legacy
from src.training import single_fold_meta as meta_trainer
from src.config_bridge import to_legacy_cfg
from src.wandb_ext import init_wandb, finish_wandb
from src.utils.general_utils import set_seed
from src.utils.torch_utils import get_device

logger = logging.getLogger(__name__)


def _experiment_name(cfg: dict) -> str:
    exp = cfg.get("experiment_setup", {}).get("experiment_name")
    if exp:
        return str(exp)
    model_type = cfg.get("model", {}).get("type", "model")
    loss_type = cfg.get("training", {}).get("loss", {}).get("type", "loss")
    return f"{model_type}_{loss_type}"


def _run_fold(cfg: dict, fold_id, exp_name: str, base_path: Path, seed: int,
              wandb_group: str | None) -> float | None:
    """Replicates the legacy single-fold orchestration, routed by ``training_mode``
    (image-only -> single_fold; image_meta -> single_fold_meta)."""
    set_seed(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_str = str(fold_id).replace(" ", "_").replace("/", "-")
    training_mode = cfg.get("training_mode", "image")

    log_dir = (legacy._get_path_from_config(cfg, "log_dir", "outputs/tensorboard", base_path)
               / exp_name / f"fold_{fold_str}" / timestamp)
    ckpt_dir = (legacy._get_path_from_config(cfg, "ckpt_dir", "outputs/checkpoints", base_path)
                / exp_name / f"fold_{fold_str}" / timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = legacy._get_path_from_config(cfg, "labels_csv", base_path=base_path)
    train_root = legacy._get_path_from_config(cfg, "train_root", base_path=base_path)
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}. "
                                "Run scripts/download_isic2019.py + prepare_isic2019.py.")
    if not train_root.is_dir():
        raise FileNotFoundError(f"Train root invalid: {train_root}")

    df = pd.read_csv(labels_csv)
    label2idx = {lbl: i for i, lbl in enumerate(sorted(df["label"].unique()))}
    device = get_device(cfg.get("experiment_setup", {}).get("device"))

    fold_val = fold_id
    if pd.api.types.is_numeric_dtype(df["fold"].dtype):
        try:
            fold_val = int(fold_id)
        except (TypeError, ValueError):
            pass
    if fold_val not in df["fold"].unique():
        raise ValueError(f"Fold '{fold_val}' not in CSV. Available: {df['fold'].unique().tolist()}")

    train_df = df[df["fold"] != fold_val].reset_index(drop=True)
    val_df = df[df["fold"] == fold_val].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Fold {fold_val}: empty train ({len(train_df)}) or val ({len(val_df)}).")
    logger.info(f"Fold {fold_val} [{training_mode}]: {len(train_df)} train / {len(val_df)} val samples.")

    run = init_wandb(cfg, run_name=f"{exp_name}_fold{fold_str}", group=wandb_group,
                     extra_config={"fold_id": fold_val, "seed": seed, "training_mode": training_mode})
    try:
        if training_mode == "image_meta":
            meta_csv = legacy._get_path_from_config(cfg, "meta_csv", base_path=base_path)
            if not meta_csv.exists():
                raise FileNotFoundError(f"Meta CSV not found: {meta_csv}. "
                                        "Run scripts/prepare_isic2019.py to build it.")
            meta_df = pd.read_csv(meta_csv)
            return meta_trainer.train_one_fold_with_meta(
                fold_id=fold_id, train_df=train_df, val_df=val_df, meta_df=meta_df,
                cfg=cfg, label2idx_model=label2idx, label2idx_eval=label2idx,
                image_root=train_root, log_dir=log_dir, ckpt_dir=ckpt_dir,
                exp_name=exp_name, device=device,
            )
        return legacy.train_one_fold(
            fold_id=fold_id, train_df=train_df, val_df=val_df, cfg=cfg,
            label2idx=label2idx, train_root=train_root, log_dir=log_dir,
            ckpt_dir=ckpt_dir, exp_name=exp_name, device=device,
        )
    finally:
        finish_wandb()


def run(cfg: DictConfig):
    """Core logic; callable directly (e.g. from src/hpo.py)."""
    cfg_dict = to_legacy_cfg(cfg)
    run_cfg = cfg_dict.get("run", {})
    seed = run_cfg.get("seed") or cfg_dict.get("experiment_setup", {}).get("seed", 42)
    exp_name = _experiment_name(cfg_dict)
    base_path = Path.cwd()

    if not run_cfg.get("all_folds", False):
        return _run_fold(cfg_dict, run_cfg.get("fold_id", 0), exp_name, base_path,
                         seed, wandb_group=None)

    # Cross-validation: one W&B run per fold, grouped under the experiment name.
    labels_csv = legacy._get_path_from_config(cfg_dict, "labels_csv", base_path=base_path)
    folds = sorted(pd.read_csv(labels_csv)["fold"].unique().tolist())
    logger.info(f"Cross-validating over {len(folds)} folds: {folds}")
    results = {}
    for fold in folds:
        logger.info(f"\n{'='*60}\n  FOLD {fold}  ({exp_name})\n{'='*60}")
        results[fold] = _run_fold(cfg_dict, fold, exp_name, base_path, seed,
                                  wandb_group=exp_name)
    valid = [v for v in results.values() if v is not None]
    if valid:
        logger.info(f"CV mean best-metric over {len(valid)} folds: {sum(valid)/len(valid):.4f}")
    return results


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
