"""Weights & Biases run setup + a transparent TensorBoard->W&B scalar mirror.

The legacy trainers log through a raw ``torch.utils.tensorboard.SummaryWriter``
(``writer.add_scalar(tag, value, step)``). Rather than thread a logger object
through the deep training loop, we monkeypatch ``SummaryWriter.add_scalar`` once
so every scalar is also forwarded to the active W&B run. Scalars are logged
without an explicit step (W&B auto-increments) because the loop mixes per-batch
global steps with per-epoch steps, which would otherwise be non-monotonic.
"""
from __future__ import annotations

import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Validation tags worth a run-level summary (best value over training).
_VAL_MAX_TAGS = (
    "Val/Acc_epoch", "Val/F1_macro", "Val/AUROC_macro", "Val/Sensitivity_macro",
)
_VAL_MIN_TAGS = ("Val/Loss_epoch",)

_PATCHED = False


def _patch_summary_writer(wandb):
    """Make every ``SummaryWriter.add_scalar`` also log to the active W&B run."""
    global _PATCHED
    if _PATCHED:
        return
    original = SummaryWriter.add_scalar

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        original(self, tag, scalar_value, global_step, *args, **kwargs)
        if wandb.run is not None:
            try:
                value = float(scalar_value.item() if hasattr(scalar_value, "item")
                              else scalar_value)
                wandb.log({tag: value})
            except Exception:  # never let logging break training
                pass

    SummaryWriter.add_scalar = add_scalar
    _PATCHED = True


def init_wandb(cfg: dict, run_name: str, group: str | None = None,
               tags=None, extra_config: dict | None = None):
    """Start a W&B run from the bridged config dict, or return None if disabled.

    Returns the ``wandb.run`` (or None). Safe to call when wandb is not installed
    or ``cfg['wandb'].enable`` is false / mode is 'disabled'.
    """
    wb_cfg = cfg.get("wandb", {}) or {}
    if not wb_cfg.get("enable", True) or wb_cfg.get("mode") == "disabled":
        logger.info("W&B logging disabled via config.")
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping W&B logging. `pip install wandb`.")
        return None

    run = wandb.init(
        project=wb_cfg.get("project", "isic2019-skin-cancer"),
        entity=wb_cfg.get("entity"),
        name=run_name,
        group=group if group is not None else wb_cfg.get("group"),
        tags=list(tags) if tags else (wb_cfg.get("tags") or []),
        mode=wb_cfg.get("mode", "online"),
        config={**cfg, **(extra_config or {})},
        reinit=True,
    )

    for tag in _VAL_MAX_TAGS:
        run.define_metric(tag, summary="max")
    for tag in _VAL_MIN_TAGS:
        run.define_metric(tag, summary="min")

    if wb_cfg.get("log_code", True):
        try:
            run.log_code(".", include_fn=lambda p: p.endswith((".py", ".yaml")))
        except Exception as e:
            logger.warning(f"W&B log_code failed: {e}")

    _patch_summary_writer(wandb)
    logger.info(f"W&B run started: {run.name} (project={run.project}).")
    return run


def finish_wandb():
    """Close the active W&B run if any."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
