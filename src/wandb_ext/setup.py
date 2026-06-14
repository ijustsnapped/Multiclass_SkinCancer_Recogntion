"""Weights & Biases run setup.

The legacy trainers log through a ``SummaryWriter``-shaped object; we hand them a
``WandbWriter`` (see ``src/wandb_ext/writer.py``) that forwards scalars/media to
the active W&B run instead of writing TensorBoard event files. This module owns
run creation, the metric schema (a shared ``epoch`` step axis), and the run-level
"best value" summaries.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Validation tags worth a run-level summary (best value over training). These match
# the unified lowercase schema emitted by both trainers.
_VAL_MAX_TAGS = (
    "val/acc", "val/f1_macro", "val/auroc_macro", "val/sensitivity_macro",
)
_VAL_MIN_TAGS = ("val/loss",)


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

    # All epoch scalars share a clean, monotonic x-axis.
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    for tag in _VAL_MAX_TAGS:
        run.define_metric(tag, step_metric="epoch", summary="max")
    for tag in _VAL_MIN_TAGS:
        run.define_metric(tag, step_metric="epoch", summary="min")

    if wb_cfg.get("log_code", True):
        try:
            run.log_code(".", include_fn=lambda p: p.endswith((".py", ".yaml")))
        except Exception as e:
            logger.warning(f"W&B log_code failed: {e}")

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
