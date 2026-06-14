"""A tiny ``SummaryWriter``-shaped object that logs straight to Weights & Biases.

The legacy trainers were written against ``torch.utils.tensorboard.SummaryWriter``
(``writer.add_scalar(tag, value, step)`` etc.). Rather than rip out every call
site, we hand them a ``WandbWriter`` that exposes the same surface but forwards to
``wandb.log`` instead of writing TensorBoard event files. This drops the
TensorBoard dependency while keeping the call sites intact.

Scalars are logged against an explicit ``epoch`` step so train/val curves share a
clean, monotonic x-axis (see ``init_wandb`` which calls ``define_metric``). Tags
ending in ``_batch`` are intentionally *not* forwarded — per-batch detail lives in
the tqdm progress bar only (see the cadence decision in the logging plan).

Everything no-ops when there is no active W&B run, so ``wandb.enable=false`` runs
(and unit tests) just skip logging.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _wandb():
    """Return the wandb module if installed and a run is active, else None."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb if wandb.run is not None else None


class WandbWriter:
    """Drop-in replacement for the subset of ``SummaryWriter`` the trainers use."""

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        # Per-batch scalars stay in the terminal only; don't clutter W&B.
        if tag.endswith("_batch") or "_batch_" in tag:
            return
        wandb = _wandb()
        if wandb is None:
            return
        try:
            value = float(scalar_value.item() if hasattr(scalar_value, "item")
                          else scalar_value)
        except (TypeError, ValueError):
            return
        payload = {tag: value}
        if global_step is not None:
            payload["epoch"] = int(global_step)
        try:
            wandb.log(payload)
        except Exception:  # never let logging break training
            pass

    def add_image(self, tag, img_tensor, global_step=None, *args, **kwargs):
        self._log_images(tag, img_tensor, global_step)

    def add_images(self, tag, img_tensor, global_step=None, *args, **kwargs):
        self._log_images(tag, img_tensor, global_step)

    def _log_images(self, tag, img_tensor, global_step):
        wandb = _wandb()
        if wandb is None:
            return
        try:
            payload = {tag: wandb.Image(img_tensor)}
            if global_step is not None:
                payload["epoch"] = int(global_step)
            wandb.log(payload)
        except Exception:
            pass

    def add_text(self, tag, text_string, global_step=None, *args, **kwargs):
        wandb = _wandb()
        if wandb is None:
            return
        try:
            payload = {tag: str(text_string)}
            if global_step is not None:
                payload["epoch"] = int(global_step)
            wandb.log(payload)
        except Exception:
            pass

    # SummaryWriter lifecycle methods the trainers call — nothing to flush/close.
    def flush(self):
        pass

    def close(self):
        pass


def make_writer() -> WandbWriter:
    """Return a writer that mirrors scalars/media to the active W&B run (or no-ops)."""
    return WandbWriter()
