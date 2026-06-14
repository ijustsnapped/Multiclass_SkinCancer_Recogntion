"""Consistent, low-noise console output + a per-fold metrics CSV.

Both trainers (image and image+meta) share these helpers so their progress bars
and per-epoch summaries look identical, and so each fold gets a plain-text
``metrics.csv`` record that survives even when W&B is disabled.

Design choices (see the logging plan):
- One concise summary **line per validated epoch**, emitted via ``tqdm.write`` so
  active progress bars don't clobber it.
- Progress bars are transient (``leave=False``) with ``dynamic_ncols=True`` so they
  adapt to the terminal instead of wrapping at a fixed width.
- Logging is configured **once**, with a compact ``HH:MM:SS`` format (no full
  date), replacing the duplicate ``basicConfig(force=True)`` calls in the trainers.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

from tqdm import tqdm

_LOGGING_CONFIGURED = False

# Column order for the per-fold metrics.csv.
CSV_FIELDS = [
    "epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc",
    "f1_macro", "auroc_macro", "sensitivity_macro", "lr", "is_best",
]


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single compact stream handler on the root logger (idempotent)."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    # Windows consoles default to cp1252, which chokes on arrows/emoji in log
    # messages ('charmap' codec errors). Force UTF-8 so any message encodes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    _LOGGING_CONFIGURED = True


def epoch_bar(loader, *, fold, epoch, n_epochs, split: str, phase: str | None = None):
    """A consistent tqdm bar: ``[fold] E{epoch}/{n_epochs} {phase} {split}``."""
    phase_part = f" {phase}" if phase else ""
    desc = f"[{fold}] E{epoch}/{n_epochs}{phase_part} {split}"
    return tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)


def _fmt(value, nd=2):
    if value is None:
        return "  - "
    try:
        if value != value:  # NaN
            return " nan"
    except TypeError:
        return "  - "
    return f"{value:.{nd}f}"


def log_epoch(*, fold, epoch, n_epochs, train_loss, train_acc, val_loss, val_acc,
              f1=None, auroc=None, sens=None, lr=None, seconds=None,
              is_best=False, phase: str | None = None) -> None:
    """Emit one concise summary line for a validated epoch (via ``tqdm.write``)."""
    phase_part = f" {phase}" if phase else ""
    parts = [
        f"[{fold}] E{epoch:>2}/{n_epochs}{phase_part}",
        f"train {_fmt(train_loss, 3)}/{_fmt(train_acc)}",
        f"val {_fmt(val_loss, 3)}/{_fmt(val_acc)}",
        f"F1 {_fmt(f1)} AUROC {_fmt(auroc)} Sens {_fmt(sens)}",
    ]
    if lr is not None:
        parts.append(f"lr {lr:.2e}")
    if seconds is not None:
        parts.append(f"{seconds:.0f}s")
    line = " | ".join(parts)
    if is_best:
        line += "  *best"
    tqdm.write(line)


class MetricsCSV:
    """Append-only per-fold metrics record written next to the run's logs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists()

    def append(self, **row) -> None:
        record = {k: row.get(k) for k in CSV_FIELDS}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(record)
