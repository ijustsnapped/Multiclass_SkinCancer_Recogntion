"""Optuna (TPE) hyperparameter optimization driven from a W&B sweep YAML.

Runs ``study.optimize()`` in-process: each trial composes the Hydra config with
sampled overrides, runs one fold via ``src.train.run``, and returns the fold's
best selection-metric. The search space is read from the same
``conf/sweep/**/*.yaml`` files used by the W&B Bayesian sweeps, so both engines
share one definition.

Pruning is intentionally omitted: the legacy training loop is a black box that
only returns its final best metric (no per-epoch callback), so there is no signal
to prune on mid-trial. TPE still focuses sampling on promising regions.

Usage:
    python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60
    python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60 \
        --study-name b0_adamw --storage sqlite:///hpo/b0_adamw.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import yaml
from hydra import compose, initialize
from optuna.samplers import TPESampler

from src.train import run as train_run
from src.wandb_ext.setup import init_wandb  # noqa: F401  (kept for parity / discoverability)

HPO_PROJECT = "isic2019-hpo"


def _fmt(value):
    """Render a Python value as a Hydra command-line override token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def load_search_space(sweep_yaml):
    """Split a W&B sweep YAML ``parameters`` block into (fixed, sampled).

    fixed:   list of ``"key=value"`` Hydra overrides (single-value params)
    sampled: list of ``(key, spec)`` for parameters Optuna should suggest
    """
    with open(sweep_yaml) as f:
        spec = yaml.safe_load(f)
    params = spec.get("parameters", {})
    fixed, sampled = [], []
    for key, pspec in params.items():
        if "values" in pspec and len(pspec["values"]) == 1:
            fixed.append(f"{key}={_fmt(pspec['values'][0])}")
        else:
            sampled.append((key, pspec))
    return fixed, sampled


def _suggest_override(trial, key, spec):
    """Translate one W&B parameter spec into an Optuna suggestion -> override."""
    if "values" in spec:
        value = trial.suggest_categorical(key, spec["values"])
    else:
        dist = spec.get("distribution", "uniform")
        # YAML parses "1e-05" as a string, so coerce bounds to numbers.
        lo, hi = spec["min"], spec["max"]
        if dist == "int_uniform":
            value = trial.suggest_int(key, int(lo), int(hi))
        elif dist == "log_uniform_values":
            value = trial.suggest_float(key, float(lo), float(hi), log=True)
        else:
            value = trial.suggest_float(key, float(lo), float(hi))
    return f"{key}={_fmt(value)}"


def make_objective(fixed_overrides, sampled):
    def objective(trial):
        overrides = list(fixed_overrides)
        overrides += [_suggest_override(trial, key, spec) for key, spec in sampled]
        overrides.append(f"wandb.project={HPO_PROJECT}")

        cfg = compose(config_name="config", overrides=overrides)
        cfg.experiment_setup.experiment_name = f"{trial.study.study_name}-t{trial.number:03d}"
        result = train_run(cfg)
        # run() returns a float (single fold) or dict (CV); reduce to a scalar.
        if isinstance(result, dict):
            vals = [v for v in result.values() if v is not None]
            return sum(vals) / len(vals) if vals else float("-inf")
        return result if result is not None else float("-inf")

    return objective


def _default_study_name(sweep_yaml):
    parts = Path(sweep_yaml).with_suffix("").parts
    return "_".join(parts[-2:]) if len(parts) >= 2 else Path(sweep_yaml).stem


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_yaml", help="W&B sweep YAML defining the search space")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default=None,
                        help="Defaults to a name derived from the sweep YAML path.")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (e.g. sqlite:///hpo/b0.db) for "
                             "resume / parallelism. Default: in-memory.")
    parser.add_argument("--seed", type=int, default=42, help="TPE sampler seed.")
    parser.add_argument("--startup-trials", type=int, default=10,
                        help="Random trials before the TPE model engages.")
    args = parser.parse_args()

    study_name = args.study_name or _default_study_name(args.sweep_yaml)
    fixed, sampled = load_search_space(args.sweep_yaml)

    sampler = TPESampler(n_startup_trials=args.startup_trials, seed=args.seed)
    print(f"Study '{study_name}': {len(fixed)} fixed, {len(sampled)} sampled params")
    print(f"Objective: Val/Sensitivity_macro (maximize) | Sampler: TPE")

    study = optuna.create_study(
        study_name=study_name, direction="maximize", sampler=sampler,
        storage=args.storage, load_if_exists=args.storage is not None,
    )

    # config_path is relative to this file (src/), matching src/train.py's "../conf".
    with initialize(version_base="1.3", config_path="../conf"):
        study.optimize(make_objective(fixed, sampled), n_trials=args.n_trials)

    print(f"\n{'='*60}\n  BEST TRIAL ({study_name})\n{'='*60}")
    best = study.best_trial
    print(f"  value: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
