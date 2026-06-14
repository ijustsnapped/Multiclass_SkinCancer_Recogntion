"""Hydra / W&B / Optuna layer for the ISIC 2019 skin-lesion classifier.

This package wraps the repo's existing PyTorch training loops (``train_single_fold``,
``train_cv``) with a composable Hydra config tree (``conf/``), Weights & Biases
logging, and an Optuna HPO runner. It deliberately does *not* reimplement the
training loop — ``src/train.py`` composes a config and hands the legacy
``train_one_fold`` exactly the nested dict it already understands.
"""
