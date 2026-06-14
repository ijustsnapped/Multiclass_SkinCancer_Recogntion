"""Translate a composed Hydra config into the dict the legacy trainers expect.

The ``conf/`` groups are authored (via ``# @package`` directives) to compose into
the *same* schema as the hand-written ``configs/*.yaml`` files, so the bridge is
thin: resolve interpolations, drop to a plain dict, and run the project's own
``cast_config_values`` so numeric strings / lists are coerced exactly as the
legacy ``main()`` entrypoints do.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.utils.general_utils import cast_config_values


def to_legacy_cfg(cfg: DictConfig) -> dict:
    """DictConfig -> plain dict in the legacy ``train_one_fold`` schema."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return cast_config_values(container)
