"""Legacy training loops (single-fold / CV, with and without metadata fusion).

Run as modules, e.g.:
    python -m src.training.single_fold b0_cross_entropy_w_sampler --fold_id 0
    python -m src.training.cv --config_file configs/effnetb3.yaml

The Hydra entrypoint ``src/train.py`` imports ``single_fold`` and calls its
``train_one_fold`` directly.
"""
