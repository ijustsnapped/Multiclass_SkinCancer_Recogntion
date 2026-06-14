"""Training loops for single-fold runs, with and without metadata fusion.

The Hydra entrypoint ``src/train.py`` imports ``single_fold`` (image-only) and
``single_fold_meta`` (image + patient metadata) and calls their
``train_one_fold`` / ``train_one_fold_with_meta`` directly. Cross-validation is
just these loops run once per fold (``run.all_folds=true``).
"""
