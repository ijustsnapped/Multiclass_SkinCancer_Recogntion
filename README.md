# Multiclass Skin Cancer Recognition (ISIC 2019)

Deep-learning pipeline for 8-class dermoscopy lesion classification on the
**ISIC 2019** challenge dataset (AK, BCC, BKL, DF, MEL, NV, SCC, VASC). It supports
EfficientNet / DINOv2 backbones, class-imbalance handling (LDAM + DRW, focal loss,
class-balanced sampling), optional metadata fusion (age / sex / anatomical site),
EMA, AMP, TensorBoard, and a Hydra + Weights & Biases + Optuna experiment layer.

---

## Repository layout

```
conf/                 Hydra config tree (compose experiments from groups)
  config.yaml           root defaults + run control
  dataset/              data paths + augmentations (isic2019)
  model/                backbones (efficientnet_b0/b3, dino_vit_s14)
  optim/                optimizer + scheduler (adamw, sgd)
  loss/                 cross_entropy, focal, ldam(+DRW)
  sampler/              default, class_balanced_sqrt
  wandb/                W&B logging settings
  experiment/           full presets (b0_ce_sampler, b3_ldam)
  sweep/                W&B / Optuna search spaces
src/                  single source tree
  train.py              Hydra entrypoint  ->  training.single_fold.train_one_fold
  hpo.py                Optuna (TPE) runner over conf/sweep/*.yaml
  config_bridge.py      composed Hydra cfg -> legacy nested-dict schema
  wandb_ext/            W&B init + TensorBoard->W&B scalar mirror
  data/                 datasets, transforms, samplers
  models/               model factory + metadata-fusion heads
  losses/               focal + LDAM
  utils/                EMA, TensorBoard logger, stats, cropping, helpers
  training/             the underlying training loops (run as modules)
    single_fold.py        single-fold trainer
    cv.py                 k-fold trainer
    single_fold_meta.py   single-fold + metadata fusion
    cv_meta.py            k-fold + metadata fusion
scripts/              dataset acquisition + preparation
  download_isic2019.py  fetch official images / ground truth / metadata
  prepare_isic2019.py   build splits/training/labels.csv + hot_one_meta.csv
configs/              original hand-written YAML configs (training/* CLIs)
```

`splits/`, `data/`, `outputs/`, `wandb/`, and `hpo/*.db` are git-ignored.

---

## 1. Get the dataset

The ISIC 2019 dataset (25,331 training images, ~9 GB) is **CC-BY-NC 4.0**. By
downloading you accept the licenses on the
[ISIC 2019](https://challenge.isic-archive.com/data/) and HAM10000 dataset pages.

```bash
# Download images + ground truth + metadata into data/raw/
python scripts/download_isic2019.py            # add --with-test for the test set

# Build the labels.csv / one-hot metadata / CV folds the trainers expect
python scripts/prepare_isic2019.py --folds 5
```

This produces:

```
data/raw/ISIC_2019_Training_Input/        25,331 JPEGs
splits/training/labels.csv                dataset,filename,label,fold
splits/training/hot_one_meta.csv          image, age_zscore, anatom_site_*, sex_*
```

`labels.csv`'s `dataset` column is a relative path from `splits/training/` to the
image folder, so the 9 GB of images are referenced in place (not copied).

**Direct download links** (if you prefer manual download):
- Images: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip`
- Ground truth: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv`
- Metadata: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv`

Mirrors: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/isic-2019),
[ISIC Challenge](https://challenge.isic-archive.com/data/).

---

## 2. Environment

```bash
# conda (recommended)
conda env create -f environment.yml      # creates env "skincancer"
conda activate skincancer

# or pip into an existing env
pip install -r requirements.txt
```

Install a CUDA build of torch matching your GPU, e.g.
`pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124`.

> On this machine the working env is the conda env **`DeepLearn`**
> (`C:\Users\Dan\Anaconda3\envs\DeepLearn`, torch 2.7.0+cu128). The Hydra/W&B/Optuna
> packages are already installed there. A WSL setup is also possible once
> virtualization is enabled in BIOS (`wsl.exe --install`), after which run the same
> `pip install -r requirements.txt` inside the distro.

---

## 3. Train (Hydra + W&B)

```bash
python -m src.train                                # default: B0 + CE + sampler, fold 0
python -m src.train experiment=b3_ldam             # a full preset
python -m src.train model=efficientnet_b3 optim=sgd loss=ldam
python -m src.train run.fold_id=2                  # validate on fold 2
python -m src.train run.all_folds=true             # cross-validate all folds
python -m src.train wandb.enable=false             # disable W&B (TensorBoard only)
```

Composition produces exactly the nested-dict config the legacy `train_one_fold`
consumes, so no training logic is duplicated. W&B logging is layered on by
mirroring every TensorBoard scalar (`Val/Sensitivity_macro`, `Val/F1_macro`, ...)
to the active run; run `wandb login` once first.

The underlying loops can still be run directly as modules:

```bash
python -m src.training.single_fold b0_cross_entropy_w_sampler --fold_id 0
python -m src.training.cv --config_file configs/effnetb3.yaml
python -m src.training.single_fold_meta config_meta_single --fold_id 0   # metadata fusion
```

---

## 4. Hyperparameter optimization

The search space lives in `conf/sweep/**/*.yaml` and is shared by two engines.

**Optuna (TPE), in-process — no W&B agent needed:**

```bash
python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60
python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60 \
    --study-name b0_adamw --storage sqlite:///hpo/b0_adamw.db   # resumable
```

**W&B Bayesian sweep:**

```bash
wandb sweep conf/sweep/efficientnet_b0/adamw.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Each Optuna trial runs one full fold and returns its best `Val/Sensitivity_macro`.
(Mid-trial pruning is omitted because the legacy loop only exposes a final metric.)

---

## Notes
- 8 classes are derived by `argmax` over the ground-truth one-hot columns; the
  trainer builds `label2idx` from `sorted(unique(label))`.
- Metadata one-hot columns are produced to match the `meta_features_names` blocks
  in `configs/*.yaml` (the `*_with_meta` trainers consume them).
- License: dataset is CC-BY-NC 4.0 (non-commercial). Respect it.
