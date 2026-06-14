# Multiclass Skin Cancer Recognition (ISIC 2019)

8-class dermoscopy lesion classifier for the ISIC 2019 dataset
(AK, BCC, BKL, DF, MEL, NV, SCC, VASC).

Backbones: EfficientNet and DINOv2. Handles class imbalance (LDAM + DRW,
focal loss, class-balanced sampling), optional metadata fusion (age / sex /
site), EMA, AMP, and TensorBoard. Experiments run through Hydra, with
Weights & Biases and Optuna on top.

## Layout

- `conf/` — Hydra configs (dataset, model, optim, loss, sampler, experiment, sweep)
- `src/` — code: `train.py`, `hpo.py`, and the `training/` loops, plus data/models/losses/utils
- `scripts/` — dataset download + prep
- `configs/` — original hand-written YAML configs
- `notebooks/` — EDA

`splits/`, `data/`, `outputs/`, `wandb/`, and `hpo/*.db` are git-ignored.

## 1. Get the dataset

ISIC 2019 is 25,331 training images (~9 GB), licensed CC-BY-NC 4.0.
Downloading means you accept the licenses on the
[ISIC 2019](https://challenge.isic-archive.com/data/) and HAM10000 pages.

```bash
python scripts/download_isic2019.py        # add --with-test for the test set
python scripts/prepare_isic2019.py --folds 5
```

That gives you:

```
data/raw/ISIC_2019_Training_Input/    25,331 JPEGs
splits/training/labels.csv            dataset,filename,label,fold
splits/training/hot_one_meta.csv      image, age_zscore, anatom_site_*, sex_*
```

The `dataset` column in `labels.csv` is a relative path from `splits/training/`
to the image folder, so the images stay where they are (no copying).

Manual download links:
- Images: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip`
- Ground truth: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv`
- Metadata: `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv`

Mirrors: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/isic-2019),
[ISIC Challenge](https://challenge.isic-archive.com/data/).

## 2. Environment

```bash
# conda
conda env create -f environment.yml      # env name: skincancer
conda activate skincancer

# or pip
pip install -r requirements.txt
```

Install a CUDA torch build that matches your GPU, e.g.
`pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124`.

Note: on this machine the env that works is the conda env `DeepLearn`
(`C:\Users\Dan\Anaconda3\envs\DeepLearn`, torch 2.7.0+cu128), with Hydra/W&B/Optuna
already installed.

## 3. Train

```bash
python -m src.train                          # default: B0 + CE + sampler, fold 0
python -m src.train experiment=b3_ldam       # a preset
python -m src.train model=efficientnet_b3 optim=sgd loss=ldam
python -m src.train run.fold_id=2            # validate on fold 2
python -m src.train run.all_folds=true       # cross-validate
python -m src.train wandb.enable=false       # TensorBoard only
```

Hydra composes the nested-dict config that the legacy `train_one_fold` expects,
so no training code is duplicated. W&B logging works by mirroring every
TensorBoard scalar to the active run. Run `wandb login` once first.

You can also run the loops directly:

```bash
python -m src.training.single_fold b0_cross_entropy_w_sampler --fold_id 0
python -m src.training.cv --config_file configs/effnetb3.yaml
python -m src.training.single_fold_meta config_meta_single --fold_id 0
```

## 4. Hyperparameter optimization

The search space lives in `conf/sweep/**/*.yaml` and is shared by both engines.

Optuna (TPE), in-process:

```bash
python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60
python -m src.hpo conf/sweep/efficientnet_b0/adamw.yaml --n-trials 60 \
    --study-name b0_adamw --storage sqlite:///hpo/b0_adamw.db   # resumable
```

W&B sweep:

```bash
wandb sweep conf/sweep/efficientnet_b0/adamw.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Each Optuna trial runs one full fold and returns its best `Val/Sensitivity_macro`.
There's no mid-trial pruning because the legacy loop only reports a final metric.

## Notes
- The 8 classes come from `argmax` over the ground-truth one-hot columns;
  `label2idx` is built from `sorted(unique(label))`.
- Metadata one-hot columns match the `meta_features_names` blocks in
  `configs/*.yaml`, used by the `*_with_meta` trainers.
- The dataset is CC-BY-NC 4.0 (non-commercial).
