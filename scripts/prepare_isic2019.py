#!/usr/bin/env python3
"""
prepare_isic2019.py

Turn the raw ISIC 2019 download into the layout the training scripts expect:

    splits/training/labels.csv          # dataset,filename,label,fold
    splits/training/hot_one_meta.csv    # image,age_zscore,anatom_site_general_*,sex_*

The ``dataset`` column is the directory (relative to ``train_root``) that holds the
images, so ``train_root / dataset / filename`` resolves to each JPEG. By default we
point it at the raw image folder via a relative path so the 9 GB of images are not
copied or moved.

Inputs (produced by scripts/download_isic2019.py):
    data/raw/ISIC_2019_Training_Input/            images
    data/raw/ISIC_2019_Training_GroundTruth.csv   one-hot diagnoses
    data/raw/ISIC_2019_Training_Metadata.csv      age / sex / anatom_site

Usage:
    python scripts/prepare_isic2019.py
    python scripts/prepare_isic2019.py --raw data/raw --out splits/training --folds 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ISIC 2019 diagnosis columns in the ground-truth CSV (UNK is not present in train).
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Anatomical sites present in the ISIC 2019 metadata; fixed order so the one-hot
# columns always match configs/*.yaml `meta_features_names`.
ANATOM_SITES = [
    "anterior torso",
    "head/neck",
    "lateral torso",
    "lower extremity",
    "oral/genital",
    "palms/soles",
    "posterior torso",
    "upper extremity",
]
SEXES = ["female", "male"]


def build_labels(gt_csv: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_csv)
    present = [c for c in CLASS_COLS if c in gt.columns]
    if len(present) != len(CLASS_COLS):
        raise ValueError(
            f"Ground-truth CSV missing expected class columns. "
            f"Found {list(gt.columns)}"
        )
    # One-hot -> single label string via argmax over the class columns.
    gt["label"] = gt[CLASS_COLS].values.argmax(axis=1)
    gt["label"] = gt["label"].map({i: c for i, c in enumerate(CLASS_COLS)})
    return gt[["image", "label"]].copy()


def add_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_train, val) in enumerate(skf.split(df["image"], df["label"])):
        df.loc[df.index[val], "fold"] = fold_idx
    assert (df["fold"] >= 0).all(), "Some rows were not assigned a fold."
    return df


def build_metadata(meta_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    out = pd.DataFrame({"image": meta["image"]})

    # --- age -> z-score (NaN preserved; the dataset fills it at load time) ---
    age = pd.to_numeric(meta.get("age_approx"), errors="coerce")
    mean, std = age.mean(), age.std()
    out["age_zscore"] = (age - mean) / (std if std and not np.isnan(std) else 1.0)

    # --- anatomical site one-hot + explicit NaN indicator ---
    site = meta.get("anatom_site_general")
    for s in ANATOM_SITES:
        out[f"anatom_site_general_{s}"] = (site == s).astype(float)
    out["anatom_site_general_nan"] = site.isna().astype(float)

    # --- sex one-hot + explicit NaN indicator ---
    sex = meta.get("sex")
    for s in SEXES:
        out[f"sex_{s}"] = (sex == s).astype(float)
    out["sex_nan"] = sex.isna().astype(float)

    return out


def relative_image_dir(images_dir: Path, out_dir: Path) -> str:
    """Path from the labels.csv directory to the image directory (POSIX style)."""
    rel = os.path.relpath(images_dir.resolve(), out_dir.resolve())
    return Path(rel).as_posix()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ISIC 2019 splits/labels for training.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"),
                    help="Directory containing the raw ISIC 2019 download.")
    ap.add_argument("--out", type=Path, default=Path("splits/training"),
                    help="Where to write labels.csv and hot_one_meta.csv.")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="Override image directory (default: <raw>/ISIC_2019_Training_Input).")
    ap.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for fold assignment.")
    args = ap.parse_args()

    raw: Path = args.raw
    out: Path = args.out
    images_dir = args.images_dir or (raw / "ISIC_2019_Training_Input")
    gt_csv = raw / "ISIC_2019_Training_GroundTruth.csv"
    meta_csv = raw / "ISIC_2019_Training_Metadata.csv"

    for p in (images_dir, gt_csv, meta_csv):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Run scripts/download_isic2019.py first."
            )

    out.mkdir(parents=True, exist_ok=True)

    # ---- labels.csv ----
    labels = build_labels(gt_csv)
    labels = add_folds(labels, n_splits=args.folds, seed=args.seed)
    labels["filename"] = labels["image"] + ".jpg"
    labels["dataset"] = relative_image_dir(images_dir, out)
    labels_out = labels[["dataset", "filename", "label", "fold"]]
    labels_path = out / "labels.csv"
    labels_out.to_csv(labels_path, index=False)

    # ---- hot_one_meta.csv ----
    meta = build_metadata(meta_csv)
    meta_path = out / "hot_one_meta.csv"
    meta.to_csv(meta_path, index=False)

    # ---- report ----
    print(f"Wrote {labels_path}  ({len(labels_out):,} rows)")
    print(f"Wrote {meta_path}  ({len(meta):,} rows, {meta.shape[1] - 1} meta features)")
    print("\nClass distribution:")
    print(labels_out["label"].value_counts().sort_index().to_string())
    print("\nFold sizes:")
    print(labels_out["fold"].value_counts().sort_index().to_string())
    print(f"\nimage dir (relative to {out}): {labels_out['dataset'].iloc[0]}")
    print("\nSet your config `paths` to:")
    print(f'    train_root: "{out.as_posix()}"')
    print(f'    labels_csv: "{labels_path.as_posix()}"')
    print(f'    meta_csv:   "{meta_path.as_posix()}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
