#!/usr/bin/env python3
"""
resize_images.py

One-time downscale of the ISIC 2019 JPEGs so training/sweeps aren't bottlenecked
decoding full-resolution images every epoch. Resizes each image so its **shorter
side** equals ``--size`` (preserving aspect ratio); images already smaller are
copied unchanged. Output mirrors the input folder with identical filenames.

Short-side 320 (default) covers both the B0 pipeline (resize 256 / crop 224) and
the B3 pipeline (resize 320 / crop 300) without upscaling, shrinks avg decoded
size ~5x, and makes a RAM cache feasible (~8 GB instead of ~50 GB).

Usage:
    python scripts/resize_images.py --update-labels
    python scripts/resize_images.py --size 288 --quality 92 --update-labels
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

VALID = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_one(src: Path, dst: Path, size: int, quality: int) -> str:
    if dst.exists():
        return "skip"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            short = min(w, h)
            if short <= size:
                shutil.copy2(src, dst)
                return "copy"
            scale = size / short
            new = (round(w * scale), round(h * scale))
            im.resize(new, Image.BILINEAR).save(dst, "JPEG", quality=quality)
            return "resized"
    except Exception as e:  # don't let one bad file kill the batch
        print(f"[warn] {src.name}: {e}")
        return "error"


def update_labels(labels_csv: Path, src_dir: Path, dst_dir: Path) -> None:
    if not labels_csv.exists():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    required = {"dataset", "filename", "label", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{labels_csv} missing required columns: {sorted(missing)}")

    labels_dir = labels_csv.parent.resolve()
    src_resolved = src_dir.resolve()
    dst_rel = Path(os.path.relpath(dst_dir.resolve(), labels_dir)).as_posix()

    dataset_paths = df["dataset"].astype(str).map(
        lambda value: (labels_dir / value).resolve()
    )
    mask = dataset_paths == src_resolved
    if not mask.any():
        current = sorted(df["dataset"].astype(str).unique().tolist())
        raise SystemExit(
            f"No rows in {labels_csv} point at {src_dir}. "
            f"Current dataset values: {current}"
        )

    backup = labels_csv.with_name(labels_csv.name + ".raw.bak")
    if not backup.exists():
        shutil.copy2(labels_csv, backup)

    df.loc[mask, "dataset"] = dst_rel
    df.to_csv(labels_csv, index=False)
    print(f"Updated {mask.sum():,} label rows to dataset={dst_rel}")
    print(f"Backup: {backup}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Downscale ISIC images by shorter side.")
    ap.add_argument("--src", type=Path, default=Path("data/raw/ISIC_2019_Training_Input"))
    ap.add_argument("--dst", type=Path, default=Path("data/resized/ISIC_2019_Training_Input"))
    ap.add_argument("--size", type=int, default=320, help="Target shorter-side length (px).")
    ap.add_argument("--quality", type=int, default=92, help="Output JPEG quality.")
    ap.add_argument("--update-labels", action="store_true",
                    help="Rewrite labels.csv rows that point at --src so training uses --dst.")
    ap.add_argument("--labels-csv", type=Path, default=Path("splits/training/labels.csv"),
                    help="labels.csv to update when --update-labels is set.")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source dir not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(args.src.iterdir()) if p.suffix.lower() in VALID]
    print(f"Resizing {len(files):,} images: {args.src} -> {args.dst} (short side {args.size}px)")

    counts: dict[str, int] = {}
    for p in tqdm(files, ncols=90, desc="resize"):
        r = resize_one(p, args.dst / p.name, args.size, args.quality)
        counts[r] = counts.get(r, 0) + 1

    print("done:", counts)
    if args.update_labels:
        update_labels(args.labels_csv, args.src, args.dst)
    else:
        print("\nPoint training at the resized images with:\n"
              "    python scripts/resize_images.py --update-labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
