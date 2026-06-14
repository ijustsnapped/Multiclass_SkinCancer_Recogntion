#!/usr/bin/env python3
"""
download_isic2019.py

Download the official ISIC 2019 challenge dataset (images, ground truth, metadata)
from the ISIC archive S3 bucket and unpack it into a local ``data/`` directory.

The ISIC 2019 dataset is released under CC-BY-NC 4.0. By downloading it you agree
to the licenses on the ISIC 2019 and HAM10000 dataset pages:
    https://challenge.isic-archive.com/data/

After running this you will have:

    data/raw/
        ISIC_2019_Training_Input/            # 25,331 JPEGs
        ISIC_2019_Training_GroundTruth.csv    # 8-class one-hot labels (+ UNK)
        ISIC_2019_Training_Metadata.csv       # age / sex / anatom_site
        ISIC_2019_Test_Input/                 # 8,238 JPEGs (optional, --with-test)

Then run ``scripts/prepare_isic2019.py`` to build the ``splits/`` layout the
training scripts expect.

Usage:
    python scripts/download_isic2019.py                 # training data only
    python scripts/download_isic2019.py --with-test     # also fetch the test set
    python scripts/download_isic2019.py --out data/raw  # custom destination
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019"

FILES = {
    "train_images": ("ISIC_2019_Training_Input.zip", True),       # (name, is_zip)
    "train_gt": ("ISIC_2019_Training_GroundTruth.csv", False),
    "train_meta": ("ISIC_2019_Training_Metadata.csv", False),
}
TEST_FILES = {
    "test_images": ("ISIC_2019_Test_Input.zip", True),
}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    sys.stdout.write(f"\r    {pct:5.1f}%  ({mb:7.1f} / {total_mb:7.1f} MB)")
    sys.stdout.flush()


def download(name: str, dest_dir: Path) -> Path:
    url = f"{BASE_URL}/{name}"
    dest = dest_dir / name
    if dest.exists():
        print(f"[skip] {name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    sys.stdout.write("\n")
    tmp.rename(dest)
    return dest


def unzip(zip_path: Path, dest_dir: Path) -> None:
    # The archives expand to a folder matching the zip stem.
    extracted = dest_dir / zip_path.stem
    if extracted.exists():
        print(f"[skip] {zip_path.stem}/ already extracted")
        return
    print(f"[unzip] {zip_path.name} -> {dest_dir}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the official ISIC 2019 dataset.")
    ap.add_argument("--out", type=Path, default=Path("data/raw"),
                    help="Destination directory (default: data/raw)")
    ap.add_argument("--with-test", action="store_true",
                    help="Also download the 3.6 GB test image set.")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the downloaded .zip archives after extraction.")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    targets = dict(FILES)
    if args.with_test:
        targets.update(TEST_FILES)

    print(f"Downloading ISIC 2019 into {out.resolve()}\n")
    for _key, (name, is_zip) in targets.items():
        path = download(name, out)
        if is_zip:
            unzip(path, out)
            if not args.keep_zip:
                path.unlink(missing_ok=True)

    print("\nDone. Next step:")
    print("    python scripts/prepare_isic2019.py --raw", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
