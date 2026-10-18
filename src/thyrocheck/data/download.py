"""Step 1 - download the Kaggle dataset into data/raw.

Two options (both need a Kaggle account):
  A) kagglehub (recommended):  pip install kagglehub
     Auth: put kaggle.json in ~/.kaggle/  OR  set KAGGLE_USERNAME + KAGGLE_KEY
  B) Kaggle CLI:  kaggle datasets download -d <slug> -p data/raw --unzip

Usage:
  python -m thyrocheck.data.download
  python -m thyrocheck.data.download --set data.kaggle_dataset=<owner>/<slug>
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from thyrocheck.config import base_argparser, load_config, resolve


def download_with_kagglehub(slug: str, dest: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[download] kagglehub not installed - trying the Kaggle CLI instead")
        return False
    print(f"[download] kagglehub -> {slug}")
    cache_path = Path(kagglehub.dataset_download(slug))
    print(f"[download] cached at {cache_path}; copying to {dest}")
    shutil.copytree(cache_path, dest, dirs_exist_ok=True)
    return True


def download_with_cli(slug: str, dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        return False
    print(f"[download] kaggle CLI -> {slug}")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    return subprocess.call(cmd) == 0


def main() -> None:
    args = base_argparser("Download the thyroid ultrasound dataset from Kaggle").parse_args()
    cfg = load_config(args.config, args.set)
    dest = resolve(cfg.data.raw_dir)
    dest.mkdir(parents=True, exist_ok=True)
    slug = cfg.data.kaggle_dataset

    ok = download_with_kagglehub(slug, dest) or download_with_cli(slug, dest)
    if not ok:
        sys.exit(
            "[download] Could not download. Either `pip install kagglehub` and set up "
            "Kaggle credentials, or download the zip manually from\n"
            f"  https://www.kaggle.com/datasets/{slug}\n"
            f"and extract it into {dest}"
        )

    n = sum(1 for p in dest.rglob("*") if p.suffix.lower() in cfg.data.image_extensions)
    print(f"[download] done - {n} images under {dest}")
    print("[download] next: python -m thyrocheck.data.prepare")


if __name__ == "__main__":
    main()
