"""Create a tiny synthetic "ultrasound" dataset in data/raw/_dummy so you can
test the whole pipeline before downloading the real Kaggle data.

    python scripts/make_dummy_data.py --per-class 20

The images are speckle noise with a dark blob (benign = round, malignant =
irregular, normal = none). They are only for checking that the code runs -
delete data/raw/_dummy before training on real data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def speckle(rng, h=360, w=560):
    base = rng.rayleigh(0.35, (h, w))
    base = cv2.GaussianBlur(base, (0, 0), 2)
    return np.clip(base / base.max() * 200, 0, 255)


def make(kind: str, rng) -> np.ndarray:
    img = speckle(rng)
    h, w = img.shape
    cy, cx = rng.integers(120, h - 120), rng.integers(150, w - 150)
    if kind == "benign":
        cv2.ellipse(img, (int(cx), int(cy)), (int(rng.integers(35, 60)), int(rng.integers(30, 50))), 0, 0, 360, 40, -1)
    elif kind == "malignant":
        pts = [(cx + r * np.cos(t), cy + r * np.sin(t))
               for t, r in zip(np.linspace(0, 2 * np.pi, 14, endpoint=False), rng.integers(20, 70, 14))]
        cv2.fillPoly(img, [np.array(pts, np.int32)], 25)
        for _ in range(6):  # micro-calcifications
            cv2.circle(img, (int(cx + rng.integers(-25, 25)), int(cy + rng.integers(-25, 25))), 2, 255, -1)
    return img.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for kind in ("benign", "malignant", "normal"):
        d = ROOT / "data" / "raw" / "_dummy" / kind
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            cv2.imwrite(str(d / f"{kind}_{i:03d}.png"), make(kind, rng))
    print(f"wrote {3 * args.per_class} dummy images to data/raw/_dummy")


if __name__ == "__main__":
    main()
