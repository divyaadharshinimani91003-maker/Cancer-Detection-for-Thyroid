"""Step 2 - index raw images, assign labels, deduplicate, stratified split.

Writes data/processed/splits.csv with columns: path, label, label_idx, split, md5
and prints a class-balance table so you can sanity-check the dataset.

Labels come from `data.label_source`:
  folder  - inferred from folder names using `data.label_keywords`, so this works
            whatever the exact Kaggle layout is (e.g. AUITD/benign/xxx.jpg or
            Dataset/Train/Malignant/xxx.png)
  voc_xml - read from Pascal-VOC Annotations/<image>.xml and mapped with
            `data.voc_label_map` (TN5000: 0 = benign, 1 = malignant)
Unmatched files are listed and skipped.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from thyrocheck.config import base_argparser, load_config, resolve, PROJECT_ROOT
from thyrocheck.data.roi import crop_square


def infer_label(path: Path, root: Path, keywords: dict[str, list[str]], classes: list[str]) -> str | None:
    """Return the class whose keyword appears in the path (deepest folder wins)."""
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    for part in reversed(parts):  # deepest folder first
        for cls in classes:
            if any(k.lower() in part for k in keywords.get(cls, [cls])):
                return cls
    return None


def voc_label(path: Path, root: Path, label_map: dict) -> str | None:
    """Read the class from a Pascal-VOC annotation (TN5000: Annotations/<stem>.xml, <name> 0/1)."""
    xml = root / "Annotations" / f"{path.stem}.xml"
    if not xml.exists():
        return None
    names = {o.findtext("name", "").strip() for o in ET.parse(xml).getroot().iter("object")}
    labels = {label_map.get(n) for n in names}
    return labels.pop() if len(labels) == 1 else None  # skip unlabelled or mixed-label images


def voc_box(path: Path, root: Path) -> tuple[int, int, int, int] | None:
    """Union of all <bndbox> boxes in the image's VOC annotation, as (xmin, ymin, xmax, ymax)."""
    xml = root / "Annotations" / f"{path.stem}.xml"
    if not xml.exists():
        return None
    boxes = [[int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
             for b in ET.parse(xml).getroot().iter("bndbox")]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def write_crops(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Crop every image to its annotated nodule (ROI) and point `path` at the crop."""
    root = resolve(cfg.data.raw_dir)
    out_dir = resolve(cfg.data.processed_dir) / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    margin = cfg.data.get("crop_margin", 0.15)
    paths, missing = [], 0
    for rel in df["path"]:
        src = PROJECT_ROOT / rel
        box = voc_box(src, root)
        if box is None:
            missing += 1
            paths.append(rel)
            continue
        dst = out_dir / f"{src.stem}.png"
        with Image.open(src) as img:
            crop_square(img.convert("RGB"), box, margin).save(dst)
        paths.append(str(dst.relative_to(PROJECT_ROOT)))
    print(f"[prepare] cropped {len(df) - missing} images to the nodule box (margin {margin}) -> {out_dir}"
          + (f"; {missing} without a box kept whole" if missing else ""))
    return df.assign(path=paths)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(cfg) -> pd.DataFrame:
    root = resolve(cfg.data.raw_dir)
    exts = {e.lower() for e in cfg.data.image_extensions}
    classes = list(cfg.data.classes)
    source = cfg.data.get("label_source", "folder")
    label_map = {str(k): v for k, v in (cfg.data.get("voc_label_map") or {}).items()}
    rows, unmatched = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # skip mask / annotation images often shipped alongside the scans
        if any(t in p.stem.lower() for t in ("mask", "_seg", "label")):
            continue
        if source == "voc_xml":
            label = voc_label(p, root, label_map)
        else:
            label = infer_label(p, root, cfg.data.label_keywords, classes)
        if label not in classes:
            label = None
        if label is None:
            unmatched.append(p)
            continue
        rows.append({"path": str(p.relative_to(PROJECT_ROOT)), "label": label})
    if unmatched:
        print(f"[prepare] {len(unmatched)} images had no class keyword in their path and were skipped, e.g.:")
        for p in unmatched[:5]:
            print("   ", p.relative_to(root))
        print("   -> check data.label_source / label_keywords / voc_label_map in configs/config.yaml")
    if not rows:
        raise SystemExit(f"[prepare] No labelled images found under {root}. Did you run the download step?")
    return pd.DataFrame(rows)


def main() -> None:
    args = base_argparser("Index, deduplicate and split the dataset").parse_args()
    cfg = load_config(args.config, args.set)
    seed = cfg.project.seed

    df = build_index(cfg)
    if cfg.data.deduplicate:
        df["md5"] = [md5(PROJECT_ROOT / p) for p in df["path"]]
        conflict = df.groupby("md5")["label"].nunique()
        bad = set(conflict[conflict > 1].index)
        if bad:
            print(f"[prepare] dropping {df['md5'].isin(bad).sum()} images whose identical copies carry different labels")
            df = df[~df["md5"].isin(bad)]
        before = len(df)
        df = df.drop_duplicates("md5").reset_index(drop=True)
        print(f"[prepare] removed {before - len(df)} exact duplicates")
    else:
        df["md5"] = ""

    if cfg.data.get("crop_to_bbox") and cfg.data.get("label_source") == "voc_xml":
        df = write_crops(df, cfg)

    classes = [c for c in cfg.data.classes if c in set(df["label"])]
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})

    test_size, val_size = cfg.data.test_size, cfg.data.val_size
    trainval, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(trainval, test_size=val_size / (1 - test_size),
                                  stratify=trainval["label"], random_state=seed)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    out = pd.concat([train, val, test]).reset_index(drop=True)

    out_path = resolve(cfg.data.splits_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["path", "label", "label_idx", "split", "md5"]].to_csv(out_path, index=False)

    print("\n[prepare] class balance per split:")
    print(pd.crosstab(out["label"], out["split"], margins=True).to_string())
    print(f"\n[prepare] classes (index order): {classes}")
    print(f"[prepare] wrote {out_path}")
    print("[prepare] next: python -m thyrocheck.training.train")


if __name__ == "__main__":
    main()
