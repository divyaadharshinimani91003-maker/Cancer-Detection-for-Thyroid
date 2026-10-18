"""Load configs/config.yaml and apply command-line overrides."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access: cfg.train.epochs"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "1e-3" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path or os.environ.get("THYROCHECK_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(apply_overrides(cfg, overrides))


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Override config values, e.g. --set train.epochs=40")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
