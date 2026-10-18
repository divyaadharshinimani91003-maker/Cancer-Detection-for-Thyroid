"""tf.data input pipelines + a shared preprocessing function used by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from thyrocheck.config import PROJECT_ROOT, resolve

AUTOTUNE = tf.data.AUTOTUNE


def load_splits(cfg) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg.data.splits_csv))
    df["abs_path"] = [str(PROJECT_ROOT / p) for p in df["path"]]
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    return list(df.sort_values("label_idx").drop_duplicates("label_idx")["label"])


def preprocess_array(img: np.ndarray, size: int, channels: int = 3) -> np.ndarray:
    """Numpy preprocessing for inference (API / Streamlit). Must match `_decode` below."""
    img_t = tf.convert_to_tensor(img)
    if img_t.shape.rank == 2:
        img_t = img_t[..., None]
    if img_t.shape[-1] == 4:
        img_t = img_t[..., :3]
    if channels == 3 and img_t.shape[-1] == 1:
        img_t = tf.image.grayscale_to_rgb(img_t)
    if channels == 1 and img_t.shape[-1] == 3:
        img_t = tf.image.rgb_to_grayscale(img_t)
    img_t = tf.image.resize(tf.cast(img_t, tf.float32), (size, size))
    return (img_t / 255.0).numpy()


def _decode(path: tf.Tensor, size: int, channels: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    img = tf.image.resize(tf.cast(img, tf.float32), (size, size))
    return img / 255.0


def _augment(img: tf.Tensor) -> tf.Tensor:
    # Ultrasound-safe augmentations: no vertical flip (probe is always on top)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0)


def make_dataset(df: pd.DataFrame, cfg, split: str, shuffle: bool | None = None) -> tf.data.Dataset:
    part = df[df["split"] == split]
    n_classes = df["label_idx"].nunique()
    size, ch = cfg.model.input_size, cfg.model.channels
    shuffle = (split == "train") if shuffle is None else shuffle

    ds = tf.data.Dataset.from_tensor_slices((part["abs_path"].values, part["label_idx"].values))
    if shuffle:
        ds = ds.shuffle(len(part), seed=cfg.project.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode(p, size, ch), tf.one_hot(y, n_classes)), num_parallel_calls=AUTOTUNE)
    if split == "train" and cfg.train.augment:
        ds = ds.map(lambda x, y: (_augment(x), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg.train.batch_size).prefetch(AUTOTUNE)
