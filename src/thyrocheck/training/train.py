"""Step 3 - train FibonacciNet.

  python -m thyrocheck.training.train
  python -m thyrocheck.training.train --set train.epochs=40 train.augment=true

Outputs (under outputs/):
  models/fibonacci_net.keras   best model by val loss
  models/class_names.json      label order used by the API
  logs/history.csv             per-epoch metrics
  figures/training_curves.png
  tensorboard/                 `tensorboard --logdir outputs/tensorboard`
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import keras

from thyrocheck.config import base_argparser, load_config, resolve, set_seed
from thyrocheck.data.dataset import class_names, load_splits, make_dataset
from thyrocheck.models.fibonacci_net import build_from_config


def paper_lr_schedule(base_lr: float, constant_epochs: int, decay: float):
    """Constant LR for `constant_epochs`, then multiply by `decay` every epoch."""
    def schedule(epoch: int, lr: float) -> float:
        return base_lr if epoch < constant_epochs else base_lr * decay ** (epoch - constant_epochs + 1)
    return schedule


def plot_history(hist: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist[key], label="train")
        ax.plot(hist[f"val_{key}"], label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = base_argparser("Train FibonacciNet on thyroid ultrasound").parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.project.seed)

    out = resolve(cfg.train.output_dir)
    for sub in ("models", "logs", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    df = load_splits(cfg)
    names = class_names(df)
    train_ds = make_dataset(df, cfg, "train")
    val_ds = make_dataset(df, cfg, "val")

    model = build_from_config(cfg, num_classes=len(names))
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.train.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    model.summary(line_length=110)

    class_weight = None
    if cfg.train.class_weights:
        y = df.loc[df["split"] == "train", "label_idx"].values
        w = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight = dict(enumerate(w))
        print("[train] class weights:", class_weight)

    model_path = resolve(cfg.train.model_path)
    callbacks = [
        keras.callbacks.LearningRateScheduler(
            paper_lr_schedule(cfg.train.learning_rate, cfg.train.lr_constant_epochs, cfg.train.lr_decay), verbose=0),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor=cfg.train.get("monitor", "val_loss"),
                                        mode="max" if "acc" in cfg.train.get("monitor", "val_loss") or "auc" in cfg.train.get("monitor", "") else "min",
                                        save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(str(out / "logs" / "history.csv")),
    ]
    try:
        import tensorboard  # noqa: F401
        callbacks.append(keras.callbacks.TensorBoard(str(out / "tensorboard")))
    except ImportError:
        print("[train] tensorboard not installed - skipping TensorBoard logs")
    if cfg.train.early_stopping_patience:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.train.early_stopping_patience, restore_best_weights=True))

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.train.epochs,
                        callbacks=callbacks, class_weight=class_weight)

    with open(model_path.parent / "class_names.json", "w") as f:
        json.dump(names, f)
    plot_history(pd.DataFrame(history.history), out / "figures" / "training_curves.png")
    print(f"[train] best model -> {model_path}")
    print("[train] next: python -m thyrocheck.evaluation.evaluate")


if __name__ == "__main__":
    main()
