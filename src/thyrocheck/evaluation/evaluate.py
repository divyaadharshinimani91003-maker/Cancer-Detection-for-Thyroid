"""Step 4 - evaluate on the held-out test split.

  python -m thyrocheck.evaluation.evaluate

Writes outputs/metrics/: test_metrics.json, classification_report.csv,
predictions.csv, and outputs/figures/: confusion_matrix.png, roc_curves.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

from thyrocheck.config import base_argparser, load_config, resolve
from thyrocheck.data.dataset import class_names, load_splits, make_dataset
from thyrocheck.models.fibonacci_net import load_model


def specificity_per_class(cm: np.ndarray) -> list[float]:
    spec = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)
    return spec


def main() -> None:
    ap = base_argparser("Evaluate a trained model on the test split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    out = resolve(cfg.train.output_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_splits(cfg)
    names = class_names(df)
    part = df[df["split"] == args.split].reset_index(drop=True)
    ds = make_dataset(df, cfg, args.split, shuffle=False)

    model = load_model(str(resolve(cfg.train.model_path)))
    probs = model.predict(ds, verbose=1)
    y_true = part["label_idx"].values
    y_pred = probs.argmax(1)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    metrics = {
        "split": args.split,
        "n_images": int(len(part)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity_per_class": dict(zip(names, specificity_per_class(cm))),
    }
    try:
        if len(names) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError as e:  # a class missing from a tiny split
        metrics["roc_auc_error"] = str(e)

    report = classification_report(y_true, y_pred, labels=range(len(names)), target_names=names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "metrics" / "classification_report.csv")
    with open(out / "metrics" / f"{args.split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = part[["path", "label"]].copy()
    pred_df["predicted"] = [names[i] for i in y_pred]
    for i, n in enumerate(names):
        pred_df[f"p_{n}"] = probs[:, i]
    pred_df.to_csv(out / "metrics" / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix ({args.split})")
    fig.tight_layout()
    fig.savefig(out / "figures" / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for i, n in enumerate(names):
        if len(np.unique(y_true == i)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
        ax.plot(fpr, tpr, label=n)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "roc_curves.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, labels=range(len(names)), target_names=names, zero_division=0))


if __name__ == "__main__":
    main()
