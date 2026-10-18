"""Grad-CAM (Selvaraju et al., 2017) for any Keras CNN.

Library use:
    cam = GradCAM(model, "last_conv")
    heatmap = cam.heatmap(batch)                  # (H, W) in [0, 1]
    overlay = overlay_heatmap(image_uint8, heatmap)

CLI - heatmaps for N test images into outputs/gradcam/:
    python -m thyrocheck.xai.gradcam --n 12
    python -m thyrocheck.xai.gradcam --image path/to/scan.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import keras


class GradCAM:
    def __init__(self, model: keras.Model, target_layer: str = "last_conv"):
        self.model = model
        self.grad_model = keras.Model(model.inputs, [model.get_layer(target_layer).output, model.output])

    def heatmap(self, batch: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """batch: (1, H, W, C) preprocessed image. Returns heatmap (H, W) in [0, 1]."""
        x = tf.convert_to_tensor(batch, tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, conv_out)               # d(score)/d(feature maps)
        weights = tf.reduce_mean(grads, axis=(1, 2))          # global-average the gradients
        cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (batch.shape[2], batch.shape[1]))


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """image: uint8 RGB (H, W, 3) or float [0,1]; returns uint8 RGB overlay."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    heat = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from thyrocheck.config import base_argparser, load_config, resolve
    from thyrocheck.inference.predictor import Predictor

    ap = base_argparser("Generate Grad-CAM heatmaps")
    ap.add_argument("--n", type=int, default=12, help="number of test images")
    ap.add_argument("--image", default=None, help="single image instead of test split")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = resolve(cfg.train.output_dir) / "gradcam"
    out.mkdir(parents=True, exist_ok=True)

    predictor = Predictor.from_config(cfg)
    if args.image:
        items = [(args.image, None)]
    else:
        import pandas as pd
        from thyrocheck.data.dataset import load_splits
        df = load_splits(cfg)
        test = df[df["split"] == "test"]
        per_class = max(1, args.n // df["label"].nunique())
        test = pd.concat([g.sample(min(len(g), per_class), random_state=0) for _, g in test.groupby("label")])
        items = list(zip(test["abs_path"], test["label"]))

    for path, true_label in items:
        res = predictor.predict_path(path, with_gradcam=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(res["image"])
        axes[0].set_title(f"true: {true_label or '?'}")
        axes[1].imshow(res["overlay"])
        axes[1].set_title(f"pred: {res['label']} ({res['confidence']:.0%})")
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        name = f"{true_label or 'img'}_{Path(path).stem}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
    print(f"[gradcam] saved {len(items)} figures to {out}")


if __name__ == "__main__":
    main()
