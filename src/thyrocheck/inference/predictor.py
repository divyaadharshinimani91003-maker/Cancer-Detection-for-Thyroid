"""Single entry point for inference - shared by the CLI, FastAPI and Streamlit."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from thyrocheck.config import resolve
from thyrocheck.data.dataset import preprocess_array
from thyrocheck.data.roi import Box, crop_square
from thyrocheck.models.fibonacci_net import load_model
from thyrocheck.xai.gradcam import GradCAM, overlay_heatmap

# Plain-language interpretation shown in the UI and the report
INTERPRETATION = {
    "benign": "Features are more consistent with a benign nodule.",
    "malignant": "Features are suspicious for malignancy. Clinical correlation and FNA biopsy should be considered.",
    "normal": "No nodule pattern detected; appearance is consistent with normal thyroid tissue.",
}


class Predictor:
    def __init__(self, model_path: str | Path, class_names: list[str], input_size: int = 224,
                 channels: int = 3, target_layer: str = "last_conv", alpha: float = 0.4,
                 crop_margin: float = 0.15, expects_roi: bool = False):
        self.model = load_model(str(model_path))
        self.class_names = class_names
        self.input_size, self.channels, self.alpha = input_size, channels, alpha
        # True when the model was trained on nodule crops: callers should pass a box
        self.crop_margin, self.expects_roi = crop_margin, expects_roi
        self.gradcam = GradCAM(self.model, target_layer)

    @classmethod
    def from_config(cls, cfg) -> "Predictor":
        model_path = resolve(cfg.train.model_path)
        names_path = model_path.parent / "class_names.json"
        names = json.loads(names_path.read_text()) if names_path.exists() else list(cfg.data.classes)
        return cls(model_path, names, cfg.model.input_size, cfg.model.channels,
                   cfg.xai.target_layer, cfg.xai.alpha, cfg.data.get("crop_margin", 0.15),
                   bool(cfg.data.get("crop_to_bbox")))

    def predict_array(self, img: np.ndarray, with_gradcam: bool = True) -> dict:
        x = preprocess_array(img, self.input_size, self.channels)[None]
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(probs.argmax())
        label = self.class_names[idx]
        result = {
            "label": label,
            "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
            "interpretation": INTERPRETATION.get(label, ""),
            "image": (x[0] * 255).astype(np.uint8) if self.channels == 3 else
                     np.repeat((x[0] * 255).astype(np.uint8), 3, -1),
        }
        if with_gradcam:
            heat = self.gradcam.heatmap(x, idx)
            result["heatmap"] = heat
            result["overlay"] = overlay_heatmap(result["image"], heat, self.alpha)
        return result

    def predict_bytes(self, data: bytes, with_gradcam: bool = True, box: Box | None = None) -> dict:
        """box: nodule (xmin, ymin, xmax, ymax) in pixels; the image is cropped to it as in training."""
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if box is not None:
            img = crop_square(img, box, self.crop_margin)
        return self.predict_array(np.array(img), with_gradcam)

    def predict_path(self, path: str | Path, with_gradcam: bool = True) -> dict:
        return self.predict_bytes(Path(path).read_bytes(), with_gradcam)


def main() -> None:
    from thyrocheck.config import base_argparser, load_config
    ap = base_argparser("Classify one ultrasound image")
    ap.add_argument("image")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    res = Predictor.from_config(cfg).predict_path(args.image, with_gradcam=False)
    print(json.dumps({k: v for k, v in res.items() if k != "image"}, indent=2))


if __name__ == "__main__":
    main()
