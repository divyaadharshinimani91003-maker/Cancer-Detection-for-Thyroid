"""FastAPI service.

Run locally:
    uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs     Web page: http://localhost:8000/

Endpoints
  GET  /health   -> model status
  POST /predict  -> label, probabilities, Grad-CAM overlay (base64 PNG)
  POST /report   -> .docx clinical report (download)

Both POST endpoints take an optional form field `box="xmin,ymin,xmax,ymax"` (pixels): the nodule
region. The model is trained on nodule crops (data.crop_to_bbox), so pass it for full scans.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402

from thyrocheck.config import load_config, resolve  # noqa: E402
from thyrocheck.data.roi import Box, parse_box  # noqa: E402
from thyrocheck.inference.predictor import Predictor  # noqa: E402
from thyrocheck.reports.report_generator import generate_report  # noqa: E402

cfg = load_config()
app = FastAPI(title="ThyroCheck AI", version="0.1.0",
              description="Thyroid nodule classification from ultrasound (FibonacciNet + Grad-CAM). Research use only.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor: Predictor | None = None
ALLOWED = {"image/png", "image/jpeg", "image/bmp", "image/tiff"}


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        model_path = resolve(cfg.train.model_path)
        if not model_path.exists() and os.environ.get("HF_MODEL_REPO"):
            from thyrocheck.hub import pull
            pull(os.environ["HF_MODEL_REPO"], cfg)
        if not model_path.exists():
            raise HTTPException(503, f"Model not found at {model_path}. Train it first or set HF_MODEL_REPO.")
        _predictor = Predictor.from_config(cfg)
    return _predictor


def _b64png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _read(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type {file.content_type}")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 15 MB")
    return data


def _box(data: bytes, box: str | None) -> Box | None:
    if not box:
        return None
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
    try:
        return parse_box(box, width, height)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/health")
def health():
    model_path = resolve(cfg.train.model_path)
    return {"status": "ok", "model_loaded": _predictor is not None, "model_file_exists": model_path.exists(),
            "expects_nodule_box": bool(cfg.data.get("crop_to_bbox"))}


@app.post("/predict")
async def predict(file: UploadFile = File(...), gradcam: bool = True, box: str | None = Form(None)):
    data = await _read(file)
    res = get_predictor().predict_bytes(data, with_gradcam=gradcam, box=_box(data, box))
    payload = {k: res[k] for k in ("label", "confidence", "probabilities", "interpretation")}
    if gradcam:
        payload["gradcam_png_base64"] = _b64png(res["overlay"])
    return payload


@app.post("/report")
async def report(file: UploadFile = File(...), patient_id: str = Form("-"), patient_name: str = Form("-"),
                 age: str = Form("-"), sex: str = Form("-"), clinician: str = Form("-"), notes: str = Form(""),
                 box: str | None = Form(None)):
    data = await _read(file)
    res = get_predictor().predict_bytes(data, with_gradcam=True, box=_box(data, box))
    path = generate_report(
        res, {"id": patient_id, "name": patient_name, "age": age, "sex": sex, "clinician": clinician, "notes": notes},
        resolve(cfg.report.output_dir), cfg.report.institution, cfg.report.disclaimer)
    return FileResponse(path, filename=path.name,
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
