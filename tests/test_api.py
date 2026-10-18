"""API smoke test - needs a trained model (run the pipeline, or the dummy pipeline, first)."""
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")
from fastapi.testclient import TestClient  # noqa: E402

from thyrocheck.config import load_config, resolve  # noqa: E402

if not resolve(load_config().train.model_path).exists():
    pytest.skip("no trained model yet", allow_module_level=True)

from api.main import app  # noqa: E402

client = TestClient(app)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(300, 400) * 255).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] in d["probabilities"]
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert len(d["gradcam_png_base64"]) > 100


def test_report():
    r = client.post("/report", files={"file": ("x.png", _png(), "image/png")}, data={"patient_id": "T-1"})
    assert r.status_code == 200
    assert r.content[:2] == b"PK"   # docx is a zip


def test_rejects_non_image():
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_with_nodule_box():
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")}, data={"box": "100,50,300,250"})
    assert r.status_code == 200
    assert r.json()["label"] in r.json()["probabilities"]


def test_rejects_box_outside_image():
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")}, data={"box": "100,50,900,250"})
    assert r.status_code == 422
