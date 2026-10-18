"""Streamlit clinician UI.

    streamlit run app/streamlit_app.py

Loads the trained .keras model directly - no API service needed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from thyrocheck.config import load_config, resolve  # noqa: E402
from thyrocheck.data.roi import crop_square  # noqa: E402

cfg = load_config()
st.set_page_config(page_title="ThyroCheck AI", page_icon="🩺", layout="wide")


@st.cache_resource
def local_predictor():
    from thyrocheck.inference.predictor import Predictor
    return Predictor.from_config(cfg)


def predict_local(data: bytes, box) -> dict:
    res = local_predictor().predict_bytes(data, with_gradcam=True, box=box)
    return res


# ---------------- sidebar
st.sidebar.title("🩺 ThyroCheck AI")
st.sidebar.subheader("Patient (for the report)")
patient = {
    "id": st.sidebar.text_input("Patient ID", "P-0001"),
    "name": st.sidebar.text_input("Name", ""),
    "age": st.sidebar.text_input("Age", ""),
    "sex": st.sidebar.selectbox("Sex", ["-", "F", "M", "Other"]),
    "clinician": st.sidebar.text_input("Referring clinician", ""),
    "notes": st.sidebar.text_area("Clinical notes", ""),
}
st.sidebar.caption("Research prototype - not for clinical decisions.")

# ---------------- main
st.title("Thyroid Nodule Classification with Explainable AI")
st.write("Upload a B-mode thyroid ultrasound image and mark the nodule. FibonacciNet predicts "
         "the class and Grad-CAM shows which regions drove the decision.")

upload = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if upload is None:
    st.info("Waiting for an image…")
    st.stop()

data = upload.getvalue()
scan = Image.open(io.BytesIO(data)).convert("RGB")
margin = cfg.data.get("crop_margin", 0.15)

# The model is trained on nodule crops (data.crop_to_bbox), so the user marks the nodule first
box = None
if cfg.data.get("crop_to_bbox"):
    st.subheader("1. Mark the nodule")
    if not st.checkbox("This image is already cropped to the nodule"):
        w, h = scan.size
        b1, b2 = st.columns([2, 1])
        with b2:
            xs = st.slider("Left / right edge (px)", 0, w, (w // 4, 3 * w // 4))
            ys = st.slider("Top / bottom edge (px)", 0, h, (h // 4, 3 * h // 4))
        if xs[1] - xs[0] < 8 or ys[1] - ys[0] < 8:
            st.warning("The box is too small - widen it around the nodule.")
            st.stop()
        box = (xs[0], ys[0], xs[1], ys[1])
        preview = scan.copy()
        ImageDraw.Draw(preview).rectangle(box, outline=(255, 64, 64), width=max(2, w // 200))
        b1.image(preview, caption="Drag the sliders until the red box surrounds the nodule",
                 use_container_width=True)
        b2.image(crop_square(scan, box, margin), caption="What the model sees", use_container_width=True)
    if not st.button("Analyse nodule", type="primary") and "res_key" not in st.session_state:
        st.stop()
    st.subheader("2. Result")

key = (upload.name, len(data), box)
try:
    with st.spinner("Analysing…"):
        if st.session_state.get("res_key") != key:
            st.session_state.res = predict_local(data, box)
            st.session_state.res_key = key
        res = st.session_state.res
except Exception as e:  # noqa: BLE001
    st.error(f"Prediction failed: {e}")
    st.stop()

c1, c2, c3 = st.columns([1, 1, 1])
c1.image(crop_square(scan, box, margin) if box else scan, caption="Input", use_container_width=True)
c2.image(res["overlay"], caption="Grad-CAM", use_container_width=True)
with c3:
    color = {"malignant": "red", "benign": "green", "normal": "blue"}.get(res["label"], "gray")
    st.markdown(f"### Prediction: :{color}[{res['label'].upper()}]")
    st.metric("Confidence", f"{res['confidence']:.1%}")
    st.write(res.get("interpretation", ""))
    probs = pd.DataFrame({"probability": res["probabilities"]})
    st.bar_chart(probs)

st.markdown("---")
if st.button("📄 Generate clinical report (.docx)"):
    from thyrocheck.reports.report_generator import generate_report
    path = generate_report(res, patient, resolve(cfg.report.output_dir),
                           cfg.report.institution, cfg.report.disclaimer)
    st.download_button("Download report", path.read_bytes(), file_name=path.name,
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
