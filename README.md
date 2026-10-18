# ThyroCheck AI

Thyroid nodule classification from ultrasound with a custom **FibonacciNet** CNN, **Grad-CAM** explainability, automated **python-docx** clinical reports, a **FastAPI** backend, a **Streamlit** UI, a small **HTML/JS** page, and **Google Cloud Run** deployment.

> Research / learning prototype. Not a medical device. Never upload real patient data to a public deployment.

---

## Pipeline at a glance

```
Kaggle (AUITD)                                                                  ┌─> FastAPI  /predict /report ─> HTML/JS page
   │ 1 download        2 prepare               3 train          4 evaluate      │
   ▼                   ▼                       ▼                ▼               ├─> Streamlit UI
data/raw ──> label from folder names ──> FibonacciNet ──> metrics, CM, ROC ──> 5 Grad-CAM ──> 6 .docx report
             dedupe (md5) + stratified     Adam 1e-4, paper                    │
             80/10/10 split -> splits.csv  LR schedule, bs 8                   └─> 7 Docker -> GCP Cloud Run  (+ model on Hugging Face Hub)
```

| Step | Command | Output |
|---|---|---|
| 1. Download | `python -m thyrocheck.data.download` | `data/raw/` |
| 2. Prepare | `python -m thyrocheck.data.prepare` | `data/processed/splits.csv` + class-balance table |
| 3. Train | `python -m thyrocheck.training.train` | `outputs/models/fibonacci_net.keras`, curves |
| 4. Evaluate | `python -m thyrocheck.evaluation.evaluate` | `outputs/metrics/*`, confusion matrix, ROC |
| 5. Grad-CAM | `python -m thyrocheck.xai.gradcam --n 12` | `outputs/gradcam/*.png` |
| 6. Report | `python -m thyrocheck.reports.report_generator <img>` | `outputs/reports/*.docx` |
| 7. Serve | `uvicorn api.main:app --port 8000` / `streamlit run app/streamlit_app.py` | web UI |
| 8. Deploy | `bash deploy/deploy_cloud_run.sh <GCP_PROJECT_ID>` | public Cloud Run URLs |

Or run steps 1–6 in one go: `bash run_pipeline.sh`

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                       # makes `thyrocheck` importable

# 0) Sanity check with synthetic images (~3 min on CPU, no Kaggle account needed)
bash run_pipeline.sh --dummy
rm -rf data/raw/_dummy data/processed outputs          # clean up before real training
```

### Real data: Kaggle AUITD

Dataset: **Algerian Ultrasound Images Thyroid Dataset (AUITD)** — `azouzmaroua/algeria-ultrasound-images-thyroid-dataset-auitd`
3 classes: benign, malignant, normal.

1. Kaggle → *Settings* → *API* → **Create New Token** → save `kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
2. `bash run_pipeline.sh` (or run the steps one by one).

`prepare` finds the class of each image by searching its folder path for keywords (`data.label_keywords` in `configs/config.yaml`). If it reports skipped images, check the folder names it prints and add them as keywords.

**Other datasets.** Any dataset whose class is in the folder name works — change `data.kaggle_dataset`, `data.classes` and `data.label_keywords`. For a benign-vs-malignant task on AUITD set `data.classes: [benign, malignant]`. DDTI (TI-RADS scores in XML) needs a small custom label step.

### Train on a free Kaggle GPU (recommended)

Training on CPU is slow. Open `notebooks/kaggle_train.ipynb` on Kaggle, attach the AUITD dataset, enable *GPU T4*, upload this project as a Kaggle dataset, and run all cells. Download `outputs/models/` back into this project for serving.

---

## Configuration

Everything lives in `configs/config.yaml`. Override from the command line:

```bash
python -m thyrocheck.training.train --set train.epochs=40 train.augment=true train.class_weights=true
python -m thyrocheck.training.train --set model.convs_per_block=2          # bigger model
python -m thyrocheck.training.train --set model.use_pcb_2_4=false model.use_pcb_3_5=false   # ablation
```

Defaults follow the paper: 224×224 input, Adam, lr 1e-4 constant for 13 epochs then ×0.9 per epoch, 25 epochs, batch size 8, stratified 80/10/10 (the paper uses 60/20/20), no augmentation, no early stopping.

---

## Project structure

```
thyrocheck-ai/
├── configs/config.yaml              # all hyper-parameters & paths
├── src/thyrocheck/
│   ├── config.py                    # YAML loader + --set overrides
│   ├── data/
│   │   ├── download.py              # Kaggle download (kagglehub / CLI)
│   │   ├── prepare.py               # labels from folders, md5 dedupe, stratified split
│   │   └── dataset.py               # tf.data pipeline + shared preprocessing
│   ├── models/fibonacci_net.py      # FibonacciNet + Avg2MaxPooling2D layer
│   ├── training/train.py            # training loop, paper LR schedule, callbacks
│   ├── evaluation/evaluate.py       # accuracy, F1, specificity, AUC, CM, ROC
│   ├── xai/gradcam.py               # Grad-CAM + overlays
│   ├── reports/report_generator.py  # python-docx clinical report
│   ├── inference/predictor.py       # one Predictor used by CLI, API and UI
│   └── hub.py                       # push/pull model to Hugging Face Hub
├── api/main.py                      # FastAPI: /health /predict /report, serves web/
├── app/streamlit_app.py             # Streamlit UI (local model or API mode)
├── web/index.html                   # plain HTML + JavaScript client
├── deploy/                          # Dockerfiles, Cloud Build, Cloud Run script
├── notebooks/kaggle_train.ipynb     # train on Kaggle GPU
├── scripts/make_dummy_data.py       # synthetic data for smoke tests
├── tests/                           # pytest: model, Grad-CAM, API
├── docs/ARCHITECTURE.md             # model details & design decisions
├── run_pipeline.sh  docker-compose.yml  requirements.txt  pyproject.toml
```

---

## Serving

```bash
uvicorn api.main:app --reload --port 8000     # http://localhost:8000 (HTML page) and /docs (Swagger)
streamlit run app/streamlit_app.py            # http://localhost:8501
docker compose up --build                     # both, in containers
```

`curl -F "file=@scan.png" http://localhost:8000/predict`

## Deploy to Google Cloud Run

```bash
gcloud auth login && gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
bash deploy/deploy_cloud_run.sh <PROJECT_ID> europe-west3
```

The free tier covers light demo traffic. Delete the services afterwards (`gcloud run services delete thyrocheck-api thyrocheck-ui`).

## Share the model on Hugging Face

```bash
huggingface-cli login
python -m thyrocheck.hub push --repo <hf-user>/thyrocheck-fibonaccinet
```
Set `HF_MODEL_REPO=<hf-user>/thyrocheck-fibonaccinet` on the API and it downloads the model on first request instead of baking it into the image.

---

## Doing it like research

- **Leakage.** Several images often come from the same patient. AUITD has no patient IDs, so the split is image-level; `prepare` removes byte-identical duplicates, but near-duplicates can still inflate scores. Say this in your write-up.
- **Report more than accuracy.** Classes are imbalanced, so look at macro-F1, per-class sensitivity/specificity and AUC (`outputs/metrics/`).
- **Ablations.** Toggle `use_pcb_2_4` / `use_pcb_3_5` to reproduce the paper's ablation idea on thyroid data.
- **Baselines.** Compare against a fine-tuned ImageNet backbone (e.g. EfficientNet-B0, MobileNetV2) under the same split.
- **Repeat.** Run 3–5 seeds (`--set project.seed=…`) and report mean ± std.
- **Check Grad-CAM.** Heatmaps on calipers, text overlays or image borders mean the model is learning shortcuts — crop/mask them.

## References

- S. Roy et al., *Fibonacci-Net: A Lightweight CNN model for Automatic Brain Tumor Classification*, arXiv:2503.13928 (2025).
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- M. Azouz, *Algerian Ultrasound Images Thyroid Dataset (AUITD)*, Kaggle.
