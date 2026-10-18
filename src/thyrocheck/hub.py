"""Hugging Face Hub helpers: share the trained model and pull it at deploy time.

Push (after training):
    huggingface-cli login
    python -m thyrocheck.hub push --repo <your-username>/thyrocheck-fibonaccinet

Pull (e.g. inside the API container):
    python -m thyrocheck.hub pull --repo <your-username>/thyrocheck-fibonaccinet
The API also pulls automatically on first request if the env var HF_MODEL_REPO is set
and the model file is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thyrocheck.config import load_config, resolve

FILES = ["fibonacci_net.keras", "class_names.json"]


def push(repo: str, cfg) -> None:
    from huggingface_hub import HfApi
    model_dir = resolve(cfg.train.model_path).parent
    api = HfApi()
    api.create_repo(repo, exist_ok=True, private=True)
    for f in FILES:
        api.upload_file(path_or_fileobj=str(model_dir / f), path_in_repo=f, repo_id=repo)
    metrics = resolve(cfg.train.output_dir) / "metrics" / "test_metrics.json"
    card = ["---", "tags: [medical-imaging, thyroid, ultrasound, keras, grad-cam]", "---",
            "# ThyroCheck AI - FibonacciNet", "",
            "Thyroid ultrasound classifier (research prototype, not for clinical use).", ""]
    if metrics.exists():
        card += ["## Test metrics", "```json", json.dumps(json.loads(metrics.read_text()), indent=2), "```"]
    api.upload_file(path_or_fileobj="\n".join(card).encode(), path_in_repo="README.md", repo_id=repo)
    print(f"[hub] pushed to https://huggingface.co/{repo}")


def pull(repo: str, cfg) -> Path:
    from huggingface_hub import hf_hub_download
    model_dir = resolve(cfg.train.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        hf_hub_download(repo_id=repo, filename=f, local_dir=str(model_dir))
    print(f"[hub] downloaded model to {model_dir}")
    return model_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    (push if args.action == "push" else pull)(args.repo, cfg)


if __name__ == "__main__":
    main()
