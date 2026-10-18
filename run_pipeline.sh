#!/usr/bin/env bash
# End-to-end pipeline:  download -> prepare -> train -> evaluate -> Grad-CAM -> sample report
#
#   bash run_pipeline.sh            # real Kaggle data (needs Kaggle credentials)
#   bash run_pipeline.sh --dummy    # synthetic images, 20 quick epochs - checks the code runs in ~3 min
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [[ "${1:-}" == "--dummy" ]]; then
  python scripts/make_dummy_data.py --per-class 20
  EXTRA=(--set train.epochs=20 train.learning_rate=1e-3 train.lr_constant_epochs=12 train.monitor=val_auc)
else
  python -m thyrocheck.data.download
  EXTRA=()
fi

python -m thyrocheck.data.prepare
python -m thyrocheck.training.train "${EXTRA[@]}"
python -m thyrocheck.evaluation.evaluate
python -m thyrocheck.xai.gradcam --n 9

SAMPLE=$(python - <<'EOF'
import pandas as pd
from thyrocheck.config import load_config, resolve
df = pd.read_csv(resolve(load_config().data.splits_csv))
print(df[df.split == "test"].iloc[0].path)
EOF
)
python -m thyrocheck.reports.report_generator "$SAMPLE" --patient-id DEMO-001

echo
echo "Done. Look in outputs/: models/, metrics/, figures/, gradcam/, reports/"
echo "Serve it:  uvicorn api.main:app --port 8000     |    streamlit run app/streamlit_app.py"
