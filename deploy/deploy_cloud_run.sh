#!/usr/bin/env bash
# Deploy API + Streamlit UI to Google Cloud Run.
#
# One-time setup:
#   gcloud auth login
#   gcloud config set project <PROJECT_ID>
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#
# Usage (from the project root, after training so outputs/models/ exists):
#   bash deploy/deploy_cloud_run.sh <PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:?usage: deploy_cloud_run.sh <PROJECT_ID> [REGION]}"
REGION="${2:-europe-west3}"          # Frankfurt
REPO="thyrocheck"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

test -f outputs/models/fibonacci_net.keras || { echo "Train the model first (outputs/models/fibonacci_net.keras missing)"; exit 1; }

gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

echo ">> building images with Cloud Build"
gcloud builds submit --config deploy/cloudbuild.yaml --substitutions=_REGISTRY="$REGISTRY" .

echo ">> deploying API"
gcloud run deploy thyrocheck-api --image "$REGISTRY/api:latest" --region "$REGION" \
  --memory 2Gi --cpu 1 --allow-unauthenticated --max-instances 2
API_URL=$(gcloud run services describe thyrocheck-api --region "$REGION" --format 'value(status.url)')

echo ">> deploying Streamlit UI"
gcloud run deploy thyrocheck-ui --image "$REGISTRY/ui:latest" --region "$REGION" \
  --memory 512Mi --allow-unauthenticated --max-instances 2 --set-env-vars "API_URL=${API_URL}"
UI_URL=$(gcloud run services describe thyrocheck-ui --region "$REGION" --format 'value(status.url)')

echo
echo "API + HTML page : $API_URL   (docs: $API_URL/docs)"
echo "Streamlit UI    : $UI_URL"
echo "Note: --allow-unauthenticated makes these public. Never upload real patient data to a public demo."
