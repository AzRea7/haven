#!/usr/bin/env bash
set -euo pipefail

# Change into repo root (this script should live there)
cd "$(dirname "$0")"

echo "=== [0] Activate virtualenv (adjust path if needed) ==="
# If you use .venv:
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source haven/Scripts/activate
else
  echo "No .venv found. Make sure your Python env is active before running."
fi

echo "=== [1] Export backend env vars (adjust values) ==="
export DATABASE_URL="sqlite:///haven.db"
export ZILLOW_API_KEY="596249a5-6b4a-43cc-98d2-afae3d9e1709"

# These match how your backend expects model paths in config
export RENT_MODEL_PATH="models/rent_quantiles.pkl"
export FLIP_MODEL_PATH="models/flip_model.pkl"

mkdir -p data/curated models

echo "=== [2] Parallel ingest from HasData Zillow API ==="
# Use YOUR zip codes here; 48009/48363 are just examples from the repo
python scripts/ingest_properties_parallel.py \
  --zip 48009 48363 \
  --workers 4 \
  --limit 300

echo "=== [3] Build features in parallel from DB -> properties.parquet & rent_training.parquet ==="
python scripts/build_features_parallel.py

# At this point you should have:
#   data/curated/properties.parquet
#   data/curated/rent_training.parquet
# produced by build_features_parallel.py

echo "=== [4] Train LightGBM rent quantile model from rent_training.parquet ==="
python scripts/train_rent_quantiles.py

# scripts/train_rent_quantiles.py expects exactly:
#   data/curated/rent_training.parquet
# and uses these feature columns:
#   bedrooms, bathrooms, sqft, zipcode_encoded, property_type_encoded, is_small_unit

echo "=== [5] Train flip model from properties.parquet ==="
python scripts/train_flip.py

# train_flip.py loads data/curated/flip_training.parquet by default
# (you’ll generate that from your own labeled deals). It then saves a calibrated
# classifier to models/flip_model.pkl and stamps feature_names_in_.

echo "=== [6] Optional sanity check: debug a single deal through analyze_deal_with_defaults ==="
python scripts/debug_analyze_deal.py

echo "=== [7] Optional sanity check: debug rent estimator outputs ==="
python scripts/debug_rent_estimator.py

echo "Pipeline complete."
echo "Next steps:"
echo "  1) Start backend:   uvicorn haven.api.main:app --reload --port 8000"
echo "  2) Start frontend:  cd frontend && npm run dev"
