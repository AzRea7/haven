#!/usr/bin/env bash
set -euo pipefail

# Activate runtime settings here if needed (migrations, wait-for-db, etc.)
# Example: wait for Postgres if your app needs it
# until pg_isready -h "${DB_HOST:-localhost}" -U "${DB_USER:-postgres}" -d "${DB_NAME:-postgres}" -q; do
#   echo "Waiting for database..." ; sleep 2
# done

echo "Starting Haven API..."
# If your app is a module under src/, run it like:
# exec python -m haven.api
# or if you have a main entry point:
exec uvicorn haven.api:app --host 0.0.0.0 --port 8000