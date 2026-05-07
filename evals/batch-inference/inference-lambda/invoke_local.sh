#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env.local ]; then
  export $(grep -v '^#' .env.local | xargs)
fi

sam local invoke RunInferenceFunction \
  -e events/event.json \
  --profile "${AWS_PROFILE:-default}" \
  --env-vars <(echo "{\"RunInferenceFunction\": {\"BUCKET_NAME\": \"${BUCKET_NAME}\", \"SECRET_ARN\": \"${SECRET_ARN}\"}}")
