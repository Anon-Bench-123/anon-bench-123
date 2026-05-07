#!/usr/bin/env bash
# Fetch AWS Bedrock batch inference output from S3.
#
# Downloads the latest batch inference results for a given eval_id and epoch.
# Automatically detects the most recent Bedrock job subfolder and syncs its
# contents to the local output directory.
#
# Source: s3://<bucket>/<job_type>_outputs/<eval_id>/<epoch>/<job_id>/
# Destination: ./output/<job_type>/<eval_id>/<epoch>/
#
# Arguments:
#   --job-type    Either 'solver' or 'scorer' (default: 'solver')
#   --eval-id     Evaluation identifier (required)
#   --epoch       Epoch number (required)
#   --output-base Base directory for results (default: './output')
#
# Environment Variables:
#   S3_BUCKET     Override the source S3 bucket (default: l2-bench-batch-inference-tmp)
#
# Usage:
#   bash fetch_bedrock_batch_inference_output.sh --eval-id <eval_id> --epoch <epoch>
#   bash fetch_bedrock_batch_inference_output.sh --job-type scorer --eval-id <eval_id> --epoch <epoch>

set -euo pipefail

# ──────────────────────────────────────────────
# Constants — matches create_batch_inference_job.sh
# ──────────────────────────────────────────────
DEFAULT_S3_BUCKET="l2-bench-batch-inference-tmp"
S3_BUCKET="${S3_BUCKET:-$DEFAULT_S3_BUCKET}"

# ──────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────
usage() {
    echo "Usage: $0 --job-type <solver|scorer> --eval-id <eval_id> --epoch <epoch> [--output-base <base_output_folder>]"
    echo ""
    echo "  --job-type    Either 'solver' or 'scorer' (default: 'solver')"
    echo "  --eval-id     Evaluation ID (e.g., 'eval-001')"
    echo "  --epoch       Epoch number (e.g., '1')"
    echo "  --output-base Base directory for results (default: './output')"
    echo ""
    echo "  Environment Variables:"
    echo "    S3_BUCKET           Override the source S3 bucket (default: ${DEFAULT_S3_BUCKET})"
    exit 1
}

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
JOB_TYPE="solver"
EVAL_ID=""
EPOCH=""
OUTPUT_BASE="./output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-type)    JOB_TYPE="$2";    shift 2 ;;
        --eval-id)     EVAL_ID="$2";     shift 2 ;;
        --epoch)       EPOCH="$2";       shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ "$JOB_TYPE" != "solver" && "$JOB_TYPE" != "scorer" ]]; then
    echo "Error: --job-type must be 'solver' or 'scorer', got: '${JOB_TYPE}'" >&2
    exit 1
fi

# ──────────────────────────────────────────────
# Validate required arguments
# ──────────────────────────────────────────────
missing=()
[[ -z "$EVAL_ID" ]] && missing+=("--eval-id")
[[ -z "$EPOCH"   ]] && missing+=("--epoch")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing required arguments: ${missing[*]}" >&2
    usage
fi

# ──────────────────────────────────────────────
# Derive paths
# ──────────────────────────────────────────────
S3_BASE_URI="s3://${S3_BUCKET}/${JOB_TYPE}_outputs/${EVAL_ID}/${EPOCH}/"
LOCAL_PATH="${OUTPUT_BASE}/${JOB_TYPE}/${EVAL_ID}/${EPOCH}"

# ──────────────────────────────────────────────
# Identify latest Bedrock subfolder
# ──────────────────────────────────────────────
echo "Looking for the latest ${JOB_TYPE} output in ${S3_BASE_URI}..."

# List recursively and sort by timestamp (first two columns of aws s3 ls output)
# and take the last one.
LATEST_LINE=$(aws s3 ls "${S3_BASE_URI}" --recursive | sort | tail -n 1)

if [ -z "$LATEST_LINE" ]; then
    echo "Error: No output files found in ${S3_BASE_URI}"
    exit 1
fi

# aws s3 ls output format: 2024-04-14 07:00:00 1234 path/from/root/file.json
# awk '{print $4}' gets the object path from the bucket root
OBJECT_PATH=$(echo "$LATEST_LINE" | awk '{print $4}')

# In an s3 recursive list, the OBJECT_PATH starts with the filter prefix.
# We want the folder immediately after the epoch folder.
EXPECTED_PREFIX="${JOB_TYPE}_outputs/${EVAL_ID}/${EPOCH}/"
RELATIVE_PATH="${OBJECT_PATH#$EXPECTED_PREFIX}"

# The first segment is the Bedrock Job ID / arbitrary folder
SUBFOLDER=$(echo "$RELATIVE_PATH" | cut -d'/' -f1)

TARGET_S3_URI="${S3_BASE_URI}${SUBFOLDER}/"

# ──────────────────────────────────────────────
# Fetch data
# ──────────────────────────────────────────────
# Capitalize JOB_TYPE for display
JOB_TYPE_CAP="$(echo "${JOB_TYPE:0:1}" | tr '[:lower:]' '[:upper:]')${JOB_TYPE:1}"

echo "──────────────────────────────────────────────"
echo "  Fetching Latest ${JOB_TYPE_CAP} Output"
echo "──────────────────────────────────────────────"
echo "  S3 Source:   ${TARGET_S3_URI}"
echo "  Local Dest:  ${LOCAL_PATH}"
echo "──────────────────────────────────────────────"
echo ""

# Ensure the local directory exists
mkdir -p "${LOCAL_PATH}"

# Sync files from the specific job subfolder to local
# This "flattens" the arbitrary folder out.
aws s3 sync "${TARGET_S3_URI}" "${LOCAL_PATH}"

echo ""
echo "Fetch complete."
