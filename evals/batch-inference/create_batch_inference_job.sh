#!/usr/bin/env bash
# Create an AWS Bedrock batch inference job for L2-bench evaluation.
#
# Uploads the input JSONL to S3 and creates a Bedrock batch inference job.
# Used for both solver (task completion) and scorer (judge) inference.
#
# Usage:
#   bash create_batch_inference_job.sh --job-type solver --eval-id <eval_id> --epoch <epoch> \
#       --model <model_id> --input input/solver/<eval_id>/input.jsonl

set -euo pipefail

# ──────────────────────────────────────────────
# Constants — change these to reconfigure the
# target bucket and IAM role.
# ──────────────────────────────────────────────
DEFAULT_S3_BUCKET="l2-bench-batch-inference-tmp"
DEFAULT_SERVICE_ROLE_NAME="l2-bench-batch-inference"

S3_BUCKET="${S3_BUCKET:-$DEFAULT_S3_BUCKET}"
SERVICE_ROLE_NAME="${SERVICE_ROLE_NAME:-$DEFAULT_SERVICE_ROLE_NAME}"

# ──────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────
usage() {
    echo "Usage: $0 --job-type <solver|scorer> --eval-id <eval_id> --epoch <epoch> --model <model_id> --input <jsonl_file>"
    echo ""
    echo "  --job-type   Either 'solver' or 'scorer'"
    echo "  --eval-id    Evaluation ID (matches the Python scripts)"
    echo "  --epoch      Epoch number (matches the Python scripts)"
    echo "  --model      Bedrock model ID"
    echo "  --input      Path to the local input JSONL file"
    echo ""
    echo "  Environment Variables:"
    echo "    S3_BUCKET           Override the target S3 bucket (default: ${DEFAULT_S3_BUCKET})"
    echo "    SERVICE_ROLE_NAME   Override the IAM service role name (default: ${DEFAULT_SERVICE_ROLE_NAME})"
    exit 1
}

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
JOB_TYPE=""
EVAL_ID=""
EPOCH=""
MODEL=""
JSONL_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-type)  JOB_TYPE="$2";  shift 2 ;;
        --eval-id)   EVAL_ID="$2";   shift 2 ;;
        --epoch)     EPOCH="$2";     shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --input)     JSONL_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# ──────────────────────────────────────────────
# Validate required arguments
# ──────────────────────────────────────────────
missing=()
[[ -z "$JOB_TYPE"   ]] && missing+=("--job-type")
[[ -z "$EVAL_ID"    ]] && missing+=("--eval-id")
[[ -z "$EPOCH"      ]] && missing+=("--epoch")
[[ -z "$MODEL"      ]] && missing+=("--model")
[[ -z "$JSONL_FILE" ]] && missing+=("--input")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing required arguments: ${missing[*]}" >&2
    usage
fi

# ──────────────────────────────────────────────
# Validate job type
# ──────────────────────────────────────────────
if [[ "$JOB_TYPE" != "solver" && "$JOB_TYPE" != "scorer" ]]; then
    echo "Error: --job-type must be 'solver' or 'scorer', got: '${JOB_TYPE}'" >&2
    exit 1
fi

# ──────────────────────────────────────────────
# Validate the input file exists
# ──────────────────────────────────────────────
if [ ! -f "$JSONL_FILE" ]; then
    echo "Error: JSONL file not found: $JSONL_FILE" >&2
    exit 1
fi

JSONL_FILE_NAME="$(basename "$JSONL_FILE")"

# ──────────────────────────────────────────────
# Derive names / ARNs
# ──────────────────────────────────────────────
TIMESTAMP="$(date -u '+%Y%m%d%H%M%S')"
JOB_NAME="${EVAL_ID}-epoch${EPOCH}-${TIMESTAMP}"

# Resolve the service-role ARN from the account that owns the current credentials
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SERVICE_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SERVICE_ROLE_NAME}"

INPUT_S3_URI="s3://${S3_BUCKET}/${JOB_TYPE}_jobs/${EVAL_ID}/${EPOCH}/${JSONL_FILE_NAME}"
OUTPUT_S3_URI="s3://${S3_BUCKET}/${JOB_TYPE}_outputs/${EVAL_ID}/${EPOCH}/"

# ──────────────────────────────────────────────
# Confirmation Dialog
# ──────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "  Review Batch Inference Job Details"
echo "──────────────────────────────────────────────"
echo "  Job Type:    ${JOB_TYPE}"
echo "  Eval ID:     ${EVAL_ID}"
echo "  Epoch:       ${EPOCH}"
echo "  Job Name:    ${JOB_NAME}"
echo "  Model:       ${MODEL}"
echo "  Input Local: ${JSONL_FILE}"
echo "  Input S3:    ${INPUT_S3_URI}"
echo "  Output S3:   ${OUTPUT_S3_URI}"
echo "  Role ARN:    ${SERVICE_ROLE_ARN}"
echo "──────────────────────────────────────────────"
echo ""

read -p "Proceed with upload and job creation? (y/N) " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ──────────────────────────────────────────────
# Upload input data to S3
# ──────────────────────────────────────────────
echo "Uploading input data to ${INPUT_S3_URI} ..."
aws s3 cp "$JSONL_FILE" "$INPUT_S3_URI"
echo "Upload complete."

# ──────────────────────────────────────────────
# Create the Bedrock batch inference job
# ──────────────────────────────────────────────
echo ""
echo "Creating Bedrock batch inference job: ${JOB_NAME}"
echo "  Model:       ${MODEL}"
echo "  Input:       ${INPUT_S3_URI}"
echo "  Output:      ${OUTPUT_S3_URI}"
echo "  Role ARN:    ${SERVICE_ROLE_ARN}"
echo ""

JOB_ARN="$(aws bedrock create-model-invocation-job \
    --region "$AWS_REGION" \
    --query jobArn \
    --output text \
    --cli-input-json "{
        \"jobName\": \"${JOB_NAME}\",
        \"modelId\": \"${MODEL}\",
        \"roleArn\": \"${SERVICE_ROLE_ARN}\",
        \"modelInvocationType\": \"Converse\",
        \"inputDataConfig\": {\"s3InputDataConfig\": {\"s3Uri\": \"${INPUT_S3_URI}\"}},
        \"outputDataConfig\": {\"s3OutputDataConfig\": {\"s3Uri\": \"${OUTPUT_S3_URI}\"}},
        \"tags\": [{\"key\": \"eval_id\", \"value\": \"${EVAL_ID}\"}, {\"key\": \"epoch\", \"value\": \"${EPOCH}\"}]
    }")"

echo "Batch inference job created successfully."
echo "Job ARN: ${JOB_ARN}"