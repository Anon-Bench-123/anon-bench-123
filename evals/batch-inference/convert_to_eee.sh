#!/bin/bash
# Convert Inspect AI evaluation logs to Every Eval Ever (EEE) format.
#
# Processes an Inspect AI log file through two stages:
# 1. Fix known issues (negative working_time, replay/ model prefix)
# 2. Convert to EEE format using the official every_eval_ever converter
#
# Output is organized by benchmark/developer/model for EEE submission.
#
# Arguments:
#   log_path       Path to Inspect AI evaluation log file (required)
#   output_dir     Base directory for EEE output (default: eee_output)
#   benchmark_name Benchmark name in output path (default: l2-bench)
#
# Output structure:
#   <output_dir>/<benchmark_name>/<developer>/<model>/<uuid>.json
#   <output_dir>/<benchmark_name>/<developer>/<model>/<uuid>_samples.jsonl
#
# Usage:
#   bash convert_to_eee.sh logs/eval-001/log.json
#   bash convert_to_eee.sh logs/eval-001/log.json eee_output l2-bench
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <log_path> [output_dir] [benchmark_name]"
    exit 1
fi

LOG_PATH="$1"
OUTPUT_DIR="${2:-eee_output}"
BENCHMARK_NAME="${3:-l2-bench}"

if [ ! -f "$LOG_PATH" ]; then
    echo "Error: File not found: $LOG_PATH"
    exit 1
fi

# Fix inspect log (negative working_time, strip replay/ prefix)
echo "Fixing inspect log..."
uv run fix_inspect_log.py "$LOG_PATH"

# Run official EEE converter
echo "Running EEE converter..."
TEMP_DIR=$(mktemp -d)
uv run python -m every_eval_ever.converters.inspect \
    --log_path "$LOG_PATH" \
    --output_dir "$TEMP_DIR" \
    --source_organization_name OUP \
    --evaluator_relationship third_party

# Reorganize output
echo "Reorganizing output..."
find "$TEMP_DIR" -name "*.json" | while read json_file; do
    model_id=$(uv run python -c "import json; print(json.load(open('$json_file'))['model_info']['id'])")
    developer="${model_id%%/*}"
    model="${model_id##*/}"

    dest_dir="$OUTPUT_DIR/$BENCHMARK_NAME/$developer/$model"
    mkdir -p "$dest_dir"

    uuid=$(basename "$json_file" .json)

    json_basename=$(basename "$json_file")
    cp "$json_file" "$dest_dir/$json_basename"
    echo "Wrote $dest_dir/$json_basename"

    jsonl_file="${json_file%.json}_samples.jsonl"
    if [ -f "$jsonl_file" ]; then
        jsonl_basename=$(basename "$jsonl_file")
        cp "$jsonl_file" "$dest_dir/$jsonl_basename"
        echo "Wrote $dest_dir/$jsonl_basename"
    fi
done

rm -rf "$TEMP_DIR"
echo "Done"
