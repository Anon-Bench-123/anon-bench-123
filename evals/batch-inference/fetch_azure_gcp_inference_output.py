"""Fetch Lambda inference outputs from S3 and convert to Bedrock format.

Downloads per-task output files written by the inference Lambda (OpenAI, Gemini,
or Claude) and converts them to AWS Bedrock Converse format for downstream
processing by the scorer and replay evaluation scripts.

Reads task IDs from: input/solver/<eval_id>/input.jsonl
Fetches outputs from: s3://<bucket>/outputs/<eval_id>/<epoch>/<task_id>/output.json
Writes combined output to: output/solver/<eval_id>/<epoch>/input.jsonl.out

Usage:
    uv run fetch_azure_gcp_inference_output.py --eval_id <eval_id> --epoch <epoch> \\
        --model_type <openai|gemini|claude>
"""

import argparse
import json
import sys
from pathlib import Path

import boto3
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def fetch_inference_output(
    eval_id: str,
    epoch: int,
    model_type: str,
    bucket: str = "l2-bench-openai-google-inference",
) -> None:
    """Fetch inference outputs from S3 and convert to Bedrock Converse format.

    Args:
        eval_id: Evaluation identifier matching the input JSONL path.
        epoch: Epoch number for this inference run.
        model_type: Provider type ('openai', 'gemini', or 'claude').
        bucket: S3 bucket containing the Lambda output files.
    """
    input_path = Path("./input/solver") / eval_id / "input.jsonl"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    tasks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    output_dir = Path("./output/solver") / eval_id / str(epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "input.jsonl.out"

    s3 = boto3.client("s3")
    written = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for task in tasks:
            task_id = task["recordId"]
            key = f"outputs/{eval_id}/{epoch}/{task_id}/output.json"
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                response = json.loads(obj["Body"].read().decode("utf-8"))
            except Exception as e:
                logger.warning(f"Failed to fetch {key}: {e}")
                failed += 1
                continue

            if model_type == "openai":
                message = response["output"][0]
                text = message["content"][0]["text"]
                role = message["role"]
                usage = response["usage"]
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
            elif model_type == "gemini":
                candidate = response["candidates"][0]
                text = candidate["content"]["parts"][0]["text"]
                role = candidate["content"]["role"]
                usage = response["usage_metadata"]
                input_tokens = usage["prompt_token_count"]
                output_tokens = usage["candidates_token_count"]
            elif model_type == "claude":
                text = response["content"][0]["text"]
                role = response["role"]
                usage = response["usage"]
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]

            record = {
                "recordId": task_id,
                "modelInput": task["modelInput"],
                "modelOutput": {
                    "output": {
                        "message": {
                            "role": role,
                            "content": [{"text": text}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                    },
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"Written: {written}, Failed: {failed} → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OpenAI/Gemini inference outputs from S3 and convert to Bedrock Converse format."
    )
    parser.add_argument("--eval_id", type=str, required=True, help="Evaluation ID.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["openai", "gemini", "claude"],
        help="Model type.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="l2-bench-openai-google-inference",
        help="S3 bucket name.",
    )
    args = parser.parse_args()

    fetch_inference_output(
        eval_id=args.eval_id,
        epoch=args.epoch,
        model_type=args.model_type,
        bucket=args.bucket,
    )


if __name__ == "__main__":
    main()
