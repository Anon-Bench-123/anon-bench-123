"""Send L2-bench inference tasks to SQS for Lambda-based processing.

Uploads the solver input JSONL to S3 and enqueues one SQS message per task.
Each message triggers a Lambda function that calls Azure OpenAI, Gemini, or Claude
and writes the result back to S3. Used for models not available via AWS Bedrock.

Reads input from: input/solver/<eval_id>/input.jsonl
Uploads input to: s3://<bucket>/inputs/<eval_id>/input.jsonl

Usage:
    uv run send_sqs_messages.py --queue_url <sqs_url> --eval_id <eval_id> \\
        --epoch <epoch> --model_type <openai|gemini|claude> --model <model_name>
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import boto3
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def send_sqs_messages(
    queue_url: str,
    eval_id: str,
    epoch: int,
    model_type: str,
    model: str,
    bucket: str = "l2-bench-openai-google-inference",
    max_retries: int = 3,
    state_machine_arn: str | None = None,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
) -> None:
    """Upload input JSONL to S3 and enqueue SQS messages for inference.

    Args:
        queue_url: SQS queue URL (create one per eval_id/epoch combination).
        eval_id: Evaluation identifier matching the input JSONL path.
        epoch: Epoch number for this inference run.
        model_type: Provider type ('openai', 'gemini', or 'claude').
        model: Model/deployment name (e.g., 'gpt-4o', 'gemini-2.0-flash').
        bucket: S3 bucket for input/output data.
        max_retries: Max retry attempts per task in Lambda.
        state_machine_arn: Optional Step Functions ARN to start after enqueuing.
        dry_run: If True, print actions without uploading or sending.
        task_ids: Specific task IDs to process; defaults to all tasks in input.
    """
    input_path = Path("./input/solver") / eval_id / "input.jsonl"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    all_task_ids = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            all_task_ids.append(json.loads(line)["recordId"])

    if task_ids is not None:
        unknown = set(task_ids) - set(all_task_ids)
        if unknown:
            logger.error(f"Unknown task IDs: {unknown}")
            sys.exit(1)
    else:
        task_ids = all_task_ids

    if dry_run:
        logger.info(f"[dry_run] Would upload {input_path} to s3://{bucket}/inputs/{eval_id}/input.jsonl")
        for task_id in task_ids:
            msg = {"model_type": model_type, "model": model, "eval_id": eval_id, "epoch": epoch, "task_id": task_id, "max_retries": max_retries}
            logger.info(f"[dry_run] Message: {json.dumps(msg)}")
        logger.info(f"[dry_run] Total messages: {len(task_ids)}")
        if state_machine_arn:
            logger.info(f"[dry_run] Would start state machine: {state_machine_arn}")
        return

    print(f"\n{'='*60}")
    print(f"Model:        {model_type}/{model}")
    print(f"Eval ID:      {eval_id}")
    print(f"Epoch:        {epoch}")
    print(f"Tasks:        {len(task_ids)}")
    print(f"Queue:        {queue_url}")
    print(f"S3 Bucket:    {bucket}")
    if state_machine_arn:
        print(f"State Machine: {state_machine_arn}")
    print(f"{'='*60}\n")

    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        logger.info("Aborted by user.")
        return

    s3 = boto3.client("s3")
    s3.upload_file(str(input_path), bucket, f"inputs/{eval_id}/input.jsonl")
    logger.info(f"Uploaded {input_path} to s3://{bucket}/inputs/{eval_id}/input.jsonl")

    sqs = boto3.client("sqs")
    sent = 0
    failed = 0

    for i in range(0, len(task_ids), 10):
        batch = task_ids[i : i + 10]
        entries = [
            {
                "Id": str(uuid.uuid4()),
                "MessageBody": json.dumps({
                    "model_type": model_type,
                    "model": model,
                    "eval_id": eval_id,
                    "epoch": epoch,
                    "task_id": task_id,
                    "max_retries": max_retries,
                }),
            }
            for task_id in batch
        ]
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        sent += len(resp.get("Successful", []))
        failed += len(resp.get("Failed", []))
        for failure in resp.get("Failed", []):
            logger.error(f"Failed to send message: {failure}")

    logger.info(f"Done. Sent: {sent}, Failed: {failed}")

    if state_machine_arn:
        sfn = boto3.client("stepfunctions")
        execution = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"{eval_id}-epoch{epoch}-{uuid.uuid4().hex[:8]}",
            input=json.dumps({"queueUrl": queue_url}),
        )
        logger.info(f"Started state machine execution: {execution['executionArn']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload input JSONL to S3 and enqueue SQS messages for inference.")
    parser.add_argument("--queue_url", type=str, required=True, help="SQS queue URL.")
    parser.add_argument("--eval_id", type=str, required=True, help="Evaluation ID.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type, e.g. openai.")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. gpt-5.4.")
    parser.add_argument("--bucket", type=str, default="l2-bench-openai-google-inference", help="S3 bucket name.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per task.")
    parser.add_argument("--state_machine_arn", type=str, default=None, help="Step Functions state machine ARN. If provided, starts execution after enqueuing.")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without uploading or sending.")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None, help="Specific task IDs to process (default: all tasks).")
    args = parser.parse_args()

    send_sqs_messages(
        queue_url=args.queue_url,
        eval_id=args.eval_id,
        epoch=args.epoch,
        model_type=args.model_type,
        model=args.model,
        bucket=args.bucket,
        max_retries=args.max_retries,
        state_machine_arn=args.state_machine_arn,
        dry_run=args.dry_run,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()
