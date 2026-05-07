"""Re-run failed Bedrock batch inference tasks locally.

When AWS Bedrock batch inference fails for specific tasks, this script re-runs
them locally using the Bedrock Converse API or Azure Anthropic API. Results are
written directly to the output JSONL file in Bedrock batch output format.

Reads input from: input/<job_type>/<eval_id>/input.jsonl (solver) or input/<job_type>/<eval_id>/<epoch>/input.jsonl (scorer)
Updates output at: output/<job_type>/<eval_id>/<epoch>/input.jsonl.out

Usage:
    uv run bedrock_inference_local_fallback.py --eval_id <eval_id> --epoch <epoch> \\
        --model <model_id> --task_ids <id1> <id2> ... [--provider bedrock|azure-anthropic]
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import boto3
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


def convert_converse_to_batch_format(converse_response: dict, model_input: dict, record_id: str) -> dict:
    """Convert Bedrock converse() response to batch output format.

    Args:
        converse_response: Response from bedrock_client.converse().
        model_input: Original model input from the task.
        record_id: Task ID.

    Returns:
        Dict in Bedrock batch output format.
    """
    output = converse_response.get("output", {})
    message = output.get("message", {})
    usage = converse_response.get("usage", {})
    metrics = converse_response.get("metrics", {})

    content = []
    for item in message.get("content", []):
        if "reasoningContent" in item:
            content.append({
                "image": None,
                "searchResult": None,
                "citationsContent": None,
                "toolUse": None,
                "guardContent": None,
                "document": None,
                "cachePoint": None,
                "text": None,
                "video": None,
                "audio": None,
                "toolResult": None,
                "reasoningContent": item["reasoningContent"],
            })
        elif "text" in item:
            content.append({
                "image": None,
                "searchResult": None,
                "citationsContent": None,
                "toolUse": None,
                "guardContent": None,
                "document": None,
                "cachePoint": None,
                "text": item["text"],
                "video": None,
                "audio": None,
                "toolResult": None,
                "reasoningContent": None,
            })

    return {
        "modelInput": model_input,
        "modelOutput": {
            "output": {
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": content,
                }
            },
            "stopReason": converse_response.get("stopReason"),
            "trace": None,
            "performanceConfig": None,
            "additionalModelResponseFields": None,
            "usage": {
                "serverToolUsage": {"webSearchRequests": None},
                "cacheReadInputTokens": None,
                "cacheWriteInputTokens": None,
                "cacheDetails": None,
                "inputTokens": usage.get("inputTokens"),
                "outputTokens": usage.get("outputTokens"),
                "totalTokens": usage.get("totalTokens"),
                "cacheWriteInputTokenCount": None,
                "cacheReadInputTokenCount": None,
            },
            "serviceTier": None,
            "metrics": {"latencyMs": metrics.get("latencyMs")},
        },
        "recordId": record_id,
    }


def convert_anthropic_to_batch_format(response: dict, model_input: dict, record_id: str) -> dict:
    """Convert Anthropic messages API response to Bedrock batch output format.

    Args:
        response: Response from Anthropic messages API.
        model_input: Original model input from the task.
        record_id: Task ID.

    Returns:
        Dict in Bedrock batch output format.
    """
    content = []
    for item in response.get("content", []):
        if item.get("type") == "thinking":
            content.append({
                "image": None,
                "searchResult": None,
                "citationsContent": None,
                "toolUse": None,
                "guardContent": None,
                "document": None,
                "cachePoint": None,
                "text": None,
                "video": None,
                "audio": None,
                "toolResult": None,
                "reasoningContent": {
                    "reasoningText": {"text": item.get("thinking", ""), "signature": item.get("signature")},
                },
            })
        elif item.get("type") == "text":
            content.append({
                "image": None,
                "searchResult": None,
                "citationsContent": None,
                "toolUse": None,
                "guardContent": None,
                "document": None,
                "cachePoint": None,
                "text": item.get("text", ""),
                "video": None,
                "audio": None,
                "toolResult": None,
                "reasoningContent": None,
            })

    usage = response.get("usage", {})
    return {
        "modelInput": model_input,
        "modelOutput": {
            "output": {
                "message": {
                    "role": response.get("role", "assistant"),
                    "content": content,
                }
            },
            "stopReason": response.get("stop_reason"),
            "trace": None,
            "performanceConfig": None,
            "additionalModelResponseFields": None,
            "usage": {
                "serverToolUsage": {"webSearchRequests": None},
                "cacheReadInputTokens": None,
                "cacheWriteInputTokens": None,
                "cacheDetails": None,
                "inputTokens": usage.get("input_tokens"),
                "outputTokens": usage.get("output_tokens"),
                "totalTokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0) if usage else None,
                "cacheWriteInputTokenCount": None,
                "cacheReadInputTokenCount": None,
            },
            "serviceTier": None,
            "metrics": {"latencyMs": None},
        },
        "recordId": record_id,
    }


async def run_azure_anthropic_inference(client: AsyncAnthropic, task: dict, model: str) -> dict:
    """Run inference using Azure Anthropic API and convert to Bedrock batch format.

    Args:
        client: Anthropic client configured for Azure endpoint.
        task: Task dict with 'recordId' and 'modelInput'.
        model: Model name (e.g., 'claude-opus-4-7').

    Returns:
        Dict in Bedrock batch output format.
    """
    model_input = task["modelInput"]
    messages = model_input["messages"]
    system = model_input.get("system", [])
    inference_config = model_input.get("inferenceConfig", {})

    anthropic_messages = [
        {"role": m["role"], "content": m["content"][0]["text"]}
        for m in messages
    ]
    kwargs = {"model": model, "messages": anthropic_messages}
    if system:
        kwargs["system"] = "\n".join(s["text"] for s in system)
    kwargs["max_tokens"] = inference_config.get("maxTokens") or 16384
    if inference_config.get("temperature") is not None:
        kwargs["temperature"] = inference_config["temperature"]

    response = await client.messages.create(**kwargs)
    response_dict = json.loads(response.model_dump_json())
    return convert_anthropic_to_batch_format(response_dict, model_input, task["recordId"])


async def run_bedrock_inference(task: dict, model: str) -> dict:
    """Run inference using Bedrock Converse API and convert to batch format.

    Args:
        task: Task dict with 'recordId' and 'modelInput'.
        model: Bedrock model ID (e.g., 'us.anthropic.claude-haiku-4-5-20251001-v1:0').

    Returns:
        Dict in Bedrock batch output format.
    """
    model_input = task["modelInput"]
    messages = model_input["messages"]
    system = model_input.get("system", [])
    inference_config = model_input.get("inferenceConfig", {})

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: bedrock_client.converse(
            modelId=model,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
        ),
    )

    response.pop("ResponseMetadata", None)
    return convert_converse_to_batch_format(response, model_input, task["recordId"])


async def process_task(
    task: dict,
    model: str,
    provider: str,
    azure_client: AsyncAnthropic | None,
) -> tuple[str, dict | None]:
    """Run inference for a single task and return the result.

    Args:
        task: Task dict with 'recordId' and 'modelInput'.
        model: Model ID or name.
        provider: Either 'bedrock' or 'azure-anthropic'.
        azure_client: Anthropic client (used if provider is 'azure-anthropic').

    Returns:
        Tuple of (task_id, result_dict) or (task_id, None) on failure.
    """
    task_id = task["recordId"]
    logger.info(f"Starting task {task_id}")

    try:
        if provider == "azure-anthropic":
            result = await run_azure_anthropic_inference(azure_client, task, model)
        else:
            result = await run_bedrock_inference(task, model)
        logger.info(f"Task {task_id} completed")
        return task_id, result
    except Exception as e:
        logger.error(f"Task {task_id} failed: {type(e).__name__}: {e}")
        return task_id, None


async def main(eval_id: str, epoch: int, model: str, task_ids: list[str], job_type: str, provider: str) -> None:
    """Run inference for specified tasks and update the output JSONL file.

    Args:
        eval_id: Evaluation identifier matching the input/output paths.
        epoch: Epoch number for this inference run.
        model: Bedrock model ID (e.g., 'us.anthropic.claude-haiku-4-5-20251001-v1:0') or Azure Anthropic model name (e.g., 'claude-opus-4-7').
        task_ids: List of task IDs to re-run.
        job_type: Either 'solver' or 'scorer'.
        provider: Either 'bedrock' or 'azure-anthropic'.
    """
    output_path = Path(f"./output/{job_type}/{eval_id}/{epoch}/input.jsonl.out")
    input_path = Path(f"./input/{job_type}/{eval_id}/{epoch}/input.jsonl") if job_type == "scorer" else Path(f"./input/{job_type}/{eval_id}/input.jsonl")

    if not output_path.exists():
        logger.error(f"Output file not found: {output_path}")
        return

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    input_by_id = {}
    with open(input_path) as f:
        for line in f:
            parsed = json.loads(line)
            input_by_id[parsed["recordId"]] = parsed

    missing_inputs = set(task_ids) - set(input_by_id.keys())
    if missing_inputs:
        logger.error(f"Task IDs not found in input: {missing_inputs}")
        return

    tasks_to_run = [input_by_id[tid] for tid in task_ids]
    logger.info(f"Running {len(tasks_to_run)} tasks with model={model} provider={provider}")

    azure_client = None
    if provider == "azure-anthropic":
        azure_client = AsyncAnthropic(
            api_key=os.environ["AZURE_ANTHROPIC_API_KEY"],
            base_url=os.environ["AZURE_ANTHROPIC_API_ENDPOINT"],
        )

    results = await asyncio.gather(*[
        process_task(task, model, provider, azure_client)
        for task in tasks_to_run
    ])

    results_by_id = {tid: result for tid, result in results if result is not None}
    failed_ids = [tid for tid, result in results if result is None]

    if failed_ids:
        logger.warning(f"Failed tasks: {failed_ids}")

    if not results_by_id:
        logger.error("No successful results to write")
        return

    lines = output_path.read_text().splitlines()
    new_lines = []
    replaced = 0
    found_ids = set()

    for line in lines:
        parsed = json.loads(line)
        record_id = parsed.get("recordId")
        if record_id in results_by_id:
            new_lines.append(json.dumps(results_by_id[record_id], ensure_ascii=False))
            replaced += 1
            found_ids.add(record_id)
        else:
            new_lines.append(line)

    # Append any results not found in existing file
    added = 0
    for tid, result in results_by_id.items():
        if tid not in found_ids:
            new_lines.append(json.dumps(result, ensure_ascii=False))
            added += 1

    output_path.write_text("\n".join(new_lines) + "\n")
    logger.info(f"Replaced {replaced}, added {added} lines in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run failed Bedrock batch tasks locally")
    parser.add_argument("--eval_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task_ids", type=str, nargs="+", required=True)
    parser.add_argument("--job_type", type=str, choices=["solver", "scorer"], default="solver")
    parser.add_argument("--provider", type=str, choices=["bedrock", "azure-anthropic"], default="bedrock")
    args = parser.parse_args()

    asyncio.run(main(
        eval_id=args.eval_id,
        epoch=args.epoch,
        model=args.model,
        task_ids=args.task_ids,
        job_type=args.job_type,
        provider=args.provider,
    ))
