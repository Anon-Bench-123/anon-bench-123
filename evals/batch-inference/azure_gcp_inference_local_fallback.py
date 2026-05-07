"""Re-run failed Azure/GCP inference tasks locally.

When Lambda-based batch inference fails for specific tasks, this script re-runs
them locally using the same Azure OpenAI, Gemini, or Claude APIs. Results are
written to S3 in the same format as Lambda output for downstream processing.

Reads input from: input/solver/<eval_id>/input.jsonl
Writes output to: s3://<bucket>/outputs/<eval_id>/<epoch>/<task_id>/output.json

Usage:
    uv run azure_gcp_inference_local_fallback.py --eval_id <eval_id> --epoch <epoch> \\
        --model_type <openai|gemini|claude> --model <model_name> --task_ids <id1> <id2> ...
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import boto3
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv()

s3_client = boto3.client("s3")


async def run_openai_inference(client: AsyncAzureOpenAI | AsyncOpenAI, task: dict, model: str) -> dict:
    """Run inference using Azure OpenAI or OpenRouter Responses API.

    Args:
        client: Azure OpenAI or OpenRouter client.
        task: Task dict with 'recordId' and 'modelInput'.
        model: Model name (e.g., 'gpt-5.4').

    Returns:
        Raw API response as dict.
    """
    model_input = task["modelInput"]
    messages = model_input["messages"]
    system = model_input.get("system", [])
    inference_config = model_input.get("inferenceConfig", {})

    input_text = messages[0]["content"][0]["text"]
    kwargs = {"model": model, "input": input_text}
    if system:
        kwargs["instructions"] = "\n".join(s["text"] for s in system)
    if inference_config.get("maxTokens") is not None:
        kwargs["max_output_tokens"] = inference_config["maxTokens"]
    if inference_config.get("temperature") is not None:
        kwargs["temperature"] = inference_config["temperature"]

    response = await client.responses.create(**kwargs)
    return json.loads(response.model_dump_json())


async def run_gemini_inference(client: genai.Client, task: dict, model: str) -> dict:
    """Run inference using Google Gemini API.

    Args:
        client: Google GenAI client.
        task: Task dict with 'recordId' and 'modelInput'.
        model: Model name (e.g., 'gemini-3.1-pro-preview').

    Returns:
        Raw API response as dict.
    """
    model_input = task["modelInput"]
    messages = model_input["messages"]
    system = model_input.get("system", [])
    inference_config = model_input.get("inferenceConfig", {})

    contents = messages[0]["content"][0]["text"]
    config_kwargs = {}
    if system:
        config_kwargs["system_instruction"] = "\n".join(s["text"] for s in system)
    if inference_config.get("maxTokens") is not None:
        config_kwargs["max_output_tokens"] = inference_config["maxTokens"]
    if inference_config.get("temperature") is not None:
        config_kwargs["temperature"] = inference_config["temperature"]

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return json.loads(response.model_dump_json())


async def run_claude_inference(client: AsyncAnthropic, task: dict, model: str) -> dict:
    """Run inference using Azure Anthropic API.

    Args:
        client: Anthropic client configured for Azure endpoint.
        task: Task dict with 'recordId' and 'modelInput'.
        model: Model name (e.g., 'claude-opus-4-7').

    Returns:
        Raw API response as dict.
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
    kwargs["max_tokens"] = inference_config.get("maxTokens") or 4096
    if inference_config.get("temperature") is not None:
        kwargs["temperature"] = inference_config["temperature"]

    response = await client.messages.create(**kwargs)
    return json.loads(response.model_dump_json())


async def process_task(
    task: dict,
    model_type: str,
    model: str,
    eval_id: str,
    epoch: int,
    bucket: str,
    openai_client: AsyncAzureOpenAI | AsyncOpenAI | None,
    gemini_client: genai.Client | None,
    claude_client: AsyncAnthropic | None,
) -> None:
    """Run inference for a single task and upload result to S3.

    Args:
        task: Task dict with 'recordId' and 'modelInput'.
        model_type: Provider type ('openai', 'gemini', or 'claude').
        model: Model name.
        eval_id: Evaluation identifier.
        epoch: Epoch number.
        bucket: S3 bucket for output.
        openai_client: OpenAI client (used if model_type is 'openai').
        gemini_client: Gemini client (used if model_type is 'gemini').
        claude_client: Anthropic client (used if model_type is 'claude').
    """
    task_id = task["recordId"]
    logger.info(f"Starting task {task_id}")

    try:
        if model_type == "openai":
            result = await run_openai_inference(openai_client, task, model)
        elif model_type == "gemini":
            result = await run_gemini_inference(gemini_client, task, model)
        elif model_type == "claude":
            result = await run_claude_inference(claude_client, task, model)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        s3_key = f"outputs/{eval_id}/{epoch}/{task_id}/output.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(result),
            ContentType="application/json",
        )
        logger.info(f"Task {task_id} completed, saved to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Task {task_id} failed: {type(e).__name__}: {e}")


async def main(
    eval_id: str,
    epoch: int,
    model_type: str,
    model: str,
    task_ids: list[str],
    bucket: str,
    use_open_router: bool = False,
) -> None:
    """Run inference for specified tasks and upload results to S3.

    Args:
        eval_id: Evaluation identifier matching the input JSONL path.
        epoch: Epoch number for this inference run.
        model_type: Provider type ('openai', 'gemini', or 'claude').
        model: Model name (e.g., 'gpt-5.4', 'gemini-3.1-pro-preview', 'claude-opus-4-7').
        task_ids: List of task IDs to re-run.
        bucket: S3 bucket for output data.
        use_open_router: If True, use OpenRouter instead of Azure OpenAI.
    """
    input_path = Path("./input/solver") / eval_id / "input.jsonl"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    tasks_to_run = []
    with open(input_path) as f:
        for line in f:
            parsed = json.loads(line)
            if parsed["recordId"] in task_ids:
                tasks_to_run.append(parsed)

    found_ids = {t["recordId"] for t in tasks_to_run}
    missing = set(task_ids) - found_ids
    if missing:
        logger.warning(f"Task IDs not found in input: {missing}")

    logger.info(f"Running {len(tasks_to_run)} tasks with model_type={model_type} model={model}")

    openai_client = None
    gemini_client = None
    claude_client = None

    if model_type == "openai":
        if use_open_router:
            openai_client = AsyncOpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            openai_client = AsyncAzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
                api_version="2025-03-01-preview",
            )
    elif model_type == "gemini":
        gemini_client = genai.Client(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])
    elif model_type == "claude":
        claude_client = AsyncAnthropic(
            api_key=os.environ["AZURE_ANTHROPIC_API_KEY"],
            base_url=os.environ["AZURE_ANTHROPIC_API_ENDPOINT"],
        )

    await asyncio.gather(*[
        process_task(
            task=task,
            model_type=model_type,
            model=model,
            eval_id=eval_id,
            epoch=epoch,
            bucket=bucket,
            openai_client=openai_client,
            gemini_client=gemini_client,
            claude_client=claude_client,
        )
        for task in tasks_to_run
    ])

    logger.info("All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference locally for failed tasks")
    parser.add_argument("--eval_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["openai", "gemini", "claude"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task_ids", type=str, nargs="+", required=True)
    parser.add_argument("--bucket", type=str, default="l2-bench-openai-google-inference")
    parser.add_argument("--use_open_router", action="store_true", help="Use OpenRouter for OpenAI models")
    args = parser.parse_args()

    asyncio.run(main(
        eval_id=args.eval_id,
        epoch=args.epoch,
        model_type=args.model_type,
        model=args.model,
        task_ids=args.task_ids,
        bucket=args.bucket,
        use_open_router=args.use_open_router,
    ))
