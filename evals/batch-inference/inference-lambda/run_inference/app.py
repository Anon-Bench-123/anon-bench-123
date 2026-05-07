import json
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
from botocore.config import Config
from google import genai
from google.genai import types
from anthropic import AnthropicFoundry
from openai import AzureOpenAI

S3_RETRIES = 5
S3_RETRY_DELAY = 2

s3_config = Config(retries={"max_attempts": 3, "mode": "adaptive"})
secrets_client = boto3.client("secretsmanager", region_name="us-east-1")
s3_client = boto3.client("s3", config=s3_config)


def s3_get_object_with_retry(bucket, key):
    for attempt in range(S3_RETRIES):
        try:
            return s3_client.get_object(Bucket=bucket, Key=key)
        except Exception:
            if attempt == S3_RETRIES - 1:
                raise
            time.sleep(S3_RETRY_DELAY * (2 ** attempt))


def s3_put_object_with_retry(bucket, key, body, content_type):
    for attempt in range(S3_RETRIES):
        try:
            return s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        except Exception:
            if attempt == S3_RETRIES - 1:
                raise
            time.sleep(S3_RETRY_DELAY * (2 ** attempt))


def get_secret():
    resp = secrets_client.get_secret_value(SecretId=os.environ["SECRET_ARN"])
    return json.loads(resp["SecretString"])

_secret = None

def get_cached_secret():
    global _secret
    if _secret is None:
        _secret = get_secret()
    return _secret


def lambda_handler(event, context):
    raw_body = event["Records"][0]["body"]
    body = raw_body if isinstance(raw_body, dict) else json.loads(raw_body)
    model_type = body["model_type"]
    model = body["model"]
    run_name = body["eval_id"]
    epoch = body["epoch"]
    task_id = body["task_id"]
    max_retries = body.get("max_retries", 3)

    logger.info(f"Starting task_id={task_id} model={model} model_type={model_type} eval_id={run_name} epoch={epoch}")

    bucket = os.environ["BUCKET_NAME"]

    logger.info(f"Fetching input from s3://{bucket}/inputs/{run_name}/input.jsonl")
    obj = s3_get_object_with_retry(bucket, f"inputs/{run_name}/input.jsonl")
    logger.info("S3 fetch complete")
    row = None
    for line in obj["Body"].read().decode("utf-8").splitlines():
        parsed = json.loads(line)
        if parsed.get("recordId") == task_id:
            row = parsed
            break

    model_input = row["modelInput"]
    messages = model_input["messages"]
    system = model_input.get("system", [])
    inference_config = model_input.get("inferenceConfig", {})
    max_tokens = inference_config.get("maxTokens")
    temperature = inference_config.get("temperature")

    logger.info(f"Found task_id={task_id}, input length={len(messages[0]['content'][0]['text'])} chars")

    logger.info("Fetching secrets")
    secret = get_cached_secret()
    logger.info("Secrets fetched")
    openai_client = AzureOpenAI(
        api_key=secret["AZURE_OPENAI_API_KEY"],
        azure_endpoint=secret["AZURE_OPENAI_API_ENDPOINT"],
        api_version="2025-03-01-preview",
    )

    gemini_client = genai.Client(api_key=secret["GOOGLE_CLOUD_API_KEY"])

    anthropic_client = AnthropicFoundry(
        api_key=secret["AZURE_ANTHROPIC_API_KEY"],
        base_url=secret["AZURE_ANTHROPIC_API_ENDPOINT"],
    )

    response = None
    if model_type == "openai":
        for attempt in range(max_retries):
            try:
                logger.info(f"OpenAI API call attempt {attempt + 1}/{max_retries}")
                input_text = messages[0]["content"][0]["text"]
                kwargs = {"model": model, "input": input_text}
                if system:
                    kwargs["instructions"] = "\n".join(s["text"] for s in system)
                if max_tokens is not None:
                    kwargs["max_output_tokens"] = max_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = openai_client.responses.create(**kwargs)
                logger.info("OpenAI API call succeeded")
                break
            except Exception as e:
                logger.error(f"OpenAI API call failed attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
    elif model_type == "gemini":
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API call attempt {attempt + 1}/{max_retries}")
                contents = messages[0]["content"][0]["text"]
                config_kwargs = {}
                if system:
                    config_kwargs["system_instruction"] = "\n".join(s["text"] for s in system)
                if max_tokens is not None:
                    config_kwargs["max_output_tokens"] = max_tokens
                if temperature is not None:
                    config_kwargs["temperature"] = temperature
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                logger.info("Gemini API call succeeded")
                break
            except Exception as e:
                logger.error(f"Gemini API call failed attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
    elif model_type == "claude":
        for attempt in range(max_retries):
            try:
                logger.info(f"Claude API call attempt {attempt + 1}/{max_retries}")
                anthropic_messages = [
                    {"role": m["role"], "content": m["content"][0]["text"]}
                    for m in messages
                ]
                kwargs = {"model": model, "messages": anthropic_messages}
                if system:
                    kwargs["system"] = "\n".join(s["text"] for s in system)
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = 4096
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = anthropic_client.messages.create(**kwargs)
                logger.info("Claude API call succeeded")
                break
            except Exception as e:
                logger.error(f"Claude API call failed attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logger.info(f"Writing output to s3://{bucket}/outputs/{run_name}/{epoch}/{task_id}/output.json")
    s3_put_object_with_retry(
        bucket,
        f"outputs/{run_name}/{epoch}/{task_id}/output.json",
        response.model_dump_json(),
        "application/json",
    )
    logger.info(f"Task {task_id} completed successfully")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "lambda completed"}),
    }
