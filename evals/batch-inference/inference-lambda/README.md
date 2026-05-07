# inference-lambda

SAM-based Lambda function for running batch inference against Azure OpenAI. Triggered by SQS, reads input from S3, writes output back to S3.

## How it works

1. A message is enqueued to SQS with `model_type`, `model`, `run_name`, `epoch`, and `task_id`.
2. The Lambda reads the corresponding row from `s3://<bucket>/inputs/<run_name>/input.jsonl` (matched by `recordId`).
3. It calls the Azure OpenAI Responses API and writes the full response to `s3://<bucket>/outputs/<run_name>/<epoch>/<task_id>/output.json`.

**Important:** Create a separate SQS queue for each `(eval_id, epoch)` combination. The state machine monitors a single queue and waits for it to drain, so mixing different runs in the same queue will cause incorrect completion detection.

```bash
aws sqs create-queue --queue-name l2-bench-<eval_id>-epoch<epoch>
```

## Prerequisites

- [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- Docker (for local invoke)
- AWS credentials with access to the S3 bucket and Secrets Manager secret

## Local testing

Copy `.env.local.example` to `.env.local` and fill in your values:

```
BUCKET_NAME=l2-bench-openai-inference
SECRET_ARN=arn:aws:secretsmanager:us-east-1:<account>:secret:<secret-name>
AWS_PROFILE=default
```

Edit `events/event.json` to set the test input, then run:

```bash
sam build
./invoke_local.sh
```

The event body fields:

| Field | Description |
|---|---|
| `model_type` | Must be `openai` |
| `model` | Azure OpenAI deployment name |
| `run_name` | Matches the input JSONL prefix in S3 |
| `epoch` | Used in the output S3 path |
| `task_id` | Matches `recordId` in the input JSONL |
| `max_retries` | Optional, default `3` |

## Deployment

```bash
sam build
sam deploy
```

Parameters are pre-configured in `samconfig.toml`. The deploy will prompt for changeset confirmation before applying.

## Logs

```bash
sam logs -n RunInferenceFunction --stack-name l2-bench-inference-lambda --tail
```

## Cleanup

```bash
sam delete --stack-name l2-bench-inference-lambda
```
