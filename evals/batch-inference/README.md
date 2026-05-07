# L2 Bench Batch Inference

This repository contains tools for running batch inference on L2-bench tasks using AWS Bedrock.

## Setup

### 1. Install Dependencies
Ensure you have [uv](https://docs.astral.sh/uv/) installed, then run:

```bash
uv sync
```

### 2. Configure Environment
Create a .env file with the following variables:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1
```

### 3. Create Symlinks
Create symbolic links to the task data and resources.

```bash
ln -s /path/to/elt-bench-data/l2-bench_tasks.csv l2-bench_tasks.csv
ln -s /path/to/elt-bench-data/resources_for_tasks resources_for_tasks
```

### 4. Inference Lambda Setup (for Azure/GCP models)
For running inference with OpenAI, Gemini, or Claude models, deploy the Lambda function in `inference-lambda/`. See `inference-lambda/README.md` for SAM deployment instructions.

Create an AWS Secrets Manager secret with the following keys (only include the ones you need):

- `AZURE_OPENAI_API_KEY` — Azure OpenAI API key
- `AZURE_OPENAI_API_ENDPOINT` — Azure OpenAI endpoint URL
- `GOOGLE_CLOUD_API_KEY` — Google Cloud API key for Gemini
- `AZURE_ANTHROPIC_API_KEY` — Azure Anthropic API key
- `AZURE_ANTHROPIC_API_ENDPOINT` — Azure Anthropic endpoint URL

**Important:** Create a separate SQS queue for each `(eval_id, epoch)` combination. The state machine monitors a single queue and waits for it to drain, so mixing different runs in the same queue will cause incorrect completion detection.

## Workflow

### 5. Generate Solver Input
Generate the JSONL input file for the solver model.

**Note:** Even for Azure/GCP inference, use this script to generate AWS Bedrock format first. The downstream Azure/GCP inference code expects input in this format.

```bash
uv run solver_model_batch_inference_input.py --eval_id <eval_id> [--inference_config '<json>']
```

Options:
- `--inference_config`: JSON string for model parameters, e.g. `'{"maxTokens": 4096, "temperature": 0.7}'`
- `--test_number`: Only output the first N tasks
- `--task_ids`: List of specific task IDs to include

*Output: `input/solver/<eval_id>/input.jsonl`*

### 6. Create Solver Batch Job

#### AWS Bedrock
Create a Bedrock batch inference job for the solver.

```bash
bash create_batch_inference_job.sh --job-type solver --eval-id <eval_id> --epoch <epoch> --model <model_id> --input input/solver/<eval_id>/input.jsonl
```

The `inference_config` from step 5 is embedded in the input JSONL and used automatically by Bedrock.

#### Azure/GCP (OpenAI, Gemini, Claude)
For non-Bedrock models, use the Lambda-based inference pipeline:

1. Send tasks to SQS queue:
```bash
uv run send_sqs_messages.py --queue_url <sqs_url> --eval_id <eval_id> --epoch <epoch> --model_type <openai|gemini|claude> --model <model_name>
```

2. After completion, fetch results:
```bash
uv run fetch_azure_gcp_inference_output.py --eval_id <eval_id> --epoch <epoch> --model_type <openai|gemini|claude>
```

This fetches per-task outputs from S3 and converts them to Bedrock Converse format at `output/solver/<eval_id>/<epoch>/input.jsonl.out`.

### 7. Fetch Solver Output
After the solver job completes, fetch the results from S3.

```bash
bash fetch_bedrock_batch_inference_output.sh --job-type solver --eval-id <eval_id> --epoch <epoch>
```
*(Note: `--job-type solver` is the default.)*
*Output: `output/solver/<eval_id>/<epoch>/...`*

### 8. Local Fallback for Failed Tasks
If some tasks fail during batch inference, re-run them locally.

#### Bedrock models
```bash
uv run bedrock_inference_local_fallback.py --eval_id <eval_id> --epoch <epoch> --model <model_id> --task_ids <id1> <id2> ...
```

Options:
- `--job_type`: `solver` (default) or `scorer`
- `--provider`: `bedrock` (default) or `azure-anthropic`

#### Azure/GCP models
```bash
uv run azure_gcp_inference_local_fallback.py --eval_id <eval_id> --epoch <epoch> --model_type <openai|gemini|claude> --model <model_name> --task_ids <id1> <id2> ...
```

### 9. Generate Scorer Input
Generate the JSONL input file for the scorer (judge) model based on the solver's output.

```bash
uv run scorer_model_batch_inference_input.py --eval_id <eval_id> --epoch <epoch> [--inference_config '<json>']
```

Options:
- `--inference_config`: JSON string for model parameters, e.g. `'{"maxTokens": 4096, "temperature": 0}'`
- `--judge_prompt_version`: Version of judge prompt to use (default: `v1`)

*Output: `input/scorer/<eval_id>/<epoch>/input.jsonl`*

### 10. Create Scorer Batch Job
Create a Bedrock batch inference job for the scorer.

```bash
bash create_batch_inference_job.sh --job-type scorer --eval-id <eval_id> --epoch <epoch> --model <model_id> --input input/scorer/<eval_id>/<epoch>/input.jsonl
```

### 11. Fetch Scorer Output
After the scorer job completes, fetch the results from S3.

```bash
bash fetch_bedrock_batch_inference_output.sh --job-type scorer --eval-id <eval_id> --epoch <epoch>
```
*Output: `output/scorer/<eval_id>/<epoch>/...`*

### 12. Run Replay Evaluation
Run the final evaluation by replaying the solver and scorer outputs through Inspect AI.

```bash
uv run replay_eval_combine_epoch.py --eval-id <eval_id> --solver-model-name <solver_model> --scorer-model-name <scorer_model>
```

Options:
- `--filter-published` / `--no-filter-published`: Filter to published tasks only (default: enabled)

The script auto-detects epochs from `output/solver/<eval_id>/` and combines results across all epochs.

### 13. Convert to EEE
Convert the Inspect AI evaluation log to Every Eval Ever format.

```bash
bash convert_to_eee.sh <path_to_eval_log> [output_dir] [benchmark_name]
```

Arguments:
- `path_to_eval_log`: Path to the `.eval` log file from replay evaluation
- `output_dir`: Output directory (default: `eee_output`)
- `benchmark_name`: Benchmark name for output path (default: `l2-bench`)

The script automatically fixes log issues (negative `working_time`, `replay/` model prefix) before running the EEE converter.
