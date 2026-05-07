# Anon-Bench

Response generation and LLM-as-judge scoring for Anon-Bench tasks. This directory contains:
1. **Source code** (`src/elt_bench_eval/`) — the inspect-ai solver and scorer pipeline


In addition, you can find the following datasets in Huggingface
1. [Anon-Bench tasks](https://huggingface.co/datasets/anon-bench-org/anon-bench-123) - tasks and resources for tasks used in Anon-Bench
2. [Evaluation logs](https://huggingface.co/datasets/anon-bench-org/anon-bench-inspect-ai-log) — scored inspect-ai `.eval` log files for 9 models (1,000 tasks x 3 epochs each)
3. [Raw outputs](https://huggingface.co/datasets/anon-bench-org/anon-bench-raw-output) - raw inference outputs from LLMs that are fed to the inspect-ai log generation pipeline
4. [Test run outputs](https://huggingface.co/datasets/anon-bench-org/anon-bench-test-run-data) — data from judge tuning and stability analysis


## Quick Start for running the benchmark

### Prerequisites
- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- AWS credentials with Bedrock access configured in the repo root `.env` file
- The root `.env` must contain: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and `BEDROCK_MODEL_ID`


### Run the benchmark
To benchmark an LLM on Anon-Bench:

```bash
cd evals

uv venv
source .venv/bin/activate
uv pip install -e "."

# Run the evaluation (solver + scorer)
uv run python -m elt_bench_eval.eval \
    --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --log-dir logs/my-eval-run

# Smoke test with 2 samples
uv run python -m elt_bench_eval.eval \
    --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --log-dir logs/smoke-test \
    --sample-limit 2

# View logs
inspect view --log-dir logs/
```

This runs the model (solver) on Anon-Bench tasks, scores responses using LLM-as-judge (scorer), and saves results to inspect-ai `.eval` logs. The default scorer setting is to use Claude Sonnet 4.6 (via AWS Bedrock) with temperature set to 0.0

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Solver model identifier (e.g., `bedrock/...`, `openai/...`) |
| `--log-dir` | (required) | Directory for output `.eval` logs |
| `--epochs` | 1 | Number of evaluation epochs |
| `--sample-limit` | 0 (all) | Limit to first N samples (0 = run all 1,000 tasks) |
| `--task-ids` | None | Specific task IDs to evaluate |
| `--solver-temperature` | 0.0 | Sampling temperature for solver |
| `--solver-max-tokens` | 4096 | Max output tokens for solver |
| `--scorer-model` | None | Override scorer model (default: uses solver model) |
| `--continue-on-fail` | True | Continue on transient errors |



## Scoring Formula

Task-level scores are computed from per-criterion binary Pass/Fail verdicts using a weighted formula:

```
task_score = sum(passed_weights) / sum(positive_weights_only)
```

- **Positive criteria** (weight > 0): Passing adds the criterion's weight to the numerator.
- **Negative criteria** (weight < 0): These act as **penalties**. If the judge determines the undesirable behaviour is present (i.e. the criterion "passes"), the negative weight is added to the numerator, *reducing* the score. If the undesirable behaviour is absent (criterion does not pass), 0 is added — no penalty.
- **Denominator**: Only positive weights contribute. This means the denominator represents the maximum achievable score ignoring penalties.
- **Score range**: Scores can go negative when many negative criteria activate; these are clipped to 0 during aggregation in `analysis/model_results_process.py`.

**Example:** A task with 3 criteria (weights: +5, +3, −2). The model passes all three:
- Numerator: 5 + 3 + (−2) = 6
- Denominator: 5 + 3 = 8
- Score: 6/8 = 75%

If the negative criterion does *not* pass (good — the undesirable behaviour is absent):
- Numerator: 5 + 3 + 0 = 8
- Denominator: 8
- Score: 8/8 = 100%

**Code references:**
- `evals/src/elt_bench_eval/score.py:161-166` — core scoring logic


## Scores of different models

Scored 1,000 published Anon-Bench tasks across 9 frontier models, 3 epochs each (27,000 total samples). Used Claude Sonnet 4.6 with the reference-guided classifier prompt (v1) as the production judge.

- **Code:** [`evals/batch-inference/`](batch-inference/) (solver + scorer batch scripts)
- **Logs:** See the [Huggingface repositories](https://huggingface.co/anon-bench-org).


| Directory | Model ID | Mean Score |
|-----------|----------|------------|
| `full-solver-claude-opus-4.7/` | `anthropic/claude-opus-4.7` | 85.5% |
| `full-solver-gpt-5.4/` | `openai/gpt-5.4` | 84.1% |
| `full-solver-gemini-3.1-pro-preview/` | `google/gemini-3.1-pro-preview` | 83.4% |
| `full-solver-gemini-3-flash-preview/` | `google/gemini-3-flash-preview` | 80.7% |
| `full-solver-deepseek-v3.2/` | `deepseek/deepseek.v3.2` | 80.2% |
| `full-solver-kimi-k2.5/` | `moonshotai/kimi-k2.5` | 79.1% |
| `full-solver-claude-haiku-4.5/` | `anthropic/claude-haiku-4.5` | 78.8% |
| `full-solver-qwen3-32b/` | `alibaba/qwen3-32b` | 65.8% |
| `full-solver-magistral-small/` | `mistral/magistral-small-2509` | 50.7% |


## Source Files

| File | Description |
|------|-------------|
| `src/elt_bench_eval/eval.py` | CLI entry point for Anon-Bench evaluation (solver + scorer) |
| `src/elt_bench_eval/run.py` | CLI entry point for GPC response generation (solver only, no scoring) |
| `src/elt_bench_eval/task.py` | Defines inspect-ai `Task` with `generate()` solver |
| `src/elt_bench_eval/score.py` | LLM-as-judge scorer: per-criterion binary verdicts, weighted aggregation (see [Scoring Formula](#scoring-formula)) |
| `src/elt_bench_eval/dataset.py` | Parses `Anon-Bench_tasks.csv` and resource files, builds inspect-ai `MemoryDataset` |
| `src/elt_bench_eval/criteria.py` | Criterion parsing and lookup from task CSV |
| `src/elt_bench_eval/prompts/` | Judge prompt templates (v1, v2, etc.) |
| `src/elt_bench_eval/bedrock_patch.py` | Patches Bedrock client timeout for long responses |

## Batch Inference Workflow

All evaluation runs were executed via the [`batch-inference/`](batch-inference/) pipeline, which orchestrates:

1. **Prepare** — Convert inspect-ai task definitions to Bedrock Batch input JSONL
2. **Submit** — Submit batch jobs to AWS Bedrock (Converse API)
3. **Poll** — Monitor job completion status
4. **Download** — Retrieve output JSONL from S3
5. **Replay** — Convert Bedrock outputs back into inspect-ai `.eval` log format
6. **Score** — Run scorer over replayed solver outputs (for pipelines where solver and scorer are separate)
7. **Combine** — Merge multi-epoch results into single `.eval` files

**Inference providers:**
- **Bedrock Batch** (direct): Claude Opus 4.7, Claude Haiku 4.5, DeepSeek V3.2, Kimi K2.5, Qwen3-32B, Magistral Small
- **Lambda + SQS**: GPT-5.4 (Azure OpenAI Responses API), Gemini 3.1 Pro, Gemini 3 Flash (GCP Vertex AI)

The Lambda+SQS approach was used for models not available on Bedrock, routing requests through AWS Lambda functions that call external APIs and return results via SQS for replay into the same `.eval` format.

## License

This repository contains both evaluation code and a benchmark dataset which are licensed separately:

- **Dataset:** All data files in the repo root and `resources_for_tasks/` directory (including `.csv` files) are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0)](LICENSE-DATA).
- **Code:** All Python scripts and evaluation pipelines are licensed under the [MIT License](LICENSE-CODE).