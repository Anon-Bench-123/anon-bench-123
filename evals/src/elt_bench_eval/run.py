"""
Main orchestration script for ELT-Bench response generation.

Loads environment configuration, builds the inspect-ai dataset and task,
and runs the evaluation against AWS Bedrock. Responses are saved in the
inspect-ai eval log; downstream extraction is handled by
validation/oup_app/format_dataset.py.

Version | Date       | Author       | Change comment
--------|------------|--------------|---------------
0.2.0   | 2026-03-01 | Claude       | Remove extract.py dep; response extraction now in format_dataset.py
0.1.3   | 2026-03-01 | Claude       | Increase max_connections 3→50, read_timeout 300→600
0.1.2   | 2026-02-27 | Claude       | Add fail_on_error + content filter handling
0.1.1   | 2026-02-27 | Claude       | Add Bedrock read_timeout config
0.1.0   | 2026-02-27 | Claude       | Initial version
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


from elt_bench_eval.bedrock_patch import patch_bedrock_timeout


def main() -> None:
    """
    Orchestrate the full ELT-Bench response generation pipeline.

    Steps
    -----
    1. Load environment variables (AWS credentials + model ID)
    2. Build the inspect-ai dataset from CSV + resources
    3. Create the Task with generation config
    4. Run eval() against Bedrock model
    5. Save eval log (response extraction handled by format_dataset.py)
    """
    # AIDEV-NOTE: Load .env from repo root (two levels up from evals/src/)
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    load_dotenv(env_path)
    logger.info(f"Loaded env from {env_path}")

    # Validate required env vars
    bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID")
    if not bedrock_model_id:
        logger.error("BEDROCK_MODEL_ID not set in environment")
        sys.exit(1)

    # AIDEV-NOTE: inspect-ai Bedrock model string format is "bedrock/<model-id>"
    model_str = f"bedrock/{bedrock_model_id}"
    logger.info(f"Using model: {model_str}")

    csv_path = repo_root / "elt-bench_tasks.csv"
    resources_dir = repo_root / "resources_for_tasks"
    log_dir = str(Path(__file__).resolve().parents[2] / "logs")

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    if not resources_dir.exists():
        logger.error(f"Resources directory not found: {resources_dir}")
        sys.exit(1)

    # Lazy imports to avoid loading inspect-ai before env is configured
    from inspect_ai import eval as inspect_eval

    from elt_bench_eval.task import create_elt_bench_task

    # Patch Bedrock timeout before running eval
    patch_bedrock_timeout(read_timeout=600)

    logger.info("Building inspect-ai task...")
    task = create_elt_bench_task(csv_path, resources_dir)

    # AIDEV-NOTE: max_connections=50 is safe for Bedrock cross-region Claude Sonnet 4.6
    # (10K RPM, 5M TPM). At 50 concurrent × ~5K tokens/req ≈ 125K TPM (2.5% of limit).
    # fail_on_error=False allows the eval to continue past individual sample failures
    # (e.g. Bedrock content filtering policy blocks).
    logger.info("Running evaluation...")
    logs = inspect_eval(
        task,
        model=model_str,
        log_dir=log_dir,
        max_connections=50,
        fail_on_error=False,
    )

    eval_log = logs[0]
    if eval_log.status not in ("success", "error"):
        logger.error(f"Eval terminated with status: {eval_log.status}")
        if eval_log.error:
            logger.error(f"Error: {eval_log.error}")
        sys.exit(1)

    if eval_log.status == "error":
        logger.warning(
            "Eval completed with errors (some samples may have failed due to "
            "content filtering or API issues)"
        )

    # AIDEV-NOTE: Response extraction from eval logs is now handled downstream
    # by validation/oup_app/format_dataset.py using read_eval_log().
    n_samples = len(eval_log.samples) if eval_log.samples else 0
    logger.success(f"Pipeline complete: {n_samples} samples evaluated")
    logger.info(f"View inspect logs: inspect view --log-dir {log_dir}")
    logger.info(
        "To extract responses, run: uv run validation/oup_app/format_dataset.py"
    )


if __name__ == "__main__":
    main()
