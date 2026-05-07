"""
eval_set entry point for L2-Bench evaluation.

Usage:
    uv run python -m elt_bench_eval.eval \
        --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --log-dir logs/run-001

    # Smoke test with 2 samples
    uv run python -m elt_bench_eval.eval \
        --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --log-dir logs/smoke-test \
        --sample-limit 2

    # Run with custom solver config and scorer
    uv run python -m elt_bench_eval.eval \
        --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --log-dir logs/run-001 \
        --epochs 2 --sample-limit 10 \
        --solver-max-tokens 8192 --solver-temperature 0.5 \
        --scorer-model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --csv-path path/to/data.csv --resources-dir path/to/resources
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig, get_model

from elt_bench_eval.bedrock_patch import patch_bedrock_timeout
from elt_bench_eval.score import ScorerSetting
from elt_bench_eval.task import create_l2_bench_eval_task
from pydantic import BaseModel


class EvalRunParams(BaseModel):
    """Parameters for a single evaluation run.

    Attributes
    ----------
    solver_model_name : str
        Model identifier passed to ``get_model`` (e.g. ``bedrock/...``).
    solver_model_base_url : str or None
        Optional base URL override for the solver model API.
    solver_model_config : GenerateConfig
        Generation configuration for the solver model.
    log_dir : str
        Directory where eval logs are written.
    epochs : int
        Number of evaluation epochs.
    retry_on_error : int or None
        Number of retries on transient errors (``None`` disables retries).
    continue_on_fail : bool
        If ``True``, keep running remaining samples after a failure.
    scorer_setting : ScorerSetting or None
        Optional scorer model and generation configuration.
    csv_path : Path or None
        Path to the tasks CSV file. Uses the repo default when ``None``.
    resources_dir : Path or None
        Path to the task resources directory. Uses the repo default when ``None``.
    first_n_samples : int or None
        Limit evaluation to the first *n* samples.
    sample_range : tuple of (int, int) or None
        Slice range ``(start, end)`` applied to the dataset. Overrides
        ``first_n_samples`` when set.
    """

    solver_model_name: str
    solver_model_base_url: str | None = None
    solver_model_config: GenerateConfig = GenerateConfig(max_tokens=4096, temperature=0.0)
    log_dir: str
    epochs: int = 1
    retry_on_error: int | None = 1
    continue_on_fail: bool = True
    scorer_setting: ScorerSetting | None = None
    csv_path: Path | None = None
    resources_dir: Path | None = None
    first_n_samples: int | None = None
    sample_range: tuple[int, int] | None = None # will override first_n_samples
    task_ids: list[int] | None = None # will override sample_range


def run_eval(params: EvalRunParams):
    """Execute an L2-Bench evaluation run.

    Parameters
    ----------
    params : EvalRunParams
        Fully-populated run parameters including model, scorer, and dataset
        settings.
    """
    patch_bedrock_timeout(read_timeout=600)

    solver_model = get_model(
        model=params.solver_model_name,
        base_url=params.solver_model_base_url,
        config=params.solver_model_config,
    )

    task = create_l2_bench_eval_task(
        scorer_setting=params.scorer_setting,
        csv_path=params.csv_path,
        resources_dir=params.resources_dir,
        first_n_samples=params.first_n_samples,
        sample_range=params.sample_range,
        task_ids=params.task_ids
    )

    success, logs = eval_set(
        tasks=[task],
        model=solver_model,
        log_dir=params.log_dir,
        epochs=params.epochs,
        retry_on_error=params.retry_on_error,
        continue_on_fail=params.continue_on_fail
    )


def main():
    """CLI entry point for L2-Bench evaluation."""
    parser = argparse.ArgumentParser(description="Run L2-Bench eval_set")
    parser.add_argument("--model", required=True, help="Solver model name")
    parser.add_argument("--log-dir", required=True, help="Log directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env", help="Path to .env file")    
    
    parser.add_argument("--solver-max-tokens", type=int, default=4096, help="Solver max output tokens")
    parser.add_argument("--solver-temperature", type=float, default=0.0, help="Solver sampling temperature")
    parser.add_argument("--solver-top-p", type=float, default=None, help="Solver top-p (nucleus sampling)")
    parser.add_argument("--solver-top-k", type=int, default=None, help="Solver top-k sampling")
    parser.add_argument("--solver-frequency-penalty", type=float, default=None, help="Solver frequency penalty")
    parser.add_argument("--solver-presence-penalty", type=float, default=None, help="Solver presence penalty")
    parser.add_argument("--solver-seed", type=int, default=None, help="Solver random seed")
    parser.add_argument("--solver-stop-seqs", nargs="*", default=None, help="Solver stop sequences")
    parser.add_argument("--solver-num-choices", type=int, default=None, help="Solver number of choices")
    parser.add_argument("--solver-best-of", type=int, default=None, help="Solver best-of sampling count")
    parser.add_argument("--solver-max-retries", type=int, default=None, help="Solver max retries")
    parser.add_argument("--solver-timeout", type=int, default=None, help="Solver timeout in seconds")
    parser.add_argument("--solver-max-connections", type=int, default=None, help="Solver max connections")
    parser.add_argument("--solver-reasoning-tokens", type=int, default=None, help="Solver reasoning/thinking token budget")
    parser.add_argument("--solver-reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None, help="Solver reasoning effort level")
    
    parser.add_argument("--scorer-model", default=None, help="Scorer model name")
    parser.add_argument("--scorer-max-tokens", type=int, default=None, help="Scorer max output tokens")
    parser.add_argument("--scorer-temperature", type=float, default=None, help="Scorer sampling temperature")
    parser.add_argument("--scorer-top-p", type=float, default=None, help="Scorer top-p (nucleus sampling)")
    parser.add_argument("--scorer-top-k", type=int, default=None, help="Scorer top-k sampling")
    parser.add_argument("--scorer-frequency-penalty", type=float, default=None, help="Scorer frequency penalty")
    parser.add_argument("--scorer-presence-penalty", type=float, default=None, help="Scorer presence penalty")
    parser.add_argument("--scorer-seed", type=int, default=None, help="Scorer random seed")
    parser.add_argument("--scorer-stop-seqs", nargs="*", default=None, help="Scorer stop sequences")
    parser.add_argument("--scorer-num-choices", type=int, default=None, help="Scorer number of choices")
    parser.add_argument("--scorer-best-of", type=int, default=None, help="Scorer best-of sampling count")
    parser.add_argument("--scorer-max-retries", type=int, default=None, help="Scorer max retries")
    parser.add_argument("--scorer-timeout", type=int, default=None, help="Scorer timeout in seconds")
    parser.add_argument("--scorer-max-connections", type=int, default=None, help="Scorer max connections")
    parser.add_argument("--scorer-reasoning-tokens", type=int, default=None, help="Scorer reasoning/thinking token budget")
    parser.add_argument("--scorer-reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None, help="Scorer reasoning effort level")
    
    parser.add_argument(
        "--task-ids", nargs="*", type=int, default=None,
        help="List of task IDs to evaluate",
    )
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--resources-dir", type=Path, default=None)
    parser.add_argument("--prompt-version", default=None, help="Judge prompt version")
    args = parser.parse_args()

    load_dotenv(args.env_file)

    params = EvalRunParams(
        solver_model_name=args.model,
        solver_model_config=GenerateConfig(
            max_tokens=args.solver_max_tokens,
            temperature=args.solver_temperature,
            top_p=args.solver_top_p,
            top_k=args.solver_top_k,
            frequency_penalty=args.solver_frequency_penalty,
            presence_penalty=args.solver_presence_penalty,
            seed=args.solver_seed,
            stop_seqs=args.solver_stop_seqs,
            num_choices=args.solver_num_choices,
            best_of=args.solver_best_of,
            max_retries=args.solver_max_retries,
            timeout=args.solver_timeout,
            max_connections=args.solver_max_connections,
            reasoning_tokens=args.solver_reasoning_tokens,
            reasoning_effort=args.solver_reasoning_effort,
        ),
        log_dir=args.log_dir,
        epochs=args.epochs,
        continue_on_fail=args.continue_on_fail,
        scorer_setting=ScorerSetting(
            model=args.scorer_model,
            max_retries=args.scorer_max_retries,
            scorer_model_config=GenerateConfig(
                max_tokens=args.scorer_max_tokens,
                temperature=args.scorer_temperature,
                top_p=args.scorer_top_p,
                top_k=args.scorer_top_k,
                frequency_penalty=args.scorer_frequency_penalty,
                presence_penalty=args.scorer_presence_penalty,
                seed=args.scorer_seed,
                stop_seqs=args.scorer_stop_seqs,
                num_choices=args.scorer_num_choices,
                best_of=args.scorer_best_of,
                max_retries=args.scorer_max_retries,
                timeout=args.scorer_timeout,
                max_connections=args.scorer_max_connections,
                reasoning_tokens=args.scorer_reasoning_tokens,
                reasoning_effort=args.scorer_reasoning_effort,
            ),
            prompt_version=args.prompt_version or "v1",
        ) if any([
            args.scorer_model, args.scorer_max_tokens, args.scorer_temperature,
            args.scorer_top_p, args.scorer_top_k, args.scorer_frequency_penalty,
            args.scorer_presence_penalty, args.scorer_seed, args.scorer_stop_seqs,
            args.scorer_num_choices, args.scorer_best_of, args.scorer_max_retries,
            args.scorer_timeout, args.scorer_max_connections, args.scorer_reasoning_tokens,
            args.scorer_reasoning_effort, args.prompt_version,
        ]) else None,
        csv_path=args.csv_path,
        resources_dir=args.resources_dir,
        first_n_samples=args.sample_limit if args.sample_limit > 0 else None,
        task_ids=args.task_ids,
    )

    run_eval(params)


if __name__ == "__main__":
    main()
