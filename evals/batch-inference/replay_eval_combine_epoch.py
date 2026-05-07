"""Replay batch inference results through Inspect AI for final evaluation.

Combines solver and scorer outputs from all epochs and runs them through the
Inspect AI evaluation framework. Uses a custom ReplayModelAPI that returns
pre-computed batch inference results instead of making live API calls.

Reads solver output from: output/solver/<eval_id>/<epoch>/input.jsonl.out
Reads scorer output from: output/scorer/<eval_id>/<epoch>/input.jsonl.out
Writes evaluation logs to: logs/<eval_id>/

Usage:
    uv run replay_eval_combine_epoch.py --eval-id <eval_id> \\
        --solver-model-name <solver_model> --scorer-model-name <scorer_model>
"""

import asyncio
from inspect_ai import Task, eval_set
from inspect_ai.model import (
    ModelAPI,
    ModelOutput,
    ModelUsage,
    GenerateConfig,
    ChatMessage,
    ChatMessageAssistant,
    get_model,
    modelapi,
    Model,
)
from inspect_ai.model._model_output import ChatCompletionChoice
from inspect_ai.tool import ToolInfo, ToolChoice
from inspect_ai.solver import Generate, generate, TaskState, solver
from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    accuracy,
    stderr,
    Scorer,
)
from inspect_ai.scorer import scorer, mean, stderr, Scorer
from inspect_ai.model._model_output import as_stop_reason

from elt_bench_eval.dataset import create_inspect_dataset, create_filtered_inspect_dataset
from elt_bench_eval.criteria import get_criteria_getter, Criterion

from pathlib import Path
from typing import Any, Literal
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def replay_eval(
    eval_id: str,
    solver_model_name: str,
    scorer_model_name: str,
    scorer_retry_attempts: int = 3,
    scorer_retry_temperature: float = 0.2,
    csv_path: Path = Path("./l2-bench_tasks.csv"),
    filter_published: bool = True,
):
    """Replay batch inference results through Inspect AI evaluation.

    Auto-detects epochs from the solver output directory and combines results
    across all epochs. Tasks are filtered to those present in all epochs.

    Args:
        eval_id: Evaluation identifier matching the output directory structure.
        solver_model_name: Name of the solver model (for logging purposes).
        scorer_model_name: Name of the scorer model (used for retry fallback).
        scorer_retry_attempts: Number of retry attempts if scorer output is invalid.
        scorer_retry_temperature: Temperature for scorer retry calls.
        csv_path: Path to the L2-bench tasks CSV file.
        filter_published: If True, only include tasks with is_published_v1-0 == 1.
    """
    # Define singletons here
    criteria_getter = get_criteria_getter(csv_path=csv_path)
    solver_outputs = {}  # {epoch: {record_id: item}}
    scorer_outputs = {}  # {epoch: {record_id: item}}

    # Auto-detect epochs from directory structure
    solver_base = Path(f"output/solver/{eval_id}")
    epochs = sorted([int(d.name) for d in solver_base.iterdir() if d.is_dir() and d.name.isdigit()])

    for epoch in epochs:
        solver_outputs[epoch] = {}
        scorer_outputs[epoch] = {}

        solver_output_path = Path(f"output/solver/{eval_id}/{epoch}/input.jsonl.out")
        scorer_output_path = Path(f"output/scorer/{eval_id}/{epoch}/input.jsonl.out")

        with solver_output_path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                solver_outputs[epoch][item["recordId"]] = item

        with scorer_output_path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                scorer_outputs[epoch][item["recordId"]] = item

    class ReplayModelAPI(ModelAPI):
        def __init__(
            self,
            model_name: str,  # not used, and not to be confused by the 'model' parameter in get_model function
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
        ) -> None:
            """Initialize the replay model API.

            Args:
                model_name: Model name (not used, required by ModelAPI interface).
                config: Generation configuration.
                **model_args: Must include 'epoch', 'replay_role' ('solver' or 'scorer'),
                    'task_id', and optionally 'criterion_id' for scorer role.
            """
            super().__init__(model_name=model_name, config=config)

            self.record_id = ""
            self.epoch: int | None = model_args.get("epoch")
            self.replay_role: Literal["solver", "scorer"] | None = model_args.get(
                "replay_role"
            )

            if self.replay_role:

                task_id = model_args.get("task_id")

                if self.replay_role == "solver":
                    self.record_id = str(task_id)
                elif self.replay_role == "scorer":
                    criterion_id = model_args.get("criterion_id")
                    self.record_id = f"{task_id}/{criterion_id}"
                else:
                    raise ValueError(f"Unknown role: {self.replay_role}")

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            """Return pre-computed batch inference result for Inspect AI evaluation.

            Implements the ModelAPI interface so Inspect AI can use the batch
            inference results. Looks up the result from loaded JSONL files by
            replay_role (solver/scorer), epoch, and record_id.

            Args:
                input: Chat messages (unused).
                tools: Tool definitions (unused).
                tool_choice: Tool choice setting (unused).
                config: Generation config (unused).

            Returns:
                ModelOutput with the batch inference result.
            """
            generated_content_text = ""
            stop_reason = "not_running"
            usage_data = {"inputTokens": 0, "outputTokens": 0}

            if self.replay_role:
                all_records = (
                    solver_outputs if self.replay_role == "solver" else scorer_outputs
                )
                epoch_records = all_records.get(self.epoch, {})
                record = epoch_records.get(self.record_id)

                if not record:
                    raise ValueError(
                        f"Record {self.record_id} for epoch {self.epoch} not found"
                    )

                model_output = record["modelOutput"]
                generated_content_text = model_output["output"]["message"]["content"][
                    0
                ]["text"]
                stop_reason = model_output["stopReason"]
                usage_data = model_output["usage"]

            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(content=generated_content_text),
                        stop_reason=as_stop_reason(stop_reason),
                    )
                ],
                usage=ModelUsage(
                    input_tokens=usage_data.get("inputTokens", 0),
                    output_tokens=usage_data.get("outputTokens", 0),
                ),
            )

    @modelapi(name="replay")
    def replay():
        """Register ReplayModelAPI with Inspect AI under the 'replay' prefix.

        Returns:
            ReplayModelAPI class for Inspect AI model registration.
        """
        return ReplayModelAPI

    @solver
    def replay_solver():
        """Inspect AI solver that retrieves batch inference solver output.

        Returns:
            Async solve function that looks up solver output by task_id and epoch.
        """
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            solver_model = get_model(
                model=f"replay/{solver_model_name}",  # This parameter is model, not model_name
                memoize=False,
                eval_id=eval_id,
                epoch=state.epoch,
                replay_role="solver",
                task_id=state.sample_id,
            )

            solver_model_output = await solver_model.generate(state.input_text)

            state.messages.append(solver_model_output.choices[0].message)
            state.output = solver_model_output

            return state

        return solve

    @scorer(metrics=[mean(), stderr()])
    def replay_scorer() -> Scorer:
        """Inspect AI scorer that retrieves batch inference scorer output.

        Computes final task score by aggregating per-criterion judgements
        from the batch inference results, weighted by criterion weights.

        Returns:
            Async score function that computes weighted task score from criterion judgements.
        """
        async def get_criterion_score(
            criterion: Criterion, state: TaskState, target: Target
        ) -> bool:
            """Get judgement for a single criterion from batch inference output.

            Falls back to live scorer model call if the batch result is invalid.

            Args:
                criterion: The criterion to evaluate.
                state: Current task state with sample_id and epoch.
                target: Target answer (unused).

            Returns:
                True if criterion is satisfied, False otherwise.
            """
            task_id = state.sample_id
            criterion_id = criterion.criterion_id

            scorer_model = get_model(
                model=f"replay/{scorer_model_name}",
                memoize=False,
                eval_id=eval_id,
                epoch=state.epoch,
                replay_role="scorer",
                task_id=task_id,
                criterion_id=criterion_id,
            )

            scorer_record_id = f"{task_id}/{criterion_id}"
            scorer_input = scorer_outputs[state.epoch][scorer_record_id]["modelInput"]["messages"][
                0
            ]["content"][0]["text"]

            response = await scorer_model.generate(scorer_input)

            completion = response.completion.strip()

            # Get the last line to find the final verdict
            last_line = completion.splitlines()[-1].strip()

            if "true" in last_line:
                return True
            elif "false" in last_line:
                return False
            else:
                retry_last_line = ""
                retry_scorer_model = get_model(model=scorer_model_name, config=GenerateConfig(temperature=scorer_retry_temperature))
                for _ in range(scorer_retry_attempts):
                    retry_response = await retry_scorer_model.generate(scorer_input)
                    retry_completion = retry_response.completion.strip()
                    retry_last_line = retry_completion.splitlines()[-1].strip()
                    if "true" in retry_last_line or "false" in retry_last_line:
                        break
                if "true" in retry_last_line:
                    return True
                elif "false" in retry_last_line:
                    return False
                else:
                    raise ValueError(
                        f"LLM judge fails to generate valid value in line: '{last_line}'"
                    )

            return True if judgement == "true" else False

        async def score(state: TaskState, target: Target) -> Score:
            """Compute weighted task score from all criterion judgements.

            Args:
                state: Current task state with sample_id.
                target: Target answer (unused).

            Returns:
                Score with value as weighted sum of criterion judgements.
            """
            task_id = state.sample_id
            criteria = criteria_getter.get_criteria_of_task(task_id)
            scoring_jobs = [
                get_criterion_score(criterion, state, target) for criterion in criteria
            ]

            judgements = await asyncio.gather(*scoring_jobs)

            scores = [
                (criterion.weight if judgement else 0)
                for judgement, criterion in zip(judgements, criteria)
            ]

            maximum_possible_score = sum(
                [criterion.weight for criterion in criteria if criterion.weight > 0]
            )
            task_score = sum(scores) / maximum_possible_score

            return Score(
                value=task_score,
                metadata={
                    "scoring_results": [
                        {
                            "criterion_id": criterion.criterion_id,
                            "weight": criterion.weight,
                            "judgement": judgement,
                        }
                        for criterion, judgement in zip(criteria, judgements)
                    ]
                },
            )

        return score

    def get_task() -> Task:
        """Create Inspect AI Task with replay solver and scorer.

        Returns:
            Task configured with the filtered dataset, replay_solver, and replay_scorer.
        """
        if filter_published:
            dataset = create_filtered_inspect_dataset(
                Path("./l2-bench_tasks.csv"),
                resources_dir=Path("./resources_for_tasks"),
            )
        else:
            dataset = create_inspect_dataset(
                Path("./l2-bench_tasks.csv"),
                resources_dir=Path("./resources_for_tasks"),
            )

        # Get intersection of task IDs across all epochs
        valid_ids_sets = [set(solver_outputs[epoch].keys()) for epoch in epochs]
        valid_ids = set.intersection(*valid_ids_sets) if valid_ids_sets else set()

        filtered_dataset = [sample for sample in dataset if sample.id in valid_ids]

        return Task(
            name=eval_id,
            dataset=filtered_dataset,
            solver=[replay_solver()],
            scorer=[replay_scorer()],
        )

    def run_replay() -> None:
        """Execute the Inspect AI evaluation with replay models."""
        solver_model = get_model(
            model=f"replay/{solver_model_name}", eval_id=eval_id
        )

        eval_set(
            tasks=[get_task()],
            model=solver_model,  # dummy, never called
            log_dir=f"logs/{eval_id}",
            epochs=len(epochs),
            retry_attempts=0,
            continue_on_fail=True,
        )

    run_replay()


def main():
    parser = argparse.ArgumentParser(
        description="Replay evaluation with specified parameters."
    )
    parser.add_argument("--eval-id", type=str, required=True, help="The evaluation ID.")
    parser.add_argument(
        "--solver-model-name",
        type=str,
        required=True,
        help="The name of the solver model.",
    )
    parser.add_argument(
        "--scorer-model-name",
        type=str,
        required=True,
        help="The name of the scorer model.",
    )
    parser.add_argument(
        "--filter-published",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter to published tasks only (is_published_v1-0 == 1). Use --no-filter-published to disable.",
    )

    args = parser.parse_args()

    replay_eval(
        args.eval_id,
        args.solver_model_name,
        args.scorer_model_name,
        filter_published=args.filter_published,
    )


if __name__ == "__main__":
    main()
