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

from elt_bench_eval.dataset import create_inspect_dataset
from elt_bench_eval.criteria import get_criteria_getter, Criterion

from pathlib import Path
from typing import Any, Literal
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def replay_eval(
    eval_id: str,
    epoch: int,
    solver_model_name: str,
    scorer_model_name: str,
    scorer_retry_attempts: int = 3,
    scorer_retry_temperature: float = 0.2,
    csv_path: Path = Path("./l2-bench_tasks.csv"),
):

    # Define singletons here
    criteria_getter = get_criteria_getter(csv_path=csv_path)
    solver_outputs = {}
    scorer_outputs = {}

    solver_output_path = Path(f"output/solver/{eval_id}/{epoch}/input.jsonl.out")
    scorer_output_path = Path(f"output/scorer/{eval_id}/{epoch}/input.jsonl.out")

    with solver_output_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            solver_outputs[item["recordId"]] = item

    with scorer_output_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            scorer_outputs[item["recordId"]] = item

    class ReplayModelAPI(ModelAPI):
        def __init__(
            self,
            model_name: str,  # not used, and not to be confused by the 'model' parameter in get_model function
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
        ) -> None:
            super().__init__(model_name=model_name, config=config)

            self.record_id = ""
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

            generated_content_text = ""
            stop_reason = "not_running"
            usage_data = {"inputTokens": 0, "outputTokens": 0}

            if self.replay_role:
                records = (
                    solver_outputs if self.replay_role == "solver" else scorer_outputs
                )
                record = records.get(self.record_id)

                if not record:
                    raise ValueError(
                        f"Record {self.record_id} not found in file {solver_output_path if self.replay_role == 'solve' else scorer_output_path}"
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
        return ReplayModelAPI

    @solver
    def replay_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            solver_model = get_model(
                model=f"replay/{solver_model_name}",  # This parameter is model, not model_name
                memoize=False,
                eval_id=eval_id,
                epoch=epoch,
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

        async def get_criterion_score(
            criterion: Criterion, state: TaskState, target: Target
        ) -> bool:

            task_id = state.sample_id
            criterion_id = criterion.criterion_id

            scorer_model = get_model(
                model=f"replay/{scorer_model_name}",
                memoize=False,
                eval_id=eval_id,
                epoch=epoch,
                replay_role="scorer",
                task_id=task_id,
                criterion_id=criterion_id,
            )

            scorer_record_id = f"{task_id}/{criterion_id}"
            scorer_input = scorer_outputs[scorer_record_id]["modelInput"]["messages"][
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

        dataset = create_inspect_dataset(
            Path("./l2-bench_tasks.csv"),
            resources_dir=Path("./resources_for_tasks"),
        )

        valid_ids = [key for key in list(solver_outputs.keys())]

        filtered_dataset = [sample for sample in dataset if sample.id in valid_ids]

        return Task(
            name=f"{eval_id}-{epoch}",
            dataset=filtered_dataset,
            solver=[replay_solver()],
            scorer=[replay_scorer()],
        )

    def run_replay() -> None:
        solver_model = get_model(
            model=f"replay/{solver_model_name}", eval_id=eval_id, epoch=epoch
        )

        eval_set(
            tasks=[get_task()],
            model=solver_model,  # dummy, never called
            log_dir=f"logs/{eval_id}-{epoch}",
            retry_attempts=0,
            continue_on_fail=True,
        )

    run_replay()


def main():
    parser = argparse.ArgumentParser(
        description="Replay evaluation with specified parameters."
    )
    parser.add_argument("--eval-id", type=str, required=True, help="The evaluation ID.")
    parser.add_argument("--epoch", type=int, required=True, help="The epoch number.")
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

    args = parser.parse_args()

    replay_eval(
        args.eval_id, args.epoch, args.solver_model_name, args.scorer_model_name
    )


if __name__ == "__main__":
    main()
