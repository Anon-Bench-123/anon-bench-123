"""
LLM-as-judge scorer for L2-Bench evaluation.

Scores each sample by evaluating every criterion independently via a judge
model, then computing a weighted aggregate score.
"""

import asyncio
from pathlib import Path
from inspect_ai.scorer import scorer, mean, stderr, Scorer
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    accuracy,
    stderr,
    Scorer,
)
from inspect_ai.model import get_model, GenerateConfig
from elt_bench_eval.criteria import get_criteria_getter, Criterion
from elt_bench_eval.prompts.prompt_getter import get_judge_prompt, get_retry_prompt
from pydantic import BaseModel



class ScorerSetting(BaseModel):
    """Configuration for the scorer judge model.

    Attributes
    ----------
    model : str
        Model identifier for the judge (e.g. ``bedrock/...``).
    base_url : str or None
        Optional base URL override for the judge model API.
    scorer_model_config : GenerateConfig
        Generation configuration for the judge model.
    max_retries : int
        Number of retries when the judge returns an invalid value.
    """

    model: str
    base_url: str | None = None
    scorer_model_config: GenerateConfig = GenerateConfig(reasoning_tokens=1024)
    max_retries: int = 0
    prompt_version: str = "v1"


@scorer(metrics=[mean(), stderr()])
def l2_bench_scorer(csv_path: Path, setting: ScorerSetting) -> Scorer:
    """Create an inspect-ai scorer that evaluates responses against L2-Bench criteria.

    Parameters
    ----------
    csv_path : Path
        Path to the ``l2-bench_tasks.csv`` file.
    setting : ScorerSetting
        Judge model and generation configuration.

    Returns
    -------
    Scorer
        An async scoring function compatible with inspect-ai.
    """

    criteria_getter = get_criteria_getter(csv_path)
    judge_prompt_template = get_judge_prompt(setting.prompt_version)
    retry_prompt_suffix = get_retry_prompt()

    judge_model = get_model(
        model=setting.model,
        role="scorer",
        base_url=setting.base_url,
        config=setting.scorer_model_config,
    )

    async def get_criterion_score(
        criterion: Criterion, state: TaskState, target: Target
    ) -> bool:
        """Judge whether a single criterion is met.

        Parameters
        ----------
        criterion : Criterion
            The criterion to evaluate.
        state : TaskState
            Current task state containing the solver's response.
        target : Target
            Reference answer for the task.

        Returns
        -------
        bool
            ``True`` if the criterion is met, ``False`` otherwise.

        Raises
        ------
        ValueError
            If the solver produced no output or the judge fails to return a
            valid ``'true'``/``'false'`` value after retries.
        """
        input_messages = state.input if isinstance(state.input, list) else []
        task_text = next((msg.text for msg in input_messages if msg.role == "user"), "")
        ai_response = state.output.completion
        
        if not ai_response or ai_response == "":
            raise ValueError("Solver model fails to generate any content")

        reference_answer = target.text

        prompt = judge_prompt_template.format(
            task_text=task_text,
            reference_answer=reference_answer,
            ai_response=ai_response,
            criterion_description=criterion.description
        )
        

        judgement = ""
        for i in range(setting.max_retries + 1):
            if i == 0:
                response = await judge_model.generate(prompt)
            else:
                response = await judge_model.generate(
                    prompt + retry_prompt_suffix
                )
            completion = response.completion.strip()
            judgement = completion.splitlines()[-1].strip().lower() if completion else ""
            if judgement == "true" or judgement == "false":
                break
        
        if judgement != "true" and judgement != "false":
            raise ValueError(f"LLM judge fails to generate valid value: {judgement}")
        
        return True if judgement == "true" else False
      

    async def score(state: TaskState, target: Target) -> Score:
        """Score a single sample by aggregating all criterion judgements.

        Parameters
        ----------
        state : TaskState
            Current task state containing the solver's response.
        target : Target
            Reference answer for the task.

        Returns
        -------
        Score
            Weighted score in ``[0, 1]`` with per-criterion metadata.
        """
        task_id = state.sample_id
        criteria = criteria_getter.get_criteria_of_task(task_id)
        scoring_jobs = [
            get_criterion_score(criterion, state, target) for criterion in criteria
        ]

        judgements = await asyncio.gather(*scoring_jobs)
        
        scores = [(criterion.weight if judgement else 0) for judgement, criterion in zip(judgements, criteria)]

        maximum_possible_score = sum(
            [criterion.weight for criterion in criteria if criterion.weight > 0]
        )
        task_score = sum(scores) / maximum_possible_score

        return Score(
            value=task_score,
            metadata={
                "scoring_results": [{
                    'criterion_id': criterion.criterion_id,
                    'weight': criterion.weight,
                    'judgement': judgement
                } for criterion, judgement in zip(criteria, judgements)]
            }
        )

    return score
