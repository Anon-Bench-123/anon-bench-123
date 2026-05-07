"""
inspect-ai Task definition for ELT-Bench response generation.

Defines a generation-only Task (no scoring) using the generate() solver
with configurable temperature and max_tokens.

Version | Date       | Author       | Change comment
--------|------------|--------------|---------------
0.2.0   | 2026-03-01 | Claude       | Version bump (no functional change to this file)
0.1.0   | 2026-02-27 | Claude       | Initial version
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from inspect_ai.dataset import MemoryDataset

from elt_bench_eval.dataset import create_inspect_dataset
from elt_bench_eval.score import l2_bench_scorer, ScorerSetting


def create_l2_bench_eval_task(
    scorer_setting: ScorerSetting | None = None,
    csv_path: Path | None = None,
    resources_dir: Path | None = None,
    **kwargs # for internal testing params like first_n_samples, sample_range and dataset
) -> Task:
    """Create an inspect-ai Task for L2-Bench evaluation with scoring.

    Parameters
    ----------
    scorer_setting : ScorerSetting or None
        Judge model and generation configuration. Falls back to a default
        Sonnet 4.6 scorer when ``None``.
    csv_path : Path or None
        Path to ``l2-bench_tasks.csv``. Uses the repo default when ``None``.
    resources_dir : Path or None
        Path to the task resources directory. Uses the repo default when
        ``None``.
    **kwargs
        Internal testing parameters: ``first_n_samples`` (int),
        ``sample_range`` (tuple of int), ``dataset`` (MemoryDataset).

    Returns
    -------
    Task
        inspect-ai Task configured for generation and scoring.
    """

    if not scorer_setting:
        scorer_setting = ScorerSetting(
            model="bedrock/us.anthropic.claude-sonnet-4-6",
            scorer_model_config=GenerateConfig(temperature=0.0),
        )

    _repo_root = Path(__file__).resolve().parents[3]
    _default_csv = _repo_root / "l2-bench_tasks.csv"
    _default_resources = _repo_root / "resources_for_tasks"

    if not csv_path:
        csv_path = _default_csv

    if not resources_dir:
        resources_dir = _default_resources

    full_dataset = create_inspect_dataset(csv_path, resources_dir)
    defected_ids = [str(v) for v in [540]]

    clean_dataset = MemoryDataset(
        samples=[
            sample for sample in full_dataset.samples if sample.id not in defected_ids
        ],
        name="l2-bench-samples",
    )
    
    dataset = clean_dataset
    
    first_n_samples = kwargs.get('first_n_samples')
    if first_n_samples and isinstance(first_n_samples, int):
        dataset = clean_dataset[:first_n_samples]
    
    sample_range = kwargs.get('sample_range')
    if sample_range and isinstance(sample_range, tuple) and len(sample_range) == 2 and all(isinstance(index, int) for index in sample_range):
        dataset = clean_dataset[sample_range[0] : sample_range[1]]

    task_ids = kwargs.get('task_ids')
    if task_ids and isinstance(task_ids, list):
        str_task_ids = [str(tid) for tid in task_ids]
        clean_ids = {sample.id for sample in clean_dataset.samples}
        missing = [tid for tid in str_task_ids if tid not in clean_ids]
        if missing and not kwargs.get("dataset"):
            raise ValueError(f"Task IDs not found in dataset: {missing}")
        dataset = MemoryDataset(
            samples=[s for s in clean_dataset.samples if s.id in str_task_ids],
            name="l2-bench-samples",
        )

    if kwargs.get("dataset") and isinstance(kwargs.get("dataset"), MemoryDataset):
        dataset = kwargs.get("dataset")

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=l2_bench_scorer(
            csv_path=csv_path,
            setting=scorer_setting,
        ),
        name="l2-bench-eval",
        version=1,
        metadata={
            "benchmark": "l2-bench",
            "purpose": "response_evaluation",
        },
    )


# AIDEV-NOTE: max_tokens=4096 is ~2x the max reference answer length (2896 words).
# Median ref answer is 744 words (~967 tokens), so 4096 provides ample headroom.
def create_elt_bench_task(
    csv_path: Path,
    resources_dir: Path,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Task:
    """
    Create an inspect-ai Task for ELT-Bench response generation.

    Parameters
    ----------
    csv_path : Path
        Path to elt-bench_tasks.csv.
    resources_dir : Path
        Path to resources_for_tasks directory.
    max_tokens : int
        Maximum output tokens. Default 4096 (~2x max reference answer).
    temperature : float
        Sampling temperature. Default 0.0 for reproducibility.

    Returns
    -------
    Task
        inspect-ai Task configured for generation-only evaluation.
    """
    dataset = create_inspect_dataset(csv_path, resources_dir)

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=None,
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        name="elt-bench-generation",
        version=1,
        metadata={
            "benchmark": "elt-bench",
            "purpose": "response_generation",
        },
    )
