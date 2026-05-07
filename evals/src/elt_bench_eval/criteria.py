"""
Criteria parsing for ELT-Bench evaluation.

Reads task-specific, consensus, and universal criteria from l2-bench_tasks.csv
and returns structured Criterion objects for scoring.

"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from elt_bench_eval.dataset import load_tasks_csv


class Criterion(BaseModel):
    """A single scoring criterion for a task.

    Attributes
    ----------
    criterion_type : {'task', 'consensus', 'universal'}
        Category of the criterion.
    criterion_id : str
        Unique identifier for the criterion.
    name : str
        Display name (typically same as ``criterion_id``).
    description : str
        Human-readable description of what is being evaluated.
    weight : int
        Integer weight used for scoring.
    """

    criterion_type: Literal['task', 'consensus', 'universal']
    criterion_id: str
    name: str | None = None
    description: str
    weight: int


class CriteriaGetter:
    def __init__(self, csv_path: str | Path):
        """Initialise the getter by loading the tasks CSV.

        Parameters
        ----------
        csv_path : str or Path
            Path to the ``l2-bench_tasks.csv`` file.
        """
        self._csv_path = Path(csv_path)
        self._tasks_df = load_tasks_csv(self._csv_path)
    
    def _parse_criteria(
        self,
        col: Literal['task_criteria', 'consensus_criteria', 'universal_criteria'],
        raw: str,
        task_id: str | None = None,
    ) -> list[Criterion]:
        """Parse a pipe-delimited criteria string into Criterion objects.

        Parameters
        ----------
        col : {'task_criteria', 'consensus_criteria', 'universal_criteria'}
            The CSV column name, used to derive ``criterion_type``.
        raw : str
            Pipe-delimited string from the CSV, e.g.
            ``"01a-01. Includes reference... ==7 | 01a-02. References... ==7"``.
        task_id : str or None
            When provided and *col* is ``'task_criteria'``, the criterion_id
            is prefixed as ``'{task_id}-{id}'`` for traceability.

        Returns
        -------
        list of Criterion
            Parsed criteria with IDs, descriptions, and weights extracted.
        """
        if not raw or pd.isna(raw):
            return []

        criterion_type: Literal['task', 'consensus', 'universal'] = col.replace("_criteria", "")  # type: ignore[assignment]

        criteria = []
        for entry in str(raw).split("|"):
            entry = entry.strip()
            if not entry:
                continue

            weight_match = re.search(r"==(-?\d+)\s*$", entry)
            if not weight_match:
                raise ValueError(
                    f"Could not parse weight from criterion: task_id: {task_id}, col: {col}, entry: {entry!r}"
                )
            weight = int(weight_match.group(1))
            text = entry[: weight_match.start()].strip()

            id_match = re.match(r"^(.+?)\.\s+", text)
            if id_match:
                raw_id = id_match.group(1).strip()
                description = text[id_match.end():].strip()
            else:
                raise ValueError(
                    f"Could not parse criterion ID from entry: task_id: {task_id}, col: {col}, text: {text!r}"
                )

            if criterion_type == "task" and task_id is not None:
                criterion_id = f"{task_id}-{raw_id}"
            else:
                criterion_id = raw_id

            criteria.append(
                Criterion(
                    criterion_type=criterion_type,
                    criterion_id=criterion_id,
                    description=description,
                    weight=weight,
                )
            )

        return criteria


    def get_criteria_of_task(self, task_id: int | str) -> list[Criterion]:
        """Get all criteria (task, consensus, universal) for a given task.

        Parameters
        ----------
        task_id : int or str
            The ``task_key`` value to look up in ``l2-bench_tasks.csv``.

        Returns
        -------
        list of Criterion
            Combined list of task-specific, consensus, and universal criteria.

        Raises
        ------
        ValueError
            If *task_id* is not found in the CSV.
        """

        row = self._tasks_df[self._tasks_df["task_key"] == str(task_id)]
        if row.empty:
            raise ValueError(f"Task ID {task_id} not found in {self._csv_path.name}")
        row = row.iloc[0]

        criteria: list[Criterion] = []
        for col in ("task_criteria", "consensus_criteria", "universal_criteria"):
            criteria.extend(self._parse_criteria(col, row.get(col, ""), task_id=str(task_id)))

        return criteria


_criteria_getter: CriteriaGetter | None = None


def get_criteria_getter(csv_path: str | Path) -> CriteriaGetter:
    """Return a module-level singleton ``CriteriaGetter``.

    Parameters
    ----------
    csv_path : str or Path
        Path to the ``l2-bench_tasks.csv`` file.

    Returns
    -------
    CriteriaGetter
        Lazily-initialised singleton instance.
    """
    global _criteria_getter
    if _criteria_getter is None:
        _criteria_getter = CriteriaGetter(csv_path)
        logger.info("CriteriaGetter loaded")
    return _criteria_getter