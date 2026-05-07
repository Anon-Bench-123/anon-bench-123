"""
Data preparation: CSV + resources -> inspect-ai Dataset.

Reads elt-bench_tasks.csv and resource files from resources_for_tasks/,
then constructs an inspect-ai MemoryDataset of Sample objects.

Version | Date       | Author       | Change comment
--------|------------|--------------|---------------
0.2.0   | 2026-03-01 | Claude       | Remove reference_answer from Sample.target
0.1.0   | 2026-02-27 | Claude       | Initial version
"""

from pathlib import Path

import pandas as pd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from loguru import logger


def load_tasks_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read elt-bench_tasks.csv into a DataFrame.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with all task columns.

    Notes
    -----
    Uses encoding='utf-8-sig' to handle the UTF-8 BOM marker present in the file.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"task_key": str})
    logger.info(f"Loaded {len(df)} tasks from {csv_path.name}")
    return df


def load_resource_content(filename: str, resources_dir: Path) -> str:
    """
    Read a resource file's content from the resources_for_tasks directory.

    Parameters
    ----------
    filename : str
        Filename of the resource (e.g. 'teaching_ef5e_preint_contents.md').
    resources_dir : Path
        Path to the resources_for_tasks directory.

    Returns
    -------
    str
        The full text content of the resource file, or a placeholder if not found.
    """
    filepath = resources_dir / filename
    if not filepath.exists():
        logger.warning(f"Resource file not found: {filename}")
        return f"[Resource not available: {filename}]"
    return filepath.read_text(encoding="utf-8")


def build_user_message(task_text: str, resource_contents: dict[str, str]) -> str:
    """
    Combine task text with resource content.

    Appends resource content after the task text with clear markdown delimiters.
    The task text already references resources by filename (e.g. [filename.md]),
    so the model can cross-reference.

    Parameters
    ----------
    task_text : str
        The task column value from the CSV.
    resource_contents : dict[str, str]
        Mapping of resource filename to its content.

    Returns
    -------
    str
        The complete user message with embedded resource content.
    """
    if not resource_contents:
        return task_text

    parts = [task_text, ""]
    for filename, content in resource_contents.items():
        parts.append(f"---\n## Resource: {filename}\n\n{content}")

    return "\n".join(parts)


# AIDEV-NOTE: Each Sample uses per-task system prompts via ChatMessageSystem in input,
# not the global system_message() solver (which only supports a single static prompt).
def create_inspect_dataset(csv_path: Path, resources_dir: Path) -> MemoryDataset:
    """
    Build an inspect-ai MemoryDataset from the ELT-Bench CSV.

    Each row becomes a Sample with:
    - input: list[ChatMessage] with system message + user message (task + resources)
    - target: empty string (generation-only pipeline, no scorer uses target)
    - id: task_key (as str)
    - metadata: {competency, version}

    Parameters
    ----------
    csv_path : Path
        Path to elt-bench_tasks.csv.
    resources_dir : Path
        Path to resources_for_tasks directory.

    Returns
    -------
    MemoryDataset
        Dataset of Samples ready for inspect-ai evaluation.
    """
    df = load_tasks_csv(csv_path)
    resource_link_cols = [
        "task_resource_file_link_1",
        "task_resource_file_link_2",
        "task_resource_file_link_3",
    ]

    samples = []
    missing_resources = 0

    for _, row in df.iterrows():
        # Load resource content for any non-empty resource links
        resource_contents: dict[str, str] = {}
        for col in resource_link_cols:
            filename = str(row[col]).strip() if pd.notna(row[col]) else ""
            if filename:
                content = load_resource_content(filename, resources_dir)
                if content.startswith("[Resource not available"):
                    missing_resources += 1
                resource_contents[filename] = content

        user_message = build_user_message(str(row["task"]), resource_contents)
        system_prompt = str(row["task_system_prompt"])

        sample = Sample(
            input=[
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_message),
            ],
            target=str(row.get("reference_answer", "")),
            id=str(row["task_key"]),
            metadata={
                "competency": str(row["competency"]),
                "version": str(row["version_at_time_of_draft"]),
            },
        )
        samples.append(sample)

    if missing_resources > 0:
        logger.warning(f"{missing_resources} resource file(s) not found")

    logger.info(f"Created dataset with {len(samples)} samples")
    return MemoryDataset(samples=samples, name="elt-bench")


def create_filtered_inspect_dataset(
    csv_path: Path,
    resources_dir: Path,
) -> MemoryDataset:
    """
    Build an inspect-ai MemoryDataset filtered to published tasks.

    Wraps create_inspect_dataset and filters to tasks where is_published_v1-0 == 1.

    Parameters
    ----------
    csv_path : Path
        Path to elt-bench_tasks.csv.
    resources_dir : Path
        Path to resources_for_tasks directory.

    Returns
    -------
    MemoryDataset
        Filtered dataset of Samples ready for inspect-ai evaluation.
    """
    dataset = create_inspect_dataset(csv_path, resources_dir)

    df = load_tasks_csv(csv_path)
    published_ids = set(
        df[df["is_published_v1-0"] == 1]["task_key"].astype(str).tolist()
    )

    filtered_samples = [s for s in dataset if s.id in published_ids]
    logger.info(
        f"Filtered to {len(filtered_samples)} published samples "
        f"(from {len(dataset)} total)"
    )

    return MemoryDataset(samples=filtered_samples, name="elt-bench")
