"""Generate JSONL input for L2-bench solver model batch inference.

Reads L2-bench task definitions from l2-bench_tasks.csv, loads task resources from
resources_for_tasks/, and produces a JSONL file in AWS Bedrock batch inference format.
This format is also used as the canonical input for Azure/GCP inference pipelines.

Output: input/solver/<eval_id>/input.jsonl

Usage:
    uv run solver_model_batch_inference_input.py --eval_id <eval_id> [--inference_config '<json>']
"""

from typing import Any
import argparse
import sys
import json
from pathlib import Path
from loguru import logger
import pandas as pd
from elt_bench_eval.dataset import load_tasks_csv, load_resource_content, build_user_message

def create_batch_inference_input(
    eval_id: str,
    solver_input_base_path: Path = Path('./input/solver'),
    inference_config: dict | None = None,
    csv_path: Path = Path('./l2-bench_tasks.csv'),
    resources_dir: Path = Path('./resources_for_tasks'),
    test_number: int | None = None,
    task_ids: list[str] | None = None,
) -> None:
    """Generate batch inference input JSONL from L2-bench tasks.

    Args:
        eval_id: Evaluation identifier, used as subdirectory name in output path.
        solver_input_base_path: Base directory for output files.
        inference_config: Model parameters (e.g. {"maxTokens": 4096, "temperature": 0.7}).
        csv_path: Path to l2-bench_tasks.csv.
        resources_dir: Directory containing task resource files.
        test_number: Limit output to first N tasks.
        task_ids: Specific task IDs to include (overrides test_number).
    """
    output_path = (
        solver_input_base_path / eval_id / 'input.jsonl'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_tasks_csv(csv_path)
    resource_link_cols = [
        "task_resource_file_link_1",
        "task_resource_file_link_2",
        "task_resource_file_link_3",
    ]

    inputs = []
    missing_resources = 0

    # Filter by task_ids if provided (overrides test_number)
    if task_ids is not None:
        working_df = df[df["task_key"].isin(task_ids)]
    elif test_number is not None:
        working_df = df.head(test_number)
    else:
        working_df = df

    for _, row in working_df.iterrows():
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

        system_prompt = str(row["task_system_prompt"]) if pd.notna(row["task_system_prompt"]) else ""

        model_input: dict[str, Any] = {
            'messages': [
                {
                    'role': 'user',
                    'content': [{"text" : user_message}]
                }
            ]
        }

        if system_prompt:
            model_input['system'] = [{"text" : system_prompt}]

        if inference_config:
            model_input['inferenceConfig'] = inference_config

        input = {
            'recordId': str(row["task_key"]),
            'modelInput': model_input
        }

        inputs.append(input)

    if missing_resources > 0:
        logger.warning(f"{missing_resources} resource file(s) not found")

    with open(output_path, 'w', encoding='utf-8') as f:
        for input in inputs:
            json_record = json.dumps(input)
            f.write(json_record + '\n')

    logger.info(f"Created input jsonl for batch inference with {len(inputs)} records")


def main():
    parser = argparse.ArgumentParser(description="Generate model batch inference input JSONL.")
    parser.add_argument("--eval_id", type=str, required=True, help="Evaluation ID.")
    parser.add_argument(
        "--solver_input_base_path",
        type=Path,
        default=None,
        help="Base path for output JSONL file.",
    )
    parser.add_argument("--csv_path", type=Path, default=None, help="Path to the tasks CSV file.")
    parser.add_argument("--resources_dir", type=Path, default=None, help="Directory containing task resources.")
    parser.add_argument("--inference_config", type=str, default=None, help="JSON string representing model parameters.")
    parser.add_argument("--test_number", type=int, default=None, help="Only output the first N tasks.")
    parser.add_argument("--task_ids", nargs="*", type=str, default=None, help="List of task IDs to include (overrides --test_number).")

    args = parser.parse_args()

    # Parse inference_config JSON string
    inference_config = None
    if args.inference_config:
        try:
            inference_config = json.loads(args.inference_config)
            if not isinstance(inference_config, dict):
                logger.error("--inference_config must be a JSON object (dictionary).")
                sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --inference_config as JSON: {e}")
            sys.exit(1)

    # Filter out None values to respect function defaults
    kwargs = {
        'eval_id': args.eval_id,
    }
    if args.solver_input_base_path is not None:
        kwargs['solver_input_base_path'] = args.solver_input_base_path
    if args.csv_path is not None:
        kwargs['csv_path'] = args.csv_path
    if args.resources_dir is not None:
        kwargs['resources_dir'] = args.resources_dir
    if inference_config:
        kwargs['inference_config'] = inference_config
    if args.test_number is not None:
        kwargs['test_number'] = args.test_number
    if args.task_ids is not None:
        kwargs['task_ids'] = args.task_ids

    create_batch_inference_input(**kwargs)


if __name__ == "__main__":
    main()
