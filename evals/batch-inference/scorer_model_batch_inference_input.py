"""Generate scorer (judge) model batch inference input from solver output.

Reads solver model output and generates JSONL input for the scorer model.
Each solver response is evaluated against multiple criteria, producing one
scorer input record per (task, criterion) pair.

Reads solver output from: output/solver/<eval_id>/<epoch>/input.jsonl.out
Writes scorer input to: input/scorer/<eval_id>/<epoch>/input.jsonl

Usage:
    uv run scorer_model_batch_inference_input.py --eval_id <eval_id> --epoch <epoch> \\
        [--inference_config '<json>'] [--judge_prompt_version <version>]
"""

from elt_bench_eval.prompts.prompt_getter import get_judge_prompt
from typing import Any
import argparse
import sys
import json
from pathlib import Path
from loguru import logger
import pandas as pd
from elt_bench_eval.dataset import create_inspect_dataset
from elt_bench_eval.criteria import get_criteria_getter, Criterion


def create_batch_inference_input(
    eval_id: str,
    epoch: int,
    solver_output_base_path: Path = Path('./output/solver'), 
    scorer_input_base_path: Path = Path('./input/scorer'),
    inference_config: dict | None = None,
    judge_prompt_version: str = 'v2',
    csv_path: Path = Path("./l2-bench_tasks.csv"),
    resources_dir: Path = Path("./resources_for_tasks"),
) -> None:
    """Generate scorer batch inference input JSONL from solver output.

    For each task in the solver output, creates one scorer input record per
    evaluation criterion. The record ID format is '<task_id>/<criterion_id>'.

    Args:
        eval_id: Evaluation identifier matching the solver output path.
        epoch: Epoch number for this inference run.
        solver_output_base_path: Base directory for solver output files.
        scorer_input_base_path: Base directory for scorer input files.
        inference_config: Model parameters (e.g., {'maxTokens': 4096, 'temperature': 0}).
        judge_prompt_version: Version of judge prompt template to use (e.g., 'v1', 'v2').
        csv_path: Path to the L2-bench tasks CSV file.
        resources_dir: Directory containing task resource files.
    """
    inspect_dataset = create_inspect_dataset(csv_path, resources_dir)
    criteria_getter = get_criteria_getter(csv_path)
    inputs = []

    with open(solver_output_base_path / eval_id / str(epoch) / "input.jsonl.out", encoding='utf-8') as f:
        for solver_line in f:
            solver_item = json.loads(solver_line)
            task_id = solver_item['recordId']
            task = [sample for sample in inspect_dataset if sample.id == task_id][0]
            solver_input = solver_item['modelInput']
            task_text = solver_input['messages'][0]['content'][0]['text']
            
            content_list = solver_item['modelOutput']['output']['message']['content']
            ai_response = None
            for content_item in content_list:
                if content_item.get('text'):
                    ai_response = content_item['text']
                    break

            if not ai_response:
                raise ValueError(f"Solver model fails to generate any content for task_id={task_id}")
            
            reference_answer = task.target
            judge_prompt_template = get_judge_prompt(judge_prompt_version)
            
            criteria = criteria_getter.get_criteria_of_task(task_id)
            
            for criterion in criteria:                
                scorer_prompt = judge_prompt_template.format(
                    task_text=task_text,
                    reference_answer=reference_answer,
                    ai_response=ai_response,
                    criterion_description=criterion.description
                )
                
                model_input: dict[str, Any] = {
                    "messages": [{"role": "user", "content": [{"text": scorer_prompt}]}]
                }
                
                if inference_config:
                    model_input["inferenceConfig"] = inference_config
                
                input = {"recordId": f"{task_id}/{criterion.criterion_id}", "modelInput": model_input}
                
                inputs.append(input)

    
    output_path = scorer_input_base_path / f"{eval_id}" / f"{epoch}" / "input.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for input in inputs:
            json_record = json.dumps(input)
            f.write(json_record + "\n")

    logger.info(f"Created input jsonl for batch inference with {len(inputs)} records")        

            
def main():
    parser = argparse.ArgumentParser(
        description="Generate model batch inference input JSONL."
    )
    parser.add_argument("--eval_id", type=str, required=True, help="Evaluation ID.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    parser.add_argument(
        "--solver_output_base_path",
        type=Path,
        default=None,
        help="Base path for solver output files.",
    )
    parser.add_argument(
        "--scorer_input_base_path",
        type=Path,
        default=None,
        help="Base path for scorer input files.",
    )
    parser.add_argument(
        "--judge_prompt_version",
        type=str,
        default="v1",
        help="Version of the judge prompt to use.",
    )
    parser.add_argument(
        "--csv_path", type=Path, default=None, help="Path to the tasks CSV file."
    )
    parser.add_argument(
        "--resources_dir",
        type=Path,
        default=None,
        help="Directory containing task resources.",
    )
    parser.add_argument(
        "--inference_config",
        type=str,
        default=None,
        help="JSON string representing model parameters.",
    )

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

    kwargs = {
        'eval_id': args.eval_id,
        'epoch': args.epoch,
    }
    if args.solver_output_base_path is not None:
        kwargs['solver_output_base_path'] = args.solver_output_base_path
    if args.scorer_input_base_path is not None:
        kwargs['scorer_input_base_path'] = args.scorer_input_base_path
    if args.judge_prompt_version is not None:
        kwargs['judge_prompt_version'] = args.judge_prompt_version
    if args.csv_path is not None:
        kwargs['csv_path'] = args.csv_path
    if args.resources_dir is not None:
        kwargs['resources_dir'] = args.resources_dir
    if inference_config:
        kwargs['inference_config'] = inference_config

    create_batch_inference_input(**kwargs)


if __name__ == "__main__":
    main()