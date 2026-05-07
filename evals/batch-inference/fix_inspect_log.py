import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from loguru import logger


def fix_inspect_log(log_path: Path) -> None:
    log = read_eval_log(str(log_path))
    modified = False

    # Fix negative working_time values
    for sample in log.samples:
        if sample.working_time is not None and sample.working_time < 0:
            logger.info(f"Sample {sample.id}: working_time {sample.working_time} -> 0.0")
            sample.working_time = 0.0
            modified = True

    # Strip "replay/" prefix from model name
    if log.eval.model.startswith("replay/"):
        old_model = log.eval.model
        log.eval.model = log.eval.model[7:]  # len("replay/") == 7
        logger.info(f"Model: {old_model} -> {log.eval.model}")
        modified = True

    if modified:
        write_eval_log(log, str(log_path))
        logger.info(f"Fixed {log_path}")
    else:
        logger.info(f"No fixes needed for {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix inspect-ai log files for EEE conversion")
    parser.add_argument("log_path", type=Path, help="Path to .eval log file")
    args = parser.parse_args()

    if not args.log_path.exists():
        logger.error(f"File not found: {args.log_path}")
        return

    fix_inspect_log(args.log_path)


if __name__ == "__main__":
    main()
