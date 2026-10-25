import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_run_args(extra_args: tuple[str, ...] = ()) -> argparse.Namespace:
    """
    Shared CLI for the training / evaluation scripts (used by run_all_trainings.sh).

        --pretrained-dir NAME   results folder name (timestamp) of the pretrained run -> overrides TIME_STAMP / GNN_TIME
        --test                  dry run: print the resolved parameters, report a dummy results folder and exit (no training)
        --result-file PATH      the results folder name of this run is written to this file (read by the shell script)
        --no-show               don't open plot windows (for the pipeline)
        --<extra>               additional string args, e.g. extra_args=("run", "constraint-run") -> args.run, args.constraint_run
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=None,
        help="results folder name (timestamp) of the pretrained run, overrides TIME_STAMP"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="dry run: no training, report a dummy results folder and exit"
    )

    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="file the results folder name is written to"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="don't open plot windows (plots are only saved)"
    )

    for name in extra_args:
        parser.add_argument(f"--{name}", type=str, default=None)

    return parser.parse_args()


def report_results_folder(folder_name: str, result_file: str | None) -> None:
    print(f"RESULTS_FOLDER={folder_name}", flush=True)
    if result_file:
        Path(result_file).write_text(folder_name)


def exit_if_test(args: argparse.Namespace, script_file: str, **resolved) -> None:
    """In --test mode: print what the script would use, report a dummy timestamp folder and exit."""
    if not args.test:
        return

    name = Path(script_file).stem
    print(f"[TEST] {name}")
    print(f"[TEST]   --pretrained-dir = {args.pretrained_dir}")
    for key, value in resolved.items():
        print(f"[TEST]   {key} = {value}")

    dummy_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_results_folder(dummy_timestamp, args.result_file)
    sys.exit(0)