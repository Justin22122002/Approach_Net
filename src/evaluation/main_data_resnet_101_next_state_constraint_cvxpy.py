from evaluation.main_data_resnet_101_next_state import ResNet101NextStateRollout
from ml.utils.run_cli import exit_if_test, parse_run_args

if __name__ == "__main__":
    args = parse_run_args(("run",))

    RUN: str = args.run or "2026-10-11_18-23-44"
    RESULT_DIR = f"ml/results/multiple_airports_landing_next_state_resnet101_torchvision/{RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR)

    ResNet101NextStateRollout(RESULT_DIR, constrained=True).run_airports(
        output_folder="multiple_airports_next_state_resnet101_rollout_constraint",
        data_root="./data",
    )
