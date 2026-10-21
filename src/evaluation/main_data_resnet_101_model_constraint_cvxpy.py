from evaluation.main_data_resnet_101_model import ResNet101ActionRollout
from ml.utils.run_cli import exit_if_test, parse_run_args

if __name__ == "__main__":
    args = parse_run_args(("run", "constraint-run"))

    RUN: str = args.run or "2026-10-10_20-02-35"
    CONSTRAINT_RUN: str = args.constraint_run or "2026-10-11_05-34-36"

    RESULT_DIR = f"ml/results/multiple_airports_landing_action_resnet101_torchvision/{RUN}"
    RESULT_DIR_CONSTRAINT_TRAINING = f"ml/results/multiple_airports_landing_action_constraint_resnet101/{CONSTRAINT_RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR, RESULT_DIR_CONSTRAINT_TRAINING=RESULT_DIR_CONSTRAINT_TRAINING)

    ResNet101ActionRollout(
        RESULT_DIR,
        constrained=True
    ).run_airports(
        output_folder="multiple_airports_resnet101_rollout_constraint",
        data_root="./data",
    )

    ResNet101ActionRollout(
        RESULT_DIR_CONSTRAINT_TRAINING,
        constrained=True,
        best_model_name="best_base_model.pth"
    ).run_airports(
        output_folder="multiple_airports_resnet101_rollout_constraint_training",
        data_root="./data",
    )
