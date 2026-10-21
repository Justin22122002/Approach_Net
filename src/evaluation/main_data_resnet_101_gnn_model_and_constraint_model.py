import cvxpy as cp

from evaluation.utils.multi_rollout_evaluator import MultiAircraftRollout
from ml.utils.run_cli import exit_if_test, parse_run_args


if __name__ == "__main__":
    args = parse_run_args(("run", "constraint-run"))

    # 2026-10-12_21-14-14, 2026-10-16_06-21-55
    RUN: str = args.run or "2026-10-16_06-21-55"
    # 2026-10-13_12-57-59, 2026-10-16_22-17-12
    CONSTRAINT_RUN: str = args.constraint_run or "2026-10-16_22-17-12"

    RESULT_DIR_CONSTRAINT_TRAINING = f"ml/results/multiple_airports_landing_action_constraint_resnet101_torchvision_gnn/{CONSTRAINT_RUN}"
    RESULT_DIR_CONSTRAINT = f"ml/results/multiple_airports_landing_action_resnet101_torchvision_gnn/{RUN}"
    RESULT_DIR = f"ml/results/multiple_airports_landing_action_resnet101_torchvision_gnn/{RUN}"

    exit_if_test(
        args, __file__,
        RESULT_DIR=RESULT_DIR,
        RESULT_DIR_CONSTRAINT=RESULT_DIR_CONSTRAINT,
        RESULT_DIR_CONSTRAINT_TRAINING=RESULT_DIR_CONSTRAINT_TRAINING,
    )

    MultiAircraftRollout(
        RESULT_DIR_CONSTRAINT_TRAINING,
        constrained=True,
        solver=cp.MOREAU,
    ).run_airports(
        output_folder="multiple_airports_resnet_101_gnn_rollout_constraint_training",
        data_root="./data",
    )

    MultiAircraftRollout(
        RESULT_DIR_CONSTRAINT,
        constrained=True,
        solver=cp.MOREAU,
        best_model_name="best_model.pth"
    ).run_airports(
        output_folder="multiple_airports_resnet_101_gnn_rollout_constraint",
        data_root="./data",
    )

    MultiAircraftRollout(
        RESULT_DIR,
        solver=cp.MOREAU,
    ).run_airports(
        output_folder="multiple_airports_resnet_101_gnn_rollout",
        data_root="./data",
    )
