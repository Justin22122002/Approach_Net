import cvxpy as cp

from evaluation.utils.multi_rollout_evaluator import MultiAircraftRollout
from ml.utils.run_cli import exit_if_test, parse_run_args


if __name__ == "__main__":
    args = parse_run_args(("run", "constraint-run"))

    # 2026-10-10_23-39-39, 2026-10-12_11-33-15, 2026-10-16_06-08-04
    RUN: str = args.run or "2026-10-16_06-08-04"
    # 2026-10-11_17-13-54, 2026-10-12_18-10-46, 2026-10-16_20-10-17
    CONSTRAINT_RUN: str = args.constraint_run or "2026-10-16_20-10-17"

    RESULT_DIR_CONSTRAINT_TRAINING = f"ml/results/multiple_airports_landing_action_constraint_ft_transformer_gnn/{CONSTRAINT_RUN}"
    RESULT_DIR_CONSTRAINT = f"ml/results/multiple_airports_landing_action_ft_transformer_gnn/{RUN}"
    RESULT_DIR = f"ml/results/multiple_airports_landing_action_ft_transformer_gnn/{RUN}"

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
        output_folder="multiple_airports_ft_transformer_gnn_rollout_constraint_training",
        data_root="./data",
    )

    MultiAircraftRollout(
        RESULT_DIR_CONSTRAINT,
        constrained=True,
        solver=cp.MOREAU,
        best_model_name="best_model.pth"
    ).run_airports(
        output_folder="multiple_airports_ft_transformer_gnn_rollout_constraint",
        data_root="./data",
    )

    MultiAircraftRollout(
        RESULT_DIR,
        solver=cp.MOREAU,
    ).run_airports(
        output_folder="multiple_airports_ft_transformer_gnn_rollout",
        data_root="./data",
    )
