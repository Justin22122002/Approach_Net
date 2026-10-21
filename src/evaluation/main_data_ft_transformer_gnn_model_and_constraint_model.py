import cvxpy as cp

from evaluation.utils.multi_rollout_evaluator import MultiAircraftRollout

# 2026-10-11_17-13-54, 2026-10-12_18-10-46, 2026-10-16_20-10-17
RESULT_DIR_CONSTRAINT_TRAINING = "ml/results/multiple_airports_landing_action_constraint_ft_transformer_gnn/2026-10-16_20-10-17"
# 2026-10-10_23-39-39, 2026-10-12_11-33-15, 2026-10-16_06-08-04
RESULT_DIR_CONSTRAINT = "ml/results/multiple_airports_landing_action_ft_transformer_gnn/2026-10-16_06-08-04"
RESULT_DIR = "ml/results/multiple_airports_landing_action_ft_transformer_gnn/2026-10-16_06-08-04"


if __name__ == "__main__":
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
