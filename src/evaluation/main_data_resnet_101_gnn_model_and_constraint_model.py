import cvxpy as cp

from evaluation.utils.multi_rollout_evaluator import MultiAircraftRollout

# 2026-10-13_12-57-59, 2026-10-16_22-17-12
RESULT_DIR_CONSTRAINT_TRAINING = "ml/results/multiple_airports_landing_action_constraint_resnet101_torchvision_gnn/2026-10-16_22-17-12"

# 2026-10-12_21-14-14, 2026-10-16_06-21-55
RESULT_DIR_CONSTRAINT = "ml/results/multiple_airports_landing_action_resnet101_torchvision_gnn/2026-10-16_06-21-55"
RESULT_DIR = "ml/results/multiple_airports_landing_action_resnet101_torchvision_gnn/2026-10-16_06-21-55"


if __name__ == "__main__":
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
