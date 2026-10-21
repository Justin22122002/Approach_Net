from evaluation.main_data_ft_transformer_model import FTTransformerActionRollout
from ml.utils.run_cli import exit_if_test, parse_run_args

if __name__ == "__main__":
    args = parse_run_args(("run", "constraint-run"))

    RUN: str = args.run or "2026-10-10_20-33-57"
    CONSTRAINT_RUN: str = args.constraint_run or "2026-10-11_06-09-41"

    RESULT_DIR = f"ml/results/multiple_airports_landing_action_ft_transformer/{RUN}"
    RESULT_DIR_CONSTRAINT_TRAINING = f"ml/results/multiple_airports_landing_action_constraint_ft_transformer/{CONSTRAINT_RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR, RESULT_DIR_CONSTRAINT_TRAINING=RESULT_DIR_CONSTRAINT_TRAINING)

    FTTransformerActionRollout(
        RESULT_DIR,
        constrained=True,
    ).run_airports(
        output_folder="multiple_airports_ft_transformer_rollout_constraint",
        data_root="./data",
    )

    FTTransformerActionRollout(
        RESULT_DIR_CONSTRAINT_TRAINING,
        constrained=True,
        best_model_name="best_base_model.pth",
    ).run_airports(
        output_folder="multiple_airports_ft_transformer_rollout_constraint_training",
        data_root="./data",
    )
