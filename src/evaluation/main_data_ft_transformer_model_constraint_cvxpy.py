from evaluation.main_data_ft_transformer_model import FTTransformerActionRollout

if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_ft_transformer_augmented/2026-08-05_21-33-34"
    FTTransformerActionRollout(
        RESULT_DIR,
        constrained=True,
    ).run_airports("multiple_airports_ft_transformer_rollout_constraint")

    # 2026-10-04_01-07-00, 2026-10-03_15-01-43 2026-10-06_20-40-23
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_constrained_ft_transformer/2026-10-04_01-07-00"
    FTTransformerActionRollout(
        RESULT_DIR,
        constrained=True,
        best_model_name="best_base_model.pth",
    ).run_airports("multiple_airports_ft_transformer_rollout_constraint_training")