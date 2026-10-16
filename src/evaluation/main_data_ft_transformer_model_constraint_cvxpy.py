from evaluation.main_data_ft_transformer_model import FTTransformerActionRollout

if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_ft_transformer/2026-10-10_20-33-57"
    FTTransformerActionRollout(
        RESULT_DIR,
        constrained=True,
    ).run_airports("multiple_airports_ft_transformer_rollout_constraint")

    RESULT_DIR = "../ml/results/multiple_airports_landing_action_constraint_ft_transformer/2026-10-11_06-09-41"
    FTTransformerActionRollout(
        RESULT_DIR,
        constrained=True,
        best_model_name="best_base_model.pth",
    ).run_airports("multiple_airports_ft_transformer_rollout_constraint_training")