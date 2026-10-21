from evaluation.main_data_ft_transformer_next_state import FTTransformerNextStateRollout

if __name__ == "__main__":
    RESULT_DIR = "ml/results/multiple_airports_landing_next_state_ft_transformer/2026-10-11_16-49-30"
    FTTransformerNextStateRollout(RESULT_DIR, constrained=True).run_airports(
        output_folder="multiple_airports_ft_transformer_next_state_rollout_constraint",
        data_root="./data",
    )