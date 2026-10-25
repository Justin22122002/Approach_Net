from evaluation.main_data_ft_transformer_next_state import FTTransformerNextStateRollout
from ml.utils.run_cli import exit_if_test, parse_run_args

if __name__ == "__main__":
    args = parse_run_args(("run",))

    RUN: str = args.run or "2026-10-11_16-49-30"
    RESULT_DIR = f"ml/results/multiple_airports_landing_next_state_ft_transformer/{RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR)

    FTTransformerNextStateRollout(RESULT_DIR, constrained=True).run_airports(
        output_folder="multiple_airports_ft_transformer_next_state_rollout_constraint",
        data_root="./data",
    )
