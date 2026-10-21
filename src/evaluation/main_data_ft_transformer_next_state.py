from typing_extensions import override

from evaluation.utils.rollout_evaluator import NextStateRollout
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from ml.utils.run_cli import exit_if_test, parse_run_args


class FTTransformerNextStateRollout(NextStateRollout):
    name = "ft_transformer_next_state"

    @override
    def create_model(self, input_dim, output_dim):
        return LandingActionFTTransformer(input_dim=input_dim, action_dim=output_dim)


if __name__ == "__main__":
    args = parse_run_args(("run",))

    RUN: str = args.run or "2026-10-11_16-49-30"
    RESULT_DIR = f"ml/results/multiple_airports_landing_next_state_ft_transformer/{RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR)

    FTTransformerNextStateRollout(RESULT_DIR).run_airports(
        output_folder="multiple_airports_ft_transformer_next_state_rollout",
        data_root="./data",
    )
