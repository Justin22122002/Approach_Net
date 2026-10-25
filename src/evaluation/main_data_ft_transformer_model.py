from typing_extensions import override

from evaluation.utils.rollout_evaluator import ActionRollout
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from ml.utils.run_cli import exit_if_test, parse_run_args


class FTTransformerActionRollout(ActionRollout):
    name = "ft_transformer"

    @override
    def create_model(self, input_dim, output_dim):
        return LandingActionFTTransformer(input_dim=input_dim, action_dim=output_dim)


if __name__ == "__main__":
    args = parse_run_args(("run",))

    # 2026-08-05_21-33-34
    RUN: str = args.run or "2026-10-10_20-33-57"
    RESULT_DIR = f"ml/results/multiple_airports_landing_action_ft_transformer/{RUN}"

    exit_if_test(args, __file__, RESULT_DIR=RESULT_DIR)

    FTTransformerActionRollout(RESULT_DIR).run_airports(
        output_folder="multiple_airports_ft_transformer_rollout",
        data_root="./data",
    )
