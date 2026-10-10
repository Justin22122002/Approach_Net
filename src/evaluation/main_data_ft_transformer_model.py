from typing_extensions import override

from evaluation.utils.rollout_evaluator import ActionRollout
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer


class FTTransformerActionRollout(ActionRollout):
    name = "ft_transformer"

    @override
    def create_model(self, input_dim, output_dim):
        return LandingActionFTTransformer(input_dim=input_dim, action_dim=output_dim)


if __name__ == "__main__":
    # 2026-08-06_07-08-10, 2026-08-05_21-33-34 -> flip augmentation
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_ft_transformer_augmented/2026-08-05_21-33-34"
    FTTransformerActionRollout(RESULT_DIR).run_airports("multiple_airports_ft_transformer_rollout")