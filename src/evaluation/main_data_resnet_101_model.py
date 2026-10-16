from typing_extensions import override

from evaluation.utils.rollout_evaluator import ActionRollout
from ml.utils.models.landing_resnet_101 import LandingActionResNet101


class ResNet101ActionRollout(ActionRollout):
    name = "resnet101"

    @override
    def create_model(self, input_dim, output_dim):
        return LandingActionResNet101(input_dim=input_dim, output_dim=output_dim)


if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_resnet101_torchvision/2026-10-10_20-02-35"
    ResNet101ActionRollout(RESULT_DIR).run_airports("multiple_airports_resnet101_rollout")