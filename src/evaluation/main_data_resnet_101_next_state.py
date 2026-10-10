from typing_extensions import override

from evaluation.utils.rollout_evaluator import NextStateRollout
from ml.utils.models.landing_resnet_101 import LandingActionResNet101


class ResNet101NextStateRollout(NextStateRollout):
    name = "resnet101_next_state"

    @override
    def create_model(self, input_dim, output_dim):
        return LandingActionResNet101(input_dim=input_dim, output_dim=output_dim)


if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_next_state_resnet101_torchvision/2026-08-06_13-05-41"
    ResNet101NextStateRollout(RESULT_DIR).run_airports("multiple_airports_next_state_resnet101_rollout")