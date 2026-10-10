from evaluation.main_data_resnet_101_model import ResNet101ActionRollout

if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_resnet101_torchvision_augmented/2026-08-05_14-54-31"
    ResNet101ActionRollout(
        RESULT_DIR,
        constrained=True
    ).run_airports("multiple_airports_resnet101_rollout_constraint")

    # 2026-10-05_02-28-26, 2026-10-05_02-31-22
    RESULT_DIR = "../ml/results/multiple_airports_landing_action_constrained_resnet101/2026-10-05_02-31-22"
    ResNet101ActionRollout(
        RESULT_DIR,
        constrained=True,
        best_model_name="best_base_model.pth"
    ).run_airports("multiple_airports_resnet101_rollout_constraint_training")