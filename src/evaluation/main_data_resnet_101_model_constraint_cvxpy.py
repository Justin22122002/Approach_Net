from evaluation.main_data_resnet_101_model import ResNet101ActionRollout

if __name__ == "__main__":
    RESULT_DIR = "ml/results/multiple_airports_landing_action_resnet101_torchvision/2026-10-10_20-02-35"
    ResNet101ActionRollout(
        RESULT_DIR,
        constrained=True
    ).run_airports(
        output_folder="multiple_airports_resnet101_rollout_constraint",
        data_root="./data",
    )

    # new model 2026-10-11_05-34-36
    RESULT_DIR = "ml/results/multiple_airports_landing_action_constraint_resnet101/2026-10-11_05-34-36"
    ResNet101ActionRollout(
        RESULT_DIR,
        constrained=True,
        best_model_name="best_base_model.pth"
    ).run_airports(
        output_folder="multiple_airports_resnet101_rollout_constraint_training",
        data_root="./data",
    )