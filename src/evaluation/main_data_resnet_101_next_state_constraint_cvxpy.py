from evaluation.main_data_resnet_101_next_state import ResNet101NextStateRollout

if __name__ == "__main__":
    RESULT_DIR = "../ml/results/multiple_airports_landing_next_state_resnet101_torchvision/2026-08-06_13-05-41"
    ResNet101NextStateRollout(RESULT_DIR, constrained=True).run_airports("multiple_airports_next_state_resnet101_rollout_constraint")