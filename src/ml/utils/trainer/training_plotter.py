from pathlib import Path

import matplotlib.pyplot as plt
from dataclasses import fields

from ml.utils.trainer.training_metrics import BaseTrainingMetrics, TrainingMetrics


class TrainingPlotter[METRICS: BaseTrainingMetrics]:
    def __init__(self, metrics: list[METRICS]) -> None:
        self.metrics: list[METRICS] = metrics

    def plot_all(
            self,
            is_classification: bool = False,
            groups: dict[str, list[str]] = None,
            group_by_type: bool = True,
            save_path: str | Path = None
    ) -> None:
        """
        does not save a pig of the figure, when no save_path is given
        """
        if not self.metrics:
            print("No metrics found to plot.")
            return

        all_fields: list[str] = [f.name for f in fields(self.metrics[0]) if f.name != 'epoch']
        epochs: list[int] = [m.epoch for m in self.metrics]

        if groups is not None:
            active_groups: dict[str, list[str]] = {k: v for k, v in groups.items() if v}

        else:
            if group_by_type:
                temp_groups: dict[str, list[str]] = {
                    "Loss": [f for f in all_fields if "loss" in f.lower()],
                    "Metrics": [f for f in all_fields if "loss" not in f.lower()]
                }
            else:
                temp_groups: dict[str, list[str]] = {f: [f] for f in all_fields}

            active_groups: dict[str, list[str]] = {k: v for k, v in temp_groups.items() if v}

        fig, axes = plt.subplots(len(active_groups), 1, figsize=(10, 5 * len(active_groups)), squeeze=False)
        axes = axes.flatten()

        for i, (group_name, field_names) in enumerate(active_groups.items()):
            for field in field_names:
                values = [getattr(m, field) for m in self.metrics]
                axes[i].plot(epochs, values, label=field, marker='o', linestyle='--')

            axes[i].set_title(f'Evolution of {group_name}')
            axes[i].set_xlabel('Epoch')
            label = 'Percentage' if (is_classification and "accuracy" in group_name.lower()) else group_name
            axes[i].set_ylabel(label)
            axes[i].legend()
            axes[i].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    dummy_metrics = [
        TrainingMetrics(
            train_loss=0.5 - (i * 0.04),
            train_accuracy=0.6 + (i * 0.03),
            val_loss=0.55 - (i * 0.03),
            val_accuracy=0.58 + (i * 0.025),
            epoch=i + 1
        ) for i in range(10)
    ]

    plotter = TrainingPlotter(dummy_metrics)

    plotter.plot_all(is_classification=True, group_by_type=True)

    plotter.plot_all(is_classification=True, group_by_type=False)

    custom_groups = {
        "loss values": ["train_loss", "val_loss"],
        "accuracy values": ["train_accuracy", "val_accuracy"]
    }
    plotter.plot_all(is_classification=True, groups=custom_groups)

if __name__ == "__main__":
    main()
