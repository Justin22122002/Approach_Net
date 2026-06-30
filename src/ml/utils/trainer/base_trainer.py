import pickle
import random
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.utils.trainer.training_metrics import BaseTrainingMetrics, TrainingMetrics
from ml.utils.trainer.training_plotter import TrainingPlotter


class BaseTrainer[INPUTS: Tensor, OUTPUTS: Tensor, LABELS: Tensor, METRICS: BaseTrainingMetrics](metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer, # todo: set model type to a generic argument
        loss: _Loss | Callable[[Tensor, Tensor], Tensor], # torch.nn.modules.loss
        device: torch.device,
        scheduler: LRScheduler | None = None,
        use_early_stopping: bool = True,
        patience: int = 10,
        best_model_name: str = "best_model.pth",
        results_path: Path | None = None,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.optimizer: optim.Optimizer = optimizer
        self.loss: _Loss | Callable[[Tensor, Tensor], Tensor] = loss
        self.device: torch.device = device
        self.scheduler: LRScheduler | None = scheduler
        self.use_early_stopping: bool = use_early_stopping
        self.patience: int = patience
        self.best_model_name: str = best_model_name
        self.best_val_loss: float = float("inf")
        self.patience_counter: float = 0
        self.metrics_list: list[METRICS | TrainingMetrics] = []
        self.results_path: Path | None = results_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.is_classification: bool = self._is_classification_loss(loss)

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor) -> OUTPUTS:
        pass

    @abstractmethod
    def calculate_loss(self, outputs: OUTPUTS, labels: LABELS) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_accuracy(self, outputs: OUTPUTS, labels: LABELS) -> float:
        pass

    @property
    def plotter(self) -> TrainingPlotter:
        return TrainingPlotter(self.metrics_list)

    def plot_metrics(
        self,
        group_by_type: bool,
        groups: dict[str, list[str]] = None
    ) -> None:
        self.plotter.plot_all(
            is_classification=self.is_classification,
            group_by_type=group_by_type,
            groups=groups,
            save_path=self.results_path / "training_plot.png"
        )

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def set_seed(seed: int = 42):
        print(f"Random Seed: {seed}")

        random.seed(seed)  # Python
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # PyTorch CPU
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _is_classification_loss(self, loss_fn: _Loss | Callable) -> bool:
        classification_losses = (
            nn.CrossEntropyLoss,
            nn.NLLLoss,
            nn.BCELoss,
            nn.BCEWithLogitsLoss,
            nn.MultiLabelMarginLoss,
        )
        return isinstance(loss_fn, classification_losses)

    def format_metric(self, value: float) -> str:
        if self.is_classification:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"

    def early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            print("Early stopping triggered!")
            return True
        return False

    def save_best_model(self, val_loss: float) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)
        if val_loss < self.best_val_loss:
            torch.save(self.model.state_dict(), self.results_path / self.best_model_name)
            print(f"Best model updated with validation loss {val_loss:.4f}.")

    def train(
        self,
        train_loader: DataLoader[tuple[INPUTS, OUTPUTS]],
        validation_loader: DataLoader[tuple[INPUTS, OUTPUTS]],
        epochs: int
    ) -> None:
        for epoch in range(epochs):
            self.model.train()
            total_loss: float = 0.0
            total_accuracy: float = 0.0

            num_batches = len(train_loader)

            progress = tqdm(
                iterable=enumerate(train_loader),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for _, batch in progress:
                inputs: INPUTS
                labels: LABELS
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 1. Forward pass
                outputs: OUTPUTS = self.forward_pass(inputs)

                # 2. Calculate loss
                loss = self.calculate_loss(outputs, labels)

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                loss.backward()

                # 5. Optimizer step (gradient descent)
                self.optimizer.step()

                # Calculate training accuracy
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

                progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss: float = total_loss / len(train_loader)
            avg_accuracy: float = total_accuracy / len(train_loader)
            val_loss, val_accuracy = self.evaluate(validation_loader)

            self.metrics_list.append(
                TrainingMetrics(
                    train_loss=avg_loss,
                    train_accuracy=avg_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    epoch=epoch + 1,
                ),
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Metric: {self.format_metric(avg_accuracy)} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Metric: {self.format_metric(val_accuracy)}"
            )

            self.save_best_model(val_loss)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                    self.scheduler.step()

            if self.use_early_stopping and self.early_stopping(val_loss):
                break

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

    def evaluate(self, loader: DataLoader[tuple[INPUTS, OUTPUTS]]) -> tuple[float, float]:
        self.model.eval()
        total_loss: float = 0.0
        total_accuracy: float = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs)
                loss = self.calculate_loss(outputs, labels)
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

        return total_loss / len(loader), total_accuracy / len(loader)

    def save_metrics(self, filename: str = "metrics.pkl") -> None:
        if not self.results_path:
            raise ValueError("results_path is not defined in the Trainer.")

        self.results_path.mkdir(parents=True, exist_ok=True)
        file_path = self.results_path / filename

        with open(file_path, "wb") as f:
            pickle.dump(self.metrics_list, f)

    def load_metrics(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for loading metrics: {path}")

        with open(path, "rb") as f:
            self.metrics_list = pickle.load(f)

