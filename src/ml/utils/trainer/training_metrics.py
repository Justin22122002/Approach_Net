from dataclasses import dataclass

@dataclass
class BaseTrainingMetrics:
    epoch: int

@dataclass
class TrainingMetrics(BaseTrainingMetrics):
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
