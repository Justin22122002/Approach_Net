import torch
from torch import nn
from torchvision.models import resnet101


class LandingActionResNet101(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3) -> None:
        super().__init__()
        self.backbone = resnet101(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B, input_dim) -> (B, input_dim, 1, 1)
        return self.backbone(x)