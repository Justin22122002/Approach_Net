from torch import nn, Tensor

# todo: make an load method
# todo: save and load model config via a base class that supports those methods, or make put the config in a seperate file
class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))


class LandingActionResNet(nn.Module):
    def __init__(
        self,
        input_dim:  int   = 19,
        hidden_dim: int   = 256,
        output_dim: int   = 3,
        n_blocks:   int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)