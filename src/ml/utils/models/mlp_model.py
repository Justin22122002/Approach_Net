from torch import nn, Tensor

# todo: make an load method
# todo: save and load model config via a base class that supports those methods, or make put the config in a seperate file
class LandingActionMLP(nn.Module):
    def __init__(
        self,
        input_dim:  int   = 19,
        hidden_dim: int   = 128,
        output_dim: int   = 3,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_features: int = input_dim

        for _ in range(n_layers):
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_features = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
