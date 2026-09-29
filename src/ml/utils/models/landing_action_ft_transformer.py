import torch
from torch import nn
from rtdl_revisiting_models import FTTransformer


class LandingActionFTTransformer(nn.Module):
    """
    implementation (Gorishniy et al., "Revisiting Deep Learning Models for
    Tabular Data", NeurIPS 2021; package: rtdl_revisiting_models).

    note: All 15 input features are continuous -- no categorical features here.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 3,
        n_blocks: int = 3,
        d_block: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model: FTTransformer = FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=action_dim, # d_out=None,  # no Head -> forward() yields CLS-Embedding
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim), all continuous -> x_cat=None
        return self.model(x, None)