from torch import nn, Tensor
from iTransformer.iTransformer import iTransformer


class LandingActionITransformer(nn.Module):
    def __init__(
        self,
        num_variates: int,
        lookback_len: int,
        dim: int = 128,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 32,
        action_dim: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.backbone = iTransformer(
            num_variates=num_variates,
            lookback_len=lookback_len,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            pred_length=1,
            num_tokens_per_variate=1,
            num_mem_tokens=0,
            use_reversible_instance_norm=False,
            attn_dropout=dropout,
            ff_dropout=dropout,
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(num_variates * dim),
            nn.Linear(num_variates * dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, action_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        b = self.backbone
        assert x.shape[1:] == (b.lookback_len, b.num_variates)

        x = x.transpose(1, 2)      # b n v -> b v n
        x = b.mlp_in(x)            # b v n -> b v d   (num_tokens_per_variate=1)

        x = b.expand_streams(x)
        first_values = None
        for attn, ff in b.layers:
            attn_out, values = attn(x, value_residual=first_values)
            first_values = first_values if first_values is not None else values
            x = x + attn_out
            x = ff(x) + x
        x = b.reduce_streams(x)

        return x  # (B, num_variates, dim)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.encode(x)          # (B, V, D)
        return self.action_head(tokens.flatten(1))  # (B, action_dim)