# removes the warning from the torch geometric library
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit._script")

from pathlib import Path

import torch
from torch import nn, Tensor
from rtdl_revisiting_models import FTTransformer
from torch_geometric.nn import TransformerConv


def fully_connected_edge_index(batch_idx: Tensor) -> Tensor:
    device = batch_idx.device
    edges: list[Tensor] = []
    for b in batch_idx.unique():
        node_idx = (batch_idx == b).nonzero(as_tuple=True)[0]  # (n_b,)
        n_b = node_idx.numel()
        if n_b <= 1:
            continue  # single aircraft -> no edges, TransformerConv still handles it (root_weight)
        src = node_idx.repeat_interleave(n_b)  # (n_b*n_b,)
        dst = node_idx.repeat(n_b)             # (n_b*n_b,)
        mask = src != dst
        edges.append(torch.stack([src[mask], dst[mask]], dim=0))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edges, dim=1)  # (2, E)


class NodeEncoder(nn.Module):
    """
    d_out=None -> forward() returns the raw [CLS] embedding
    input (N, input_dim) -> output (N, d_block)
    """

    def __init__(
        self,
        input_dim: int,
        d_block: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_block: int = d_block
        self.backbone: FTTransformer = FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=None,  # no head -> forward() yields the [CLS] embedding, (N, d_block)
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x, None)  # (N, input_dim) -> (N, d_block)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        input_dim: int,
        d_block: int = 192,
        **kwargs,
    ) -> "NodeEncoder":
        """
        Loads backbone weights from a LandingActionFTTransformer checkpoint
        """
        encoder = cls(input_dim=input_dim, d_block=d_block, **kwargs)

        raw_state: dict[str, Tensor] = torch.load(checkpoint_path, map_location="cpu")
        stripped: dict[str, Tensor] = {
            k[len("model."):]: v for k, v in raw_state.items() if k.startswith("model.")
        }
        own_state: dict[str, Tensor] = encoder.backbone.state_dict()
        filtered: dict[str, Tensor] = {
            k: v for k, v in stripped.items() if k in own_state and v.shape == own_state[k].shape
        }

        missing, _ = encoder.backbone.load_state_dict(filtered, strict=False)
        print(
            f"[NodeEncoder.from_pretrained] loaded {len(filtered)}/{len(own_state)} backbone tensors "
            f"from {checkpoint_path} (missing={len(missing)}, dropped_head_keys={len(stripped) - len(filtered)})"
        )
        return encoder


class MultiAircraftLandingGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,               # 15, MultiAircraftLandingActionDataset.input_dim
        action_dim: int = 3,          # thrust, load_factor, roll
        d_block: int = 192,           # FT-Transformer [CLS] dim == GNN node dim
        n_ft_blocks: int = 3,
        ft_attention_heads: int = 8,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        pretrained_encoder_path: str | Path | None = None,  # None -> encoder trained from scratch
    ) -> None:
        super().__init__()
        assert d_block % gnn_heads == 0, "d_block must be divisible by gnn_heads (TransformerConv concat)"

        self.encoder: NodeEncoder
        if pretrained_encoder_path is not None:
            self.encoder = NodeEncoder.from_pretrained(
                pretrained_encoder_path,
                input_dim=input_dim,
                d_block=d_block,
                n_blocks=n_ft_blocks,
                attention_n_heads=ft_attention_heads,
            )
        else:
            self.encoder = NodeEncoder(
                input_dim=input_dim,
                d_block=d_block,
                n_blocks=n_ft_blocks,
                attention_n_heads=ft_attention_heads,
            )

        # heads * (d_block // heads) == d_block -> residual add below stays shape-consistent
        self.convs: nn.ModuleList = nn.ModuleList([
            TransformerConv(
                in_channels=d_block,
                out_channels=d_block // gnn_heads,
                heads=gnn_heads,
                dropout=gnn_dropout
            )
            for _ in range(gnn_layers)
        ])
        self.norms: nn.ModuleList = nn.ModuleList([nn.LayerNorm(d_block) for _ in range(gnn_layers)])

        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(d_block, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: Tensor, batch_idx: Tensor) -> Tensor:
        # x: (N, input_dim), batch_idx: (N,) graph id per node -> out: (N, action_dim)
        edge_index = fully_connected_edge_index(batch_idx)  # (2, E)

        h = self.encoder(x)  # (N, d_block)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index))  # (N, d_block), residual + PostNorm

        return self.head(h)  # (N, action_dim)


if __name__ == "__main__":
    torch.manual_seed(0)

    sizes = [5, 3, 1]  # 3 timesteps with 5 / 3 / 1 aircraft present
    input_dim = 15
    N = sum(sizes)

    x = torch.randn(N, input_dim)
    batch_idx = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    y = torch.randn(N, 3)

    model = MultiAircraftLandingGNN(input_dim=input_dim)
    out = model(x, batch_idx)
    print(f"x: {tuple(x.shape)}  batch_idx: {tuple(batch_idx.shape)}  ->  out: {tuple(out.shape)}")

    loss = nn.functional.mse_loss(out, y)
    loss.backward()

    n_params = sum(1 for _ in model.parameters())
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"params: {n_params}  with grad: {n_with_grad}")
    assert n_with_grad == n_params, "gradients did not reach every parameter!"
    print("OK - gradients flow through encoder, GNN layers and head.")