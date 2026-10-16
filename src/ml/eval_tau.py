import math
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm

from ml.pre_process_datasets import load_or_build_dataset
from ml.utils.models.constraint_layer_cvxpy import AircraftCombinedQPLayer
from ml.utils.trainer.constrained_landing_action_trainer import ConstrainedLandingActionTrainer

HIGH_Q: float = 0.9
W_AT_HIGH: float = 0.1

USE_APPROACH_INFO: bool = True

class _LabelPassthrough(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.y_norm: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self.y_norm


def exp_weight(delta: np.ndarray, tau: float) -> np.ndarray:
    finite = np.isfinite(delta)
    return np.where(finite, np.exp(-np.where(finite, delta, 0.0) / tau), 0.0)


def _pad(a: np.ndarray, batch_size: int) -> np.ndarray:
    n = len(a)
    if n == batch_size:
        return a
    return np.concatenate([a, np.repeat(a[-1:], batch_size - n, axis=0)])


@torch.no_grad()
def project_labels(
    dataset,
    qp_layer: AircraftCombinedQPLayer,
    device,
    batch_size: int = 256,
    desc: str = "label-proj",
    use_approach_info: bool = USE_APPROACH_INFO,
) -> dict[int, np.ndarray]:
    passthrough: _LabelPassthrough = qp_layer.base_model
    qp_layer.to(device).eval()
    y_mean = torch.as_tensor(dataset.y_mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(dataset.y_std, dtype=torch.float32, device=device)

    total_batches = sum((len(entry[0]) + batch_size - 1) // batch_size for entry in dataset._cache.values())
    projected: dict[int, np.ndarray] = {}
    with tqdm(total=total_batches, desc=desc, unit="batch") as pbar:
        for f, (X, Y, L, AP) in dataset._cache.items():
            Y_proj = np.empty_like(Y)
            for s in range(0, len(X), batch_size):
                n = len(X[s:s + batch_size])
                x_raw = torch.from_numpy(_pad(X[s:s + batch_size], batch_size)).to(device)
                y_raw = torch.from_numpy(_pad(Y[s:s + batch_size], batch_size)).to(device)
                limits = torch.from_numpy(_pad(L[s:s + batch_size], batch_size)).to(device)
                appr = torch.from_numpy(_pad(AP[s:s + batch_size], batch_size)).to(device)
                passthrough.y_norm = (y_raw - y_mean) / y_std

                # same slicing as ConstrainedLandingActionTrainer._unpack
                if use_approach_info:
                    faf_pos, if_pos, runway_length_m = appr[:, 0:2], appr[:, 2:4], appr[:, 4]
                else:
                    faf_pos = if_pos = runway_length_m = None

                out_phys = qp_layer(
                    torch.zeros(batch_size, x_raw.shape[1], device=device),  # x_norm
                    x_raw[:, 7:10],          # aircraft_state
                    x_raw[:, 0:5],           # aircraft_dyn
                    limits,                  # aircraft_limits
                    x_raw[:, [5, 6, 10]],    # aircraft_pos
                    x_raw[:, [11, 12, 14]],  # runway_pos
                    faf_pos,
                    if_pos,
                    runway_length_m,
                )
                Y_proj[s:s + n] = out_phys[:n].cpu().numpy()
                pbar.update(1)
            projected[f] = Y_proj.astype(np.float32)
    return projected


if __name__ == "__main__":
    ConstrainedLandingActionTrainer.set_seed()

    dataset_name: str = "20260802_0920"
    airport_list_icao: list[str] = ["CYVR", "CYUL", "CYYZ"]
    cap_data: bool = False
    folder_path: Path = Path("./data/")

    device = ConstrainedLandingActionTrainer.get_device()

    per_airport, train_Y = [], []
    for icao in airport_list_icao:
        dataset = load_or_build_dataset(icao, folder_path, dataset_name, cap_data=cap_data)
        train_flights, _ = dataset.split_flights(val_frac=0.2, seed=42)
        per_airport.append((icao, dataset, train_flights))
        train_Y += [dataset._cache[f][1] for f in train_flights]

    all_X = np.concatenate([ds._cache[f][0] for _, ds, fl in per_airport for f in fl])
    all_Y = np.concatenate(train_Y)
    x_mean, x_std = all_X.mean(0), all_X.std(0) + 1e-8
    y_mean, y_std = all_Y.mean(0), all_Y.std(0) + 1e-8
    del all_X, all_Y

    qp_layer = AircraftCombinedQPLayer(
        base_model=_LabelPassthrough(),
        y_mean=y_mean,
        y_std=y_std,
        debug=False,
        solver=cp.MOREAU,  # cp.SCS
    )

    deltas = []
    for icao, dataset, train_flights in per_airport:
        dataset.set_normalization_stats(x_mean, x_std, y_mean, y_std)
        projected = project_labels(dataset, qp_layer, device, desc=f"label-proj {icao}")
        for f in train_flights:
            Y = dataset._cache[f][1]
            deltas.append((((projected[f] - Y) / y_std) ** 2).mean(1))

    delta = np.concatenate(deltas)
    finite = delta[np.isfinite(delta)]

    delta_hi = float(np.quantile(finite, HIGH_Q))
    tau = delta_hi / math.log(1.0 / W_AT_HIGH)
    w = exp_weight(delta, tau)

    print("\n" + "=" * 60)
    print(f"Samples: {len(delta)}   not-finite: {(~np.isfinite(delta)).mean():.2%}")
    print(f"Delta q{int(HIGH_Q * 100)} = {delta_hi:.5f}")
    print(f"w_mean = {w.mean():.3f}   w<{W_AT_HIGH:g}: {(w < W_AT_HIGH).mean():.2%}")
    print(f"-> tau = {tau:.5f}   (q{int(HIGH_Q * 100)} -> w={W_AT_HIGH:g})")
    print("=" * 60)