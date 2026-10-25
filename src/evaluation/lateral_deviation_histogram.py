from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.trajectory_evaluator_all import EXPERIMENTS

DATA_ROOT = Path("../data")
AIRPORTS = ["CYVR", "CYUL", "CYYZ"]
OUT_DIR = DATA_ROOT / "plots"

LATERAL_TOL_M = 15.0
WINDOW_TOL_M = 2500.0
BIN_WIDTH_M = 500.0

COLOR, INK, INK_MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def load_lateral(exp: dict) -> np.ndarray:
    parts = []
    for icao in AIRPORTS:
        p = DATA_ROOT / icao / exp["output_dir_name"] / f"{Path(exp['traj_filename']).stem}_evaluation.csv"
        if p.exists():
            parts.append(pd.read_csv(p)["lateral_dev_m"].dropna().abs().to_numpy())
    return np.concatenate(parts) if parts else np.array([])


if __name__ == "__main__":
    data = {exp["label"]: load_lateral(exp) for exp in EXPERIMENTS}
    data = {k: v for k, v in data.items() if len(v)}
    if not data:
        raise FileNotFoundError(f"no *_evaluation.csv found under {DATA_ROOT} -> run the evaluation main first")

    x_hi = float(np.ceil(max(v.max() for v in data.values()) / BIN_WIDTH_M) * BIN_WIDTH_M)
    bins = np.arange(0.0, x_hi + BIN_WIDTH_M, BIN_WIDTH_M)

    cols = min(3, len(data))
    rows = int(np.ceil(len(data) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True, squeeze=False)

    for i, (ax, (label, v)) in enumerate(zip(axes.flat, data.items())):
        ax.hist(v, bins=bins, weights=np.full(len(v), 100.0 / len(v)), color=COLOR, edgecolor="white", linewidth=0.5)
        for x in (LATERAL_TOL_M, WINDOW_TOL_M):
            ax.axvline(x, color=INK_MUTED, lw=1.0, ls=(0, (2, 2)))
        ax.set_xlim(0.0, x_hi)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8, labelbottom=True)
        ax.set_title(f"{label}\nn={len(v)}  median={np.median(v):.1f} m  <=15 m: {100 * np.mean(v <= LATERAL_TOL_M):.1f} %",
                     fontsize=9, color=INK, loc="left")
        if i % cols == 0:
            ax.set_ylabel(f"flights [%] per {BIN_WIDTH_M:g} m", fontsize=9, color=INK_MUTED)
        if i + cols >= len(data):
            ax.set_xlabel("lateral deviation at closest point [m]", fontsize=9, color=INK_MUTED)
    for ax in axes.flat[len(data):]:
        ax.axis("off")

    fig.suptitle("Lateral deviation per experiment - dashed: 15 m strict / 2500 m window tolerance", fontsize=11, color=INK)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "lateral_deviation_all.png", dpi=200)
    fig.savefig(OUT_DIR / "lateral_deviation_all.pdf")
    print(f"[saved] {OUT_DIR / 'lateral_deviation_all.png'}")
    plt.show()