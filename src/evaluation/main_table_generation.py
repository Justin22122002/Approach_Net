from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("resnet101", "ft_transformer")

BACKBONE_LABELS = {
    "resnet101": "ResNet-101",
    "ft_transformer": "FT-Transformer",
}

BACKBONE_CITATIONS = {
    "resnet101": "he2016deep",                    # He et al., Deep Residual Learning (ResNet)
    "ft_transformer": "gorishniy2021revisiting",  # Gorishniy et al., Revisiting Deep Learning Models for Tabular Data
}
OPTIMIZER_CITATION = "wachter2006ipopt"
OPTIMIZER_LABEL = "Ipopt-Optimizer"
OPTIMIZER_LAYER_MODE = "rnav_optimizer"

OURS_LABEL = "({\\bf ours})"

# (dynamics, hard-constraint layer, H-layer during training) per row, in table order
VARIANTS: tuple[tuple[bool, bool, bool], ...] = (
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
)

LAYER_MODE_VALUE = {
    ("resnet101", False, False, False): "resnet101_next_state_baseline",
    ("resnet101", True, False, False): "resnet101_action_baseline",
    ("resnet101", False, True, False): "resnet101_next_state_constraint",
    ("resnet101", True, True, False): "resnet101_action_constraint",
    ("resnet101", True, True, True): "resnet101_action_constraint_training",
    ("ft_transformer", False, False, False): "ft_transformer_next_state_baseline",
    ("ft_transformer", True, False, False): "ft_transformer_action_baseline",
    ("ft_transformer", False, True, False): "ft_transformer_next_state_constraint",
    ("ft_transformer", True, True, False): "ft_transformer_action_constraint",
    ("ft_transformer", True, True, True): "ft_transformer_action_constraint_training",
}

# (column in pooled row, header, higher_is_better)
RATE_COLUMNS = [
    ("succ_rate_s1", "\\shortstack{\\texttt{succ@1}\\\\(\\%)}", True),
    ("succ_rate_s2", "\\shortstack{\\texttt{succ@2}\\\\(\\%)}", True),
    ("succ_rate_s3", "\\shortstack{\\texttt{succ@3}\\\\(\\%)}", True),
    ("window_auc", "\\shortstack{AUC\\\\(\\%)}", True),
]

# (key, header, digits, higher_is_better)
METRICS = [
    ("lateral_dev_m", "\\shortstack{Lateral\\\\Dev. (m)}", 2, False),
    ("vertical_dev_m", "\\shortstack{Vertical\\\\Dev. (m)}", 2, False),
    ("angle_dev_deg", "\\shortstack{HDG ($\\chi$)\\\\Dev. ($^\\circ$)}", 2, False),
    ("gamma_dev_deg", "\\shortstack{GP ($\\gamma$)\\\\Dev. ($^\\circ$)}", 2, False),
    ("velocity_dev_ms", "\\shortstack{Velocity\\\\Dev. (m/s)}", 2, False),
    ("landing_time_s", "\\shortstack{Landing\\\\Time (s)}", 2, False),
]
SOLVING_TIME_HEADER = "\\shortstack{Solving\\\\Time (s)}"

ALL_PREFIX = "all_"
SUCCESS_PREFIX = "success_only_"

N_LABEL_COLS = 4  # Model, D, H, T
N_COLS = N_LABEL_COLS + len(RATE_COLUMNS) + len(METRICS) + 1

CHECK = "\\checkmark"
CROSS = "--"


def pooled_mean_std(n: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, float]:
    valid = (n > 0) & np.isfinite(mean)
    n, mean, std = n[valid], mean[valid], np.nan_to_num(std[valid])
    if n.sum() == 0:
        return float("nan"), float("nan")

    pooled_mean = np.sum(n * mean) / n.sum()
    pooled_var = np.sum(n * (std ** 2 + (mean - pooled_mean) ** 2)) / n.sum()
    return float(pooled_mean), float(np.sqrt(pooled_var))


def pool_rows(sub: pd.DataFrame, prefix: str, n_col: str) -> dict:
    n_total = int(sub["n_total"].sum()) if len(sub) else 0
    row: dict = {"n_total": n_total}

    weights = sub["n_total"].to_numpy(dtype=float) if len(sub) else np.array([])
    for col, _, _ in RATE_COLUMNS:
        if n_total > 0 and col in sub:
            row[col] = 100.0 * float(np.average(sub[col].to_numpy(dtype=float), weights=weights))
        else:
            row[col] = float("nan")

    for key, _, _, _ in METRICS:
        if len(sub) == 0:
            row[f"{key}_mean"], row[f"{key}_std"] = float("nan"), float("nan")
            continue
        n = sub[n_col].fillna(0).to_numpy(dtype=float)
        mean = sub[f"{prefix}{key}_mean"].to_numpy(dtype=float)
        std = sub[f"{prefix}{key}_std"].to_numpy(dtype=float)
        row[f"{key}_mean"], row[f"{key}_std"] = pooled_mean_std(n, mean, std)

    return row


def pool_layer_mode(df: pd.DataFrame, layer_mode_value: str, prefix: str, n_col: str) -> dict:
    sub = df[df["layer_mode"] == layer_mode_value]
    if sub.empty:
        print(f"[WARN] no rows for layer_mode '{layer_mode_value}' -> '--'")
    return pool_rows(sub, prefix, n_col)


BEST_SHADE = "gray!12"
SECOND_SHADE = "gray!12"


def _highlight(text: str, rank: int) -> str:
    if rank == 1:
        return f"\\cellcolor{{{BEST_SHADE}}}\\textbf{{{text}}}"
    if rank == 2:
        return f"\\cellcolor{{{SECOND_SHADE}}}\\textit{{{text}}}"
    return text


def fmt(mean: float, std: float, digits: int, show_std: bool = False, rank: int = 0) -> str:
    if np.isnan(mean):
        return "--"
    text = f"{mean:.{digits}f}" if not show_std else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    return _highlight(text, rank)


def fmt_rate(rate: float, rank: int = 0) -> str:
    if np.isnan(rate):
        return "--"
    return _highlight(f"{rate:.2f}", rank)


def _rank_indices(values: list[float], higher_is_better: bool, digits: int = 2) -> dict[int, int]:
    """row index -> 1 (best) / 2 (second best), compared on the printed (rounded) values; ties share a rank"""
    finite = [(i, round(v, digits)) for i, v in enumerate(values) if not np.isnan(v)]
    distinct = sorted({v for _, v in finite}, reverse=higher_is_better)
    ranks: dict[int, int] = {}
    for rank, val in enumerate(distinct[:2], start=1):
        ranks.update({i: rank for i, v in finite if v == val})
    return ranks


def _build_table_rows(df: pd.DataFrame, prefix: str, n_col: str, show_std: bool) -> list[str]:
    rows: list[tuple[str, tuple[bool, bool, bool], dict]] = []
    for model in MODELS:
        for variant in VARIANTS:
            pooled = pool_layer_mode(df, LAYER_MODE_VALUE[(model, *variant)], prefix, n_col)
            rows.append((model, variant, pooled))

    optimizer_row = pool_rows(df[df["layer_mode"] == OPTIMIZER_LAYER_MODE], prefix, n_col)

    # best / second best over the learned methods only (optimizer row is a reference, never highlighted)
    rate_ranks = {
        col: _rank_indices([p[col] for _, _, p in rows], higher) for col, _, higher in RATE_COLUMNS
    }
    metric_ranks = {
        key: _rank_indices([p[f"{key}_mean"] for _, _, p in rows], higher, digits)
        for key, _, digits, higher in METRICS
    }

    def emit(label_cells: list[str], pooled: dict, row_idx: int | None) -> str:
        rate_cells = [
            fmt_rate(pooled[col], rank=rate_ranks[col].get(row_idx, 0))
            for col, _, _ in RATE_COLUMNS
        ]
        metric_cells = [
            fmt(pooled[f"{key}_mean"], pooled[f"{key}_std"], digits, show_std,
                rank=metric_ranks[key].get(row_idx, 0))
            for key, _, digits, _ in METRICS
        ]
        solving_cell = "--"  # TODO: solving time
        return " & ".join(label_cells + rate_cells + metric_cells + [solving_cell]) + " \\\\"

    lines: list[str] = []
    prev_model: str | None = None
    prev_h: bool | None = None
    for row_idx, (model, (dyn, hard, trained), pooled) in enumerate(rows):
        new_model_block = model != prev_model
        new_h_block = new_model_block or hard != prev_h

        if prev_model is not None:
            if new_model_block:
                lines.append("\\midrule")
            elif new_h_block:
                lines.append(f"\\cmidrule(lr){{2-{N_COLS}}}")

        backbone = BACKBONE_LABELS[model]
        if new_h_block:
            if hard:
                label = f"ApproachNet ({backbone})"
            else:
                citation = BACKBONE_CITATIONS.get(model)
                label = f"{backbone}~\\cite{{{citation}}}" if citation else backbone
        elif hard and dyn and not trained:
            label = OURS_LABEL
        else:
            label = ""

        label_cells = [
            label,
            CHECK if dyn else CROSS,
            CHECK if hard else CROSS,
            CHECK if trained else CROSS,
        ]
        lines.append(emit(label_cells, pooled, row_idx))

        prev_model, prev_h = model, hard

    lines.append("\\midrule")
    optimizer_label = f"{OPTIMIZER_LABEL}~\\cite{{{OPTIMIZER_CITATION}}}"
    lines.append(emit([optimizer_label, CHECK, CHECK, CROSS], optimizer_row, None))

    return lines


# ----------------------------------------------------------------------------------------------------------
# multi-aircraft table: only the three GNN models (joint scene, separation QP active in the H rows)
# ----------------------------------------------------------------------------------------------------------
MULTI_BACKBONE_LABEL = "FT-Transformer-GNN"
MULTI_BACKBONE_CITATION: str | None = None   # e.g. "velickovic2018graph" -- None -> no \\cite

# (label, layer_mode, D, H, T) in table order
MULTI_ROWS: tuple[tuple[str, str, bool, bool, bool], ...] = (
    (MULTI_BACKBONE_LABEL, "ft_gnn_baseline", True, False, False),
    (f"ApproachNet ({MULTI_BACKBONE_LABEL})", "ft_gnn_constraint", True, True, False),
    ("", "ft_gnn_constraint_training", True, True, True),
)


def _emit_row(label_cells: list[str], pooled: dict, row_idx: int | None,
              rate_ranks: dict, metric_ranks: dict, show_std: bool) -> str:
    rate_cells = [fmt_rate(pooled[col], rank=rate_ranks[col].get(row_idx, 0)) for col, _, _ in RATE_COLUMNS]
    metric_cells = [
        fmt(pooled[f"{key}_mean"], pooled[f"{key}_std"], digits, show_std, rank=metric_ranks[key].get(row_idx, 0))
        for key, _, digits, _ in METRICS
    ]
    solving_cell = "--"  # TODO: solving time
    return " & ".join(label_cells + rate_cells + metric_cells + [solving_cell]) + " \\\\"


def build_multi_table(df: pd.DataFrame, over_all_flights: bool = True, show_std: bool = False) -> str:
    prefix = ALL_PREFIX if over_all_flights else SUCCESS_PREFIX
    n_col = "all_n" if over_all_flights else "success_only_n"

    pooled_rows = [pool_layer_mode(df, mode, prefix, n_col) for _, mode, *_ in MULTI_ROWS]
    rate_ranks = {col: _rank_indices([p[col] for p in pooled_rows], higher) for col, _, higher in RATE_COLUMNS}
    metric_ranks = {
        key: _rank_indices([p[f"{key}_mean"] for p in pooled_rows], higher, digits)
        for key, _, digits, higher in METRICS
    }

    header = (
        ["Model", "D", "H", "T"]
        + [h for _, h, _ in RATE_COLUMNS]
        + [h for _, h, _, _ in METRICS]
        + [SOLVING_TIME_HEADER]
    )

    body: list[str] = []
    prev_h: bool | None = None
    for row_idx, ((label, _, dyn, hard, trained), pooled) in enumerate(zip(MULTI_ROWS, pooled_rows)):
        if prev_h is not None and hard != prev_h:
            body.append(f"\\cmidrule(lr){{2-{N_COLS}}}")
        if label == MULTI_BACKBONE_LABEL and MULTI_BACKBONE_CITATION:
            label = f"{label}~\\cite{{{MULTI_BACKBONE_CITATION}}}"
        cells = [label] + [CHECK if f else CROSS for f in (dyn, hard, trained)]
        body.append(_emit_row(cells, pooled, row_idx, rate_ranks, metric_ranks, show_std))
        prev_h = hard

    lines: list[str] = [
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l ccc " + "c" * (N_COLS - N_LABEL_COLS) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
    ]
    return "\n".join(lines)


def build_layer_mode_table(df: pd.DataFrame, over_all_flights: bool = True, show_std: bool = False) -> str:
    prefix = ALL_PREFIX if over_all_flights else SUCCESS_PREFIX
    n_col = "all_n" if over_all_flights else "success_only_n"

    header = (
        ["Model", "D", "H", "T"]
        + [h for _, h, _ in RATE_COLUMNS]
        + [h for _, h, _, _ in METRICS]
        + [SOLVING_TIME_HEADER]
    )

    lines: list[str] = [
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l ccc " + "c" * (N_COLS - N_LABEL_COLS) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *_build_table_rows(df, prefix, n_col, show_std),
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    csv_path = Path("../data/summary_experiments.csv")
    df = pd.read_csv(csv_path)

    latex = build_layer_mode_table(df, over_all_flights=True, show_std=False)
    print(latex)
    Path("layer_mode_table.tex").write_text(latex)
    print("\n[saved] layer_mode_table.tex")

    latex_std = build_layer_mode_table(df, over_all_flights=True, show_std=True)
    Path("layer_mode_table_std.tex").write_text(latex_std)
    print("[saved] layer_mode_table_std.tex")

    # multi-aircraft (GNN) table
    latex_multi = build_multi_table(df, over_all_flights=True, show_std=False)
    print("\n" + latex_multi)
    Path("multi_aircraft_table.tex").write_text(latex_multi)
    print("\n[saved] multi_aircraft_table.tex")

    latex_multi_std = build_multi_table(df, over_all_flights=True, show_std=True)
    Path("multi_aircraft_table_std.tex").write_text(latex_multi_std)
    print("[saved] multi_aircraft_table_std.tex")