from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("resnet101", "ft_transformer")
LAYER_MODES = ("baseline", "combined")

BACKBONE_LABELS = {
    "resnet101": "ResNet-101",
    "ft_transformer": "FT-Transformer",
}

BACKBONE_CITATIONS = {
    "resnet101": "he2016deep",               # He et al., Deep Residual Learning (ResNet)
    "ft_transformer": "gorishniy2021revisiting",  # Gorishniy et al., Revisiting Deep Learning Models for Tabular Data
}
OPTIMIZER_CITATION = "wachter2006ipopt"
OPTIMIZER_LABEL = "Ipopt-Optimizer"

OURS_LABEL = "({\\bf ours})"


def _model_label(model: str, lm: str) -> str:
    if lm == "combined":
        return "ApproachNet"
    return BACKBONE_LABELS[model]


DYNAMICS_MODES = ("without_dynamics", "with_dynamics")
LAYER_MODE_VALUE = {
    ("resnet101", "with_dynamics", "baseline"): "resnet101_action_baseline",
    ("resnet101", "with_dynamics", "combined"): "resnet101_action_constraint",
    ("resnet101", "without_dynamics", "baseline"): "resnet101_next_state_baseline",
    ("resnet101", "without_dynamics", "combined"): "resnet101_next_state_ils_funnel",
    ("ft_transformer", "with_dynamics", "baseline"): "ft_transformer_action_baseline",
    ("ft_transformer", "with_dynamics", "combined"): "ft_transformer_action_constraint",
    ("ft_transformer", "without_dynamics", "baseline"): "ft_transformer_next_state_baseline",
    ("ft_transformer", "without_dynamics", "combined"): "ft_transformer_next_state_ils_funnel",
}

RATE_COLUMNS = [
    ("success_rate", True),
    ("succ_rate_s2", True),
    ("window_auc", True),
]

METRICS = [
    ("lateral_dev_m", "\\shortstack{Lateral\\\\Dev. (m)}", 2, False),
    ("vertical_dev_m", "\\shortstack{Vertical\\\\Dev. (m)}", 2, False),
    ("angle_dev_deg", "\\shortstack{Angle\\\\Dev. ($^\\circ$)}", 2, False),
    ("gamma_dev_deg", "\\shortstack{Gamma\\\\Dev. ($^\\circ$)}", 2, False),
    ("velocity_dev_ms", "\\shortstack{Velocity\\\\Dev. (m/s)}", 2, False),
    ("landing_time_s", "\\shortstack{Landing\\\\Time (s)}", 2, False),
]
ALL_PREFIX = "all_"
SUCCESS_PREFIX = "success_only_"

OPTIMIZER_LAYER_MODE = "rnav_optimizer"

QUALITATIVE_IMAGE_PATH = "comparison_figure.png"

BENCHMARK_TIMING_CSV = Path("../data/CYVR/benchmark_method_comparison.csv")
SOLVING_TIME_HEADER = "\\shortstack{Solving\\\\Time (s)}"


def pooled_mean_std(n: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, float]:
    valid = n > 0
    n, mean, std = n[valid], mean[valid], std[valid]
    if n.sum() == 0:
        return float("nan"), float("nan")

    pooled_mean = np.sum(n * mean) / n.sum()
    pooled_var = np.sum(n * (std ** 2 + (mean - pooled_mean) ** 2)) / n.sum()
    return pooled_mean, float(np.sqrt(pooled_var))


def pool_rows(sub: pd.DataFrame, prefix: str, n_col: str) -> dict:
    n_total = sub["n_total"].sum()
    n_group = sub[n_col].sum()
    success_rate = 100.0 * sub["success_only_n"].sum() / n_total if n_total > 0 else float("nan")

    row = {"n_total": int(n_total), "n_group": int(n_group), "success_rate": success_rate}

    weights = sub["n_total"].to_numpy(dtype=float)
    if n_total > 0 and len(sub) > 0:
        auc_vals = sub["window_auc"].to_numpy(dtype=float)
        row["window_auc"] = 100.0 * float(np.average(auc_vals, weights=weights))
        s2_vals = sub["succ_rate_s2"].to_numpy(dtype=float)
        row["succ_rate_s2"] = 100.0 * float(np.average(s2_vals, weights=weights))
    else:
        row["window_auc"] = float("nan")
        row["succ_rate_s2"] = float("nan")

    for key, _, _, _ in METRICS:
        n = sub[n_col].to_numpy(dtype=float)
        mean = sub[f"{prefix}{key}_mean"].to_numpy(dtype=float)
        std = sub[f"{prefix}{key}_std"].to_numpy(dtype=float)
        row[f"{key}_mean"], row[f"{key}_std"] = pooled_mean_std(n, mean, std)

    return row


def pool_layer_mode(df: pd.DataFrame, layer_mode_value: str, prefix: str, n_col: str) -> dict:
    sub = df[df["layer_mode"] == layer_mode_value]
    return pool_rows(sub, prefix, n_col)


def load_solving_times(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        print(f"[WARN] Timing-CSV nicht gefunden: {csv_path} -> Solving-Time-Spalte bleibt leer ('--')")
        return {}

    df_timing = pd.read_csv(csv_path)
    solving_times: dict[str, float] = {}
    for _, row in df_timing.iterrows():
        solving = row.get("avg_solving_time_s")
        if pd.isna(solving):
            continue
        solving_times[str(row["model"])] = float(solving)
    return solving_times


def fmt(mean: float, std: float, digits: int, show_std: bool = False, bold: bool = False) -> str:
    if np.isnan(mean):
        return "--"
    text = f"{mean:.{digits}f}" if not show_std else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    return f"\\cellcolor{{gray!12}}\\textbf{{{text}}}" if bold else text


def fmt_rate(rate: float, bold: bool = False) -> str:
    if np.isnan(rate):
        return "--"
    text = f"{rate:.2f}"
    return f"\\cellcolor{{gray!12}}\\textbf{{{text}}}" if bold else text


def fmt_solving_time(solving_time: float, bold: bool = False) -> str:
    if solving_time is None or np.isnan(solving_time):
        return "--"
    text = f"{solving_time:.2f}"
    return f"\\cellcolor{{gray!12}}\\textbf{{{text}}}" if bold else text


def _best_indices(values: list[float], higher_is_better: bool) -> set[int]:
    finite = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if not finite:
        return set()
    best_val = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
    return {i for i, v in finite if v == best_val}


def _build_table_rows(
    df: pd.DataFrame, prefix: str, n_col: str, show_std: bool, solving_times: dict[str, float],
) -> list[str]:
    rows: list[tuple[str, str, str, str, dict]] = []
    for model in MODELS:
        for lm in LAYER_MODES:
            for dyn in DYNAMICS_MODES:
                layer_mode_value = LAYER_MODE_VALUE[(model, dyn, lm)]
                pooled = pool_layer_mode(df, layer_mode_value, prefix, n_col)
                rows.append((model, dyn, lm, layer_mode_value, pooled))

    optimizer_row = pool_rows(df[df["layer_mode"] == OPTIMIZER_LAYER_MODE], prefix, n_col)
    optimizer_solving_time = solving_times.get(OPTIMIZER_LAYER_MODE, float("nan"))

    best_rate_idx: dict[str, set[int]] = {
        col: _best_indices([pooled[col] for _, _, _, _, pooled in rows], higher)
        for col, higher in RATE_COLUMNS
    }
    best_metric_idx: dict[str, set[int]] = {
        key: _best_indices([pooled[f"{key}_mean"] for _, _, _, _, pooled in rows], higher_is_better)
        for key, _, _, higher_is_better in METRICS
    }
    best_solving_idx: set[int] = _best_indices(
        [solving_times.get(lmv, float("nan")) for _, _, _, lmv, _ in rows], False,
    )

    n_metrics = len(METRICS)
    CHECK = "\\checkmark"
    CROSS = "--"

    def metric_vals(pooled: dict, row_idx: int) -> list[str]:
        return [
            fmt(
                pooled[f"{key}_mean"], pooled[f"{key}_std"], digits, show_std,
                bold=(row_idx in best_metric_idx[key]),
            )
            for key, _, digits, _ in METRICS
        ]

    def emit_case(label_cells: list[str], pooled: dict, row_idx: int | None, solving_time: float) -> str:
        if row_idx is None:
            rate_cells = [
                fmt_rate(pooled["success_rate"]),
                fmt_rate(pooled["succ_rate_s2"]),
                fmt_rate(pooled["window_auc"]),
            ]
            metrics_cells = [
                fmt(pooled[f"{key}_mean"], pooled[f"{key}_std"], digits, show_std)
                for key, _, digits, _ in METRICS
            ]
            solving_cell = fmt_solving_time(solving_time)
        else:
            rate_cells = [
                fmt_rate(pooled[col], bold=(row_idx in best_rate_idx[col]))
                for col, _ in RATE_COLUMNS
            ]
            metrics_cells = metric_vals(pooled, row_idx)
            solving_cell = fmt_solving_time(solving_time, bold=(row_idx in best_solving_idx))

        row = label_cells + rate_cells + metrics_cells + [solving_cell]
        return " & ".join(row) + " \\\\"

    lines: list[str] = []
    prev_model = None
    prev_lm = None
    for row_idx, (model, dyn, lm, layer_mode_value, pooled) in enumerate(rows):
        new_model_block = model != prev_model
        new_lm_block = lm != prev_lm or new_model_block

        if prev_model is not None:
            if new_model_block:
                lines.append("\\midrule")
            elif new_lm_block:
                lines.append("\\cmidrule(lr){2-" + str(7 + n_metrics) + "}")

        if new_lm_block:
            backbone = BACKBONE_LABELS[model]
            if lm == "combined":
                label = f"ApproachNet ({backbone})"
            else:
                citation = BACKBONE_CITATIONS.get(model)
                label = f"{backbone} \\cite{{{citation}}}" if citation else backbone
        elif lm == "combined":
            label = OURS_LABEL
        else:
            label = ""

        dyn_cell = CHECK if dyn == "with_dynamics" else CROSS
        hc_cell = CHECK if lm == "combined" else CROSS
        solving_time = solving_times.get(layer_mode_value, float("nan"))
        lines.append(emit_case([label, dyn_cell, hc_cell], pooled, row_idx, solving_time))

        prev_model, prev_lm = model, lm

    lines.append("\\midrule")
    optimizer_label = f"{OPTIMIZER_LABEL} \\cite{{{OPTIMIZER_CITATION}}}"
    lines.append(emit_case([optimizer_label, CHECK, CHECK], optimizer_row, None, optimizer_solving_time))

    return lines


def build_layer_mode_figure(
    df: pd.DataFrame,
    over_all_flights: bool = True,
    show_std: bool = False,
    timing_csv_path: Path = BENCHMARK_TIMING_CSV,
) -> str:
    prefix = ALL_PREFIX if over_all_flights else SUCCESS_PREFIX
    n_col = "all_n" if over_all_flights else "success_only_n"

    solving_times = load_solving_times(timing_csv_path)

    n_flights_total = pool_layer_mode(
        df, LAYER_MODE_VALUE[("resnet101", "with_dynamics", "baseline")], prefix, n_col
    )["n_total"]
    n_airports = int(df["icao"].nunique()) if "icao" in df else 3

    n_metrics = len(METRICS)
    header = (
        ["Model", "D", "H", "\\shortstack{\\texttt{succ@1}\\\\(\\%)}",
         "\\shortstack{\\texttt{succ@2}\\\\(\\%)}", "\\shortstack{AUC\\\\(\\%)}"]
        + [label for _, label, _, _ in METRICS]
        + [SOLVING_TIME_HEADER]
    )

    table_rows = _build_table_rows(df, prefix, n_col, show_std, solving_times)

    lines: list[str] = []
    lines.append("\\begin{figure*}[t]")
    lines.append("\\centering")
    lines.append("")
    lines.append("\\begin{minipage}[t]{0.69\\textwidth}")
    lines.append("\\vspace{0pt}")
    lines.append("\\raggedright")
    lines.append("\\refstepcounter{table}")
    lines.append("\\label{tab:qualitative}")
    lines.append("{\\footnotesize")
    lines.append("\\textbf{Table \\thetable: Quantitative Evaluation.}")
    lines.append(
        f"Approach performance by backbone architecture and presence of D/H layers, averaged across "
        f"${n_airports}$ airports and ${n_flights_total}$ test approaches.\\\\ "
        "\\texttt{succ@s} = success ratio at $s$ times reference tolerances; "
        "\\mbox{AUC = $\\int\\nolimits_0^1 \\texttt{auc@s} \\,ds$} (see App.~\\ref{app:tolerances})"
    )
    lines.append("\\par}")
    lines.append("")
    lines.append("\\vspace{0.1cm}")
    lines.append("")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l cc " + "c" * (3 + n_metrics + 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    lines.extend(table_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("")
    lines.append("\\medskip")
    lines.append("\\raggedright")
    lines.append("\\vspace{-0.15cm}")
    lines.append("D = flight mechanics ; H = hard constraints layer")
    lines.append("\\end{minipage}")
    lines.append("\\hfill")
    lines.append("\\begin{minipage}[t]{0.29\\textwidth}")
    lines.append("\\vspace{0pt}")
    lines.append("\\raggedright")
    lines.append("\\refstepcounter{figure}")
    lines.append("\\label{fig:qualitative}")
    lines.append("{\\footnotesize")
    lines.append("\\textbf{Figure \\thefigure: \\mbox{Qualitative Evaluation.}}")
    lines.append("}")
    lines.append("\\par}")
    lines.append("")
    lines.append("\\vspace{0.1cm}")
    lines.append(f"\\centering\\includegraphics[width=0.85\\linewidth]{{{QUALITATIVE_IMAGE_PATH}}}")
    lines.append("")
    lines.append("\\end{minipage}")
    lines.append("\\vspace{-0.5cm}")
    lines.append("\\end{figure*}")

    return "\n".join(lines)


if __name__ == "__main__":
    csv_path = Path("../data/qp_layer_mode_summary_8experiments.csv")
    df = pd.read_csv(csv_path)

    latex = build_layer_mode_figure(df, over_all_flights=True, show_std=False)
    print(latex)
    Path("layer_mode_figure.tex").write_text(latex)
    print("\n[saved] layer_mode_figure.tex")

    latex_std = build_layer_mode_figure(df, over_all_flights=True, show_std=True)
    Path("layer_mode_figure_std.tex").write_text(latex_std)
    print("[saved] layer_mode_figure_std.tex")