from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def atomic_write_parquet(df: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)


def append_trajectory(df_new: DataFrame, path: Path) -> None:
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    atomic_write_parquet(df_all, path)


def already_processed(status_path: Path, icao24: str, flight_id: int) -> bool:
    df = load_status_log(status_path)
    if df.empty:
        return False

    mask = (df["icao24"] == icao24) & (df["flight_id"] == flight_id)
    return bool(mask.any())


def append_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(status, default=str) + "\n")


def load_status_log(path: Path) -> DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)
    if "success" in df.columns:
        df["success"] = df["success"].fillna(False).astype(bool)

    return df