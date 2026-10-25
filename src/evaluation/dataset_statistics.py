from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset


@dataclass
class DatasetMetrics:
    base_path: Path
    n_flights: int = 0
    n_flights_total_raw: int = 0
    n_flights_dropped_no_aircraft: int = 0
    n_trackpoints: int = 0
    trackpoints_per_flight: dict = field(default_factory=dict)  # summary stats
    time_start: pd.Timestamp | None = None
    time_end: pd.Timestamp | None = None
    time_span: pd.Timedelta | None = None
    flight_duration_stats: dict = field(default_factory=dict)  # in seconds
    n_aircraft_known: int = 0
    n_aircraft_total: int = 0
    n_runway_assignments: int = 0
    runway_counts: dict = field(default_factory=dict)
    _per_flight_counts: pd.Series = field(default_factory=pd.Series, repr=False, compare=False)
    _durations_s: pd.Series = field(default_factory=pd.Series, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "base_path": str(self.base_path),
            "n_flights": self.n_flights,
            "n_flights_total_raw": self.n_flights_total_raw,
            "n_flights_dropped_no_aircraft": self.n_flights_dropped_no_aircraft,
            "n_trackpoints": self.n_trackpoints,
            **{f"trackpoints_{k}": v for k, v in self.trackpoints_per_flight.items()},
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_span": self.time_span,
            **{f"duration_s_{k}": v for k, v in self.flight_duration_stats.items()},
            "n_aircraft_known": self.n_aircraft_known,
            "n_aircraft_total": self.n_aircraft_total,
            "n_runway_assignments": self.n_runway_assignments,
            "runway_counts": self.runway_counts,
        }


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


def compute_dataset_metrics(base_path: str | Path) -> DatasetMetrics:
    base_path = Path(base_path)

    lazy = LazyFlightParquetDataset(base_path)

    metrics = DatasetMetrics(base_path=base_path)

    valid_icao24: set[str] = set()
    if lazy.aircraft_path.exists():
        df_ac = pd.read_parquet(lazy.aircraft_path)
        metrics.n_aircraft_total = len(df_ac)
        if "mass" in df_ac.columns:
            df_ac_valid = df_ac[df_ac["mass"].notna()]
        else:
            df_ac_valid = df_ac
        metrics.n_aircraft_known = len(df_ac_valid)
        valid_icao24 = set(df_ac_valid["icao24"].astype(str))

    cols = ["icao24", "flight_id", "time"]
    df_arrivals_probe = pd.read_parquet(lazy.flight_path, columns=None)
    available_cols = [c for c in cols if c in df_arrivals_probe.columns]
    df_arr_all = df_arrivals_probe[available_cols] if available_cols else df_arrivals_probe

    metrics.n_flights_total_raw = df_arr_all.drop_duplicates(["icao24", "flight_id"]).shape[0]

    # nur Trajektorien behalten, deren icao24 tatsächlich ein Aircraft-Modell hat
    df_arr = df_arr_all[df_arr_all["icao24"].astype(str).isin(valid_icao24)]

    metrics.n_flights = df_arr.drop_duplicates(["icao24", "flight_id"]).shape[0]
    metrics.n_flights_dropped_no_aircraft = metrics.n_flights_total_raw - metrics.n_flights

    metrics.n_trackpoints = len(df_arr)

    per_flight_counts = df_arr.groupby(["icao24", "flight_id"]).size()
    metrics.trackpoints_per_flight = _stats(per_flight_counts)
    metrics._per_flight_counts = per_flight_counts

    if "time" in df_arr.columns:
        time_col = pd.to_datetime(df_arr["time"])
        metrics.time_start = time_col.min()
        metrics.time_end = time_col.max()
        metrics.time_span = metrics.time_end - metrics.time_start

        df_time = df_arr.assign(_time=time_col)
        dur = (
            df_time.groupby(["icao24", "flight_id"])["_time"]
            .agg(lambda t: (t.max() - t.min()).total_seconds())
        )
        metrics.flight_duration_stats = _stats(dur)
        metrics._durations_s = dur

    if lazy.runway_path.exists():
        df_rwy = pd.read_parquet(lazy.runway_path)
        df_rwy = df_rwy[df_rwy["icao24"].astype(str).isin(valid_icao24)]
        metrics.n_runway_assignments = len(df_rwy)
        if "runway_bearing" in df_rwy.columns:
            metrics.runway_counts = (
                df_rwy["runway_bearing"].round(0).astype("Int64").value_counts().to_dict()
            )

    return metrics


def print_metrics(metrics: DatasetMetrics) -> None:
    print(f"\n=== {metrics.base_path} ===")
    print(f"Flights (mit gültigem Aircraft-Typ):   {metrics.n_flights}")
    print(f"Flights total (vor Aircraft-Filter):   {metrics.n_flights_total_raw}")
    print(f"  davon verworfen (kein Aircraft):     {metrics.n_flights_dropped_no_aircraft}")
    print(f"Trackpoints total (gefiltert):          {metrics.n_trackpoints}")
    tp = metrics.trackpoints_per_flight
    if tp.get("mean") is not None:
        print(f"  per flight: mean={tp['mean']:.1f}  median={tp['median']:.1f}  "
              f"min={tp['min']:.0f}  max={tp['max']:.0f}  std={tp['std']:.1f}")
    if metrics.time_start is not None:
        print(f"Time range:                            {metrics.time_start} -> {metrics.time_end}")
        print(f"Time span:                             {metrics.time_span}")
    d = metrics.flight_duration_stats
    if d.get("mean") is not None:
        print(f"Flight duration [s]: mean={d['mean']:.1f}  median={d['median']:.1f}  "
              f"min={d['min']:.1f}  max={d['max']:.1f}  std={d['std']:.1f}")
    print(f"Aircraft: known={metrics.n_aircraft_known} / total={metrics.n_aircraft_total}")
    print(f"Runway assignments:                    {metrics.n_runway_assignments}")
    if metrics.runway_counts:
        print(f"Runway bearing counts:                 {metrics.runway_counts}")


def aggregate_metrics(all_metrics: list[DatasetMetrics]) -> dict:
    if not all_metrics:
        return {}

    n_flights = sum(m.n_flights for m in all_metrics)
    n_flights_total_raw = sum(m.n_flights_total_raw for m in all_metrics)
    n_flights_dropped_no_aircraft = sum(m.n_flights_dropped_no_aircraft for m in all_metrics)
    n_trackpoints = sum(m.n_trackpoints for m in all_metrics)
    n_aircraft_known = sum(m.n_aircraft_known for m in all_metrics)
    n_aircraft_total = sum(m.n_aircraft_total for m in all_metrics)
    n_runway_assignments = sum(m.n_runway_assignments for m in all_metrics)

    all_counts = pd.concat([m._per_flight_counts for m in all_metrics if not m._per_flight_counts.empty])
    all_durations = pd.concat([m._durations_s for m in all_metrics if not m._durations_s.empty])

    starts = [m.time_start for m in all_metrics if m.time_start is not None]
    ends = [m.time_end for m in all_metrics if m.time_end is not None]

    runway_counts: dict = {}
    for m in all_metrics:
        for k, v in m.runway_counts.items():
            runway_counts[k] = runway_counts.get(k, 0) + v

    return {
        "icao": "ALL",
        "split": "combined",
        "n_flights": n_flights,
        "n_flights_total_raw": n_flights_total_raw,
        "n_flights_dropped_no_aircraft": n_flights_dropped_no_aircraft,
        "n_trackpoints": n_trackpoints,
        **{f"trackpoints_{k}": v for k, v in _stats(all_counts).items()},
        "time_start": min(starts) if starts else None,
        "time_end": max(ends) if ends else None,
        **{f"duration_s_{k}": v for k, v in _stats(all_durations).items()},
        "n_aircraft_known": n_aircraft_known,
        "n_aircraft_total": n_aircraft_total,
        "n_runway_assignments": n_runway_assignments,
        "runway_counts": runway_counts,
    }


def print_aggregate(agg: dict) -> None:
    print(f"\n=== TOTAL / gepoolt über {agg['n_flights']} Flüge (mit Aircraft-Typ), {len(agg.get('runway_counts', {}))} Runways ===")
    print(f"Flights total (mit Aircraft-Typ):       {agg['n_flights']}")
    print(f"Flights total (vor Aircraft-Filter):    {agg['n_flights_total_raw']}")
    print(f"  davon verworfen (kein Aircraft):      {agg['n_flights_dropped_no_aircraft']}")
    print(f"Trackpoints total:                      {agg['n_trackpoints']}")
    if agg.get("trackpoints_mean") is not None:
        print(f"  per flight: mean={agg['trackpoints_mean']:.1f}  median={agg['trackpoints_median']:.1f}  "
              f"min={agg['trackpoints_min']:.0f}  max={agg['trackpoints_max']:.0f}  std={agg['trackpoints_std']:.1f}")
    if agg.get("time_start") is not None:
        print(f"Time range (über alle Splits):          {agg['time_start']} -> {agg['time_end']}")
    if agg.get("duration_s_mean") is not None:
        print(f"Flight duration [s]: mean={agg['duration_s_mean']:.1f}  median={agg['duration_s_median']:.1f}  "
              f"min={agg['duration_s_min']:.1f}  max={agg['duration_s_max']:.1f}  std={agg['duration_s_std']:.1f}")
    print(f"Aircraft: known={agg['n_aircraft_known']} / total={agg['n_aircraft_total']}")
    print(f"Runway assignments:                     {agg['n_runway_assignments']}")


if __name__ == "__main__":
    # (icao, split, base_path)
    DATASET_PATHS: list[tuple[str, str, str]] = [
        ("CYVR", "train", "../data/CYVR/train/20260802_0920"),
        ("CYVR", "test", "../data/CYVR/test/20260709_1207"),
        ("CYUL", "train", "../data/CYUL/train/20260802_0920"),
        ("CYUL", "test", "../data/CYUL/test/20260709_1207"),
        ("CYYZ", "train", "../data/CYYZ/train/20260802_0920"),
        ("CYYZ", "test", "../data/CYYZ/test/20260709_1207"),
    ]

    rows = []
    all_metrics: list[DatasetMetrics] = []
    for icao, split, path in DATASET_PATHS:
        p = Path(path)
        if not p.parent.exists():
            print(f"[SKIP] {icao}/{split}: {p} not found")
            continue
        m = compute_dataset_metrics(p)
        print_metrics(m)
        all_metrics.append(m)
        row = {"icao": icao, "split": split, **m.to_dict()}
        rows.append(row)

    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        print_aggregate(agg)
        rows.append(agg)

    if rows:
        summary_df = pd.DataFrame(rows)
        out_csv = Path("dataset_metrics_summary.csv")
        summary_df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] {out_csv.resolve()}")