from pathlib import Path

import numpy as np

from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from utils.airport import Airport


def compute_stats(dataset: LandingActionDataset, flight_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = [dataset._cache[f][0] for f in flight_indices if f in dataset._cache]
    ys = [dataset._cache[f][1] for f in flight_indices if f in dataset._cache]

    all_x = np.concatenate(xs, axis=0)
    all_y = np.concatenate(ys, axis=0)

    x_mean, x_std = all_x.mean(0), all_x.std(0) + 1e-8
    y_mean, y_std = all_y.mean(0), all_y.std(0) + 1e-8
    return x_mean, x_std, y_mean, y_std


if __name__ == "__main__":

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    results_path = Path("results/airport_normalizations")
    results_path.mkdir(parents=True, exist_ok=True)

    for airport_icao in airport_list_icao:
        print(f"\n--- Verarbeite {airport_icao} ---")
        airport = Airport.load_from_folder(base_path=Path("../data/"), icao=airport_icao)

        # 20260708_1247 -> old with 2000
        # 20260802_0920 -> new with 10000
        flight_dataset = LazyFlightParquetDataset(f"../data/{airport_icao}/train/20260802_0920")
        dataset = LandingActionDataset(
            flight_dataset=flight_dataset,
            mode="bspline_state",
            max_dt=5.0,
            airport_center_latlon=(airport.lat, airport.lon),
            airport_elevation_m=airport.arinc_424_runway[0].runway_altitude * 0.3048,
        )

        print(f"Samples: {len(dataset)}, input_dim: {dataset.input_dim}")

        try:
            dataset.flight_level_split(val_frac=0.2, seed=42)
        except Exception as e:
            print(f"[WARN] Split für {airport_icao} fehlgeschlagen: {e}")
            continue

        if dataset.x_mean is not None:
            np.savetxt(
                results_path / f"{airport_icao}_normalization_x.csv",
                np.vstack([dataset.x_mean, dataset.x_std]),
                delimiter=",",
                comments="",
            )

            np.savetxt(
                results_path / f"{airport_icao}_normalization_y.csv",
                np.vstack([dataset.y_mean, dataset.y_std]),
                delimiter=",",
                comments="",
            )

            print(f"Stats gespeichert! Check X-Position (Easting) Mean: {dataset.x_mean[5]:.2f}")

        train_flights, val_flights = dataset.split_flights(val_frac=0.2, seed=42)

        try:
            val_x_mean, val_x_std, val_y_mean, val_y_std = compute_stats(dataset, val_flights)
        except ValueError as e:
            print(f"[{airport_icao}] [WARN] Val-Stats konnten nicht berechnet werden: {e}")
            continue

        np.savetxt(
            results_path / f"{airport_icao}_val_normalization_x.csv",
            np.vstack([val_x_mean, val_x_std]),
            delimiter=",",
            comments="",
        )
        np.savetxt(
            results_path / f"{airport_icao}_val_normalization_y.csv",
            np.vstack([val_y_mean, val_y_std]),
            delimiter=",",
            comments="",
        )

        if not (np.isfinite(val_x_mean).all() and np.isfinite(val_x_std).all()
                and np.isfinite(val_y_mean).all() and np.isfinite(val_y_std).all()):
            print(f"[{airport_icao}] [WARN] Val-Normalisierung enthält NaN/Inf!")
        else:
            print(f"[{airport_icao}] Val-Stats gespeichert, keine NaN/Inf")