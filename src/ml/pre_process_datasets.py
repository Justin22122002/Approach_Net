import pickle
from pathlib import Path

from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from utils.airport import Airport


def save_dataset(dataset: LandingActionDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] saved {path} ({len(dataset)} samples)")


def load_dataset(path: str | Path) -> LandingActionDataset:
    path = Path(path)
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    print(f"[cache] loaded {path} ({len(dataset)} samples)")
    return dataset


def load_or_build_dataset(
    airport_icao: str,
    folder_path: Path,
    dataset_name: str,
    cap_data: bool = True,
) -> LandingActionDataset:
    cache_name = f"landing_action_dataset_cap{int(cap_data)}.pkl"
    cache_path = folder_path / airport_icao / "train" / dataset_name / "dataset_cache" / cache_name
    if cache_path.exists():
        return load_dataset(cache_path)

    airport = Airport.load_from_folder(base_path=folder_path, icao=airport_icao)
    flight_dataset = LazyFlightParquetDataset(f"{folder_path}/{airport_icao}/train/{dataset_name}")

    dataset = LandingActionDataset(
        flight_dataset=flight_dataset,
        mode="bspline_state",
        max_dt=5.0,
        airport_center_latlon=(airport.lat, airport.lon),
        airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
        augment_rotations_deg=[180.0],
        provide_raw_features=True,
        cap_data=cap_data,
    )
    save_dataset(dataset, cache_path)
    return dataset


if __name__ == "__main__":
    dataset_name: str = "20260802_0920"
    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]
    folder_path: Path = Path("../data/")  # or ./data/ for wsl or direct scripts execution from the src folder of this project
    CAP_DATA: bool = True

    all_datasets: list[LandingActionDataset] = []
    for airport_icao in airport_list_icao:
        dataset = load_or_build_dataset(airport_icao, folder_path, dataset_name, cap_data=CAP_DATA)
        print(f"[{airport_icao}] samples: {len(dataset)}, input_dim: {dataset.input_dim}")
        all_datasets.append(dataset)