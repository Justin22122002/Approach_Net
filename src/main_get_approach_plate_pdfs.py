from pathlib import Path
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://data.skylinkapi.com/v3.1"

API_KEY = os.environ["APPROCH_PLATE_API_KEY"]

HEADERS = {
    "x-api-key": API_KEY,
}

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_approach_charts(icao: str) -> list[dict]:
    icao = icao.upper().strip()

    response = requests.get(
        f"{BASE_URL}/charts/{icao}/APP",
        headers=HEADERS,
        timeout=(10, 60),
    )

    if response.status_code == 404:
        raise RuntimeError(f"Airport {icao} was not found.")

    response.raise_for_status()

    data = response.json()

    return data.get("charts", {}).get("APP", [])


def download_chart(
    chart: dict,
    output_dir: Path,
) -> Path:
    name = chart["name"]
    url = chart["url"]

    filename = sanitize_filename(name) + ".pdf"
    output_path = output_dir / filename

    print(f"Downloading: {name}")
    print(f"URL:        {url}")

    response = requests.get(
        url,
        timeout=(10, 120),
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved:      {output_path}")
    print()

    return output_path


def download_approach_charts(
    icao: str,
    output_root: Path = Path("data"),
) -> None:

    icao = icao.upper().strip()

    output_dir = output_root / icao / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = get_approach_charts(icao)

    print(f"Found {len(charts)} approach charts for {icao}")
    print()

    for chart in charts:
        try:
            download_chart(
                chart=chart,
                output_dir=output_dir,
            )
        except requests.RequestException as exc:
            print(f"ERROR downloading {chart.get('name', '<unknown>')}:")
            print(exc)
            print()


if __name__ == "__main__":
    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        "CYYC",
        "CYEG",
    ]
    for icao in airport_list_icao:
        download_approach_charts(icao)
