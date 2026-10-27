"""Download raw data: NESO day-ahead demand forecast performance + historic demand,
and Elexon BMRS WINDFOR (wind forecast) + FUELHH wind outturn."""
import json, pathlib, time
import pandas as pd, requests

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BMRS = "https://data.elexon.co.uk/bmrs/api/v1"
NESO_DUMP = "https://api.neso.energy/datastore/dump/"

NESO = {
    "neso_da_perf.csv": "08e41551-80f8-4e28-a416-ea473a695db9",
    "hist_demand_2021.csv": "18c69c42-f20d-46f0-84e9-e279045befc6",
    "hist_demand_2022.csv": "bb44a1b5-75b1-4db2-8491-257f23385006",
    "hist_demand_2023.csv": "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    "hist_demand_2024.csv": "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    "hist_demand_2025.csv": "b2bde559-3455-4021-b179-dfe60c0337b0",
    "hist_demand_2026.csv": "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}

def get(url, **params):
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except Exception as e:
            print("retry", url, e); time.sleep(2 * (i + 1))
    raise RuntimeError(url)

def fetch_neso():
    for name, rid in NESO.items():
        (RAW / name).write_bytes(get(NESO_DUMP + rid).content)
        print(name)

def fetch_bmrs(dataset, start, end, key_from, key_to, fmt, extra=None, step="7D"):
    out = RAW / f"{dataset.lower()}.parquet"
    rows = []
    edges = pd.date_range(start, end, freq=step).append(pd.DatetimeIndex([pd.Timestamp(end)]))
    for a, b in zip(edges[:-1], edges[1:]):
        p = {key_from: a.strftime(fmt), key_to: b.strftime(fmt), **(extra or {})}
        rows += get(f"{BMRS}/datasets/{dataset}/stream", **p).json()
        print(dataset, a.date(), len(rows))
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_parquet(out)
    return df

if __name__ == "__main__":
    fetch_neso()
    # WINDFOR: chunk by publish time (forecasts published 8x/day, hourly to ~D+2)
    fetch_bmrs("WINDFOR", "2021-03-25", "2026-10-23", "publishDateTimeFrom", "publishDateTimeTo",
               "%Y-%m-%dT%H:%MZ")
    # FUELHH wind outturn, by settlement date (inclusive range -> step minus a day handled by dedupe)
    fetch_bmrs("FUELHH", "2021-04-01", "2026-10-22", "settlementDateFrom", "settlementDateTo",
               "%Y-%m-%d", {"fuelType": "WIND"}, step="14D")

# NESO wind BOA volumes (curtailment/bids on wind BMUs), used by NESO's own wind-forecast incentive
BOA = {"2021_22": "094ad551-630a-4302-b259-800a27122cfb", "2022_23": "62da8915-4b87-4d77-a8f5-89084de1830d",
       "2023_24": "1edaa5fe-86fe-41b4-95b9-55e9ec45fab0", "2024_25": "d3fbf6c1-7688-4486-8716-b5af0c895a5a",
       "2025_26": "7bf83942-3590-4fe5-9980-49a9546e30a9", "2026_27": "45598dcd-ea9c-4911-95f2-7946c5f3b034"}

def fetch_boa():
    for k, rid in BOA.items():
        (RAW / f"wind_boa_{k}.csv").write_bytes(get(NESO_DUMP + rid).content)
        print("boa", k)
