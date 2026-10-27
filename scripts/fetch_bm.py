"""Per-BMU Elexon data for wind units, one parquet per dataset-month (resumable):
B1610 metered output, PN physical notifications, MELS max export limits, BOALF bid-offer acceptances.
Usage: python fetch_bm.py 2021-01 2026-10 [--embedded] [DATASET ...]
--embedded: the E_ (embedded BM) units, into data/raw/bm_embedded (their B1610 to 2026-10-11 is read from
/Users/matt/code/genloadfactor/data/raw/b1610 by prepare_bm.py, so only the final month is fetched here)."""
import pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMBEDDED = "--embedded" in sys.argv
RAW = ROOT / "data" / "raw" / ("bm_embedded" if EMBEDDED else "bm")
BMRS = "https://data.elexon.co.uk/bmrs/api/v1"
_u = pd.read_csv(ROOT / "data" / "sites.csv").elexonBmUnit
UNITS = _u[_u.str.startswith("E_" if EMBEDDED else "T_")].tolist()
BATCH = 20

def get(ds, params):
    for i in range(6):
        try:
            r = requests.get(f"{BMRS}/datasets/{ds}/stream", params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("retry", ds, params.get("from"), e, flush=True); time.sleep(3 * (i + 1))
    raise RuntimeError(ds)

def month(ds, ym):
    f = RAW / ds / f"{ym}.parquet"
    if f.exists():
        return
    p = pd.Period(ym)
    a, b = p.start_time, min(p.end_time.ceil("D"), pd.Timestamp.now().floor("D"))
    rows = []
    for i in range(0, len(UNITS), BATCH):
        rows += get(ds, {"from": a.strftime("%Y-%m-%dT%H:%MZ"), "to": b.strftime("%Y-%m-%dT%H:%MZ"),
                         "bmUnit": UNITS[i:i + BATCH]})
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f)
    print(ds, ym, len(rows), flush=True)

if __name__ == "__main__":
    months = [str(p) for p in pd.period_range(sys.argv[1], sys.argv[2], freq="M")]
    sets = [a for a in sys.argv[3:] if not a.startswith("--")] or ["B1610", "PN", "MELS", "BOALF"]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda job: month(*job), [(d, m) for d in sets for m in months]))
