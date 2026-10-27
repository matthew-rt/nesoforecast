"""REMIT unavailability messages for wind (onshore + offshore), every revision, from the Elexon Insights API.

Weekly publish-time lists filtered by fuelType, then message details in bulk (/remit?messageId=..). Every revision
is kept with its publishTime, so the outage state known at any forecast issue time can be reconstructed.
Cached per week / per batch; reruns only fetch what is missing.
Usage: python fetch_remit.py [start] [end]   (default 2020-07-01 -> today)"""
import pathlib, sys, time
import pandas as pd, requests

OUT = pathlib.Path("/Users/matt/SCORESdata/elexon_remit_wind")  # shared store, see DATA_INDEX.md
API = "https://data.elexon.co.uk/bmrs/api/v1"
FUELS = ["Wind Offshore", "Wind Onshore"]
BATCH = 50

def get(path, params):
    for i in range(8):
        r = requests.get(API + path, params=params, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            print("throttled", r.status_code, flush=True); time.sleep(30 * (i + 1)); continue
        r.raise_for_status()
        return r.json()["data"]
    raise RuntimeError("gave up on " + path)

def lists(start, end):
    (OUT / "lists").mkdir(parents=True, exist_ok=True)
    if not (OUT / "lists" / ".all_revisions").exists():  # first lists were fetched latest-revision-only
        for f in (OUT / "lists").glob("*.parquet"):
            f.unlink()
        (OUT / "lists" / ".all_revisions").touch()
    ids = set()
    for d in pd.date_range(start, end, freq="7D"):
        f = OUT / "lists" / f"{d:%Y-%m-%d}.parquet"
        if not f.exists() or d + pd.Timedelta("7D") > pd.Timestamp.now():  # refetch the current week
            rows = []
            for fuel in FUELS:
                rows += get("/remit/list/by-publish", {"from": f"{d:%Y-%m-%d}T00:00Z",
                                                       "to": f"{d + pd.Timedelta('7D'):%Y-%m-%d}T00:00Z", "fuelType": fuel,
                                                       "latestRevisionOnly": "false"})  # default drops revisions
            pd.DataFrame(rows, columns=["id", "mrid", "revisionNumber", "createdTime", "publishTime"]).to_parquet(f, index=False)
            time.sleep(0.2)
        ids.update(pd.read_parquet(f)["id"].astype(int))
    return sorted(ids)

def details(ids):
    (OUT / "messages").mkdir(parents=True, exist_ok=True)
    have = set()
    for f in (OUT / "messages").glob("*.parquet"):
        have.update(pd.read_parquet(f, columns=["id"])["id"].astype(int))
    todo = [i for i in ids if i not in have]
    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        rows = get("/remit", [("messageId", i) for i in chunk])
        df = pd.json_normalize(rows)
        nested = [c for c in ("outageProfile", "relatedInformation") if c in df]  # absent in some batches
        df.astype({c: str for c in nested}).to_parquet(OUT / "messages" / f"batch_{chunk[0]}_{chunk[-1]}.parquet", index=False)
        time.sleep(0.2)
    print(f"{len(ids)} ids, fetched {len(todo)} new", flush=True)

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-07-01"  # catch outages announced before 2021
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.now().strftime("%Y-%m-%d")
    details(lists(start, end))
