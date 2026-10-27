"""Per-site Global Wind Atlas downscaling factor for ERA5, plus hub-height / rotor estimates.

ERA5 at 0.25 deg smooths terrain and coastlines. For each site:
  k = mean over turbines of GWA speed at hub height / GWA mean speed at 100 m over the ERA5 grid box(es)
using the same bilinear weights we use to interpolate ERA5 to the site, so k * ERA5_ws100 ~ hub-height speed."""
import pathlib, re
import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, GWA = ROOT / "data", ROOT / "data" / "raw" / "gwa"
HEIGHTS = [50, 100, 150]
ERA5_LAT0, ERA5_LON0, ERA5_RES = 49.5, -8.5, 0.25  # grid origin of our ERA5 download

def rotor_from_model(m):
    """Rotor diameter from model strings like 'V164-10.0', 'SG 8.0-167 DD', 'SWT-3.6-107', 'V90-3.0'."""
    if not isinstance(m, str):
        return np.nan
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m)]
    d = [x for x in nums if 40 <= x <= 260]
    return d[0] if d else np.nan

LAT_TO_MSL = 2.5  # m; offshore planning docs quote hub height above Lowest Astronomical Tide

def hub_heights():
    """Researched hub heights (planning/developer sources, data/raw/sites/hub_heights.csv).
    Stated hub height where given, else consented/as-built tip height minus rotor radius."""
    h = pd.read_csv(DATA / "raw" / "sites" / "hub_heights.csv")
    hub = h.hub_direct_m.fillna(h.tip_m - h.rotor_m / 2)
    h["hub_m"] = hub - np.where(h.datum == "LAT", LAT_TO_MSL, 0)
    h["hub_low_conf"] = h.basis.str.contains("low confidence")
    return h.set_index("loc")[["hub_m", "rotor_m", "tip_m", "hub_low_conf", "basis", "source"]]

def site_table():
    s = pd.read_csv(DATA / "sites.csv").dropna(subset=["lat"])
    s["loc"] = s.osm_id.fillna("repd" + s.repd_id.astype(str))
    l = s.groupby("loc").agg(lat=("lat", "first"), lon=("lon", "first"), offshore=("offshore", "first"),
                             osm_id=("osm_id", "first"), hub_m=("hub_m", "first"), rotor_m=("rotor_m", "first"),
                             tip_m=("tip_m", "first"), repd_tip_m=("repd_tip_m", "first"),
                             rated_mw=("rated_mw", "first"), repd_turbine_mw=("repd_turbine_mw", "first"),
                             model=("model", "first"), units=("elexonBmUnit", list))
    h = hub_heights()
    l["rotor_est"] = h.rotor_m.reindex(l.index).fillna(l.rotor_m).fillna(l.model.map(rotor_from_model))
    l["hub_est"] = h.hub_m.reindex(l.index)
    l["hub_low_conf"] = h.hub_low_conf.reindex(l.index)
    l["hub_source"] = h.source.reindex(l.index)
    # sites without researched hub heights (the embedded E_ units, added 2026-10-24): first plausible (40-160 m) of
    # OSM hub tag, tip - rotor/2, 0.58 x tip, type median; all flagged low confidence
    tip = l.tip_m.fillna(l.repd_tip_m)
    steps = [("OSM height:hub", l.hub_m), ("tip - rotor/2 from OSM/REPD", tip - l.rotor_est / 2),
             ("0.58 x tip (median hub/tip of researched onshore sites)", 0.58 * tip),
             ("type median", l.offshore.map(l.hub_est.groupby(l.offshore).median()))]
    miss = l.hub_est.isna()
    for name, v in steps:
        ok = miss & v.between(40, 160)  # OSM height tags are sometimes wrong (Baillie: tip 65 m with a 90 m rotor)
        l.loc[ok, "hub_est"], l.loc[ok, "hub_source"], l.loc[ok, "hub_low_conf"] = v[ok], name + " (not researched)", True
        miss &= ~ok
    return l.drop(columns=["hub_m", "rotor_m", "tip_m", "repd_tip_m", "repd_turbine_mw"])

def bilinear(lat, lon):
    """ERA5 node indices (i=lat row from south, j=lon col) and weights for bilinear interpolation."""
    y, x = (lat - ERA5_LAT0) / ERA5_RES, (lon - ERA5_LON0) / ERA5_RES
    i0, j0 = int(np.floor(y)), int(np.floor(x)); fy, fx = y - i0, x - j0
    return [((i0, j0), (1 - fy) * (1 - fx)), ((i0 + 1, j0), fy * (1 - fx)),
            ((i0, j0 + 1), (1 - fy) * fx), ((i0 + 1, j0 + 1), fy * fx)]

def at_height(v, h):
    """Interpolate GWA layers (HEIGHTS) to height h with a piecewise power law."""
    v = np.asarray(v, float)  # shape (..., 3)
    k = 0 if h <= 100 else 1
    a = np.log(v[..., k + 1] / v[..., k]) / np.log(HEIGHTS[k + 1] / HEIGHTS[k])
    return v[..., k] * (h / HEIGHTS[k]) ** a

def main():
    l = site_table()
    t = pd.read_parquet(DATA / "turbines.parquet")
    rs = {h: rasterio.open(GWA / f"gwa_GBR_ws{h}.tif") for h in HEIGHTS}
    box_cache = {}

    def box_mean(i, j):
        if (i, j) not in box_cache:
            la, lo = ERA5_LAT0 + i * ERA5_RES, ERA5_LON0 + j * ERA5_RES
            h = ERA5_RES / 2
            vals = []
            for r in rs.values():
                a = r.read(1, window=from_bounds(lo - h, la - h, lo + h, la + h, r.transform))
                vals.append(np.nanmean(np.where(a > 0, a, np.nan)))
            box_cache[(i, j)] = np.array(vals)
        return box_cache[(i, j)]

    rows = []
    for loc, s in l.iterrows():
        tt = t[t.osm_id == s.osm_id]
        lats, lons = (tt.lat.values, tt.lon.values) if len(tt) else (np.array([s.lat]), np.array([s.lon]))
        pts = np.stack([[v[0] for v in rs[h].sample(zip(lons, lats))] for h in HEIGHTS], axis=-1)
        pts = pts[np.all(pts > 0, axis=1)]
        turb_hub = np.nanmean(at_height(pts, s.hub_est))
        turb_100 = np.nanmean(pts[:, 1])
        w = bilinear(s.lat, s.lon)
        cell = sum(wt * box_mean(*ij) for ij, wt in w)
        rows.append({"loc": loc, "n_pts": len(pts), "gwa_turb_100": turb_100, "gwa_turb_hub": turb_hub,
                     "gwa_cell_100": cell[1], "k_gwa": turb_hub / cell[1], "k_gwa_100": turb_100 / cell[1]})
    out = l.join(pd.DataFrame(rows).set_index("loc"))
    out.drop(columns=["units"]).to_parquet(DATA / "site_meta.parquet")
    out[["units"]].explode("units").rename(columns={"units": "bmu"}).reset_index() \
        .to_parquet(DATA / "bmu_loc.parquet")
    print(out[["offshore", "hub_est", "rotor_est", "gwa_turb_hub", "gwa_cell_100", "k_gwa", "k_gwa_100"]]
          .groupby("offshore").describe().T.round(3).to_string())
    return out

if __name__ == "__main__":
    main()
