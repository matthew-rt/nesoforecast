"""Site registry for BM-registered wind units (transmission T_ and embedded E_): location + turbine metadata.

BMU -> OSM wind-farm polygon via (1) EIC code, (2) REPD id (OSUKED dictionary or manual match below),
(3) REPD point falling inside a polygon. Location is the centroid of OSM turbines inside the polygon,
falling back to the polygon centroid, then the REPD point (REPD offshore coordinates are often landfalls)."""
import json, pathlib
import numpy as np, pandas as pd, shapely, shapely.ops
from pyproj import Transformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data" / "raw" / "sites", ROOT / "data"

# REPD Ref IDs for sites the OSUKED dictionary doesn't link (mostly post-2021 builds); matched by name + capacity
MANUAL = {"SOFOW": 6556, "HOWAO": 2525, "HOWBO": 2502, "DBAWO": 2541, "DBBWO": 2541, "INCWO": 2527, "MOWEO": 2534,
          "MOWWO": 6158, "SGRWO": 2540, "VKNGW": 6828, "NNGAO": 2522, "KLGLW": 4386, "NOKYW": 6607,
          "GNFSW": 2490, "GORDW": 3662, "RREW": 2496, "RRWW": 2497, "LIMKW": 6005, "DUNGW": 4118,
          "BOWLW": 2546, "THNTO": 2499, "EAAO": 2524, "SANQW": 4441, "DOUGW": 5393, "PLOEW": 4480, "ENHLW": 3502, "STRNW": 4625, "CGTHW": 4664, "KILBW": 3202,
          "BENBW": 3503, "KYPEW": 4628, "DWEXW": 6826, "MILWW": 4682, "GLWSW": 4368, "MKHLW": 3260,
          "GLNKW": 4371, "HAGHW": 7456, "HAHAW": 7456, "MIDMW": 4440, "CRMLW": 4120, "BROCW": 6708,
          "CRDEW": 4633, "CRGHW": 4619, "WDRGW": 7562, "ACHRW": 4331, "TWSHW": 6590, "HALSW": 4636,
          "WHIHW": 4502, "BDCHW": 4646, "MYGPW": 4437, "FSDLW": 3886, "GNAPW": 4593, "TRLGW": 4073,
          "PGBIW": 4434, "KTHLW": 4516}
MANUAL_UNIT = {"T_CUMHW-1": 5455, "T_CUMHW-2": 3328, "T_CRYRW-4": 4631, "T_AKGLW-2": 4332, "T_AKGLW-3": 4591,
               "T_WISTW-2": 4431,
               # embedded (E_) units: OSUKED covers few of them; matched on name + capacity + operator, 2026-10-24
               "E_ABRTW-1": 4337, "E_ASLVW-1": 4645, "E_BETHW-1": 4669, "E_BLARW-1": 4605, "E_BNWKW-1": 4122,
               "E_BRDUW-1": 3119, "E_BRNLW-1": 4348, "E_BTUIW-2": 4525, "E_BTUIW-3": 3387, "E_BURBO": 2487,
               "E_CLFLW-1": 3448, "E_CMSTW-2": 4351, "E_CNCLW-1": 3846, "E_CRGTW-1": 4524, "E_GDSTW-1": 3666,
               "E_GLCHW-1": 4638, "E_GLOFW-1": 3311, "E_GRGRW-1": 4213, "E_HBHDW-1": 4225, "E_HLTWW-1": 4014,
               "E_HRHLW-1": 4157, "E_HYWDW-1": 2479, "E_KHLLW-1": 4988, "E_KINCW-1": 6136, "E_MINSW-1": 4517,
               "E_MOYEW-1": 3647, "E_PIBUW-1": 6147, "E_STRHW-1": 7476, "E_SWBKW-1": 4449, "E_TLYMW-1": 4599,
               "E_TULWW-1": 3190, "E_TULWW-2": 4365}
# OSM wind-farm ids for farms whose OSM object has no (or a wrong) repd:id tag
MANUAL_OSM = {"SOFOW": "w1157893252", "DBAWO": "w889977926", "DBBWO": "w1157893250", "KENNW": "r15073682",
              "TWSHW": "r15523672", "DWEXW": "r19405446", "HAGHW": "r19405219", "HAHAW": "r19405219",
              "T_GORDW-2": "r13615192", "T_WLNYO-2": "r20197306", "T_WLNYW-1": "r20197306",
              "T_WLNYO-3": "r20197356", "T_WLNYO-4": "r20197356"}  # keys: site code or BMU

# OSM objects that contain two farms with different turbines: split by the turbines' rated MW
SPLIT = {"T_BRBEO-1": ("w327949356", 8.0), "E_BURBO": ("w327949356", 3.6)}  # Burbo Bank Extension / Burbo Bank

def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")

def bmus():
    b = pd.DataFrame(json.load(open(SRC / "elexon_bmunits.json")))
    b = b[(b.fuelType == "WIND") & b.bmUnitType.isin(["T", "E"])].copy()  # E_ = embedded BM units (in FUELHH)
    b["cap"] = b.generationCapacity.astype(float)
    b["site"] = b.elexonBmUnit.str.extract(r"[TE]_([A-Z0-9]+)")[0]
    ids = pd.read_csv(SRC / "osuked_ids.csv", dtype=str)
    m = ids.assign(bmu=ids.sett_bmu_id.str.split(", ")).explode("bmu").drop_duplicates("bmu")
    b = b.merge(m[["bmu", "name", "old_repd_id", "new_repd_id"]], left_on="elexonBmUnit", right_on="bmu", how="left")
    r = repd()
    old2new = r.dropna(subset=["old_id"]).drop_duplicates("old_id").set_index("old_id").repd_id
    first = lambda s: s.str.split(", ").str[0]
    b["repd_id"] = pd.to_numeric(first(b.new_repd_id), errors="coerce")
    b["repd_id"] = b.repd_id.where(b.repd_id.isin(r.repd_id), first(b.old_repd_id).map(old2new))
    b["repd_id"] = b.site.map(MANUAL).fillna(b.repd_id)
    b["repd_id"] = b.elexonBmUnit.map(MANUAL_UNIT).fillna(b.repd_id)
    b = b.drop_duplicates("elexonBmUnit")
    return b[["elexonBmUnit", "site", "bmUnitName", "leadPartyName", "eic", "cap", "repd_id"]]

def repd():
    r = pd.read_csv(SRC / "repd.csv", encoding="latin1", dtype=str)
    r = r[r["Technology Type"].str.contains("Wind", na=False)].copy()
    r["repd_id"] = pd.to_numeric(r["Ref ID"], errors="coerce")
    r["old_id"] = r["Old Ref ID"].str.strip()
    x, y = num(r["X-coordinate"]).values, num(r["Y-coordinate"]).values
    r["repd_lon"], r["repd_lat"] = Transformer.from_crs(27700, 4326, always_xy=True).transform(x, y)
    r["offshore"] = r["Technology Type"].str.contains("Offshore")
    r["repd_tip_m"] = num(r["Height of Turbines (m)"]).values
    r["repd_turbine_mw"] = num(r["Turbine Capacity (MW)"]).values
    return r[["repd_id", "old_id", "Site Name", "offshore", "repd_lat", "repd_lon", "repd_tip_m", "repd_turbine_mw"]] \
        .rename(columns={"Site Name": "repd_name"})

def ring(pts):
    return [(p["lon"], p["lat"]) for p in pts]

def osm_plants():
    """Wind-farm ways/relations. Most GB farms are type=site relations listing turbines as `generator` members;
    others are multipolygons or closed ways. geom = boundary if mapped, else hull of member turbines."""
    rows = []
    for e in json.load(open(SRC / "osm_plant_geom.json"))["elements"]:
        t, members, g = e.get("tags", {}), [], None
        if e["type"] == "way":
            g = shapely.Polygon(ring(e["geometry"])) if len(e["geometry"]) >= 4 else None
        else:
            ms = e.get("members", [])
            members = [m["ref"] for m in ms if m["type"] == "node" and m.get("role") == "generator"]
            lines = [shapely.LineString(ring(m["geometry"])) for m in ms
                     if m["type"] == "way" and m.get("role") in ("outer", "") and len(m.get("geometry", [])) >= 2]
            if lines:
                g = shapely.union_all(list(shapely.ops.polygonize(lines)))
            pts = [(m["lon"], m["lat"]) for m in ms if m["type"] == "node" and "lat" in m]
            if (g is None or g.is_empty) and pts:
                g = shapely.MultiPoint(pts).convex_hull.buffer(0.002)
        if g is None or g.is_empty:
            continue
        rows.append({"osm_id": f"{e['type'][0]}{e['id']}", "osm_name": t.get("name"), "geom": shapely.make_valid(g),
                     "members": members,
                     "eics": [s.strip() for s in t.get("ref:EU:ENTSOE_EIC", "").split(";") if s.strip()],
                     "repd_ids": [int(s) for s in t.get("repd:id", "").replace(",", ";").split(";") if s.strip().isdigit()]})
    return pd.DataFrame(rows)

def osm_turbines(plants):
    e = json.load(open(SRC / "osm_wind.json"))["elements"]
    g = pd.DataFrame([{"node": x["id"], "lat": x["lat"], "lon": x["lon"], **x["tags"]} for x in e
                      if x["type"] == "node" and x["tags"].get("power") == "generator"])
    member_of = plants.explode("members").dropna(subset=["members"]).drop_duplicates("members") \
        .set_index("members").osm_id
    g["osm_id"] = g.node.map(member_of)
    # turbines not listed in a site relation: assign by boundary (~200 m buffer for hand-drawn outlines)
    free = g.osm_id.isna().values
    tree = shapely.STRtree(plants.geom.values)
    ti, pi = tree.query(shapely.points(g.lon[free], g.lat[free]), predicate="dwithin", distance=0.003)
    hit = pd.DataFrame({"t": g.index[free][ti], "osm_id": plants.osm_id.values[pi],
                        "area": shapely.area(plants.geom.values)[pi]}).sort_values("area").drop_duplicates("t")
    g.loc[hit.t, "osm_id"] = hit.osm_id.values
    out = pd.DataFrame({"node": g.node, "lat": g.lat, "lon": g.lon, "osm_id": g.osm_id,
                        "hub_m": num(g.get("height:hub")).values, "rotor_m": num(g.get("rotor:diameter")).values,
                        "tip_m": num(g.get("height")).values, "model": g.get("model"),
                        "manufacturer": g.get("manufacturer"),
                        "rated_mw": num(g.get("generator:output:electricity")).values})
    # generator:output:electricity is usually "3.6 MW" but sometimes kW
    kw = g.get("generator:output:electricity").astype(str).str.contains("kW", na=False).values
    out.loc[kw, "rated_mw"] /= 1000
    return out

def link(b, plants, r):
    b = b.merge(r, on="repd_id", how="left")
    eic = plants.explode("eics").dropna(subset=["eics"]).drop_duplicates("eics").set_index("eics").osm_id
    plants = plants.assign(area=shapely.area(plants.geom.values)).sort_values("area")
    rid = plants.explode("repd_ids").dropna(subset=["repd_ids"]).drop_duplicates("repd_ids").set_index("repd_ids").osm_id
    b["osm_id"], b["link"] = b.eic.map(eic), np.where(b.eic.map(eic).notna(), "eic", None)
    man = b.elexonBmUnit.map(MANUAL_OSM).fillna(b.site.map(MANUAL_OSM))
    b.loc[man.notna(), "osm_id"], b.loc[man.notna(), "link"] = man[man.notna()], "manual_osm"
    m = b.osm_id.isna() & b.repd_id.notna()
    b.loc[m, "osm_id"] = b.loc[m, "repd_id"].map(rid)
    b.loc[m & b.osm_id.notna(), "link"] = "repd_tag"
    # spatial: onshore REPD point inside a polygon (offshore REPD points are unreliable)
    tree = shapely.STRtree(plants.geom.values)
    m = b.osm_id.isna() & b.repd_lat.notna() & ~b.offshore.astype("boolean").fillna(False).astype(bool)
    for i in b.index[m]:
        hit = tree.query(shapely.Point(b.repd_lon[i], b.repd_lat[i]), predicate="dwithin", distance=0.005)
        if len(hit):
            b.loc[i, ["osm_id", "link"]] = plants.osm_id.values[hit[0]], "repd_point"
    return b

def main():
    plants, r = osm_plants(), repd()
    b = link(bmus(), plants, r)
    t = osm_turbines(plants)
    for bmu, (oid, mw) in SPLIT.items():
        sub = f"{oid}_{mw:g}"
        t.loc[(t.osm_id == oid) & (t.rated_mw == mw), "osm_id"] = sub
        b.loc[b.elexonBmUnit == bmu, ["osm_id", "link"]] = sub, "manual_split"
        plants = pd.concat([plants, plants[plants.osm_id == oid].assign(osm_id=sub)])
    t.to_parquet(OUT / "turbines.parquet")
    agg = t.dropna(subset=["osm_id"]).groupby("osm_id").agg(
        n_turb=("lat", "size"), t_lat=("lat", "mean"), t_lon=("lon", "mean"), hub_m=("hub_m", "median"),
        rotor_m=("rotor_m", "median"), tip_m=("tip_m", "median"), rated_mw=("rated_mw", "median"),
        model=("model", lambda s: s.mode().iat[0] if s.notna().any() else None))
    p = plants.set_index("osm_id")
    p["p_lon"], p["p_lat"] = shapely.get_x(shapely.centroid(p.geom.values)), shapely.get_y(shapely.centroid(p.geom.values))
    b = b.join(p[["osm_name", "p_lat", "p_lon"]], on="osm_id").join(agg, on="osm_id")
    b["lat"] = b.t_lat.fillna(b.p_lat).fillna(b.repd_lat)
    b["lon"] = b.t_lon.fillna(b.p_lon).fillna(b.repd_lon)
    b["loc_src"] = np.select([b.t_lat.notna(), b.p_lat.notna(), b.repd_lat.notna()], ["osm_turbines", "osm_polygon", "repd"], None)
    b["offshore"] = b.offshore.fillna(b.elexonBmUnit.str.match(r"[TE]_\w+O(-|$)")).astype(bool)
    cols = ["elexonBmUnit", "site", "bmUnitName", "cap", "offshore", "lat", "lon", "loc_src", "link", "osm_id", "osm_name",
            "repd_id", "repd_name", "n_turb", "hub_m", "rotor_m", "tip_m", "repd_tip_m", "rated_mw", "repd_turbine_mw", "model"]
    b = b[cols].sort_values("elexonBmUnit")
    b.to_csv(OUT / "sites.csv", index=False)
    print(b.loc_src.value_counts(dropna=False), "\nlink:", b.link.value_counts(dropna=False).to_dict())
    print("MW by loc_src:", b.groupby("loc_src", dropna=False).cap.sum().round().to_dict())
    return b

if __name__ == "__main__":
    main()
