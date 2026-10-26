"""
Step 3.4 - Clean the TLE files and measure how much height each object lost.
Run:  python 04_clean_tles.py
Outputs (in data/clean/):
    tle_<storm>.csv           cleaned TLEs with altitude (km)
    decay_<storm>.csv         one row per object: altitude before/after the storm,
                              altitude loss, B*, object type, maneuver flag
This "altitude loss" is the GROUND TRUTH you will test your risk score against (RQ2).
"""
import numpy as np
import pandas as pd
from config import STORMS, RAW, CLEAN, EARTH_RADIUS_KM

MU = 398600.4418  # km^3/s^2, Earth's gravity constant

def altitude_from_mean_motion(n_rev_per_day):
    """Mean altitude (km) from mean motion (rev/day), via Kepler's third law."""
    n = n_rev_per_day * 2 * np.pi / 86400.0          # rad/s
    a = (MU / n**2) ** (1 / 3)                         # semi-major axis, km
    return a - EARTH_RADIUS_KM

for name, info in STORMS.items():
    f = RAW / "tle" / f"{name}.csv"
    if not f.exists():
        print(f"{name}: no TLE file yet, skipping (run 03_download_tles.py)")
        continue
    t = pd.read_csv(f, low_memory=False)
    t.columns = [c.upper() for c in t.columns]
    need = ["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "BSTAR"]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise ValueError(f"{name}: columns missing {missing}")
    t["EPOCH"] = pd.to_datetime(t["EPOCH"])
    t["ALT_KM"] = altitude_from_mean_motion(t["MEAN_MOTION"].astype(float))
    if "OBJECT_TYPE" not in t.columns:
        t["OBJECT_TYPE"] = "UNKNOWN"

    # remove impossible values and exact duplicates
    n0 = len(t)
    t = t[(t["ALT_KM"] > 100) & (t["ALT_KM"] < 1000)]
    t = t.drop_duplicates(subset=["NORAD_CAT_ID", "EPOCH"]).sort_values(["NORAD_CAT_ID", "EPOCH"])
    t.to_csv(CLEAN / f"tle_{name}.csv", index=False)

    start = pd.Timestamp(info["start"])
    rows = []
    for sat, g in t.groupby("NORAD_CAT_ID"):
        before = g[(g["EPOCH"] >= start - pd.Timedelta(days=3)) & (g["EPOCH"] < start)]
        after = g[(g["EPOCH"] >= start + pd.Timedelta(days=3)) & (g["EPOCH"] < start + pd.Timedelta(days=5))]
        if len(before) < 2 or len(after) < 1:
            continue                                   # not enough TLEs to judge
        # a rise of more than 0.5 km between TLEs = engine burn (maneuver)
        steps = g.loc[(g["EPOCH"] >= start - pd.Timedelta(days=3)) &
                      (g["EPOCH"] < start + pd.Timedelta(days=5)), "ALT_KM"].diff()
        rows.append({
            "norad_id": sat,
            "name": g["OBJECT_NAME"].iloc[-1],
            "object_type": g["OBJECT_TYPE"].iloc[-1],
            "alt_before_km": before["ALT_KM"].median(),
            "alt_after_km": after["ALT_KM"].median(),
            "bstar_before": before["BSTAR"].astype(float).median(),
            "maneuver": bool((steps > 0.5).any()),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        print(f"{name}: no objects with TLEs both before and after the storm")
        continue
    d["alt_loss_km"] = d["alt_before_km"] - d["alt_after_km"]
    d.to_csv(CLEAN / f"decay_{name}.csv", index=False)
    clean = d[~d["maneuver"]]
    print(f"{name}: {n0:,} TLE rows -> {len(t):,} after cleaning; "
          f"{len(d):,} objects measured, {d['maneuver'].sum():,} flagged as maneuvering; "
          f"median altitude loss (non-maneuvering) = {clean['alt_loss_km'].median():.3f} km")
