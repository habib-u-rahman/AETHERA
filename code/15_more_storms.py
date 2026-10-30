"""
Step 13 (journal revision) - Part A: five new storms for RQ2, download + clean.

Does NOT modify or overwrite code/03_download_tles.py, code/04_clean_tles.py, config.py, or
any existing data/results file. The download/clean logic is copied verbatim from those two
scripts (same piece size, same rate limits, same filters, same cleaning rules, same maneuver
test) and applied only to NEW_STORMS below - a separate dict, not config.STORMS. Output goes to
data/raw/tle/<storm>.csv (cached raw TLEs, same as script 03 would produce) and
data/clean/tle_<storm>.csv / decay_<storm>.csv (same as script 04 would produce).

Space-Track login is read only from .env (ST_USER / ST_PASS), never printed.

Run:
  python 15_more_storms.py --list-only    (step 1: identify Kp>=7 storms 2015-2025 from the
                                             cleaned OMNI data, print them - NO network calls)
  python 15_more_storms.py --download     (step 2: download + clean the 5 confirmed storms;
                                             resumable - piece files already on disk are
                                             skipped, same as script 03; self-limits to a time
                                             budget per invocation so it exits cleanly rather
                                             than risking an external kill - re-run to continue)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from config import CLEAN, RAW, ROOT, EARTH_RADIUS_KM, ST_LOGIN, ST_QUERY, MIN_MEAN_MOTION, MAX_ECCENTRICITY

load_dotenv()
STEP6 = ROOT / "results" / "step6"
STEP6.mkdir(parents=True, exist_ok=True)
PIECE_DAYS = 2
MU = 398600.4418

# ====================================================================== the 5 new storms (confirmed with the user)
NEW_STORMS = {
    "2015_stpatrick": {"start": "2015-03-17", "window": ("2015-03-10", "2015-03-27")},
    "2017_september": {"start": "2017-09-07", "window": ("2017-08-31", "2017-09-17")},
    "2023_april":     {"start": "2023-04-23", "window": ("2023-04-16", "2023-05-03")},
    "2024_october":   {"start": "2024-10-10", "window": ("2024-10-03", "2024-10-20")},
    "2025_november":  {"start": "2025-11-12", "window": ("2025-11-05", "2025-11-22")},
}
# storms inside the RF training period (2000-01-02 .. 2018-12-31) - no out-of-sample RF forecast
IN_SAMPLE_STORMS = {"2015_stpatrick", "2017_september"}

# ====================================================================== step 1: candidate list (no network)
def list_candidates():
    omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
    sub = omni.loc["2015-01-01":"2025-12-31", "kp"]
    hi = sub[sub >= 5]
    gaps = hi.index.to_series().diff() > pd.Timedelta(hours=48)
    group_id = gaps.cumsum()
    events = []
    for gid, g in hi.groupby(group_id):
        onset = g.index.min()
        win = sub.loc[onset - pd.Timedelta(hours=24): g.index.max() + pd.Timedelta(hours=24)]
        events.append({"onset": onset, "max_kp": win.max(), "max_kp_time": win.idxmax()})
    ev = pd.DataFrame(events)
    ev = ev[ev.max_kp >= 7].reset_index(drop=True)
    print(ev.to_string())
    print(f"\n{len(ev)} qualifying storms (max Kp>=7), 2015-2025.")
    print("\nConfirmed selection (see NEW_STORMS in this script):")
    for name, info in NEW_STORMS.items():
        note = " [IN-SAMPLE: RF forecast N/A, will use observed Kp]" if name in IN_SAMPLE_STORMS else " [out-of-sample: RF forecast available]"
        print(f"  {name}: start={info['start']}  window={info['window']}{note}")

# ====================================================================== step 2a: download (verbatim logic from 03_download_tles.py)
def login(s, user, pw):
    r = s.post(ST_LOGIN, data={"identity": user, "password": pw}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text:
        raise SystemExit("Login failed - check ST_USER/ST_PASS in .env.")

def get_piece(s, a, b, out):
    url = (f"{ST_QUERY}/class/gp_history"
           f"/EPOCH/{a}--{b}"
           f"/MEAN_MOTION/%3E{MIN_MEAN_MOTION}"
           f"/ECCENTRICITY/%3C{MAX_ECCENTRICITY}"
           f"/orderby/NORAD_CAT_ID,EPOCH/format/csv")
    tmp = out.with_suffix(".part")
    with s.get(url, timeout=(30, 900), stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(out)

def download(time_budget_s=1100):
    user, pw = os.environ.get("ST_USER"), os.environ.get("ST_PASS")
    if not user or not pw:
        raise SystemExit("Set ST_USER and ST_PASS in .env first.")
    start_time = time.time()
    with requests.Session() as s:
        login(s, user, pw)
        print("Logged in to Space-Track.")
        for name, info in NEW_STORMS.items():
            if time.time() - start_time > time_budget_s:
                print(f"Time budget ({time_budget_s:.0f}s) reached - stopping cleanly. Re-run to continue.")
                return
            final = RAW / "tle" / f"{name}.csv"
            if final.exists() and final.stat().st_size > 10_000:
                print(f"{name}: already downloaded, skipping")
                continue

            piece_dir = RAW / "tle" / f"{name}_pieces"
            piece_dir.mkdir(exist_ok=True)
            win_start, win_end = (pd.Timestamp(x) for x in info["window"])
            edges = list(pd.date_range(win_start, win_end, freq=f"{PIECE_DAYS}D"))
            if edges[-1] < win_end:
                edges.append(win_end)

            ok = True
            for a, b in zip(edges[:-1], edges[1:]):
                if time.time() - start_time > time_budget_s:
                    print(f"Time budget ({time_budget_s:.0f}s) reached mid-storm ({name}) - stopping cleanly. Re-run to continue.")
                    return
                a_s, b_s = a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")
                out = piece_dir / f"{a_s}_{b_s}.csv"
                if out.exists():
                    print(f"{name}: piece {a_s} to {b_s} already done")
                    continue
                for attempt in (1, 2, 3):
                    try:
                        print(f"{name}: piece {a_s} to {b_s} (attempt {attempt})...")
                        get_piece(s, a_s, b_s, out)
                        print(f"   saved {out.stat().st_size/1e6:.1f} MB")
                        break
                    except requests.RequestException as e:
                        print(f"   broke: {type(e).__name__}. Waiting 60 s, then retrying.")
                        time.sleep(60)
                        try:
                            login(s, user, pw)
                        except Exception:
                            pass
                else:
                    print(f"{name}: piece {a_s} to {b_s} FAILED 3 times. Run the script again later.")
                    ok = False
                time.sleep(15)   # stay far below the rate limit, same as script 03

            if not ok:
                continue
            parts = sorted(piece_dir.glob("*.csv"))
            df = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
            df = df.drop_duplicates()
            df.to_csv(final, index=False)
            print(f"{name}: joined {len(parts)} pieces -> {len(df):,} TLE rows -> {final}")
    print("Download pass complete (or time-budget-limited - check above for any storms still pending).")

# ====================================================================== step 2b: clean (verbatim logic from 04_clean_tles.py)
def altitude_from_mean_motion(n_rev_per_day):
    n = n_rev_per_day * 2 * np.pi / 86400.0
    a = (MU / n**2) ** (1 / 3)
    return a - EARTH_RADIUS_KM

def clean():
    for name, info in NEW_STORMS.items():
        f = RAW / "tle" / f"{name}.csv"
        if not f.exists():
            print(f"{name}: no TLE file yet, skipping (run --download first)")
            continue
        out_tle = CLEAN / f"tle_{name}.csv"
        if out_tle.exists():
            print(f"{name}: already cleaned, skipping")
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

        n0 = len(t)
        t = t[(t["ALT_KM"] > 100) & (t["ALT_KM"] < 1000)]
        t = t.drop_duplicates(subset=["NORAD_CAT_ID", "EPOCH"]).sort_values(["NORAD_CAT_ID", "EPOCH"])
        t.to_csv(out_tle, index=False)

        start = pd.Timestamp(info["start"])
        rows = []
        for sat, g in t.groupby("NORAD_CAT_ID"):
            before = g[(g["EPOCH"] >= start - pd.Timedelta(days=3)) & (g["EPOCH"] < start)]
            after = g[(g["EPOCH"] >= start + pd.Timedelta(days=3)) & (g["EPOCH"] < start + pd.Timedelta(days=5))]
            if len(before) < 2 or len(after) < 1:
                continue
            steps = g.loc[(g["EPOCH"] >= start - pd.Timedelta(days=3)) &
                          (g["EPOCH"] < start + pd.Timedelta(days=5)), "ALT_KM"].diff()
            rows.append({
                "norad_id": sat, "name": g["OBJECT_NAME"].iloc[-1], "object_type": g["OBJECT_TYPE"].iloc[-1],
                "alt_before_km": before["ALT_KM"].median(), "alt_after_km": after["ALT_KM"].median(),
                "bstar_before": before["BSTAR"].astype(float).median(), "maneuver": bool((steps > 0.5).any()),
            })
        d = pd.DataFrame(rows)
        if d.empty:
            print(f"{name}: no objects with TLEs both before and after the storm")
            continue
        d["alt_loss_km"] = d["alt_before_km"] - d["alt_after_km"]
        d.to_csv(CLEAN / f"decay_{name}.csv", index=False)
        clean_d = d[~d["maneuver"]]
        print(f"{name}: {n0:,} TLE rows -> {len(t):,} after cleaning; "
              f"{len(d):,} objects measured, {d['maneuver'].sum():,} flagged as maneuvering; "
              f"median altitude loss (non-maneuvering) = {clean_d['alt_loss_km'].median():.3f} km")

if __name__ == "__main__":
    if "--list-only" in sys.argv:
        list_candidates()
    elif "--download" in sys.argv:
        budget = float(os.environ.get("STORMS_TIME_BUDGET_S", 1100))
        download(time_budget_s=budget)
        clean()
    else:
        print(__doc__)
