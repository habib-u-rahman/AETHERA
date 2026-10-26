"""
Step 3.3 - Download historical TLEs from Space-Track for the three storm windows.
Before running, set your Space-Track login as environment variables (see config.py).
Run:  python 03_download_tles.py
Output: data/raw/tle/<storm>.csv  (one file per storm)

This version downloads each storm in SMALL PIECES (2 days at a time) and saves
each piece as soon as it arrives. If the connection breaks, just run the script
again: pieces you already have are skipped, only missing pieces are downloaded.

IMPORTANT rules from Space-Track (or your account can be blocked):
  * Do NOT run this script again and again in a loop.
  * Max 30 requests/minute and 300/hour. This script waits 15 s between pieces.
"""
import os
import time
import pandas as pd
import requests
from config import STORMS, RAW, ST_LOGIN, ST_QUERY, MIN_MEAN_MOTION, MAX_ECCENTRICITY

PIECE_DAYS = 2

user, pw = os.environ.get("ST_USER"), os.environ.get("ST_PASS")
if not user or not pw:
    raise SystemExit("Set ST_USER and ST_PASS first (see config.py).")

def login(s):
    r = s.post(ST_LOGIN, data={"identity": user, "password": pw}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text:
        raise SystemExit("Login failed - check your username/password.")

def get_piece(s, a, b, out):
    """Download one piece [a, b) and stream it straight into a file."""
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
    tmp.replace(out)              # only a COMPLETE piece gets its final name

with requests.Session() as s:
    login(s)
    print("Logged in to Space-Track.")

    for name, info in STORMS.items():
        final = RAW / "tle" / f"{name}.csv"
        if final.exists() and final.stat().st_size > 10_000:
            print(f"{name}: already downloaded, skipping")
            continue

        piece_dir = RAW / "tle" / f"{name}_pieces"
        piece_dir.mkdir(exist_ok=True)
        start, end = (pd.Timestamp(x) for x in info["window"])
        edges = list(pd.date_range(start, end, freq=f"{PIECE_DAYS}D"))
        if edges[-1] < end:
            edges.append(end)

        ok = True
        for a, b in zip(edges[:-1], edges[1:]):
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
                        login(s)
                    except Exception:
                        pass
            else:
                print(f"{name}: piece {a_s} to {b_s} FAILED 3 times. Run the script again later.")
                ok = False
            time.sleep(15)        # stay far below the rate limit

        if not ok:
            continue
        # join all pieces into one file
        parts = sorted(piece_dir.glob("*.csv"))
        df = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
        df = df.drop_duplicates()
        df.to_csv(final, index=False)
        print(f"{name}: joined {len(parts)} pieces -> {len(df):,} TLE rows -> {final}")

print("Done.")