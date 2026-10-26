"""
config.py - settings shared by all Step 3 scripts.
Edit this file only if you want to change years, folders or storm windows.
"""
from pathlib import Path

# ---------- folders (relative to the AETHERA project folder) ----------
ROOT = Path(__file__).resolve().parent.parent      # AETHERA/
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"
for p in (RAW / "omni", RAW / "tle", CLEAN, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------- OMNI (solar wind + Kp + Dst + F10.7, hourly) ----------
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
OMNI_YEARS = range(2000, 2026)          # 2000 ... 2025  (2025 = extra test year)

# Train / test split (never mix them!)
TRAIN_START, TRAIN_END = "2000-01-01", "2021-12-31 23:00"
TEST_START, TEST_END = "2022-01-01", "2024-12-31 23:00"
# October 2003 window: removed from training, used as an extra test
HOLDOUT_2003 = ("2003-10-15", "2003-11-15 23:00")

# ---------- the three storms ----------
# start = first day of strong activity (Kp >= 5); window = TLEs we download
STORMS = {
    "2003_halloween": {"start": "2003-10-29", "window": ("2003-10-22", "2003-11-08")},
    "2022_starlink":  {"start": "2022-02-03", "window": ("2022-01-27", "2022-02-13")},
    "2024_gannon":    {"start": "2024-05-10", "window": ("2024-05-03", "2024-05-19")},
}

# ---------- Space-Track (TLE history) ----------
# Your username and password are read from environment variables, NOT written here.
# Windows (PowerShell):  $env:ST_USER="you@mail.com";  $env:ST_PASS="yourpassword"
# Mac/Linux/Colab:       import os; os.environ["ST_USER"]="..."; os.environ["ST_PASS"]="..."
ST_LOGIN = "https://www.space-track.org/ajaxauth/login"
ST_QUERY = "https://www.space-track.org/basicspacedata/query"
# Only low orbits: mean motion > 15 rev/day (about below 560 km), near-circular orbits
MIN_MEAN_MOTION = 15.0
MAX_ECCENTRICITY = 0.01

EARTH_RADIUS_KM = 6378.137
