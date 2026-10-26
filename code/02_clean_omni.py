"""
Step 3.2 - Clean the OMNI files and make one hourly table.
Run:  python 02_clean_omni.py
Outputs (in data/clean/):
    omni_hourly.csv      all years, cleaned
    train.csv            2000-2021 (without Oct-Nov 2003)
    test.csv             2022-2024
    test_2003.csv        Oct-Nov 2003 (extra test storm)
    data_report.txt      how much data is missing, number of storm hours, etc.
"""
import numpy as np
import pandas as pd
from config import RAW, CLEAN, OMNI_YEARS, TRAIN_START, TRAIN_END, TEST_START, TEST_END, HOLDOUT_2003

# OMNI2 word numbers (1-based) -> our column names, and the "missing" fill value
COLS = {
    1: ("year", None), 2: ("doy", None), 3: ("hour", None),
    9: ("B", 999.9), 16: ("By", 999.9), 17: ("Bz", 999.9),
    24: ("density", 999.9), 25: ("speed", 9999.0), 29: ("pressure", 99.99),
    36: ("E_field", 999.99), 39: ("kp10", 99), 41: ("dst", 99999),
    51: ("f107", 999.9),
}

frames = []
for year in OMNI_YEARS:
    f = RAW / "omni" / f"omni2_{year}.dat"
    if not f.exists():
        print("Missing file, run 01_download_omni.py first:", f)
        continue
    raw = pd.read_csv(f, sep=r"\s+", header=None)
    if raw.shape[1] < 55:
        raise ValueError(f"{f.name}: expected 55 columns, got {raw.shape[1]}")
    df = pd.DataFrame({name: raw[w - 1] for w, (name, _) in COLS.items()})
    for w, (name, fill) in COLS.items():
        if fill is not None:
            df.loc[np.isclose(df[name].astype(float), fill), name] = np.nan
    frames.append(df)

df = pd.concat(frames, ignore_index=True)
df["time"] = (pd.to_datetime(df["year"].astype(str), format="%Y")
              + pd.to_timedelta(df["doy"] - 1, unit="D")
              + pd.to_timedelta(df["hour"], unit="h"))
df["kp"] = df["kp10"] / 10.0           # OMNI stores Kp x 10 (e.g. 33 = 3+ = 3.3)
df = df.drop(columns=["year", "doy", "hour", "kp10"]).set_index("time").sort_index()
df = df[~df.index.duplicated()]

# Fill SHORT gaps only (up to 3 hours) by straight-line interpolation.
# Longer gaps stay empty; the model step will skip them.
missing_before = df.isna().mean() * 100

def fill_short_gaps(s, max_len=3):
    """Interpolate only gaps of <= max_len hours; leave longer gaps empty."""
    gap = s.isna()
    gap_id = (~gap).cumsum()
    gap_len = gap.groupby(gap_id).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~(gap & (gap_len <= max_len)), filled)

df = df.apply(fill_short_gaps)
missing_after = df.isna().mean() * 100

df["storm"] = (df["kp"] >= 5).astype(int)   # NOAA G1 or stronger
df.to_csv(CLEAN / "omni_hourly.csv")

# ---------- split ----------
hold = (df.index >= HOLDOUT_2003[0]) & (df.index <= HOLDOUT_2003[1])
train = df.loc[TRAIN_START:TRAIN_END]
train = train[~((train.index >= HOLDOUT_2003[0]) & (train.index <= HOLDOUT_2003[1]))]
test = df.loc[TEST_START:TEST_END]
test03 = df[hold]
train.to_csv(CLEAN / "train.csv"); test.to_csv(CLEAN / "test.csv"); test03.to_csv(CLEAN / "test_2003.csv")

# ---------- report ----------
lines = [
    "AETHERA - OMNI data report",
    f"Time range: {df.index.min()} to {df.index.max()}  ({len(df):,} hours)",
    "",
    "Missing values (%)   before -> after filling gaps of <= 3 h",
]
for c in missing_before.index:
    lines.append(f"  {c:10s} {missing_before[c]:6.2f} -> {missing_after[c]:6.2f}")
lines += [
    "",
    f"Train hours: {len(train):,}   storm hours (Kp>=5): {train['storm'].sum():,} ({train['storm'].mean()*100:.2f}%)",
    f"Test hours:  {len(test):,}   storm hours (Kp>=5): {test['storm'].sum():,} ({test['storm'].mean()*100:.2f}%)",
    f"2003 test:   {len(test03):,}   storm hours (Kp>=5): {test03['storm'].sum():,}",
    "",
    "Max Kp in each storm window:",
    f"  Oct 2003: {df.loc['2003-10-28':'2003-11-01','kp'].max()}",
    f"  Feb 2022: {df.loc['2022-02-02':'2022-02-05','kp'].max()}",
    f"  May 2024: {df.loc['2024-05-09':'2024-05-12','kp'].max()}",
]
report = "\n".join(lines)
(CLEAN / "data_report.txt").write_text(report)
print(report)
