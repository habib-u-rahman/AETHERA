"""
Step 12b (journal revision) - storm-conditional interval coverage.

Reports, for 2022-2024 and 2025 separately and for each horizon (3/6/12/24h), the coverage
of the 50/80/90% split-conformal intervals from code/14_probabilistic.py, restricted to hours
with observed Kp>=5 and, separately, Kp>=7.

Does NOT modify code/14_probabilistic.py or any of its saved results. 14_probabilistic.py has
no __main__ guard (importing it would re-run its whole pipeline, including rewriting its output
files, as a side effect), so the calibration logic needed to reproduce per-case interval bounds
is copied verbatim below rather than imported - same data, same seed, same method, deterministic
so the numbers are identical to what 14_probabilistic.py already computed internally.

Run:  python 14b_storm_coverage.py
Output: results/step5/storm_coverage.csv (also printed)
"""
import joblib
import numpy as np
import pandas as pd
from config import CLEAN, RESULTS, HOLDOUT_2003

STEP5 = RESULTS / "step5"
HORIZONS = [3, 6, 12, 24]
LOOKBACK = 24
FEATURES = ["B", "By", "Bz", "density", "speed", "pressure", "E_field", "dst", "f107", "kp"]
N_BINS = 5
INTERVAL_LEVELS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
STORM_THRESHOLDS = [5, 7]

# ====================================================================== data + features (verbatim from 14_probabilistic.py / 06_rq1_forecast.py)
df = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
df = df[FEATURES].asfreq("h")
t_index = df.index
X_all = df.to_numpy(dtype=np.float32)
n = len(df)
kp = df["kp"].to_numpy(dtype=np.float32)

Y_all = np.full((n, len(HORIZONS)), np.nan, dtype=np.float32)
for j, h in enumerate(HORIZONS):
    Y_all[:-h, j] = kp[h:]

bad = np.isnan(X_all).any(axis=1).astype(np.int32)
bad_in_window = np.convolve(bad, np.ones(LOOKBACK, dtype=np.int32))[:n]
valid = (bad_in_window == 0) & ~np.isnan(Y_all).any(axis=1)
valid[:LOOKBACK - 1] = False

def period(a, b):
    return (t_index >= pd.Timestamp(a)) & (t_index <= pd.Timestamp(b))

H = max(HORIZONS)
hold_a = pd.Timestamp(HOLDOUT_2003[0]); hold_b = pd.Timestamp(HOLDOUT_2003[1])
near_2003 = period(hold_a - pd.Timedelta(hours=LOOKBACK + H), hold_b + pd.Timedelta(hours=LOOKBACK))
def end_minus(day):
    return pd.Timestamp(day + " 23:00") - pd.Timedelta(hours=H)
sets = {
    "train":     valid & period("2000-01-02", end_minus("2018-12-31")) & ~near_2003,
    "val":       valid & period("2019-01-02", end_minus("2021-12-31")),
    "test":      valid & period("2022-01-02", end_minus("2024-12-31")),
    "test_2025": valid & period("2025-01-02", end_minus("2025-12-31")),
}

feat = {}
for c in FEATURES:
    s = df[c]
    feat[f"{c}_now"] = s
    for w in (3, 6, 12, 24):
        feat[f"{c}_mean{w}"] = s.rolling(w, min_periods=w).mean()
feat["Bz_min3"] = df["Bz"].rolling(3).min()
feat["Bz_min24"] = df["Bz"].rolling(24).min()
feat["speed_max24"] = df["speed"].rolling(24).max()
feat["kp_max24"] = df["kp"].rolling(24).max()
feat["kp_change3"] = df["kp"] - df["kp"].shift(3)
F_all = pd.DataFrame(feat).to_numpy(dtype=np.float32)
f_ok = ~np.isnan(F_all).any(axis=1)
for k in sets:
    sets[k] = sets[k] & f_ok

rf = joblib.load(RESULTS / "rf_kp.joblib")

def tree_preds(mask):
    X = F_all[mask]
    return np.stack([np.clip(tree.predict(X), 0, 9) for tree in rf.estimators_])

tp_val = tree_preds(sets["val"]); y_val = Y_all[sets["val"]]
pt_val = tp_val.mean(axis=0)
tp_test = tree_preds(sets["test"]); y_test = Y_all[sets["test"]]
pt_test = tp_test.mean(axis=0)
tp_2025 = tree_preds(sets["test_2025"]); y_2025 = Y_all[sets["test_2025"]]
pt_2025 = tp_2025.mean(axis=0)

bin_edges = {}
bin_residuals = {}
for j, h in enumerate(HORIZONS):
    edges = np.quantile(pt_val[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1])
    bin_edges[h] = edges
    bin_id = np.digitize(pt_val[:, j], edges)
    resid = y_val[:, j] - pt_val[:, j]
    bin_residuals[h] = [resid[bin_id == b] for b in range(N_BINS)]

def quantile_forecast(point_pred, h, tau):
    bin_id = np.digitize(point_pred, bin_edges[h])
    out = np.empty_like(point_pred, dtype=np.float64)
    for b in range(N_BINS):
        m = bin_id == b
        if not m.any():
            continue
        r = bin_residuals[h][b]
        q = np.quantile(r, tau) if len(r) else 0.0
        out[m] = point_pred[m] + q
    return np.clip(out, 0, 9)

# ====================================================================== NEW: storm-conditional coverage
rows = []
for period_name, pt, y in [("2022-2024", pt_test, y_test), ("2025", pt_2025, y_2025)]:
    for j, h in enumerate(HORIZONS):
        pj, yj = pt[:, j], y[:, j]
        for thr in STORM_THRESHOLDS:
            m = yj >= thr
            row = {"period": period_name, "horizon_h": h, "group": f"Kp>={thr}", "n": int(m.sum())}
            for level, (lo_t, hi_t) in INTERVAL_LEVELS.items():
                if m.sum() == 0:
                    row[f"coverage_{level}"] = np.nan
                    continue
                q_lo = quantile_forecast(pj, h, lo_t)[m]
                q_hi = quantile_forecast(pj, h, hi_t)[m]
                row[f"coverage_{level}"] = float(np.mean((yj[m] >= q_lo) & (yj[m] <= q_hi)))
            rows.append(row)

out = pd.DataFrame(rows)
STEP5.mkdir(parents=True, exist_ok=True)
out.to_csv(STEP5 / "storm_coverage.csv", index=False)
print(out.to_string(index=False))
print(f"\nSaved {STEP5 / 'storm_coverage.csv'}")
