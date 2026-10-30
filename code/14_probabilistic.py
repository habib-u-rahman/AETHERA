"""
Step 12 (journal revision) - turn the RQ1 Random Forest into a probabilistic forecaster,
with NO new training data: split-conformal prediction intervals + isotonic storm probability,
calibrated on the untouched 2019-2021 validation set, evaluated on 2022-2024 and 2025.

Does NOT modify or overwrite any existing script or result file. All new output goes to
results/step5/.

Run:  python 14_probabilistic.py
Outputs in results/step5/:
  prob_metrics.csv, reliability_bins.csv, summary.txt,
  fig_prob_reliability.png, fig_prob_gannon.png
Then zips results/step5/ + this script into step5_results.zip.
"""
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CLEAN, RESULTS, HOLDOUT_2003, STORMS, ROOT

STEP5 = RESULTS / "step5"
STEP5.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 6, 12, 24]
LOOKBACK = 24
FEATURES = ["B", "By", "Bz", "density", "speed", "pressure", "E_field", "dst", "f107", "kp"]
STORM_KP = 5.0
SEED = 42
N_BINS = 5
INTERVAL_LEVELS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
CRPS_TAU_GRID = np.linspace(0.025, 0.975, 39)
TARGET_RMSE = {3: 0.807, 6: 0.997, 12: 1.151, 24: 1.265}   # the published RQ1 numbers to reproduce

# ====================================================================== 1. data + features (verbatim from 06_rq1_forecast.py / 09_test_2025.py)
print("Loading data...")
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

# ====================================================================== 2. load (or, failing that, retrain identically) the RQ1 Random Forest
RF_PATH = RESULTS / "rf_kp.joblib"
if RF_PATH.exists():
    print(f"Loading existing {RF_PATH} ...")
    rf = joblib.load(RF_PATH)
else:
    print("results/rf_kp.joblib not found - retraining identically to code/06_rq1_forecast.py ...")
    rf_mask = sets["train"].copy()
    idx = np.flatnonzero(rf_mask); rf_mask[:] = False; rf_mask[idx[::3]] = True
    w = np.where((Y_all[rf_mask] >= STORM_KP).any(axis=1), 3.0, 1.0)
    rf = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5, max_features=0.5,
                               n_jobs=-1, random_state=SEED).fit(F_all[rf_mask], Y_all[rf_mask], sample_weight=w)

def rf_point(mask):
    return np.clip(rf.predict(F_all[mask]).astype(np.float32), 0, 9)

print("\n=== Verifying point forecasts reproduce rq1_forecasts.csv (test set, 2022-2024) ===")
p_test = rf_point(sets["test"])
y_test = Y_all[sets["test"]]
all_pass = True
for j, h in enumerate(HORIZONS):
    rmse = float(np.sqrt(np.mean((p_test[:, j] - y_test[:, j]) ** 2)))
    ok = abs(rmse - TARGET_RMSE[h]) < 0.002
    all_pass &= ok
    print(f"  {h:>2d}h: RMSE = {rmse:.3f}  (target {TARGET_RMSE[h]:.3f})  {'OK' if ok else '** MISMATCH **'}")
print("Point-forecast check:", "PASSED - this is exactly the RQ1 model." if all_pass else "FAILED - see above.")

# ====================================================================== 3. per-tree predictions
def tree_preds(mask):
    """Shape (n_trees, n_samples, n_horizons)."""
    X = F_all[mask]
    return np.stack([np.clip(tree.predict(X), 0, 9) for tree in rf.estimators_])

print("\nComputing per-tree predictions for val / test / test_2025 ...")
tp_val = tree_preds(sets["val"]); y_val = Y_all[sets["val"]]
tp_test = tree_preds(sets["test"])
tp_2025 = tree_preds(sets["test_2025"]); y_2025 = Y_all[sets["test_2025"]]
pt_val = tp_val.mean(axis=0); pt_test = tp_test.mean(axis=0); pt_2025 = tp_2025.mean(axis=0)

# ====================================================================== 4a. split-conformal, binned by predicted Kp
bin_edges = {}      # h -> array of N_BINS-1 interior edges (quantiles of validation predictions)
bin_residuals = {}  # h -> list of residual arrays, one per bin
for j, h in enumerate(HORIZONS):
    edges = np.quantile(pt_val[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1])
    bin_edges[h] = edges
    bin_id = np.digitize(pt_val[:, j], edges)
    resid = y_val[:, j] - pt_val[:, j]
    bin_residuals[h] = [resid[bin_id == b] for b in range(N_BINS)]

def quantile_forecast(point_pred, h, tau):
    """point_pred: array of point predictions for horizon h. Returns q_tau(x) for each x,
    using the validation-calibrated, predicted-Kp-binned residual quantiles (split-conformal)."""
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

# ====================================================================== 4b. storm probability: tree fraction + isotonic calibration
isotonic = {}
for j, h in enumerate(HORIZONS):
    raw_frac_val = (tp_val[:, :, j] >= STORM_KP).mean(axis=0)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_frac_val, (y_val[:, j] >= STORM_KP).astype(float))
    isotonic[h] = iso

def storm_prob(tp, h_idx, h):
    raw_frac = (tp[:, :, h_idx] >= STORM_KP).mean(axis=0)
    return isotonic[h].predict(raw_frac)

# ====================================================================== 5. evaluation
def crps_from_quantiles(y, point_pred, h):
    """CRPS via the quantile/pinball-loss integral identity, using CRPS_TAU_GRID."""
    total = np.zeros_like(y, dtype=np.float64)
    for tau in CRPS_TAU_GRID:
        q = quantile_forecast(point_pred, h, tau)
        total += np.where(y >= q, (y - q) * tau, (q - y) * (1 - tau))
    return 2.0 * total / len(CRPS_TAU_GRID)

def evaluate(period_name, tp, point_pred, y):
    rows = []
    rel_rows = []
    for j, h in enumerate(HORIZONS):
        yj, pj, tpj = y[:, j], point_pred[:, j], tp[:, :, j]
        row = {"period": period_name, "horizon_h": h, "n": len(yj)}

        for level, (lo_t, hi_t) in INTERVAL_LEVELS.items():
            q_lo = quantile_forecast(pj, h, lo_t)
            q_hi = quantile_forecast(pj, h, hi_t)
            row[f"coverage_{level}"] = float(np.mean((yj >= q_lo) & (yj <= q_hi)))
            row[f"width_{level}"] = float(np.mean(q_hi - q_lo))

        crps = crps_from_quantiles(yj, pj, h)
        train_y = Y_all[sets["train"], j]
        crps_clim_vals = np.zeros_like(yj, dtype=np.float64)
        for tau in CRPS_TAU_GRID:
            qc = np.quantile(train_y, tau)
            crps_clim_vals += np.where(yj >= qc, (yj - qc) * tau, (qc - yj) * (1 - tau))
        crps_clim_vals = 2.0 * crps_clim_vals / len(CRPS_TAU_GRID)
        row["CRPS"] = float(crps.mean())
        row["CRPS_climatology"] = float(crps_clim_vals.mean())
        row["CRPSS"] = 1.0 - row["CRPS"] / row["CRPS_climatology"]

        p_cal = storm_prob(tp, j, h)
        obs_storm = (yj >= STORM_KP).astype(float)
        brier = float(np.mean((p_cal - obs_storm) ** 2))
        p_clim = float(np.mean(Y_all[sets["train"], j] >= STORM_KP))
        brier_clim = float(np.mean((p_clim - obs_storm) ** 2))
        row["Brier"] = brier
        row["Brier_climatology"] = brier_clim
        row["BSS"] = 1.0 - brier / brier_clim if brier_clim > 0 else np.nan
        row["climatological_storm_prob"] = p_clim
        row["ROC_AUC"] = float(roc_auc_score(obs_storm, p_cal)) if obs_storm.std() > 0 else np.nan
        rows.append(row)

        bins = np.linspace(0, 1, 11)
        bin_id = np.clip(np.digitize(p_cal, bins[1:-1]), 0, 9)
        for b in range(10):
            m = bin_id == b
            if m.sum() == 0:
                continue
            rel_rows.append({"period": period_name, "horizon_h": h, "bin": b,
                             "bin_lo": bins[b], "bin_hi": bins[b + 1], "n": int(m.sum()),
                             "mean_predicted_prob": float(p_cal[m].mean()),
                             "observed_frequency": float(obs_storm[m].mean())})
    return rows, rel_rows

rows_test, rel_test = evaluate("2022-2024", tp_test, pt_test, y_test)
rows_2025, rel_2025 = evaluate("2025", tp_2025, pt_2025, y_2025)
metrics = pd.DataFrame(rows_test + rows_2025)
reliability = pd.DataFrame(rel_test + rel_2025)
metrics.to_csv(STEP5 / "prob_metrics.csv", index=False)
reliability.to_csv(STEP5 / "reliability_bins.csv", index=False)

# ====================================================================== 6a. figure: reliability diagrams at 3h and 12h
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, h in zip(axes, (3, 12)):
    sub = reliability[(reliability.period == "2022-2024") & (reliability.horizon_h == h)].sort_values("bin")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    ax.plot(sub.mean_predicted_prob, sub.observed_frequency, "o-", color="tab:blue", label="RF (calibrated)")
    ax.set_xlabel("Forecast probability of Kp >= 5"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability, {h}h ahead (2022-2024)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(STEP5 / "fig_prob_reliability.png", dpi=200); plt.close(fig)

# ====================================================================== 6b. figure: May 2024 Gannon storm, RF + 80% interval, 3h and 12h
a, b = STORMS["2024_gannon"]["window"]
win_mask = sets["test"] & period(a, b)
t_win = t_index[win_mask]
p_win = rf_point(win_mask)
y_win = Y_all[win_mask]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, h in zip(axes, (3, 12)):
    j = HORIZONS.index(h)
    q_lo = quantile_forecast(p_win[:, j], h, 0.10)
    q_hi = quantile_forecast(p_win[:, j], h, 0.90)
    x = t_win + pd.Timedelta(hours=h)
    ax.fill_between(x, q_lo, q_hi, color="tab:green", alpha=0.25, label="80% interval")
    ax.plot(x, p_win[:, j], color="tab:green", lw=1, label="RF point forecast")
    ax.plot(x, y_win[:, j], color="black", lw=1.5, label="Observed Kp")
    ax.axhline(STORM_KP, color="tab:red", ls="--", lw=0.8)
    ax.set_title(f"May 2024 (Gannon) storm: forecast made {h}h ahead"); ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Kp"); axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(STEP5 / "fig_prob_gannon.png", dpi=200); plt.close(fig)

# ====================================================================== 7. summary.txt
lines = ["AETHERA - RQ1 Random Forest made probabilistic (split-conformal + isotonic), no new training data",
         "", "Point-forecast reproduction check (test=2022-2024):"]
for h in HORIZONS:
    lines.append(f"  {h}h target RMSE {TARGET_RMSE[h]:.3f}")
lines.append("PASSED - rf_kp.joblib is exactly the RQ1 Random Forest." if all_pass else "FAILED - see script output.")
lines.append("")
lines.append("Calibration: split-conformal prediction intervals (residual quantiles binned into "
            f"{N_BINS} bins by predicted Kp, fit on 2019-2021 validation only), and isotonic regression "
            "on the fraction of trees predicting Kp>=5 (fit on validation only). No refitting on test data.")
lines.append("")
for period_name in ["2022-2024", "2025"]:
    sub = metrics[metrics.period == period_name].set_index("horizon_h")
    lines.append(f"=== {period_name} ===")
    cols = ["n", "coverage_50", "width_50", "coverage_80", "width_80", "coverage_90", "width_90",
            "CRPS", "CRPS_climatology", "CRPSS", "Brier", "Brier_climatology", "BSS", "ROC_AUC"]
    lines.append(sub[cols].round(4).to_string()); lines.append("")
summary = "\n".join(lines)
(STEP5 / "summary.txt").write_text(summary)
print("\n" + summary)

# ====================================================================== 8. package
out_zip = ROOT / "step5_results.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in STEP5.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(ROOT))
    z.write(Path(__file__), f"code/{Path(__file__).name}")
print(f"\n{out_zip}: {out_zip.stat().st_size:,} bytes ({out_zip.stat().st_size/1e6:.2f} MB)")
