"""
Step 4 - RQ1: Forecast Kp 3, 6, 12 and 24 hours ahead.
Compares three models:
    1. Persistence   - "Kp in the future = Kp now"   (the simple baseline)
    2. Random Forest - classic machine learning on summary features
    3. LSTM          - deep learning on the last 24 hours of data

Run:  python 06_rq1_forecast.py           (about 20-60 minutes on a laptop)
Quick test first (5 min, smaller data):  python 06_rq1_forecast.py --quick

Outputs in results/:
    rq1_metrics.csv            all scores (model x horizon x test set)
    rq1_summary.txt            the main table, easy to read
    rq1_forecasts.csv          every forecast on the test sets (needed later for RQ2)
    fig_rq1_rmse.png           error vs forecast horizon, for each model
    fig_rq1_<storm>.png        true vs forecast Kp during each storm
    lstm_kp.keras              the trained LSTM model
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from config import CLEAN, RESULTS, HOLDOUT_2003, STORMS

QUICK = "--quick" in sys.argv
SEED = 42
np.random.seed(SEED)

HORIZONS = [3, 6, 12, 24]          # hours ahead
LOOKBACK = 24                       # hours of history the LSTM sees
FEATURES = ["B", "By", "Bz", "density", "speed", "pressure", "E_field", "dst", "f107", "kp"]
STORM_KP = 5.0                      # NOAA G1 storm threshold
STORM_WEIGHT = 3.0                  # storm hours count 3x in training (they are rare)

# ------------------------------------------------------------------ data
print("Loading data...")
df = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
df = df[FEATURES].asfreq("h")       # make sure every hour exists (gaps -> NaN)
t_index = df.index
X_all = df.to_numpy(dtype=np.float32)
n = len(df)
kp = df["kp"].to_numpy(dtype=np.float32)

# targets: Kp at t + h for each horizon
Y_all = np.full((n, len(HORIZONS)), np.nan, dtype=np.float32)
for j, h in enumerate(HORIZONS):
    Y_all[:-h, j] = kp[h:]

# a time t is usable if the last LOOKBACK hours of inputs and all targets exist
bad = np.isnan(X_all).any(axis=1).astype(np.int32)
bad_in_window = np.convolve(bad, np.ones(LOOKBACK, dtype=np.int32))[:n]   # counts NaN rows in [t-23, t]
valid = (bad_in_window == 0) & ~np.isnan(Y_all).any(axis=1)
valid[:LOOKBACK - 1] = False

def period(a, b):
    return (t_index >= pd.Timestamp(a)) & (t_index <= pd.Timestamp(b))

H = max(HORIZONS)
hold_a = pd.Timestamp(HOLDOUT_2003[0]); hold_b = pd.Timestamp(HOLDOUT_2003[1])
# keep a 24 h safety gap around the 2003 test window so no training sample "sees" it
near_2003 = period(hold_a - pd.Timedelta(hours=LOOKBACK + H), hold_b + pd.Timedelta(hours=LOOKBACK))
def end_minus(day):      # last time t whose target t+H is still inside 'day'
    return pd.Timestamp(day + " 23:00") - pd.Timedelta(hours=H)
sets = {
    "train":     valid & period("2000-01-02", end_minus("2018-12-31")) & ~near_2003,
    "val":       valid & period("2019-01-02", end_minus("2021-12-31")),
    "test":      valid & period("2022-01-02", end_minus("2024-12-31")),
    "test_2003": valid & period(hold_a + pd.Timedelta(hours=LOOKBACK), hold_b - pd.Timedelta(hours=H)),
}
if QUICK:   # use every 6th training hour only, to test the code quickly
    idx = np.flatnonzero(sets["train"]); keep = np.zeros(n, bool); keep[idx[::6]] = True
    sets["train"] = keep
for k, m in sets.items():
    print(f"  {k:10s}: {m.sum():,} samples")

# normalise with TRAIN statistics only (never look at test data)
mu = np.nanmean(X_all[sets["train"]], axis=0)
sd = np.nanstd(X_all[sets["train"]], axis=0) + 1e-6
Xn_all = (X_all - mu) / sd

def sequences(mask):
    """Return LSTM inputs of shape (samples, LOOKBACK, features) for the given time mask."""
    ends = np.flatnonzero(mask)
    win = np.lib.stride_tricks.sliding_window_view(Xn_all, LOOKBACK, axis=0)   # (n-L+1, F, L)
    return np.transpose(win[ends - LOOKBACK + 1], (0, 2, 1)).astype(np.float32)

def weights(mask):
    storm = (Y_all[mask] >= STORM_KP).any(axis=1)
    return np.where(storm, STORM_WEIGHT, 1.0).astype(np.float32)

# ------------------------------------------------------------------ features for Random Forest
print("Building Random Forest features...")
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

# ------------------------------------------------------------------ model 1: persistence
preds = {}
def persistence(mask):
    return np.repeat(kp[mask][:, None], len(HORIZONS), axis=1)

# ------------------------------------------------------------------ model 2: Random Forest
print("Training Random Forest...")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=60 if QUICK else 150, max_depth=18, min_samples_leaf=5,
                           max_features=0.5, n_jobs=-1, random_state=SEED)
rf_mask = sets["train"].copy()
if not QUICK:   # every 3rd hour is enough for the forest (Kp itself changes every 3 h)
    idx = np.flatnonzero(rf_mask); rf_mask[:] = False; rf_mask[idx[::3]] = True
rf.fit(F_all[rf_mask], Y_all[rf_mask], sample_weight=weights(rf_mask))
print(f"  done in {time.time()-t0:.0f} s")

# ------------------------------------------------------------------ model 3: LSTM
print("Training LSTM...")
import tensorflow as tf
import keras
from keras import layers
tf.random.set_seed(SEED)
Xtr, Ytr, Wtr = sequences(sets["train"]), Y_all[sets["train"]], weights(sets["train"])
Xva, Yva = sequences(sets["val"]), Y_all[sets["val"]]
model = keras.Sequential([
    keras.Input(shape=(LOOKBACK, len(FEATURES))),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(HORIZONS)),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
t0 = time.time()
model.fit(Xtr, Ytr, sample_weight=Wtr, validation_data=(Xva, Yva),
          epochs=5 if QUICK else 40, batch_size=256, verbose=2,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
print(f"  done in {time.time()-t0:.0f} s")
model.save(RESULTS / "lstm_kp.keras")
del Xtr

# ------------------------------------------------------------------ evaluation
def scores(y, p, thr=STORM_KP):
    out = {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
           "MAE": float(np.mean(np.abs(p - y))),
           "r": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan}
    s = y >= STORM_KP
    out["RMSE_storm"] = float(np.sqrt(np.mean((p[s] - y[s]) ** 2))) if s.any() else np.nan
    yes_p, yes_t = p >= thr, s
    a = np.sum(yes_p & yes_t); b = np.sum(yes_p & ~yes_t)
    c = np.sum(~yes_p & yes_t); d = np.sum(~yes_p & ~yes_t)
    pod = a / (a + c) if a + c else np.nan
    pofd = b / (b + d) if b + d else np.nan
    out["POD"] = pod                                   # storms caught
    out["FAR"] = b / (a + b) if a + b else np.nan      # false alarms
    out["TSS"] = pod - pofd
    den = (a + c) * (c + d) + (a + b) * (b + d)
    out["HSS"] = 2 * (a * d - b * c) / den if den else np.nan
    out["n"] = int(len(y)); out["n_storm"] = int(s.sum())
    return out

def predict_all(m):
    P = {"Persistence": persistence(m),
         "RandomForest": rf.predict(F_all[m]).astype(np.float32),
         "LSTM": model.predict(sequences(m), batch_size=1024, verbose=0)}
    return {k: np.clip(v, 0, 9) for k, v in P.items()}

# Models trained to minimise error tend to under-predict rare peaks, so a forecast of
# Kp = 4.3 may already mean "storm coming". We pick, on the VALIDATION set only, the
# alarm level (2.33 ... 5.0) that gives the best HSS, then use it unchanged on the test sets.
# (HSS is used here because, for rare events, maximising TSS rewards raising far too many
#  false alarms; HSS balances hits against false alarms.)
ALARM_LEVELS = [2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0]
P_val, y_val = predict_all(sets["val"]), Y_all[sets["val"]]
best_thr = {}
for name, p in P_val.items():
    for j, h in enumerate(HORIZONS):
        hss = [scores(y_val[:, j], p[:, j], thr)["HSS"] for thr in ALARM_LEVELS]
        best_thr[(name, h)] = ALARM_LEVELS[int(np.nanargmax(hss))]

rows, fc = [], []
for set_name in ("test", "test_2003"):
    m = sets[set_name]
    y = Y_all[m]
    P = predict_all(m)
    for name, p in P.items():
        for j, h in enumerate(HORIZONS):
            base = scores(y[:, j], p[:, j])
            thr = best_thr[(name, h)]
            tuned = scores(y[:, j], p[:, j], thr)
            rows.append({"test_set": set_name, "model": name, "horizon_h": h, **base,
                         "alarm_level": thr, "POD_tuned": tuned["POD"], "FAR_tuned": tuned["FAR"],
                         "TSS_tuned": tuned["TSS"], "HSS_tuned": tuned["HSS"]})
    f = pd.DataFrame({"time": t_index[m], "test_set": set_name})
    for j, h in enumerate(HORIZONS):
        f[f"kp_true_{h}h"] = y[:, j]
        for name, p in P.items():
            f[f"{name}_{h}h"] = p[:, j]
    fc.append(f)

met = pd.DataFrame(rows)
met.to_csv(RESULTS / "rq1_metrics.csv", index=False)
fcast = pd.concat(fc, ignore_index=True)
fcast.to_csv(RESULTS / "rq1_forecasts.csv", index=False)

lines = ["AETHERA - RQ1 results (Kp forecast)",
         "TSS/HSS: storm alarm when forecast >= 5.  *_tuned: alarm level chosen (max HSS) on 2019-2021 validation data.", ""]
for set_name in ("test", "test_2003"):
    lines.append(f"=== Test set: {set_name} ===")
    for metric in ("RMSE", "RMSE_storm", "r", "TSS", "HSS", "alarm_level", "TSS_tuned", "HSS_tuned", "POD_tuned", "FAR_tuned"):
        tab = met[met.test_set == set_name].pivot(index="model", columns="horizon_h", values=metric)
        tab = tab.loc[["Persistence", "RandomForest", "LSTM"]]
        lines.append(f"{metric} (columns = hours ahead)")
        lines.append(tab.round(3).to_string()); lines.append("")
summary = "\n".join(lines)
(RESULTS / "rq1_summary.txt").write_text(summary)
print(summary)

# ------------------------------------------------------------------ figures
colors = {"Persistence": "tab:gray", "RandomForest": "tab:green", "LSTM": "tab:blue"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for ax, metric, title in zip(axes, ("RMSE", "RMSE_storm"), ("All hours, 2022-2024", "Storm hours (Kp >= 5), 2022-2024")):
    for name, col in colors.items():
        d = met[(met.test_set == "test") & (met.model == name)]
        ax.plot(d.horizon_h, d[metric], "o-", color=col, label=name)
    ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("RMSE (Kp units)")
    ax.set_title(title); ax.set_xticks(HORIZONS); ax.grid(alpha=0.3)
axes[0].legend()
fig.tight_layout(); fig.savefig(RESULTS / "fig_rq1_rmse.png", dpi=200); plt.close(fig)

for storm, info in STORMS.items():
    a, b = info["window"]
    d = fcast[(fcast.time >= a) & (fcast.time <= b)]
    if d.empty:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, h in zip(axes, (3, 12)):
        ax.plot(d.time + pd.Timedelta(hours=h), d[f"kp_true_{h}h"], color="black", lw=1.5, label="Observed Kp")
        for name, col in colors.items():
            ax.plot(d.time + pd.Timedelta(hours=h), d[f"{name}_{h}h"], color=col, lw=1, label=name)
        ax.axhline(STORM_KP, color="tab:red", ls="--", lw=0.8)
        ax.set_title(f"{storm}: forecast made {h} h ahead"); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Kp"); axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(RESULTS / f"fig_rq1_{storm}.png", dpi=200); plt.close(fig)
print("\nFigures and files saved in", RESULTS)