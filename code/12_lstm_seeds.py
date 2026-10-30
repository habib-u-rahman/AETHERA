"""
Step 10 (journal revision) - LSTM SEED STABILITY.

Retrains ONLY the LSTM from code/06_rq1_forecast.py, for seeds 1-5, with the exact same
architecture, data, features and training settings as script 06 (copied verbatim below).
Persistence and Random Forest are seed-independent (RF uses the same seed=42 as script 06
in every run) so they are trained/computed ONCE and reused across the 5 LSTM seeds, exactly
reproducing script 06's own numbers - not retrained 5 times for no reason.
Also evaluates on the 2025 holdout (same test_2025 definition as code/09_test_2025.py).

Does NOT modify or overwrite results/lstm_kp.keras or any other existing result file.
All output goes to results/step2/.

Run:  python 12_lstm_seeds.py            (full run, ~5 x 10-40 min depending on hardware)
      python 12_lstm_seeds.py --quick    (fast smoke test, smaller data / fewer epochs, to check the code before the real run)
Outputs in results/step2/:
  lstm_seeds_metrics.csv       RMSE and TSS (at the per-seed validation-tuned alarm level), by seed x test_set x horizon
  lstm_seeds_forecasts.csv     per-hour forecasts, same columns as results/rq1_forecasts.csv, plus a 'seed' column
"""
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import CLEAN, RESULTS, HOLDOUT_2003, STORMS

QUICK = "--quick" in sys.argv
STEP2 = RESULTS / "step2"
STEP2.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 6, 12, 24]
LOOKBACK = 24
FEATURES = ["B", "By", "Bz", "density", "speed", "pressure", "E_field", "dst", "f107", "kp"]
STORM_KP = 5.0
STORM_WEIGHT = 3.0
SEEDS = [1, 2, 3, 4, 5]
RF_SEED = 42   # unchanged from script 06 - RF is not part of this seed-stability test

# ------------------------------------------------------------------ data (verbatim from 06/09, extended with test_2025)
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
has_2025 = sets["test_2025"].sum() > 0
if not has_2025:
    print("WARNING: no 2025 data found - only 'test' (2022-2024) will be evaluated.")
    del sets["test_2025"]
if QUICK:
    idx = np.flatnonzero(sets["train"]); keep = np.zeros(n, bool); keep[idx[::6]] = True
    sets["train"] = keep
for k, m in sets.items():
    print(f"  {k:10s}: {m.sum():,} samples")

mu = np.nanmean(X_all[sets["train"]], axis=0)
sd = np.nanstd(X_all[sets["train"]], axis=0) + 1e-6
Xn_all = (X_all - mu) / sd

def sequences(mask):
    ends = np.flatnonzero(mask)
    win = np.lib.stride_tricks.sliding_window_view(Xn_all, LOOKBACK, axis=0)
    return np.transpose(win[ends - LOOKBACK + 1], (0, 2, 1)).astype(np.float32)

def weights(mask):
    storm = (Y_all[mask] >= STORM_KP).any(axis=1)
    return np.where(storm, STORM_WEIGHT, 1.0).astype(np.float32)

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

def persistence(mask):
    return np.repeat(kp[mask][:, None], len(HORIZONS), axis=1)

print(f"Training Random Forest once (seed={RF_SEED}, shared across all LSTM seeds)...")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=60 if QUICK else 150, max_depth=18, min_samples_leaf=5,
                           max_features=0.5, n_jobs=-1, random_state=RF_SEED)
rf_mask = sets["train"].copy()
if not QUICK:
    idx = np.flatnonzero(rf_mask); rf_mask[:] = False; rf_mask[idx[::3]] = True
rf.fit(F_all[rf_mask], Y_all[rf_mask], sample_weight=weights(rf_mask))
print(f"  done in {time.time()-t0:.0f} s")

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
    out["POD"] = pod
    out["FAR"] = b / (a + b) if a + b else np.nan
    out["TSS"] = pod - pofd
    den = (a + c) * (c + d) + (a + b) * (b + d)
    out["HSS"] = 2 * (a * d - b * c) / den if den else np.nan
    out["n"] = int(len(y)); out["n_storm"] = int(s.sum())
    return out

ALARM_LEVELS = [2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0]

def train_lstm(seed):
    """Verbatim architecture/compile/fit settings from code/06_rq1_forecast.py, only the seed differs."""
    np.random.seed(seed)
    random.seed(seed)
    import tensorflow as tf
    import keras
    from keras import layers
    tf.random.set_seed(seed)
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
    print(f"  seed {seed} done in {time.time()-t0:.0f} s")
    return model

def predict_all(m, model):
    P = {"Persistence": persistence(m),
         "RandomForest": rf.predict(F_all[m]).astype(np.float32),
         "LSTM": model.predict(sequences(m), batch_size=1024, verbose=0)}
    return {k: np.clip(v, 0, 9) for k, v in P.items()}

TEST_SETS = [k for k in ("test", "test_2025") if k in sets]
MET_CSV = STEP2 / "lstm_seeds_metrics.csv"
FC_CSV = STEP2 / "lstm_seeds_forecasts.csv"
DONE_FILE = STEP2 / "_lstm_seeds_done.json"
import json
done = json.loads(DONE_FILE.read_text()) if DONE_FILE.exists() else []
if not done:   # fresh start - clear any stale/partial output (e.g. from a --quick smoke test)
    if MET_CSV.exists(): MET_CSV.unlink()
    if FC_CSV.exists(): FC_CSV.unlink()
remaining_seeds = [s for s in SEEDS if s not in done]
print(f"Seeds already done: {done or 'none'}. Remaining: {remaining_seeds}")

for seed in remaining_seeds:
    print(f"\n=== LSTM seed {seed} ===")
    met_rows, fc_rows = [], []
    model = train_lstm(seed)

    P_val, y_val = predict_all(sets["val"], model), Y_all[sets["val"]]
    best_thr = {}
    for name, p in P_val.items():
        for j, h in enumerate(HORIZONS):
            hss = [scores(y_val[:, j], p[:, j], thr)["HSS"] for thr in ALARM_LEVELS]
            best_thr[(name, h)] = ALARM_LEVELS[int(np.nanargmax(hss))]

    for set_name in TEST_SETS:
        m = sets[set_name]
        y = Y_all[m]
        P = predict_all(m, model)
        for name, p in P.items():
            for j, h in enumerate(HORIZONS):
                base = scores(y[:, j], p[:, j])
                thr = best_thr[(name, h)]
                tuned = scores(y[:, j], p[:, j], thr)
                met_rows.append({"seed": seed, "test_set": set_name, "model": name, "horizon_h": h, **base,
                                  "alarm_level": thr, "POD_tuned": tuned["POD"], "FAR_tuned": tuned["FAR"],
                                  "TSS_tuned": tuned["TSS"], "HSS_tuned": tuned["HSS"]})
        f = pd.DataFrame({"time": t_index[m], "test_set": set_name, "seed": seed})
        for j, h in enumerate(HORIZONS):
            f[f"kp_true_{h}h"] = y[:, j]
            for name, p in P.items():
                f[f"{name}_{h}h"] = p[:, j]
        fc_rows.append(f)
    del model

    pd.DataFrame(met_rows).to_csv(MET_CSV, mode="a", header=not MET_CSV.exists(), index=False)
    pd.concat(fc_rows, ignore_index=True).to_csv(FC_CSV, mode="a", header=not FC_CSV.exists(), index=False)
    done.append(seed)
    DONE_FILE.write_text(json.dumps(done))
    print(f"  seed {seed} saved to disk ({len(done)}/{len(SEEDS)} seeds done)")

met = pd.read_csv(MET_CSV)
fcast = pd.read_csv(FC_CSV)

print("\n=== LSTM RMSE across seeds (test set = 2022-2024) ===")
lstm_rmse = met[(met.model == "LSTM") & (met.test_set == "test")].pivot(index="seed", columns="horizon_h", values="RMSE")
print(lstm_rmse.round(3).to_string())
print("\nstd across seeds, per horizon:")
print(lstm_rmse.std().round(4).to_string())
if "test_2025" in TEST_SETS:
    print("\n=== LSTM RMSE across seeds (test set = 2025) ===")
    lstm_rmse_25 = met[(met.model == "LSTM") & (met.test_set == "test_2025")].pivot(index="seed", columns="horizon_h", values="RMSE")
    print(lstm_rmse_25.round(3).to_string())
    print("\nstd across seeds, per horizon:")
    print(lstm_rmse_25.std().round(4).to_string())
print("\nFiles saved in", STEP2)
print("results/lstm_kp.keras was NOT touched.")
