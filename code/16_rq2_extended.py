"""
Step 14 (journal revision) - Part A (RQ2 extended to 8 storms, leave-one-storm-out ML,
bootstrap CIs) + Part B (NRLMSIS 2.0 density score R_msis).

Does NOT modify or overwrite code/07_rq2_risk.py, config.py, or any existing result file.
Ground-truth and risk-score logic is copied verbatim from code/07_rq2_risk.py, extended to run
over ALL_STORMS (config.STORMS + the 5 new storms from code/15_more_storms.py) instead of just
config.STORMS. Output goes to results/step6/.

Run:  python 16_rq2_extended.py
(resumable: each stage checkpoints to results/step6/_stage*.csv / _density_<storm>.csv so a
 kill mid-run only loses the in-progress stage, not everything before it - just re-run)
Outputs in results/step6/:
  rq2x_objects.csv, rq2x_summary.csv, summary.txt, fig_rq2x_skill.png
Then zips results/step6/ + code/15_more_storms.py + this script into step6_results.zip
(no raw TLE files, no credentials).
"""
import datetime as dt
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pymsis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from config import CLEAN, RESULTS, STORMS, ROOT

STEP6 = RESULTS / "step6"
STEP6.mkdir(parents=True, exist_ok=True)
SEED = 42
BOOT_SEED = 20261026
N_BOOT = 2000
PRE_DAYS, AFTER_DAYS, TOL_DAYS = 6, 3, 0.5
H_SCALE, H_REF = 60.0, 400.0
TOP_FRACTION = 0.10
PASSIVE = {"DEBRIS", "ROCKET BODY"}

NEW_STORMS = {
    "2015_stpatrick": {"start": "2015-03-17", "window": ("2015-03-10", "2015-03-27")},
    "2017_september": {"start": "2017-09-07", "window": ("2017-08-31", "2017-09-17")},
    "2023_april":     {"start": "2023-04-23", "window": ("2023-04-16", "2023-05-03")},
    "2024_october":   {"start": "2024-10-10", "window": ("2024-10-03", "2024-10-20")},
    "2025_november":  {"start": "2025-11-12", "window": ("2025-11-05", "2025-11-22")},
}
IN_SAMPLE_STORMS = {"2015_stpatrick", "2017_september"}
ALL_STORMS = {**STORMS, **NEW_STORMS}

def kp_multiplier(kp):
    return np.select([kp <= 3, kp <= 5, kp <= 7, kp <= 8], [1.0, 1.5, 3.0, 6.0], default=10.0)

# ====================================================================== Kp forecast lookup (extended: test + test_2003 + test_2025, with in-sample flag)
omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
fc_main = pd.read_csv(RESULTS / "rq1_forecasts.csv", parse_dates=["time"])        # test (2022-2024) + test_2003
fc_2025 = pd.read_csv(RESULTS / "rq1_2025_forecasts.csv", parse_dates=["time"])
fc_2025 = fc_2025[fc_2025.test_set == "test_2025"]
fc = pd.concat([fc_main, fc_2025], ignore_index=True).drop_duplicates("time").sort_values("time")

def rf_forecast_before(t, storm_name):
    """Returns (kp_forecast_or_observed, issue_time, out_of_sample: bool)."""
    if storm_name in IN_SAMPLE_STORMS:
        return float(omni.loc[:t, "kp"].iloc[-1]), t, False
    issued = fc[(fc.time <= t) & (fc.time >= t - pd.Timedelta(days=2))]
    if issued.empty:
        return float(omni.loc[:t, "kp"].iloc[-1]), t, False
    last = issued.sort_values("time").iloc[-1]
    return float(max(last[f"RandomForest_{h}h"] for h in (3, 6, 12, 24))), last.time, True

storm_kp = {}
oos_flag = {}
print("=== Storm Kp forecasts (verbatim method from 07_rq2_risk.py) ===")
for name, info in ALL_STORMS.items():
    t0 = pd.Timestamp(info["start"])
    kp_fc, t_issue, oos1 = rf_forecast_before(t0, name)
    win = omni.loc[t0 - pd.Timedelta(hours=12):t0 + pd.Timedelta(days=2), "kp"]
    t_storm = win[win >= 5].index.min() if (win >= 5).any() else t0
    kp_fc_start, t_issue_start, oos2 = rf_forecast_before(t_storm, name)
    kp_obs = float(omni.loc[t0:t0 + pd.Timedelta(days=AFTER_DAYS), "kp"].max())
    storm_kp[name] = (kp_fc, kp_obs, t_issue, kp_fc_start, t_issue_start, t_storm)
    oos_flag[name] = oos1 and oos2
    print(f"{name}: forecast max Kp {kp_fc:.2f} | storm start ({t_storm}) forecast {kp_fc_start:.2f} | "
          f"observed max Kp {kp_obs:.2f} | out-of-sample={oos_flag[name]}")

# ====================================================================== Stage 1: ground truth + Part A risk scores (verbatim from 07_rq2_risk.py, over ALL_STORMS)
STAGE1 = STEP6 / "_stage1_objects.csv"
if STAGE1.exists():
    print(f"\nStage 1 checkpoint found, loading {STAGE1}")
    d = pd.read_csv(STAGE1)
else:
    print("\n=== Stage 1: ground truth + risk scores ===")
    rows = []
    for name, info in ALL_STORMS.items():
        f = CLEAN / f"tle_{name}.csv"
        if not f.exists():
            print("missing", f); continue
        t = pd.read_csv(f, low_memory=False, parse_dates=["EPOCH"])
        t0 = pd.Timestamp(info["start"])
        t["day"] = (t["EPOCH"] - t0).dt.total_seconds() / 86400.0
        t = t[(t["day"] >= -PRE_DAYS) & (t["day"] <= AFTER_DAYS + TOL_DAYS)]
        if "INCLINATION" not in t.columns:
            t["INCLINATION"] = np.nan
        for sat, g in t.groupby("NORAD_CAT_ID"):
            g = g.sort_values("day")
            pre = g[g["day"] < 0]
            post = g[(g["day"] >= AFTER_DAYS - TOL_DAYS) & (g["day"] <= AFTER_DAYS + TOL_DAYS)]
            if len(pre) < 4 or pre["day"].max() - pre["day"].min() < 3 or len(post) < 1:
                continue
            if (g["ALT_KM"].diff() > 0.5).any():
                continue
            slope, intercept = np.polyfit(pre["day"], pre["ALT_KM"], 1)
            expected_lin = intercept + slope * AFTER_DAYS
            arg = 1 + slope * AFTER_DAYS / H_SCALE
            expected = intercept + H_SCALE * np.log(arg) if (slope < 0 and arg > 0.05) else expected_lin
            observed = post["ALT_KM"].median()
            h0 = intercept
            if not (200 <= h0 <= 600):
                continue
            rows.append({
                "storm": name, "norad_id": sat, "name": g["OBJECT_NAME"].iloc[-1],
                "object_type": str(g["OBJECT_TYPE"].iloc[-1]).upper(),
                "alt_km": h0, "pre_rate_km_day": slope,
                "bstar": float(pre["BSTAR"].astype(float).median()),
                "inclination": float(pre["INCLINATION"].astype(float).median()),
                "excess_decay_km": expected - observed,
                "excess_decay_linear_km": expected_lin - observed,
            })
    d = pd.DataFrame(rows)
    d["passive"] = d["object_type"].isin(PASSIVE)
    d["kp_forecast"] = d["storm"].map(lambda s: storm_kp[s][0])
    d["kp_observed"] = d["storm"].map(lambda s: storm_kp[s][1])
    d["kp_forecast_start"] = d["storm"].map(lambda s: storm_kp[s][3])
    d["kp_forecast_available"] = d["storm"].map(lambda s: oos_flag[s])

    b = d["bstar"].clip(lower=1e-6)
    d["score_altitude"] = -d["alt_km"]
    d["score_decay_rate"] = -d["pre_rate_km_day"]
    d["score_physics"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_forecast"])
    d["score_physics_obsKp"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_observed"])
    d["score_physics_startKp"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_forecast_start"])
    d.to_csv(STAGE1, index=False)
    print(f"Stage 1: {len(d):,} objects across {d.storm.nunique()} storms. Saved {STAGE1}")

# ====================================================================== Stage 2: ML leave-one-storm-out (verbatim from 07_rq2_risk.py, 8 storms)
STAGE2 = STEP6 / "_stage2_objects.csv"
if STAGE2.exists():
    print(f"\nStage 2 checkpoint found, loading {STAGE2}")
    d = pd.read_csv(STAGE2)
else:
    print("\n=== Stage 2: leave-one-storm-out ML score ===")
    b = d["bstar"].clip(lower=1e-6)
    def ml_scores(kp_col):
        out = pd.Series(np.nan, index=d.index)
        feats = ["alt_km", "log_bstar", "inclination", kp_col]
        X = d.assign(log_bstar=np.log10(b))[feats].fillna(d.assign(log_bstar=np.log10(b))[feats].median())
        y = d["excess_decay_km"].clip(-5, 50)
        for test_storm in d["storm"].unique():
            tr = (d["storm"] != test_storm) & d["passive"]
            te = d["storm"] == test_storm
            if tr.sum() < 50:
                continue
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=SEED, loss="huber")
            m.fit(X[tr], y[tr])
            out[te] = m.predict(X[te])
        return out
    d["score_ml"] = ml_scores("kp_forecast")
    d["score_ml_obsKp"] = ml_scores("kp_observed")
    d.to_csv(STAGE2, index=False)
    print(f"Stage 2 done. Saved {STAGE2}")

# ====================================================================== Stage 3 (Part B): NRLMSIS 2.0 density -> R_msis
def kp_to_ap(kp):
    """Standard Kp (thirds notation) -> ap lookup table, linearly interpolated."""
    kp_tab = np.array([0,0.33,0.67,1,1.33,1.67,2,2.33,2.67,3,3.33,3.67,4,4.33,4.67,
                        5,5.33,5.67,6,6.33,6.67,7,7.33,7.67,8,8.33,8.67,9])
    ap_tab = np.array([0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,
                        48,56,67,80,94,111,132,154,179,207,236,300,400])
    return np.interp(kp, kp_tab, ap_tab)

daily_kp_max = omni["kp"].resample("D").max()
daily_f107 = omni["f107"].resample("D").mean()
daily_f107a = daily_f107.rolling(81, min_periods=40).mean()   # trailing 81-day mean

LAT_GRID = np.arange(-80, 81, 10)
LT_GRID = np.arange(0, 24, 3)
GLAT, GLT = np.meshgrid(LAT_GRID, LT_GRID)
GLAT, GLT = GLAT.ravel(), GLT.ravel()
GLON = GLT * 15.0   # UT fixed at 00:00, so local time (h) * 15 deg/h = longitude
N_GRID = len(GLAT)

def msis_mean_density(date, alts, f107, f107a, ap):
    """Vectorized: alts is per-object (n,), returns mean density (n,) over the lat/LT grid."""
    n = len(alts)
    dates = np.array([date] * (n * N_GRID))
    lons = np.tile(GLON, n)
    lats = np.tile(GLAT, n)
    alts_rep = np.repeat(alts, N_GRID)
    f107s = np.full(n * N_GRID, f107)
    f107as = np.full(n * N_GRID, f107a)
    aps = np.full((n * N_GRID, 1), ap)
    out = pymsis.calculate(dates, lons, lats, alts_rep, f107s, f107as, aps)
    rho = out[:, pymsis.Variable.MASS_DENSITY].reshape(n, N_GRID)
    return rho.mean(axis=1)

print("\n=== Stage 3: NRLMSIS 2.0 density per storm (quiet / storm-max-Ap / storm-RFKp) ===")
ratio_rows = []
density_frames = []
t_start = time.time()
for name, info in ALL_STORMS.items():
    dpath = STEP6 / f"_density_{name}.csv"
    if dpath.exists():
        print(f"{name}: density checkpoint found, loading")
        density_frames.append(pd.read_csv(dpath))
        continue
    sub = d[d.storm == name]
    if sub.empty:
        continue
    onset = pd.Timestamp(ALL_STORMS[name]["start"]).normalize()
    alts = sub["alt_km"].to_numpy()
    norad = sub["norad_id"].to_numpy()

    quiet_day = onset - pd.Timedelta(days=1)
    f107_q = float(daily_f107.get(quiet_day, daily_f107.loc[:quiet_day].iloc[-1]))
    f107a_q = float(daily_f107a.get(quiet_day, daily_f107a.loc[:quiet_day].dropna().iloc[-1]))
    ap_q = float(kp_to_ap(daily_kp_max.get(quiet_day, 2.0)))
    rho_quiet = msis_mean_density(dt.datetime(quiet_day.year, quiet_day.month, quiet_day.day), alts, f107_q, f107a_q, ap_q)

    win3 = daily_kp_max.loc[onset: onset + pd.Timedelta(days=AFTER_DAYS)]
    peak_day = win3.idxmax() if len(win3) else onset
    f107_s = float(daily_f107.get(peak_day, daily_f107.loc[:peak_day].iloc[-1]))
    f107a_s = float(daily_f107a.get(peak_day, daily_f107a.loc[:peak_day].dropna().iloc[-1]))
    ap_s = float(kp_to_ap(win3.max() if len(win3) else 5.0))
    rho_storm = msis_mean_density(dt.datetime(peak_day.year, peak_day.month, peak_day.day), alts, f107_s, f107a_s, ap_s)

    if oos_flag[name]:
        kp_rf = storm_kp[name][0]
        ap_rf = float(kp_to_ap(kp_rf))
        rho_rfkp = msis_mean_density(dt.datetime(onset.year, onset.month, onset.day), alts, f107_s, f107a_s, ap_rf)
    else:
        rho_rfkp = np.full(len(alts), np.nan)

    out = pd.DataFrame({"storm": name, "norad_id": norad, "rho_quiet": rho_quiet,
                        "rho_storm_obsAp": rho_storm, "rho_storm_rfKp": rho_rfkp})
    out.to_csv(dpath, index=False)
    density_frames.append(out)
    print(f"{name}: {len(sub):,} objects, quiet-day Ap={ap_q:.1f} F10.7={f107_q:.0f} | "
          f"storm peak-day Ap={ap_s:.1f} F10.7={f107_s:.0f} ({peak_day.date()}) | "
          f"[{time.time()-t_start:.0f}s elapsed]")

density = pd.concat(density_frames, ignore_index=True)
d = d.merge(density, on=["storm", "norad_id"], how="left")
b = d["bstar"].clip(lower=1e-6)
d["score_physics_msis"] = b * d["rho_storm_obsAp"]
d["score_physics_msis_rfKp"] = b * d["rho_storm_rfKp"]
d["density_storm_quiet_ratio"] = d["rho_storm_obsAp"] / d["rho_quiet"]

OBJ_OUT = STEP6 / "rq2x_objects.csv"
d.to_csv(OBJ_OUT, index=False)
print(f"\nSaved {OBJ_OUT} ({len(d):,} rows)")

# ====================================================================== storm/quiet density ratio by altitude band
print("\n=== Density ratio (storm obsAp / quiet) by storm and altitude band ===")
bins = [200, 300, 350, 400, 450, 500, 600]
d["alt_band"] = pd.cut(d["alt_km"], bins)
ratio_tab = d.groupby(["storm", "alt_band"], observed=True)["density_storm_quiet_ratio"].agg(["median", "count"])
print(ratio_tab.to_string())

# ====================================================================== Stage 4: evaluation (rho / AUC / bootstrap), Part A + Part B combined
METHODS = {"Altitude only": "score_altitude", "Pre-storm decay": "score_decay_rate",
           "Physics score": "score_physics", "Physics (NRLMSIS)": "score_physics_msis",
           "ML (forecast Kp)": "score_ml", "ML (observed Kp)": "score_ml_obsKp"}

rng_global = np.random.default_rng(BOOT_SEED)

def rho_auc(sub, col):
    thr = sub["excess_decay_km"].quantile(1 - TOP_FRACTION)
    label = (sub["excess_decay_km"] >= thr).astype(int)
    s = sub[col]
    ok = s.notna()
    if ok.sum() < 20 or label[ok].nunique() < 2:
        return np.nan, np.nan
    rho = spearmanr(s[ok], sub.loc[ok, "excess_decay_km"]).correlation
    auc = roc_auc_score(label[ok], s[ok])
    return rho, auc

def bootstrap_ci(sub, col, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(sub)
    rhos, aucs = [], []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        samp = sub.iloc[rng.choice(idx_all, n, replace=True)]
        r, a = rho_auc(samp, col)
        rhos.append(r); aucs.append(a)
    rhos, aucs = np.array(rhos, dtype=float), np.array(aucs, dtype=float)
    def ci(x):
        x = x[~np.isnan(x)]
        return (np.nan, np.nan) if len(x) < 100 else (np.percentile(x, 2.5), np.percentile(x, 97.5))
    return ci(rhos), ci(aucs)

print("\n=== Stage 4: within-storm rho/AUC + bootstrap CIs ===")
rows = []
passive_counts = d[d.passive].groupby("storm").size()
for group_name, gmask in (("PASSIVE", d["passive"]), ("PAYLOADS", ~d["passive"])):
    for storm in ALL_STORMS:
        sub = d[gmask & (d.storm == storm)]
        if len(sub) < 20:
            continue
        for mname, col in METHODS.items():
            rho, auc = rho_auc(sub, col)
            (rho_lo, rho_hi), (auc_lo, auc_hi) = bootstrap_ci(sub, col, seed=BOOT_SEED + hash((storm, mname)) % 10000)
            rows.append({"storm": storm, "group": group_name, "method": mname, "n": len(sub),
                        "rho": rho, "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi,
                        "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})

# pooled two ways, PASSIVE and PAYLOADS
for group_name, gmask in (("PASSIVE", d["passive"]), ("PAYLOADS", ~d["passive"])):
    pas = d[gmask]
    for mname, col in METHODS.items():
        rho, auc = rho_auc(pas, col)
        (rho_lo, rho_hi), (auc_lo, auc_hi) = bootstrap_ci(pas, col, seed=BOOT_SEED + hash(("ALL", mname)) % 10000)
        rows.append({"storm": "POOLED_ALL_OBJECTS", "group": group_name, "method": mname, "n": len(pas),
                    "rho": rho, "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi,
                    "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})

    # mean-of-within-storm-rho (equal weight per storm), stratified bootstrap (resample objects WITHIN each storm)
    storms_here = [s for s in ALL_STORMS if (gmask & (d.storm == s)).sum() >= 20]
    for mname, col in METHODS.items():
        per_storm_rho = [rho_auc(d[gmask & (d.storm == s)], col)[0] for s in storms_here]
        per_storm_rho = [r for r in per_storm_rho if not np.isnan(r)]
        mean_rho = float(np.mean(per_storm_rho)) if per_storm_rho else np.nan
        rng = np.random.default_rng(BOOT_SEED + hash(("MEAN", mname)) % 10000)
        boot_means = []
        storm_subs = {s: d[gmask & (d.storm == s)] for s in storms_here}
        for _ in range(N_BOOT):
            draw = []
            for s, ssub in storm_subs.items():
                n = len(ssub)
                samp = ssub.iloc[rng.choice(n, n, replace=True)]
                r, _ = rho_auc(samp, col)
                if not np.isnan(r):
                    draw.append(r)
            if draw:
                boot_means.append(np.mean(draw))
        boot_means = np.array(boot_means)
        lo, hi = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)) if len(boot_means) >= 100 else (np.nan, np.nan)
        rows.append({"storm": "MEAN_OF_WITHIN_STORM_RHO", "group": group_name, "method": mname,
                    "n": len(storms_here), "rho": mean_rho, "rho_ci_lo": lo, "rho_ci_hi": hi,
                    "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan})

summary_df = pd.DataFrame(rows)
summary_df.to_csv(STEP6 / "rq2x_summary.csv", index=False)

# ====================================================================== figure
storms_order = list(ALL_STORMS.keys())
fig, ax = plt.subplots(figsize=(14, 5))
w = 0.13
x = np.arange(len(storms_order))
for i, (mname, col) in enumerate(METHODS.items()):
    vals = [summary_df[(summary_df.storm == s) & (summary_df.group == "PASSIVE") & (summary_df.method == mname)]["rho"]
            .pipe(lambda s: s.iloc[0] if len(s) else np.nan) for s in storms_order]
    ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, vals, w, label=mname)
ax.set_xticks(x); ax.set_xticklabels(storms_order, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Spearman rho (passive objects)"); ax.set_title("RQ2 extended: risk-score skill per storm (8 storms)")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
fig.tight_layout(); fig.savefig(STEP6 / "fig_rq2x_skill.png", dpi=200); plt.close(fig)

# ====================================================================== summary.txt
lines = ["AETHERA - RQ2 EXTENDED (8 storms) + NRLMSIS 2.0 density score", ""]
lines.append("Kp-forecast availability per storm:")
for name in ALL_STORMS:
    tag = "out-of-sample (RF forecast used)" if oos_flag[name] else "NOT AVAILABLE (in RF training period - observed Kp used instead)"
    lines.append(f"  {name}: {tag}")
lines.append("")
lines.append("Passive objects per storm:")
for name in ALL_STORMS:
    n = int(passive_counts.get(name, 0))
    lines.append(f"  {name}: {n:,}")
lines.append("")
lines.append(f"=== Pooled Spearman rho, PASSIVE objects (two ways; bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}) ===")
for mname in METHODS:
    r1 = summary_df[(summary_df.storm == "POOLED_ALL_OBJECTS") & (summary_df.group == "PASSIVE") & (summary_df.method == mname)].iloc[0]
    r2 = summary_df[(summary_df.storm == "MEAN_OF_WITHIN_STORM_RHO") & (summary_df.group == "PASSIVE") & (summary_df.method == mname)].iloc[0]
    lines.append(f"  {mname:20s} all-objects-pooled rho={r1.rho:.3f} [{r1.rho_ci_lo:.3f},{r1.rho_ci_hi:.3f}]  |  "
                f"mean-of-within-storm rho={r2.rho:.3f} [{r2.rho_ci_lo:.3f},{r2.rho_ci_hi:.3f}] (n_storms={int(r2.n)})")
lines.append("")
lines.append("=== Within-storm rho / AUC (top 10%), PASSIVE objects ===")
for name in ALL_STORMS:
    sub = summary_df[(summary_df.storm == name) & (summary_df.group == "PASSIVE")]
    if sub.empty:
        continue
    lines.append(f"  --- {name} (n={int(sub.n.iloc[0])}) ---")
    for _, r in sub.iterrows():
        lines.append(f"    {r.method:20s} rho={r.rho:6.3f} [{r.rho_ci_lo:6.3f},{r.rho_ci_hi:6.3f}]  "
                     f"AUC={r.auc:5.3f} [{r.auc_ci_lo:5.3f},{r.auc_ci_hi:5.3f}]")
lines.append("")
lines.append("=== NRLMSIS 2.0 method notes ===")
lines.append("Grid: latitudes -80..80 step 10 deg (17), local times 0..21h step 3h (8) -> 136 points/object,")
lines.append("averaged to one mean density per object per scenario. Local time implemented as longitude at a")
lines.append("fixed UT=00:00 reference (lon = LT*15 deg). F10.7a = trailing 81-day mean (not centered - centered")
lines.append("is not available for forecast-based scenarios, so a uniform trailing definition is used throughout).")
lines.append("Ap converted from Kp via the standard Kp->ap lookup table. geomagnetic_activity left at MSIS default")
lines.append("(daily-Ap-based storm-time coupling), not the full 7-component historical ap array.")
lines.append("R_msis = B* x rho_storm(h), using the OBSERVED max-Ap-in-first-3-days scenario (available for all 8")
lines.append("storms). The RF-forecast-Kp density variant (score_physics_msis_rfKp) is saved in rq2x_objects.csv")
lines.append("for the 6 out-of-sample storms only, as a secondary comparison, but is not the primary R_msis.")
lines.append("")
lines.append("=== Storm/quiet density ratio by altitude band (median, obsAp storm-day vs quiet-day-before) ===")
lines.append(ratio_tab.to_string())
summary = "\n".join(lines)
(STEP6 / "summary.txt").write_text(summary)
print("\n" + summary[:3000] + "\n... (full summary.txt saved)")

# ====================================================================== cleanup internal checkpoints before zipping + secret scan + zip
for p in list(STEP6.glob("_stage*.csv")) + list(STEP6.glob("_density_*.csv")):
    p.unlink()

leak = []
for p in list(STEP6.rglob("*")) + [ROOT / "code" / "15_more_storms.py", ROOT / "code" / "16_rq2_extended.py"]:
    if p.is_file() and p.suffix in (".csv", ".txt", ".py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(api[_-]?key|password|secret|token)\s*[:=]\s*['\"]", text, re.I) or \
           re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            leak.append(str(p))
print("\nSecret/email scan:", "CLEAN - nothing found" if not leak else f"FOUND IN: {leak}")

out_zip = ROOT / "step6_results.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in STEP6.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(ROOT))
    z.write(ROOT / "code" / "15_more_storms.py", "code/15_more_storms.py")
    z.write(Path(__file__), f"code/{Path(__file__).name}")
print(f"\n{out_zip}: {out_zip.stat().st_size:,} bytes ({out_zip.stat().st_size/1e6:.2f} MB)")
