"""
Step 14b (journal revision) - RQ2 baseline-contamination and forecast-NaN diagnostics.

Does NOT modify or overwrite results/step6/ or any existing result file - loads
results/step6/rq2x_objects.csv read-only and works on an in-memory corrected copy.
All new output goes to results/step6b/.

FIX (see finding #4 in summary.txt): in code/16_rq2_extended.py, rf_forecast_before() had a
branch "if storm_name in IN_SAMPLE_STORMS: return <last observed Kp>, t, False" - this silently
used the last OBSERVED Kp as a stand-in "forecast" for the 2015 and 2017 storms (which predate
the RF's training cutoff, so no real out-of-sample forecast exists for them). That fallback value
then flowed into kp_forecast, kp_forecast_start, score_physics, score_physics_startKp, and the
"ML (forecast Kp)" score - none of which were ever actually NaN for those two storms, contrary to
what "forecast not available" should mean. This script patches those columns to NaN for 2015/2017
before recomputing anything, and reports what the old (wrong) values looked like.

Run:  python 16b_rq2_baseline_check.py
Outputs in results/step6b/: rq2x_baseline_check.csv, summary.txt
Then zips results/step6b/ + this script into step6b_results.zip.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from config import CLEAN, RESULTS, ROOT

STEP6 = RESULTS / "step6"
STEP6B = RESULTS / "step6b"
STEP6B.mkdir(parents=True, exist_ok=True)
SEED_BOOT = 20261026
N_BOOT = 2000
TOP_FRACTION = 0.10
PRE_DAYS, AFTER_DAYS, TOL_DAYS = 6, 3, 0.5
H_SCALE, H_REF = 60.0, 400.0
IN_SAMPLE_STORMS = {"2015_stpatrick", "2017_september"}
CONTAM_KP = 6.0   # G2

ALL_STORMS_WINDOWS = {   # start date only (needed for baseline/post windows) - copied from configs,
    "2003_halloween": "2003-10-29", "2022_starlink": "2022-02-03", "2024_gannon": "2024-05-10",
    "2015_stpatrick": "2015-03-17", "2017_september": "2017-09-07", "2023_april": "2023-04-23",
    "2024_october": "2024-10-10", "2025_november": "2025-11-12",
}

def g_level(kp):
    return int(min(5, max(0, np.floor(kp + 1e-6) - 4)))

omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)

# ====================================================================== 1 + 2: baseline/post window Kp, contamination rule
print("=== 1-2: pre-onset baseline and post-onset window Kp, contamination check ===")
win_rows = []
for storm, start in ALL_STORMS_WINDOWS.items():
    t0 = pd.Timestamp(start)
    pre = omni.loc[t0 - pd.Timedelta(days=PRE_DAYS): t0 - pd.Timedelta(hours=1), "kp"]
    post = omni.loc[t0: t0 + pd.Timedelta(days=AFTER_DAYS), "kp"]
    pre_max, post_max = float(pre.max()), float(post.max())
    contaminated = pre_max >= CONTAM_KP
    win_rows.append({"storm": storm, "pre_max_kp": pre_max, "pre_max_G": f"G{g_level(pre_max)}",
                     "post_max_kp": post_max, "post_max_G": f"G{g_level(post_max)}",
                     "baseline_contaminated": contaminated})
    print(f"  {storm}: pre-onset max Kp={pre_max:.2f} (G{g_level(pre_max)})  "
          f"post-onset max Kp={post_max:.2f} (G{g_level(post_max)})  "
          f"{'** CONTAMINATED (Kp>=6 in baseline) **' if contaminated else 'clean baseline'}")
win_df = pd.DataFrame(win_rows)
CLEAN_STORMS = win_df[~win_df.baseline_contaminated].storm.tolist()
CONTAM_STORMS = win_df[win_df.baseline_contaminated].storm.tolist()
print(f"\nContaminated storms: {CONTAM_STORMS or 'none'}")
print(f"Clean-baseline storms: {CLEAN_STORMS}")

# ====================================================================== 4: load + report + fix the forecast-Kp NaN bug
print("\n=== 4: forecast-Kp fix for 2015/2017 ===")
d = pd.read_csv(STEP6 / "rq2x_objects.csv")
before = {}
for s in IN_SAMPLE_STORMS:
    sub = d[d.storm == s]
    before[s] = {
        "kp_forecast (old, wrongly filled)": sub["kp_forecast"].median(),
        "kp_observed (for comparison)": sub["kp_observed"].median(),
        "score_physics (old)": sub["score_physics"].median(),
        "score_ml (ML forecast-Kp, old)": sub["score_ml"].median(),
    }
    print(f"  {s}: " + ", ".join(f"{k}={v:.3f}" for k, v in before[s].items()))
print("  Reason: rf_forecast_before() in code/16_rq2_extended.py fell back to the LAST OBSERVED")
print("  Kp (not NaN) for any storm in IN_SAMPLE_STORMS, because that fallback branch was originally")
print("  written as a generic 'no nearby forecast row found' fallback and was reused unconditionally")
print("  for in-sample storms too, rather than being skipped in favour of a real NaN.")

FORECAST_COLS = ["kp_forecast", "kp_forecast_start", "score_physics", "score_physics_startKp", "score_ml"]
for c in FORECAST_COLS:
    d.loc[d.storm.isin(IN_SAMPLE_STORMS), c] = np.nan
print(f"  FIXED: {FORECAST_COLS} are now NaN for {sorted(IN_SAMPLE_STORMS)}.")
print("  Note: this does not retrain the leave-one-storm-out ML model excluding 2015/2017's old")
print("  (fallback-based) kp_forecast values from OTHER storms' training folds - that is a separate,")
print("  smaller residual contamination of the training data this script does not address.")

# ====================================================================== 5: per-storm diagnostics
print("\n=== 5: per-storm diagnostics ===")
diag_rows = []
for storm in ALL_STORMS_WINDOWS:
    sub = d[(d.storm == storm) & d.passive]
    if sub.empty:
        continue
    ok = sub["excess_decay_km"].notna()
    rho_lin, _ = (np.nan, np.nan)
    s = sub["score_physics_obsKp"]   # observed-Kp physics score is always available, used for the linear-vs-exp sensitivity check
    m = s.notna() & sub["excess_decay_linear_km"].notna()
    if m.sum() >= 20:
        rho_lin = spearmanr(s[m], sub.loc[m, "excess_decay_linear_km"]).correlation
    diag_rows.append({
        "storm": storm, "n_passive": len(sub),
        "median_excess_decay_km": float(sub["excess_decay_km"].median()),
        "frac_negative_excess_decay": float((sub["excess_decay_km"] < 0).mean()),
        "median_pre_storm_decay_rate_km_day": float(sub["pre_rate_km_day"].median()),
        "rho_physics_obsKp_vs_excess_decay_linear": rho_lin,
    })
diag_df = pd.DataFrame(diag_rows)
print(diag_df.round(4).to_string(index=False))

# ====================================================================== 6: April 2023 robustness, post-storm measured at +4 and +5 days
print("\n=== 6: April 2023 robustness (post-storm altitude at +4 and +5 days instead of +3) ===")
def rebuild_april_ground_truth(after_days):
    f = CLEAN / "tle_2023_april.csv"
    t = pd.read_csv(f, low_memory=False, parse_dates=["EPOCH"])
    t0 = pd.Timestamp(ALL_STORMS_WINDOWS["2023_april"])
    t["day"] = (t["EPOCH"] - t0).dt.total_seconds() / 86400.0
    t = t[(t["day"] >= -PRE_DAYS) & (t["day"] <= after_days + TOL_DAYS)]
    rows = []
    for sat, g in t.groupby("NORAD_CAT_ID"):
        g = g.sort_values("day")
        pre = g[g["day"] < 0]
        post = g[(g["day"] >= after_days - TOL_DAYS) & (g["day"] <= after_days + TOL_DAYS)]
        if len(pre) < 4 or pre["day"].max() - pre["day"].min() < 3 or len(post) < 1:
            continue
        if (g["ALT_KM"].diff() > 0.5).any():
            continue
        slope, intercept = np.polyfit(pre["day"], pre["ALT_KM"], 1)
        arg = 1 + slope * after_days / H_SCALE
        expected = intercept + H_SCALE * np.log(arg) if (slope < 0 and arg > 0.05) else intercept + slope * after_days
        observed = post["ALT_KM"].median()
        h0 = intercept
        if not (200 <= h0 <= 600):
            continue
        rows.append({"norad_id": sat, "object_type": str(g["OBJECT_TYPE"].iloc[-1]).upper(),
                     "excess_decay_km": expected - observed})
    gt = pd.DataFrame(rows)
    gt["passive"] = gt["object_type"].isin({"DEBRIS", "ROCKET BODY"})
    merged = gt[gt.passive].merge(d[(d.storm == "2023_april")][["norad_id", "score_physics"]], on="norad_id", how="inner")
    ok = merged["score_physics"].notna() & merged["excess_decay_km"].notna()
    rho = spearmanr(merged.loc[ok, "score_physics"], merged.loc[ok, "excess_decay_km"]).correlation if ok.sum() >= 20 else np.nan
    return rho, int(ok.sum())

april_robust = {}
for ad in (3, 4, 5):
    rho, n = rebuild_april_ground_truth(ad)
    april_robust[ad] = (rho, n)
    print(f"  +{ad} days: rho(physics score, excess_decay) = {rho:.3f} (n={n})" if not np.isnan(rho)
          else f"  +{ad} days: insufficient data")

# ====================================================================== 3: recompute rho/AUC + bootstrap, passive only, corrected table
METHODS = {"Altitude only": "score_altitude", "Pre-storm decay": "score_decay_rate",
           "Physics score": "score_physics", "Physics (NRLMSIS)": "score_physics_msis",
           "ML (forecast Kp)": "score_ml", "ML (observed Kp)": "score_ml_obsKp"}

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
    for _ in range(n_boot):
        samp = sub.iloc[rng.integers(0, n, n)]
        r, a = rho_auc(samp, col)
        rhos.append(r); aucs.append(a)
    rhos, aucs = np.array(rhos, float), np.array(aucs, float)
    def ci(x):
        x = x[~np.isnan(x)]
        return (np.nan, np.nan) if len(x) < 100 else (np.percentile(x, 2.5), np.percentile(x, 97.5))
    return ci(rhos), ci(aucs)

CKPT = STEP6B / "_within_storm.csv"
if CKPT.exists():
    print(f"\nWithin-storm checkpoint found, loading {CKPT}")
    within = pd.read_csv(CKPT)
else:
    print("\n=== 3a: within-storm rho/AUC + bootstrap (passive objects, corrected table) ===")
    rows = []
    for storm in ALL_STORMS_WINDOWS:
        sub = d[(d.storm == storm) & d.passive]
        if len(sub) < 20:
            continue
        for mname, col in METHODS.items():
            rho, auc = rho_auc(sub, col)
            (rho_lo, rho_hi), (auc_lo, auc_hi) = bootstrap_ci(sub, col, seed=SEED_BOOT + hash((storm, mname)) % 10000)
            rows.append({"storm": storm, "method": mname, "n": len(sub), "rho": rho,
                        "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi, "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
        print(f"  {storm} done")
    within = pd.DataFrame(rows)
    within.to_csv(CKPT, index=False)

def pooled_both_ways(storm_list, label):
    out = []
    pas = d[d.passive & d.storm.isin(storm_list)]
    for mname, col in METHODS.items():
        rho, auc = rho_auc(pas, col)
        (rho_lo, rho_hi), (auc_lo, auc_hi) = bootstrap_ci(pas, col, seed=SEED_BOOT + hash((label, "ALL", mname)) % 10000)
        out.append({"scope": label, "pooling": "all_objects_pooled", "method": mname, "n": len(pas),
                    "rho": rho, "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi, "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
        per_storm = within[(within.storm.isin(storm_list)) & (within.method == mname)]
        rhos = per_storm["rho"].dropna().to_numpy()
        mean_rho = float(rhos.mean()) if len(rhos) else np.nan
        rng = np.random.default_rng(SEED_BOOT + hash((label, "MEAN", mname)) % 10000)
        storm_subs = {s: d[d.passive & (d.storm == s)] for s in storm_list if (d.passive & (d.storm == s)).sum() >= 20}
        boot_means = []
        for _ in range(N_BOOT):
            draw = []
            for s, ssub in storm_subs.items():
                n = len(ssub)
                samp = ssub.iloc[rng.integers(0, n, n)]
                r, _ = rho_auc(samp, col)
                if not np.isnan(r):
                    draw.append(r)
            if draw:
                boot_means.append(np.mean(draw))
        boot_means = np.array(boot_means)
        lo, hi = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)) if len(boot_means) >= 100 else (np.nan, np.nan)
        out.append({"scope": label, "pooling": "mean_of_within_storm_rho", "method": mname, "n": len(storm_subs),
                    "rho": mean_rho, "rho_ci_lo": lo, "rho_ci_hi": hi, "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan})
    return out

print("\n=== 3b: pooled, both ways, (a) all 8 storms and (b) clean-baseline storms only ===")
pooled_rows = pooled_both_ways(list(ALL_STORMS_WINDOWS.keys()), "ALL_8_STORMS")
pooled_rows += pooled_both_ways(CLEAN_STORMS, "CLEAN_BASELINE_ONLY")
pooled = pd.DataFrame(pooled_rows)

# ====================================================================== save
within["scope"] = "per_storm"
full = pd.concat([within, pooled], ignore_index=True)
full.to_csv(STEP6B / "rq2x_baseline_check.csv", index=False)
win_df.to_csv(STEP6B / "baseline_windows.csv", index=False)
diag_df.to_csv(STEP6B / "storm_diagnostics.csv", index=False)

lines = ["AETHERA - RQ2 baseline-contamination check + forecast-NaN fix", ""]
lines.append("=== 1-2: pre/post window Kp and contamination (Kp>=6 / G2+ in the 6-day pre-onset baseline) ===")
lines.append(win_df.round(2).to_string(index=False))
lines.append(f"\nContaminated: {CONTAM_STORMS or 'none'}  |  Clean-baseline: {CLEAN_STORMS}")
lines.append("")
lines.append("=== 4: forecast-Kp NaN fix ===")
for s in IN_SAMPLE_STORMS:
    lines.append(f"  {s} - values used BEFORE the fix (wrong): " + ", ".join(f"{k}={v:.3f}" for k, v in before[s].items()))
lines.append(f"  FIXED: {FORECAST_COLS} are now NaN for {sorted(IN_SAMPLE_STORMS)} in this script's corrected table.")
lines.append("  Cause: rf_forecast_before()'s in-sample branch returned the last OBSERVED Kp (not NaN) as a")
lines.append("  stand-in forecast, so 'ML (forecast Kp)' and the forecast-based physics score were silently")
lines.append("  computed on a fake forecast for 2015/2017, instead of being skipped. This is why 2015's")
lines.append("  'ML (forecast Kp)' rho was a nonsensical -0.157 in the original results/step6 report.")
lines.append("  Residual caveat: other storms' leave-one-storm-out ML training folds still include 2015/2017's")
lines.append("  old (fallback-based) kp_forecast as a training feature value - not re-fit here.")
lines.append("")
lines.append("=== 5: per-storm diagnostics ===")
lines.append(diag_df.round(4).to_string(index=False))
lines.append("")
lines.append("=== 6: April 2023 robustness (physics score rho vs excess_decay, post-storm altitude at +N days) ===")
for ad, (rho, n) in april_robust.items():
    lines.append(f"  +{ad} days: rho={rho:.3f} (n={n})" + ("  <- main result" if ad == 3 else ""))
lines.append("")
lines.append("=== 3: recomputed rho/AUC, PASSIVE objects, corrected table (within-storm; bootstrap 95% CI, "
             f"{N_BOOT} resamples, seed {SEED_BOOT}) ===")
for storm in ALL_STORMS_WINDOWS:
    sub = within[within.storm == storm]
    if sub.empty:
        continue
    lines.append(f"  --- {storm} (n={int(sub.n.iloc[0])}) ---")
    for _, r in sub.iterrows():
        lines.append(f"    {r.method:20s} rho={r.rho:6.3f} [{r.rho_ci_lo:6.3f},{r.rho_ci_hi:6.3f}]  "
                     f"AUC={r.auc:5.3f} [{r.auc_ci_lo:5.3f},{r.auc_ci_hi:5.3f}]")
lines.append("")
lines.append("=== Pooled rho, both ways, (a) all 8 storms vs (b) clean-baseline storms only ===")
for scope in ("ALL_8_STORMS", "CLEAN_BASELINE_ONLY"):
    lines.append(f"  --- {scope} ---")
    for mname in METHODS:
        r1 = pooled[(pooled.scope == scope) & (pooled.pooling == "all_objects_pooled") & (pooled.method == mname)].iloc[0]
        r2 = pooled[(pooled.scope == scope) & (pooled.pooling == "mean_of_within_storm_rho") & (pooled.method == mname)].iloc[0]
        lines.append(f"    {mname:20s} all-pooled rho={r1.rho:.3f} [{r1.rho_ci_lo:.3f},{r1.rho_ci_hi:.3f}]  |  "
                     f"mean-of-storms rho={r2.rho:.3f} [{r2.rho_ci_lo:.3f},{r2.rho_ci_hi:.3f}] (n_storms={int(r2.n)})")
summary = "\n".join(lines)
(STEP6B / "summary.txt").write_text(summary)
print("\n" + summary)

CKPT.unlink()

# ====================================================================== zip
out_zip = ROOT / "step6b_results.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in STEP6B.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(ROOT))
    z.write(Path(__file__), f"code/{Path(__file__).name}")
print(f"\n{out_zip}: {out_zip.stat().st_size:,} bytes ({out_zip.stat().st_size/1e6:.2f} MB)")
