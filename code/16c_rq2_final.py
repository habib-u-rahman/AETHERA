"""
Step 14c (journal revision) - RQ2 final fix-up: within-storm scores cleanly independent of any
Kp-scaling (which is a per-storm CONSTANT and therefore cannot affect within-storm rank
correlation - multiplying every object's score by the same positive constant preserves Spearman
rho exactly), a clean no-fallback leave-one-storm-out ML refit, baseline-window sensitivity
applied uniformly to all 8 storms, and the between-storm Kp question kept strictly separate.

Does NOT modify results/step6/ or results/step6b/ - loads results/step6/rq2x_objects.csv
read-only (it already has rho_quiet from the Part B NRLMSIS run, so no new density computation
is needed here) and data/clean/tle_<storm>.csv read-only (for the alternative-baseline check).
All new output goes to results/step6c/.

Run:  python 16c_rq2_final.py
Outputs in results/step6c/: rq2x_final_scores.csv, rq2x_final_summary.csv, summary.txt,
fig_rq2x_final_skill.png. Then zips results/step6c/ + this script into step6c_results.zip.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CLEAN, RESULTS, ROOT

STEP6 = RESULTS / "step6"
STEP6C = RESULTS / "step6c"
STEP6C.mkdir(parents=True, exist_ok=True)
SEED = 42
SEED_BOOT = 20261026
N_BOOT = 2000
TOP_FRACTION = 0.10
PRE_DAYS, AFTER_DAYS, TOL_DAYS = 6, 3, 0.5
H_SCALE, H_REF = 60.0, 400.0
IN_SAMPLE_STORMS = {"2015_stpatrick", "2017_september"}
ALL_STORMS_START = {
    "2003_halloween": "2003-10-29", "2022_starlink": "2022-02-03", "2024_gannon": "2024-05-10",
    "2015_stpatrick": "2015-03-17", "2017_september": "2017-09-07", "2023_april": "2023-04-23",
    "2024_october": "2024-10-10", "2025_november": "2025-11-12",
}
OOS_STORMS = [s for s in ALL_STORMS_START if s not in IN_SAMPLE_STORMS]

def kp_multiplier(kp):
    return np.select([kp <= 3, kp <= 5, kp <= 7, kp <= 8], [1.0, 1.5, 3.0, 6.0], default=10.0)

omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)

# ====================================================================== load + R_rank, R_msis_rank (no new density compute - reuses rho_quiet)
d = pd.read_csv(STEP6 / "rq2x_objects.csv")
b = d["bstar"].clip(lower=1e-6)
d["R_rank"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE)
d["R_msis_rank"] = b * d["rho_quiet"]

# ====================================================================== item 2: clean leave-one-storm-out ML, NO fallback
print("=== 2: ML_obs (8 storms) and ML_fc (6 out-of-sample storms only), no fallback ===")
def ml_loso(storms_included, kp_col):
    X_cols = ["alt_km", "log_bstar", "inclination", kp_col]
    out = pd.Series(np.nan, index=d.index)
    sub_all = d[d.storm.isin(storms_included)].copy()
    sub_all["log_bstar"] = np.log10(sub_all["bstar"].clip(lower=1e-6))
    med = sub_all[["alt_km", "log_bstar", "inclination", kp_col]].median()
    for test_storm in storms_included:
        tr_mask = (sub_all.storm != test_storm) & sub_all.passive
        te_mask = sub_all.storm == test_storm
        if tr_mask.sum() < 50:
            continue
        Xtr = sub_all.loc[tr_mask, ["alt_km", "log_bstar", "inclination", kp_col]].fillna(med)
        ytr = sub_all.loc[tr_mask, "excess_decay_km"].clip(-5, 50)
        Xte = sub_all.loc[te_mask, ["alt_km", "log_bstar", "inclination", kp_col]].fillna(med)
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED, loss="huber")
        m.fit(Xtr, ytr)
        out.loc[sub_all.index[te_mask]] = m.predict(Xte)
    return out

d["ML_obs"] = ml_loso(list(ALL_STORMS_START.keys()), "kp_observed")
d["ML_fc"] = np.nan
d.loc[d.storm.isin(OOS_STORMS), "ML_fc"] = ml_loso(OOS_STORMS, "kp_forecast")[d.storm.isin(OOS_STORMS)]
print(f"  ML_obs: fit/predicted over all 8 storms.")
print(f"  ML_fc: fit/predicted over {OOS_STORMS} only - 2015/2017 excluded entirely (no fallback).")

OUT_OBJ = STEP6C / "rq2x_final_scores.csv"
d.to_csv(OUT_OBJ, index=False)

# ====================================================================== item 1: within-storm rho/AUC + bootstrap, 6 clean scores, passive + payloads
METHODS = {"Altitude only": "score_altitude", "Pre-storm decay": "score_decay_rate",
           "R_rank": "R_rank", "R_msis_rank": "R_msis_rank", "ML_obs": "ML_obs", "ML_fc": "ML_fc"}

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

CKPT = STEP6C / "_within_storm.csv"
if CKPT.exists():
    print(f"\nWithin-storm checkpoint found, loading {CKPT}")
    within = pd.read_csv(CKPT)
else:
    print("\n=== 1: within-storm rho/AUC + bootstrap (passive + payloads, all 8 storms) ===")
    rows = []
    for group_name, gmask in (("PASSIVE", d["passive"]), ("PAYLOADS", ~d["passive"])):
        for storm in ALL_STORMS_START:
            sub = d[gmask & (d.storm == storm)]
            if len(sub) < 20:
                continue
            for mname, col in METHODS.items():
                rho, auc = rho_auc(sub, col)
                (rho_lo, rho_hi), (auc_lo, auc_hi) = bootstrap_ci(sub, col, seed=SEED_BOOT + hash((group_name, storm, mname)) % 10000)
                rows.append({"group": group_name, "storm": storm, "method": mname, "n": len(sub), "rho": rho,
                            "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi, "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
            print(f"  {group_name} {storm} done")
    within = pd.DataFrame(rows)
    within.to_csv(CKPT, index=False)

# ====================================================================== item 3: summary across storms (median/mean/count>=0.9), all-8 vs excl-Oct2024
print("\n=== 3: cross-storm summary, all 8 vs excluding October 2024 only ===")
summary_rows = []
for group_name in ("PASSIVE", "PAYLOADS"):
    for scope_name, excluded in (("ALL_8_STORMS", set()), ("EXCLUDE_2024_OCTOBER", {"2024_october"})):
        sub = within[(within.group == group_name) & (~within.storm.isin(excluded))]
        for mname in METHODS:
            r = sub[sub.method == mname]["rho"].dropna()
            summary_rows.append({"group": group_name, "scope": scope_name, "method": mname,
                                 "n_storms": len(r), "median_rho": float(r.median()) if len(r) else np.nan,
                                 "mean_rho": float(r.mean()) if len(r) else np.nan,
                                 "n_storms_rho_ge_0.9": int((r >= 0.9).sum())})
cross_df = pd.DataFrame(summary_rows)
print(cross_df.round(3).to_string(index=False))

# ====================================================================== item 4: alternative baseline sensitivity (uniform rule, all 8 storms)
print("\n=== 4: alternative pre-storm baseline sensitivity (all 8 storms, passive, R_rank) ===")
def alt_baseline_end(onset):
    look = omni.loc[onset - pd.Timedelta(days=10): onset - pd.Timedelta(hours=1), "kp"]
    hi = look[look >= 5]
    if hi.empty:
        return onset, False
    gaps = hi.index.to_series().diff() > pd.Timedelta(hours=48)
    gid = gaps.cumsum()
    starts = hi.groupby(gid).apply(lambda g: g.index.min())
    return starts.iloc[-1], True

def rebuild_ground_truth_alt_baseline(storm):
    onset = pd.Timestamp(ALL_STORMS_START[storm])
    base_end, shifted = alt_baseline_end(onset)
    f = CLEAN / f"tle_{storm}.csv"
    t = pd.read_csv(f, low_memory=False, parse_dates=["EPOCH"])
    lo = min(base_end - pd.Timedelta(days=PRE_DAYS), onset - pd.Timedelta(days=PRE_DAYS))
    hi = onset + pd.Timedelta(days=AFTER_DAYS + TOL_DAYS)
    t = t[(t["EPOCH"] >= lo) & (t["EPOCH"] <= hi)]
    rows = []
    for sat, g in t.groupby("NORAD_CAT_ID"):
        g = g.sort_values("EPOCH")
        pre = g[(g["EPOCH"] >= base_end - pd.Timedelta(days=PRE_DAYS)) & (g["EPOCH"] < base_end)]
        post = g[(g["EPOCH"] >= onset + pd.Timedelta(days=AFTER_DAYS - TOL_DAYS)) &
                 (g["EPOCH"] <= onset + pd.Timedelta(days=AFTER_DAYS + TOL_DAYS))]
        if len(pre) < 4 or (pre["EPOCH"].max() - pre["EPOCH"].min()).total_seconds() / 86400 < 3 or len(post) < 1:
            continue
        window = g[(g["EPOCH"] >= base_end - pd.Timedelta(days=PRE_DAYS)) & (g["EPOCH"] <= onset + pd.Timedelta(days=AFTER_DAYS + TOL_DAYS))]
        if (window["ALT_KM"].diff() > 0.5).any():
            continue
        pre_days_num = (pre["EPOCH"] - base_end).dt.total_seconds() / 86400.0
        slope, intercept = np.polyfit(pre_days_num, pre["ALT_KM"], 1)
        t_after_from_base = (onset - base_end).total_seconds() / 86400.0 + AFTER_DAYS
        arg = 1 + slope * t_after_from_base / H_SCALE
        expected = intercept + H_SCALE * np.log(arg) if (slope < 0 and arg > 0.05) else intercept + slope * t_after_from_base
        observed = post["ALT_KM"].median()
        h0 = intercept + slope * (onset - base_end).total_seconds() / 86400.0   # altitude AT ONSET under this fit
        if not (200 <= h0 <= 600):
            continue
        rows.append({"norad_id": sat, "object_type": str(g["OBJECT_TYPE"].iloc[-1]).upper(),
                     "excess_decay_km_alt": expected - observed})
    gt = pd.DataFrame(rows)
    if gt.empty:
        return np.nan, np.nan, shifted, base_end
    gt["passive"] = gt["object_type"].isin({"DEBRIS", "ROCKET BODY"})
    gt = gt[gt.passive].merge(d[d.storm == storm][["norad_id", "R_rank"]], on="norad_id", how="inner")
    ok = gt["R_rank"].notna() & gt["excess_decay_km_alt"].notna()
    rho = spearmanr(gt.loc[ok, "R_rank"], gt.loc[ok, "excess_decay_km_alt"]).correlation if ok.sum() >= 20 else np.nan
    frac_neg = float((gt["excess_decay_km_alt"] < 0).mean()) if len(gt) else np.nan
    return rho, frac_neg, shifted, base_end

alt_rows = []
for storm in ALL_STORMS_START:
    rho, frac_neg, shifted, base_end = rebuild_ground_truth_alt_baseline(storm)
    alt_rows.append({"storm": storm, "baseline_shifted": shifted, "alt_baseline_end": base_end,
                     "rho_Rrank_alt_baseline": rho, "frac_negative_excess_decay_alt": frac_neg})
    print(f"  {storm}: baseline_shifted={shifted} (ends {base_end})  rho(R_rank)={rho}  frac_neg={frac_neg}")
alt_df = pd.DataFrame(alt_rows)

# ====================================================================== item 5: between-storm Kp question, kept separate, pooled only
print("\n=== 5: between-storm Kp question (pooled rho, PASSIVE, all objects) ===")
pooled_rows = []
pas_all = d[d.passive]
rho, auc = rho_auc(pas_all, "score_physics_obsKp")
(rho_lo, rho_hi), _ = bootstrap_ci(pas_all, "score_physics_obsKp", seed=SEED_BOOT + 1)
pooled_rows.append({"variant": "observed max Kp, all 8 storms", "n": len(pas_all), "rho": rho, "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi})
pas_oos = d[d.passive & d.storm.isin(OOS_STORMS)]
rho2, auc2 = rho_auc(pas_oos, "score_physics")
(rho_lo2, rho_hi2), _ = bootstrap_ci(pas_oos, "score_physics", seed=SEED_BOOT + 2)
pooled_rows.append({"variant": "RF forecast Kp, 6 out-of-sample storms only", "n": len(pas_oos), "rho": rho2, "rho_ci_lo": rho_lo2, "rho_ci_hi": rho_hi2})
pooled_df = pd.DataFrame(pooled_rows)
print(pooled_df.round(3).to_string(index=False))

# ====================================================================== save tables
within.to_csv(STEP6C / "rq2x_final_within_storm.csv", index=False)
cross_df.to_csv(STEP6C / "rq2x_final_summary.csv", index=False)
alt_df.to_csv(STEP6C / "rq2x_final_alt_baseline.csv", index=False)
pooled_df.to_csv(STEP6C / "rq2x_final_pooled_kp.csv", index=False)

# ====================================================================== figure: rho per storm per score (PASSIVE), storms on x-axis
storms_order = list(ALL_STORMS_START.keys())
fig, ax = plt.subplots(figsize=(13, 5))
w = 0.13
x = np.arange(len(storms_order))
for i, mname in enumerate(METHODS):
    vals = [within[(within.storm == s) & (within.group == "PASSIVE") & (within.method == mname)]["rho"]
            .pipe(lambda ser: ser.iloc[0] if len(ser) else np.nan) for s in storms_order]
    ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, vals, w, label=mname)
ax.set_xticks(x); ax.set_xticklabels(storms_order, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Spearman rho (passive objects)")
ax.set_title("RQ2 final: within-storm skill, Kp-independent scores (8 storms)")
ax.axhline(0, color="black", lw=0.6)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.38))
fig.tight_layout(); fig.savefig(STEP6C / "fig_rq2x_final_skill.png", dpi=200); plt.close(fig)

# ====================================================================== summary.txt
lines = ["AETHERA - RQ2 FINAL: Kp-independent within-storm scores, clean no-fallback ML, baseline sensitivity", ""]
lines.append("Note: within a single storm, any Kp-based multiplier is a CONSTANT across all objects, so it")
lines.append("cannot change within-storm Spearman rank correlation (multiplying every score by the same")
lines.append("positive constant preserves rank order exactly). R_rank = B*exp(-(h-400)/60) and")
lines.append("R_msis_rank = B*rho_quiet(h) are therefore the clean within-storm comparators; the Kp question")
lines.append("is evaluated separately in section 5 below (pooled across storms, where different storms'")
lines.append("different Kp constants DO matter for cross-storm rank comparison).")
lines.append("")
lines.append("=== 2: ML refit, no fallback ===")
lines.append(f"  ML_obs: leave-one-storm-out over all 8 storms, feature = observed max Kp (always real).")
lines.append(f"  ML_fc: leave-one-storm-out over {OOS_STORMS} only - 2015/2017 excluded from BOTH training")
lines.append("  and testing (not just output-NaN'd); no fallback value used anywhere for this model.")
lines.append("")
lines.append("=== 1: within-storm rho/AUC (top 10%), bootstrap 95% CI, PASSIVE objects ===")
for storm in ALL_STORMS_START:
    sub = within[(within.storm == storm) & (within.group == "PASSIVE")]
    if sub.empty:
        continue
    lines.append(f"  --- {storm} (n={int(sub.n.iloc[0])}) ---")
    for _, r in sub.iterrows():
        rho_s = f"{r.rho:6.3f} [{r.rho_ci_lo:6.3f},{r.rho_ci_hi:6.3f}]" if not np.isnan(r.rho) else "   NaN"
        auc_s = f"{r.auc:5.3f} [{r.auc_ci_lo:5.3f},{r.auc_ci_hi:5.3f}]" if not np.isnan(r.auc) else "  NaN"
        lines.append(f"    {r.method:14s} rho={rho_s}  AUC={auc_s}")
lines.append("")
lines.append("=== 1 (PAYLOADS) within-storm rho, bootstrap 95% CI ===")
for storm in ALL_STORMS_START:
    sub = within[(within.storm == storm) & (within.group == "PAYLOADS")]
    if sub.empty:
        continue
    lines.append(f"  --- {storm} (n={int(sub.n.iloc[0])}) ---")
    for _, r in sub.iterrows():
        rho_s = f"{r.rho:6.3f} [{r.rho_ci_lo:6.3f},{r.rho_ci_hi:6.3f}]" if not np.isnan(r.rho) else "   NaN"
        lines.append(f"    {r.method:14s} rho={rho_s}")
lines.append("")
lines.append("=== 3: cross-storm summary (median/mean rho, # storms rho>=0.9) ===")
lines.append(cross_df.round(3).to_string(index=False))
lines.append("")
lines.append("=== 4: alternative baseline sensitivity (ends at the most recent earlier Kp>=5 period, 10-day lookback) ===")
lines.append(alt_df.round(3).to_string(index=False))
lines.append("")
lines.append("=== 5: between-storm Kp question (kept separate, pooled rho, PASSIVE, all objects) ===")
lines.append(pooled_df.round(3).to_string(index=False))
summary = "\n".join(lines)
(STEP6C / "summary.txt").write_text(summary)
print("\n" + summary)

CKPT.unlink()

# ====================================================================== zip
out_zip = ROOT / "step6c_results.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in STEP6C.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(ROOT))
    z.write(Path(__file__), f"code/{Path(__file__).name}")
print(f"\n{out_zip}: {out_zip.stat().st_size:,} bytes ({out_zip.stat().st_size/1e6:.2f} MB)")
