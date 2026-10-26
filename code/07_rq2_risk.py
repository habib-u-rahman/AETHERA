"""
Step 5 - RQ2: Does a per-satellite risk score find the objects that lost the most height?

For every object and every storm we compute:
  TRUTH (what really happened)
    excess_decay_km = how much MORE height the object lost in the 3 days after storm onset
                      than it would have lost at its normal, pre-storm rate.
  RISK SCORES (what we could have said BEFORE the storm)
    1. Altitude only         - lower objects = more risk (simple baseline)
    2. Physics score         - B* x air density at that height x storm strength (from Kp forecast)
    3. ML score              - Gradient Boosting trained on the OTHER two storms
                               (leave-one-storm-out, so it never sees the storm it is tested on)
    3b. ML score with observed Kp - same, but using the real Kp (shows the cost of forecast errors)

Main test set: PASSIVE objects only (debris and rocket bodies), because active satellites
fire their engines and hide the effect of the storm.

Run:  python 07_rq2_risk.py         (a few minutes)
Needs: data/clean/tle_<storm>.csv  (Step 3), data/clean/omni_hourly.csv, results/rq1_forecasts.csv (Step 4)

Outputs in results/:
    rq2_objects.csv          one row per object per storm: truth + all scores
    rq2_summary.txt          the main results table
    fig_rq2_scatter.png      physics score vs real excess decay, one panel per storm
    fig_rq2_skill.png        Spearman and ROC-AUC for each method and storm
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from config import CLEAN, RESULTS, STORMS

SEED = 42
PRE_DAYS = 6            # pre-storm window used to measure the normal decay rate
AFTER_DAYS = 3          # we measure the effect 3 days after onset
TOL_DAYS = 0.5          # TLEs within +/- 0.5 day of onset + 3 days
H_SCALE = 60.0          # km, approximate density scale height in the 300-550 km range
H_REF = 400.0           # km, reference altitude (only scales the score)
TOP_FRACTION = 0.10     # "high decay" = top 10% of excess decay within a storm
PASSIVE = {"DEBRIS", "ROCKET BODY"}

# Storm-time density enhancement factor vs Kp (step function used by Kukreja et al. 2026)
def kp_multiplier(kp):
    return np.select([kp <= 3, kp <= 5, kp <= 7, kp <= 8], [1.0, 1.5, 3.0, 6.0], default=10.0)

# ------------------------------------------------------------------ storm strength (forecast + observed)
omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
fc = pd.read_csv(RESULTS / "rq1_forecasts.csv", parse_dates=["time"])
storm_kp = {}
def rf_forecast_before(t):
    """Latest RF forecast issued at or before time t (max over the 4 horizons)."""
    issued = fc[(fc.time <= t) & (fc.time >= t - pd.Timedelta(days=2))]
    if issued.empty:
        return float(omni.loc[:t, "kp"].iloc[-1]), t      # fallback: persistence
    last = issued.sort_values("time").iloc[-1]
    return float(max(last[f"RandomForest_{h}h"] for h in (3, 6, 12, 24))), last.time

for name, info in STORMS.items():
    t0 = pd.Timestamp(info["start"])
    kp_fc, t_issue = rf_forecast_before(t0)                   # issued at 00:00 UT on onset day
    win = omni.loc[t0 - pd.Timedelta(hours=12):t0 + pd.Timedelta(days=2), "kp"]
    t_storm = win[win >= 5].index.min() if (win >= 5).any() else t0
    kp_fc_start, t_issue_start = rf_forecast_before(t_storm)  # issued when Kp first reached 5
    kp_obs = float(omni.loc[t0:t0 + pd.Timedelta(days=AFTER_DAYS), "kp"].max())
    storm_kp[name] = (kp_fc, kp_obs, t_issue, kp_fc_start, t_issue_start, t_storm)
    print(f"{name}: forecast at {t_issue}: max Kp {kp_fc:.2f} | storm conditions from {t_storm}, "
          f"forecast at {t_issue_start}: max Kp {kp_fc_start:.2f} | observed max Kp {kp_obs:.2f}")

# ------------------------------------------------------------------ ground truth per object
rows = []
for name, info in STORMS.items():
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
        if (g["ALT_KM"].diff() > 0.5).any():          # engine burn -> skip
            continue
        slope, intercept = np.polyfit(pre["day"], pre["ALT_KM"], 1)   # km/day (negative = decaying)
        expected_lin = intercept + slope * AFTER_DAYS
        # Without a storm, a falling object still speeds up as it enters denser air.
        # With an exponential atmosphere (scale height H) the no-storm path is
        #   h(t) = h0 + H * ln(1 + slope * t / H)
        arg = 1 + slope * AFTER_DAYS / H_SCALE
        expected = intercept + H_SCALE * np.log(arg) if (slope < 0 and arg > 0.05) else expected_lin
        observed = post["ALT_KM"].median()
        h0 = intercept                                   # altitude at onset, from the pre-storm fit
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
if d.empty:
    raise SystemExit("No objects passed the checks - send me the output above.")
d["passive"] = d["object_type"].isin(PASSIVE)
d["kp_forecast"] = d["storm"].map(lambda s: storm_kp[s][0])
d["kp_observed"] = d["storm"].map(lambda s: storm_kp[s][1])
d["kp_forecast_start"] = d["storm"].map(lambda s: storm_kp[s][3])

# ------------------------------------------------------------------ risk scores
b = d["bstar"].clip(lower=1e-6)                        # B* <= 0 is noise; treat as very small
d["score_altitude"] = -d["alt_km"]                     # lower = riskier
d["score_decay_rate"] = -d["pre_rate_km_day"]          # already falling fast = riskier
d["score_physics"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_forecast"])
d["score_physics_obsKp"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_observed"])
d["score_physics_startKp"] = b * np.exp(-(d["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(d["kp_forecast_start"])

def ml_scores(kp_col):
    out = pd.Series(np.nan, index=d.index)
    feats = ["alt_km", "log_bstar", "inclination", kp_col]
    X = d.assign(log_bstar=np.log10(b))[feats].fillna(d.assign(log_bstar=np.log10(b))[feats].median())
    y = d["excess_decay_km"].clip(-5, 50)               # limit the effect of a few extreme objects
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
d.to_csv(RESULTS / "rq2_objects.csv", index=False)

# ------------------------------------------------------------------ evaluation
METHODS = {"Altitude only": "score_altitude", "Pre-storm decay": "score_decay_rate", "Physics score": "score_physics",
           "ML (forecast Kp)": "score_ml", "ML (observed Kp)": "score_ml_obsKp"}

def evaluate(sub):
    res = {}
    if len(sub) < 20:
        return None
    thr = sub["excess_decay_km"].quantile(1 - TOP_FRACTION)
    label = (sub["excess_decay_km"] >= thr).astype(int)
    for mname, col in METHODS.items():
        s = sub[col]
        ok = s.notna()
        if ok.sum() < 20 or label[ok].nunique() < 2:
            res[mname] = (np.nan, np.nan); continue
        rho = spearmanr(s[ok], sub.loc[ok, "excess_decay_km"]).correlation
        auc = roc_auc_score(label[ok], s[ok])
        res[mname] = (rho, auc)
    return res

lines = ["AETHERA - RQ2 results (per-satellite risk)", "",
         "Truth = excess altitude loss 3 days after onset (vs. pre-storm decay rate).",
         f"High decay = top {int(TOP_FRACTION*100)}% within each storm. ML is leave-one-storm-out.", ""]
for name, (kpf, kpo, ti, kps, tis, tst) in storm_kp.items():
    lines.append(f"{name}: RF forecast at {ti}: max Kp {kpf:.2f} | at storm start ({tst}): {kps:.2f} | observed max Kp {kpo:.2f}")
lines.append("")
lines.append("Effect of the density correction on the truth (median excess km, passive):")
for storm in STORMS:
    sub = d[d["passive"] & (d["storm"] == storm)]
    if len(sub):
        lines.append(f"  {storm:16s} linear baseline {sub['excess_decay_linear_km'].median():.3f}  ->  corrected {sub['excess_decay_km'].median():.3f}")
lines.append("")
skill = []
for group, gmask in (("PASSIVE (debris + rocket bodies)", d["passive"]), ("PAYLOADS (active or dead satellites)", ~d["passive"])):
    lines.append(f"=== {group} ===")
    lines.append(f"{'storm':16s} {'n':>6s} {'median excess km':>17s} | " + " | ".join(f"{m:>17s}" for m in METHODS))
    lines.append(f"{'':16s} {'':>6s} {'':>17s} | " + " | ".join(f"{'rho / AUC':>17s}" for _ in METHODS))
    for storm in STORMS:
        sub = d[gmask & (d["storm"] == storm)]
        r = evaluate(sub)
        if r is None:
            lines.append(f"{storm:16s} {len(sub):6d}   (too few objects)"); continue
        lines.append(f"{storm:16s} {len(sub):6d} {sub['excess_decay_km'].median():17.3f} | " +
                     " | ".join(f"{r[m][0]:7.3f} / {r[m][1]:6.3f}" for m in METHODS))
        if group.startswith("PASSIVE"):
            for m in METHODS:
                skill.append({"storm": storm, "method": m, "spearman": r[m][0], "auc": r[m][1], "n": len(sub)})
    lines.append("")

# pooled across storms: here the storm strength (Kp) matters, not only altitude and B*
pas = d[d["passive"]]
lines.append("=== PASSIVE, all three storms pooled (Spearman rho) ===")
for m, col in METHODS.items():
    ok = pas[col].notna()
    lines.append(f"  {m:18s} rho = {spearmanr(pas.loc[ok, col], pas.loc[ok, 'excess_decay_km']).correlation:.3f}  (n={ok.sum()})")
for k, col in (("Physics, Kp forecast at storm start", "score_physics_startKp"), ("Physics, observed Kp", "score_physics_obsKp")):
    lines.append(f"  {k:36s} rho = {spearmanr(pas[col], pas['excess_decay_km']).correlation:.3f}")
lines.append("")

# ablation on the physics score: remove one ingredient at a time (pooled passive objects)
lines.append("=== Ablation of the physics score (passive, pooled, Spearman rho) ===")
bb = pas["bstar"].clip(lower=1e-6); dens = np.exp(-(pas["alt_km"] - H_REF) / H_SCALE); km = kp_multiplier(pas["kp_forecast"])
abl = {"Full score (B* x density x Kp)": bb * dens * km, "Without Kp": bb * dens,
       "Without B*": dens * km, "Without altitude": bb * km}
for k, v in abl.items():
    lines.append(f"  {k:32s} rho = {spearmanr(v, pas['excess_decay_km']).correlation:.3f}")
summary = "\n".join(lines)
(RESULTS / "rq2_summary.txt").write_text(summary)
print("\n" + summary)

# ------------------------------------------------------------------ figures
storms = [s for s in STORMS if ((d["storm"] == s) & d["passive"]).sum() > 0]
fig, axes = plt.subplots(1, len(storms), figsize=(4 * len(storms), 3.6))
axes = np.atleast_1d(axes)
for ax, s in zip(axes, storms):
    sub = d[(d["storm"] == s) & d["passive"]]
    ax.scatter(np.log10(sub["score_physics"]), sub["excess_decay_km"].clip(-5, 60), s=6, alpha=0.5)
    rho = spearmanr(sub["score_physics"], sub["excess_decay_km"]).correlation
    ax.set_title(f"{s} (n={len(sub)}, rho={rho:.2f})", fontsize=10)
    ax.set_xlabel("log10 physics risk score"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Excess altitude loss (km)")
fig.tight_layout(); fig.savefig(RESULTS / "fig_rq2_scatter.png", dpi=200); plt.close(fig)

sk = pd.DataFrame(skill)
if not sk.empty:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    w = 0.16
    for i, m in enumerate(METHODS):
        part = sk[sk.method == m].set_index("storm").reindex(storms)
        x = np.arange(len(storms)) + (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(x, part["spearman"], w, label=m)
        axes[1].bar(x, part["auc"], w, label=m)
    for ax, t in zip(axes, ("Spearman rank correlation", "ROC-AUC (top 10% decay)")):
        ax.set_xticks(np.arange(len(storms))); ax.set_xticklabels(storms, fontsize=9)
        ax.set_title(t); ax.grid(axis="y", alpha=0.3)
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(RESULTS / "fig_rq2_skill.png", dpi=200); plt.close(fig)
print("\nFiles saved in", RESULTS)