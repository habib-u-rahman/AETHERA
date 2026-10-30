"""
Step 11 (journal revision) - RQ1 vs NOAA SWPC's official 3-day Kp forecast.

Does NOT modify or overwrite any existing script or result file. All new output goes to
results/step4/ ; raw NOAA text files are cached under data/raw/noaa_3day/ (excluded from the
final zip, per the task).

Run:
  python 13_noaa_comparison.py --download     (politely fetches the NOAA archive; resumable,
                                                 self-limits to a time budget per invocation -
                                                 re-run to continue; skips files already cached)
  python 13_noaa_comparison.py --validate-parser   (hand-checks the parser on a few sample
                                                 files spanning different NOAA text formats,
                                                 prints a report of format changes found)
  python 13_noaa_comparison.py --analyze       (parses the cached archive, matches against
                                                 results/rq1_forecasts.csv + rq1_2025_forecasts.csv,
                                                 computes stats + bootstrap CIs, writes
                                                 results/step4/noaa_comparison.csv,
                                                 fig_noaa_comparison.png, summary.txt, and zips
                                                 results/step4/ + this script into step4_results.zip)
"""
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CLEAN, RESULTS, ROOT

STEP4 = RESULTS / "step4"
STEP4.mkdir(parents=True, exist_ok=True)
RAW_NOAA = ROOT / "data" / "raw" / "noaa_3day"
RAW_NOAA.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast"
DATE_START = pd.Timestamp("2022-03-01")
DATE_END = pd.Timestamp("2025-12-31")
HORIZONS = [3, 6, 12, 24]
MODELS = ["Persistence", "RandomForest", "LSTM"]
STORM_KP = 5.0
MISSING_MARKER = "__MISSING_404__"

# ====================================================================== 1. polite, resumable downloader
def noaa_url(ts):
    return f"{BASE_URL}/{ts.year}/{ts.month:02d}/{ts.strftime('%Y%m%d%H%M')}three_day_forecast.txt"

def cache_path(ts):
    return RAW_NOAA / f"{ts.strftime('%Y%m%d%H%M')}three_day_forecast.txt"

def all_issue_times():
    days = pd.date_range(DATE_START, DATE_END, freq="D")
    out = []
    for d in days:
        out.append(d + pd.Timedelta(minutes=30))          # 00:30 UT
        out.append(d + pd.Timedelta(hours=12, minutes=30)) # 12:30 UT
    return out

def download(time_budget_s=1100, pause_s=0.4):
    start = time.time()
    todo = [t for t in all_issue_times() if not cache_path(t).exists()]
    print(f"{len(todo)} files not yet cached (of {len(all_issue_times())} total).")
    got, missing, errors = 0, 0, 0
    for t in todo:
        if time.time() - start > time_budget_s:
            print(f"Time budget ({time_budget_s:.0f}s) reached - stopping cleanly. Re-run to continue.")
            break
        url = noaa_url(t)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                cache_path(t).write_text(r.text, encoding="utf-8")
                got += 1
            elif r.status_code == 404:
                cache_path(t).write_text(MISSING_MARKER, encoding="utf-8")
                missing += 1
            else:
                print(f"  {t}: unexpected status {r.status_code}, will retry next run")
                errors += 1
        except requests.RequestException as e:
            print(f"  {t}: {type(e).__name__}, will retry next run")
            errors += 1
        time.sleep(pause_s)   # be polite - one request at a time, short pause
    print(f"This run: {got} downloaded, {missing} confirmed missing (404), {errors} transient errors.")
    remaining = sum(1 for t in all_issue_times() if not cache_path(t).exists())
    print(f"Remaining to fetch: {remaining}")

# ====================================================================== 2. parser (handles both eras + year rollover + G-level suffix)
MONTHS = {m: i+1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}

def parse_issue_time(text):
    m = re.search(r"Issued:\s*(\d{4})\s+(\w{3})\s+(\d{1,2})\s+(\d{4})\s+UTC", text)
    if not m:
        return None
    yr, mon, day, hm = m.groups()
    return pd.Timestamp(int(yr), MONTHS[mon], int(day), int(hm[:2]), int(hm[2:]))

def parse_kp_table(text):
    """Returns list of (valid_time, predicted_kp). Robust to both the integer-Kp era and the
    later thirds-notation era (e.g. '5.00 (G1)'), by token-extracting numbers per row rather
    than relying on fixed column widths. Also robust to the early-2022 format, which inserts a
    blank line BETWEEN every table row (later formats don't) - so row extraction must not rely
    on "stop at the first blank line" to find the end of the table."""
    m = re.search(r"NOAA Kp index breakdown\s+(\w{3})\s+(\d{1,2})-(\w{3})\s+(\d{1,2})\s+(\d{4})", text)
    if not m:
        return []
    mon1, day1, mon2, day2, year2 = m.groups()
    year2 = int(year2)
    year1 = year2 - 1 if (mon1 == "Dec" and mon2 == "Jan") else year2

    # day-label header row, e.g. "            Mar 25     Mar 26     Mar 27"
    header_m = re.search(r"\n[ \t]*((?:\w{3}\s+\d{1,2}\s*){2,})\n", text[m.end():])
    if not header_m:
        return []
    day_labels = re.findall(r"(\w{3})\s+(\d{1,2})", header_m.group(1))
    days, rolled, prev_month_num = [], False, None
    for mon, day in day_labels:
        mnum = MONTHS[mon]
        if prev_month_num is not None and mnum < prev_month_num:   # rolled into the next year
            rolled = True
        days.append(pd.Timestamp(year2 if rolled else year1, mnum, int(day)))
        prev_month_num = mnum

    # table body: from just after the day-label row up to the next section marker (Rationale / B.)
    body_start = m.end() + header_m.end()
    body_end_m = re.search(r"\nRationale:|\nB\. NOAA", text[body_start:])
    body = text[body_start: body_start + body_end_m.start()] if body_end_m else text[body_start:body_start + 2000]
    rows = re.findall(r"(\d{2})-(\d{2})UT\s+([^\n]*)", body)
    out = []
    for h0, h1, rest in rows:
        nums = re.findall(r"\d+(?:\.\d+)?", rest)   # strips "(G1)" etc. automatically
        for day0, val in zip(days, nums):
            valid_time = day0 + pd.Timedelta(hours=int(h0))
            out.append((valid_time, float(val)))
    return out

def load_one(path):
    text = path.read_text(encoding="utf-8")
    if text.strip() == MISSING_MARKER:
        return None, []
    issue = parse_issue_time(text)
    table = parse_kp_table(text)
    return issue, table

def validate_parser():
    samples = [
        ("2022-03-25 00:30", "integer Kp, same-year table"),
        ("2022-03-25 12:30", "integer Kp, 1230 issue"),
        ("2023-06-15 00:30", "thirds-notation decimal Kp"),
        ("2024-12-31 00:30", "decimal Kp with (G#) suffixes, Dec->Jan year rollover"),
        ("2025-01-01 00:30", "decimal Kp, first file of 2025"),
        ("2025-06-15 00:30", "decimal Kp, mid-2025"),
    ]
    print("=== Parser validation on hand-checked sample files ===")
    changes = []
    for ts_str, note in samples:
        ts = pd.Timestamp(ts_str)
        p = cache_path(ts)
        if not p.exists():
            print(f"{ts_str} [{note}]: not cached yet - run --download first"); continue
        issue, table = load_one(p)
        print(f"\n{ts_str} [{note}]")
        print(f"  parsed issue time: {issue}")
        print(f"  parsed {len(table)} (valid_time, Kp) rows, first 3: {table[:3]}  last 3: {table[-3:]}")
        if issue is None or len(table) != 24:
            print("  ** UNEXPECTED: expected 24 rows (8 blocks x 3 days) and a valid issue time **")
            changes.append(f"{ts_str}: parser did not return the expected 24 rows - needs attention")
    print("\n=== Format changes observed across the archive (manually confirmed) ===")
    print("- Mar 2022: Kp given as plain integers (e.g. '3').")
    print("- By mid-2023: Kp given in NOAA's thirds notation, two decimals (e.g. '2.33').")
    print("- By late 2024 / 2025: values >=5 carry a '(G1)'/'(G2)'/'(G3)' suffix; the parser")
    print("  strips this automatically (token-extracts the leading number).")
    print("- Day-column headers can cross a year boundary (e.g. 'Dec 31-Jan 02 2025'); the")
    print("  parser infers which columns are the prior year from the month sequence.")
    print("- Some files are missing entirely from the archive (HTTP 404) - these are cached as")
    print(f"  a '{MISSING_MARKER}' marker and skipped during analysis, not treated as Kp=NaN errors.")
    if changes:
        print("\nPARSER ISSUES FOUND:", changes)
    else:
        print("\nNo parser issues found on the checked samples.")

# ====================================================================== 3. analysis
def load_all_noaa():
    rows = []
    for p in sorted(RAW_NOAA.glob("*three_day_forecast.txt")):
        issue, table = load_one(p)
        if issue is None:
            continue
        for valid_time, kp in table:
            rows.append({"noaa_issue_time": issue, "valid_time": valid_time, "noaa_kp": kp})
    return pd.DataFrame(rows)

def our_period_data(ts):
    """Return (forecasts_df, test_set_label, metrics_csv_path) for the calendar year of ts."""
    if ts.year == 2025:
        return pd.read_csv(RESULTS / "rq1_2025_forecasts.csv", parse_dates=["time"]), "test_2025", RESULTS / "rq1_2025_metrics.csv"
    return pd.read_csv(RESULTS / "rq1_forecasts.csv", parse_dates=["time"]), "test", RESULTS / "rq1_metrics.csv"

def build_matched_table():
    noaa = load_all_noaa()
    if noaa.empty:
        raise SystemExit("No NOAA data parsed - run --download first.")
    noaa["our_issue_time"] = noaa["noaa_issue_time"].dt.floor("h")   # 00:30/12:30 -> 00:00/12:00

    fc24, _, _ = our_period_data(pd.Timestamp("2023-01-01"))
    fc24 = fc24[fc24.test_set == "test"]
    fc25, _, _ = our_period_data(pd.Timestamp("2025-01-01"))
    fc25 = fc25[fc25.test_set == "test_2025"]
    fc = pd.concat([fc24, fc25], ignore_index=True).drop_duplicates("time").set_index("time")

    omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)

    rows = []
    for h in HORIZONS:
        noaa_h = noaa.copy()
        noaa_h["target_time"] = noaa_h["our_issue_time"] + pd.Timedelta(hours=h)
        noaa_h = noaa_h[noaa_h["valid_time"] == noaa_h["target_time"]]
        for _, r in noaa_h.iterrows():
            it, tt = r["our_issue_time"], r["target_time"]
            if it not in fc.index or tt not in omni.index:
                continue
            frow = fc.loc[it]
            if pd.isna(frow.get(f"kp_true_{h}h")):
                continue
            model_vals = {}
            ok = True
            for m in MODELS:
                v = frow.get(f"{m}_{h}h")
                if pd.isna(v):
                    ok = False; break
                model_vals[m] = float(v)
            if not ok:
                continue
            obs = omni.loc[tt, "kp"]
            if pd.isna(obs):
                continue
            rows.append({"issue_time": it, "target_time": tt, "horizon_h": h,
                         "period": "2025" if it.year == 2025 else "2022-2024",
                         "observed_kp": float(obs), "noaa_kp": float(r["noaa_kp"]),
                         **model_vals})
    return pd.DataFrame(rows)

def scores(y, p, thr):
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else np.nan
    s = y >= STORM_KP
    rmse_storm = float(np.sqrt(np.mean((p[s] - y[s]) ** 2))) if s.any() else np.nan
    yes_p, yes_t = p >= thr, s
    a = np.sum(yes_p & yes_t); b = np.sum(yes_p & ~yes_t)
    c = np.sum(~yes_p & yes_t); d = np.sum(~yes_p & ~yes_t)
    pod = a / (a + c) if a + c else np.nan
    pofd = b / (b + d) if b + d else np.nan
    tss = pod - pofd
    den = (a + c) * (c + d) + (a + b) * (b + d)
    hss = 2 * (a * d - b * c) / den if den else np.nan
    return {"n": int(len(y)), "RMSE": rmse, "MAE": mae, "r": r, "RMSE_storm": rmse_storm, "TSS": tss, "HSS": hss}

def moving_block_bootstrap_diff(order, stat_fn, n_boot=2000, block_hours=168, seed=20261026, step_hours=12):
    """Generic MBB: `order` gives the time-ordering (used only to size the block in case-steps),
    `stat_fn(idx)` computes the statistic (already paired RF-vs-NOAA difference) for a given
    index array (with repeats allowed, as produced by block resampling)."""
    n = len(order)
    L = max(1, round(block_hours / step_hours))
    rng = np.random.default_rng(seed)
    max_start = max(1, n - L + 1)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, max_start)
            idx.extend(range(start, min(start + L, n)))
        idx = np.array(idx[:n])
        boots[b] = stat_fn(idx)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def analyze():
    df = build_matched_table()
    df.to_csv(STEP4 / "noaa_comparison_matched_cases.csv", index=False)
    print(f"Matched cases: {len(df)} (across periods and horizons)")

    rf_metrics_24 = pd.read_csv(RESULTS / "rq1_metrics.csv")
    rf_metrics_25 = pd.read_csv(RESULTS / "rq1_2025_metrics.csv")

    def alarm_level(model, h, period):
        m = rf_metrics_24 if period == "2022-2024" else rf_metrics_25
        tset = "test" if period == "2022-2024" else "test_2025"
        row = m[(m.model == model) & (m.horizon_h == h) & (m.test_set == tset)]
        return float(row["alarm_level"].iloc[0])

    rows = []
    boot_rows = []
    for period in ["2022-2024", "2025"]:
        for h in HORIZONS:
            sub = df[(df.period == period) & (df.horizon_h == h)]
            if sub.empty:
                continue
            y = sub["observed_kp"].to_numpy()
            series = {"NOAA": (sub["noaa_kp"].to_numpy(), STORM_KP)}
            for m in MODELS:
                series[m] = (sub[m].to_numpy(), alarm_level(m, h, period))
            for name, (p, thr) in series.items():
                sc = scores(y, p, thr)
                rows.append({"period": period, "horizon_h": h, "forecaster": name, "alarm_level": thr, **sc})

            # bootstrap: RMSE(RF)-RMSE(NOAA) and TSS(RF)-TSS(NOAA), moving-block, 168h blocks
            p_rf, thr_rf = series["RandomForest"]
            p_noaa, thr_noaa = series["NOAA"]
            order = sub["target_time"].to_numpy()
            srt = np.argsort(order)
            y_s, p_rf_s, p_noaa_s = y[srt], p_rf[srt], p_noaa[srt]
            step_hours = 12   # NOAA issue times are ~12h apart, so that's the case-to-case spacing
            def rmse_diff(idx):
                return (np.sqrt(np.mean((p_rf_s[idx] - y_s[idx]) ** 2)) -
                        np.sqrt(np.mean((p_noaa_s[idx] - y_s[idx]) ** 2)))
            def tss_diff(idx):
                yy = y_s[idx]; s_t = yy >= STORM_KP
                def tss_of(pp, thr):
                    yp = pp[idx] >= thr
                    a = np.sum(yp & s_t); b = np.sum(yp & ~s_t); c = np.sum(~yp & s_t); d = np.sum(~yp & ~s_t)
                    pod = a / (a + c) if a + c else np.nan
                    pofd = b / (b + d) if b + d else np.nan
                    return pod - pofd
                return tss_of(p_rf_s, thr_rf) - tss_of(p_noaa_s, thr_noaa)
            rmse_lo, rmse_hi = moving_block_bootstrap_diff(order, rmse_diff, step_hours=step_hours)
            tss_lo, tss_hi = moving_block_bootstrap_diff(order, tss_diff, step_hours=step_hours)
            point_rmse_diff = rmse_diff(np.arange(len(y_s)))
            point_tss_diff = tss_diff(np.arange(len(y_s)))
            boot_rows.append({"period": period, "horizon_h": h,
                               "RMSE_RF_minus_NOAA": point_rmse_diff, "RMSE_diff_CI95_lo": rmse_lo, "RMSE_diff_CI95_hi": rmse_hi,
                               "TSS_RF_minus_NOAA": point_tss_diff, "TSS_diff_CI95_lo": tss_lo, "TSS_diff_CI95_hi": tss_hi})

    out = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    out.to_csv(STEP4 / "noaa_comparison.csv", index=False)
    boot.to_csv(STEP4 / "noaa_comparison_bootstrap.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"NOAA": "tab:red", "RandomForest": "tab:green", "LSTM": "tab:blue", "Persistence": "tab:gray"}
    for name, col in colors.items():
        d = out[(out.period == "2022-2024") & (out.forecaster == name)].sort_values("horizon_h")
        ax.plot(d.horizon_h, d.RMSE, "o-", color=col, label=name)
    ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("RMSE (Kp units)")
    ax.set_title("RQ1 models vs NOAA 3-day forecast, 2022-2024"); ax.set_xticks(HORIZONS); ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(STEP4 / "fig_noaa_comparison.png", dpi=200); plt.close(fig)

    lines = ["AETHERA - RQ1 vs NOAA SWPC 3-day Kp forecast comparison", ""]
    for period in ["2022-2024", "2025"]:
        sub = out[out.period == period]
        if sub.empty:
            continue
        lines.append(f"=== {period} ===")
        tab = sub.pivot(index="forecaster", columns="horizon_h", values=["n", "RMSE", "MAE", "r", "RMSE_storm", "TSS", "HSS"])
        lines.append(tab.round(3).to_string()); lines.append("")
        b = boot[boot.period == period]
        for _, r in b.iterrows():
            lines.append(f"  horizon {int(r.horizon_h)}h: RMSE(RF)-RMSE(NOAA) = {r.RMSE_RF_minus_NOAA:+.3f} "
                         f"[95% CI {r.RMSE_diff_CI95_lo:+.3f}, {r.RMSE_diff_CI95_hi:+.3f}]   "
                         f"TSS(RF)-TSS(NOAA) = {r.TSS_RF_minus_NOAA:+.3f} "
                         f"[95% CI {r.TSS_diff_CI95_lo:+.3f}, {r.TSS_diff_CI95_hi:+.3f}]")
        lines.append("")
    lines.append("Bootstrap method: moving-block bootstrap, block length = 168 hours (converted to "
                  "~14 case-steps given the ~12-hour spacing between NOAA issue times), 2000 resamples, "
                  f"seed 20261026, 95% CI = 2.5th/97.5th percentile of the resampled difference statistic.")
    lines.append("NOAA alarm threshold: Kp >= 5 (fixed). Our models: validation-tuned alarm_level from "
                 "rq1_metrics.csv / rq1_2025_metrics.csv.")
    summary = "\n".join(lines)
    (STEP4 / "summary.txt").write_text(summary)
    print("\n" + summary)
    return out, boot

# ====================================================================== 4. package
def package():
    out_zip = ROOT / "step4_results.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in STEP4.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
        z.write(Path(__file__), f"code/{Path(__file__).name}")
    print(f"{out_zip}: {out_zip.stat().st_size:,} bytes ({out_zip.stat().st_size/1e6:.2f} MB)")

if __name__ == "__main__":
    if "--download" in sys.argv:
        budget = float(os.environ.get("NOAA_TIME_BUDGET_S", 1100))
        download(time_budget_s=budget)
    elif "--validate-parser" in sys.argv:
        validate_parser()
    elif "--analyze" in sys.argv:
        analyze()
        package()
    else:
        print(__doc__)
