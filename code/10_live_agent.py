"""
Step 8 - AETHERA LIVE PROTOTYPE: an autonomous agent that, every few hours,
  1. fetches real-time space weather from NOAA SWPC (solar wind, Kp, Dst, F10.7),
  2. fetches current orbits (TLEs) of low-orbit debris from CelesTrak,
  3. forecasts Kp 3-24 h ahead with the Random Forest from RQ1,
  4. ranks objects by the physics risk score from RQ2,
  5. writes a plain-language alert with the grounded agent from RQ3 (Groq),
  6. checks the alert and saves a report (Markdown + JSON + figure).

The workflow is a LangGraph graph:
  fetch_weather -> fetch_tles -> forecast -> risk -> assess -> retrieve -> write -> check -(rewrite once)-> save

Run once:                    python 10_live_agent.py
Run every 3 hours (Ctrl+C):  python 10_live_agent.py --every 3
Without the language model:  python 10_live_agent.py --no-llm      (template alert, no Groq key needed)

Needs: data/clean/omni_hourly.csv (to build the Random Forest the first time; then saved as results/rf_kp.joblib)
       GROQ_API_KEY set in the terminal (unless --no-llm)
Outputs: results/live/alert_<UTC time>.md, facts_<UTC time>.json, fig_live_<UTC time>.png

Rules we respect: CelesTrak allows ONE download per 2-hour update -> TLEs are cached for 2 hours and
errors are never retried in a loop. NOAA data are public and updated every minute.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from langgraph.graph import StateGraph, END
from config import CLEAN, RESULTS, HOLDOUT_2003

LIVE = RESULTS / "live"; LIVE.mkdir(parents=True, exist_ok=True)
TLE_CACHE = CLEAN.parent / "live_tle"; TLE_CACHE.mkdir(parents=True, exist_ok=True)
NO_LLM = "--no-llm" in sys.argv
HORIZONS = [3, 6, 12, 24]
FEATURES = ["B", "By", "Bz", "density", "speed", "pressure", "E_field", "dst", "f107", "kp"]
RF_ALARM = {3: 4.33, 6: 4.0, 12: 3.67, 24: 3.33}          # tuned alarm levels from RQ1
DEBRIS_GROUPS = ["cosmos-1408-debris", "fengyun-1c-debris", "cosmos-2251-debris", "iridium-33-debris"]
H_SCALE, H_REF, MU, RE = 60.0, 400.0, 398600.4418, 6378.137
HEADERS = {"User-Agent": "AETHERA-student-research-prototype"}

NOAA = {
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    # NOAA moved real-time solar wind to SOLAR-1/ACE (RTSW feeds) in 2026; the old "products/solar-wind"
    # files stopped updating in June 2025. Stale feeds are rejected automatically (see first_working).
    "plasma": ["https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
               "https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json"],
    "mag": ["https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
            "https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json"],
}

# ====================================================================== helpers
def get_table(url):
    """Download a NOAA JSON product. Works for both formats: list of objects, or list of lists with a header row."""
    r = requests.get(url, timeout=60, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], list):
        df = pd.DataFrame(data[1:], columns=data[0])
    else:
        df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
    for c in df.columns:
        if c not in ("time_tag", "time", "source", "reporting_schedule", "avg_begin_date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time")

def first_working(urls, max_age_h=6):
    """First source that works AND is up to date (newest value at most max_age_h hours old)."""
    notes = []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for u in urls:
        try:
            df = get_table(u)
            age = (now - df["time"].max()).total_seconds() / 3600 if len(df) else 1e9
            if age <= max_age_h:
                return df, u
            notes.append(f"{u.split('/')[-1]}: stale ({age:.0f} h old)")
        except Exception as e:
            notes.append(f"{u.split('/')[-1]}: {type(e).__name__}")
    raise RuntimeError("no up-to-date solar wind source: " + "; ".join(notes))

SW_HIST = CLEAN.parent / "live_sw"; SW_HIST.mkdir(parents=True, exist_ok=True)
def remember(df, name, cols):
    """Keep our own history of real-time data (the RTSW feeds only hold a short recent period)."""
    f = SW_HIST / f"{name}.csv"
    part = df[["time"] + cols].dropna(subset=cols, how="all")
    if f.exists():
        part = pd.concat([pd.read_csv(f, parse_dates=["time"]), part])
    part = part.drop_duplicates("time", keep="last").sort_values("time")
    part = part[part["time"] >= part["time"].max() - pd.Timedelta(days=10)]
    part.to_csv(f, index=False)
    return part

def g_level(kp):
    return int(min(5, max(0, np.floor(kp + 1e-6) - 4)))

def kp_multiplier(kp):
    return float(np.select([kp <= 3, kp <= 5, kp <= 7, kp <= 8], [1.0, 1.5, 3.0, 6.0], default=10.0))

def rf_features(df):
    """EXACTLY the same 55 features as in 06_rq1_forecast.py."""
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
    return pd.DataFrame(feat)

def load_or_build_rf():
    """The Random Forest of RQ1, rebuilt with the same data, settings and seed (saved after the first time)."""
    path = RESULTS / "rf_kp.joblib"
    if path.exists():
        return joblib.load(path)
    print("Building the Random Forest from RQ1 (first time only, about 1-2 minutes)...")
    df = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)[FEATURES].asfreq("h")
    t = df.index; n = len(df); H, L = max(HORIZONS), 24
    kp = df["kp"].to_numpy(np.float32); X = df.to_numpy(np.float32)
    Y = np.full((n, len(HORIZONS)), np.nan, np.float32)
    for j, h in enumerate(HORIZONS):
        Y[:-h, j] = kp[h:]
    bad = np.isnan(X).any(axis=1).astype(np.int32)
    valid = (np.convolve(bad, np.ones(L, np.int32))[:n] == 0) & ~np.isnan(Y).any(axis=1); valid[:L - 1] = False
    per = lambda a, b: (t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))
    ha, hb = pd.Timestamp(HOLDOUT_2003[0]), pd.Timestamp(HOLDOUT_2003[1])
    near = per(ha - pd.Timedelta(hours=L + H), hb + pd.Timedelta(hours=L))
    train = valid & per("2000-01-02", pd.Timestamp("2018-12-31 23:00") - pd.Timedelta(hours=H)) & ~near
    F = rf_features(df).to_numpy(np.float32)
    train &= ~np.isnan(F).any(axis=1)
    idx = np.flatnonzero(train); m = np.zeros(n, bool); m[idx[::3]] = True
    w = np.where((Y[m] >= 5).any(axis=1), 3.0, 1.0)
    rf = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5, max_features=0.5,
                               n_jobs=-1, random_state=42).fit(F[m], Y[m], sample_weight=w)
    joblib.dump(rf, path, compress=3)
    return rf

# ====================================================================== knowledge base + language model (as in RQ3)
KB = [
 "The Kp index measures global geomagnetic activity on a scale from 0 to 9. Values of 5 or more mean a geomagnetic storm.",
 "NOAA geomagnetic storm scale: G1 (minor) corresponds to Kp 5, G2 (moderate) to Kp 6, G3 (strong) to Kp 7, G4 (severe) to Kp 8 and G5 (extreme) to Kp 9.",
 "G1 minor storm: weak fluctuations in power grids are possible, there can be a minor impact on satellite operations, and aurora may be seen at high latitudes.",
 "G2 moderate storm: satellite ground control may need to correct satellite orientation, and changes in atmospheric drag can affect orbit predictions.",
 "G3 strong storm: surface charging can occur on satellite components, drag on low Earth orbit satellites can increase, and orientation problems may need correction.",
 "G4 severe storm: satellites may experience surface charging and tracking problems, and orientation problems may need correction.",
 "G5 extreme storm: satellites may experience extensive surface charging and problems with orientation, uplink and downlink communication, and tracking.",
 "During a geomagnetic storm the upper atmosphere heats up and expands. Air density at satellite altitudes increases, so drag slows satellites in low Earth orbit and they lose altitude faster.",
 "Objects at lower altitudes and objects with a large drag term (B-star) relative to their mass lose the most altitude during storms. Debris and rocket bodies cannot correct their orbits.",
 "Stronger drag makes orbit predictions less accurate, so tracking and collision-avoidance warnings become less certain during and after a storm.",
 "Forecasts of geomagnetic activity are most reliable a few hours ahead; forecasts a day ahead have low skill because the solar wind is measured only about one hour before it reaches Earth.",
 "Satellite operators can respond to storm warnings by monitoring orbits more often, expecting less accurate orbit predictions, and planning any orbit corrections after the storm.",
]
VEC = TfidfVectorizer(stop_words="english").fit(KB); KB_MAT = VEC.transform(KB)
def retrieve(q, k=4):
    sims = (KB_MAT @ VEC.transform([q]).T).toarray().ravel()
    return [KB[i] for i in np.argsort(-sims)[:k]]

RAG_SYS = ("You are AETHERA's space weather alert writer. Write a short alert (80 to 150 words) in simple English "
           "for non-experts. Use ONLY the numbers, dates and object names given in FACTS, and only the statements "
           "in REFERENCE TEXT. Do not invent numbers, dates, satellites or impacts. State the NOAA G-level exactly as "
           "given in FACTS and do not describe the storm with a stronger word than that level. Structure: (1) what is "
           "happening now, (2) what is expected in the next 24 hours, (3) which objects face the highest drag risk, "
           "(4) what this means in practice.")
_client, MODEL = None, None
def llm(system, user):
    global _client, MODEL
    if _client is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY first, or run with --no-llm.")
        _client = Groq(api_key=key)
        avail = {m.id for m in _client.models.list().data}
        MODEL = os.environ.get("GROQ_MODEL") or next(m for m in ("openai/gpt-oss-120b", "llama-3.3-70b-versatile",
                                                                 "openai/gpt-oss-20b") if m in avail)
    extra = {"reasoning_effort": "low"} if MODEL.startswith("openai/gpt-oss") else {}
    r = _client.chat.completions.create(model=MODEL, temperature=0, max_tokens=1500, **extra,
                                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()

def check_alert(text, facts, g_max):
    """Short version of the RQ3 checks: unsupported numbers, G-level and severity words."""
    text = re.sub(r"[‐-―−  ]", "-", text)
    vals = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(k))); walk(v)
        elif isinstance(x, list): [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool): vals.add(abs(float(x)))
        elif isinstance(x, str): vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", x))
    walk(facts)
    exact_ok = {0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0}       # scale thresholds: exact only
    clean = re.sub(r"(?m)^\s*[\(\[]?\d+[.)\]]\s+", " ", text)
    nums = [float(n) for n in re.findall(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)", clean)]
    bad = sorted({x for x in nums if x not in exact_ok and
                  not any(abs(x - v) <= max(0.15, 0.02 * v) or x == round(v) for v in vals)})
    neg = re.compile(r"\b(below|under|less than|not|no|without|threshold)\b|>=|\u2265|<", re.I)   # real negations only
    sev = {"minor": 1, "moderate": 2, "strong": 3, "severe": 4, "extreme": 5}
    over = []
    for sen in re.split(r"(?<=[.!?])\s+|\n+", text):
        if neg.search(sen):
            continue
        over += [int(g) for g in re.findall(r"\bG-?([1-5])\b", sen) if int(g) > g_max]
        over += [sev[w.lower()] for w in re.findall(r"\b(minor|moderate|strong|severe|extreme)\b(?=\s*(?:geomagnetic|storm|\)))", sen, re.I)
                 if sev[w.lower()] > g_max]
    probs = []
    if bad: probs.append(f"these numbers are not in FACTS: {bad}")
    if over: probs.append(f"the storm level or severity word is stronger than the data support (max G{g_max})")
    return probs

# ====================================================================== the agent (LangGraph)
class S(TypedDict, total=False):
    now: str; weather: dict; hourly: object; tles: object; forecast: dict; top: list
    facts: dict; g_max: int; quality: str; refs: list; draft: str; problems: list; rewrites: int; log: list

def fetch_weather(st):
    log = []
    kp = get_table(NOAA["kp"])[["time", "Kp"]].rename(columns={"Kp": "kp"})
    dst = get_table(NOAA["dst"])[["time", "dst"]]
    f107 = get_table(NOAA["f107"])
    f107 = f107[f107["frequency"] == 2800] if "frequency" in f107 else f107
    plasma, u1 = first_working(NOAA["plasma"]); mag, u2 = first_working(NOAA["mag"])
    for df_ in (plasma, mag):
        if "active" in df_ and (df_["active"] == True).any():
            df_.drop(df_.index[df_["active"] != True], inplace=True)    # the spacecraft NOAA is using right now
    plasma = plasma.rename(columns={"proton_speed": "speed", "proton_density": "density"})
    mag = mag.rename(columns={"bt": "B", "by_gsm": "By", "bz_gsm": "Bz"})
    plasma = remember(plasma, "plasma", ["speed", "density"])
    mag = remember(mag, "mag", ["B", "By", "Bz"])
    msg = (f"solar wind: {u1.split('/')[-1]} + {u2.split('/')[-1]}; saved history {plasma['time'].min()} to {plasma['time'].max()}")
    print("  " + msg); log.append(msg)
    # hourly averages, like OMNI
    sw = (plasma.set_index("time")[["speed", "density"]].resample("h").mean()
          .join(mag.set_index("time")[["B", "By", "Bz"]].resample("h").mean(), how="outer"))
    hourly = sw.copy()
    end = hourly.index.max()
    full = pd.date_range(end - pd.Timedelta(days=3), end, freq="h")      # always 3 days, even on the first run
    hourly = hourly.reindex(full)
    d = dst.set_index("time")["dst"].resample("h").mean().reindex(full)
    k = kp.set_index("time")["kp"].resample("h").ffill().reindex(full)
    # Kp is published every 3 h and real-time Dst often lags by a day or more: carry the last value forward
    log.append(f"Kp lag {(end - kp['time'].max()).total_seconds()/3600:.0f} h, Dst lag {(end - dst['time'].max()).total_seconds()/3600:.0f} h (last values carried forward)")
    hourly["dst"] = d.ffill(limit=72)
    hourly["kp"] = k.ffill(limit=6)
    hourly["f107"] = float(f107.sort_values("time")["flux"].dropna().iloc[-1])
    hourly["pressure"] = 1.6726e-6 * hourly["density"] * hourly["speed"] ** 2       # nPa
    hourly["E_field"] = -hourly["speed"] * hourly["Bz"] * 1e-3                        # mV/m
    hourly = hourly[FEATURES].interpolate(limit=3, limit_area="inside")
    swcols = ["B", "By", "Bz", "density", "speed", "pressure", "E_field"]
    have = hourly[swcols].dropna()
    if have.empty:
        raise RuntimeError("no complete hour of real-time solar wind yet")
    now = have.index[-1]
    window = hourly.loc[now - pd.Timedelta(hours=35):now].copy()
    n_sw = int(window.loc[now - pd.Timedelta(hours=23):now, swcols].notna().all(axis=1).sum())
    quality = "full"
    if n_sw < 24:
        # first runs: fewer than 24 h of saved solar wind -> fill older hours with the earliest value (clearly flagged)
        window[swcols] = window[swcols].bfill()
        quality = f"reduced (only {n_sw} of 24 hours of real-time solar wind available)"
    window[["kp", "dst", "f107"]] = window[["kp", "dst", "f107"]].ffill().bfill()
    if window.loc[now - pd.Timedelta(hours=23):now].isna().any(axis=None):
        raise RuntimeError("the last 24 hours still have gaps after filling")
    msg = f"latest complete hour: {now} UTC; data quality: {quality}"
    print("  " + msg); log.append(msg)
    return {"hourly": window, "now": str(now), "log": log, "quality": quality}

def fetch_tles(st):
    frames, log = [], st["log"]
    for grp in DEBRIS_GROUPS:
        f = TLE_CACHE / f"{grp}.csv"
        if not (f.exists() and time.time() - f.stat().st_mtime < 2 * 3600):    # CelesTrak: max once per 2 h
            try:
                r = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?GROUP={grp}&FORMAT=csv",
                                 timeout=60, headers=HEADERS)
                if r.status_code == 200 and r.text.startswith("OBJECT_NAME"):
                    f.write_text(r.text)
                else:
                    log.append(f"CelesTrak {grp}: HTTP {r.status_code}, using the saved copy if any")
            except Exception as e:
                log.append(f"CelesTrak {grp}: {type(e).__name__}, using the saved copy if any")
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError("no TLE data available (CelesTrak unreachable and no saved copy)")
    t = pd.concat(frames, ignore_index=True).drop_duplicates("NORAD_CAT_ID")
    n = t["MEAN_MOTION"].astype(float) * 2 * np.pi / 86400
    t["alt_km"] = (MU / n ** 2) ** (1 / 3) - RE
    t = t[(t["alt_km"] > 200) & (t["alt_km"] < 600) & (t["ECCENTRICITY"].astype(float) < 0.01)]
    log.append(f"{len(t)} low-orbit debris objects from CelesTrak")
    return {"tles": t, "log": log}

def forecast(st):
    rf = load_or_build_rf()
    F = rf_features(st["hourly"]).iloc[[-1]]
    if F.isna().any(axis=None):
        raise RuntimeError("the last 24 hours have gaps; cannot forecast now")
    p = np.clip(rf.predict(F.to_numpy(np.float32))[0], 0, 9)
    return {"forecast": {f"{h}h": round(float(v), 1) for h, v in zip(HORIZONS, p)}}

def risk(st):
    t = st["tles"].copy()
    kpf = max(st["forecast"].values())
    b = t["BSTAR"].astype(float).clip(lower=1e-6)
    t["R"] = b * np.exp(-(t["alt_km"] - H_REF) / H_SCALE) * kp_multiplier(kpf)
    top = t.nlargest(5, "R")
    return {"top": [{"name": str(r.OBJECT_NAME).strip(), "norad_id": int(r.NORAD_CAT_ID), "altitude_km": int(round(r.alt_km))}
                    for r in top.itertuples()]}

def assess(st):
    h = st["hourly"]; last = h.iloc[-1]; fc = st["forecast"]
    kp_now = round(float(last["kp"]), 1); kp24 = round(float(h["kp"].iloc[-24:].max()), 1)
    expected = [k for k, v in fc.items() if v >= RF_ALARM[int(k[:-1])]]
    g_max = max(g_level(kp_now), g_level(kp24), g_level(max(fc.values())))
    facts = {"issue_time_utc": pd.Timestamp(st["now"]).strftime("%Y-%m-%d %H:%M"),
             "kp_now": kp_now, "noaa_g_level_now": f"G{g_level(kp_now)}" if g_level(kp_now) else "none (below storm level)",
             "kp_max_last_24h": kp24, "solar_wind_speed_km_s": int(round(last["speed"])),
             "imf_bz_nT": round(float(last["Bz"]), 1), "dst_nT": int(round(last["dst"])),
             "kp_forecast": fc, "storm_level_expected_within": expected or "not expected in the next 24 hours",
             "forecast_note": "Forecasts more than 12 hours ahead have low skill.",
             "highest_drag_risk_objects": st["top"]}
    if st.get("quality", "full") != "full":
        facts["data_quality_note"] = ("The forecast uses less than 24 hours of real-time solar wind, "
                                      "so it is less reliable than usual.")
    return {"facts": facts, "g_max": g_max, "rewrites": 0}

def retrieve_node(st):
    f = st["facts"]
    return {"refs": retrieve(f"Kp {f['kp_now']} {f['noaa_g_level_now']} geomagnetic storm satellite drag forecast")}

def write(st):
    f = st["facts"]
    if NO_LLM:
        obj = ", ".join(f"{o['name']} ({o['altitude_km']} km)" for o in f["highest_drag_risk_objects"])
        return {"draft": (f"At {f['issue_time_utc']} UTC the Kp index is {f['kp_now']} (NOAA level: {f['noaa_g_level_now']}). "
                          f"Forecast Kp: {f['kp_forecast']['3h']} in 3 h, {f['kp_forecast']['12h']} in 12 h and "
                          f"{f['kp_forecast']['24h']} in 24 h; forecasts beyond 12 hours have low skill. "
                          f"Objects with the highest drag risk: {obj}.")}
    user = "FACTS:" + json.dumps(f, indent=1) + "\nREFERENCE TEXT:\n- " + "\n- ".join(st["refs"])
    if st.get("problems"):
        user += "\nYOUR PREVIOUS ALERT:\n" + st["draft"] + "\nPROBLEMS:\n- " + "\n- ".join(st["problems"]) + "\nRewrite it and fix every problem."
    return {"draft": llm(RAG_SYS, user)}

def check(st):
    return {"problems": check_alert(st["draft"], st["facts"], st["g_max"])}

def mark(st):
    return {"rewrites": st.get("rewrites", 0) + 1}

def save(st):
    stamp = pd.Timestamp(st["now"]).strftime("%Y%m%d_%H%M")
    f = st["facts"]
    (LIVE / f"facts_{stamp}.json").write_text(json.dumps({**f, "alert": st["draft"], "model": MODEL or "template",
                                                          "rewrites": st.get("rewrites", 0),
                                                          "remaining_problems": st.get("problems", []),
                                                          "log": st["log"]}, indent=1))
    rows = "\n".join(f"| {i+1} | {o['name']} | {o['norad_id']} | {o['altitude_km']} |" for i, o in enumerate(f["highest_drag_risk_objects"]))
    md = (f"# AETHERA live alert - {f['issue_time_utc']} UTC\n\n{st['draft']}\n\n## Data used\n\n"
          f"- Kp now: {f['kp_now']} ({f['noaa_g_level_now']}); max last 24 h: {f['kp_max_last_24h']}\n"
          f"- Solar wind {f['solar_wind_speed_km_s']} km/s, Bz {f['imf_bz_nT']} nT, Dst {f['dst_nT']} nT\n"
          f"- Kp forecast (Random Forest): {f['kp_forecast']}\n"
          f"- Data quality: {st.get('quality', 'full')}\n\n"
          f"## Highest drag risk (debris, physics score)\n\n| # | Object | NORAD | Altitude (km) |\n|---|---|---|---|\n{rows}\n\n"
          f"_Model: {MODEL or 'template (no LLM)'}; rewrites: {st.get('rewrites', 0)}; checks passed: {not st.get('problems')}_\n")
    (LIVE / f"alert_{stamp}.md").write_text(md, encoding="utf-8")
    h = st["hourly"].iloc[-72:]; now = pd.Timestamp(st["now"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.step(h.index, h["kp"], where="post", color="black", label="Observed Kp (NOAA, real time)")
    ax.plot([now + pd.Timedelta(hours=int(k[:-1])) for k in f["kp_forecast"]], list(f["kp_forecast"].values()),
            "o--", color="tab:green", label="AETHERA forecast (Random Forest)")
    ax.axhline(5, color="tab:red", ls="--", lw=0.8); ax.axvline(now, color="grey", ls=":")
    ax.set_ylim(0, 9); ax.set_ylabel("Kp"); ax.set_title(f"AETHERA live run, {f['issue_time_utc']} UTC"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(LIVE / f"fig_live_{stamp}.png", dpi=200); plt.close(fig)
    print(f"\n{md}\nSaved in {LIVE}")
    return {}

g = StateGraph(S)
for name, fn in (("fetch_weather", fetch_weather), ("fetch_tles", fetch_tles), ("forecast", forecast), ("risk", risk),
                 ("assess", assess), ("retrieve", retrieve_node), ("write", write), ("check", check), ("mark", mark), ("save", save)):
    g.add_node(name, fn)
g.set_entry_point("fetch_weather")
for a, b in (("fetch_weather", "fetch_tles"), ("fetch_tles", "forecast"), ("forecast", "risk"), ("risk", "assess"),
             ("assess", "retrieve"), ("retrieve", "write"), ("write", "check"), ("mark", "write"), ("save", END)):
    g.add_edge(a, b)
g.add_conditional_edges("check", lambda st: "rewrite" if st["problems"] and st.get("rewrites", 0) < 1 and not NO_LLM else "save",
                        {"rewrite": "mark", "save": "save"})
AGENT = g.compile()

def run_once():
    print(f"AETHERA live run at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    try:
        AGENT.invoke({"log": []})
    except Exception as e:
        print(f"Run failed: {type(e).__name__}: {e}  (it will be tried again at the next scheduled time)")

if __name__ == "__main__":
    every = None
    if "--every" in sys.argv:
        every = max(2.0, float(sys.argv[sys.argv.index("--every") + 1]))       # never more often than CelesTrak allows
    run_once()
    while every:
        print(f"Next run in {every} hours (Ctrl+C to stop)...")
        time.sleep(every * 3600)
        run_once()