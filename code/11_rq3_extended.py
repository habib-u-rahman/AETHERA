"""
Step 9 (journal revision) - RQ3 EXTENDED: 100 scenarios x 3 systems x 3 models.

Everything that evaluates or writes an alert (knowledge base, prompts, the Flesch/error
checker, the LangGraph agent, temperature=0) is copied VERBATIM from code/08_rq3_alerts.py.
Nothing there was changed. The only new logic is: (a) building 60 extra scenarios on top of
the original 40 (which are loaded unchanged from results/rq3_scenarios.json), and (b) looping
the three systems over three models instead of one, with per-model resumable caching.

Does NOT modify or overwrite anything under results/ (only results/ itself) or code/08_rq3_alerts.py.
All new output goes to results/step2/.

BEFORE RUNNING:
  GROQ_API_KEY must be set in the project's .env (loaded automatically, never printed).
Run:
  python 11_rq3_extended.py --dry-run     (no API calls, checks everything works)
  python 11_rq3_extended.py               (real run; respects Groq rate limits, safe to
                                            stop and re-run on later days - finished alerts
                                            are cached and skipped)
Outputs in results/step2/:
  rq3x_scenarios.json, rq3x_cache_<model>.json (one per model), rq3x_alerts.jsonl,
  rq3x_summary.txt
"""
import json
import os
import re
import sys
import time
from typing import TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from langgraph.graph import StateGraph, END
from scipy.stats import binomtest
from config import CLEAN, RESULTS, STORMS

load_dotenv()
DRY = "--dry-run" in sys.argv
STEP2 = RESULTS / "step2"
STEP2.mkdir(parents=True, exist_ok=True)

SEED_ORIGINAL = 7            # code/08's scenario seed (unchanged scenarios only use this indirectly, via the saved file)
SEED_EXTEND = 20261026       # fixed seed for the 60 NEW scenarios
N_STORM_TOTAL, N_QUIET_TOTAL = 70, 30
N_STORM_NEW, N_QUIET_NEW = 40, 20   # 30+10 original + 40+20 new = 70+30
RF_ALARM = {3: 4.33, 6: 4.0, 12: 3.67, 24: 3.33}   # tuned alarm levels from RQ1 (Random Forest) - unchanged from 08

MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]
SYSTEMS = ["Plain LLM", "RAG", "AETHERA agent"]
ORIGINAL_CACHE = RESULTS / "rq3_cache_openai_gpt-oss-120b.json"   # read-only, never written to

# ====================================================================== copied verbatim from code/08_rq3_alerts.py
def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)

def flesch(text):
    """Flesch reading ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)."""
    sents = max(1, len([x for x in re.split(r"[.!?]+", text) if x.strip()]))
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text)]
    if not words:
        return float("nan")
    return 206.835 - 1.015 * len(words) / sents - 84.6 * sum(syllables(w) for w in words) / len(words)

def g_level(kp):
    """NOAA G-scale from Kp: 5->G1, 6->G2, 7->G3, 8->G4, 9->G5 (0 = below storm level)."""
    return int(min(5, max(0, np.floor(kp + 1e-6) - 4)))

KB_BUILTIN = [
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
 "A southward interplanetary magnetic field (negative Bz) and a fast solar wind allow more solar wind energy to enter the magnetosphere and drive geomagnetic storms.",
 "The Dst index measures the strength of the ring current; more negative Dst values mean a stronger geomagnetic storm.",
 "Forecasts of geomagnetic activity are most reliable a few hours ahead; forecasts a day ahead have low skill because the solar wind is measured only about one hour before it reaches Earth.",
 "Satellite operators can respond to storm warnings by monitoring orbits more often, expecting less accurate orbit predictions, and planning any orbit corrections after the storm.",
]
def build_kb():
    chunks = list(KB_BUILTIN)
    if not DRY:
        try:
            import requests
            html = requests.get("https://www.swpc.noaa.gov/noaa-scales-explanation", timeout=20).text
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text); text = re.sub(r"\s+", " ", text)
            words = text.split()
            for i in range(0, len(words), 90):
                c = " ".join(words[i:i + 110])
                if "Kp" in c or "satellite" in c.lower():
                    chunks.append(c)
        except Exception:
            pass
    return chunks

KB = build_kb()
VEC = TfidfVectorizer(stop_words="english").fit(KB)
KB_MAT = VEC.transform(KB)
KB_NUMBERS = {float(x) for c in KB for x in re.findall(r"\d+(?:\.\d+)?", c)}

def retrieve(query, k=4):
    sims = (KB_MAT @ VEC.transform([query]).T).toarray().ravel()
    return [KB[i] for i in np.argsort(-sims)[:k]]

PLAIN_SYS = ("You are a space weather assistant. Write a short alert (80 to 150 words) in simple English "
             "for non-experts such as satellite operators and students, based on the data provided.")
RAG_SYS = ("You are AETHERA's space weather alert writer. Write a short alert (80 to 150 words) in simple English "
           "for non-experts. Use ONLY the numbers, dates and object names given in FACTS, and only the statements "
           "in REFERENCE TEXT. Do not invent numbers, dates, satellites or impacts. State the NOAA G-level exactly as "
           "given in FACTS. Structure: (1) what is happening now, (2) what is expected in the next 24 hours, "
           "(3) which objects face the highest drag risk, if listed, (4) what this means in practice.")

def rag_prompt(facts, refs, assessment=""):
    return ("FACTS:" + json.dumps(facts, indent=1) + "\nREFERENCE TEXT:\n- " + "\n- ".join(refs) +
            (f"\nASSESSMENT: {assessment}" if assessment else ""))

def rag_query(facts):
    g = facts["noaa_g_level_now"]
    return f"Kp {facts['kp_now']} {g} geomagnetic storm satellite drag low Earth orbit forecast"

def fact_numbers(facts):
    vals = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(k)))
                walk(v)
        elif isinstance(x, list): [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool): vals.add(float(x))
        elif isinstance(x, str): vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", x))
    walk(facts)
    return vals

def supported(x, vals):
    return any(abs(x - abs(v)) <= max(0.15, 0.02 * abs(v)) or (x == round(abs(v))) for v in vals)

SEVERITY = {"minor": 1, "moderate": 2, "strong": 3, "severe": 4, "extreme": 5}
NEGATION = re.compile(r"\b(below|under|less than|not|no|without|threshold)\b|≥|>=|<", re.I)

def norm(t):
    return re.sub(r"[‐-―−  ]", "-", t)

def check_alert(text, s):
    facts = s["facts"]
    text = norm(text)
    allowed = fact_numbers(facts)
    exact_ok = KB_NUMBERS | {1.0, 2.0}
    clean = re.sub(r"(?m)^\s*[\(\[]?\d+[.)\]]\s+", " ", text)
    clean = re.sub(r"\(\d\)", " ", clean)
    nums = [float(n) for n in re.findall(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)", clean)]
    unsupported = [x for x in nums if not supported(x, allowed) and x not in exact_ok]
    kp_facts = {facts["kp_now"], facts["kp_max_last_24h"], *facts["kp_forecast"].values()}
    kp_claims = [float(m) for m in re.findall(
        r"\bKp\b[^0-9.\n]{0,20}?(?<![A-Za-z])(\d+(?:\.\d+)?)(?!\s*-?\s*(?:h\b|hr|hours?|-hour|UTC|:))", text, flags=re.I)]
    kp_errors = [x for x in kp_claims if not any(abs(x - v) <= 0.35 for v in kp_facts) and x not in (5, 6, 7, 8, 9)]
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    g_claims, sev_claims = [], []
    for sen in sentences:
        if NEGATION.search(sen):
            continue
        g_claims += [int(g) for g in re.findall(r"\bG-?([1-5])\b", sen)]
        for m in re.finditer(r"\b(minor|moderate|strong|severe|extreme)\b(?=\s*(?:-\s*)?(?:geomagnetic|storm|level|\)))"
                             r"|\(\s*(minor|moderate|strong|severe|extreme)\s*\)", sen, flags=re.I):
            sev_claims.append(SEVERITY[(m.group(1) or m.group(2)).lower()])
    g_over = [g for g in g_claims if g > s["g_max_supported"]]
    sev_over = [v for v in sev_claims if v > s["g_max_supported"]]
    g_ok = bool(re.search(rf"\bG-?{s['g_now']}\b", text)) if s["type"] == "storm" else True
    known_ids = {str(o["norad_id"]) for o in facts.get("highest_drag_risk_objects", [])}
    fact_names = " ".join(norm(o["name"]).upper() for o in facts.get("highest_drag_risk_objects", []))
    invented = [i for i in re.findall(r"\b(\d{5})\b", text) if i not in known_ids and not supported(float(i), allowed)]
    for fam in set(re.findall(r"\b(COSMOS|STARLINK|IRIDIUM|FENGYUN|ONEWEB|SL-\d+|CZ-\d\w*)\b", text.upper())):
        if fam not in fact_names:
            invented.append(fam)
    problems = []
    if unsupported: problems.append(f"these numbers are not in FACTS or REFERENCE TEXT: {sorted(set(unsupported))}")
    if kp_errors: problems.append(f"these Kp values do not match FACTS: {sorted(set(kp_errors))}")
    if g_over: problems.append(f"the G-level is overstated; the highest supported level is G{s['g_max_supported']}")
    if sev_over: problems.append(f"a storm is described with a severity word stronger than G{s['g_max_supported']} "
                                 f"(minor=G1, moderate=G2, strong=G3, severe=G4, extreme=G5)")
    if not g_ok: problems.append(f"the current NOAA level must be stated as G{s['g_now']}")
    if invented: problems.append(f"these objects are not in FACTS: {sorted(set(invented))}")
    return {"n_numbers": len(nums), "unsupported_numbers": len(unsupported), "kp_errors": len(kp_errors),
            "g_correct": int(g_ok), "g_overstated": int(bool(g_over)), "severity_overstated": int(bool(sev_over)),
            "invented_objects": len(invented), "flesch": flesch(text), "words": len(text.split()), "problems": problems}

# ====================================================================== LLM call (Groq) - same retry/temperature logic as 08, parameterised by model
_client = None
def get_client():
    global _client
    if _client is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY in .env first.")
        _client = Groq(api_key=key)
    return _client

def llm(system, user, model):
    if DRY:
        part = user.split("FACTS:", 1)[1] if "FACTS:" in user else user.split("DATA:", 1)[-1]
        try:
            f = json.loads(part.split("\nREFERENCE TEXT:")[0])
        except Exception:
            f = {}
        return (f"Space weather alert for {f.get('issue_time_utc','now')} UTC. The Kp index is {f.get('kp_now','?')} "
                f"({f.get('noaa_g_level_now','?')}). Drag on low satellites may increase. Kp could reach 7.5 tomorrow.")
    client = get_client()
    reasoning = model.startswith("openai/gpt-oss") or model.startswith("qwen/")
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    for attempt in range(8):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=1500 if reasoning else 400, **extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            time.sleep(3)
            if text:
                return text
            print("   empty answer, retrying...")
        except Exception as e:
            name, msg = type(e).__name__, str(e)
            if name in ("NotFoundError", "AuthenticationError", "PermissionDeniedError", "BadRequestError"):
                raise SystemExit(f"Groq error that waiting cannot fix ({name}): {msg[:300]}")
            if "per day" in msg or "TPD" in msg or "RPD" in msg:
                raise SystemExit(f"Groq DAILY limit reached for {model}. Finished alerts are saved - run again tomorrow.")
            wait = 20 * (attempt + 1)
            print(f"   API problem ({name}); waiting {wait} s...")
            time.sleep(wait)
    raise SystemExit(f"Groq API kept failing for {model} - run again later.")

def plain_alert(s, model):
    return llm(PLAIN_SYS, "DATA:\n" + json.dumps(s["facts"], indent=1), model)

def rag_alert(s, model):
    return llm(RAG_SYS, rag_prompt(s["facts"], retrieve(rag_query(s["facts"]))), model)

# ---------------------------------------------------------------- AETHERA agent (LangGraph) - unchanged graph/edges/logic
class AgentState(TypedDict, total=False):
    scenario: dict
    model: str
    assessment: str
    refs: list
    draft: str
    problems: list
    rewrites: int

def node_monitor(st):
    return {"rewrites": 0}

def node_assess(st):
    f, s = st["scenario"]["facts"], st["scenario"]
    lvl = f["noaa_g_level_now"]
    a = (f"Current level: {lvl}. Highest level supported by the data in the next 24 hours: "
         f"{'G'+str(s['g_max_supported']) if s['g_max_supported'] else 'below storm level'}.")
    if f.get("highest_drag_risk_objects"):
        a += " List the objects in highest_drag_risk_objects as the most exposed to drag."
    return {"assessment": a}

def node_retrieve(st):
    return {"refs": retrieve(rag_query(st["scenario"]["facts"]))}

def node_write(st):
    user = rag_prompt(st["scenario"]["facts"], st["refs"], st["assessment"])
    if st.get("problems"):
        user += ("\nYOUR PREVIOUS ALERT:\n" + st["draft"] + "\nIT HAD THESE PROBLEMS:\n- " +
                 "\n- ".join(st["problems"]) + "\nRewrite the alert and fix every problem.")
    return {"draft": llm(RAG_SYS, user, st["model"])}

def node_check(st):
    return {"problems": check_alert(st["draft"], st["scenario"])["problems"]}

def node_mark(st):
    return {"rewrites": st.get("rewrites", 0) + 1}

def route(st):
    return "rewrite" if st["problems"] and st.get("rewrites", 0) < 1 else "done"

g = StateGraph(AgentState)
for name, fn in (("monitor", node_monitor), ("assess", node_assess), ("retrieve", node_retrieve),
                 ("write", node_write), ("check", node_check), ("mark", node_mark)):
    g.add_node(name, fn)
g.set_entry_point("monitor")
g.add_edge("monitor", "assess"); g.add_edge("assess", "retrieve"); g.add_edge("retrieve", "write")
g.add_edge("write", "check")
g.add_conditional_edges("check", route, {"rewrite": "mark", "done": END})
g.add_edge("mark", "write")
AGENT = g.compile()

def agent_alert(s, model):
    out = AGENT.invoke({"scenario": s, "model": model})
    return out["draft"], out.get("rewrites", 0)

# ====================================================================== NEW: scenario pool + extension (same construction rules as 08)
def build_pools():
    omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
    fc = pd.read_csv(RESULTS / "rq1_forecasts.csv", parse_dates=["time"])
    fc = fc[fc.test_set == "test"].set_index("time")
    rq2 = pd.read_csv(RESULTS / "rq2_objects.csv") if (RESULTS / "rq2_objects.csv").exists() else None
    test = omni.loc["2022-01-02":"2024-12-30"]
    storm_times = []
    for day, gday in test[test["kp"] >= 5].groupby(test[test["kp"] >= 5].index.date):
        cand = [t for t in gday.index if t in fc.index]
        if cand:
            storm_times.append(cand[0])
    daily_max = test["kp"].resample("D").max()
    quiet_days = [d for d, v in daily_max.items() if v < 3 and (d + pd.Timedelta(hours=12)) in fc.index]
    return omni, fc, rq2, sorted(storm_times), quiet_days

def build_one_scenario(t, idx, omni, fc, rq2, onset_dates):
    t = pd.Timestamp(t)
    row, f = omni.loc[t], fc.loc[t]
    fcast = {f"{h}h": round(float(f[f"RandomForest_{h}h"]), 1) for h in (3, 6, 12, 24)}
    expected = [f"{h}h" for h in (3, 6, 12, 24) if f[f"RandomForest_{h}h"] >= RF_ALARM[h]]
    kp_now = round(float(row["kp"]), 1)
    kp24 = round(float(omni.loc[t - pd.Timedelta(hours=23):t, "kp"].max()), 1)
    facts = {
        "issue_time_utc": t.strftime("%Y-%m-%d %H:%M"),
        "kp_now": kp_now, "noaa_g_level_now": f"G{g_level(kp_now)}" if g_level(kp_now) else "none (below storm level)",
        "kp_max_last_24h": kp24,
        "solar_wind_speed_km_s": int(round(row["speed"])) if pd.notna(row["speed"]) else None,
        "imf_bz_nT": round(float(row["Bz"]), 1) if pd.notna(row["Bz"]) else None,
        "dst_nT": int(round(row["dst"])) if pd.notna(row["dst"]) else None,
        "kp_forecast": fcast,
        "storm_level_expected_within": expected or "not expected in the next 24 hours",
        "forecast_note": "Forecasts more than 12 hours ahead have low skill.",
    }
    storm_key = onset_dates.get(str(t.date()))
    if rq2 is not None and storm_key:
        top = rq2[(rq2.storm == storm_key) & rq2.passive].nlargest(5, "score_physics")
        facts["highest_drag_risk_objects"] = [
            {"name": str(r["name"]).strip(), "norad_id": int(r["norad_id"]), "altitude_km": int(round(r["alt_km"]))}
            for _, r in top.iterrows()]
    return {"id": f"S{idx:02d}" if idx < 100 else f"S{idx}", "type": "storm" if kp_now >= 5 else "quiet",
            "g_now": g_level(kp_now),
            "g_max_supported": max(g_level(kp_now), g_level(kp24), g_level(max(fcast.values()))),
            "facts": facts}

def build_100_scenarios():
    orig_path = RESULTS / "rq3_scenarios.json"
    if not orig_path.exists():
        raise SystemExit(f"Missing {orig_path} - run code/08_rq3_alerts.py (or at least its scenario step) first.")
    original = json.loads(orig_path.read_text())
    if len(original) != 40:
        raise SystemExit(f"Expected 40 original scenarios, found {len(original)}.")
    used_times = {pd.Timestamp(s["facts"]["issue_time_utc"]) for s in original}
    used_quiet_days = {t.normalize() for t in used_times if (t.hour == 12)}

    omni, fc, rq2, storm_times, quiet_days = build_pools()
    onset_dates = {str(pd.Timestamp(v["start"]).date()): k for k, v in STORMS.items()}

    remaining_storms = [t for t in storm_times if t not in used_times]
    if len(remaining_storms) < N_STORM_NEW:
        raise SystemExit(f"Only {len(remaining_storms)} unused storm hours available, need {N_STORM_NEW}.")
    idx = np.linspace(0, len(remaining_storms) - 1, N_STORM_NEW).astype(int)
    new_storm_times = sorted({remaining_storms[i] for i in idx})
    while len(new_storm_times) < N_STORM_NEW:   # linspace can collide on duplicates; top up from the front
        for t in remaining_storms:
            if t not in new_storm_times:
                new_storm_times.append(t)
            if len(new_storm_times) == N_STORM_NEW:
                break
    new_storm_times = sorted(new_storm_times)[:N_STORM_NEW]

    remaining_quiet = [d for d in quiet_days if d.normalize() not in used_quiet_days]
    rng = np.random.default_rng(SEED_EXTEND)
    picked_days = rng.choice(remaining_quiet, N_QUIET_NEW, replace=False)
    new_quiet_times = sorted(pd.Timestamp(d) + pd.Timedelta(hours=12) for d in picked_days)

    new_scenarios = []
    for i, t in enumerate(new_storm_times + new_quiet_times):
        new_scenarios.append(build_one_scenario(t, 41 + i, omni, fc, rq2, onset_dates))
    all_scen = original + new_scenarios
    assert len(all_scen) == 100
    assert sum(s["type"] == "storm" for s in all_scen) == N_STORM_TOTAL
    assert sum(s["type"] == "quiet" for s in all_scen) == N_QUIET_TOTAL
    return all_scen

# ====================================================================== run + evaluate, per model
def model_cache_path(model):
    return STEP2 / f"rq3x_cache_{model.replace('/', '_')}.json"

def load_cache(model):
    p = model_cache_path(model)
    cache = json.loads(p.read_text()) if p.exists() else {}
    if model == "openai/gpt-oss-120b" and ORIGINAL_CACHE.exists():
        orig = json.loads(ORIGINAL_CACHE.read_text())
        for k, v in orig.items():
            if k.startswith("_"):
                continue
            sid = k.split("|", 1)[0]
            if sid in {f"S{n:02d}" for n in range(1, 41)} and k not in cache:
                cache[k] = v   # reuse cached alerts for the original 40 scenarios - no new API calls
    return cache

def save_cache(model, cache):
    """Atomic write (tmp file + rename) so a kill mid-save can never corrupt the cache."""
    p = model_cache_path(model)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    os.replace(tmp, p)

FN = {"Plain LLM": plain_alert, "RAG": rag_alert, "AETHERA agent": agent_alert}

TIME_BUDGET_S = float(os.environ.get("RQ3X_TIME_BUDGET_S", 1200))   # exit cleanly before any external kill
START_TIME = time.time()
class TimeBudgetExceeded(Exception):
    pass

def run_model(model, scen):
    print(f"\n=== Model: {model} ===")
    cache = load_cache(model)
    rows = []
    try:
        for s in scen:
            for system in SYSTEMS:
                key = f"{s['id']}|{system}"
                if key not in cache:
                    if time.time() - START_TIME > TIME_BUDGET_S:
                        raise TimeBudgetExceeded()
                    print(f"{s['id']} ({s['type']}) - {system} [{model}]...")
                    res = FN[system](s, model)
                    text, rew = (res if isinstance(res, tuple) else (res, 0))
                    cache[key] = {"text": text, "rewrites": rew}
                    if not DRY:
                        save_cache(model, cache)
                c = cache[key]
                m = check_alert(c["text"], s)
                rows.append({"scenario": s["id"], "type": s["type"], "issue_time": s["facts"]["issue_time_utc"],
                             "model": model, "system": system, "rewrites": c["rewrites"],
                             **{k: v for k, v in m.items() if k != "problems"},
                             "problems": " | ".join(m["problems"]), "alert": c["text"], "facts": s["facts"]})
    except SystemExit as e:
        print(f"STOPPED for {model}: {e}")
        save_cache(model, cache)
    except TimeBudgetExceeded:
        print(f"Time budget ({TIME_BUDGET_S:.0f}s) reached while on {model} - stopping cleanly. Re-run to continue.")
        save_cache(model, cache)
        raise
    return rows

def main():
    scen = build_100_scenarios()
    (STEP2 / "rq3x_scenarios.json").write_text(json.dumps(scen, indent=1))
    print(f"{len(scen)} scenarios ({sum(s['type']=='storm' for s in scen)} storm, "
          f"{sum(s['type']=='quiet' for s in scen)} quiet). KB: {len(KB)} passages. Models: {MODELS}"
          f"{'  [DRY RUN]' if DRY else ''}")

    all_rows = []
    budget_hit = False
    for model in MODELS:
        try:
            all_rows += run_model(model, scen)
        except TimeBudgetExceeded:
            budget_hit = True
            break
    df = pd.DataFrame(all_rows)
    if budget_hit:
        print("Writing partial outputs for what's completed so far; re-run this script to continue.")

    with open(STEP2 / "rq3x_alerts.jsonl", "w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            rec = {"scenario_id": r["scenario"], "storm_or_quiet": r["type"], "issue_time_utc": r["issue_time"],
                   "model": r["model"], "system": r["system"], "facts": r["facts"], "alert_text": r["alert"],
                   "rewrites_used": int(r["rewrites"]),
                   "error_flags": {"n_numbers": int(r["n_numbers"]), "unsupported_numbers": int(r["unsupported_numbers"]),
                                   "kp_errors": int(r["kp_errors"]), "g_correct": int(r["g_correct"]),
                                   "g_overstated": int(r["g_overstated"]), "severity_overstated": int(r["severity_overstated"]),
                                   "invented_objects": int(r["invented_objects"]), "problems": r["problems"]},
                   "flesch_reading_ease": float(r["flesch"]), "words": int(r["words"])}
            fh.write(json.dumps(rec) + "\n")

    def any_err(sub):
        return ((sub.unsupported_numbers > 0) | (sub.kp_errors > 0) | (sub.g_overstated > 0) | (sub.severity_overstated > 0)
                | (sub.invented_objects > 0) | ((sub.type == "storm") & (sub.g_correct == 0))).astype(int)

    lines = ["AETHERA - RQ3 EXTENDED results (100 scenarios x 3 systems x 3 models)",
             f"Models: {MODELS}", f"Scenarios: {len(scen)} ({sum(s['type']=='storm' for s in scen)} storm, "
             f"{sum(s['type']=='quiet' for s in scen)} quiet){'  [DRY RUN - fake text]' if DRY else ''}", ""]
    for model in MODELS:
        sub_m = df[df.model == model]
        if sub_m.empty:
            lines.append(f"=== {model}: no completed alerts yet ===\n"); continue
        lines.append(f"=== {model} ===")
        for system in SYSTEMS:
            sub = sub_m[sub_m.system == system]
            if sub.empty:
                lines.append(f"  {system:16s} no completed alerts yet"); continue
            err = any_err(sub)
            lines.append(f"  {system:16s} n={len(sub):3d}  error%={100*err.mean():5.1f}  "
                         f"severity_overstated%={100*sub.severity_overstated.mean():5.1f}  "
                         f"flesch={sub.flesch.mean():5.2f}  words={sub.words.mean():5.1f}  "
                         f"rewrites%={100*(sub.rewrites>0).mean():5.1f}")
        E = {}
        for system in SYSTEMS:
            sub = sub_m[sub_m.system == system].set_index("scenario")
            if len(sub):
                E[system] = any_err(sub)
        for a, b in (("Plain LLM", "RAG"), ("Plain LLM", "AETHERA agent"), ("RAG", "AETHERA agent")):
            if a in E and b in E:
                x, y = E[a], E[b].reindex(E[a].index)
                common = x.index.intersection(y.dropna().index)
                x, y = x.loc[common], y.loc[common]
                n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
                if n01 + n10:
                    p = binomtest(n10, n01 + n10, 0.5).pvalue
                    lines.append(f"  McNemar exact, any error, {a} vs {b}: only {a} wrong={n10}, only {b} wrong={n01}, p={p:.4f}")
        lines.append("")
    summary = "\n".join(lines)
    (STEP2 / "rq3x_summary.txt").write_text(summary)
    print("\n" + summary)
    print("Files saved in", STEP2)

if __name__ == "__main__":
    main()
