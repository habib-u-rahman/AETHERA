"""
Step 6 - RQ3: Are alerts from a RAG-based agent more accurate and readable than a plain LLM?

Three systems write a short space-weather alert for the SAME situations, using the SAME model:
  1. Plain LLM      - gets the facts and is asked to write an alert.
  2. RAG            - also gets reference text retrieved from a small knowledge base
                      (NOAA geomagnetic storm scale, Kp, satellite drag), plus strict rules.
  3. AETHERA agent  - a LangGraph agent: monitor -> assess -> retrieve -> write -> check,
                      and if the check finds a problem, it rewrites ONCE with feedback.

Situations ("scenarios"): 30 storm hours and 10 quiet hours from the 2022-2024 test period,
built from YOUR data (omni_hourly.csv, rq1_forecasts.csv, rq2_objects.csv).

Automatic evaluation of every alert:
  * unsupported numbers  - numbers in the alert that are not in the facts or reference text
  * Kp errors            - a Kp value that does not match any Kp in the facts
  * G-level correct      - storm alerts must state the right NOAA G-level (G1-G5)
  * G-level overstated   - alert claims a higher G-level than the data supports
  * invented objects     - satellite names/IDs not in the facts
  * Flesch reading ease  - higher = easier to read (60-70 = plain English)

BEFORE RUNNING:
  1. pip install groq langgraph
  2. Get a free key at https://console.groq.com/keys and set it in the terminal:
        $env:GROQ_API_KEY="gsk_..."          (PowerShell)
Run:
  python 08_rq3_alerts.py --dry-run     (no API calls, checks everything works, 1 minute)
  python 08_rq3_alerts.py               (real run, about 30-60 minutes because of free-tier limits)
If it stops because of the daily limit, run it again later: finished alerts are saved and skipped.

Outputs in results/:
  rq3_scenarios.json, rq3_alerts.csv, rq3_summary.txt, fig_rq3.png, rq3_cache.json
"""
import json
import os
import re
import sys
import time
from typing import TypedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from langgraph.graph import StateGraph, END
from config import CLEAN, RESULTS, STORMS

DRY = "--dry-run" in sys.argv
# The script picks the first model in this list that your Groq key can use.
# You can force one with:  $env:GROQ_MODEL="openai/gpt-oss-120b"
MODEL_PREFERENCE = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                    "llama-3.1-8b-instant", "qwen/qwen3.8-27b"]
MODEL = os.environ.get("GROQ_MODEL", "") or "(chosen at start)"
SEED = 7
N_STORM, N_QUIET = 30, 10
RF_ALARM = {3: 4.33, 6: 4.0, 12: 3.67, 24: 3.33}   # tuned alarm levels from RQ1 (Random Forest)

# ---------------------------------------------------------------- Flesch reading ease (built in)
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

# ====================================================================== 1. scenarios
def g_level(kp):
    """NOAA G-scale from Kp: 5->G1, 6->G2, 7->G3, 8->G4, 9->G5 (0 = below storm level)."""
    return int(min(5, max(0, np.floor(kp + 1e-6) - 4)))

def build_scenarios():
    omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)
    fc = pd.read_csv(RESULTS / "rq1_forecasts.csv", parse_dates=["time"])
    fc = fc[fc.test_set == "test"].set_index("time")
    rq2 = pd.read_csv(RESULTS / "rq2_objects.csv") if (RESULTS / "rq2_objects.csv").exists() else None
    rng = np.random.default_rng(SEED)

    test = omni.loc["2022-01-02":"2024-12-30"]
    storm_times = []
    for day, g in test[test["kp"] >= 5].groupby(test[test["kp"] >= 5].index.date):
        cand = [t for t in g.index if t in fc.index]
        if cand:
            storm_times.append(cand[0])
    must = [t for t in storm_times if str(t.date()) in ("2022-02-03", "2024-05-10")]
    rest = [t for t in storm_times if t not in must]
    pick = list(np.array(rest)[np.linspace(0, len(rest) - 1, N_STORM - len(must)).astype(int)]) if rest else []
    storm_times = sorted(set(must + pick))
    daily_max = test["kp"].resample("D").max()
    quiet_days = [d for d, v in daily_max.items() if v < 3 and (d + pd.Timedelta(hours=12)) in fc.index]
    quiet_times = [d + pd.Timedelta(hours=12) for d in rng.choice(quiet_days, N_QUIET, replace=False)]

    onset_dates = {str(pd.Timestamp(v["start"]).date()): k for k, v in STORMS.items()}
    scen = []
    for i, t in enumerate(sorted(storm_times) + sorted(quiet_times)):
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
        scen.append({"id": f"S{i+1:02d}", "type": "storm" if kp_now >= 5 else "quiet",
                     "g_now": g_level(kp_now),
                     "g_max_supported": max(g_level(kp_now), g_level(kp24), g_level(max(fcast.values()))),
                     "facts": facts})
    return scen

# ====================================================================== 2. knowledge base (RAG)
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
        try:   # optional: add official NOAA page text if reachable
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

# ====================================================================== 3. LLM call (Groq)
_client = None
def get_client():
    global _client, MODEL
    if _client is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY first (see the top of this file).")
        _client = Groq(api_key=key)
        available = sorted(m.id for m in _client.models.list().data)
        if "--list-models" in sys.argv:
            print("Models your key can use:\n  " + "\n  ".join(available)); raise SystemExit
        wanted = os.environ.get("GROQ_MODEL")
        if wanted:
            if wanted not in available:
                raise SystemExit(f"GROQ_MODEL={wanted} is not available. Available: {available}")
            MODEL = wanted
        else:
            MODEL = next((m for m in MODEL_PREFERENCE if m in available), None)
            if MODEL is None:
                raise SystemExit(f"None of the preferred models is available. Available: {available}\n"
                                 "Pick one and set it, e.g.  $env:GROQ_MODEL=\"<model id>\"")
        print(f"Using Groq model: {MODEL}")
    return _client

def llm(system, user):
    if DRY:   # fake answer for testing the pipeline without an API key
        part = user.split("FACTS:", 1)[1] if "FACTS:" in user else user.split("DATA:", 1)[-1]
        try:
            f = json.loads(part.split("\nREFERENCE TEXT:")[0])
        except Exception:
            f = {}
        return (f"Space weather alert for {f.get('issue_time_utc','now')} UTC. The Kp index is {f.get('kp_now','?')} "
                f"({f.get('noaa_g_level_now','?')}). Drag on low satellites may increase. Kp could reach 7.5 tomorrow.")
    client = get_client()
    reasoning = MODEL.startswith("openai/gpt-oss") or MODEL.startswith("qwen/")
    extra = {"reasoning_effort": "low"} if MODEL.startswith("openai/gpt-oss") else {}
    for attempt in range(8):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=1500 if reasoning else 400, **extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some models show their thinking
            time.sleep(3)                          # stay under the per-minute limits
            if text:
                return text
            print("   empty answer, retrying...")
        except Exception as e:
            name, msg = type(e).__name__, str(e)
            if name in ("NotFoundError", "AuthenticationError", "PermissionDeniedError", "BadRequestError"):
                raise SystemExit(f"Groq error that waiting cannot fix ({name}): {msg[:300]}")
            if "per day" in msg or "TPD" in msg or "RPD" in msg:
                raise SystemExit("Groq DAILY limit reached. Your finished alerts are saved - run again tomorrow.")
            wait = 20 * (attempt + 1)
            print(f"   API problem ({name}); waiting {wait} s...")
            time.sleep(wait)
    raise SystemExit("Groq API kept failing - run again later.")

# ====================================================================== 4. the three systems
PLAIN_SYS = ("You are a space weather assistant. Write a short alert (80 to 150 words) in simple English "
             "for non-experts such as satellite operators and students, based on the data provided.")
RAG_SYS = ("You are AETHERA's space weather alert writer. Write a short alert (80 to 150 words) in simple English "
           "for non-experts. Use ONLY the numbers, dates and object names given in FACTS, and only the statements "
           "in REFERENCE TEXT. Do not invent numbers, dates, satellites or impacts. State the NOAA G-level exactly as "
           "given in FACTS. Structure: (1) what is happening now, (2) what is expected in the next 24 hours, "
           "(3) which objects face the highest drag risk, if listed, (4) what this means in practice.")

def plain_alert(s):
    return llm(PLAIN_SYS, "DATA:\n" + json.dumps(s["facts"], indent=1))

def rag_prompt(facts, refs, assessment=""):
    return ("FACTS:" + json.dumps(facts, indent=1) + "\nREFERENCE TEXT:\n- " + "\n- ".join(refs) +
            (f"\nASSESSMENT: {assessment}" if assessment else ""))

def rag_query(facts):
    g = facts["noaa_g_level_now"]
    return f"Kp {facts['kp_now']} {g} geomagnetic storm satellite drag low Earth orbit forecast"

def rag_alert(s):
    return llm(RAG_SYS, rag_prompt(s["facts"], retrieve(rag_query(s["facts"]))))

# ---------------------------------------------------------------- checks (used by evaluation AND agent)
def fact_numbers(facts):
    vals = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(k)))   # e.g. "24h" -> 24
                walk(v)
        elif isinstance(x, list): [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool): vals.add(float(x))
        elif isinstance(x, str): vals.update(float(n) for n in re.findall(r"\d+(?:\.\d+)?", x))
    walk(facts)
    return vals

def supported(x, vals):
    # numbers are read without their sign (e.g. "Bz -12.3"), so compare sizes
    return any(abs(x - abs(v)) <= max(0.15, 0.02 * abs(v)) or (x == round(abs(v))) for v in vals)

SEVERITY = {"minor": 1, "moderate": 2, "strong": 3, "severe": 4, "extreme": 5}
NEGATION = re.compile(r"\b(below|under|less than|not|no|without|threshold)\b|\u2265|>=|<", re.I)   # real negations only

def norm(t):
    """Unify the many hyphen/space characters language models use."""
    return re.sub(r"[\u2010-\u2015\u2212\u00a0\u202f]", "-", t)

def check_alert(text, s):
    facts = s["facts"]
    text = norm(text)
    allowed = fact_numbers(facts)                                  # facts: small rounding allowed
    exact_ok = KB_NUMBERS | {1.0, 2.0}                              # reference-text numbers (e.g. Kp 5..9): exact only
    clean = re.sub(r"(?m)^\s*[\(\[]?\d+[.)\]]\s+", " ", text)        # list numbering "1." "2)" at line start
    clean = re.sub(r"\(\d\)", " ", clean)                              # inline "(1)" markers
    nums = [float(n) for n in re.findall(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)", clean)]
    unsupported = [x for x in nums if not supported(x, allowed) and x not in exact_ok]
    # Kp values: a number after "Kp" that is NOT a time span ("Kp in the past 24 hours", "Kp in 3 h")
    kp_facts = {facts["kp_now"], facts["kp_max_last_24h"], *facts["kp_forecast"].values()}
    kp_claims = [float(m) for m in re.findall(
        r"\bKp\b[^0-9.\n]{0,20}?(?<![A-Za-z])(\d+(?:\.\d+)?)(?!\s*-?\s*(?:h\b|hr|hours?|-hour|UTC|:))", text, flags=re.I)]
    kp_errors = [x for x in kp_claims if not any(abs(x - v) <= 0.35 for v in kp_facts) and x not in (5, 6, 7, 8, 9)]
    # G-levels and severity words count as CLAIMS only in sentences without negation/threshold words
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
    # objects: 5-digit catalogue numbers must be in FACTS; named debris families must appear in FACTS
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

# ---------------------------------------------------------------- AETHERA agent (LangGraph)
class AgentState(TypedDict, total=False):
    scenario: dict
    assessment: str
    refs: list
    draft: str
    problems: list
    rewrites: int

def node_monitor(st):          # read the latest facts (here: the scenario)
    return {"rewrites": 0}

def node_assess(st):           # rule-based severity, NOT left to the language model
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
    return {"draft": llm(RAG_SYS, user)}

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

def agent_alert(s):
    out = AGENT.invoke({"scenario": s})
    return out["draft"], out.get("rewrites", 0)

# ====================================================================== 5. run + evaluate
RESCORE = "--rescore" in sys.argv          # re-check saved alerts without calling Groq
if RESCORE:
    caches = sorted(RESULTS.glob("rq3_cache_*.json"), key=lambda p: p.stat().st_mtime)
    if not caches:
        raise SystemExit("No saved alerts found (results/rq3_cache_*.json). Run without --rescore first.")
    MODEL = json.loads(caches[-1].read_text()).get("_model") or caches[-1].stem.replace("rq3_cache_", "").replace("openai_", "openai/")
    print(f"Re-scoring saved alerts from {caches[-1].name} (no API calls).")
elif not DRY or "--list-models" in sys.argv:
    get_client()
scen = build_scenarios()
(RESULTS / "rq3_scenarios.json").write_text(json.dumps(scen, indent=1))
print(f"{len(scen)} scenarios ({sum(s['type']=='storm' for s in scen)} storm, {sum(s['type']=='quiet' for s in scen)} quiet). "
      f"Knowledge base: {len(KB)} passages. Model: {MODEL}{'  [DRY RUN]' if DRY else ''}")
CACHE = RESULTS / f"rq3_cache_{MODEL.replace('/', '_')}.json"   # one cache per model, never mixed
cache = {} if DRY or not CACHE.exists() else json.loads(CACHE.read_text())
rows = []
for s in scen:
    for system, fn in (("Plain LLM", plain_alert), ("RAG", rag_alert), ("AETHERA agent", agent_alert)):
        key = f"{s['id']}|{system}"
        if key not in cache:
            if RESCORE:
                raise SystemExit(f"Alert {key} is missing in the saved file - run without --rescore to finish it.")
            print(f"{s['id']} ({s['type']}, {s['facts']['issue_time_utc']}) - {system}...")
            res = fn(s)
            text, rew = (res if isinstance(res, tuple) else (res, 0))
            cache[key] = {"text": text, "rewrites": rew}
            cache["_model"] = MODEL
            if not DRY:
                CACHE.write_text(json.dumps(cache, indent=1))
        c = cache[key]
        m = check_alert(c["text"], s)
        rows.append({"scenario": s["id"], "type": s["type"], "issue_time": s["facts"]["issue_time_utc"],
                     "system": system, "rewrites": c["rewrites"], **{k: v for k, v in m.items() if k != "problems"},
                     "problems": " | ".join(m["problems"]), "alert": c["text"]})
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "rq3_alerts.csv", index=False)

SYS = ["Plain LLM", "RAG", "AETHERA agent"]
def agg(sub):
    return pd.Series({
        "alerts": len(sub),
        "alerts with >=1 unsupported number (%)": 100 * (sub.unsupported_numbers > 0).mean(),
        "unsupported numbers per alert": sub.unsupported_numbers.mean(),
        "alerts with a wrong Kp value (%)": 100 * (sub.kp_errors > 0).mean(),
        "storm alerts with correct G-level (%)": 100 * sub[sub.type == "storm"].g_correct.mean() if (sub.type == "storm").any() else np.nan,
        "alerts overstating G-level (%)": 100 * sub.g_overstated.mean(),
        "alerts with too-strong severity word (%)": 100 * sub.severity_overstated.mean(),
        "alerts with ANY error (%)": 100 * ((sub.unsupported_numbers > 0) | (sub.kp_errors > 0) | (sub.g_overstated > 0)
                                           | (sub.severity_overstated > 0) | (sub.invented_objects > 0)
                                           | ((sub.type == "storm") & (sub.g_correct == 0))).mean(),
        "alerts with invented objects (%)": 100 * (sub.invented_objects > 0).mean(),
        "Flesch reading ease (mean)": sub.flesch.mean(),
        "words (mean)": sub.words.mean(),
        "agent rewrites used (%)": 100 * (sub.rewrites > 0).mean(),
    })
summ = pd.DataFrame({k: agg(v) for k, v in df.groupby("system")})[SYS]
lines = ["AETHERA - RQ3 results (AI-generated alerts)", f"Model: {MODEL}{'  [DRY RUN - fake text]' if DRY else ''}",
         f"{len(scen)} scenarios from 2022-2024; same facts for every system.", "", summ.round(2).to_string(), ""]
try:
    from scipy.stats import wilcoxon
    for a, b in (("Plain LLM", "AETHERA agent"), ("Plain LLM", "RAG")):
        x = df[df.system == a].set_index("scenario").unsupported_numbers
        y = df[df.system == b].set_index("scenario").unsupported_numbers.reindex(x.index)
        if (x - y).abs().sum() > 0:
            p = wilcoxon(x, y).pvalue
            lines.append(f"Wilcoxon test, unsupported numbers, {a} vs {b}: p = {p:.4f}")
except Exception:
    pass
# McNemar exact test on "alert has any error" (paired by scenario)
from scipy.stats import binomtest
def any_err(sub):
    return ((sub.unsupported_numbers > 0) | (sub.kp_errors > 0) | (sub.g_overstated > 0) | (sub.severity_overstated > 0)
            | (sub.invented_objects > 0) | ((sub.type == "storm") & (sub.g_correct == 0))).astype(int)
E = {k: any_err(v.set_index("scenario")) for k, v in df.groupby("system")}
for a, b in (("Plain LLM", "AETHERA agent"), ("Plain LLM", "RAG"), ("RAG", "AETHERA agent")):
    x, y = E[a], E[b].reindex(E[a].index)
    n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    if n01 + n10:
        p = binomtest(n10, n01 + n10, 0.5).pvalue
        lines.append(f"McNemar exact test, any error, {a} vs {b}: only {a} wrong = {n10}, only {b} wrong = {n01}, p = {p:.4f}")
summary = "\n".join(lines)
(RESULTS / "rq3_summary.txt").write_text(summary)
print("\n" + summary)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
cols = ["tab:gray", "tab:orange", "tab:blue"]
for ax, (metric, title) in zip(axes, (("alerts with >=1 unsupported number (%)", "Alerts with unsupported numbers (%)"),
                                      ("storm alerts with correct G-level (%)", "Storm alerts with correct G-level (%)"),
                                      ("Flesch reading ease (mean)", "Flesch reading ease (higher = easier)"))):
    ax.bar(SYS, summ.loc[metric].values, color=cols)
    ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(RESULTS / "fig_rq3.png", dpi=200); plt.close(fig)
print("\nFiles saved in", RESULTS)
