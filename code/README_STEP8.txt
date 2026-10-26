AETHERA - STEP 8: LIVE PROTOTYPE (+ one RQ3 re-score)
=====================================================

PART A - Re-score RQ3 (10 seconds, no API key needed)
  Replace 08_rq3_alerts.py with the new one, then run:
      python 08_rq3_alerts.py --rescore
  Send me the new results/rq3_summary.txt

PART B - Live prototype
  Put 10_live_agent.py in your code folder.
  1. Test without the language model:
         python 10_live_agent.py --no-llm
  2. Real run with the AI alert (Groq key needed):
         $env:GROQ_API_KEY="gsk_..."
         python 10_live_agent.py
  The first run builds results/rf_kp.joblib (your RQ1 Random Forest, 1-2 minutes). Later runs are fast.
  Optional, automatic every 3 hours (leave the terminal open, Ctrl+C to stop):
         python 10_live_agent.py --every 3

  Output: results/live/alert_<time>.md, facts_<time>.json, fig_live_<time>.png

  IF YOUR NETWORK BLOCKS NOAA OR CELESTRAK (like NASA before):
  1. Run  python 10_live_agent.py --no-llm  once on your laptop anyway: even if fetching fails,
     it first builds results/rf_kp.joblib.  (If it says "Run failed", that is fine for now.)
  2. In Google Colab: upload 10_live_agent.py, config.py and results/rf_kp.joblib
     (make folders: code/, results/  and put the files in them), then run:
         !pip install groq langgraph
         %cd code
         import os; os.environ["GROQ_API_KEY"]="gsk_..."
         !python 10_live_agent.py
     and download the files from results/live/.

  CelesTrak allows ONE download per 2 hours: the agent saves the TLEs and reuses them,
  so never run it more often than every 2 hours.

SEND ME
  * the new rq3_summary.txt (Part A)
  * results/live/alert_<time>.md and fig_live_<time>.png (Part B)
