AETHERA - STEP 6: AI ALERTS (RQ3)
=================================
Put 08_rq3_alerts.py in your AETHERA/code/ folder.

1. Install two libraries (one time):
       python -m pip install groq langgraph

2. Get your free Groq key: https://console.groq.com/keys  -> "Create API Key" -> copy it (starts with gsk_)
   Set it in the terminal (every new terminal):
       $env:GROQ_API_KEY="gsk_your_key_here"
   Never put the key inside a file you share or upload.

3. Test without using the API (1 minute):
       python 08_rq3_alerts.py --dry-run
   The numbers it prints are FAKE - this only checks that everything works.

4. Real run (about 30-60 minutes, it waits between calls because of free limits):
       python 08_rq3_alerts.py
   If it stops with "DAILY limit reached", just run the same command again later or tomorrow.
   Finished alerts are saved in results/rq3_cache.json and are not repeated.

SEND ME
  * results/rq3_summary.txt
  * results/fig_rq3.png
  * results/rq3_alerts.csv   (all alerts - I will check a sample by hand)
