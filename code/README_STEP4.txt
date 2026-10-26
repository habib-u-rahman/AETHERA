AETHERA - STEP 4: FORECAST MODEL (RQ1)
======================================
Put 06_rq1_forecast.py in your AETHERA/code/ folder (next to config.py).

1. QUICK TEST first (about 5 minutes, uses less data, only to check it runs):
       python 06_rq1_forecast.py --quick

2. If the quick test works, run the FULL version (about 20-60 minutes):
       python 06_rq1_forecast.py
   You will see the LSTM training "epochs" one by one. It stops by itself
   when it stops improving (early stopping). Don't close the terminal.

WHAT IT DOES
  Trains on 2000-2018, tunes on 2019-2021, tests on 2022-2024 and Oct-Nov 2003.
  Compares Persistence (baseline), Random Forest and LSTM at 3, 6, 12, 24 h ahead.

SEND ME AFTER THE FULL RUN
  * results/rq1_summary.txt
  * results/fig_rq1_rmse.png
  * results/fig_rq1_2024_gannon.png  and  fig_rq1_2022_starlink.png
