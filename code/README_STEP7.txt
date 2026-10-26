AETHERA - STEP 7: EXTRA TEST YEAR 2025
======================================
Nothing is retrained. Your saved LSTM is reused and the Random Forest is rebuilt with the
same data and seed, so your 2022-2024 results in the paper do NOT change.

1. Replace config.py with the new one (only change: OMNI_YEARS now goes up to 2025).
2. Put 09_test_2025.py in your code folder.
3. Run, in order:
       python 01_download_omni.py     (downloads ONLY the new 2025 file; the others are skipped)
       python 02_clean_omni.py        (rebuilds omni_hourly.csv including 2025)
       python 09_test_2025.py         (2-3 minutes)

SEND ME
  * results/rq1_2025_summary.txt
  * results/fig_rq1_2025_rmse.png
  * results/fig_rq1_2025_storm.png
  * the last 3 lines printed by 09_test_2025.py

CHECK: in rq1_2025_summary.txt the "test" (2022-2024) numbers must be the same as in your
earlier rq1_summary.txt. If they are, everything is consistent.
