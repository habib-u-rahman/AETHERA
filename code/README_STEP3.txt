AETHERA - STEP 3: DOWNLOAD AND CLEAN THE DATA
=============================================

Put these files in your  AETHERA/code/  folder.
Open a terminal IN that folder and run the scripts in order.

  1. python 01_download_omni.py     (5-15 min)  solar wind + Kp + Dst + F10.7, 2000-2024
  2. python 02_clean_omni.py        (1-2 min)   cleans it and makes train/test files
  3. (set your Space-Track login - see below)
     python 03_download_tles.py     (10-30 min) satellite TLEs for the 3 storms
  4. python 04_clean_tles.py        (1-5 min)   altitude before/after each storm
  5. python 05_check_data.py        (1 min)     makes check figures in results/

Space-Track login (step 3) - your password is never written in the code:
  Windows PowerShell:   $env:ST_USER="your@email.com"
                        $env:ST_PASS="yourpassword"
  Google Colab:         import os
                        os.environ["ST_USER"]="your@email.com"
                        os.environ["ST_PASS"]="yourpassword"

Do NOT run 03_download_tles.py many times. Space-Track blocks accounts that
download the same history again and again. The script skips files you already have.

If 03_download_tles.py stops with a timeout or a very big file:
  open config.py and change MIN_MEAN_MOTION from 15.0 to 15.3 (only objects below
  about 450 km), then run it again.

WHAT TO SEND BACK TO ME WHEN DONE
  * data/clean/data_report.txt
  * results/fig_kp_2000_2024.png
  * results/fig_storm_2022_starlink.png
  * the last lines printed by 04_clean_tles.py
