"""
Step 3.1 - Download OMNI hourly data (one file per year, 2000-2024).
Run:  python 01_download_omni.py
Output: data/raw/omni/omni2_YYYY.dat  (about 25 files, ~6 MB each)

This version tries several NASA servers and retries, because the main server
sometimes does not answer from some countries or networks.
Files that already exist are skipped, so you can safely run it again.
"""
import time
import requests
from config import OMNI_YEARS, RAW

MIRRORS = [
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "https://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "http://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "http://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
]

def download(year):
    for url_t in MIRRORS:
        url = url_t.format(year=year)
        for attempt in (1, 2):
            try:
                print(f"  trying {url} (attempt {attempt})")
                r = requests.get(url, timeout=(20, 300))   # 20 s to connect, 5 min to download
                r.raise_for_status()
                if len(r.content) > 1_000_000:
                    return r.content
                print("  file too small, trying next")
            except requests.RequestException as e:
                print(f"  failed: {type(e).__name__}")
                time.sleep(3)
    return None

failed = []
for year in OMNI_YEARS:
    out = RAW / "omni" / f"omni2_{year}.dat"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"{year}: already downloaded")
        continue
    print(f"{year}: downloading")
    data = download(year)
    if data is None:
        print(f"{year}: COULD NOT DOWNLOAD from any server")
        failed.append(year)
        continue
    out.write_bytes(data)
    print(f"{year}: saved ({len(data)/1e6:.1f} MB)")
    time.sleep(1)

if failed:
    print("\nThese years failed:", failed)
    print("Your network is probably blocking NASA. Use Google Colab instead (see README).")
else:
    print("\nDone. All files are in", RAW / "omni")