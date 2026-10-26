"""
Step 3.5 - Quick visual check of the data. ALWAYS look at your data before modelling.
Run:  python 05_check_data.py
Outputs (in results/):
    fig_kp_2000_2024.png        Kp over all years (should rise/fall with the 11-year solar cycle)
    fig_storm_<storm>.png       Kp around each storm + altitude loss vs starting altitude
"""
import pandas as pd
import matplotlib.pyplot as plt
from config import CLEAN, RESULTS, STORMS

omni = pd.read_csv(CLEAN / "omni_hourly.csv", index_col="time", parse_dates=True)

fig, ax = plt.subplots(figsize=(11, 3.5))
omni["kp"].resample("27D").max().plot(ax=ax, lw=0.8, color="tab:blue", label="Max Kp per 27 days")
ax.axhline(5, color="tab:red", ls="--", lw=0.8, label="Storm threshold (Kp = 5)")
ax.set_ylabel("Kp"); ax.set_title("Kp index, 2000-2024"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(RESULTS / "fig_kp_2000_2024.png", dpi=150); plt.close(fig)

for name, info in STORMS.items():
    a, b = info["window"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    omni.loc[a:b, "kp"].plot(ax=axes[0], color="tab:blue")
    axes[0].axhline(5, color="tab:red", ls="--", lw=0.8)
    axes[0].axvline(pd.Timestamp(info["start"]), color="grey", ls=":")
    axes[0].set_title(f"Kp around {name}"); axes[0].set_ylabel("Kp")
    f = CLEAN / f"decay_{name}.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[~d["maneuver"]]
        axes[1].scatter(d["alt_before_km"], d["alt_loss_km"], s=4, alpha=0.5)
        axes[1].set_xlabel("Altitude before storm (km)"); axes[1].set_ylabel("Altitude loss (km)")
        axes[1].set_title("Lower objects should lose more height")
    else:
        axes[1].text(0.5, 0.5, "No TLE data yet", ha="center", va="center"); axes[1].axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / f"fig_storm_{name}.png", dpi=150); plt.close(fig)
print("Figures saved in", RESULTS)
