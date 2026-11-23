"""Extended comparison with published business-jet and regional turbofans.

Data: Wikipedia "Turbofan" article, table "Commercial turbofans in production"
(accessed 22 Nov 2026), plus the individual engine articles used in Week 3.
Ranges in that table are paired max-with-max (largest thrust with largest fan),
which is an assumption: the table does not say which variant each end belongs to.
Rows that conflict with another source are kept but flagged and left out of the fit.
Run:  python validation/published_engines.py
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "validation"); os.makedirs(OUT, exist_ok=True)

# name, thrust kN, fan diameter m, bypass ratio, length m, dry mass t, source, flag
T = "Wikipedia 'Turbofan' table"
DATA = [
    ("PW600", 6.0, 0.36, "1.8-2.8", 0.67, 0.15, T, ""),
    ("FJ33", 6.7, 0.53, "-", 0.98, 0.14, T, ""),
    ("FJ44 (largest)", 15.6, 0.57, "3.3-4.1", 2.09, 0.24, T, ""),
    ("PW500", 13.3, 0.70, "3.9", 1.52, 0.28, T, ""),
    ("TFE731 (largest)", 22.2, 0.78, "2.66-3.9", 2.08, 0.45, T, ""),
    ("HTF7000", 28.9, 0.87, "4.4", 2.29, 0.62, T, ""),
    ("AE 3007", 33.7, 1.11, "5.0", 2.71, 0.72, T, ""),
    ("CF34 (smallest)", 41.0, 1.25, "5.3-6.3", 2.62, 0.74, T, ""),
    ("Silvercrest", 50.9, 1.08, "5.9", 1.90, 1.09, T, ""),
    ("PW800", 69.7, 1.30, "5.5", None, 1.4, T, "table length equals the fan size; length not used"),
    ("PW300 (largest)", 35.6, 0.97, "3.8-4.5", 2.07, 0.47, T, "0.97 m is given as overall engine diameter in the PW300 article; not used in fit"),
    ("HF120", 9.1, 0.66, "2.9", 1.51, 0.211, T, "HF120 article gives 0.66 m as overall diameter; a second HF120 row (BPR 4.43, 0.54 m) conflicts; not used in fit"),
    ("CFE738", 24.9, 0.902, "5.3", 1.73, None, "Wikipedia 'CFE CFE738'", ""),
    ("PW545A", 16.92, 0.693, "4.10", 1.74, None, "Wikipedia 'PW500'", ""),
    ("PW535A", 15.12, 0.716, "2.55", 1.62, None, "Wikipedia 'PW500'", ""),
    ("PW530A", 12.84, 0.584, "3.7", 1.52, None, "Wikipedia 'PW500'", ""),
]
w3 = json.load(open(os.path.join(ROOT, "results", "week3", "week3_results.json")))
ours = {"F": 20.0, "D": w3["sizing"]["fan"]["D_tip"], "L": w3["layout"]["length_fan_to_core_exit"]}

DIAM_BAD = lambda flag: "diameter" in flag
fit = [(F, D) for n, F, D, *_rest, flag in DATA if not DIAM_BAD(flag) and 10 <= F <= 35]
k = np.array([D / math.sqrt(F) for F, D in fit])
res = {"fit_points": len(fit), "k_mean": float(k.mean()), "k_min": float(k.min()), "k_max": float(k.max()),
       "D_20kN_mean": float(k.mean() * math.sqrt(20)), "D_20kN_min": float(k.min() * math.sqrt(20)),
       "D_20kN_max": float(k.max() * math.sqrt(20)), "ours_k": ours["D"] / math.sqrt(20), "ours_D": ours["D"]}
tw = [F / (m * 9.80665) for n, F, D, b, L, m, s, flag in DATA if m and 10 <= F <= 35 and not flag]
res["thrust_to_weight_10_35kN"] = {"mean": float(np.mean(tw)), "min": float(min(tw)), "max": float(max(tw))}
tw_ = res["thrust_to_weight_10_35kN"]
res["mass_estimate_kg"] = {"at_mean_TW": 20.0 / (tw_["mean"] * 9.80665) * 1000, "at_highest_TW": 20.0 / (tw_["max"] * 9.80665) * 1000, "at_lowest_TW": 20.0 / (tw_["min"] * 9.80665) * 1000}
res["data"] = [dict(zip(["engine", "thrust_kN", "fan_D_m", "bpr", "length_m", "mass_t", "source", "flag"], d)) for d in DATA]
json.dump(res, open(os.path.join(OUT, "published_engines.json"), "w"), indent=2)

C = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(figsize=(7, 4.4))
good = [d for d in DATA if not DIAM_BAD(d[7])]; bad = [d for d in DATA if DIAM_BAD(d[7])]
ax.plot([d[1] for d in good], [d[2] for d in good], "o", color=C[0], ms=7, label="Published fan diameter")
ax.plot([d[1] for d in bad], [d[2] for d in bad], "o", mfc="white", mec=C[0], ms=7, label="Flagged (likely overall diameter)")
OFF = {"PW500": (-30, 6), "PW535A": (2, 7), "PW545A": (6, 4), "PW530A": (-46, -4), "FJ44": (6, -6), "CFE738": (-12, 7)}
for d in DATA:
    nm = d[0].split(" (")[0]
    ax.annotate(nm, (d[1], d[2]), xytext=OFF.get(nm, (5, -3)), textcoords="offset points", fontsize=7)
ax.plot([20], [ours["D"]], "D", color=C[1], ms=8, label="This design (rev. A)")
Fs = np.linspace(5, 72, 100)
ax.fill_between(Fs, k.min() * np.sqrt(Fs), k.max() * np.sqrt(Fs), color="#d0d0cc", alpha=0.5, lw=0, label="Range of D/√F, 10–35 kN engines")
ax.plot(Fs, k.mean() * np.sqrt(Fs), "--", color="#52514e", lw=1, label="Mean D ∝ √F")
ax.set_xlabel("Takeoff thrust [kN]"); ax.set_ylabel("Fan diameter [m]")
ax.set_title("Fan diameter against thrust, published engines", fontweight="bold", fontsize=10)
ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.grid(True, color="#e4e3df")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fan_diameter_extended.png"), dpi=200, facecolor="white")
print(json.dumps({k_: v for k_, v in res.items() if k_ != "data"}, indent=1))
