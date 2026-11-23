"""Reproduce every Week 3 result: revised cycle, takeoff check, sensitivity study,
component sizing, flowpath layout and comparison with published engines.

Usage (from the repository root):  python scripts/week3_run_all.py
(The pyCycle cross-check is a separate script: python validation/pycycle_check.py)
"""
import sys, os, json, csv, math
from dataclasses import replace, asdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from turbofan.cycle import run
from turbofan.offdesign import run_offdesign_T41
from turbofan.sizing import size_engine
from turbofan.layout import build_layout
from turbofan.design_case import WEEK2, REV_A, TAKEOFF_T4_LIMIT, FLAT_RATING_DT, TAKEOFF_THRUST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "week3")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "figure.dpi": 200,
    "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k != "gas"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


out = {}
G = lambda t: t * 1e6  # kg/(N s) -> g/(kN s)

# ---------------- 1. design revision steps ----------------
steps = [("Week 2 design", WEEK2),
         ("+ core-side fan PR 1.50", replace(WEEK2, fpr_core=1.50)),
         ("+ turbine cooling 5% + 3% (rotor inlet 1450 K)", REV_A)]
rev = []
for name, inp in steps:
    d = run(inp)
    sz = size_engine(d)
    rev.append({"step": name, "tsfc": G(d["tsfc"]), "m0": d["m0"], "Fs": d["Fs"], "Tt4": d["inputs"]["Tt4"],
                "hpc_pr": d["hpc_pr"], "psi_hub": sz["fan"]["psi_hub"], "D_fan": sz["fan"]["D_tip"],
                "eta_th": d["eta_th"], "eta_prop": d["eta_prop"]})
out["revision_steps"] = rev

# ---------------- 2. revised design point ----------------
d = run(REV_A)
st = d["stations"]
order = ["0", "2", "13", "17", "21", "25", "3", "4", "41", "44", "45", "5", "7"]
names = {"0": "Freestream", "2": "Fan face", "13": "Fan exit, bypass", "17": "Bypass nozzle inlet", "21": "Fan exit, core",
         "25": "HPC inlet", "3": "HPC exit", "4": "Combustor exit", "41": "HPT rotor inlet (after NGV cooling)",
         "44": "HPT exit (before rotor cooling)", "45": "LPT inlet", "5": "LPT exit", "7": "Core nozzle inlet"}
srows = [{"station": k, "name": names[k], "Tt_K": st[k]["Tt"], "pt_kPa": st[k]["pt"] / 1000, "m_kg_s": d["flows"][k]} for k in order]
with open(os.path.join(RES, "revA_stations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=srows[0].keys()); w.writeheader(); w.writerows(srows)
out["design"] = {k: d[k] for k in ["T0", "p0", "V0", "f", "f_burner", "w_fan", "w_fan_core", "w_hpc", "w_hpt", "w_lpt",
                                   "hpc_pr", "hpt_pr", "lpt_pr", "Fs", "tsfc", "m0", "m_core", "m_byp", "m_fuel",
                                   "A8", "A18", "eta_th", "eta_prop", "eta_overall", "velocity_ratio"]}
out["design"]["Tt4"] = d["inputs"]["Tt4"]
out["design"]["tsfc_lb"] = d["tsfc"] * 3600 * 9.80665
out["design"]["m0_corrected"] = d["m0"] * math.sqrt(st["2"]["Tt"] / 288.15) / (st["2"]["pt"] / 101325)
out["stations"] = srows
out["inputs"] = asdict(REV_A)

# ---------------- 3. takeoff check (rotor inlet temperature limited) ----------------
def F_TO(T41, dT):
    return run_offdesign_T41(d, 0.0, 0.0, T41, dT)["F"]
T41_isa = brentq(lambda t: F_TO(t, 0.0) - TAKEOFF_THRUST, 1350, 1700, xtol=0.01)
T41_hot = brentq(lambda t: F_TO(t, FLAT_RATING_DT) - TAKEOFF_THRUST, 1400, 1750, xtol=0.01)
to_isa = run_offdesign_T41(d, 0, 0, T41_isa, 0.0)
to_hot = run_offdesign_T41(d, 0, 0, T41_hot, FLAT_RATING_DT)
lim_hot = run_offdesign_T41(d, 0, 0, TAKEOFF_T4_LIMIT, FLAT_RATING_DT)
out["takeoff"] = {"T41_isa": T41_isa, "T41_hot": T41_hot, "isa": to_isa, "hot": to_hot, "F_hot_at_limit": lim_hot["F"]}

# ---------------- 4. sensitivity study ----------------
base = d
sens_def = [
    ("Fan polytropic efficiency", "eta_p_fan", 0.01, "±0.01"),
    ("HPC polytropic efficiency", "eta_p_hpc", 0.01, "±0.01"),
    ("HPT polytropic efficiency", "eta_p_hpt", 0.01, "±0.01"),
    ("LPT polytropic efficiency", "eta_p_lpt", 0.01, "±0.01"),
    ("Combustor pressure loss", "burner_dp", 0.01, "±1 point"),
    ("Inlet pressure recovery", "inlet_recovery", 0.005, "±0.005"),
    ("Bypass duct pressure ratio", "bypass_duct", 0.005, "±0.005"),
    ("Turbine cooling air (total)", "cooling", 0.02, "±2 points"),
    ("Rotor inlet temperature", "rotor_inlet_T", 50.0, "±50 K"),
    ("Overall pressure ratio", "opr", 2.0, "±2"),
    ("Bypass ratio", "bpr", 0.5, "±0.5"),
    ("Fan pressure ratio (bypass)", "fpr", 0.1, "±0.1"),
]
sens = []
for label, key, dv, txt in sens_def:
    res = {}
    for sgn in (+1, -1):
        if key == "cooling":
            k = 1 + sgn * dv / (REV_A.cool_ngv + REV_A.cool_rotor)
            inp = replace(REV_A, cool_ngv=REV_A.cool_ngv * k, cool_rotor=REV_A.cool_rotor * k)
        else:
            inp = replace(REV_A, **{key: getattr(REV_A, key) + sgn * dv})
        r = run(inp)
        res[sgn] = ((r["tsfc"] / base["tsfc"] - 1) * 100, (r["m0"] / base["m0"] - 1) * 100)
    sens.append({"parameter": label, "change": txt, "dTSFC_plus": res[1][0], "dTSFC_minus": res[-1][0],
                 "dm0_plus": res[1][1], "dm0_minus": res[-1][1]})
out["sensitivity"] = sens

srt = sorted(sens, key=lambda s: max(abs(s["dTSFC_plus"]), abs(s["dTSFC_minus"])))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
y = np.arange(len(srt))
for ax, kp, km, title in [(axes[0], "dTSFC_plus", "dTSFC_minus", "Change in cruise TSFC [%]"),
                          (axes[1], "dm0_plus", "dm0_minus", "Change in engine air mass flow [%]")]:
    ax.barh(y + 0.2, [s[kp] for s in srt], height=0.38, color=C[0], label="parameter increased")
    ax.barh(y - 0.2, [s[km] for s in srt], height=0.38, color=C[1], label="parameter decreased")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(title); ax.grid(axis="y", visible=False)
axes[0].set_yticks(y); axes[0].set_yticklabels([f"{s['parameter']} ({s['change']})" for s in srt], fontsize=8)
axes[0].legend(loc="lower left", fontsize=8)
axes[0].set_title("Fuel consumption"); axes[1].set_title("Engine size")
save(fig, "fig_sensitivity.png")

# ---------------- 5. sizing and layout ----------------
sz = size_engine(d)
L = build_layout(sz)
out["sizing"] = sz
out["layout"] = {k: v for k, v in L.items() if k not in ("lines", "rows")}
out["layout_rows"] = L["rows"]
with open(os.path.join(RES, "flowpath_lines_mm.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["line", "point", "x_mm", "r_mm"])
    for n, pts in L["lines"].items():
        for i, (x, r) in enumerate(pts):
            w.writerow([n, i, round(x * 1000, 2), round(r * 1000, 2)])
with open(os.path.join(RES, "blade_rows_mm.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["row", "x_le_mm", "x_te_mm", "r_hub_le_mm", "r_tip_le_mm", "r_hub_te_mm", "r_tip_te_mm"])
    for r in L["rows"]:
        w.writerow([r["name"]] + [round(r[k] * 1000, 2) for k in ("x_le", "x_te", "r_hub_le", "r_tip_le", "r_hub_te", "r_tip_te")])

# flowpath figure
fig, ax = plt.subplots(figsize=(11, 4.6))
lines = L["lines"]
style = {"nacelle_outer": (INK2, "Nacelle"), "bypass_outer": (C[1], "Bypass duct"), "bypass_inner": (C[1], None),
         "core_casing": (C[0], "Core flowpath"), "core_hub": (C[0], None), "spinner_and_fan_hub": (INK2, None)}
for n, pts in lines.items():
    col, lab = style[n]
    ax.plot([p[0] * 1000 for p in pts], [p[1] * 1000 for p in pts], color=col, lw=1.4, label=lab)
grp_col = {"fan": "#9ec5ee", "bypass": "#f5c2a8", "core": "#cfe0f3"}
for r in L["rows"]:
    fc = "#f2b8b8" if r["name"] == "Combustor" else ("#f6d58e" if r["name"].startswith(("HPT", "LPT")) else grp_col[r["stream"]])
    ax.fill([r["x_le"] * 1000, r["x_te"] * 1000, r["x_te"] * 1000, r["x_le"] * 1000],
            [r["r_hub_le"] * 1000, r["r_hub_te"] * 1000, r["r_tip_te"] * 1000, r["r_tip_le"] * 1000],
            color=fc, ec=INK2, lw=0.4)
ax.axhline(0, color=INK2, lw=0.8, ls="-.")
labels = [("Fan", 30, 360), ("Bypass OGV", 140, 330), ("4-stage axial HPC", 330, 165), ("Centrifugal\nimpeller", 560, 120),
          ("Combustor", 750, 215), ("2-stage HPT", 870, 105), ("3-stage LPT", 1090, 110), ("Core nozzle", 1390, 60),
          ("Bypass nozzle exit", 1000, 345)]
for t, x0, y0 in labels:
    ax.text(x0, y0, t, fontsize=7.5, ha="center", color=INK)
ax.set_aspect("equal"); ax.set_xlabel("Axial position from fan leading edge [mm]"); ax.set_ylabel("Radius [mm]")
ax.set_title("Engine flowpath (upper half, revision A)")
ax.legend(loc="upper right", fontsize=7.5)
ax.set_ylim(-20, 460)
save(fig, "fig_flowpath.png")

# ---------------- 6. published engine data ----------------
pub = [
    {"engine": "PW305A", "F_kN": 20.81, "bpr": 4.5, "opr": None, "D_fan_m": None, "D_engine_m": 0.970, "length_m": 2.062,
     "tsfc_to": 0.407, "tsfc_cr": 0.685, "layout": "1 fan; 4A+1C HPC; 2 HPT; 3 LPT", "N_hp": None, "N_lp": None,
     "source": "Wikipedia, Pratt & Whitney Canada PW300 (series data)"},
    {"engine": "CFE738", "F_kN": 24.9, "bpr": 5.3, "opr": 35, "D_fan_m": 0.902, "D_engine_m": None, "length_m": 1.73,
     "tsfc_to": 0.369, "tsfc_cr": 0.645, "layout": "1 fan; 5A+1C HPC; 2 HPT; 3 LPT", "N_hp": None, "N_lp": None,
     "airflow_kg_s": 240 * 0.45359237, "source": "Wikipedia, CFE CFE738"},
    {"engine": "PW545A", "F_kN": 16.92, "bpr": 4.10, "opr": None, "D_fan_m": 27.3 * 0.0254, "D_engine_m": None,
     "length_m": 68.6 * 0.0254, "tsfc_to": 0.44, "tsfc_cr": None, "layout": "1 fan + booster; 2A+1C HPC; 1 HPT; 3 LPT",
     "N_hp": None, "N_lp": None, "source": "Wikipedia, Pratt & Whitney Canada PW500"},
    {"engine": "PW535A", "F_kN": 15.12, "bpr": 2.55, "opr": 13.3, "D_fan_m": 28.2 * 0.0254, "D_engine_m": None,
     "length_m": 63.9 * 0.0254, "tsfc_to": None, "tsfc_cr": None, "layout": "1 fan + booster; 2A+1C HPC; 1 HPT; 2 LPT",
     "N_hp": None, "N_lp": None, "source": "Wikipedia, Pratt & Whitney Canada PW500"},
    {"engine": "PW530A", "F_kN": 12.84, "bpr": 3.7, "opr": 13.3, "D_fan_m": 23.0 * 0.0254, "D_engine_m": None,
     "length_m": 60.0 * 0.0254, "tsfc_to": 0.44, "tsfc_cr": None, "layout": "1 fan; 2A+1C HPC; 1 HPT; 2 LPT",
     "N_hp": None, "N_lp": None, "source": "Wikipedia, Pratt & Whitney Canada PW500"},
    {"engine": "HF120", "F_kN": 9.1, "bpr": 2.9, "opr": 24, "D_fan_m": None, "D_engine_m": 25.8 * 0.0254, "length_m": 59.5 * 0.0254,
     "tsfc_to": None, "tsfc_cr": None, "layout": "1 fan + 2 LP stages; 1C HPC; 1 HPT; 2 LPT", "N_hp": 48777, "N_lp": 19055,
     "source": "Wikipedia, GE Honda HF120"},
]
out["published"] = pub
ours = {"F_kN": TAKEOFF_THRUST / 1000, "bpr": REV_A.bpr, "opr": REV_A.opr, "D_fan_m": sz["fan"]["D_tip"],
        "D_nacelle_m": L["D_nacelle"], "length_fan_to_core_exit_m": L["length_fan_to_core_exit"],
        "length_overall_m": L["length_overall"], "tsfc_to": to_isa["tsfc"] * 3600 * 9.80665,
        "tsfc_cr": d["tsfc"] * 3600 * 9.80665, "N_hp": sz["hpc_axial"]["N_rpm"], "N_lp": sz["fan"]["N_rpm"],
        "airflow_TO_kg_s": to_isa["m0"]}
out["ours"] = ours
# speed scaling of HF120 to 20 kN (same blade speeds: N ~ 1/D ~ 1/sqrt(F))
k = math.sqrt(9.1 / 20.0)
out["hf120_scaled_speeds"] = {"N_hp": 48777 * k, "N_lp": 19055 * k}

fig, ax = plt.subplots(figsize=(6.4, 4.0))
pts = [p for p in pub if p["D_fan_m"]]
ax.plot([p["F_kN"] for p in pts], [p["D_fan_m"] for p in pts], "o", color=C[0], ms=8, label="Published engines (fan diameter)")
for p in pts:
    ax.annotate(p["engine"], (p["F_kN"], p["D_fan_m"]), xytext=(6, -3), textcoords="offset points", fontsize=8)
ax.plot([20], [ours["D_fan_m"]], "D", color=C[1], ms=8, label="This design (revision A)")
ax.annotate("This design", (20, ours["D_fan_m"]), xytext=(6, -10), textcoords="offset points", fontsize=8)
Fs_ = np.linspace(10, 27, 50)
kfit = np.mean([p["D_fan_m"] / math.sqrt(p["F_kN"]) for p in pts])
ax.plot(Fs_, kfit * np.sqrt(Fs_), "--", color=INK2, lw=1, label="Fit D ∝ √F to published engines")
ax.set_xlabel("Takeoff thrust [kN]"); ax.set_ylabel("Fan diameter [m]")
ax.set_title("Fan diameter compared with published engines")
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "fig_fan_diameter.png")
out["fan_diameter_fit_coeff"] = kfit
out["fan_diameter_expected_20kN"] = kfit * math.sqrt(20)

with open(os.path.join(RES, "week3_results.json"), "w") as fh:
    json.dump(clean(out), fh, indent=2)
print("Done. Rev A TSFC %.3f g/(kN s), m0 %.2f kg/s, fan D %.3f m, T41 for 20 kN: %.0f K (ISA), %.0f K (ISA+15)"
      % (G(d["tsfc"]), d["m0"], sz["fan"]["D_tip"], T41_isa, T41_hot))
