"""Reproduce every Week 2 result, table and figure.

Usage (from the repository root):  python scripts/week2_run_all.py
Outputs go to results/ (CSV, JSON) and results/figures/ (PNG).
"""
import sys, os, json, csv, math
from dataclasses import replace, asdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from turbofan.cycle import DesignInputs, run, optimum_fpr
from turbofan.offdesign import run_offdesign
from turbofan.requirement import toc_requirement, AIRCRAFT
from turbofan.design_case import WEEK2 as FINAL, TAKEOFF_T4_LIMIT, FLAT_RATING_DT, TAKEOFF_THRUST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- chart style (validated palette, one axis per chart, direct labels) ----------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "figure.dpi": 200,
    "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False,
})
TSFC_UNIT = "TSFC [g/(kN·s)]"


def g_kNs(tsfc):
    return tsfc * 1e6


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mark_design(ax, x, y, text="Selected design", dx=8, dy=8):
    ax.plot([x], [y], marker="o", ms=8, mfc="white", mec=INK, mew=1.6, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK)


out = {}

# ---------- 1. thrust requirement ----------
req = toc_requirement()
out["requirement"] = req
out["aircraft"] = AIRCRAFT

# ---------- 2. sweeps ----------
base = replace(FINAL)

# 2a: FPR sweep at several BPR
fprs = np.linspace(1.4, 2.6, 49)
sweep_fpr = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for i, bpr in enumerate([3.0, 4.0, 5.0, 6.0]):
    ts, fs = [], []
    for fpr in fprs:
        try:
            r = run(replace(base, bpr=bpr, fpr=fpr))
            ts.append(g_kNs(r["tsfc"])); fs.append(r["Fs"])
        except ValueError:
            ts.append(float("nan")); fs.append(float("nan"))
        sweep_fpr.append({"bpr": bpr, "fpr": fpr, "tsfc_g_per_kNs": ts[-1], "Fs": fs[-1]})
    axes[0].plot(fprs, ts, color=C[i]); axes[1].plot(fprs, fs, color=C[i])
    k = int(np.nanargmin(ts))
    last = max(j for j in range(len(ts)) if not math.isnan(ts[j]))
    axes[0].plot(fprs[k], ts[k], "o", ms=5, color=C[i])
    off = {3.0: 0, 4.0: 5, 5.0: -6, 6.0: 0}[bpr]
    if bpr == 6.0:
        axes[0].annotate("BPR 6", (fprs[last], ts[last]), xytext=(-6, 4), textcoords="offset points", fontsize=8, color=INK, ha="right")
    else:
        axes[0].annotate(f"BPR {bpr:g}", (fprs[last], ts[last]), xytext=(4, off), textcoords="offset points", fontsize=8, color=INK, va="center")
    axes[1].annotate(f"BPR {bpr:g}", (fprs[last], fs[last]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
for ax in axes:
    ax.axvline(1.8, color=INK2, lw=1, ls="--"); ax.set_xlabel("Fan pressure ratio")
    ax.set_xlim(1.4, 2.85)
axes[0].text(1.81, axes[0].get_ylim()[1], "single-stage fan\nlimit (1.8)", fontsize=7.5, color=INK2, va="top")
axes[0].set_ylabel(TSFC_UNIT); axes[0].set_title("Fuel consumption vs fan pressure ratio")
axes[1].set_ylabel("Specific thrust [N/(kg/s)]"); axes[1].set_title("Specific thrust vs fan pressure ratio")
fig.text(0.5, -0.03, f"Cruise, 12,000 m, Mach 0.80, OPR {base.opr:g}, Tt4 {base.Tt4:g} K. Dots mark minimum TSFC.", ha="center", fontsize=8, color=INK2)
save(fig, "fig_sweep_fpr_bpr.png")

# 2b: OPR sweep at several T4
oprs = np.linspace(15, 35, 41)
sweep_opr = []
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for i, T4 in enumerate([1350.0, 1450.0, 1550.0]):
    ts, fs, t3 = [], [], []
    for opr in oprs:
        r = run(replace(base, opr=opr, Tt4=T4))
        ts.append(g_kNs(r["tsfc"])); fs.append(r["Fs"]); t3.append(r["stations"]["3"]["Tt"])
        sweep_opr.append({"Tt4": T4, "opr": opr, "tsfc_g_per_kNs": ts[-1], "Fs": fs[-1], "Tt3": t3[-1]})
    for ax, y in zip(axes[:2], [ts, fs]):
        ax.plot(oprs, y, color=C[i])
    if i == 0:
        axes[2].plot(oprs, t3, color=INK2)
    axes[0].annotate(f"Tt4 {T4:g} K", (oprs[-1], ts[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
    axes[1].annotate(f"Tt4 {T4:g} K", (oprs[-1], fs[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
for ax in axes:
    ax.axvline(23, color=INK2, lw=1, ls="--"); ax.set_xlabel("Overall pressure ratio"); ax.set_xlim(15, 40)
axes[0].set_ylabel(TSFC_UNIT); axes[0].set_title("TSFC vs OPR")
axes[1].set_ylabel("Specific thrust [N/(kg/s)]"); axes[1].set_title("Specific thrust vs OPR")
axes[2].set_ylabel("HPC exit temperature Tt3 [K]"); axes[2].set_title("Compressor exit temperature")
axes[2].set_xlim(15, 36)
fig.text(0.5, -0.03, "Cruise, BPR 5.0, FPR 1.80. Dashed line: selected OPR 23. Tt3 does not depend on Tt4, so one line is shown.", ha="center", fontsize=8, color=INK2)
save(fig, "fig_sweep_opr_t4.png")

# 2c: optimum FPR vs BPR and T4 (table)
opt_table = []
for bpr in [3.0, 4.0, 5.0, 6.0]:
    for T4 in [1350.0, 1450.0, 1550.0]:
        fo = optimum_fpr(replace(base, bpr=bpr, Tt4=T4))
        r = run(replace(base, bpr=bpr, Tt4=T4, fpr=fo))
        opt_table.append({"bpr": bpr, "Tt4": T4, "fpr_opt": fo, "tsfc_g_per_kNs": g_kNs(r["tsfc"]), "Fs": r["Fs"]})
out["optimum_fpr_table"] = opt_table

# ---------- 3. cruise T4 selection against takeoff requirement ----------
def hot_day_takeoff(T4c, T4lim=TAKEOFF_T4_LIMIT):
    d = run(replace(base, Tt4=T4c))
    return run_offdesign(d, 0.0, 0.0, T4lim, FLAT_RATING_DT)["F"]

T4_balanced = {lim: brentq(lambda t: hot_day_takeoff(t, lim) - TAKEOFF_THRUST, 1300, 1560)
               for lim in (1600.0, 1650.0, 1700.0)}
out["T4_cruise_balanced"] = T4_balanced

t4c = np.linspace(1300, 1560, 27)
sel = []
for T4c in t4c:
    d = run(replace(base, Tt4=T4c))
    F_hot = run_offdesign(d, 0, 0, TAKEOFF_T4_LIMIT, FLAT_RATING_DT)["F"]
    sel.append({"Tt4_cruise": T4c, "tsfc_g_per_kNs": g_kNs(d["tsfc"]), "m0": d["m0"], "F_takeoff_hot": F_hot})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
axes[0].plot(t4c, [s["F_takeoff_hot"] / 1000 for s in sel], color=C[0])
axes[0].axhline(20, color=INK2, lw=1, ls="--"); axes[0].text(1305, 20.3, "20 kN requirement", fontsize=7.5, color=INK2)
axes[0].set_ylabel("Takeoff thrust, ISA+15, Tt4 1650 K [kN]"); axes[0].set_title("Hot-day takeoff thrust")
axes[1].plot(t4c, [s["tsfc_g_per_kNs"] for s in sel], color=C[0]); axes[1].set_ylabel(TSFC_UNIT); axes[1].set_title("Cruise TSFC")
axes[2].plot(t4c, [s["m0"] for s in sel], color=C[0]); axes[2].set_ylabel("Cruise air mass flow [kg/s]"); axes[2].set_title("Engine size (mass flow)")
for ax in axes:
    ax.axvline(FINAL.Tt4, color=INK2, lw=1, ls="--"); ax.set_xlabel("Cruise turbine inlet temperature [K]")
fig.text(0.5, -0.03, "Each point is a separate engine sized for 4.0 kN at cruise. Dashed vertical line: selected Tt4 = 1450 K.", ha="center", fontsize=8, color=INK2)
save(fig, "fig_t4_selection.png")

# ---------- 4. final design point ----------
d = run(FINAL)
st = d["stations"]
order = ["0", "2", "13", "17", "18", "21", "25", "3", "4", "45", "5", "7", "8"]
names = {"0": "Freestream", "2": "Fan face", "13": "Fan exit, bypass", "17": "Bypass nozzle inlet",
         "18": "Bypass nozzle throat", "21": "Fan exit, core", "25": "HPC inlet", "3": "HPC exit",
         "4": "HPT inlet", "45": "HPT exit / LPT inlet", "5": "LPT exit", "7": "Core nozzle inlet", "8": "Core nozzle throat"}
station_rows = []
for k in order:
    s = st[k]; g = s["gas"]
    station_rows.append({"station": k, "name": names[k], "Tt_K": s["Tt"], "pt_kPa": s["pt"] / 1000,
                         "m_kg_s": d["flows"][k], "cp": g.cp(s["Tt"]), "gamma": g.gamma(s["Tt"])})
with open(os.path.join(RES, "design_point_stations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=station_rows[0].keys()); w.writeheader(); w.writerows(station_rows)

summary = {k: d[k] for k in ["T0", "p0", "V0", "f", "w_fan", "w_hpc", "w_hpt", "w_lpt", "hpc_pr", "hpt_pr", "lpt_pr",
                             "eta_is_fan", "eta_is_hpc", "F_core_spec", "F_byp_spec", "Fs", "tsfc", "m0", "m_core",
                             "m_byp", "m_fuel", "A8", "A18", "eta_th", "eta_prop", "eta_overall", "velocity_ratio",
                             "Veff_core", "Veff_byp"]}
summary["nozzle_core"] = d["nozzle_core"]; summary["nozzle_bypass"] = d["nozzle_bypass"]
summary["thrust_core"] = d["F_core_spec"] * d["m_core"]
summary["thrust_bypass"] = d["F_byp_spec"] * d["m_byp"]
summary["ram_drag"] = d["m0"] * d["V0"]
summary["tsfc_lb_per_lbf_h"] = d["tsfc"] * 3600 * 9.80665
summary["m0_corrected"] = d["m0"] * math.sqrt(st["2"]["Tt"] / 288.15) / (st["2"]["pt"] / 101325)
out["design"] = summary
out["inputs"] = asdict(FINAL)
out["stations"] = station_rows

# T-s diagram (drawn with air properties throughout, for illustration)
from turbofan.thermo import air as _air
AIR = _air()
def entropy(T, p):
    return AIR.phi(T) - AIR.R * math.log(p / 101325.0)
s_ref = entropy(st["0"]["Tt"], st["0"]["pt"])
def pt(k):
    return (entropy(st[k]["Tt"], st[k]["pt"]) - s_ref, st[k]["Tt"])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
segs = []
pts = [pt(k) for k in ["0", "2", "25", "3"]]
# combustor: constant-pressure heating with linear pressure loss
T3, T4_, p3, p4 = st["3"]["Tt"], st["4"]["Tt"], st["3"]["pt"], st["4"]["pt"]
for x in np.linspace(0, 1, 40)[1:]:
    T = T3 + (T4_ - T3) * x; p = p3 + (p4 - p3) * x
    pts.append((entropy(T, p) - s_ref, T))
pts += [pt(k) for k in ["45", "5", "7"]]
ax.plot([q[0] for q in pts], [q[1] for q in pts], color=C[0], label="Core stream")
for k in ["3", "4", "45", "5"]:
    x, y = pt(k); ax.plot(x, y, "o", color=C[0], ms=5); ax.annotate(k, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=8)
x, y = pt("25"); ax.plot(x, y, "o", color=C[0], ms=5); ax.annotate("21/25", (x, y), xytext=(8, -10), textcoords="offset points", fontsize=8)
bp = [pt(k) for k in ["2", "13", "17"]]
ax.plot([q[0] for q in bp], [q[1] for q in bp], color=C[1], label="Bypass stream")
x, y = pt("17"); ax.plot(x, y, "o", color=C[1], ms=5); ax.annotate("13/17", (x, y), xytext=(-40, 6), textcoords="offset points", fontsize=8)
x, y = pt("2"); ax.plot(x, y, "o", color=INK, ms=5); ax.annotate("0/2", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=8)
ax.set_xlabel("Specific entropy relative to freestream [J/(kg·K)]"); ax.set_ylabel("Total temperature Tt [K]")
ax.set_title("Cruise design point: T-s diagram (total conditions)")
ax.legend(loc="upper left")
save(fig, "fig_ts_diagram.png")

# ---------- 5. takeoff estimate ----------
to_isa = run_offdesign(d, 0, 0, TAKEOFF_T4_LIMIT, 0.0)
to_hot = run_offdesign(d, 0, 0, TAKEOFF_T4_LIMIT, FLAT_RATING_DT)
T4_isa_20 = brentq(lambda t: run_offdesign(d, 0, 0, t, 0.0)["F"] - TAKEOFF_THRUST, 1350, 1700)
T4_hot_20 = brentq(lambda t: run_offdesign(d, 0, 0, t, FLAT_RATING_DT)["F"] - TAKEOFF_THRUST, 1350, 1750)
rated_isa = run_offdesign(d, 0, 0, T4_isa_20, 0.0)
rated_hot = run_offdesign(d, 0, 0, T4_hot_20, FLAT_RATING_DT)
self_check = run_offdesign(d, FINAL.altitude, FINAL.mach, FINAL.Tt4)
out["takeoff"] = {"at_limit_isa": to_isa, "at_limit_hot": to_hot, "T4_for_20kN_isa": T4_isa_20,
                  "T4_for_20kN_hot": T4_hot_20, "rated_isa": rated_isa, "rated_hot": rated_hot,
                  "offdesign_self_check_F": self_check["F"], "offdesign_self_check_fpr": self_check["fpr"],
                  "offdesign_self_check_bpr": self_check["bpr"]}

t4s = np.linspace(1400, 1700, 31)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, (dT, lab) in enumerate([(0.0, "ISA"), (FLAT_RATING_DT, "ISA+15")]):
    Fs_ = [run_offdesign(d, 0, 0, t, dT)["F"] / 1000 for t in t4s]
    ax.plot(t4s, Fs_, color=C[i])
    ax.annotate(lab, (t4s[-1], Fs_[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
ax.axhline(20, color=INK2, lw=1, ls="--"); ax.text(1405, 20.25, "20 kN requirement", fontsize=7.5, color=INK2)
ax.axvline(TAKEOFF_T4_LIMIT, color=INK2, lw=1, ls=":"); ax.text(TAKEOFF_T4_LIMIT - 4, ax.get_ylim()[0] + 0.5, "Tt4 limit\n1650 K", fontsize=7.5, color=INK2, ha="right")
ax.set_xlim(1400, 1735)
ax.set_xlabel("Takeoff turbine inlet temperature [K]"); ax.set_ylabel("Sea-level static thrust [kN]")
ax.set_title("Takeoff thrust of the engine sized at cruise")
save(fig, "fig_takeoff.png")

# ---------- save ----------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k != "gas"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

for name, rows in [("sweep_fpr_bpr.csv", sweep_fpr), ("sweep_opr_t4.csv", sweep_opr), ("t4_selection.csv", sel)]:
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with open(os.path.join(RES, "week2_results.json"), "w") as fh:
    json.dump(clean(out), fh, indent=2)
print("Done. Net thrust %.1f N, TSFC %.3f g/(kN s), m0 %.2f kg/s" % (d["Fs"] * d["m0"], g_kNs(d["tsfc"]), d["m0"]))
