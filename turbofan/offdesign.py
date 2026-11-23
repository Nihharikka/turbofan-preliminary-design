"""Simplified off-design (matching) model for the fixed-geometry engine.

Used to estimate takeoff thrust of the engine sized at the design point.
The method follows the classic textbook "performance analysis" approach
(Mattingly, Elements of Propulsion, off-design chapter) without component
maps. Assumptions:

* Component polytropic efficiencies and pressure losses stay at design values.
* HPT nozzle guide vanes and LPT nozzle guide vanes are choked, so the HPT
  expansion ratio stays at its design value and the corrected flow
  m4*sqrt(Tt4)/pt4 is constant.
* Nozzle throat areas A8 and A18 stay at their design values.
* The core-side fan pressure rise stays in the same proportion to the
  bypass-side rise as at the design point.

Unknowns: fan pressure ratio and LPT expansion ratio.
Equations: core nozzle continuity and bypass nozzle continuity, with
bypass ratio set by the LP spool power balance.
"""
import math
from dataclasses import replace
from scipy.optimize import fsolve
from . import components as c
from .atmosphere import isa
from .thermo import air


def _flow_param(m, s):
    return m * math.sqrt(s["Tt"]) / s["pt"]


def design_constants(des):
    """Fixed geometry quantities from a design-point result."""
    st = des["stations"]
    return {
        "Q4": _flow_param(des["flows"]["4"], st["4"]),
        "hpt_pr": des["hpt_pr"],
        "A8": des["A8"],
        "A18": des["A18"],
        "fpr_des": des["inputs"]["fpr"],
        "core_ratio": (des["fpr_core"] - 1) / (des["inputs"]["fpr"] - 1),
        "lpt_pr_des": des["lpt_pr"],
    }


def run_offdesign(des, altitude, mach, Tt4, dT_isa=0.0):
    """Operate the engine sized at `des` at a new flight condition and Tt4."""
    p = des["inputs"]
    k = design_constants(des)
    gas = air()
    T0, p0 = isa(altitude)
    T0 += dT_isa
    s0, V0 = c.freestream(T0, p0, mach, gas)
    s2 = c.duct(s0, p["inlet_recovery"])

    def core(fpr, lpt_pr):
        s13, w_fan = c.compressor(s2, fpr, p["eta_p_fan"])
        # core-side pressure rise kept in the same proportion as at the design point
        s21, w_fan_core = c.compressor(s2, 1 + k["core_ratio"] * (fpr - 1), p["eta_p_fan"])
        s25 = c.duct(s21, p["transition_duct"])
        # HP spool: HPT expansion ratio fixed (choked NGVs) -> HPT work -> HPC work -> HPC PR.
        # Cooling fractions are fixed at their design values.
        from .thermo import products, mix
        cn, cr = p.get("cool_ngv", 0.0), p.get("cool_rotor", 0.0)
        m_b = 1.0 - cn - cr

        def mixed(sa, ma, sb, mb):
            if mb <= 0:
                return sa
            g = mix(sa["gas"], ma, sb["gas"], mb)
            h = (ma * sa["gas"].h(sa["Tt"]) + mb * sb["gas"].h(sb["Tt"])) / (ma + mb)
            return c.state(g.T_from_h(h), sa["pt"], g)

        f_b = 0.02
        Tt3 = s25["Tt"] * 2.2
        for _ in range(40):
            s3t = c.state(Tt3, 1.0, s25["gas"])
            s4t = c.state(Tt4, 1.0, products(f_b))
            m4 = m_b * (1 + f_b)
            s41t = mixed(s4t, m4, s3t, cn)
            m41 = m4 + cn
            s44t, w_hpt = c.turbine_for_pr(s41t, k["hpt_pr"], p["eta_p_hpt"])
            w_hpc = w_hpt * p["eta_mech"] * m41
            Tt3_new = s25["gas"].T_from_h(s25["gas"].h(s25["Tt"]) + w_hpc)
            hpc_pr = math.exp((s25["gas"].phi(Tt3_new) - s25["gas"].phi(s25["Tt"])) * p["eta_p_hpc"] / s25["gas"].R)
            s3 = c.state(Tt3_new, s25["pt"] * hpc_pr, s25["gas"])
            s4, f_new = c.combustor(s3, Tt4, p["burner_dp"], p["eta_burner"], p["lhv"])
            done = abs(f_new - f_b) < 1e-11 and abs(Tt3_new - Tt3) < 1e-7
            f_b, Tt3 = f_new, Tt3_new
            if done:
                break
        m4 = m_b * (1 + f_b)
        f = f_b * m_b
        m4_abs = k["Q4"] * s4["pt"] / math.sqrt(Tt4)
        m_core = m4_abs / m4
        s41 = mixed(s4, m4, s3, cn)
        m41 = m4 + cn
        s44, w_hpt = c.turbine_for_pr(s41, k["hpt_pr"], p["eta_p_hpt"])
        s45 = mixed(s44, m41, s3, cr)
        s5, w_lpt = c.turbine_for_pr(s45, lpt_pr, p["eta_p_lpt"])
        bpr = (w_lpt * p["eta_mech"] * (1 + f) - w_fan_core) / w_fan
        m4 = m_core * (1 + f)   # total core-nozzle flow, used in the nozzle balance
        s7 = c.duct(s5, p["core_jetpipe"])
        s17 = c.duct(s13, p["bypass_duct"])
        nz8 = c.convergent_nozzle(s7, p0)
        nz18 = c.convergent_nozzle(s17, p0)
        return dict(s13=s13, s25=s25, s3=s3, s4=s4, s41=s41, s45=s45, s5=s5, s7=s7, s17=s17, f=f, m_core=m_core,
                    m4=m4, bpr=bpr, nz8=nz8, nz18=nz18, hpc_pr=hpc_pr, w_fan=w_fan)

    def resid(x):
        fpr, lpt_pr = x
        r = core(fpr, lpt_pr)
        e1 = r["nz8"]["flux"] * k["A8"] / r["m4"] - 1.0
        e2 = r["nz18"]["flux"] * k["A18"] / (r["bpr"] * r["m_core"]) - 1.0
        return [e1, e2]

    sol = None
    for scale in (1.0, 0.95, 1.05, 0.9, 1.1, 0.85):
        guess = [1 + (k["fpr_des"] - 1) * scale, k["lpt_pr_des"] * scale]
        try:
            x, info, ier, msg = fsolve(resid, guess, full_output=True, xtol=1e-11)
        except (ValueError, RuntimeError):
            continue
        if ier == 1 and max(abs(v) for v in resid(x)) < 1e-8:
            sol = x
            break
    if sol is None:
        raise RuntimeError("Off-design match did not converge")
    fpr, lpt_pr = sol
    r = core(fpr, lpt_pr)
    m_core, f, bpr = r["m_core"], r["f"], r["bpr"]
    m_byp = bpr * m_core
    nz8, nz18 = r["nz8"], r["nz18"]
    F = (m_core * (1 + f) * nz8["V"] + k["A8"] * (nz8["p"] - p0)
         + m_byp * nz18["V"] + k["A18"] * (nz18["p"] - p0) - (m_core + m_byp) * V0)
    m0 = m_core + m_byp
    return {"altitude": altitude, "mach": mach, "Tt4": Tt4, "T0": T0, "p0": p0, "F": F,
            "tsfc": f * m_core / F if F > 0 else float("nan"), "fpr": fpr, "lpt_pr": lpt_pr, "bpr": bpr,
            "hpc_pr": r["hpc_pr"], "opr": r["s3"]["pt"] / s2["pt"], "Tt3": r["s3"]["Tt"], "Tt45": r["s45"]["Tt"], "Tt41": r["s41"]["Tt"],
            "Tt5": r["s5"]["Tt"], "m0": m0, "m_core": m_core, "f": f, "Tt2": s2["Tt"], "pt2": s2["pt"],
            "m0_corr": m0 * math.sqrt(s2["Tt"] / 288.15) / (s2["pt"] / 101325.0),
            "choked8": nz8["choked"], "choked18": nz18["choked"], "V8": nz8["V"], "V18": nz18["V"]}


def run_offdesign_T41(des, altitude, mach, T41, dT_isa=0.0):
    """Off-design point with the HPT rotor inlet temperature Tt41 specified."""
    from scipy.optimize import brentq
    if des["inputs"].get("cool_ngv", 0.0) <= 0:
        return run_offdesign(des, altitude, mach, T41, dT_isa)
    T4 = brentq(lambda t: run_offdesign(des, altitude, mach, t, dT_isa)["Tt41"] - T41, T41, T41 + 300, xtol=1e-5)
    return run_offdesign(des, altitude, mach, T4, dT_isa)
