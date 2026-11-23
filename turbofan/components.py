"""Component models. Each function works on total (stagnation) conditions.

A flow state is a small dict: {"Tt": K, "pt": Pa, "gas": Gas}.
"""
import math
from scipy.optimize import brentq
from .thermo import products


def state(Tt, pt, gas):
    return {"Tt": Tt, "pt": pt, "gas": gas}


def freestream(T0, p0, M0, gas):
    """Total conditions of the freestream (station 0)."""
    V0 = M0 * gas.sound_speed(T0)
    Tt0 = gas.T_from_h(gas.h(T0) + 0.5 * V0**2)
    pt0 = p0 * math.exp((gas.phi(Tt0) - gas.phi(T0)) / gas.R)
    return state(Tt0, pt0, gas), V0


def duct(s, pt_ratio):
    """Adiabatic duct with total pressure recovery pt_out/pt_in."""
    return state(s["Tt"], s["pt"] * pt_ratio, s["gas"])


def compressor(s, pr, eta_p):
    """Compression with polytropic efficiency. Returns exit state and specific work [J/kg]."""
    g = s["gas"]
    phi_out = g.phi(s["Tt"]) + g.R * math.log(pr) / eta_p
    Tt_out = g.T_from_phi(phi_out)
    work = g.h(Tt_out) - g.h(s["Tt"])
    return state(Tt_out, s["pt"] * pr, g), work


def isentropic_efficiency(s_in, s_out):
    """Isentropic efficiency of a compression between two states (for reporting)."""
    g = s_in["gas"]
    phi_s = g.phi(s_in["Tt"]) + g.R * math.log(s_out["pt"] / s_in["pt"])
    Tts = g.T_from_phi(phi_s)
    return (g.h(Tts) - g.h(s_in["Tt"])) / (g.h(s_out["Tt"]) - g.h(s_in["Tt"]))


def combustor(s, Tt4, dp_frac, eta_b, lhv, gas_model="real"):
    """Find fuel-air ratio to reach Tt4. Energy balance on sensible enthalpies:

        h_air(Tt3) + f * eta_b * LHV = (1 + f) * h_prod(Tt4)
    """
    h3 = s["gas"].h(s["Tt"])
    if gas_model == "ideal":
        g = s["gas"]
        f = (g.h(Tt4) - h3) / (eta_b * lhv)  # textbook ideal form: fuel mass neglected
        return state(Tt4, s["pt"] * (1 - dp_frac), g), f

    def resid(f):
        return (1 + f) * products(f).h(Tt4) - h3 - f * eta_b * lhv

    f = brentq(resid, 1e-6, 0.06, xtol=1e-12)
    return state(Tt4, s["pt"] * (1 - dp_frac), products(f)), f


def turbine_for_work(s, work, eta_p):
    """Turbine that delivers specific work [J/kg of turbine gas]. Returns exit state and pressure ratio."""
    g = s["gas"]
    Tt_out = g.T_from_h(g.h(s["Tt"]) - work)
    ln_pr = (g.phi(s["Tt"]) - g.phi(Tt_out)) / (eta_p * g.R)
    pr = math.exp(ln_pr)
    return state(Tt_out, s["pt"] / pr, g), pr


def turbine_for_pr(s, pr, eta_p):
    """Turbine with a fixed expansion ratio pt_in/pt_out. Returns exit state and specific work."""
    g = s["gas"]
    phi_out = g.phi(s["Tt"]) - eta_p * g.R * math.log(pr)
    Tt_out = g.T_from_phi(phi_out)
    return state(Tt_out, s["pt"] / pr, g), g.h(s["Tt"]) - g.h(Tt_out)


def _static(s, p):
    """Isentropic expansion from total state to static pressure p: returns T, V."""
    g = s["gas"]
    T = g.T_from_phi(g.phi(s["Tt"]) + g.R * math.log(p / s["pt"]))
    V = math.sqrt(max(2.0 * (g.h(s["Tt"]) - g.h(T)), 0.0))
    return T, V


def critical_pressure(s):
    """Static pressure at which the isentropic flow reaches Mach 1."""
    g = s["gas"]

    def resid(p):
        T, V = _static(s, p)
        return V - g.sound_speed(T)

    return brentq(resid, 0.35 * s["pt"], 0.75 * s["pt"], xtol=1e-6)


def convergent_nozzle(s, p_amb, fully_expanded=False):
    """Ideal convergent nozzle. Returns exit static p, T, V, Mach, choked flag, mass flux [kg/(s m^2)].

    fully_expanded=True expands to ambient pressure regardless of choking
    (ideal convergent-divergent nozzle; used only for textbook verification).
    """
    g = s["gas"]
    p_crit = critical_pressure(s) if s["pt"] > p_amb * 1.0001 else 0.0
    choked = p_amb <= p_crit
    p_exit = p_crit if (choked and not fully_expanded) else p_amb
    T, V = _static(s, p_exit)
    rho = p_exit / (g.R * T)
    return {"p": p_exit, "T": T, "V": V, "M": V / g.sound_speed(T), "choked": choked,
            "flux": rho * V, "p_crit": p_crit, "npr": s["pt"] / p_amb}
