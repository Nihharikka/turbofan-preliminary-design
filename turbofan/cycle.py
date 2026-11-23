"""Design-point cycle of a two-spool, separate-flow turbofan.

Station numbers follow SAE AS755:
 0 freestream, 2 fan face, 13 fan bypass exit, 17 bypass nozzle inlet,
 18 bypass nozzle throat (exit), 21 fan core exit, 25 HPC inlet, 3 HPC exit,
 4 combustor exit / HPT inlet, 45 HPT exit / LPT inlet, 5 LPT exit,
 7 core nozzle inlet, 8 core nozzle throat (exit).
"""
from dataclasses import dataclass, asdict
from . import components as c
from .atmosphere import isa
from .thermo import air, IdealGas, LHV_JETA1


@dataclass
class DesignInputs:
    # Flight condition
    altitude: float = 12000.0      # m
    mach: float = 0.80
    # Cycle
    bpr: float = 4.5               # bypass ratio, m13 / m21
    fpr: float = 1.70              # fan pressure ratio, bypass (outer) side
    fpr_core: float = None         # fan pressure ratio, core (hub) side; None = same as fpr
    opr: float = 23.0              # overall pressure ratio pt3 / pt2
    Tt4: float = 1500.0            # K, turbine inlet temperature
    # Component performance
    inlet_recovery: float = 0.99
    eta_p_fan: float = 0.89
    eta_p_hpc: float = 0.88
    transition_duct: float = 0.99  # pt25 / pt21
    burner_dp: float = 0.04        # fractional total pressure loss
    eta_burner: float = 0.995
    eta_p_hpt: float = 0.89
    eta_p_lpt: float = 0.90
    eta_mech: float = 0.99
    bypass_duct: float = 0.985     # pt17 / pt13
    core_jetpipe: float = 0.99     # pt7 / pt5
    lhv: float = LHV_JETA1
    # Turbine cooling air, as fractions of HPC exit (core) flow, bled at station 3
    cool_ngv: float = 0.0          # re-enters before the HPT rotor (does HPT work)
    cool_rotor: float = 0.0        # re-enters after the HPT rotor (does LPT work only)
    # If set, Tt4 is adjusted so that the HPT rotor inlet temperature Tt41 equals this value
    rotor_inlet_T: float = None
    # Requirement
    thrust_required: float = 4000.0  # N, net thrust per engine at this condition
    fully_expanded_nozzles: bool = False  # True only for verification
    gas_model: str = "real"        # "real" or "ideal" (verification only)
    ideal_neglect_fuel: bool = False


def run(inp: DesignInputs):
    """Run the design-point cycle. Returns a results dict with every station.

    If inp.rotor_inlet_T is set, the combustor exit temperature Tt4 is found so that
    the rotor inlet temperature Tt41 (after NGV cooling air mixes in) equals it.
    """
    if inp.rotor_inlet_T is not None and (inp.cool_ngv > 0):
        from dataclasses import replace
        from scipy.optimize import brentq
        T4 = brentq(lambda t: _run(replace(inp, Tt4=t, rotor_inlet_T=None))["stations"]["41"]["Tt"] - inp.rotor_inlet_T,
                    inp.rotor_inlet_T, inp.rotor_inlet_T + 300, xtol=1e-6)
        r = _run(replace(inp, Tt4=T4, rotor_inlet_T=None))
        r["inputs"]["rotor_inlet_T"] = inp.rotor_inlet_T
        return r
    if inp.rotor_inlet_T is not None:
        from dataclasses import replace
        return _run(replace(inp, Tt4=inp.rotor_inlet_T))
    return _run(inp)


def _run(inp: DesignInputs):
    gas = air() if inp.gas_model == "real" else IdealGas()
    T0, p0 = isa(inp.altitude)
    s0, V0 = c.freestream(T0, p0, inp.mach, gas)
    st = {"0": s0}

    st["2"] = c.duct(s0, inp.inlet_recovery)
    st["13"], w_fan = c.compressor(st["2"], inp.fpr, inp.eta_p_fan)
    fpr_core = inp.fpr if inp.fpr_core is None else inp.fpr_core
    st["21"], w_fan_core = c.compressor(st["2"], fpr_core, inp.eta_p_fan)
    st["25"] = c.duct(st["21"], inp.transition_duct)
    hpc_pr = inp.opr / (fpr_core * inp.transition_duct)
    st["3"], w_hpc = c.compressor(st["25"], hpc_pr, inp.eta_p_hpc)
    from .thermo import mix
    m_b = 1.0 - inp.cool_ngv - inp.cool_rotor          # burner air per kg core air
    st["4"], f_b = c.combustor(st["3"], inp.Tt4, inp.burner_dp, inp.eta_burner, inp.lhv, inp.gas_model)
    f = f_b * m_b                                       # fuel per kg core air
    mf = 0.0 if inp.ideal_neglect_fuel else f
    m4 = m_b * (1 + (0.0 if inp.ideal_neglect_fuel else f_b))

    def _mix(sa, ma, sb, mb):
        if mb <= 0:
            return sa
        g = mix(sa["gas"], ma, sb["gas"], mb) if hasattr(sa["gas"], "x") else sa["gas"]
        h = (ma * sa["gas"].h(sa["Tt"]) + mb * sb["gas"].h(sb["Tt"])) / (ma + mb)
        return c.state(g.T_from_h(h), sa["pt"], g)

    # NGV cooling air mixes in before the rotor (station 41)
    st["41"] = _mix(st["4"], m4, st["3"], inp.cool_ngv)
    m41 = m4 + inp.cool_ngv
    # HP spool: turbine work per kg of rotor gas
    w_hpt = w_hpc / (inp.eta_mech * m41)
    st["44"], hpt_pr = c.turbine_for_work(st["41"], w_hpt, inp.eta_p_hpt)
    # rotor cooling air rejoins after the HPT rotor (station 45)
    st["45"] = _mix(st["44"], m41, st["3"], inp.cool_rotor)
    m45 = m41 + inp.cool_rotor                          # = 1 + mf
    # LP spool drives the fan (bypass and core sides)
    w_lpt = (inp.bpr * w_fan + w_fan_core) / (inp.eta_mech * m45)
    st["5"], lpt_pr = c.turbine_for_work(st["45"], w_lpt, inp.eta_p_lpt)

    st["7"] = c.duct(st["5"], inp.core_jetpipe)
    st["17"] = c.duct(st["13"], inp.bypass_duct)
    if st["7"]["pt"] <= p0 * 1.01 or st["17"]["pt"] <= p0 * 1.01:
        raise ValueError("Infeasible cycle: the core cannot drive the fan at these settings")
    nz_core = c.convergent_nozzle(st["7"], p0, inp.fully_expanded_nozzles)
    nz_byp = c.convergent_nozzle(st["17"], p0, inp.fully_expanded_nozzles)

    # Thrust per kg/s of core air and of bypass air
    F_core = (1 + mf) * nz_core["V"] - V0 + (nz_core["p"] - p0) * (1 + mf) / nz_core["flux"]
    F_byp = nz_byp["V"] - V0 + (nz_byp["p"] - p0) / nz_byp["flux"]
    Fs = (F_core + inp.bpr * F_byp) / (1 + inp.bpr)     # N per kg/s total air
    tsfc = f / ((1 + inp.bpr) * Fs)                     # kg/(N s)

    # Sizing
    m0 = inp.thrust_required / Fs
    m_core = m0 / (1 + inp.bpr)
    m_byp = m0 - m_core
    m_fuel = f * m_core
    A8 = m_core * (1 + mf) / nz_core["flux"]
    A18 = m_byp / nz_byp["flux"]

    # Efficiencies (per kg core air basis). Effective jet velocities include
    # pressure thrust, so that choked nozzles are treated consistently.
    Veff_core = (F_core + V0) / (1 + mf)
    Veff_byp = F_byp + V0
    ke_out = 0.5 * (1 + mf) * Veff_core**2 + 0.5 * inp.bpr * Veff_byp**2
    ke_in = 0.5 * (1 + inp.bpr) * V0**2
    eta_th = (ke_out - ke_in) / (f * inp.lhv)
    eta_prop = (Fs * (1 + inp.bpr) * V0) / (ke_out - ke_in) if V0 > 0 else 0.0

    flows = {"0": m0, "2": m0, "13": m_byp, "17": m_byp, "18": m_byp, "21": m_core, "25": m_core,
             "3": m_core, "4": m_core * m4, "41": m_core * m41, "44": m_core * m41, "45": m_core + m_fuel,
             "5": m_core + m_fuel, "7": m_core + m_fuel, "8": m_core + m_fuel}
    st["8"] = {"Tt": st["7"]["Tt"], "pt": st["7"]["pt"], "gas": st["7"]["gas"]}
    st["18"] = {"Tt": st["17"]["Tt"], "pt": st["17"]["pt"], "gas": st["17"]["gas"]}

    return {
        "inputs": asdict(inp), "T0": T0, "p0": p0, "V0": V0, "stations": st, "flows": flows,
        "f": f, "f_burner": f_b, "m4_per_core": m4, "m41_per_core": m41, "w_fan": w_fan, "w_fan_core": w_fan_core, "fpr_core": fpr_core, "w_hpc": w_hpc, "w_hpt": w_hpt, "w_lpt": w_lpt,
        "hpc_pr": hpc_pr, "hpt_pr": hpt_pr, "lpt_pr": lpt_pr,
        "eta_is_fan": c.isentropic_efficiency(st["2"], st["13"]),
        "eta_is_hpc": c.isentropic_efficiency(st["25"], st["3"]),
        "nozzle_core": nz_core, "nozzle_bypass": nz_byp,
        "F_core_spec": F_core, "F_byp_spec": F_byp,
        "Fs": Fs, "tsfc": tsfc, "m0": m0, "m_core": m_core, "m_byp": m_byp, "m_fuel": m_fuel,
        "A8": A8, "A18": A18, "eta_th": eta_th, "eta_prop": eta_prop, "eta_overall": eta_th * eta_prop,
        "velocity_ratio": Veff_byp / Veff_core, "Veff_core": Veff_core, "Veff_byp": Veff_byp,
    }


def optimum_fpr(inp: DesignInputs, lo=1.3, hi=2.6):
    """Fan pressure ratio that minimises TSFC for the other inputs held fixed."""
    from scipy.optimize import minimize_scalar
    from dataclasses import replace

    def obj(fpr):
        try:
            return run(replace(inp, fpr=fpr))["tsfc"]
        except Exception:
            return 1.0

    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return r.x
