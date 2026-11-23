"""Verification tests. Run with:  python -m pytest -v tests/"""
import math
from dataclasses import replace
import pytest
from turbofan.atmosphere import isa
from turbofan.thermo import air, products, stoich_far
from turbofan.cycle import DesignInputs, run


def test_isa_against_iso2533_table():
    # ISO 2533 tabulated values
    assert isa(0) == pytest.approx((288.15, 101325.0), rel=1e-6)
    T, p = isa(11000)
    assert T == pytest.approx(216.65, abs=1e-3) and p == pytest.approx(22632, rel=1e-4)
    T, p = isa(12000)
    assert T == pytest.approx(216.65, abs=1e-3) and p == pytest.approx(19330, rel=1e-4)


def test_air_properties_against_tables():
    # Standard dry-air tables (e.g. Cengel & Boles Table A-2b): cp in kJ/(kg K)
    a = air()
    for T, cp in [(300, 1.005), (500, 1.029), (1000, 1.142)]:
        assert a.cp(T) / 1000 == pytest.approx(cp, abs=0.004)
    assert a.gamma(300) == pytest.approx(1.400, abs=0.002)


def test_stoichiometric_far():
    # Kerosene stoichiometric fuel-air ratio is about 0.068
    assert stoich_far() == pytest.approx(0.068, abs=0.001)


def mattingly_ideal_turbofan(M0, T0, gamma, cp, hpr, Tt4, pic, pif, alpha):
    """Closed-form ideal turbofan (Mattingly, Elements of Propulsion, ideal parametric analysis)."""
    R = (gamma - 1) / gamma * cp
    a0 = math.sqrt(gamma * R * T0)
    tau_r = 1 + (gamma - 1) / 2 * M0**2
    tau_lam = Tt4 / T0
    tau_c = pic ** ((gamma - 1) / gamma)
    tau_f = pif ** ((gamma - 1) / gamma)
    tau_t = 1 - tau_r / tau_lam * (tau_c - 1 + alpha * (tau_f - 1))
    V9_a0 = math.sqrt(2 / (gamma - 1) * tau_lam / (tau_r * tau_c) * (tau_r * tau_c * tau_t - 1))
    V19_a0 = math.sqrt(2 / (gamma - 1) * (tau_r * tau_f - 1))
    Fs = a0 / (1 + alpha) * (V9_a0 - M0 + alpha * (V19_a0 - M0))
    f = cp * T0 * (tau_lam - tau_r * tau_c) / hpr
    S = f / ((1 + alpha) * Fs)
    return Fs, S


@pytest.mark.parametrize("bpr,fpr,opr,Tt4", [(4.5, 1.7, 23, 1500), (2.0, 2.2, 15, 1400), (8.0, 1.5, 35, 1600)])
def test_cycle_code_matches_ideal_closed_form(bpr, fpr, opr, Tt4):
    """With ideal gas, ideal components and fully expanded nozzles, the station-by-station
    code must reproduce the textbook closed-form equations."""
    inp = DesignInputs(bpr=bpr, fpr=fpr, opr=opr, Tt4=Tt4, gas_model="ideal", ideal_neglect_fuel=True,
                       fully_expanded_nozzles=True, inlet_recovery=1, eta_p_fan=1, eta_p_hpc=1,
                       transition_duct=1, burner_dp=0, eta_burner=1, eta_p_hpt=1, eta_p_lpt=1,
                       eta_mech=1, bypass_duct=1, core_jetpipe=1)
    r = run(inp)
    T0, _ = isa(inp.altitude)
    Fs, S = mattingly_ideal_turbofan(inp.mach, T0, 1.4, 1004.5, inp.lhv, Tt4, opr, fpr, bpr)
    assert r["Fs"] == pytest.approx(Fs, rel=1e-6)
    assert r["tsfc"] == pytest.approx(S, rel=1e-6)


def test_spool_power_balances_close():
    r = run(DesignInputs())
    st, inp, f = r["stations"], DesignInputs(), r["f"]
    # HP spool
    hpc = st["3"]["gas"].h(st["3"]["Tt"]) - st["25"]["gas"].h(st["25"]["Tt"])
    hpt = (1 + f) * (st["4"]["gas"].h(st["4"]["Tt"]) - st["4"]["gas"].h(st["45"]["Tt"]))
    assert hpt * inp.eta_mech == pytest.approx(hpc, rel=1e-8)
    # LP spool
    fan = (1 + inp.bpr) * (st["13"]["gas"].h(st["13"]["Tt"]) - st["2"]["gas"].h(st["2"]["Tt"]))
    lpt = (1 + f) * (st["45"]["gas"].h(st["45"]["Tt"]) - st["45"]["gas"].h(st["5"]["Tt"]))
    assert lpt * inp.eta_mech == pytest.approx(fan, rel=1e-8)


def test_combustor_energy_balance():
    r = run(DesignInputs())
    st, f, inp = r["stations"], r["f"], DesignInputs()
    lhs = st["3"]["gas"].h(st["3"]["Tt"]) + f * inp.eta_burner * inp.lhv
    rhs = (1 + f) * products(f).h(st["4"]["Tt"])
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_thrust_matches_requirement_and_mass_balance():
    r = run(DesignInputs())
    assert r["Fs"] * r["m0"] == pytest.approx(4000.0, rel=1e-9)
    assert r["flows"]["8"] == pytest.approx(r["m_core"] + r["m_fuel"], rel=1e-12)
    assert r["m_byp"] / r["m_core"] == pytest.approx(4.5, rel=1e-12)


def test_offdesign_reproduces_design_point():
    """Running the matching model at the design condition must return the design point."""
    from turbofan.offdesign import run_offdesign
    from turbofan.design_case import WEEK2 as FINAL
    d = run(FINAL)
    o = run_offdesign(d, FINAL.altitude, FINAL.mach, FINAL.Tt4)
    assert o["F"] == pytest.approx(FINAL.thrust_required, rel=1e-6)
    assert o["fpr"] == pytest.approx(FINAL.fpr, rel=1e-6)
    assert o["bpr"] == pytest.approx(FINAL.bpr, rel=1e-6)
    assert o["opr"] == pytest.approx(FINAL.opr, rel=1e-6)


# ---------------- Week 3 additions: split fan and turbine cooling ----------------

def test_revA_spool_balances_with_cooling_and_split_fan():
    from turbofan.design_case import REV_A
    r = run(REV_A)
    st, f = r["stations"], r["f"]
    h = lambda k: st[k]["gas"].h(st[k]["Tt"])
    # HP spool: rotor gas (burner gas + NGV coolant) expands 41 -> 44
    w_hpc = h("3") - h("25")
    w_hpt = r["m41_per_core"] * (h("41") - st["41"]["gas"].h(st["44"]["Tt"]))
    assert w_hpt * REV_A.eta_mech == pytest.approx(w_hpc, rel=1e-8)
    # LP spool: bypass side and core side of the fan
    fan = REV_A.bpr * (h("13") - h("2")) + (h("21") - h("2"))
    lpt = (1 + f) * (h("45") - st["45"]["gas"].h(st["5"]["Tt"]))
    assert lpt * REV_A.eta_mech == pytest.approx(fan, rel=1e-8)
    assert st["41"]["Tt"] == pytest.approx(REV_A.rotor_inlet_T, abs=1e-4)


def test_revA_cooling_mixing_conserves_energy_and_mass():
    from turbofan.design_case import REV_A
    r = run(REV_A)
    st = r["stations"]
    h = lambda k, T=None: st[k]["gas"].h(st[k]["Tt"] if T is None else T)
    m4, m41 = r["m4_per_core"], r["m41_per_core"]
    assert m41 - m4 == pytest.approx(REV_A.cool_ngv, rel=1e-12)
    lhs = m4 * h("4") + REV_A.cool_ngv * h("3")
    assert m41 * h("41") == pytest.approx(lhs, rel=1e-9)
    lhs2 = m41 * st["41"]["gas"].h(st["44"]["Tt"]) + REV_A.cool_rotor * h("3")
    assert (1 + r["f"]) * h("45") == pytest.approx(lhs2, rel=1e-9)


def test_revA_offdesign_reproduces_design_point():
    from turbofan.offdesign import run_offdesign_T41
    from turbofan.design_case import REV_A
    d = run(REV_A)
    o = run_offdesign_T41(d, REV_A.altitude, REV_A.mach, REV_A.rotor_inlet_T)
    assert o["F"] == pytest.approx(REV_A.thrust_required, rel=1e-6)
    assert o["bpr"] == pytest.approx(REV_A.bpr, rel=1e-6)
    assert o["opr"] == pytest.approx(REV_A.opr, rel=1e-6)
