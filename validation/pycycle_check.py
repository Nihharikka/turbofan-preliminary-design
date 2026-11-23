"""Independent cross-check of the design point (and takeoff) with NASA pyCycle.

pyCycle (Hendricks and Gray, 2019) is NASA's open-source cycle code built on
OpenMDAO. It uses chemical-equilibrium (CEA) thermodynamics, isentropic
component efficiencies and, off-design, scaled generic component maps.

Run from the repository root:   python validation/pycycle_check.py
Requires:  pip install om-pycycle
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openmdao.api as om
import pycycle.api as pyc

from turbofan.cycle import run
from turbofan.design_case import WEEK2 as FINAL
from turbofan.offdesign import run_offdesign
from turbofan.sizing import size_engine

LBF = 4.4482216152605
FT = 0.3048


def turbine_isentropic(s_in, s_out):
    g = s_in["gas"]
    T_s = g.T_from_phi(g.phi(s_in["Tt"]) + g.R * math.log(s_out["pt"] / s_in["pt"]))
    return (g.h(s_in["Tt"]) - g.h(s_out["Tt"])) / (g.h(s_in["Tt"]) - g.h(T_s))


class TF(pyc.Cycle):
    """Two-spool separate-flow turbofan, same layout as the project model."""

    def initialize(self):
        self.options.declare("throttle_mode", default="T4", values=["T4"])
        super().initialize()

    def setup(self):
        design = self.options["design"]
        self.options["thermo_method"] = "CEA"
        self.options["thermo_data"] = pyc.species_data.janaf
        self.add_subsystem("fc", pyc.FlightConditions())
        self.add_subsystem("inlet", pyc.Inlet())
        self.add_subsystem("fan", pyc.Compressor(map_data=pyc.FanMap, bleed_names=[], map_extrap=True),
                           promotes_inputs=[("Nmech", "LP_Nmech")])
        self.add_subsystem("splitter", pyc.Splitter())
        self.add_subsystem("duct25", pyc.Duct())
        self.add_subsystem("hpc", pyc.Compressor(map_data=pyc.HPCMap, bleed_names=[], map_extrap=True),
                           promotes_inputs=[("Nmech", "HP_Nmech")])
        self.add_subsystem("burner", pyc.Combustor(fuel_type="Jet-A(g)"))
        self.add_subsystem("hpt", pyc.Turbine(map_data=pyc.HPTMap, bleed_names=[], map_extrap=True),
                           promotes_inputs=[("Nmech", "HP_Nmech")])
        self.add_subsystem("lpt", pyc.Turbine(map_data=pyc.LPTMap, bleed_names=[], map_extrap=True),
                           promotes_inputs=[("Nmech", "LP_Nmech")])
        self.add_subsystem("duct7", pyc.Duct())
        self.add_subsystem("core_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
        self.add_subsystem("duct17", pyc.Duct())
        self.add_subsystem("byp_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
        self.add_subsystem("lp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "LP_Nmech")])
        self.add_subsystem("hp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "HP_Nmech")])
        self.add_subsystem("perf", pyc.Performance(num_nozzles=2, num_burners=1))

        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("hpc.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("core_nozz.Fg", "perf.Fg_0")
        self.connect("byp_nozz.Fg", "perf.Fg_1")
        self.connect("fan.trq", "lp_shaft.trq_0")
        self.connect("lpt.trq", "lp_shaft.trq_1")
        self.connect("hpc.trq", "hp_shaft.trq_0")
        self.connect("hpt.trq", "hp_shaft.trq_1")
        self.connect("fc.Fl_O:stat:P", "core_nozz.Ps_exhaust")
        self.connect("fc.Fl_O:stat:P", "byp_nozz.Ps_exhaust")

        bal = self.add_subsystem("balance", om.BalanceComp())
        bal.add_balance("FAR", val=0.022, lower=1e-4, eq_units="degR")
        self.connect("balance.FAR", "burner.Fl_I:FAR")
        self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
        self.promotes("balance", inputs=[("rhs:FAR", "T4_MAX")])
        if design:
            bal.add_balance("W", units="lbm/s", eq_units="lbf")
            self.connect("balance.W", "fc.W")
            self.connect("perf.Fn", "balance.lhs:W")
            self.promotes("balance", inputs=[("rhs:W", "Fn_DES")])
            bal.add_balance("lpt_PR", val=3.0, lower=1.001, upper=8, eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.lpt_PR", "lpt.PR")
            self.connect("lp_shaft.pwr_in_real", "balance.lhs:lpt_PR")
            self.connect("lp_shaft.pwr_out_real", "balance.rhs:lpt_PR")
            bal.add_balance("hpt_PR", val=3.0, lower=1.001, upper=8, eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.hpt_PR", "hpt.PR")
            self.connect("hp_shaft.pwr_in_real", "balance.lhs:hpt_PR")
            self.connect("hp_shaft.pwr_out_real", "balance.rhs:hpt_PR")
        else:
            bal.add_balance("W", units="lbm/s", lower=5., upper=400., eq_units="inch**2")
            self.connect("balance.W", "fc.W")
            self.connect("core_nozz.Throat:stat:area", "balance.lhs:W")
            bal.add_balance("BPR", lower=2., upper=10., eq_units="inch**2")
            self.connect("balance.BPR", "splitter.BPR")
            self.connect("byp_nozz.Throat:stat:area", "balance.lhs:BPR")
            bal.add_balance("lp_Nmech", val=10000., units="rpm", lower=500., eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.lp_Nmech", "LP_Nmech")
            self.connect("lp_shaft.pwr_in_real", "balance.lhs:lp_Nmech")
            self.connect("lp_shaft.pwr_out_real", "balance.rhs:lp_Nmech")
            bal.add_balance("hp_Nmech", val=30000., units="rpm", lower=500., eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.hp_Nmech", "HP_Nmech")
            self.connect("hp_shaft.pwr_in_real", "balance.lhs:hp_Nmech")
            self.connect("hp_shaft.pwr_out_real", "balance.rhs:hp_Nmech")

        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
        self.pyc_connect_flow("inlet.Fl_O", "fan.Fl_I")
        self.pyc_connect_flow("fan.Fl_O", "splitter.Fl_I")
        self.pyc_connect_flow("splitter.Fl_O1", "duct25.Fl_I")
        self.pyc_connect_flow("duct25.Fl_O", "hpc.Fl_I")
        self.pyc_connect_flow("hpc.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "hpt.Fl_I")
        self.pyc_connect_flow("hpt.Fl_O", "lpt.Fl_I")
        self.pyc_connect_flow("lpt.Fl_O", "duct7.Fl_I")
        self.pyc_connect_flow("duct7.Fl_O", "core_nozz.Fl_I")
        self.pyc_connect_flow("splitter.Fl_O2", "duct17.Fl_I")
        self.pyc_connect_flow("duct17.Fl_O", "byp_nozz.Fl_I")

        newton = self.nonlinear_solver = om.NewtonSolver()
        for k, v in dict(atol=1e-8, rtol=1e-99, iprint=-1, maxiter=60, solve_subsystems=True,
                         max_sub_solves=1000, reraise_child_analysiserror=False).items():
            newton.options[k] = v
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options["maxiter"] = 3
        ls.options["rho"] = 0.75
        self.linear_solver = om.DirectSolver()
        super().setup()


# Heat of formation of the products of burning 1 kg of C12H23 (CO2 and H2O vapour),
# JANAF values as used by pyCycle: 12 x -393.51 + 11.5 x -241.826 kJ/mol fuel.
M_FUEL = (12 * 12.011 + 23 * 1.008) / 1000.0
H_PRODUCTS_PER_KG_FUEL = -(12 * 393.51e3 + 11.5 * 241.826e3) / M_FUEL   # J/kg


def fuel_enthalpy_for_lhv(lhv):
    """Fuel enthalpy (J/kg, element reference) that gives pyCycle the same heating value.

    pyCycle adds the fuel as elements; with its default fuel enthalpy of zero the
    implied heating value is about 44.8 MJ/kg, roughly 4% above Jet A-1.
    """
    return lhv + H_PRODUCTS_PER_KG_FUEL


def build(des, sz, fuel_h=None):
    inp = des["inputs"]

    class MP(pyc.MPCycle):
        def setup(self):
            self.pyc_add_pnt("DESIGN", TF(thermo_method="CEA"))
            for k, v in {"inlet.MN": 0.62, "fan.MN": 0.45, "splitter.MN1": 0.45, "splitter.MN2": 0.45,
                         "duct25.MN": 0.50, "hpc.MN": 0.25, "burner.MN": 0.10, "hpt.MN": 0.40,
                         "lpt.MN": 0.40, "duct7.MN": 0.40, "duct17.MN": 0.45}.items():
                self.set_input_defaults("DESIGN." + k, v)
            self.set_input_defaults("DESIGN.splitter.BPR", inp["bpr"])
            self.set_input_defaults("DESIGN.LP_Nmech", sz["fan"]["N_rpm"], units="rpm")
            self.set_input_defaults("DESIGN.HP_Nmech", sz["hpc_axial"]["N_rpm"], units="rpm")
            self.pyc_add_cycle_param("inlet.ram_recovery", inp["inlet_recovery"])
            self.pyc_add_cycle_param("duct25.dPqP", 1 - inp["transition_duct"])
            self.pyc_add_cycle_param("burner.dPqP", inp["burner_dp"])
            self.pyc_add_cycle_param("duct7.dPqP", 1 - inp["core_jetpipe"])
            self.pyc_add_cycle_param("duct17.dPqP", 1 - inp["bypass_duct"])
            self.pyc_add_cycle_param("core_nozz.Cv", 1.0)
            self.pyc_add_cycle_param("byp_nozz.Cv", 1.0)
            self.pyc_add_cycle_param("lp_shaft.fracLoss", 1 - inp["eta_mech"])
            self.pyc_add_cycle_param("hp_shaft.fracLoss", 1 - inp["eta_mech"])
            self.pyc_add_cycle_param("hp_shaft.HPX", 0.0, units="hp")
            self.pyc_add_cycle_param("lp_shaft.HPX", 0.0, units="hp")
            self.pyc_add_pnt("TAKEOFF", TF(design=False, thermo_method="CEA"))
            self.set_input_defaults("TAKEOFF.fc.MN", 0.001)
            self.set_input_defaults("TAKEOFF.fc.alt", 0.0, units="ft")
            self.set_input_defaults("TAKEOFF.fc.dTs", 0.0, units="degR")
            self.pyc_use_default_des_od_conns()
            self.pyc_connect_des_od("core_nozz.Throat:stat:area", "balance.rhs:W")
            self.pyc_connect_des_od("byp_nozz.Throat:stat:area", "balance.rhs:BPR")
            super().setup()

    prob = om.Problem(reports=False)
    prob.model = MP()
    prob.setup()
    st = des["stations"]
    prob.set_val("DESIGN.fan.PR", inp["fpr"])
    prob.set_val("DESIGN.fan.eff", des["eta_is_fan"])
    prob.set_val("DESIGN.hpc.PR", des["hpc_pr"])
    prob.set_val("DESIGN.hpc.eff", des["eta_is_hpc"])
    prob.set_val("DESIGN.hpt.eff", turbine_isentropic(st["4"], st["45"]))
    prob.set_val("DESIGN.lpt.eff", turbine_isentropic(st["45"], st["5"]))
    prob.set_val("DESIGN.fc.alt", inp["altitude"] / FT, units="ft")
    prob.set_val("DESIGN.fc.MN", inp["mach"])
    prob.set_val("DESIGN.T4_MAX", inp["Tt4"], units="degK")
    prob.set_val("DESIGN.Fn_DES", inp["thrust_required"], units="N")
    prob["DESIGN.balance.FAR"] = 0.022
    prob["DESIGN.balance.W"] = des["m0"] / 0.4536
    prob["DESIGN.balance.lpt_PR"] = des["lpt_pr"]
    prob["DESIGN.balance.hpt_PR"] = des["hpt_pr"]
    prob["DESIGN.fc.balance.Pt"] = 4.3
    prob["DESIGN.fc.balance.Tt"] = 440.0
    prob["TAKEOFF.balance.FAR"] = 0.024
    prob["TAKEOFF.balance.W"] = 127.0
    prob["TAKEOFF.balance.BPR"] = 5.2
    prob["TAKEOFF.balance.lp_Nmech"] = sz["fan"]["N_rpm"]
    prob["TAKEOFF.balance.hp_Nmech"] = sz["hpc_axial"]["N_rpm"]
    prob["TAKEOFF.hpt.PR"] = des["hpt_pr"]
    prob["TAKEOFF.lpt.PR"] = 3.1
    prob["TAKEOFF.fan.map.RlineMap"] = 2.0
    prob["TAKEOFF.hpc.map.RlineMap"] = 2.0
    if fuel_h is not None:
        for pt in ("DESIGN", "TAKEOFF"):
            prob.set_val(pt + ".burner.mix_fuel.mix:h", fuel_h / 2326.0, units="Btu/lbm")
    prob.set_solver_print(level=-1)
    return prob


def collect(prob):
    g = lambda k, u=None: float(prob.get_val(k, units=u)[0]) if u else float(prob.get_val(k)[0])
    return {
        "Fn_N": g("DESIGN.perf.Fn", "N"), "W_kg_s": g("DESIGN.inlet.Fl_O:stat:W", "kg/s"),
        "TSFC_g_kNs": g("DESIGN.perf.TSFC", "lbm/h/lbf") / 3600 / 9.80665 * 1e6,
        "FAR": g("DESIGN.balance.FAR"), "OPR": g("DESIGN.perf.OPR"),
        "hpt_PR": g("DESIGN.balance.hpt_PR"), "lpt_PR": g("DESIGN.balance.lpt_PR"),
        "Tt13": g("DESIGN.fan.Fl_O:tot:T", "degK"), "Tt3": g("DESIGN.hpc.Fl_O:tot:T", "degK"),
        "pt3_kPa": g("DESIGN.hpc.Fl_O:tot:P", "kPa"),
        "Tt45": g("DESIGN.hpt.Fl_O:tot:T", "degK"), "Tt5": g("DESIGN.lpt.Fl_O:tot:T", "degK"),
        "A8": g("DESIGN.core_nozz.Throat:stat:area", "m**2"), "A18": g("DESIGN.byp_nozz.Throat:stat:area", "m**2"),
        "Fg_total_N": g("DESIGN.core_nozz.Fg", "N") + g("DESIGN.byp_nozz.Fg", "N"), "Fram_N": g("DESIGN.inlet.F_ram", "N"),
        "TO_Fn_N": g("TAKEOFF.perf.Fn", "N"), "TO_W_kg_s": g("TAKEOFF.inlet.Fl_O:stat:W", "kg/s"),
        "TO_TSFC_g_kNs": g("TAKEOFF.perf.TSFC", "lbm/h/lbf") / 3600 / 9.80665 * 1e6,
        "TO_BPR": g("TAKEOFF.balance.BPR"), "TO_fanPR": g("TAKEOFF.fan.PR"), "TO_OPR": g("TAKEOFF.perf.OPR"),
        "TO_LP_N_pct": g("TAKEOFF.balance.lp_Nmech", "rpm") / g("DESIGN.LP_Nmech", "rpm") * 100,
        "TO_HP_N_pct": g("TAKEOFF.balance.hp_Nmech", "rpm") / g("DESIGN.HP_Nmech", "rpm") * 100,
        "TO_fan_eff": g("TAKEOFF.fan.eff"), "TO_hpc_eff": g("TAKEOFF.hpc.eff"),
        "TO_hpt_eff": g("TAKEOFF.hpt.eff"), "TO_lpt_eff": g("TAKEOFF.lpt.eff"),
    }


def project(des, od):
    st = des["stations"]
    return {"Fn_N": des["Fs"] * des["m0"], "W_kg_s": des["m0"], "TSFC_g_kNs": des["tsfc"] * 1e6, "FAR": des["f"],
            "OPR": des["inputs"]["opr"], "hpt_PR": des["hpt_pr"], "lpt_PR": des["lpt_pr"], "Tt13": st["13"]["Tt"],
            "Tt3": st["3"]["Tt"], "pt3_kPa": st["3"]["pt"] / 1000, "Tt45": st["45"]["Tt"], "Tt5": st["5"]["Tt"],
            "A8": des["A8"], "A18": des["A18"], "Fg_total_N": des["Fs"] * des["m0"] + des["m0"] * des["V0"],
            "Fram_N": des["m0"] * des["V0"], "TO_Fn_N": od["F"], "TO_W_kg_s": od["m0"],
            "TO_TSFC_g_kNs": od["tsfc"] * 1e6, "TO_BPR": od["bpr"], "TO_fanPR": od["fpr"], "TO_OPR": od["opr"]}


def main():
    from dataclasses import replace
    T4_to = 1569.3675938084334
    sz = size_engine(run(FINAL))
    cases = {}
    # Case A: pyCycle defaults for the fuel; project model as designed (eta_b 0.995, LHV 43.0)
    desA = run(FINAL)
    pA = build(desA, sz); pA.set_val("TAKEOFF.T4_MAX", T4_to, units="degK"); pA.run_model()
    cases["A_pycycle_default_fuel"] = {"pycycle": collect(pA), "project": project(desA, run_offdesign(desA, 0, 0, T4_to))}
    # Case B: same fuel energy in both codes (project eta_b = 1, pyCycle fuel enthalpy matched to LHV 43.0 MJ/kg)
    desB = run(replace(FINAL, eta_burner=1.0))
    hB = fuel_enthalpy_for_lhv(FINAL.lhv)
    pB = build(desB, sz, fuel_h=hB); pB.set_val("TAKEOFF.T4_MAX", T4_to, units="degK"); pB.run_model()
    cases["B_matched_fuel_energy"] = {"pycycle": collect(pB), "project": project(desB, run_offdesign(desB, 0, 0, T4_to)),
                                      "fuel_enthalpy_J_per_kg": hB,
                                      "pycycle_default_implied_LHV": -H_PRODUCTS_PER_KG_FUEL}
    st = desA["stations"]
    cases["isentropic_efficiencies_used"] = {"fan": desA["eta_is_fan"], "hpc": desA["eta_is_hpc"],
                                             "hpt": turbine_isentropic(st["4"], st["45"]), "lpt": turbine_isentropic(st["45"], st["5"])}
    cases["takeoff_Tt4"] = T4_to
    os.makedirs("results", exist_ok=True)
    with open("results/pycycle_comparison.json", "w") as fh:
        json.dump(cases, fh, indent=2)
    for name in ("A_pycycle_default_fuel", "B_matched_fuel_energy"):
        print("\n" + name)
        print(f"{'quantity':16s}{'project':>12s}{'pyCycle':>12s}{'diff %':>9s}")
        m, p = cases[name]["project"], cases[name]["pycycle"]
        for k in m:
            print(f"{k:16s}{m[k]:12.4f}{p[k]:12.4f}{(p[k] / m[k] - 1) * 100:9.2f}")
    print("\npyCycle off-design (case B):", {k: round(v, 3) for k, v in cases["B_matched_fuel_energy"]["pycycle"].items() if k.startswith("TO_")})


if __name__ == "__main__":
    main()
