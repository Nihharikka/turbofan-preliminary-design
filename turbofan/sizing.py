"""Preliminary mean-line sizing of fan, compressors and turbines.

Uses the station data from the design-point cycle. Every rule-of-thumb value
is an argument with a default, so it can be changed and its effect studied.
Methods follow Saravanamuttoo et al. (Gas Turbine Theory, ch. 4, 5, 7),
Walsh and Fletcher (Gas Turbine Performance, ch. 5) and Dixon and Hall.

Conventions
-----------
* Stage loading   psi = dh0 / U^2   (U = mean blade speed unless stated)
* Flow coefficient phi = Cx / U
* AN^2 is reported in in^2 * rpm^2 (industry convention), with A the annulus
  area at the turbine rotor exit.
"""
import math
from dataclasses import dataclass, field, asdict
from scipy.optimize import brentq

IN2_PER_M2 = 1550.0031


def static_from_mach(s, M):
    """Static T, p, V, density and mass flux for a total state at axial Mach M."""
    g = s["gas"]

    def resid(T):
        return g.h(s["Tt"]) - g.h(T) - 0.5 * (M * g.sound_speed(T)) ** 2

    T = brentq(resid, 0.4 * s["Tt"], s["Tt"])
    p = s["pt"] * math.exp((g.phi(T) - g.phi(s["Tt"])) / g.R)
    V = M * g.sound_speed(T)
    rho = p / (g.R * T)
    return {"T": T, "p": p, "V": V, "rho": rho, "flux": rho * V, "a": g.sound_speed(T)}


def annulus_from_tip(area, r_tip):
    r_hub2 = r_tip**2 - area / math.pi
    if r_hub2 <= 0:
        raise ValueError("Annulus area too large for this tip radius")
    return math.sqrt(r_hub2)


def annulus_from_hub_tip_ratio(area, nu):
    r_tip = math.sqrt(area / (math.pi * (1 - nu**2)))
    return nu * r_tip, r_tip


def annulus_from_mean(area, r_mean):
    h = area / (2 * math.pi * r_mean)
    return r_mean - h / 2, r_mean + h / 2


@dataclass
class SizingRules:
    # Fan
    fan_face_mach: float = 0.62          # axial Mach at fan face (typical 0.55 to 0.65)
    fan_hub_tip: float = 0.35            # fan inlet hub-to-tip radius ratio
    fan_tip_speed: float = 430.0         # m/s physical, at the design point
    fan_exit_mach: float = 0.45          # axial Mach at fan exit, both streams
    fan_aspect_ratio: float = 2.2        # blade height / chord
    fan_blades: int = 22
    # HPC
    hpc_inlet_mach: float = 0.50
    hpc_inlet_hub_tip: float = 0.55
    hpc_first_tip_speed: float = 420.0   # m/s, first axial rotor tip
    hpc_axial_stages: int = 4
    hpc_axial_pr: float = 3.5            # pressure ratio of the axial block
    hpc_axial_layout: str = "constant_hub"
    hpc_axial_exit_mach: float = 0.45
    impeller_slip: float = 0.90          # slip factor
    impeller_power_input: float = 1.04   # power input factor (disk friction, windage)
    diffuser_radius_ratio: float = 1.50  # diffuser exit radius / impeller tip radius (pipe diffuser)
    # Combustor
    combustor_ref_velocity: float = 25.0  # m/s, casing reference velocity
    combustor_length_to_height: float = 2.8
    # Turbines
    hpt_stage_loading_max: float = 1.6   # dh0/U^2 per stage, cooled HPT
    hpt_mean_speed: float = 430.0        # m/s
    hpt_exit_mach: float = 0.40
    hpt_inlet_mach: float = 0.15
    lpt_stage_loading_target: float = 2.0
    lpt_stages: int = 3
    lpt_exit_mach: float = 0.40
    # General layout
    row_gap_fraction: float = 0.30       # axial gap / row axial chord
    compressor_aspect_ratio: float = 1.6
    turbine_aspect_ratio: float = 1.8


def size_engine(des, rules: SizingRules = SizingRules()):
    st, fl = des["stations"], des["flows"]
    out = {"rules": asdict(rules)}

    # ---------------- Fan ----------------
    s2 = static_from_mach(st["2"], rules.fan_face_mach)
    A2 = fl["2"] / s2["flux"]
    r_hub2, r_tip2 = annulus_from_hub_tip_ratio(A2, rules.fan_hub_tip)
    omega_lp = rules.fan_tip_speed / r_tip2
    N_lp = omega_lp * 60 / (2 * math.pi)
    U_tip = rules.fan_tip_speed
    W_tip = math.hypot(s2["V"], U_tip)
    U_hub = omega_lp * r_hub2
    r_mean2 = math.sqrt((r_tip2**2 + r_hub2**2) / 2)   # equal-area mean radius
    U_mean = omega_lp * r_mean2
    w_fan = des["w_fan"]
    h_blade = r_tip2 - r_hub2
    chord = h_blade / rules.fan_aspect_ratio
    corr = math.sqrt(st["2"]["Tt"] / 288.15)
    out["fan"] = {
        "A2": A2, "r_tip": r_tip2, "r_hub": r_hub2, "D_tip": 2 * r_tip2, "r_mean": r_mean2,
        "N_rpm": N_lp, "omega": omega_lp, "U_tip": U_tip, "U_tip_corrected": U_tip / corr,
        "U_hub": U_hub, "Cx": s2["V"], "M_rel_tip": W_tip / s2["a"],
        "psi_tip": w_fan / U_tip**2, "psi_mean": w_fan / U_mean**2, "psi_hub": des["w_fan_core"] / U_hub**2,
        "phi_mean": s2["V"] / U_mean, "blade_height": h_blade, "chord": chord,
        "blades": rules.fan_blades, "pitch_chord_tip": 2 * math.pi * r_tip2 / rules.fan_blades / chord,
        "specific_flow": fl["2"] * corr / (st["2"]["pt"] / 101325) / A2,
    }

    # fan exit: core annulus from the hub, bypass annulus above the splitter
    s21 = static_from_mach(st["21"], rules.fan_exit_mach)
    A21 = fl["21"] / s21["flux"]
    r_hub21 = r_hub2 * 1.05                     # slight hub rise through the fan
    r_split = math.sqrt(r_hub21**2 + A21 / math.pi)
    s13 = static_from_mach(st["13"], rules.fan_exit_mach)
    A13 = fl["13"] / s13["flux"]
    r_split_out = r_split + 0.006               # splitter lip thickness
    r_tip13 = math.sqrt(r_split_out**2 + A13 / math.pi)
    out["fan_exit"] = {"A21": A21, "r_hub21": r_hub21, "r_split": r_split, "A13": A13,
                       "r_split_out": r_split_out, "r_tip13": r_tip13}

    # ---------------- HPC axial block ----------------
    from . import components as c
    s25 = st["25"]
    sa25 = static_from_mach(s25, rules.hpc_inlet_mach)
    A25 = fl["25"] / sa25["flux"]
    r_hub25, r_tip25 = annulus_from_hub_tip_ratio(A25, rules.hpc_inlet_hub_tip)
    omega_hp = rules.hpc_first_tip_speed / r_tip25
    N_hp = omega_hp * 60 / (2 * math.pi)
    s_ax, w_ax = c.compressor(s25, rules.hpc_axial_pr, des["inputs"]["eta_p_hpc"])
    sa_ax = static_from_mach(s_ax, rules.hpc_axial_exit_mach)
    A_ax = fl["25"] / sa_ax["flux"]
    if rules.hpc_axial_layout == "constant_tip":
        r_tip_ax, r_hub_ax = r_tip25, annulus_from_tip(A_ax, r_tip25)
    else:  # constant hub radius: tip radius falls so the impeller eye stays small
        r_hub_ax = r_hub25
        r_tip_ax = math.sqrt(r_hub_ax**2 + A_ax / math.pi)
    r_mean_avg = 0.5 * (0.5 * (r_hub25 + r_tip25) + 0.5 * (r_hub_ax + r_tip_ax))
    U_mean_ax = omega_hp * r_mean_avg
    w_stage = w_ax / rules.hpc_axial_stages
    W1 = math.hypot(sa25["V"], rules.hpc_first_tip_speed)
    out["hpc_axial"] = {
        "A25": A25, "r_hub_in": r_hub25, "r_tip": r_tip25, "r_hub_out": r_hub_ax, "r_tip_out": r_tip_ax, "A_out": A_ax,
        "N_rpm": N_hp, "omega": omega_hp, "stages": rules.hpc_axial_stages, "pr": rules.hpc_axial_pr,
        "pr_stage": rules.hpc_axial_pr ** (1 / rules.hpc_axial_stages), "work": w_ax,
        "psi_mean": w_stage / U_mean_ax**2, "U_mean": U_mean_ax, "phi_in": sa25["V"] / (omega_hp * 0.5 * (r_hub25 + r_tip25)),
        "M_rel_tip_first": W1 / sa25["a"], "blade_height_in": r_tip25 - r_hub25,
        "blade_height_out": r_tip_ax - r_hub_ax, "layout": rules.hpc_axial_layout, "Tt_out": s_ax["Tt"], "pt_out": s_ax["pt"],
    }

    # ---------------- Centrifugal stage ----------------
    w_cent = des["w_hpc"] - w_ax
    pr_cent = des["hpc_pr"] / rules.hpc_axial_pr
    U2 = math.sqrt(w_cent / (rules.impeller_slip * rules.impeller_power_input))
    r2 = U2 / omega_hp
    out["impeller"] = {
        "work": w_cent, "pr": pr_cent, "U2": U2, "r2": r2, "D2": 2 * r2,
        "r_eye_tip": r_tip_ax, "r_eye_hub": r_hub_ax, "eye_tip_speed": omega_hp * r_tip_ax,
        "eye_to_tip_ratio": r_tip_ax / r2,
        "eye_M_rel_tip": math.hypot(sa_ax["V"], omega_hp * r_tip_ax) / sa_ax["a"],
        "r_diffuser_exit": r2 * rules.diffuser_radius_ratio,
        "Tt_in": s_ax["Tt"], "Tt_out": st["3"]["Tt"],
    }
    # impeller exit width from radial velocity ~ 0.3 U2 (typical)
    sa3 = static_from_mach(st["3"], 0.30)  # approximate exit density
    out["impeller"]["b2"] = fl["3"] / (sa3["rho"] * 0.30 * U2 * 2 * math.pi * r2)

    # ---------------- Combustor ----------------
    g3 = st["3"]["gas"]
    rho3 = st["3"]["pt"] / (g3.R * st["3"]["Tt"])
    A_ref = fl["3"] / (rho3 * rules.combustor_ref_velocity)
    r_mean_cb = 0.5 * (r2 * rules.diffuser_radius_ratio * 0.85 + 0.6 * r2)
    h_ref = A_ref / (2 * math.pi * r_mean_cb)
    out["combustor"] = {"A_ref": A_ref, "r_mean": r_mean_cb, "height": h_ref,
                        "length": rules.combustor_length_to_height * h_ref,
                        "r_inner": r_mean_cb - h_ref / 2, "r_outer": r_mean_cb + h_ref / 2}

    # ---------------- HPT ----------------
    w_hpt = des["w_hpt"]
    n_hpt = math.ceil(w_hpt / (rules.hpt_stage_loading_max * rules.hpt_mean_speed**2) - 1e-9)
    r_m_hpt = rules.hpt_mean_speed / omega_hp
    k_in = "41" if "41" in st else "4"
    k_out = "44" if "44" in st else "45"
    sh4 = static_from_mach(st[k_in], rules.hpt_inlet_mach)
    A4 = fl[k_in] / sh4["flux"]
    sh45 = static_from_mach(st[k_out], rules.hpt_exit_mach)
    A45 = fl[k_out] / sh45["flux"]
    r4 = annulus_from_mean(A4, r_m_hpt)
    r45 = annulus_from_mean(A45, r_m_hpt)
    out["hpt"] = {"stages": n_hpt, "work": w_hpt, "U_mean": rules.hpt_mean_speed, "r_mean": r_m_hpt,
                  "psi": w_hpt / n_hpt / rules.hpt_mean_speed**2, "A_in": A4, "A_out": A45,
                  "r_hub_in": r4[0], "r_tip_in": r4[1], "r_hub_out": r45[0], "r_tip_out": r45[1],
                  "blade_height_out": r45[1] - r45[0], "phi_out": sh45["V"] / rules.hpt_mean_speed,
                  "AN2": A45 * IN2_PER_M2 * N_hp**2, "U_tip_out": omega_hp * r45[1]}

    # ---------------- LPT ----------------
    w_lpt = des["w_lpt"]
    U_lpt = math.sqrt(w_lpt / (rules.lpt_stages * rules.lpt_stage_loading_target))
    r_m_lpt = U_lpt / omega_lp
    sl5 = static_from_mach(st["5"], rules.lpt_exit_mach)
    A5 = fl["5"] / sl5["flux"]
    r5 = annulus_from_mean(A5, r_m_lpt)
    sl45 = static_from_mach(st["45"], 0.30)
    A45l = fl["45"] / sl45["flux"]
    r45l = annulus_from_mean(A45l, r_m_lpt)
    out["lpt"] = {"stages": rules.lpt_stages, "work": w_lpt, "U_mean": U_lpt, "r_mean": r_m_lpt,
                  "psi": w_lpt / rules.lpt_stages / U_lpt**2, "A_in": A45l, "A_out": A5,
                  "r_hub_in": r45l[0], "r_tip_in": r45l[1], "r_hub_out": r5[0], "r_tip_out": r5[1],
                  "blade_height_out": r5[1] - r5[0], "phi_out": sl5["V"] / U_lpt,
                  "AN2": A5 * IN2_PER_M2 * N_lp**2}

    # ---------------- Nozzles ----------------
    A8, A18 = des["A8"], des["A18"]
    r_plug = 0.45 * r5[0]                      # core nozzle tail-cone radius at throat
    r8_out = math.sqrt(r_plug**2 + A8 / math.pi)
    out["nozzles"] = {"A8": A8, "r8_inner": r_plug, "r8_outer": r8_out, "A18": A18}
    out["nacelle"] = {"D_max": 2 * r_tip2 * 1.22}
    return out


def rows_axial_length(height, aspect_ratio, gap_fraction):
    chord = height / aspect_ratio
    return chord, chord * (1 + gap_fraction)
