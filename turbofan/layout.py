"""Meridional (side-view) layout of the engine flowpath from the sizing results.

Produces axial positions of every blade row and duct and the hub/casing
lines of the core and bypass streams as (x, r) polylines in metres, with
x = 0 at the fan rotor leading edge at the tip. These lines are the
reference geometry for any later CAD work.

Axial lengths use simple preliminary-design rules (chord from blade height
and aspect ratio, axial gaps as a fraction of chord, duct lengths from
radius change). They are layout estimates, not detailed design.
"""
import math


def _row(h, ar, gap):
    c = h / ar
    return c, c * gap


def build_layout(sz):
    R = sz["rules"]
    fan, fx, ax, imp, cb, hpt, lpt, nz = (sz["fan"], sz["fan_exit"], sz["hpc_axial"], sz["impeller"],
                                          sz["combustor"], sz["hpt"], sz["lpt"], sz["nozzles"])
    rows = []          # (name, x_le, x_te, r_hub_le, r_tip_le, r_hub_te, r_tip_te, stream)
    x = 0.0
    # ---- fan rotor (axial chord ~ 0.65 x true chord because of stagger) ----
    c_fan = fan["chord"] * 0.65
    rows.append(("Fan rotor", x, x + c_fan, fan["r_hub"], fan["r_tip"], fx["r_hub21"], fan["r_tip"] * 0.995, "fan"))
    x_fan_te = x + c_fan
    # ---- splitter and bypass OGV ----
    x_split = x_fan_te + 0.35 * c_fan
    h_byp = fx["r_tip13"] - fx["r_split_out"]
    c_ogv = h_byp / 2.0
    x_ogv = x_split + 0.3 * c_ogv
    rows.append(("Bypass OGV", x_ogv, x_ogv + c_ogv, fx["r_split_out"], fx["r_tip13"], fx["r_split_out"], fx["r_tip13"], "bypass"))
    # ---- core: fan core-side stator then swan-neck duct to HPC ----
    h21 = fx["r_split"] - fx["r_hub21"]
    c_ccs = h21 / 1.5
    rows.append(("Core inlet stator", x_split, x_split + c_ccs, fx["r_hub21"], fx["r_split"], fx["r_hub21"], fx["r_split"], "core"))
    x = x_split + c_ccs
    dr = 0.5 * (fx["r_hub21"] + fx["r_split"]) - 0.5 * (ax["r_hub_in"] + ax["r_tip"])
    L_swan = max(2.2 * abs(dr), 0.06)
    x_hpc = x + L_swan
    # ---- HPC axial stages (constant hub) ----
    n = ax["stages"]
    x = x_hpc
    for i in range(n):
        for kind in ("rotor", "stator"):
            frac0 = (2 * i + (0 if kind == "rotor" else 1)) / (2 * n)
            frac1 = (2 * i + (1 if kind == "rotor" else 2)) / (2 * n)
            rt0 = ax["r_tip"] + (ax["r_tip_out"] - ax["r_tip"]) * frac0
            rt1 = ax["r_tip"] + (ax["r_tip_out"] - ax["r_tip"]) * frac1
            h = rt0 - ax["r_hub_in"]
            c, g = _row(h, R["compressor_aspect_ratio"], R["row_gap_fraction"])
            rows.append((f"HPC {kind} {i + 1}", x, x + c, ax["r_hub_in"], rt0, ax["r_hub_in"], rt1, "core"))
            x += c + g
    x_imp = x
    # ---- centrifugal impeller: axial length ~ 0.28 D2 ----
    L_imp = 0.28 * imp["D2"]
    rows.append(("Impeller", x_imp, x_imp + L_imp, imp["r_eye_hub"], imp["r_eye_tip"], imp["r2"], imp["r2"], "core"))
    x_imp_te = x_imp + L_imp
    # radial diffuser at the impeller exit plane, then deswirl bend back to axial
    x_diff = x_imp_te
    L_bend = 0.6 * (imp["r_diffuser_exit"] - cb["r_mean"]) + 0.03
    x_cb = x_diff + L_bend
    rows.append(("Combustor", x_cb, x_cb + cb["length"], cb["r_inner"], cb["r_outer"], cb["r_inner"], cb["r_outer"], "core"))
    x = x_cb + cb["length"] + 0.01
    # ---- HPT ----
    for i in range(hpt["stages"]):
        for kind in ("NGV", "rotor"):
            frac = (2 * i + (0 if kind == "NGV" else 1)) / (2 * hpt["stages"])
            rh0 = hpt["r_hub_in"] + (hpt["r_hub_out"] - hpt["r_hub_in"]) * frac
            rt0 = hpt["r_tip_in"] + (hpt["r_tip_out"] - hpt["r_tip_in"]) * frac
            fr1 = frac + 1 / (2 * hpt["stages"])
            rh1 = hpt["r_hub_in"] + (hpt["r_hub_out"] - hpt["r_hub_in"]) * fr1
            rt1 = hpt["r_tip_in"] + (hpt["r_tip_out"] - hpt["r_tip_in"]) * fr1
            c, g = _row(rt0 - rh0, R["turbine_aspect_ratio"] * 0.6, R["row_gap_fraction"])
            rows.append((f"HPT {kind} {i + 1}", x, x + c, rh0, rt0, rh1, rt1, "core"))
            x += c + g
    # ---- inter-turbine duct ----
    L_itd = max(2.5 * abs(lpt["r_mean"] - hpt["r_mean"]), 0.04)
    x += L_itd
    # ---- LPT ----
    for i in range(lpt["stages"]):
        for kind in ("NGV", "rotor"):
            frac = (2 * i + (0 if kind == "NGV" else 1)) / (2 * lpt["stages"])
            fr1 = frac + 1 / (2 * lpt["stages"])
            rh0 = lpt["r_hub_in"] + (lpt["r_hub_out"] - lpt["r_hub_in"]) * frac
            rt0 = lpt["r_tip_in"] + (lpt["r_tip_out"] - lpt["r_tip_in"]) * frac
            rh1 = lpt["r_hub_in"] + (lpt["r_hub_out"] - lpt["r_hub_in"]) * fr1
            rt1 = lpt["r_tip_in"] + (lpt["r_tip_out"] - lpt["r_tip_in"]) * fr1
            c, g = _row(rt0 - rh0, R["turbine_aspect_ratio"], R["row_gap_fraction"])
            rows.append((f"LPT {kind} {i + 1}", x, x + c, rh0, rt0, rh1, rt1, "core"))
            x += c + g
    x_lpt_te = x
    # ---- exhaust: turbine exit struts + convergent core nozzle with tail cone ----
    L_exh = 0.12
    L_noz = 0.18
    x_noz_exit = x_lpt_te + L_exh + L_noz
    x_cone_end = x_noz_exit + 0.12
    # ---- bypass nozzle exit placed above the LPT ----
    x_byp_exit = x_lpt_te - 0.15
    # ---- polylines ----
    core_hub = [(rows[2][1], fx["r_hub21"]), (x_hpc, ax["r_hub_in"]), (x_imp, imp["r_eye_hub"]),
                (x_imp_te, imp["r2"] * 0.98), (x_diff + 0.012, imp["r2"] * 0.98),
                (x_cb - 0.01, cb["r_inner"]), (x_cb + cb["length"], cb["r_inner"])]
    core_cas = [(x_split, fx["r_split"]), (x_hpc, ax["r_tip"]), (x_imp, imp["r_eye_tip"]),
                (x_imp + 0.6 * L_imp, imp["r2"] * 0.98 + 0.004), (x_imp_te, imp["r2"] + imp["b2"]),
                (x_diff + 0.012, imp["r_diffuser_exit"]), (x_cb - 0.01, cb["r_outer"]), (x_cb + cb["length"], cb["r_outer"])]
    for r_ in rows:
        if r_[0].startswith(("HPT", "LPT")):
            core_hub += [(r_[1], r_[3]), (r_[2], r_[5])]
            core_cas += [(r_[1], r_[4]), (r_[2], r_[6])]
    r_plug = nz["r8_inner"]
    core_hub += [(x_lpt_te + L_exh, lpt["r_hub_out"] * 0.95), (x_noz_exit, r_plug), (x_cone_end, 0.01)]
    core_cas += [(x_lpt_te + L_exh, lpt["r_tip_out"]), (x_noz_exit, nz["r8_outer"])]
    # bypass: inner wall = core cowl, which must clear the core casing (+ casing and
    # clearance allowance); outer wall from the bypass annulus area at that radius.
    CLEAR = 0.025
    def core_outer(xq):
        pts = sorted(core_cas)
        best = 0.0
        for (x0, r0), (x1, r1) in zip(pts[:-1], pts[1:]):
            if x0 - 1e-9 <= xq <= x1 + 1e-9 and x1 > x0:
                best = max(best, r0 + (r1 - r0) * (xq - x0) / (x1 - x0))
        return best
    A13 = fx["A13"]
    x_d = x_diff + 0.012
    xs_in = [x_split, x_ogv + c_ogv + 0.03, x_imp, x_d - 0.03, x_d + 0.03, x_cb + cb["length"], x_byp_exit]
    r_in = [fx["r_split_out"], fx["r_split_out"] + 0.004]
    for xq in xs_in[2:]:
        r_in.append(max(core_outer(xq), max(core_outer(xq - 0.03), core_outer(xq + 0.03))) + CLEAR)
    r_in[2] = max(r_in[2], r_in[1])
    x_exh = x_lpt_te + L_exh
    byp_in = list(zip(xs_in, r_in)) + [(x_lpt_te, core_outer(x_lpt_te - 0.01) + 0.015),
                                       (x_exh, core_outer(x_exh) + 0.012), (x_noz_exit, nz["r8_outer"] + 0.004)]
    r_out = [fan["r_tip"] * 0.995] + [math.sqrt(ri**2 + A13 / math.pi) for ri in r_in[1:-1]]
    r_byp_exit_out = math.sqrt(r_in[-1]**2 + nz["A18"] / math.pi)
    byp_out = ([(-0.45 * 2 * fan["r_tip"], fan["r_tip"] * 1.02), (0.0, fan["r_tip"] + 0.003), (x_fan_te, fan["r_tip"] * 0.995)]
               + list(zip(xs_in[1:-1], r_out[1:])) + [(x_byp_exit, r_byp_exit_out)])
    NAC_T = 0.035
    r_nac_max = max(max(r for _, r in byp_out) + NAC_T, fan["r_tip"] + 0.06)
    sz["nacelle"]["D_max"] = 2 * r_nac_max
    D_nac = sz["nacelle"]["D_max"]
    x_rmax = max(byp_out, key=lambda p: p[1])[0]
    nac_out = [(-0.45 * 2 * fan["r_tip"], fan["r_tip"] * 1.06), (-0.2 * 2 * fan["r_tip"], fan["r_tip"] + 0.05),
               (x_fan_te + 0.05, fan["r_tip"] + 0.06), (x_rmax, D_nac / 2), (x_byp_exit, r_byp_exit_out + 0.008)]
    spinner = [(-0.30 * 2 * fan["r_tip"], 0.0), (-0.12 * 2 * fan["r_tip"], fan["r_hub"] * 0.75), (0.0, fan["r_hub"]),
               (x_fan_te, fx["r_hub21"]), (rows[2][1], fx["r_hub21"])]
    return {
        "rows": [dict(zip(("name", "x_le", "x_te", "r_hub_le", "r_tip_le", "r_hub_te", "r_tip_te", "stream"), r_)) for r_ in rows],
        "lines": {"spinner_and_fan_hub": spinner, "core_hub": core_hub, "core_casing": core_cas,
                  "bypass_inner": byp_in, "bypass_outer": byp_out, "nacelle_outer": nac_out},
        "x_inlet_lip": -0.45 * 2 * fan["r_tip"], "x_bypass_exit": x_byp_exit, "x_core_exit": x_noz_exit,
        "x_cone_end": x_cone_end, "x_combustor": x_cb, "x_impeller": x_imp,
        "length_fan_to_core_exit": x_noz_exit, "length_overall": x_cone_end - (-0.45 * 2 * fan["r_tip"]),
        "D_nacelle": D_nac,
    }
