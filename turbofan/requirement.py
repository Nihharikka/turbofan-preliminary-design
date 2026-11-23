"""Top-of-climb thrust requirement per engine, from a simple aircraft force balance.

All aircraft values are class-typical assumptions for a twin-engine light to
midsize business jet; they are stated in the Week 2 report.
"""
import math
from .atmosphere import isa, G0

AIRCRAFT = {
    "n_engines": 2,
    "takeoff_thrust_per_engine": 20000.0,  # N, design brief
    "takeoff_thrust_to_weight": 0.37,      # typical business jets 0.33 to 0.42
    "weight_fraction_toc": 0.95,           # weight at top of climb / MTOW (climb fuel burned)
    "lift_to_drag": 15.0,                  # cruise L/D, typical business jet 14 to 17
    "residual_climb_rate": 300 * 0.3048 / 60,  # m/s, 300 ft/min standard top-of-climb margin
    "installation_allowance": 0.05,        # bleed, power offtake, installation losses
    "altitude": 12000.0,
    "mach": 0.80,
}


def toc_requirement(a=AIRCRAFT):
    T0, _ = isa(a["altitude"])
    V = a["mach"] * math.sqrt(1.4 * 287.05287 * T0)
    mtow = a["n_engines"] * a["takeoff_thrust_per_engine"] / (a["takeoff_thrust_to_weight"] * G0)
    W = a["weight_fraction_toc"] * mtow * G0
    drag = W / a["lift_to_drag"]
    climb = W * a["residual_climb_rate"] / V
    total = (drag + climb) * (1 + a["installation_allowance"])
    return {"V": V, "mtow": mtow, "W": W, "drag_total": drag, "climb_total": climb,
            "per_engine_uninstalled": total / a["n_engines"]}
