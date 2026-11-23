"""International Standard Atmosphere (ISO 2533:1975), troposphere and lower stratosphere.

Valid from sea level to 20 km geopotential altitude, which covers every
condition used in this project.
"""
import math

G0 = 9.80665          # m/s^2, standard gravity
R_AIR = 287.05287     # J/(kg K), ISA gas constant for dry air
T_SL = 288.15         # K
P_SL = 101325.0       # Pa
LAPSE = -0.0065       # K/m, troposphere
H_TROPO = 11000.0     # m, tropopause


def isa(h):
    """Return static temperature [K] and pressure [Pa] at geopotential altitude h [m]."""
    if h < 0 or h > 20000:
        raise ValueError("ISA model here is valid only for 0 to 20,000 m")
    if h <= H_TROPO:
        T = T_SL + LAPSE * h
        p = P_SL * (T / T_SL) ** (-G0 / (LAPSE * R_AIR))
    else:
        T11 = T_SL + LAPSE * H_TROPO
        p11 = P_SL * (T11 / T_SL) ** (-G0 / (LAPSE * R_AIR))
        T = T11
        p = p11 * math.exp(-G0 * (h - H_TROPO) / (R_AIR * T11))
    return T, p
