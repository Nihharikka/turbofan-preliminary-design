"""Gas properties for air and kerosene combustion products.

Species properties use NASA 7-coefficient polynomials (McBride, Gordon and
Reno, NASA TM-4513 format) as distributed in the GRI-Mech 3.0 thermodynamic
data set. Mixtures are ideal gases. Two gas models are provided:

* ``Gas``      temperature-dependent properties (used for all design work)
* ``IdealGas`` constant cp and gamma (used only to verify the cycle code
               against textbook closed-form equations)

Enthalpies are sensible enthalpies relative to 298.15 K, so the combustor
energy balance uses the fuel lower heating value directly.
"""
import math
from scipy.optimize import brentq

RU = 8.314462618  # J/(mol K)
T_REF = 298.15

# [Tmid, high-range a1..a7, low-range a1..a7]
NASA7 = {
    "N2":  [1000.0, 2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15, -922.7977, 5.980528,
            3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12, -1020.8999, 3.950372],
    "O2":  [1000.0, 3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129,
            3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573],
    "AR":  [1000.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366,
            2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366],
    "CO2": [1000.0, 3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14, -48759.166, 2.27163806,
            2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222],
    "H2O": [1000.0, 3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14, -30004.2971, 4.9667701,
            4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -30293.7267, -0.849032208],
}
MOLAR_MASS = {"N2": 28.0134e-3, "O2": 31.9988e-3, "AR": 39.948e-3, "CO2": 44.0095e-3, "H2O": 18.01528e-3}

# Dry air, mole fractions (standard composition, CO2 folded into argon-free remainder)
AIR_X = {"N2": 0.78084, "O2": 0.20946, "AR": 0.00934, "CO2": 0.00036}

# Jet A-1 surrogate C12H23 (a common single-component approximation)
FUEL_C, FUEL_H = 12.0, 23.0
FUEL_M = FUEL_C * 12.011e-3 + FUEL_H * 1.008e-3  # kg/mol
LHV_JETA1 = 43.0e6  # J/kg


def _coeffs(sp, T):
    c = NASA7[sp]
    return c[1:8] if T >= c[0] else c[8:15]


def _cp_R(sp, T):
    a = _coeffs(sp, T)
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4


def _h_RT(sp, T):
    a = _coeffs(sp, T)
    return a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T


def _s_R(sp, T):
    a = _coeffs(sp, T)
    return a[0]*math.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]


class Gas:
    """Ideal-gas mixture with temperature-dependent properties (per unit mass)."""

    def __init__(self, x):
        tot = sum(x.values())
        self.x = {k: v / tot for k, v in x.items() if v > 0}
        self.M = sum(self.x[k] * MOLAR_MASS[k] for k in self.x)
        self.R = RU / self.M
        self._h_ref = self._h_abs(T_REF)

    def _h_abs(self, T):
        return sum(xi * _h_RT(k, T) * RU * T for k, xi in self.x.items()) / self.M

    def cp(self, T):
        return sum(xi * _cp_R(k, T) for k, xi in self.x.items()) * RU / self.M

    def gamma(self, T):
        cp = self.cp(T)
        return cp / (cp - self.R)

    def h(self, T):
        """Sensible enthalpy relative to 298.15 K [J/kg]."""
        return self._h_abs(T) - self._h_ref

    def phi(self, T):
        """Entropy function s°(T) [J/(kg K)]; isentropic: phi2 - phi1 = R ln(p2/p1)."""
        return sum(xi * _s_R(k, T) for k, xi in self.x.items()) * RU / self.M

    def T_from_h(self, h):
        return brentq(lambda T: self.h(T) - h, 150.0, 3400.0, xtol=1e-9)

    def T_from_phi(self, phi):
        return brentq(lambda T: self.phi(T) - phi, 150.0, 3400.0, xtol=1e-9)

    def sound_speed(self, T):
        return math.sqrt(self.gamma(T) * self.R * T)


class IdealGas:
    """Calorically perfect gas (constant cp). Used for verification only."""

    def __init__(self, cp=1004.5, gamma=1.4):
        self.cp_c, self.g = cp, gamma
        self.R = cp * (gamma - 1) / gamma
        self.M = RU / self.R

    def cp(self, T):
        return self.cp_c

    def gamma(self, T):
        return self.g

    def h(self, T):
        return self.cp_c * (T - T_REF)

    def phi(self, T):
        return self.cp_c * math.log(T)

    def T_from_h(self, h):
        return T_REF + h / self.cp_c

    def T_from_phi(self, phi):
        return math.exp(phi / self.cp_c)

    def sound_speed(self, T):
        return math.sqrt(self.g * self.R * T)


def air():
    return Gas(AIR_X)


def products(far):
    """Composition of complete lean combustion products of C12H23 with dry air.

    far: fuel-air mass ratio. Returns a Gas.
    """
    M_air = Gas(AIR_X).M
    n_air = 1.0 / M_air              # mol air per kg air
    n_fuel = far / FUEL_M            # mol fuel per kg air
    n = {k: v * n_air for k, v in AIR_X.items()}
    o2_needed = n_fuel * (FUEL_C + FUEL_H / 4)
    if o2_needed > n["O2"]:
        raise ValueError("Fuel-air ratio is richer than stoichiometric")
    n["O2"] -= o2_needed
    n["CO2"] += n_fuel * FUEL_C
    n["H2O"] = n_fuel * FUEL_H / 2
    return Gas(n)


def stoich_far():
    M_air = Gas(AIR_X).M
    n_o2 = AIR_X["O2"] / M_air
    return n_o2 / (FUEL_C + FUEL_H / 4) * FUEL_M


def mix(g1, m1, g2, m2):
    """Gas formed by mixing mass m1 of gas g1 with mass m2 of gas g2 (no reaction)."""
    n = {}
    for g, m in ((g1, m1), (g2, m2)):
        moles = m / g.M
        for k, xi in g.x.items():
            n[k] = n.get(k, 0.0) + moles * xi
    return Gas(n)
