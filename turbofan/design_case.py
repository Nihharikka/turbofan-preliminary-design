"""Design points of the project.

WEEK2  the cycle selected in Week 2 (uniform fan pressure ratio, no turbine cooling)
REV_A  Week 3 revision: lower core-side fan pressure ratio (fan hub loading) and
       turbine cooling air, with the HPT rotor inlet temperature held at 1450 K
"""
from .cycle import DesignInputs

WEEK2 = DesignInputs(bpr=5.0, fpr=1.80, opr=23.0, Tt4=1450.0, thrust_required=4000.0)

REV_A = DesignInputs(bpr=5.0, fpr=1.80, fpr_core=1.50, opr=23.0, Tt4=1450.0, rotor_inlet_T=1450.0,
                     cool_ngv=0.05, cool_rotor=0.03, thrust_required=4000.0)

FINAL = REV_A

# Takeoff rating assumptions
TAKEOFF_T4_LIMIT = 1650.0   # K, maximum takeoff turbine (rotor inlet) temperature
FLAT_RATING_DT = 15.0       # K, full thrust held up to ISA+15 at sea level
TAKEOFF_THRUST = 20000.0    # N per engine
