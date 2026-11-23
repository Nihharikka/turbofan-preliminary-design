# Preliminary Design of a Small Two-Spool Turbofan

Design-point cycle analysis, preliminary sizing and validation of a small
two-spool, separate-flow turbofan for a light-to-midsize business jet
(about 20 kN takeoff thrust class). Built during the YuvaIntern Propulsion
Systems internship (October to November 2026).

## Current design (Week 3, revision A)

| Item | Value |
|---|---|
| Design point | 12,000 m, Mach 0.80, ISA (top of climb), 4.0 kN |
| Cycle | BPR 5.0, fan PR 1.80 bypass / 1.50 core, OPR 23, HPT rotor inlet 1,450 K, 8% turbine cooling |
| Cruise TSFC | 18.5 g/(kN·s) (0.653 lb/(lbf·h)) |
| Air mass flow | 20.2 kg/s at cruise, 59 kg/s at takeoff |
| Fan | 672 mm diameter, 22 blades, 12,224 rpm |
| HP spool | 4 axial + 1 centrifugal compressor stages, 2-stage HPT, 28,676 rpm |
| LP turbine | 3 stages |
| Length / nacelle | 1.49 m fan to core exit / 0.83 m |
| Takeoff, 20 kN | rotor inlet 1,566 K (ISA), 1,639 K (ISA+15) |

Uninstalled values (no bleed, power offtake or installation losses).

## How to run

```bash
pip install -r requirements.txt
python -m pytest -v tests/            # 13 verification tests
python scripts/week2_run_all.py       # Week 2 results  -> results/
python scripts/week3_run_all.py       # Week 3 results  -> results/week3/
python validation/pycycle_check.py    # NASA pyCycle cross-check (needs pyCycle, see below)
python validation/published_engines.py  # comparison with published engines
```

Optional package for the pyCycle check: install pyCycle from GitHub
(`pip install git+https://github.com/OpenMDAO/pyCycle.git`; the 4.4.0
release on PyPI fails with NumPy 2).

`validation/Turbofan_Hand_Calculation_RevA.xlsx` repeats the design point by
hand with the textbook constant-cp method (live formulas; open in Excel).

## Repository layout

```
turbofan/
  atmosphere.py    ISA model (ISO 2533)
  thermo.py        NASA-polynomial gas properties, Jet A-1 combustion products, mixing
  components.py    inlet, duct, compressor, combustor, turbine, convergent nozzle
  cycle.py         design-point cycle: split fan, turbine cooling (SAE AS755 stations)
  offdesign.py     simplified fixed-geometry matching model (takeoff)
  requirement.py   top-of-climb thrust requirement
  sizing.py        mean-line sizing of fan, HPC (axial + centrifugal), combustor, HPT, LPT
  layout.py        meridional flowpath layout
  design_case.py   WEEK2 and REV_A design points
scripts/           reproduce each week's results and figures
validation/        pyCycle cross-check, published-engine comparison, hand-calculation workbook
tests/             verification tests
results/           generated data and figures
```

## Verification and validation

* Ideal-mode results match the closed-form ideal turbofan equations to 1e-6.
* Energy and mass balances close on every spool, the combustor and the cooling mixers.
* NASA pyCycle agrees within 0.8% on every design-point quantity once fuel
  heating value is matched.
* An independent textbook hand calculation (Excel) agrees within 1.6% on TSFC.
* Stage layout, shaft speeds and TSFC agree with published business-jet engines;
  the fan is about 13% below the published average for its thrust, but within
  the range of real engines.

## Scope

Thermodynamic cycle analysis, mean-line sizing, layout and validation, all in
Python. Detailed CAD, structural (FEA) and CFD analyses are outside the scope
of this project and are listed as future work in the final dossier.

## References

Saravanamuttoo et al., *Gas Turbine Theory*, 7th ed.; Walsh and Fletcher,
*Gas Turbine Performance*, 2nd ed.; Mattingly et al., *Aircraft Engine Design*,
2nd ed.; Dixon and Hall, *Fluid Mechanics and Thermodynamics of
Turbomachinery*, 7th ed.; Hendricks and Gray (2019), pyCycle; McBride et al.,
NASA TM-4513.
