"""CEA-style rocket performance with Reactants, Equilibrium, and Rocket.

The example keeps the public workflow intentionally small:

1. Define propellant states.
2. Combine them in Reactants.
3. Pass Reactants to Rocket.
4. Read chamber, throat, and exit station data directly.
5. Print the complete CEA-style report.

All values use SI units.
"""

from thermoprop import Propellant, Reactants, Rocket


PSIA = 6894.757293168


fuel = Propellant(
    "RP-1",
    temperature=298.15,
    pressure=400.0 * PSIA,
)

oxidizer = Propellant(
    "LOX",
    temperature=90.17,
    pressure=400.0 * PSIA,
)

reactants = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=2.6,
)


rocket = Rocket(
    reactants,
    chamber_pressure=300.0 * PSIA,
    exit_pressures=10.0 * PSIA,
    subsonic_area_ratios=[1.5, 1.2],
    supersonic_area_ratios=[5.0, 10.0, 40.0],
    frozen_at=None,
)


# Major stations are available directly.
print("Chamber temperature [K]:", rocket.chamber.temperature)
print("Throat pressure [Pa]:", rocket.throat.pressure)
print("Throat Mach number:", rocket.throat.mach)
print("Characteristic velocity [m/s]:", rocket.cstar)


# Requested stations can be selected by the value used to create them.
exit_40 = rocket.at_area_ratio(40.0, branch="supersonic")
exit_10_psia = rocket.at_pressure(10.0 * PSIA)

print("A/At=40 exit pressure [Pa]:", exit_40.pressure)
print("A/At=40 exit velocity [m/s]:", exit_40.velocity)
print("A/At=40 vacuum Isp [s]:", exit_40.isp_vac)
print("10 psia station Mach number:", exit_10_psia.mach)


# Every station delegates thermodynamic properties to its underlying
# Equilibrium or CombustionGas object.
print("A/At=40 mass fractions:")
for species, fraction in exit_40.mass_fractions.items():
    if fraction >= 1.0e-4:
        print(f"  {species:<20} {fraction:.6g}")


# Adding contraction_ratio automatically selects the finite-area combustor.
fac_rocket = Rocket(
    reactants,
    chamber_pressure=300.0 * PSIA,
    contraction_ratio=4.0,
    supersonic_area_ratios=40.0,
)

print("FAC injector pressure [Pa]:", fac_rocket.injector.pressure)
print("FAC combustor-end pressure [Pa]:", fac_rocket.chamber.pressure)
print("FAC combustor-end Mach number:", fac_rocket.chamber.mach)


# A numeric frozen_at value freezes the equilibrium composition at that
# supersonic nozzle area ratio.  The freeze station is solved automatically,
# even if the ratio is not included in supersonic_area_ratios.
area_frozen_rocket = Rocket(
    reactants,
    chamber_pressure=300.0 * PSIA,
    supersonic_area_ratios=[2.0, 10.0, 40.0],
    frozen_at=5.0,
)

print("Freeze area ratio:", area_frozen_rocket.freeze_station.area_ratio)
print("A/At=2 chemistry:", area_frozen_rocket.at_area_ratio(2.0).chemistry)
print("A/At=10 chemistry:", area_frozen_rocket.at_area_ratio(10.0).chemistry)


# The full report includes reactants, model settings, all station properties,
# performance values, and composition tables.
print(rocket.report(fractions="mole", trace=1.0e-5))
