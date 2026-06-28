"""
SP equilibrium example for an ideal equilibrium nozzle.

This example demonstrates the intended workflow:

1. Use HP equilibrium to compute the chamber stagnation products.
2. Use SP equilibrium at lower static pressures with chamber entropy held fixed.
3. Interpret a fixed low-pressure SP station as a downstream/exit-like station.
4. Bracket the throat pressure by checking where Mach crosses 1.

For an ideal, adiabatic, reversible nozzle expansion, chamber entropy is
carried downstream while static pressure changes.  The static enthalpy becomes
lower than chamber stagnation enthalpy; the difference becomes kinetic energy.
"""

from thermoprop import *


def nozzle_station(reactants, chamber, pressure_psia):
    """Return SP station, velocity, and Mach number at a specified static pressure."""
    station = Equilibrium(
        reactants,
        mode="sp",
        pressure=pressure_psia * 6894.76,
        entropy=chamber.entropy,
        guess_temperature=chamber.temperature,
    )

    velocity = (2.0 * (chamber.enthalpy - station.enthalpy)) ** 0.5
    mach_number = velocity / station.speed_of_sound

    return station, velocity, mach_number


fuel = Propellant("CH4", temperature=298.15)
oxidizer = Propellant("LOX", temperature=90.17)
reactants = Reactants(fuels=fuel, oxidizers=oxidizer, mixture_ratio=3.4)

chamber = Equilibrium(
    reactants,
    mode="hp",
    pressure=300.0 * 6894.76,
)

# This is intentionally below the expected critical pressure, so it is a
# downstream/exit-like supersonic station, not the throat.
downstream, downstream_velocity, downstream_mach = nozzle_station(
    reactants,
    chamber,
    pressure_psia=100.0,
)

# The throat is found by iterating pressure until Mach = 1.  These two points
# bracket the throat for this chamber condition, without making the example slow.
high_check, high_velocity, high_mach = nozzle_station(reactants, chamber, pressure_psia=180.0)
low_check, low_velocity, low_mach = nozzle_station(reactants, chamber, pressure_psia=170.0)

print("Chamber HP state")
print(f"  P = {chamber.pressure / 6894.76:.3f} psia")
print(f"  T = {chamber.temperature:.3f} K")
print(f"  h0 = {chamber.enthalpy:.3f} J/kg")
print(f"  s0 = {chamber.entropy:.6f} J/kg-K")

print("\nFixed downstream SP station")
print(f"  P = {downstream.pressure / 6894.76:.3f} psia")
print(f"  T = {downstream.temperature:.3f} K")
print(f"  h = {downstream.enthalpy:.3f} J/kg")
print(f"  s = {downstream.entropy:.6f} J/kg-K")
print(f"  entropy error = {downstream.entropy - chamber.entropy:.6e} J/kg-K")
print(f"  velocity = {downstream_velocity:.3f} m/s")
print(f"  Mach = {downstream_mach:.3f}")

print("\nThroat pressure bracket")
print(f"  At 180 psia: Mach = {high_mach:.3f}")
print(f"  At 170 psia: Mach = {low_mach:.3f}")
print("  Therefore the sonic throat pressure is between 170 and 180 psia for this case.")
