"""
SP equilibrium example.

SP means entropy-pressure equilibrium. It is useful for ideal nozzle expansion
because an adiabatic, reversible nozzle keeps entropy constant while pressure
changes.

A typical rocket-nozzle workflow is:

    1. Solve the chamber with HP equilibrium.
    2. Save the chamber entropy and total enthalpy.
    3. Solve downstream stations with SP equilibrium at lower pressures.
    4. Convert enthalpy drop into velocity.

This example shows the chamber, a throat-like station, and an exit-like station.
"""

from thermoprop import *


psia_to_pa = 6894.76


# ---------------------------------------------------------------------------
# Chamber HP state
# ---------------------------------------------------------------------------

fuel = Propellant("ch4", temperature=298.15)
oxidizer = Propellant("lox", temperature=90.17)

reactants = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=3.4,
)

chamber = Equilibrium(
    reactants=reactants,
    mode="hp",
    pressure=300.0 * psia_to_pa,
)

chamber_entropy = chamber.entropy
chamber_total_enthalpy = chamber.enthalpy

print("Chamber state")
print("Pressure [psia]:", chamber.pressure / psia_to_pa)
print("Temperature [K]:", chamber.temperature)
print("Entropy [J/kg-K]:", chamber.entropy)
print("Total enthalpy [J/kg]:", chamber_total_enthalpy)


# ---------------------------------------------------------------------------
# SP station at a lower pressure
# ---------------------------------------------------------------------------

throat = Equilibrium(
    reactants=reactants,
    mode="sp",
    pressure=175.0 * psia_to_pa,
    entropy=chamber_entropy,
    guess_temperature=chamber.temperature,
)

throat_velocity = (2.0 * (chamber_total_enthalpy - throat.enthalpy))**0.5
throat_mach = throat_velocity / throat.speed_of_sound

print("\nThroat-like SP station")
print("Pressure [psia]:", throat.pressure / psia_to_pa)
print("Temperature [K]:", throat.temperature)
print("Entropy [J/kg-K]:", throat.entropy)
print("Entropy difference from chamber [J/kg-K]:", throat.entropy - chamber_entropy)
print("Static enthalpy [J/kg]:", throat.enthalpy)
print("Velocity [m/s]:", throat_velocity)
print("Mach number:", throat_mach)


# ---------------------------------------------------------------------------
# Exit station at a still lower pressure
# ---------------------------------------------------------------------------

exit_station = Equilibrium(
    reactants=reactants,
    mode="sp",
    pressure=30.0 * psia_to_pa,
    entropy=chamber_entropy,
    guess_temperature=throat.temperature,
)

exit_velocity = (2.0 * (chamber_total_enthalpy - exit_station.enthalpy))**0.5
exit_mach = exit_velocity / exit_station.speed_of_sound

print("\nExit-like SP station")
print("Pressure [psia]:", exit_station.pressure / psia_to_pa)
print("Temperature [K]:", exit_station.temperature)
print("Entropy [J/kg-K]:", exit_station.entropy)
print("Entropy difference from chamber [J/kg-K]:", exit_station.entropy - chamber_entropy)
print("Static enthalpy [J/kg]:", exit_station.enthalpy)
print("Velocity [m/s]:", exit_velocity)
print("Mach number:", exit_mach)


# ---------------------------------------------------------------------------
# Updating an SP station
# ---------------------------------------------------------------------------

# In SP mode, pressure and entropy are the assigned state.
# Changing pressure moves to a new point on the same isentrope.
exit_station.update(
    pressure=20.0 * psia_to_pa,
    entropy=chamber_entropy,
    guess_temperature=exit_station.temperature,
)

print("\nExit station after update()")
print("Pressure [psia]:", exit_station.pressure / psia_to_pa)
print("Temperature [K]:", exit_station.temperature)
print("Entropy [J/kg-K]:", exit_station.entropy)
