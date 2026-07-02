"""
Propellant example.

A Propellant represents an inlet reactant such as RP-1, LOX, methane, or
hydrogen peroxide. It is the class you normally use before making Reactants and
running chemical equilibrium.

Propellant is different from Fluid:

    Fluid:
        Best for ordinary real-fluid properties, phase behavior, and CoolProp
        flash pairs.

    Propellant:
        Best for rocket propellants and combustion reactants. It keeps the
        thermochemical reference information needed by CEA equilibrium.

This example shows:

    1. Creating propellants from names and aliases.
    2. Reading common properties.
    3. Changing the state with setters.
    4. Changing the state with update().
    5. Viewing supported properties and flash inputs.

All public values use SI units.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Basic propellant states
# ---------------------------------------------------------------------------

# A propellant can be created with temperature only.
# This is common for reactants because the equilibrium solver mostly needs the
# reactant enthalpy at the feed temperature.
fuel = Propellant("rp-1", temperature=298.15)

# A pressure can also be supplied.
# This is useful when the liquid-property backend needs pressure-dependent
# engineering properties.
oxidizer = Propellant("lox", temperature=90.17, pressure=400.0 * 6894.76)

print("Fuel name:", fuel.name)
print("Fuel backend:", fuel.backend)
print("Fuel temperature [K]:", fuel.temperature)
print("Fuel enthalpy [J/kg]:", fuel.enthalpy)
print("Fuel density [kg/m3]:", fuel.density)

print("\nOxidizer name:", oxidizer.name)
print("Oxidizer CEA name:", oxidizer.cea_name)
print("Oxidizer phase:", oxidizer.phase)
print("Oxidizer pressure [Pa]:", oxidizer.pressure)
print("Oxidizer temperature [K]:", oxidizer.temperature)
print("Oxidizer enthalpy [J/kg]:", oxidizer.enthalpy)
print("Oxidizer density [kg/m3]:", oxidizer.density)


# ---------------------------------------------------------------------------
# Changing a propellant state with setters
# ---------------------------------------------------------------------------

# Setters update the object in place.
# This is useful in scripts where the same object is reused at a new condition.
fuel.temperature = 310.0
print("\nFuel temperature after setter [K]:", fuel.temperature)
print("Fuel enthalpy after setter [J/kg]:", fuel.enthalpy)

# The pressure_temperature pair setter changes both values together.
# Pair setters are useful when two values must change at the same time.
oxidizer.pressure_temperature = (450.0 * 6894.76, 91.0)
print("\nOxidizer pressure after pair setter [psia]:", oxidizer.pressure / 6894.76)
print("Oxidizer temperature after pair setter [K]:", oxidizer.temperature)


# ---------------------------------------------------------------------------
# Changing a propellant state with update()
# ---------------------------------------------------------------------------

# update() is the easiest way to change several inputs in one line.
oxidizer.update(temperature=92.0, pressure=500.0 * 6894.76)

print("\nOxidizer pressure after update [psia]:", oxidizer.pressure / 6894.76)
print("Oxidizer temperature after update [K]:", oxidizer.temperature)
print("Oxidizer density after update [kg/m3]:", oxidizer.density)


# ---------------------------------------------------------------------------
# Optional quality-corrected inlet enthalpy
# ---------------------------------------------------------------------------

# Propellant can accept quality for inlet enthalpy correction.
# This is intended for feed states with some vapor content.
# The ordinary properties still represent the resolved propellant state, while
# the enthalpy receives a two-phase correction.
lox_sat_liquid = Propellant("lox", pressure=20.0 * 6894.76, quality=0.0)
lox_sat_mixture = Propellant("lox", pressure=20.0 * 6894.76, quality=0.05)

print("\nSaturated LOX liquid temperature [K]:", lox_sat_liquid.temperature)
print("Saturated LOX liquid enthalpy [J/kg]:", lox_sat_liquid.enthalpy)
print("Saturated LOX 5% quality enthalpy [J/kg]:", lox_sat_mixture.enthalpy)
print("Quality correction active:", lox_sat_mixture.has_quality_enthalpy_correction)
print("Quality enthalpy correction [J/kg]:", lox_sat_mixture.enthalpy_correction)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

# These methods are useful when a user is not sure which names or properties are
# available. They return lists, and the show_* versions also print them.
print("\nNumber of propellant names:", len(Propellant.get_available_propellants()))
print("Number of CEA species names:", len(Propellant.get_available_cea_species()))
print("Supported flash inputs:", Propellant.supported_flash_inputs())
print("Supports density property:", Propellant.supports_property("density"))
print("Supports prandtl property:", Propellant.supports_property("prandtl"))
