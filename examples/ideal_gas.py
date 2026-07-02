"""
IdealGas example.

IdealGas uses ideal-gas thermodynamics for pure gases and gas mixtures. It is a
good choice when phase behavior is not important and the ideal-gas assumption is
reasonable.

IdealGas is useful for:

    1. Air, nitrogen, helium, hydrogen, and other gas-system models.
    2. Fast thermodynamic property calculations.
    3. Mixtures with mass or mole fractions.
    4. Flash pairs such as PT, PH, PS, rho-T, and rho-u.

This example shows constructor inputs, common properties, setters, update(), and
mixture fractions.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Pure ideal gas from pressure and temperature
# ---------------------------------------------------------------------------

nitrogen = IdealGas(
    "N2",
    pressure=101325.0,
    temperature=300.0,
)

print("Nitrogen ideal gas")
print("Name:", nitrogen.name)
print("Backend:", nitrogen.backend)
print("Pressure [Pa]:", nitrogen.pressure)
print("Temperature [K]:", nitrogen.temperature)
print("Density [kg/m3]:", nitrogen.density)
print("Enthalpy [J/kg]:", nitrogen.enthalpy)
print("Internal energy [J/kg]:", nitrogen.internal_energy)
print("Entropy [J/kg-K]:", nitrogen.entropy)
print("Cp [J/kg-K]:", nitrogen.specific_heat_cp)
print("Cv [J/kg-K]:", nitrogen.specific_heat_cv)
print("Gamma:", nitrogen.gamma)
print("Speed of sound [m/s]:", nitrogen.speed_of_sound)


# ---------------------------------------------------------------------------
# Pair setters
# ---------------------------------------------------------------------------

# pressure_temperature is the most common flash pair.
nitrogen.pressure_temperature = (2.0e5, 350.0)

print("\nAfter pressure_temperature setter")
print("Pressure [Pa]:", nitrogen.pressure)
print("Temperature [K]:", nitrogen.temperature)
print("Density [kg/m3]:", nitrogen.density)

# HP is a readable alias for enthalpy-pressure style updates.
nitrogen.HP = (nitrogen.enthalpy, 1.5e5)

print("\nAfter HP setter")
print("Pressure [Pa]:", nitrogen.pressure)
print("Temperature [K]:", nitrogen.temperature)
print("Enthalpy [J/kg]:", nitrogen.enthalpy)


# ---------------------------------------------------------------------------
# Mixture from mole fractions
# ---------------------------------------------------------------------------

air_like_gas = IdealGas(
    {"N2": 0.79, "O2": 0.21},
    basis="mole",
    pressure=101325.0,
    temperature=300.0,
)

print("\nAir-like mixture")
print("Species:", air_like_gas.species)
print("Basis:", air_like_gas.basis)
print("Mole fractions:", air_like_gas.mole_fractions)
print("Mass fractions:", air_like_gas.mass_fractions)
print("Molecular weight [kg/kmol]:", air_like_gas.molar_mass)
print("Gas constant [J/kg-K]:", air_like_gas.gas_constant)
print("Density [kg/m3]:", air_like_gas.density)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

# update() can change composition and thermodynamic state together.
air_like_gas.update(
    fluid={"N2": 0.70, "O2": 0.20, "CO2": 0.10},
    basis="mole",
    pressure=2.0e5,
    temperature=500.0,
)

print("\nAfter update()")
print("Species:", air_like_gas.species)
print("Mole fractions:", air_like_gas.mole_fractions)
print("Temperature [K]:", air_like_gas.temperature)
print("Density [kg/m3]:", air_like_gas.density)
print("Gamma:", air_like_gas.gamma)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

print("\nSupported flash inputs:", IdealGas.supported_flash_inputs())
print("Supports speed_of_sound:", IdealGas.supports_property("speed_of_sound"))
print("Supports saturation_temperature:", IdealGas.supports_property("saturation_temperature"))
