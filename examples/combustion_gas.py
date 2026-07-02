"""
CombustionGas example.

CombustionGas evaluates gas-phase CEA species and gas mixtures. It is useful
when the composition is already known and the user wants gas properties without
running a new equilibrium solve.

CombustionGas is different from Equilibrium:

    Equilibrium:
        Solves the composition from reactants and thermodynamic constraints.

    CombustionGas:
        Uses a specified gas composition and evaluates properties at a state.

This example shows:

    1. A pure gas.
    2. A gas mixture.
    3. Common flash pairs.
    4. Mass and mole fractions.
    5. Updating the state and composition.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Pure combustion gas species
# ---------------------------------------------------------------------------

co2 = CombustionGas(
    "CO2",
    pressure=101325.0,
    temperature=1200.0,
)

print("Pure CO2 gas")
print("Name:", co2.name)
print("Backend:", co2.backend)
print("Pressure [Pa]:", co2.pressure)
print("Temperature [K]:", co2.temperature)
print("Density [kg/m3]:", co2.density)
print("Enthalpy [J/kg]:", co2.enthalpy)
print("Entropy [J/kg-K]:", co2.entropy)
print("Cp [J/kg-K]:", co2.specific_heat_cp)
print("Cv [J/kg-K]:", co2.specific_heat_cv)
print("Gamma:", co2.gamma)
print("Molecular weight [kg/mol]:", co2.molar_mass)
print("Molecular weight [kg/kmol]:", co2.molar_mass * 1000.0)
print("Gas constant [J/kg-K]:", co2.gas_constant)
print("Speed of sound [m/s]:", co2.speed_of_sound)
print("Dynamic viscosity [Pa-s]:", co2.dynamic_viscosity)
print("Thermal conductivity [W/m-K]:", co2.thermal_conductivity)
print("Prandtl number:", co2.prandtl)


# ---------------------------------------------------------------------------
# Mixture from mass fractions
# ---------------------------------------------------------------------------

# A CombustionGas can also be made from a dictionary of species fractions.
# When basis="mass", the numbers are mass fractions.
products = CombustionGas(
    {"CO2": 0.35, "H2O": 0.45, "CO": 0.10, "H2": 0.10},
    basis="mass",
    pressure=2.0e6,
    temperature=3000.0,
)

print("\nGas mixture")
print("Species:", products.species)
print("Basis:", products.basis)
print("Mass fractions:", products.mass_fractions)
print("Mole fractions:", products.mole_fractions)
print("Molecular weight [kg/mol]:", products.molar_mass)
print("Molecular weight [kg/kmol]:", products.molar_mass * 1000.0)
print("Gas constant [J/kg-K]:", products.gas_constant)
print("Density [kg/m3]:", products.density)
print("Enthalpy [J/kg]:", products.enthalpy)
print("Entropy [J/kg-K]:", products.entropy)
print("Cp [J/kg-K]:", products.specific_heat_cp)
print("Cv [J/kg-K]:", products.specific_heat_cv)
print("Gamma:", products.gamma)


# ---------------------------------------------------------------------------
# Flash pair setters
# ---------------------------------------------------------------------------

# pressure_temperature is the clearest setter for users.
# The order is:
#
#     (pressure, temperature)
products.pressure_temperature = (1.5e6, 2800.0)

print("\nAfter pressure_temperature setter")
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)
print("Enthalpy [J/kg]:", products.enthalpy)

# TP is the short form used by many thermodynamics packages.
# In ThermoProp, TP means:
#
#     (temperature, pressure)
#
# This is the opposite order from pressure_temperature.
products.TP = (2600.0, 1.2e6)

print("\nAfter TP setter")
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)
print("Enthalpy [J/kg]:", products.enthalpy)

# pressure_enthalpy is useful for energy-balance problems.
# The order is:
#
#     (pressure, enthalpy)
current_enthalpy = products.enthalpy
products.pressure_enthalpy = (1.0e6, current_enthalpy)

print("\nAfter pressure_enthalpy setter")
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Enthalpy [J/kg]:", products.enthalpy)

# HP is the short form.
# The order is:
#
#     (enthalpy, pressure)
products.HP = (products.enthalpy, 8.0e5)

print("\nAfter HP setter")
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Enthalpy [J/kg]:", products.enthalpy)

# density_temperature is useful when density is known instead of pressure.
# The order is:
#
#     (density, temperature)
products.density_temperature = (0.75, 2200.0)

print("\nAfter density_temperature setter")
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)


# ---------------------------------------------------------------------------
# Mixture from mole fractions
# ---------------------------------------------------------------------------

# When basis="mole", the numbers are mole fractions.
steam_rich_products = CombustionGas(
    {"H2O": 0.70, "CO2": 0.20, "CO": 0.05, "H2": 0.05},
    basis="mole",
    pressure=101325.0,
    temperature=1800.0,
)

print("\nGas mixture from mole fractions")
print("Species:", steam_rich_products.species)
print("Basis:", steam_rich_products.basis)
print("Mole fractions:", steam_rich_products.mole_fractions)
print("Mass fractions:", steam_rich_products.mass_fractions)
print("Molecular weight [kg/kmol]:", steam_rich_products.molar_mass * 1000.0)
print("Gamma:", steam_rich_products.gamma)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

# update() can change composition and state in one call.
products.update(
    fluid={"CO2": 0.50, "H2O": 0.50},
    basis="mole",
    pressure=101325.0,
    temperature=1800.0,
)

print("\nAfter update()")
print("Basis:", products.basis)
print("Species:", products.species)
print("Mole fractions:", products.mole_fractions)
print("Mass fractions:", products.mass_fractions)
print("Pressure [Pa]:", products.pressure)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)
print("Gamma:", products.gamma)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

print("\nSupports gamma property:", CombustionGas.supports_property("gamma"))
print("Supports quality property:", CombustionGas.supports_property("quality"))
print("Supported flash inputs:", CombustionGas.supported_flash_inputs())