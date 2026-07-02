"""
TP equilibrium example.

TP means temperature-pressure equilibrium. The user assigns both static
temperature and pressure. ThermoProp solves the equilibrium composition at that
assigned thermodynamic state.

TP is useful when temperature is known directly, for example:

    1. Making property maps as a function of pressure and temperature.
    2. Evaluating equilibrium products at a known wall or film temperature.
    3. Comparing equilibrium composition changes at fixed T and P.

This example uses a dictionary input instead of a Reactants object.

A dictionary is useful when the user already knows the mixture composition and
does not need to build separate fuel and oxidizer streams. For dictionary input,
the keys are CEA species names and the values are either mass or mole fractions.

This example shows:

    1. Solving TP equilibrium from a composition dictionary.
    2. Using basis="mass" and basis="mole".
    3. Reading thermodynamic and transport properties.
    4. Reading composition outputs.
    5. Using pressure_temperature and TP setters.
    6. Using update() with solve=False and solve().
"""

from thermoprop import *


psia_to_pa = 6894.76


# ---------------------------------------------------------------------------
# Composition dictionary
# ---------------------------------------------------------------------------

# This composition represents a guessed hot combustion-gas mixture.
# It is not generated from reactants here. It is simply assigned directly.
#
# Since basis="mass", the numbers below are mass fractions.
gas_composition = {
    "CO2": 0.35,
    "H2O": 0.45,
    "CO": 0.10,
    "H2": 0.10,
}


# ---------------------------------------------------------------------------
# TP equilibrium solve from dictionary input
# ---------------------------------------------------------------------------

products = Equilibrium(
    reactants=gas_composition,
    basis="mass",
    mode="tp",
    pressure=300.0 * psia_to_pa,
    temperature=3000.0,
)

print("TP equilibrium from mass-fraction dictionary")
print("Success:", products.success)
print("Message:", products.message)
print("Iterations:", products.iterations)
print("Mode:", products.mode)
print("Pressure [psia]:", products.pressure / psia_to_pa)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)
print("Enthalpy [J/kg]:", products.enthalpy)
print("Internal energy [J/kg]:", products.internal_energy)
print("Entropy [J/kg-K]:", products.entropy)
print("Gibbs energy [J/kg]:", products.gibbs_energy)
print("Helmholtz energy [J/kg]:", products.helmholtz_energy)
print("Gas constant [J/kg-K]:", products.gas_constant)
print("Molecular weight [kg/kmol]:", products.molecular_weight)
print("Cp equilibrium [J/kg-K]:", products.specific_heat_cp_equilibrium)
print("Cv equilibrium [J/kg-K]:", products.specific_heat_cv_equilibrium)
print("Gamma equilibrium:", products.gamma_equilibrium)
print("Speed of sound equilibrium [m/s]:", products.speed_of_sound_equilibrium)
print("Dynamic viscosity [Pa-s]:", products.dynamic_viscosity)
print("Thermal conductivity [W/m-K]:", products.thermal_conductivity)
print("Prandtl number:", products.prandtl)

print("\nMajor gas mass fractions:")
print(products.gas_mass_fractions)

print("\nMajor gas mole fractions:")
print(products.gas_mole_fractions)

print("\nCondensed species:")
print(products.condensed_species)


# ---------------------------------------------------------------------------
# pressure_temperature setter
# ---------------------------------------------------------------------------

# pressure_temperature is the clearest setter for users.
# The order is:
#
#     (pressure, temperature)
products.pressure_temperature = (250.0 * psia_to_pa, 2800.0)

print("\nAfter pressure_temperature setter")
print("Pressure [psia]:", products.pressure / psia_to_pa)
print("Temperature [K]:", products.temperature)
print("Enthalpy [J/kg]:", products.enthalpy)
print("Entropy [J/kg-K]:", products.entropy)
print("Gamma:", products.gamma)


# ---------------------------------------------------------------------------
# TP setter
# ---------------------------------------------------------------------------

# TP is the short thermodynamics-style setter.
# The order is:
#
#     (temperature, pressure)
#
# This is different from pressure_temperature.
products.TP = (2600.0, 200.0 * psia_to_pa)

print("\nAfter TP setter")
print("Pressure [psia]:", products.pressure / psia_to_pa)
print("Temperature [K]:", products.temperature)
print("Density [kg/m3]:", products.density)
print("Gamma:", products.gamma)


# ---------------------------------------------------------------------------
# update(..., solve=False)
# ---------------------------------------------------------------------------

# In ordinary scripts, update() solves immediately.
# In iterative code, solve=False lets the user change several inputs first and
# then call solve() once.
products.update(
    pressure=180.0 * psia_to_pa,
    temperature=2400.0,
    solve=False,
)

print("\nAfter update(..., solve=False)")
print("Is stale before solve():", products.is_stale)

products.solve()

print("Is stale after solve():", products.is_stale)
print("Pressure [psia]:", products.pressure / psia_to_pa)
print("Temperature [K]:", products.temperature)
print("Entropy [J/kg-K]:", products.entropy)


# ---------------------------------------------------------------------------
# Mole-fraction dictionary input
# ---------------------------------------------------------------------------

# The same idea works with mole fractions by changing basis="mole".
mole_composition = {
    "CO2": 0.20,
    "H2O": 0.60,
    "CO": 0.05,
    "H2": 0.15,
}

mole_products = Equilibrium(
    reactants=mole_composition,
    basis="mole",
    mode="tp",
    pressure=300.0 * psia_to_pa,
    temperature=3000.0,
)

print("\nTP equilibrium from mole-fraction dictionary")
print("Pressure [psia]:", mole_products.pressure / psia_to_pa)
print("Temperature [K]:", mole_products.temperature)
print("Molecular weight [kg/kmol]:", mole_products.molecular_weight)
print("Gas constant [J/kg-K]:", mole_products.gas_constant)
print("Gamma:", mole_products.gamma)
print("Gas mass fractions:", mole_products.gas_mass_fractions)
print("Gas mole fractions:", mole_products.gas_mole_fractions)


# ---------------------------------------------------------------------------
# CombustionGas object from equilibrium products
# ---------------------------------------------------------------------------

# combustion_gas returns only the gas-phase product mixture.
# This is useful when downstream calculations need gas properties but do not
# need another equilibrium solve.
gas_only_products = products.combustion_gas

print("\nGas-only CombustionGas object")
print("Name:", gas_only_products.name)
print("Pressure [Pa]:", gas_only_products.pressure)
print("Temperature [K]:", gas_only_products.temperature)
print("Density [kg/m3]:", gas_only_products.density)
print("Gamma:", gas_only_products.gamma)
print("Mass fractions:", gas_only_products.mass_fractions)