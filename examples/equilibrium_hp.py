"""
HP equilibrium example.

HP means enthalpy-pressure equilibrium. This is the most common combustion mode
for rocket chambers.

The reactants define the feed enthalpy. The user assigns pressure. ThermoProp
then solves the equilibrium product composition and temperature that conserve
enthalpy at that pressure.

This example uses CombustionGas objects inside Reactants instead of ordinary
Propellant objects. This is useful for staged-combustion, gas-generator,
preburner, purge, or other cases where some feeds are already gaseous streams.

This example shows:

    1. Building Reactants from CombustionGas objects.
    2. Using fuels, oxidizers, inerts, and igniters.
    3. Solving HP equilibrium.
    4. Reading chamber-product properties.
    5. Reading product composition.
    6. Converting the products to a CombustionGas object.
    7. Updating pressure, mixture ratio, and extra feed fractions.
"""

from thermoprop import *


psia_to_pa = 6894.76


# ---------------------------------------------------------------------------
# CombustionGas feed streams
# ---------------------------------------------------------------------------

# These are gaseous reactant streams. They are not equilibrium products from
# this script. They are simply known gas streams being fed into a new
# equilibrium calculation.
fuel_gas = CombustionGas(
    "CH4",
    pressure=400.0 * psia_to_pa,
    temperature=500.0,
)

oxidizer_gas = CombustionGas(
    "O2",
    pressure=400.0 * psia_to_pa,
    temperature=500.0,
)

# Inerts can be a pure gas or a gas mixture.
inert_gas = CombustionGas(
    {
        "N2": 0.80,
        "Ar": 0.20,
    },
    basis="mole",
    pressure=400.0 * psia_to_pa,
    temperature=500.0,
)

# Igniters can also be represented as gas streams.
igniter_gas = CombustionGas(
    {
        "H2": 0.70,
        "O2": 0.30,
    },
    basis="mole",
    pressure=400.0 * psia_to_pa,
    temperature=500.0,
)


# ---------------------------------------------------------------------------
# Reactants using fuels, oxidizers, inerts, and igniters
# ---------------------------------------------------------------------------

# The mixture_ratio is still oxidizer mass divided by fuel mass.
#
# inert_fraction and igniter_fraction are relative to the main propellant mass:
#
#     propellant mass = fuel mass + oxidizer mass
#     inert mass = inert_fraction * propellant mass
#     igniter mass = igniter_fraction * propellant mass
reactants = Reactants(
    fuels=fuel_gas,
    oxidizers=oxidizer_gas,
    mixture_ratio=3.4,
    inerts=inert_gas,
    inert_fraction=0.02,
    igniters=igniter_gas,
    igniter_fraction=0.01,
)

print("CombustionGas reactants")
print("Mixture ratio:", reactants.mixture_ratio)
print("Inert fraction:", reactants.inert_fraction)
print("Igniter fraction:", reactants.igniter_fraction)
print("Fuel mass [kg basis]:", reactants.fuel_mass)
print("Oxidizer mass [kg basis]:", reactants.oxidizer_mass)
print("Inert mass [kg basis]:", reactants.inert_mass)
print("Igniter mass [kg basis]:", reactants.igniter_mass)
print("Total reactant mass [kg basis]:", reactants.total_mass)
print("Reactant enthalpy [J/kg]:", reactants.reactant_enthalpy)
print("Reactant internal energy [J/kg]:", reactants.reactant_internal_energy)

print("\nReactant mass fractions:")
print(reactants.mass_fractions)

print("\nReactant mole fractions:")
print(reactants.mole_fractions)

print("\nElement moles per kg:")
print(reactants.element_moles_per_kg)


# ---------------------------------------------------------------------------
# HP equilibrium solve
# ---------------------------------------------------------------------------

chamber = Equilibrium(
    reactants=reactants,
    mode="hp",
    pressure=400.0 * psia_to_pa,
    guess_temperature=3400.0,
)

print("\nHP chamber state")
print("Success:", chamber.success)
print("Message:", chamber.message)
print("Iterations:", chamber.iterations)
print("Outer iterations:", chamber.outer_iterations)
print("Pressure [psia]:", chamber.pressure / psia_to_pa)
print("Temperature [K]:", chamber.temperature)
print("Density [kg/m3]:", chamber.density)
print("Enthalpy [J/kg]:", chamber.enthalpy)
print("Internal energy [J/kg]:", chamber.internal_energy)
print("Entropy [J/kg-K]:", chamber.entropy)
print("Gas constant [J/kg-K]:", chamber.gas_constant)
print("Molecular weight [kg/kmol]:", chamber.molecular_weight)
print("Cp equilibrium [J/kg-K]:", chamber.specific_heat_cp_equilibrium)
print("Cv equilibrium [J/kg-K]:", chamber.specific_heat_cv_equilibrium)
print("Gamma equilibrium:", chamber.gamma_equilibrium)
print("Cp frozen [J/kg-K]:", chamber.specific_heat_cp_frozen)
print("Cv frozen [J/kg-K]:", chamber.specific_heat_cv_frozen)
print("Gamma frozen:", chamber.gamma_frozen)
print("Speed of sound [m/s]:", chamber.speed_of_sound)
print("Dynamic viscosity [Pa-s]:", chamber.dynamic_viscosity)
print("Thermal conductivity [W/m-K]:", chamber.thermal_conductivity)
print("Prandtl number:", chamber.prandtl)


# ---------------------------------------------------------------------------
# Product composition
# ---------------------------------------------------------------------------

print("\nGas mass fractions:")
print(chamber.gas_mass_fractions)

print("\nGas mole fractions:")
print(chamber.gas_mole_fractions)

print("\nAll mass fractions:")
print(chamber.mass_fractions)

print("\nCondensed species:")
print(chamber.condensed_species)

print("\nInserted condensed species during solve:")
print(chamber.inserted_condensed_species)


# ---------------------------------------------------------------------------
# CombustionGas object from equilibrium products
# ---------------------------------------------------------------------------

# combustion_gas returns only the gas-phase product mixture.
# This is useful for downstream components that need gas properties but do not
# need the full equilibrium object.
gas_products = chamber.combustion_gas

print("\nCombustionGas product object")
print("Name:", gas_products.name)
print("Pressure [Pa]:", gas_products.pressure)
print("Temperature [K]:", gas_products.temperature)
print("Density [kg/m3]:", gas_products.density)
print("Gamma:", gas_products.gamma)
print("Mass fractions:", gas_products.mass_fractions)


# ---------------------------------------------------------------------------
# Updating HP equilibrium
# ---------------------------------------------------------------------------

# HP equilibrium enthalpy is fixed by the reactants, so there is no HP setter
# that lets the user directly assign product enthalpy. To change an HP result,
# change the reactants or the pressure.
reactants.update(
    mixture_ratio=3.2,
    inert_fraction=0.01,
    igniter_fraction=0.005,
)

chamber.update(
    reactants=reactants,
    pressure=350.0 * psia_to_pa,
    guess_temperature=chamber.temperature,
)

print("\nAfter changing mixture ratio, inert fraction, igniter fraction, and pressure")
print("Mixture ratio:", reactants.mixture_ratio)
print("Inert fraction:", reactants.inert_fraction)
print("Igniter fraction:", reactants.igniter_fraction)
print("Pressure [psia]:", chamber.pressure / psia_to_pa)
print("Temperature [K]:", chamber.temperature)
print("Gamma:", chamber.gamma)
print("Gas mass fractions:", chamber.gas_mass_fractions)


# ---------------------------------------------------------------------------
# update(..., solve=False)
# ---------------------------------------------------------------------------

# solve=False is useful when several inputs are being changed before solving.
chamber.update(
    pressure=300.0 * psia_to_pa,
    guess_temperature=chamber.temperature,
    solve=False,
)

print("\nAfter update(..., solve=False)")
print("Is stale before solve():", chamber.is_stale)

chamber.solve()

print("Is stale after solve():", chamber.is_stale)
print("Pressure [psia]:", chamber.pressure / psia_to_pa)
print("Temperature [K]:", chamber.temperature)
print("Entropy [J/kg-K]:", chamber.entropy)