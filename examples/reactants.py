"""
Reactants example.

Reactants combines fuels, oxidizers, inerts, and igniters into the feed mixture
used by Equilibrium.

The most common workflow is:

    1. Create Propellant or CombustionGas objects.
    2. Pass them to Reactants.
    3. Choose a mixture ratio.
    4. Pass the Reactants object to Equilibrium.

Reactants is needed because chemical equilibrium needs more than just names. It
also needs the reactant temperatures, pressures, enthalpies, element amounts,
and relative masses.

This example shows:

    1. A simple fuel / oxidizer pair.
    2. The main Reactants getters.
    3. Mixture-ratio, inert-fraction, and igniter-fraction setters.
    4. Multiple fuels, oxidizers, inerts, and igniters.
    5. Weight setters.
    6. Mass-fraction setters.
    7. Group replacement setters.
    8. update().
    9. Reactant entry objects.
    10. element_vector(), as_dict(), cache_key(), and string output.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Simple fuel / oxidizer reactants
# ---------------------------------------------------------------------------

# Pressure is included here because reactant_internal_energy needs enough
# information to compute u = h - P / rho for liquid propellants.
fuel = Propellant("rp-1", temperature=298.15, pressure=101325.0)
oxidizer = Propellant("lox", temperature=90.17, pressure=101325.0)

reactants = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=2.3,
)

print("Simple RP-1 / LOX reactants")

print("\nName information")
print("Fuel input name:", fuel.input_name)
print("Fuel registry name:", fuel.registry_name)
print("Fuel RocketProps name:", fuel.rocketprops_name)
print("Fuel CEA name:", fuel.cea_name)

print("Oxidizer input name:", oxidizer.input_name)
print("Oxidizer registry name:", oxidizer.registry_name)
print("Oxidizer RocketProps name:", oxidizer.rocketprops_name)
print("Oxidizer CEA name:", oxidizer.cea_name)
print("Oxidizer phase:", oxidizer.phase)

print("\nRatio getters")
print("Mixture ratio:", reactants.mixture_ratio)
print("Oxidizer-to-fuel ratio:", reactants.oxidizer_to_fuel_ratio)
print("Inert fraction:", reactants.inert_fraction)
print("Igniter fraction:", reactants.igniter_fraction)

print("\nMass getters")
print("Fuel mass [kg basis]:", reactants.fuel_mass)
print("Oxidizer mass [kg basis]:", reactants.oxidizer_mass)
print("Inert mass [kg basis]:", reactants.inert_mass)
print("Igniter mass [kg basis]:", reactants.igniter_mass)
print("Propellant mass [kg basis]:", reactants.propellant_mass)
print("Total reactant mass [kg basis]:", reactants.total_mass)

print("\nMole and molecular-weight getters")
print("Total moles [mol basis]:", reactants.total_moles)
print("Total kmoles [kmol basis]:", reactants.total_kmoles)
print("Molecular weight [kg/mol]:", reactants.molecular_weight)
print("Molecular weight [kg/kmol]:", reactants.molecular_weight_kg_per_kmol)

print("\nThermochemistry getters")
print("Reactant enthalpy [J/kg]:", reactants.reactant_enthalpy)
print("Reactant internal energy [J/kg]:", reactants.reactant_internal_energy)

print("\nMass fractions")
print(reactants.mass_fractions)

print("\nMole fractions")
print(reactants.mole_fractions)

print("\nFuel mass fractions")
print(reactants.fuel_mass_fractions)

print("\nOxidizer mass fractions")
print(reactants.oxidizer_mass_fractions)

print("\nInert mass fractions")
print(reactants.inert_mass_fractions)

print("\nIgniter mass fractions")
print(reactants.igniter_mass_fractions)

print("\nElement moles")
print(reactants.element_moles)

print("\nElement moles per kg")
print(reactants.element_moles_per_kg)

print("\nInput group getters")
print("Fuel inputs:", reactants.fuel_inputs)
print("Oxidizer inputs:", reactants.oxidizer_inputs)
print("Inert inputs:", reactants.inert_inputs)
print("Igniter inputs:", reactants.igniter_inputs)

print("\nWeight getters")
print("Fuel weights:", reactants.fuel_weights)
print("Oxidizer weights:", reactants.oxidizer_weights)
print("Inert weights:", reactants.inert_weights)
print("Igniter weights:", reactants.igniter_weights)


# ---------------------------------------------------------------------------
# mixture_ratio, inert_fraction, and igniter_fraction setters
# ---------------------------------------------------------------------------

# Changing mixture_ratio rebuilds the internal mass and mole bookkeeping.
reactants.mixture_ratio = 2.6

print("\nAfter changing mixture ratio")
print("Mixture ratio:", reactants.mixture_ratio)
print("Fuel mass [kg basis]:", reactants.fuel_mass)
print("Oxidizer mass [kg basis]:", reactants.oxidizer_mass)
print("Total reactant mass [kg basis]:", reactants.total_mass)
print("Reactant enthalpy [J/kg]:", reactants.reactant_enthalpy)

# These setters work even if no inert or igniter group is currently attached.
# If there is no inert or igniter group, the corresponding added mass is zero.
reactants.inert_fraction = 0.0
reactants.igniter_fraction = 0.0

print("\nAfter changing inert_fraction and igniter_fraction")
print("Inert fraction:", reactants.inert_fraction)
print("Igniter fraction:", reactants.igniter_fraction)
print("Inert mass [kg basis]:", reactants.inert_mass)
print("Igniter mass [kg basis]:", reactants.igniter_mass)


# ---------------------------------------------------------------------------
# Multiple fuels, oxidizers, inerts, and igniters
# ---------------------------------------------------------------------------

# A group can be one object, or a list of objects.
# If a tuple is used, the second value is the relative weight inside that group.
methane = Propellant("ch4", temperature=298.15, pressure=101325.0)
ethanol = Propellant("ethanol", temperature=298.15, pressure=101325.0)
lox = Propellant("lox", temperature=90.17, pressure=101325.0)

# CombustionGas is useful for gaseous inert or igniter streams.
nitrogen = CombustionGas("N2", pressure=101325.0, temperature=300.0)
argon = CombustionGas("Ar", pressure=101325.0, temperature=300.0)

hydrogen = CombustionGas("H2", pressure=101325.0, temperature=300.0)
oxygen_gas = CombustionGas("O2", pressure=101325.0, temperature=300.0)

blended_reactants = Reactants(
    fuels=[
        (methane, 0.80),
        (ethanol, 0.20),
    ],
    oxidizers=lox,
    mixture_ratio=3.0,
    inerts=[
        (nitrogen, 0.75),
        (argon, 0.25),
    ],
    inert_fraction=0.02,
    igniters=[
        (hydrogen, 0.70),
        (oxygen_gas, 0.30),
    ],
    igniter_fraction=0.01,
)

print("\nBlended reactants")
print(blended_reactants)

print("\nInput group getters")
print("Fuel inputs:", blended_reactants.fuel_inputs)
print("Oxidizer inputs:", blended_reactants.oxidizer_inputs)
print("Inert inputs:", blended_reactants.inert_inputs)
print("Igniter inputs:", blended_reactants.igniter_inputs)

print("\nWeight getters")
print("Fuel weights:", blended_reactants.fuel_weights)
print("Oxidizer weights:", blended_reactants.oxidizer_weights)
print("Inert weights:", blended_reactants.inert_weights)
print("Igniter weights:", blended_reactants.igniter_weights)

print("\nGroup mass-fraction getters")
print("Fuel mass fractions:", blended_reactants.fuel_mass_fractions)
print("Oxidizer mass fractions:", blended_reactants.oxidizer_mass_fractions)
print("Inert mass fractions:", blended_reactants.inert_mass_fractions)
print("Igniter mass fractions:", blended_reactants.igniter_mass_fractions)

print("\nTotal mass fractions")
print(blended_reactants.mass_fractions)


# ---------------------------------------------------------------------------
# Weight setters
# ---------------------------------------------------------------------------

# These setters change the relative weights inside each group.
# The fuel group still has 1 kg total mass.
# The oxidizer group still has mixture_ratio kg total mass.
# The weights only control how each group is split between entries.
blended_reactants.fuel_weights = [0.70, 0.30]
blended_reactants.oxidizer_weights = [1.0]
blended_reactants.inert_weights = [0.60, 0.40]
blended_reactants.igniter_weights = [0.50, 0.50]

print("\nAfter weight setters")
print("Fuel weights:", blended_reactants.fuel_weights)
print("Oxidizer weights:", blended_reactants.oxidizer_weights)
print("Inert weights:", blended_reactants.inert_weights)
print("Igniter weights:", blended_reactants.igniter_weights)
print("Fuel mass fractions:", blended_reactants.fuel_mass_fractions)
print("Inert mass fractions:", blended_reactants.inert_mass_fractions)
print("Igniter mass fractions:", blended_reactants.igniter_mass_fractions)


# ---------------------------------------------------------------------------
# set_*_weights() methods
# ---------------------------------------------------------------------------

# These methods do the same thing as assigning to the weight properties.
blended_reactants.set_fuel_weights([0.85, 0.15])
blended_reactants.set_oxidizer_weights([1.0])
blended_reactants.set_inert_weights([0.90, 0.10])
blended_reactants.set_igniter_weights([0.60, 0.40])

print("\nAfter set_*_weights() methods")
print("Fuel weights:", blended_reactants.fuel_weights)
print("Oxidizer weights:", blended_reactants.oxidizer_weights)
print("Inert weights:", blended_reactants.inert_weights)
print("Igniter weights:", blended_reactants.igniter_weights)


# ---------------------------------------------------------------------------
# Mass-fraction setters
# ---------------------------------------------------------------------------

# These setters are useful when you want to specify each group using normalized
# mass fractions instead of arbitrary weights.
#
# The keys must match names already present in the group. To keep the example
# independent of exact CEA naming, this example gets the existing keys first.
fuel_names = list(blended_reactants.fuel_mass_fractions)
oxidizer_names = list(blended_reactants.oxidizer_mass_fractions)
inert_names = list(blended_reactants.inert_mass_fractions)
igniter_names = list(blended_reactants.igniter_mass_fractions)

blended_reactants.fuel_mass_fractions = {
    fuel_names[0]: 0.75,
    fuel_names[1]: 0.25,
}

blended_reactants.oxidizer_mass_fractions = {
    oxidizer_names[0]: 1.00,
}

blended_reactants.inert_mass_fractions = {
    inert_names[0]: 0.80,
    inert_names[1]: 0.20,
}

blended_reactants.igniter_mass_fractions = {
    igniter_names[0]: 0.65,
    igniter_names[1]: 0.35,
}

print("\nAfter mass-fraction setters")
print("Fuel mass fractions:", blended_reactants.fuel_mass_fractions)
print("Oxidizer mass fractions:", blended_reactants.oxidizer_mass_fractions)
print("Inert mass fractions:", blended_reactants.inert_mass_fractions)
print("Igniter mass fractions:", blended_reactants.igniter_mass_fractions)


# ---------------------------------------------------------------------------
# Group replacement setters
# ---------------------------------------------------------------------------

# These methods replace the objects inside a group.
# They are useful when the chemistry setup changes, not just the mixture ratio.
rp1 = Propellant("rp-1", temperature=298.15, pressure=101325.0)
methane_cold = Propellant("ch4", temperature=120.0, pressure=2.0e6)
oxygen_cold = Propellant("lox", temperature=90.17, pressure=101325.0)

blended_reactants.set_fuels([
    (rp1, 0.50),
    (methane_cold, 0.50),
])

blended_reactants.set_oxidizers(oxygen_cold)

blended_reactants.set_inerts([
    (nitrogen, 1.0),
])

blended_reactants.set_igniters([
    (hydrogen, 1.0),
])

print("\nAfter group replacement setters")
print("Fuel inputs:", blended_reactants.fuel_inputs)
print("Oxidizer inputs:", blended_reactants.oxidizer_inputs)
print("Inert inputs:", blended_reactants.inert_inputs)
print("Igniter inputs:", blended_reactants.igniter_inputs)
print("Fuel mass fractions:", blended_reactants.fuel_mass_fractions)
print("Oxidizer mass fractions:", blended_reactants.oxidizer_mass_fractions)
print("Inert mass fractions:", blended_reactants.inert_mass_fractions)
print("Igniter mass fractions:", blended_reactants.igniter_mass_fractions)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

# update() changes several Reactants inputs at once.
blended_reactants.update(
    mixture_ratio=2.8,
    inert_fraction=0.01,
    igniter_fraction=0.005,
    fuel_weights=[0.90, 0.10],
    oxidizer_weights=[1.0],
    inert_weights=[1.0],
    igniter_weights=[1.0],
)

print("\nAfter update() with weights")
print("Mixture ratio:", blended_reactants.mixture_ratio)
print("Inert fraction:", blended_reactants.inert_fraction)
print("Igniter fraction:", blended_reactants.igniter_fraction)
print("Fuel weights:", blended_reactants.fuel_weights)
print("Mass fractions:", blended_reactants.mass_fractions)

# update() can also use group mass fractions.
fuel_names = list(blended_reactants.fuel_mass_fractions)

blended_reactants.update(
    fuel_mass_fractions={
        fuel_names[0]: 0.60,
        fuel_names[1]: 0.40,
    },
)

print("\nAfter update() with fuel_mass_fractions")
print("Fuel mass fractions:", blended_reactants.fuel_mass_fractions)

# update() can also replace groups.
blended_reactants.update(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=2.3,
    inerts=None,
    inert_fraction=0.0,
    igniters=None,
    igniter_fraction=0.0,
)

print("\nAfter update() replacing groups")
print("Fuel inputs:", blended_reactants.fuel_inputs)
print("Oxidizer inputs:", blended_reactants.oxidizer_inputs)
print("Inert inputs:", blended_reactants.inert_inputs)
print("Igniter inputs:", blended_reactants.igniter_inputs)
print("Mixture ratio:", blended_reactants.mixture_ratio)
print("Mass fractions:", blended_reactants.mass_fractions)


# ---------------------------------------------------------------------------
# Updating after changing a contained Propellant
# ---------------------------------------------------------------------------

# Reactants stores precomputed mixture bookkeeping. If a contained Propellant
# or CombustionGas object changes state, call update() with no arguments.
fuel.temperature = 350.0
oxidizer.pressure_temperature = (2.0e6, 92.0)

blended_reactants.update()

print("\nAfter changing the contained propellants and calling update()")
print("Fuel temperature [K]:", fuel.temperature)
print("Oxidizer pressure [Pa]:", oxidizer.pressure)
print("Oxidizer temperature [K]:", oxidizer.temperature)
print("Reactant enthalpy [J/kg]:", blended_reactants.reactant_enthalpy)
print("Reactant internal energy [J/kg]:", blended_reactants.reactant_internal_energy)


# ---------------------------------------------------------------------------
# Reactant entry objects
# ---------------------------------------------------------------------------

# The entries list is what Equilibrium ultimately uses internally.
# Each entry has a role, mass, mole count, temperature, pressure, enthalpy, and
# elemental composition.
print("\nReactant entries")

for entry in blended_reactants.entries:
    print("\nEntry name:", entry.name)
    print("CEA name:", entry.cea_name)
    print("Role:", entry.role)
    print("Mass [kg basis]:", entry.mass)
    print("Moles [mol basis]:", entry.moles)
    print("Kmoles [kmol basis]:", entry.kmoles)
    print("Temperature [K]:", entry.temperature)
    print("Pressure [Pa]:", entry.pressure)
    print("Enthalpy [J/kg]:", entry.enthalpy)
    print("Internal energy [J/kg]:", entry.internal_energy)
    print("Elemental composition:", entry.elemental_composition)


# ---------------------------------------------------------------------------
# Element vector
# ---------------------------------------------------------------------------

# Equilibrium needs the element inventory as a vector.
# If no element list is given, Reactants chooses the elements that are present.
elements, element_vector = blended_reactants.element_vector()

print("\nElement vector using automatic element order")
print("Elements:", elements)
print("Vector:", element_vector)

# You can also request a specific element order.
elements, element_vector = blended_reactants.element_vector(["C", "H", "O", "N", "Ar"])

print("\nElement vector using requested element order")
print("Elements:", elements)
print("Vector:", element_vector)


# ---------------------------------------------------------------------------
# as_dict(), cache_key(), and string output
# ---------------------------------------------------------------------------

# as_dict() is useful for inspection or debugging.
reactants_dictionary = blended_reactants.as_dict()

print("\nDictionary output")
print(reactants_dictionary)

# cache_key() is mainly used by FullFlow Lookup caching. Most users will not
# need it directly, but it is useful to know it exists.
print("\nCache key")
print(blended_reactants.cache_key())

# Printing a Reactants object gives a short readable summary.
print("\nString output")
print(blended_reactants)