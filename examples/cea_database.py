"""
CEA database example.

CEA is the strict NASA CEA / CEAM database object used internally by
Equilibrium, CombustionGas, and Propellant.

Most users do not need to call CEA directly. It is useful when you want to:

    1. Check exact CEA species names.
    2. Search for available species or reactant cards.
    3. Inspect elemental composition and molecular weight.
    4. Evaluate raw NASA-polynomial thermodynamic properties.
    5. Check whether transport data exists.

Important naming detail:

    SpeciesDatabase accepts ThermoProp aliases such as "lox" or "rp-1".
    CEA uses strict CEA names such as "O2(L)" or "RP-1".

For example:

    Propellant("lox")

uses ThermoProp aliasing, but:

    CEA.resolve_name("O2(L)")

uses the exact CEA database name.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Basic database lists
# ---------------------------------------------------------------------------

print("Number of strict CEA names:", len(CEA.names))
print("Number of gas species:", len(CEA.gas_species))
print("Number of condensed species:", len(CEA.condensed_species))
print("Number of predefined reactants:", len(CEA.predefined_reactants))
print("Number of species with transport data:", len(CEA.transport_names))

print("\nFirst ten strict CEA names:")
print(CEA.names[:10])


# ---------------------------------------------------------------------------
# Searching for species
# ---------------------------------------------------------------------------

# find_species() is useful when you do not know the exact CEA spelling.
oxygen_matches = CEA.find_species("O2")
rp1_matches = CEA.find_species("RP")

print("\nCEA names containing O2:")
print(oxygen_matches[:20])

print("\nCEA names containing RP:")
print(rp1_matches[:20])


# ---------------------------------------------------------------------------
# Strict name resolution
# ---------------------------------------------------------------------------

# resolve_name() requires an exact CEA name. This is intentional because the CEA
# database has many phase-specific names.
oxygen_liquid = CEA.resolve_name("O2(L)")
oxygen_gas = CEA.resolve_name("O2")
water_gas = CEA.resolve_name("H2O")

print("\nResolved liquid oxygen name:", oxygen_liquid)
print("Resolved oxygen gas name:", oxygen_gas)
print("Resolved water gas name:", water_gas)


# ---------------------------------------------------------------------------
# Species data
# ---------------------------------------------------------------------------

print("\nO2(L) molecular weight [kg/kmol]:", CEA.molecular_weight("O2(L)"))
print("O2 molecular weight [kg/kmol]:", CEA.molecular_weight("O2"))
print("H2O molecular weight [kg/kmol]:", CEA.molecular_weight("H2O"))

print("\nH2O elemental composition:")
print(CEA.elemental_composition("H2O"))

print("\nH2O temperature ranges [K]:")
print(CEA.temperature_ranges("H2O"))

# temperature_limits() takes a list because it is often used to find the common
# temperature range shared by several species.
print("\nH2O temperature limits [K]:")
print(CEA.temperature_limits(["H2O"]))

print("\nCommon temperature limits for H2O and O2 [K]:")
print(CEA.temperature_limits(["H2O", "O2"]))

print("\nSpecies checks")
print("H2O has thermo data:", CEA.has_thermo("H2O"))
print("H2O has transport data:", CEA.has_transport("H2O"))
print("O2(L) is condensed:", CEA.is_condensed("O2(L)"))
print("H2O is gas:", CEA.is_gas("H2O"))
print("RP-1 is predefined reactant:", CEA.is_reactant("RP-1"))


# ---------------------------------------------------------------------------
# Full species description
# ---------------------------------------------------------------------------

# species() returns a CEASpecies object.
water_species = CEA.species("H2O")

print("\nH2O species object")
print("Name:", water_species.name)
print("Index:", water_species.index)
print("Molar mass [kg/mol]:", water_species.molar_mass)
print("Has thermo:", water_species.has_thermo)
print("Has transport:", water_species.has_transport)
print("Is reactant:", water_species.is_reactant)
print("Is condensed:", water_species.is_condensed)
print("Is gas:", water_species.is_gas)
print("Elemental composition:", water_species.elemental_composition)
print("Temperature ranges:", water_species.temperature_ranges)

# describe() returns the same kind of information as a dictionary.
print("\nH2O description dictionary:")
print(CEA.describe("H2O"))


# ---------------------------------------------------------------------------
# Raw thermodynamic properties
# ---------------------------------------------------------------------------

# thermo_mass() returns mass-based thermodynamic values at a temperature.
# These are raw CEA-reference values.
#
# The tuple is:
#
#     specific heat at constant pressure [J/kg-K]
#     enthalpy [J/kg]
#     entropy at standard pressure [J/kg-K]
h2o_thermo = CEA.thermo_mass("H2O", temperature=1000.0)

print("\nH2O thermo_mass at 1000 K:")
print(h2o_thermo)

print("\nH2O cp_mass at 1000 K [J/kg-K]:", CEA.cp_mass("H2O", 1000.0))
print("H2O enthalpy_mass at 1000 K [J/kg]:", CEA.enthalpy_mass("H2O", 1000.0))
print("H2O entropy_mass at 1000 K [J/kg-K]:", CEA.entropy_mass_standard("H2O", 1000.0))

# thermo_molar() gives the same type of information on a molar basis.
h2o_thermo_molar = CEA.thermo_molar("H2O", temperature=1000.0)

print("\nH2O thermo_molar at 1000 K:")
print(h2o_thermo_molar)

print("\nH2O cp_molar at 1000 K [J/mol-K]:", CEA.cp_molar("H2O", 1000.0))
print("H2O enthalpy_molar at 1000 K [J/mol]:", CEA.enthalpy_molar("H2O", 1000.0))
print("H2O entropy_molar at 1000 K [J/mol-K]:", CEA.entropy_molar_standard("H2O", 1000.0))


# ---------------------------------------------------------------------------
# Raw transport properties
# ---------------------------------------------------------------------------

# Transport data are only available for some species and temperature ranges.
print("\nH2O viscosity at 1000 K [Pa-s]:", CEA.viscosity("H2O", 1000.0))
print("H2O conductivity at 1000 K [W/m-K]:", CEA.conductivity("H2O", 1000.0))

print("\nTransport species containing H2O:")
print(CEA.find_transport_species("H2O"))


# ---------------------------------------------------------------------------
# Mole and mass fraction conversion helpers
# ---------------------------------------------------------------------------

names = ["H2O", "O2", "N2"]
mole_fractions = [0.20, 0.30, 0.50]

mass_fractions = CEA.mole_to_mass(names, mole_fractions)

print("\nMole fractions:")
print(mole_fractions)

print("\nConverted mass fractions:")
print(mass_fractions)

mole_fractions_again = CEA.mass_to_mole(names, mass_fractions)

print("\nConverted back to mole fractions:")
print(mole_fractions_again)

print("\nMixture molecular weight [kg/kmol]:")
print(CEA.mixture_molar_mass(names, mole_fractions) * 1000.0)


# ---------------------------------------------------------------------------
# SpeciesDatabase compared with CEA
# ---------------------------------------------------------------------------

# SpeciesDatabase is the ThermoProp alias / compatibility layer.
# It maps user-friendly names to backend-specific names.
print("\nThermoProp species alias examples:")
print("lox ThermoProp record:", SpeciesDatabase.record("lox"))
print("rp-1 ThermoProp record:", SpeciesDatabase.record("rp-1"))

print("\nBackend names from SpeciesDatabase:")
print("lox CEA backend name:", SpeciesDatabase.backend_name("lox", "CEA"))
print("rp-1 CEA backend name:", SpeciesDatabase.backend_name("rp-1", "CEA"))
print("water CEA backend name:", SpeciesDatabase.backend_name("water", "CEA"))

print("\nSupported ThermoProp Propellant names:")
print(SpeciesDatabase.supported_species("Propellant")[:20])

print("\nSupported ThermoProp CombustionGas names:")
print(SpeciesDatabase.supported_species("CombustionGas")[:20])