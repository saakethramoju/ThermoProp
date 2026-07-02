"""
Material example.

Material stores engineering solid-material properties such as density, thermal
conductivity, specific heat, elastic modulus, and strength values.

Material is used in heat-transfer and structural-style models where the solid
properties may depend on temperature.

This example shows:

    1. Creating a material.
    2. Reading common properties.
    3. Changing temperature with a setter.
    4. Using pressure_temperature for API consistency.
    5. Checking available properties and units.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Basic material state
# ---------------------------------------------------------------------------

steel = Material("1018", temperature=298.15)

print("Material name:", steel.name)
print("Backend:", steel.backend)
print("Category:", steel.category)
print("Phase:", steel.phase)
print("Temperature [K]:", steel.temperature)
print("Density [kg/m3]:", steel.density)
print("Specific heat [J/kg-K]:", steel.specific_heat)
print("Thermal conductivity [W/m-K]:", steel.thermal_conductivity)
print("Thermal diffusivity [m2/s]:", steel.thermal_diffusivity)
print("Coefficient of thermal expansion [1/K]:", steel.coefficient_of_thermal_expansion)
print("Elastic modulus [Pa]:", steel.elastic_modulus)
print("Yield strength [Pa]:", steel.yield_strength)
print("Ultimate strength [Pa]:", steel.ultimate_strength)


# ---------------------------------------------------------------------------
# Temperature setter
# ---------------------------------------------------------------------------

# Many material properties are temperature dependent.
steel.temperature = 800.0

print("\nAfter changing temperature")
print("Temperature [K]:", steel.temperature)
print("Specific heat [J/kg-K]:", steel.specific_heat)
print("Thermal conductivity [W/m-K]:", steel.thermal_conductivity)
print("Density [kg/m3]:", steel.density)


# ---------------------------------------------------------------------------
# update() and pressure_temperature
# ---------------------------------------------------------------------------

# Material properties are pressure-independent. Pressure is always None.
# The pressure_temperature property still exists so Material has a similar API
# shape to Fluid, IdealGas, and CombustionGas.
steel.update(temperature=500.0)

print("\nAfter update()")
print("Temperature [K]:", steel.temperature)
print("Specific heat [J/kg-K]:", steel.specific_heat)

# Use None for pressure because material properties only depend on temperature.
steel.pressure_temperature = (None, 600.0)

print("\nAfter pressure_temperature setter")
print("Pressure [Pa]:", steel.pressure)
print("Temperature [K]:", steel.temperature)
print("Thermal conductivity [W/m-K]:", steel.thermal_conductivity)


# ---------------------------------------------------------------------------
# Available properties and units
# ---------------------------------------------------------------------------

print("\nAvailable properties for this material:")
print(steel.available_properties)

print("\nUnits for thermal_conductivity:")
print(steel.units("thermal_conductivity"))

print("\nTemperature range for specific_heat:")
print(steel.temperature_range("specific_heat"))

print("\nNumber of supported materials:", len(Material.get_available_materials()))
print("Supports specific_heat:", Material.supports_property("specific_heat"))
print("Supports quality:", Material.supports_property("quality"))