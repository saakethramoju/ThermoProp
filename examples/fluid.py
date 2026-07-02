"""
Fluid example.

Fluid is the CoolProp real-fluid wrapper. It is the best choice for real-fluid
phase behavior, saturation, vapor quality, and ordinary fluid properties.

Use Fluid when you care about:

    1. Liquid, vapor, supercritical, and two-phase states.
    2. Saturation pressure and saturation temperature.
    3. Quality-based states.
    4. Real-fluid density, viscosity, conductivity, and heat capacity.

This example shows several common flash pairs and the main setters.
"""

from thermoprop import *


# ---------------------------------------------------------------------------
# Pressure-temperature state
# ---------------------------------------------------------------------------

water = Fluid(
    "water",
    pressure=101325.0,
    temperature=300.0,
)

print("Water PT state")
print("Name:", water.name)
print("Backend:", water.backend)
print("Pressure [Pa]:", water.pressure)
print("Temperature [K]:", water.temperature)
print("Phase:", water.phase)
print("Density [kg/m3]:", water.density)
print("Enthalpy [J/kg]:", water.enthalpy)
print("Entropy [J/kg-K]:", water.entropy)
print("Cp [J/kg-K]:", water.specific_heat_cp)
print("Dynamic viscosity [Pa-s]:", water.dynamic_viscosity)
print("Thermal conductivity [W/m-K]:", water.thermal_conductivity)
print("Prandtl number:", water.prandtl)


# ---------------------------------------------------------------------------
# Saturation / quality states
# ---------------------------------------------------------------------------

# Quality is vapor mass fraction: 0 is saturated liquid, 1 is saturated vapor.
saturated_liquid = Fluid("water", pressure=101325.0, quality=0.0)
saturated_vapor = Fluid("water", pressure=101325.0, quality=1.0)

print("\nSaturation states at 1 atm")
print("Saturated liquid temperature [K]:", saturated_liquid.temperature)
print("Saturated liquid density [kg/m3]:", saturated_liquid.density)
print("Saturated vapor temperature [K]:", saturated_vapor.temperature)
print("Saturated vapor density [kg/m3]:", saturated_vapor.density)
heat_of_vaporization = saturated_vapor.enthalpy - saturated_liquid.enthalpy
print("Heat of vaporization [J/kg]:", heat_of_vaporization)


# ---------------------------------------------------------------------------
# Pair setters
# ---------------------------------------------------------------------------

# pressure_temperature changes both values at the same time.
water.pressure_temperature = (2.0e5, 350.0)

print("\nAfter pressure_temperature setter")
print("Pressure [Pa]:", water.pressure)
print("Temperature [K]:", water.temperature)
print("Density [kg/m3]:", water.density)

# pressure_enthalpy is useful after an energy balance.
water.pressure_enthalpy = (101325.0, water.enthalpy)

print("\nAfter pressure_enthalpy setter")
print("Pressure [Pa]:", water.pressure)
print("Temperature [K]:", water.temperature)
print("Enthalpy [J/kg]:", water.enthalpy)

# temperature_quality is useful for saturation at a known temperature.
water.temperature_quality = (373.15, 0.0)

print("\nAfter temperature_quality setter")
print("Temperature [K]:", water.temperature)
print("Pressure [Pa]:", water.pressure)
print("Quality:", water.quality)
print("Density [kg/m3]:", water.density)


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------

# Mixtures are passed as dictionaries. The basis may be "mass" or "mole".
air = Fluid(
    {"Nitrogen": 0.79, "Oxygen": 0.21},
    basis="mole",
    pressure=101325.0,
    temperature=300.0,
)

print("\nCoolProp air-like mixture")
print("Species:", air.species)
print("Basis:", air.basis)
print("Mole fractions:", air.mole_fractions)
print("Mass fractions:", air.mass_fractions)
print("Density [kg/m3]:", air.density)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

water.update(pressure=3.0e5, temperature=400.0)

print("\nAfter update()")
print("Pressure [Pa]:", water.pressure)
print("Temperature [K]:", water.temperature)
print("Phase:", water.phase)
print("Density [kg/m3]:", water.density)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

print("\nSupported flash inputs:", Fluid.supported_flash_inputs())
print("Supports quality:", Fluid.supports_property("quality"))
print("Supports saturation_temperature:", Fluid.supports_property("saturation_temperature"))
