# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermodynamic property wrapper for real fluids, mixtures, ideal gases, and liquid rocket propellants.

It provides a clean interface around:

* CoolProp
* PYroMat
* RocketProps
* NumPy
* SciPy

## Why ThermoProp?

ThermoProp provides a unified API around CoolProp, PYroMat, and RocketProps.

Instead of remembering backend-specific syntax such as:

```python
CP.PropsSI(...)
pm.get(...)
get_prop(...)
```

users can write:

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(water.density)
print(water.enthalpy)
```

with a consistent interface for pure fluids, mixtures, ideal gases, and liquid rocket propellants.

## Installation

```bash
pip install thermoprop
```

## Features

### Fluid

`Fluid` is a CoolProp-based real-fluid wrapper.

It supports:

* Pure fluids
* Fluid mixtures
* Pressure-temperature states
* Pressure-enthalpy states
* Pressure-quality states
* Temperature-quality states
* Density-based states
* Mass-fraction and mole-fraction mixtures

### IdealGas

`IdealGas` is a PYroMat-based ideal-gas wrapper.

It supports:

* Pure ideal gases
* Ideal-gas mixtures
* Temperature states
* Enthalpy states
* Internal-energy states
* Pressure-density closure
* Cp, Cv, gamma, entropy, Gibbs energy, and speed of sound

### Propellant

`Propellant` is a RocketProps-based liquid rocket propellant wrapper.

It supports:

* Liquid rocket propellants
* Saturated-liquid properties
* Compressed-liquid properties
* Density
* Dynamic viscosity
* Kinematic viscosity
* Thermal conductivity
* Surface tension
* Vapor pressure
* Saturation temperature
* Heat of vaporization
* Critical properties

`Propellant` is intended for liquid propellant engineering properties. It is not a thermodynamic flash solver and does not calculate vapor-state properties, two-phase states, enthalpy, internal energy, or entropy.

### FluidRegistry

`FluidRegistry` maps user-friendly names and aliases to backend-specific names.

For example:

```python
from thermoprop import FluidRegistry

print(FluidRegistry.coolprop_name("rp-1"))
print(FluidRegistry.propellant_name("rp-1"))
```

outputs different backend names:

```text
n-Dodecane
RP1
```

This is intentional. `Fluid("rp-1")` uses CoolProp's `n-Dodecane` as an RP-1 surrogate, while `Propellant("rp-1")` uses RocketProps' actual `RP1` correlation.

## Thermodynamic Reference States

ThermoProp provides a unified interface to multiple thermodynamic backends.

Different property libraries may use different reference states for properties such as:

* Enthalpy
* Internal energy
* Entropy

As a result, absolute values of these properties may differ between ThermoProp classes even when pressure, temperature, and composition are identical.

For example, two wrappers representing the same physical state may report different absolute enthalpy values if their underlying thermodynamic libraries use different energy reference conventions.

This behavior is expected and does not indicate an error.

Most engineering calculations depend on property differences rather than absolute values. Properties such as:

* Temperature
* Pressure
* Density
* Specific heats
* Speed of sound
* Enthalpy differences (Δh)
* Internal-energy differences (Δu)

remain physically meaningful within each backend.

Users combining results from multiple ThermoProp wrappers should establish a consistent thermodynamic reference basis if absolute values of enthalpy, internal energy, or entropy are required.

## Pure Fluid Example

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(water.density)
print(water.enthalpy)
print(water.phase)
```

## Pressure-Enthalpy Example

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    enthalpy=2.7e6,
)

print(water.temperature)
print(water.quality)
print(water.phase)
```

## Mixture Example

```python
from thermoprop import Fluid

air_like = Fluid(
    {"nitrogen": 0.79, "oxygen": 0.21},
    basis="mole",
    pressure=101325,
    temperature=300,
)

print(air_like.density)
print(air_like.specific_heat_cp)
```

## Ideal Gas Example

```python
from thermoprop import IdealGas

nitrogen = IdealGas(
    "gn2",
    pressure=101325,
    temperature=300,
)

print(nitrogen.density)
print(nitrogen.specific_heat_ratio)
print(nitrogen.speed_of_sound)
```

## Propellant Example

```python
from thermoprop import Propellant

rp1 = Propellant(
    "rp1",
    temperature=293.15,
)

print(rp1.density)
print(rp1.dynamic_viscosity)
print(rp1.vapor_pressure)
```

## Compressed-Liquid Propellant Example

```python
from thermoprop import Propellant

lox = Propellant(
    "lox",
    pressure=3e6,
    temperature=90,
)

print(lox.density)
print(lox.dynamic_viscosity)
print(lox.saturation_pressure)
```

## Propellant Cavitation Margin Example

```python
from thermoprop import Propellant

lox = Propellant(
    "lox",
    pressure=300000,
    temperature=90,
)

margin = lox.pressure - lox.vapor_pressure

print(margin)
```

## Fluid Registry Examples

`FluidRegistry` can be used to inspect supported names, check backend support, and add custom aliases.

### Check Backend Names

```python
from thermoprop import FluidRegistry

print(FluidRegistry.coolprop_name("water"))
print(FluidRegistry.pyromat_name("gn2"))
print(FluidRegistry.propellant_name("rp-1"))
```

Example output:

```text
Water
N2
RP1
```

For PYroMat, the `ig.` prefix can also be requested:

```python
print(FluidRegistry.pyromat_name("gn2", include_prefix=True))
```

Example output:

```text
ig.N2
```

### Check Backend Support

```python
from thermoprop import FluidRegistry

print(FluidRegistry.supports_coolprop("water"))
print(FluidRegistry.supports_pyromat("gn2"))
print(FluidRegistry.supports_propellant("rp-1"))
```

### List Supported Names

```python
from thermoprop import FluidRegistry

print(FluidRegistry.names)
print(FluidRegistry.coolprop_supported_names)
print(FluidRegistry.pyromat_supported_names)
print(FluidRegistry.propellant_supported_names)
```

You can also print supported species directly:

```python
FluidRegistry.show_species()
FluidRegistry.show_coolprop_species()
FluidRegistry.show_pyromat_species()
FluidRegistry.show_propellant_species()
```

### Show Aliases

ThermoProp keeps normal fluid aliases and propellant aliases separate.

```python
from thermoprop import FluidRegistry

FluidRegistry.show_aliases()
FluidRegistry.show_propellant_aliases()
```

This avoids ambiguity. For example:

```python
print(FluidRegistry.coolprop_name("rp-1"))
print(FluidRegistry.propellant_name("rp-1"))
```

returns:

```text
n-Dodecane
RP1
```

### Add Custom Aliases

Use `add_alias()` for `Fluid` and `IdealGas` names:

```python
from thermoprop import Fluid, FluidRegistry

FluidRegistry.add_alias("my-water", "Water")

water = Fluid(
    "my-water",
    pressure=101325,
    temperature=300,
)

print(water.density)
```

Use `add_propellant_alias()` for `Propellant` names:

```python
from thermoprop import Propellant, FluidRegistry

FluidRegistry.add_propellant_alias("my-rp1", "RP1")

rp1 = Propellant(
    "my-rp1",
    temperature=293.15,
)

print(rp1.density)
```

Removing aliases works the same way:

```python
FluidRegistry.remove_alias("my-water")
FluidRegistry.remove_propellant_alias("my-rp1")
```

## Common Properties

```python
from thermoprop import Fluid

fluid = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(fluid.pressure)
print(fluid.temperature)
print(fluid.density)
print(fluid.enthalpy)
print(fluid.entropy)
print(fluid.specific_heat_cp)
print(fluid.specific_heat_cv)
print(fluid.specific_heat_ratio)
print(fluid.speed_of_sound)
print(fluid.dynamic_viscosity)
print(fluid.conductivity)
```

## Updating State Properties

ThermoProp states can be updated after creation.

### Real Fluid

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

water.pressure = 2e5
water.temperature = 350

print(water.density)
print(water.enthalpy)
```

You can also update state pairs directly:

```python
water.pressure_temperature = (2e5, 350)
water.pressure_enthalpy = (2e5, 1.5e6)
water.pressure_quality = (101325, 0.5)
water.temperature_quality = (373.15, 1.0)
```

### Ideal Gas

Ideal gases only require a thermal state such as temperature, enthalpy, or internal energy.

```python
from thermoprop import IdealGas

nitrogen = IdealGas(
    "gn2",
    temperature=300,
)

print(nitrogen.enthalpy)
print(nitrogen.internal_energy)
print(nitrogen.specific_heat_cp)
```

Pressure is optional, but it is required for pressure-dependent properties such as density and entropy:

```python
nitrogen.pressure = 101325

print(nitrogen.density)
print(nitrogen.entropy)
```

You can also update ideal-gas states:

```python
nitrogen.temperature = 500
nitrogen.pressure_temperature = (101325, 300)
nitrogen.pressure_enthalpy = (101325, nitrogen.enthalpy)
```

### Propellant

Propellants require temperature. Pressure is optional.

```python
from thermoprop import Propellant

rp1 = Propellant(
    "rp1",
    temperature=293.15,
)

print(rp1.density)
print(rp1.specific_heat_cp)
```

If pressure is omitted, saturated-liquid properties are used.

```python
rp1.pressure = 2e6

print(rp1.density)
print(rp1.dynamic_viscosity)
```

You can also update the propellant state pair directly:

```python
rp1.pressure_temperature = (2e6, 300)
```

## Propellant Limitations

`Propellant` wraps RocketProps liquid propellant correlations.

It is intended for liquid engineering properties and does not calculate:

* Vapor-state properties
* Two-phase flash states
* Enthalpy
* Internal energy
* Entropy
* Cv
* Specific heat ratio
* Speed of sound

Unsupported properties raise `NotImplementedError`.

## Ideal-Gas Viscosity Limitation

`IdealGas.dynamic_viscosity` uses Sutherland's law.

Currently, viscosity is only supported for selected pure gases, including:

* Air
* Argon
* Carbon dioxide
* Carbon monoxide
* Nitrogen
* Oxygen
* Hydrogen
* Water vapor

Mixture viscosity is not currently supported.

```python
from thermoprop import IdealGas

air = IdealGas(
    "air",
    pressure=101325,
    temperature=300,
)

print(air.dynamic_viscosity)
```

If viscosity data is unavailable for a gas, ThermoProp raises `NotImplementedError`.

## Source Code

GitHub:

https://github.com/saakethramoju/ThermoProp

## License

ThermoProp is released under the GNU General Public License v3.0.

See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
