# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermophysical property library for real fluids, fluid mixtures, ideal gases, ideal gas mixtures, liquid rocket propellants, and isotropic engineering materials.

It provides a unified API around CoolProp, PYroMat, RocketProps, and a built-in engineering material property database.

It provides a clean interface around:

* CoolProp
* PYroMat
* RocketProps
* Built-in Material Database
* NumPy
* SciPy

## Why ThermoProp?

ThermoProp provides a unified API around CoolProp, PYroMat, RocketProps, and a built-in engineering material property database.

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

with a consistent interface for pure fluids, mixtures, ideal gases, liquid rocket propellants, and engineering materials.

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
* Cp, Cv, gamma, entropy, Gibbs energy, Helmholtz energy, and speed of sound
* Thermal expansion coefficient
* Isothermal compressibility
* Selected thermodynamic partial derivatives
* Dynamic & kinematic viscosity (selected species only)
* Approximate Prandtl number
* Approximate thermal conductivity

Transport properties are provided for engineering calculations.

Dynamic viscosity is currently available only for species with
implemented Sutherland-law coefficients. Thermal conductivity and
Prandtl number are estimated using simple kinetic-theory-based
correlations and should be considered approximate.

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

### Material

`Material` is a built-in isotropic engineering material-property wrapper.

It provides temperature-dependent engineering material properties using ThermoProp's integrated material property database.

Supported properties include:

* Density
* Yield strength
* Ultimate strength
* Elastic modulus
* Torsional modulus
* Poisson ratio
* Thermal conductivity
* Specific heat
* Coefficient of thermal expansion
* Thermal diffusivity
* Melting point
* Electrical resistivity

Currently supported materials include:

#### Aluminum Alloys

* Aluminum 6061
* Aluminum 7075

#### Copper Alloys

* Copper C101
* Copper C11000
* Copper C17200
* GRCop-42
* GRCop-84

#### Carbon & Low-Alloy Steels

* 1018 Carbon Steel
* 1045 Carbon Steel
* 3140 Low-Alloy Steel
* 4140 Steel

#### Stainless Steels

* Stainless Steel 303
* Stainless Steel 304
* Stainless Steel 316
* A286 Steel

#### Nickel-Based Superalloys

* Inconel 625
* Inconel 718

#### Ceramics & Non-Metals

* Graphite

### MaterialRegistry

`MaterialRegistry` maps user-friendly aliases to canonical ThermoProp material names.

Material names can be supplied using common aliases:

```python
from thermoprop import Material

mat = Material("in718")
mat = Material("6061")
mat = Material("304ss")
```

`MaterialRegistry` can also be used directly:

```python
from thermoprop import MaterialRegistry

print(MaterialRegistry.name("in718"))
print(MaterialRegistry.name("6061"))
print(MaterialRegistry.name("304ss"))
```

Example output:

```text
Inconel 718
Aluminum 6061
Stainless Steel 304
```

Custom aliases can be added at runtime:

```python
from thermoprop import Material
from thermoprop import MaterialRegistry

MaterialRegistry.add_alias(
    "chamber alloy",
    "Inconel 718",
)

mat = Material(
    "chamber alloy",
    temperature=300,
)

print(mat.yield_strength)
```

Aliases can be removed using:

```python
MaterialRegistry.remove_alias(
    "chamber alloy"
)
```

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

## Property Introspection

All ThermoProp wrappers provide utilities for discovering supported
properties programmatically.

```python
from thermoprop import Fluid

print(Fluid.supported_properties())
```

Print supported properties:

```python
Fluid.show_supported_properties()
```

Check whether a property is supported:

```python
Fluid.supports_property("enthalpy")
```

The same interface is available for:

* Fluid
* IdealGas
* Propellant
* Material

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

## Material Example

```python
from thermoprop import Material

inc718 = Material(
    "in718",
    temperature=300,
)

print(inc718.density)
print(inc718.yield_strength)
print(inc718.thermal_conductivity)
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

## Advanced Thermodynamic Properties

`Fluid` and `IdealGas` provide additional thermodynamic properties when supported by the underlying backend.

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(water.thermal_expansion_coefficient)
print(water.isothermal_compressibility)
print(water.helmholtz_energy)
print(water.gibbs_energy)
print(water.joule_thomson_coefficient)
```

For `Fluid`, these properties are provided by CoolProp.

For `IdealGas`, some properties are analytic ideal-gas results:

* `thermal_expansion_coefficient = 1 / T`
* `isothermal_compressibility = 1 / P`
* `joule_thomson_coefficient = 0`

`Propellant` and `Material` do not support all advanced thermodynamic properties because they are not full thermodynamic equation-of-state wrappers.

## Thermodynamic Partial Derivatives

`Fluid` provides access to CoolProp first partial derivatives using:

```python
fluid.partial_derivative(
    "Hmass",
    "T",
    "P",
)
```

This evaluates:

```text
(∂h/∂T)_P
```

Common derivative shortcuts are also available:

```python
print(fluid.dhdT_const_p)
print(fluid.dhdP_const_t)

print(fluid.drhodT_const_p)
print(fluid.drhodP_const_t)

print(fluid.dTdP_const_h)
```

where:

* `dhdT_const_p` is `(∂h/∂T)_P`
* `dhdP_const_t` is `(∂h/∂P)_T`
* `drhodT_const_p` is `(∂ρ/∂T)_P`
* `drhodP_const_t` is `(∂ρ/∂P)_T`
* `dTdP_const_h` is `(∂T/∂P)_h`

The Joule-Thomson coefficient is exposed as:

```python
print(fluid.joule_thomson_coefficient)
```

which is equivalent to:

```python
fluid.dTdP_const_h
```

`IdealGas` also provides selected analytic partial derivatives for common ideal-gas relationships.

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

* Mixture properties
* Vapor-state properties
* Two-phase flash states
* Enthalpy
* Internal energy
* Entropy
* Cv
* Specific heat ratio
* Speed of sound

Unsupported properties raise `NotImplementedError`.

## Ideal-Gas Transport Properties

`IdealGas` provides approximate transport-property support.

### Dynamic Viscosity

`IdealGas.dynamic_viscosity` uses Sutherland's law.

Viscosity is currently supported only for gases with available Sutherland-law constants. Supported gases include:

* Air
* Argon
* Carbon dioxide
* Carbon monoxide
* Nitrogen
* Oxygen
* Hydrogen
* Water vapor

Ideal-gas mixtures are supported using Wilke's viscosity mixing rule when viscosity data is available for all constituent species.

```python
from thermoprop import IdealGas

air = IdealGas(
    "air",
    pressure=101325,
    temperature=300,
)

print(air.dynamic_viscosity)
```

If viscosity data is unavailable for a species, ThermoProp raises `NotImplementedError`.

### Prandtl Number

`IdealGas.prandtl` provides an approximate Prandtl number based on an Eucken-style kinetic-theory correlation.

```python
print(air.prandtl)
```

This approximation is intended for engineering calculations and should not be considered a replacement for detailed transport-property models.

### Thermal Conductivity

`IdealGas.conductivity` and `IdealGas.thermal_conductivity` provide approximate thermal conductivity estimates computed from:

```text
k = Cp μ / Pr
```

using the wrapper's specific heat, dynamic viscosity, and approximate Prandtl number.

```python
print(air.conductivity)
print(air.thermal_conductivity)
```

These estimates are intended for engineering calculations and do not use detailed transport-property databases such as those employed by CEA, Cantera, or REFPROP.

## Material Limitations

`Material` currently provides temperature-dependent isotropic engineering material properties.

It does not currently support:

* Anisotropic materials
* Composite materials
* Stress-strain curves
* Fatigue data
* Fracture mechanics properties
* Creep data
* Pressure-dependent material behavior

Attempting to access unsupported thermodynamic properties such as enthalpy, entropy, viscosity, or vapor quality will raise `NotImplementedError`.

## Acknowledgments

ThermoProp's isotropic material property database was adapted from material property data compiled and distributed through the MatProtLib project.

The author gratefully acknowledges Tyson Tran and the MatProtLib project for making these engineering material datasets publicly available.

MatProtLib:

https://github.com/tysontran/MatProtLib

## Source Code

GitHub:

https://github.com/saakethramoju/ThermoProp

## License

ThermoProp is released under the GNU General Public License v3.0.

See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
