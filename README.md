# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermodynamic property wrapper for fluids, mixtures, and ideal gases.

It provides a clean interface around:

- CoolProp
- PYroMat
- NumPy
- SciPy

## Why ThermoProp?

ThermoProp provides a unified API around CoolProp and PYroMat.

Instead of remembering backend-specific syntax such as:

```python
CP.PropsSI(...)
pm.get(...)
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

with a consistent interface for pure fluids, mixtures, and ideal gases.

## Installation

```bash
pip install thermoprop
```

## Features

### Fluid

`Fluid` is a CoolProp-based real-fluid wrapper.

It supports:

- Pure fluids
- Fluid mixtures
- Pressure-temperature states
- Pressure-enthalpy states
- Pressure-quality states
- Temperature-quality states
- Density-based states
- Mass-fraction and mole-fraction mixtures

### IdealGas

`IdealGas` is a PYroMat-based ideal-gas wrapper.

It supports:

- Pure ideal gases
- Ideal-gas mixtures
- Temperature states
- Enthalpy states
- Internal-energy states
- Pressure-density closure
- Cp, Cv, gamma, entropy, Gibbs energy, and speed of sound

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

## Ideal-Gas Viscosity Limitation

`IdealGas.dynamic_viscosity` uses Sutherland's law.

Currently, viscosity is only supported for selected pure gases, including:

- Air
- Argon
- Carbon dioxide
- Carbon monoxide
- Nitrogen
- Oxygen
- Hydrogen
- Water vapor

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