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

## Source Code

GitHub:

https://github.com/saakethramoju/ThermoProp

## License

ThermoProp is released under the GNU General Public License v3.0.

See `LICENSE` and `THIRD_PARTY_LICENSES.md`.