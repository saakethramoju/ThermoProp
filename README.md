# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/pypi/pyversions/thermoprop)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermodynamic property wrapper for fluids, mixtures, and ideal gases.

It provides a clean interface around:

- CoolProp
- PYroMat
- NumPy
- SciPy

## Installation

```bash
pip install thermoprop
```

## Pure Fluid Example

```python
from thermoprop import Fluid

water = Fluid("water", pressure=101325, temperature=300)

print(water.density)
print(water.enthalpy)
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

nitrogen = IdealGas("gn2", pressure=101325, temperature=300)

print(nitrogen.density)
print(nitrogen.specific_heat_ratio)
print(nitrogen.speed_of_sound)
```

## Source Code

https://github.com/saakethramoju/ThermoProp

## License

ThermoProp is released under the GNU General Public License v3.0.

See `LICENSE` and `THIRD_PARTY_LICENSES.md`.