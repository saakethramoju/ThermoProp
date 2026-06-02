# ThermoProp

A Python thermodynamic property wrapper for pure fluids and fluid mixtures.

ThermoProp provides a simplified interface to:

- CoolProp
- PYroMat

while exposing a consistent API for thermodynamic state calculations.

## Installation

```bash
pip install thermoprop
```

## Example

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300
)

print(water.density)
print(water.enthalpy)
```

## Dependencies

- CoolProp
- NumPy
- SciPy
- PYroMat

## Source Code

GitHub:

https://github.com/saakethramoju/ThermoProp