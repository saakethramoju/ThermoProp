# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermophysical-property and chemical-equilibrium library for engineering analysis, propulsion, thermodynamics, heat transfer, and fluid systems.

It provides a unified API for:

* Real fluids
* Fluid mixtures
* Ideal gases
* Ideal-gas mixtures
* Liquid rocket propellants
* Combustion-product gases
* Chemical-equilibrium calculations
* Engineering materials

ThermoProp integrates multiple industry-standard databases and libraries behind a consistent interface:

* CoolProp
* PYroMat
* RocketProps
* NASA CEA / CEAM thermochemical databases
* Built-in engineering material databases

---

# Installation

```bash
pip install thermoprop
```

Requires Python 3.11 or newer.

---

# Why ThermoProp?

Engineering workflows often require multiple property libraries.

A single project may need:

* CoolProp for real-fluid properties
* PYroMat for ideal-gas thermodynamics
* RocketProps for propellant properties
* NASA CEA for combustion products
* Material databases for structural analysis

Each library has its own API, naming conventions, units, and capabilities.

ThermoProp provides a common interface across all of them.

Instead of:

```python
CP.PropsSI(...)
pm.get(...)
get_prop(...)
```

you can write:

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(water.density)
print(water.enthalpy)
print(water.entropy)
```

and use nearly identical syntax throughout the package.

---

# Wrapper Selection Guide

| Need                          | Wrapper       |
| ----------------------------- | ------------- |
| Real-fluid thermodynamics     | Fluid         |
| Ideal-gas thermodynamics      | IdealGas      |
| Liquid propellant properties  | Propellant    |
| Combustion-product properties | CombustionGas |
| Reactant mixture definition   | Reactants     |
| Chemical equilibrium          | Equilibrium   |
| Engineering materials         | Material      |

---

# Core Wrappers

## Fluid

`Fluid` is a CoolProp-based real-fluid wrapper.

Supported features include:

* Pure fluids
* Fluid mixtures
* Pressure-temperature states
* Pressure-enthalpy states
* Pressure-quality states
* Temperature-quality states
* Density-based states
* Internal-energy states
* Entropy-based states
* Transport properties
* Thermodynamic derivatives
* Advanced equation-of-state properties

Example:

```python
from thermoprop import Fluid

water = Fluid(
    "water",
    pressure=101325,
    temperature=300,
)

print(water.density)
print(water.enthalpy)
print(water.entropy)
print(water.phase)
```

---

## IdealGas

`IdealGas` is a PYroMat-based ideal-gas wrapper.

Supported features include:

* Pure ideal gases
* Ideal-gas mixtures
* Temperature states
* Enthalpy states
* Internal-energy states
* Pressure-density closure
* Entropy calculations
* Gibbs free energy
* Helmholtz free energy
* Speed of sound
* Specific heat ratio
* Thermal expansion coefficient
* Isothermal compressibility
* Thermodynamic derivatives

Transport-property support includes:

* Dynamic viscosity
* Kinematic viscosity
* Thermal conductivity
* Prandtl number
* Mixture transport-property calculations

NASA CEA transport-property correlations are used whenever available.

Example:

```python
from thermoprop import IdealGas

nitrogen = IdealGas(
    "gn2",
    pressure=101325,
    temperature=300,
)

print(nitrogen.specific_heat_ratio)
print(nitrogen.speed_of_sound)
print(nitrogen.dynamic_viscosity)
```

---

## Propellant

`Propellant` provides engineering and thermodynamic properties for liquid rocket propellants, gaseous propellants, and NASA CEA reactants.

Depending on the species, ThermoProp automatically selects the most appropriate backend:

* RocketProps
* NASA CEA

Supported features include:

* Density
* Viscosity
* Conductivity
* Surface tension
* Vapor pressure
* Heat of vaporization
* Heat of formation
* Enthalpy
* Internal energy
* Entropy
* Standard entropy
* Molecular weight
* Elemental composition
* Critical properties

Common propellants include:

* LOX
* RP-1
* Methane
* Hydrogen
* MMH
* UDMH
* N₂O₄
* MON blends
* Aerozine-50

Example:

```python
from thermoprop import Propellant

rp1 = Propellant(
    "rp1",
    temperature=293.15,
)

print(rp1.density)
print(rp1.dynamic_viscosity)
```

---

## CombustionGas

`CombustionGas` evaluates thermodynamic and transport properties using NASA CEA thermochemical and transport databases.

Supported features include:

* Pure species
* Multi-species mixtures
* Mole-fraction compositions
* Mass-fraction compositions
* Thermodynamic properties
* Transport properties
* Mixture viscosity
* Mixture conductivity
* Mixture Prandtl number
* Speed of sound
* Specific heat ratio

Example:

```python
from thermoprop import CombustionGas

gas = CombustionGas(
    {
        "CO2": 0.25,
        "H2O": 0.35,
        "CO": 0.05,
        "N2": 0.35,
    },
    basis="mole",
    pressure=2e6,
    temperature=3000,
)

print(gas.specific_heat_ratio)
print(gas.dynamic_viscosity)
```

---

## Reactants

`Reactants` defines reactant mixtures independently of equilibrium calculations.

Supported features include:

* Mass-fraction mixtures
* Mole-fraction mixtures
* Element accounting
* Molecular-weight calculations
* Heat-of-formation calculations
* Thermodynamic-property evaluation
* Mixture composition inspection

Example:

```python
from thermoprop import Reactants

reactants = Reactants(
    {
        "LOX": 0.70,
        "RP-1": 0.30,
    },
    basis="mass",
)

print(reactants.elemental_composition)
```

---

## Equilibrium

`Equilibrium` performs chemical-equilibrium calculations using Gibbs free-energy minimization.

Supported features include:

* Arbitrary reactant mixtures
* Element conservation constraints
* Equilibrium composition prediction
* Equilibrium thermodynamic properties
* Frozen-composition evaluation
* Combustion-product generation

Example:

```python
from thermoprop import Reactants
from thermoprop import Equilibrium

reactants = Reactants(
    {
        "LOX": 0.70,
        "RP-1": 0.30,
    },
    basis="mass",
)

eq = Equilibrium(
    reactants,
    pressure=2e6,
    temperature=3500,
)

print(eq.composition)
```

---

## Material

`Material` provides temperature-dependent engineering material properties.

Supported properties include:

* Density
* Yield strength
* Ultimate strength
* Elastic modulus
* Shear modulus
* Poisson ratio
* Thermal conductivity
* Specific heat
* Thermal expansion coefficient
* Thermal diffusivity
* Electrical resistivity
* Melting point

Supported materials include:

* Aluminum alloys
* Copper alloys
* Carbon steels
* Stainless steels
* Nickel superalloys
* Graphite

Example:

```python
from thermoprop import Material

inc718 = Material(
    "in718",
    temperature=300,
)

print(inc718.yield_strength)
print(inc718.thermal_conductivity)
```

---

# Built-In Databases

## SpeciesDatabase

ThermoProp includes a unified species database covering:

* CoolProp fluids
* PYroMat species
* RocketProps propellants
* NASA CEA species
* NASA CEA reactants

Example:

```python
from thermoprop import species

print(species())
```

---

## MaterialDatabase

ThermoProp includes a built-in engineering-material database.

Example:

```python
from thermoprop import materials

print(materials())
```

---

## CEADatabase

ThermoProp includes direct access to NASA CEA / CEAM thermochemical and transport data.

Supported features include:

* NASA-9 thermodynamic polynomials
* Transport-property correlations
* Species discovery
* Species inspection
* Molecular-weight lookup
* Elemental-composition lookup
* Transport-coefficient lookup

Example:

```python
from thermoprop import CEA

print(CEA.gas_species)
print(CEA.elemental_composition("CO2"))
```

---

# NASA CEA Integration

ThermoProp includes a native NASA CEA / CEAM database interface.

Unlike workflows that require external Fortran executables or third-party wrappers, ThermoProp ships with parsed thermodynamic and transport databases directly accessible from Python.

Supported NASA CEA functionality includes:

* NASA-9 thermodynamic polynomials
* Thermodynamic-property evaluation
* Transport-property evaluation
* Species discovery
* Reactant definitions
* Combustion-product properties
* Chemical-equilibrium calculations

---

# Property Discovery

All wrappers provide runtime introspection utilities.

Discover supported properties:

```python
from thermoprop import Fluid

print(Fluid.supported_properties())
```

Discover supported flash inputs:

```python
print(Fluid.supported_flash_inputs())
```

Discover available species:

```python
print(Fluid.get_available_species())
```

Discover available materials:

```python
from thermoprop import Material

print(Material.get_available_materials())
```

---

# Thermodynamic Reference States

ThermoProp provides a unified API across multiple thermodynamic backends.

Different libraries use different reference-state conventions for:

* Enthalpy
* Internal energy
* Entropy

As a result, absolute values of these properties may differ between wrappers even when pressure, temperature, and composition are identical.

This behavior is expected.

Property differences remain physically meaningful within each backend.

When comparing results across wrappers, users should establish a consistent thermodynamic reference basis if absolute thermodynamic values are required.

Backends used by ThermoProp include:

* CoolProp
* PYroMat
* RocketProps
* NASA CEA

Each backend may define its own thermodynamic reference state.

---

# Updating States

ThermoProp wrappers support state updates after creation.

Example:

```python
water.pressure = 2e5
water.temperature = 350
```

or:

```python
water.pressure_temperature = (
    2e5,
    350,
)
```

State-update capabilities depend on the wrapper and selected backend.

---

# Limitations

## Fluid

* Limited by CoolProp fluid availability.
* Mixture support follows CoolProp capabilities.

## IdealGas

* Assumes ideal-gas behavior.
* Not intended for dense-gas or near-critical states.

## Propellant

* Primarily intended for engineering propellant properties.
* Property availability depends on the selected backend.

## Equilibrium

* Assumes chemical equilibrium.
* Does not model finite-rate chemistry.
* Does not model transient reaction kinetics.

## Material

* Temperature dependent only.
* No anisotropic material support.
* No fatigue data.
* No creep data.
* No fracture-mechanics data.

---

# Documentation

Full documentation:

https://saakethramoju.github.io/softwares/thermoprop/

Source code:

https://github.com/saakethramoju/ThermoProp

---

# Acknowledgments

ThermoProp incorporates or utilizes data from:

* CoolProp
* PYroMat
* RocketProps
* NASA CEA / CEAM
* MatProtLib

Special thanks to Tyson Tran and the MatProtLib project for making engineering material datasets publicly available.

The author also gratefully acknowledges the NASA Glenn Research Center and the NASA CEA development team for making thermochemical and transport datasets publicly available.

---

# License

ThermoProp is released under the GNU General Public License v3.0.

See:

* LICENSE
* THIRD_PARTY_LICENSES.md
