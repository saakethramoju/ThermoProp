# ThermoProp

[![PyPI version](https://img.shields.io/pypi/v/thermoprop)](https://pypi.org/project/thermoprop/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/thermoprop/)
[![License](https://img.shields.io/pypi/l/thermoprop)](https://github.com/saakethramoju/ThermoProp)

ThermoProp is a Python thermophysical-property and chemical-equilibrium library for engineering analysis, propulsion, thermodynamics, heat transfer, and fluid-system modeling.

It provides a unified API for:

* Real fluids
* Fluid mixtures
* Ideal gases
* Ideal-gas mixtures
* Rocket propellants
* Combustion-product gases
* CEA-style reactant mixtures
* Chemical-equilibrium calculations
* Isotropic engineering materials

ThermoProp integrates several thermodynamic and engineering-property sources behind a consistent interface:

* CoolProp
* PYroMat
* RocketProps
* NASA CEA / CEAM thermochemical and transport databases
* Built-in species and material databases

---

## Installation

```bash
pip install thermoprop
```

ThermoProp requires Python 3.11 or newer.

## 1.0.1 release focus

ThermoProp 1.0.1 is a cleanup, API-stability, and low-risk performance release. It keeps the 1.0.0 solver equations and property-model behavior intact while reducing repeated helper code, adding missing wrapper introspection support to `Equilibrium`, clarifying internal CEA-equilibrium documentation, adding snake_case `Equilibrium.combustion_gas` aliases, moving large registry data into packaged JSON files, vectorizing CEA thermo/transport helper paths, preserving class-based top-level imports, and cleaning the release source tree.

Use these wrapper-level inspection helpers consistently across the public API:

```python
from thermoprop import Fluid, IdealGas, CombustionGas, Propellant, Material, Equilibrium

print(Fluid.supported_properties())
print(CombustionGas.supported_flash_inputs())
print(Equilibrium.supports_property("temperature"))
```

---

## Why ThermoProp?

Engineering projects often need several different property libraries at the same time.

For example, one propulsion or thermal-fluid model may need:

* CoolProp for real-fluid thermodynamics
* PYroMat for ideal-gas thermodynamics
* RocketProps for liquid propellant properties
* NASA CEA data for combustion products
* Material property curves for structural or thermal analysis
* Chemical equilibrium for combustion calculations

Each backend has its own syntax, naming conventions, supported properties, and reference states.

ThermoProp provides a common interface.

Instead of writing backend-specific calls such as:

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

The same general style is used across real fluids, ideal gases, propellants, combustion gases, equilibrium products, and materials.

---

## Wrapper Selection Guide

| Need                                  | Use                |
| ------------------------------------- | ------------------ |
| Real-fluid thermodynamics             | `Fluid`            |
| CoolProp fluid mixtures               | `Fluid`            |
| Ideal-gas thermodynamics              | `IdealGas`         |
| Ideal-gas mixtures                    | `IdealGas`         |
| Rocket propellant properties          | `Propellant`       |
| CEA gas species properties            | `CombustionGas`    |
| Combustion-product gas mixtures       | `CombustionGas`    |
| Reactant mixture setup                | `Reactants`        |
| Chemical equilibrium                  | `Equilibrium`      |
| Frozen combustion-gas properties      | `CombustionGas`    |
| Equilibrium combustion-gas properties | `Equilibrium`      |
| Engineering material properties       | `Material`         |
| Direct NASA CEA data access           | `CEA`              |
| Species discovery and backend mapping | `SpeciesDatabase`  |
| Material discovery and aliases        | `MaterialDatabase` |

---

## Main Imports

```python
from thermoprop import Fluid
from thermoprop import IdealGas
from thermoprop import Propellant
from thermoprop import CombustionGas
from thermoprop import Reactants
from thermoprop import Equilibrium
from thermoprop import Material

from thermoprop import CEA
from thermoprop import SpeciesDatabase
from thermoprop import MaterialDatabase
```

ThermoProp also exposes convenience functions. The `list_*` names are recommended because they are harder to accidentally shadow with local variables; the shorter `species()` and `materials()` names remain available for compatibility.

```python
from thermoprop import list_species
from thermoprop import supported_species
from thermoprop import species_aliases
from thermoprop import add_species_alias

from thermoprop import list_materials
from thermoprop import supported_materials
from thermoprop import material_aliases
from thermoprop import add_material_alias

print(list_species()[:10])
print(list_materials())
```

---

# Quick Start

## Real Fluid

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

---

## Ideal Gas

```python
from thermoprop import IdealGas

nitrogen = IdealGas(
    "gn2",
    pressure=101325,
    temperature=300,
)

print(nitrogen.density)
print(nitrogen.specific_heat_cp)
print(nitrogen.specific_heat_ratio)
print(nitrogen.speed_of_sound)
```

---

## Liquid Propellant

```python
from thermoprop import Propellant

rp1 = Propellant(
    "rp1",
    temperature=293.15,
    pressure=2.0e6,
)

print(rp1.density)
print(rp1.dynamic_viscosity)
print(rp1.enthalpy)
```

---

## Combustion Gas

```python
from thermoprop import CombustionGas

gas = CombustionGas(
    {
        "CO2": 0.25,
        "H2O": 0.45,
        "CO": 0.10,
        "H2": 0.05,
        "N2": 0.15,
    },
    basis="mole",
    pressure=2.0e6,
    temperature=3200.0,
)

print(gas.density)
print(gas.specific_heat_cp)
print(gas.specific_heat_ratio)
print(gas.dynamic_viscosity)
print(gas.thermal_conductivity)
```

---

## Reactants and Equilibrium

```python
from thermoprop import Propellant
from thermoprop import Reactants
from thermoprop import Equilibrium

fuel = Propellant(
    "RP-1",
    temperature=298.15,
    pressure=2.0e6,
)

oxidizer = Propellant(
    "LOX",
    temperature=90.17,
    pressure=2.0e6,
)

reactants = Reactants(
    fuels=[fuel],
    oxidizers=[oxidizer],
    mixture_ratio=2.5,
)

eq = Equilibrium(
    reactants,
    mode="hp",
    pressure=2.0e6,
)

print(eq.temperature)
print(eq.mole_fractions)
print(eq.specific_heat_cp)
print(eq.speed_of_sound)
```

---

## Engineering Material

```python
from thermoprop import Material

inconel = Material(
    "in718",
    temperature=300,
)

print(inconel.density)
print(inconel.yield_strength)
print(inconel.thermal_conductivity)
```

---

# Core Wrappers

## `Fluid`

`Fluid` is a CoolProp-backed real-fluid wrapper.

It supports pure fluids and mixtures using a Fluid-like API with SI units.

### Constructor

```python
Fluid(
    fluid,
    basis="mass",
    pressure=None,
    enthalpy=None,
    temperature=None,
    quality=None,
    density=None,
    internal_energy=None,
)
```

`fluid` may be either a string or a dictionary of fractions.

```python
Fluid("water", pressure=101325, temperature=300)
```

```python
Fluid(
    {"nitrogen": 0.79, "oxygen": 0.21},
    basis="mole",
    pressure=101325,
    temperature=300,
)
```

### Supported flash inputs

`Fluid` requires exactly two thermodynamic state inputs.

Supported pairs include:

* `pressure` + `temperature`
* `pressure` + `enthalpy`
* `pressure` + `quality`
* `temperature` + `quality`
* `density` + `internal_energy`
* `pressure` + `density`
* `pressure` + `internal_energy`
* `temperature` + `density`
* `density` + `enthalpy`
* `temperature` + `enthalpy`

You can inspect supported inputs programmatically:

```python
from thermoprop import Fluid

print(Fluid.supported_flash_inputs())
```

### Common properties

```python
print(fluid.pressure)
print(fluid.temperature)
print(fluid.density)
print(fluid.specific_volume)

print(fluid.enthalpy)
print(fluid.internal_energy)
print(fluid.entropy)

print(fluid.specific_heat_cp)
print(fluid.specific_heat_cv)
print(fluid.specific_heat_ratio)

print(fluid.dynamic_viscosity)
print(fluid.kinematic_viscosity)
print(fluid.thermal_conductivity)
print(fluid.prandtl)

print(fluid.speed_of_sound)
print(fluid.phase)
print(fluid.quality)
```

### Advanced properties

When supported by CoolProp, `Fluid` also exposes:

* Thermal expansion coefficient
* Isothermal compressibility
* Helmholtz energy
* Gibbs energy
* Fundamental derivative of gas dynamics
* Fugacity coefficients
* Critical properties
* Saturation properties
* Triple-point properties
* First partial derivatives

Example:

```python
water = Fluid("water", pressure=101325, temperature=300)

print(water.thermal_expansion_coefficient)
print(water.isothermal_compressibility)
print(water.gibbs_energy)
print(water.dhdT_const_p)
```

### Updating state

```python
water.pressure_temperature = (2.0e5, 350.0)
water.pressure_enthalpy = (2.0e5, 1.5e6)
water.pressure_quality = (101325, 0.5)
```

### Mixture composition updates

```python
air = Fluid(
    {"nitrogen": 0.79, "oxygen": 0.21},
    basis="mole",
    pressure=101325,
    temperature=300,
)

air.mole_fractions = [0.78, 0.22]
```

Fractions must be finite, nonnegative, and sum to 1.

---

## `IdealGas`

`IdealGas` is a PYroMat-backed ideal-gas wrapper with additional transport-property support.

Thermodynamic properties are evaluated using PYroMat. Transport properties use NASA CEA / CEAM transport data when available, with fallback correlations where implemented.

### Constructor

```python
IdealGas(
    fluid,
    basis="mass",
    pressure=None,
    enthalpy=None,
    temperature=None,
    internal_energy=None,
    density=None,
    quality=None,
)
```

`fluid` may be either a string or a mixture dictionary.

```python
from thermoprop import IdealGas

air = IdealGas(
    "air",
    pressure=101325,
    temperature=300,
)
```

```python
gas = IdealGas(
    {"nitrogen": 0.79, "oxygen": 0.21},
    basis="mole",
    pressure=101325,
    temperature=300,
)
```

### Supported flash inputs

`IdealGas` supports thermal states from:

* `temperature`
* `enthalpy`
* `internal_energy`

It also supports density closure using:

* `pressure` + `density`
* `pressure` + `temperature`
* `pressure` + `enthalpy`
* `pressure` + `internal_energy`
* `density` + `temperature`
* `density` + `enthalpy`
* `density` + `internal_energy`

Example:

```python
gas = IdealGas(
    "nitrogen",
    pressure=101325,
    density=1.14,
)
```

### Pressure-dependent properties

Pressure is optional for some ideal-gas properties.

Pressure is required for properties such as:

* Density
* Entropy
* Gibbs energy
* Helmholtz energy
* Partial pressures
* Some partial derivatives

### Common properties

```python
print(gas.temperature)
print(gas.pressure)
print(gas.density)

print(gas.enthalpy)
print(gas.internal_energy)
print(gas.entropy)

print(gas.specific_heat_cp)
print(gas.specific_heat_cv)
print(gas.specific_heat_ratio)

print(gas.dynamic_viscosity)
print(gas.thermal_conductivity)
print(gas.prandtl)

print(gas.speed_of_sound)
```

### State updates

```python
gas.temperature = 500
gas.pressure = 2.0e5

gas.pressure_temperature = (101325, 300)
gas.pressure_enthalpy = (101325, gas.enthalpy)
gas.pressure_internal_energy = (101325, gas.internal_energy)
gas.pressure_density = (101325, 1.2)
```

Backward-compatible aliases are also available:

```python
gas.TP = (300, 101325)
gas.HP = (gas.enthalpy, 101325)
```

### Mixture composition updates

```python
gas = IdealGas(
    {"nitrogen": 0.79, "oxygen": 0.21},
    basis="mole",
    pressure=101325,
    temperature=300,
)

gas.mole_fractions = [0.80, 0.20]
gas.mass_fractions = list(gas.mass_fractions.values())
```

Composition changes re-flash the gas using the last supplied state inputs.

---

## `CombustionGas`

`CombustionGas` is a NASA CEA / CEAM ideal-gas wrapper for gas-phase CEA species and mixtures.

It evaluates thermodynamic properties using NASA-9 CEA polynomials and transport properties using CEA / CEAM transport data when available.

CEA-style fallback estimates are used when explicit transport data are unavailable.

### Constructor

```python
CombustionGas(
    fluid,
    basis="mass",
    pressure=None,
    enthalpy=None,
    temperature=None,
    internal_energy=None,
    density=None,
    quality=None,
)
```

`fluid` may be a pure gas species:

```python
from thermoprop import CombustionGas

water_vapor = CombustionGas(
    "H2O",
    pressure=101325,
    temperature=1000,
)
```

or a gas mixture:

```python
gas = CombustionGas(
    {
        "CO2": 0.30,
        "H2O": 0.50,
        "CO": 0.10,
        "H2": 0.10,
    },
    basis="mole",
    pressure=2.0e6,
    temperature=3000,
)
```

### Supported flash inputs

`CombustionGas` supports the same style of ideal-gas state inputs as `IdealGas`:

* `temperature`
* `enthalpy`
* `internal_energy`
* `pressure` + `density`
* `pressure` + `temperature`
* `pressure` + `enthalpy`
* `pressure` + `internal_energy`
* `density` + `temperature`
* `density` + `enthalpy`
* `density` + `internal_energy`

Example:

```python
gas = CombustionGas(
    {"CO2": 0.4, "H2O": 0.6},
    basis="mole",
    pressure=2.0e6,
    enthalpy=-8.0e6,
)
```

### Common properties

```python
print(gas.temperature)
print(gas.pressure)
print(gas.density)

print(gas.enthalpy)
print(gas.internal_energy)
print(gas.entropy)

print(gas.specific_heat_cp)
print(gas.specific_heat_cv)
print(gas.specific_heat_ratio)

print(gas.dynamic_viscosity)
print(gas.kinematic_viscosity)
print(gas.thermal_conductivity)
print(gas.prandtl)

print(gas.speed_of_sound)
print(gas.mole_fractions)
print(gas.mass_fractions)
```

### Composition updates

```python
gas = CombustionGas(
    {"CO2": 0.4, "H2O": 0.6},
    basis="mole",
    pressure=2.0e6,
    temperature=3000,
)

gas.mole_fractions = [0.35, 0.65]
```

Fractions must be finite, nonnegative, and sum to 1.

### Estimated transport data

Some CEA species may not have explicit transport coefficients.

You can inspect which species used estimated transport data:

```python
print(gas.estimated_transport_species)
```

---

## `Propellant`

`Propellant` is a combined RocketProps / NASA CEA wrapper for rocket propellants and CEA reactants.

It resolves names through `SpeciesDatabase`.

Depending on the propellant and phase, it may use:

* RocketProps liquid-property correlations
* NASA CEA reference data
* NASA CEA condensed-species thermodynamics
* NASA CEA gas-species thermodynamics

### Constructor

```python
Propellant(
    propellant,
    temperature,
    pressure=None,
)
```

Example:

```python
from thermoprop import Propellant

lox = Propellant(
    "LOX",
    temperature=90.17,
    pressure=2.0e6,
)

print(lox.density)
print(lox.enthalpy)
print(lox.elemental_composition)
```

### Supported state inputs

`Propellant` supports:

* `temperature`
* `pressure` + `temperature`

```python
rp1 = Propellant("RP-1", temperature=298.15)
lox = Propellant("LOX", temperature=90.17, pressure=2.0e6)
```

### Common properties

Depending on backend availability, `Propellant` can expose:

* Density
* Specific volume
* Dynamic viscosity
* Kinematic viscosity
* Thermal conductivity
* Surface tension
* Vapor pressure
* Saturation temperature
* Heat of vaporization
* Critical pressure
* Critical temperature
* Critical density
* Enthalpy
* Internal energy
* Entropy
* Standard entropy
* Specific heat
* Molecular weight
* Gas constant
* Heat of formation
* Elemental composition
* CEA polynomial temperature range
* Backend source tracking

Example:

```python
rp1 = Propellant("RP-1", temperature=298.15, pressure=2.0e6)

print(rp1.density)
print(rp1.specific_heat_cp)
print(rp1.enthalpy)
print(rp1.heat_of_formation)
print(rp1.data_sources)
```

### Source tracking

`Propellant` can report which backend supplied an evaluated property:

```python
print(rp1.data_sources)
print(rp1.property_source("density"))
```

This is useful because some properties come from RocketProps while others come from NASA CEA.

### Phase behavior

For RocketProps-backed species, liquid states are generally evaluated using RocketProps correlations. When pressure falls below vapor pressure and a compatible CEA gas species is available, ThermoProp can use the gas-phase CEA species for thermodynamic data.

---

## `Reactants`

`Reactants` defines CEA-style reactant mixtures for equilibrium calculations.

Inputs are `Propellant` objects grouped into fuels and oxidizers.

Optional weights inside each group are treated as mass weights, similar to CEA weight-percent behavior.

### Constructor

```python
Reactants(
    fuels,
    oxidizers,
    mixture_ratio,
)
```

The total basis is:

```text
fuel mass = 1 kg
oxidizer mass = O/F kg
total mass = 1 + O/F kg
```

### Single-fuel / single-oxidizer example

```python
from thermoprop import Propellant
from thermoprop import Reactants

fuel = Propellant("RP-1", temperature=298.15, pressure=2.0e6)
oxidizer = Propellant("LOX", temperature=90.17, pressure=2.0e6)

reactants = Reactants(
    fuels=[fuel],
    oxidizers=[oxidizer],
    mixture_ratio=2.5,
)

print(reactants.mass_fractions)
print(reactants.mole_fractions)
print(reactants.element_moles_per_kg)
print(reactants.reactant_enthalpy)
```

### Weighted multi-propellant example

```python
from thermoprop import Propellant
from thermoprop import Reactants

fuel_a = Propellant("RP-1", temperature=298.15, pressure=2.0e6)
fuel_b = Propellant("Methane", temperature=111.0, pressure=2.0e6)
oxidizer = Propellant("LOX", temperature=90.17, pressure=2.0e6)

reactants = Reactants(
    fuels=[
        (fuel_a, 0.8),
        (fuel_b, 0.2),
    ],
    oxidizers=[oxidizer],
    mixture_ratio=2.7,
)
```

### Common properties

```python
print(reactants.fuel_mass)
print(reactants.oxidizer_mass)
print(reactants.oxidizer_to_fuel_ratio)
print(reactants.total_mass)

print(reactants.mass_fractions)
print(reactants.mole_fractions)

print(reactants.element_moles)
print(reactants.element_moles_per_kg)

print(reactants.reactant_enthalpy)
print(reactants.reactant_internal_energy)
```

---

## `Equilibrium`

`Equilibrium` performs CEA-style chemical-equilibrium calculations using Gibbs free-energy minimization and NASA CEA thermochemical data.

It can solve from:

* `Reactants`
* `CombustionGas`

### Constructor

```python
Equilibrium(
    reactants,
    mode="hp",
    temperature=None,
    pressure=None,
    guess_temperature=3500.0,
    candidates=None,
    include_all_valid_gases=True,
    verbose=False,
    element_tol=1e-8,
    enthalpy_tol=1e-3,
    correction_tol=1e-8,
    max_iterations=200,
    trace_moles=1e-300,
    min_temperature=200.0,
    max_temperature=20000.0,
    combustion_gas_trace=1e-8,
    combustion_gas_max_species=25,
    equilibrium_derivative_temperature_step=1.0,
)
```

### HP equilibrium

HP equilibrium uses reactant enthalpy and pressure.

```python
eq = Equilibrium(
    reactants,
    mode="hp",
    pressure=2.0e6,
    guess_temperature=3500.0,
)

print(eq.temperature)
print(eq.mole_fractions)
```

### TP equilibrium

TP equilibrium uses specified temperature and pressure.

```python
eq = Equilibrium(
    reactants,
    mode="tp",
    pressure=2.0e6,
    temperature=3500.0,
)

print(eq.mole_fractions)
```

### Equilibrium from an existing combustion gas

```python
from thermoprop import CombustionGas
from thermoprop import Equilibrium

gas = CombustionGas(
    {"CO2": 0.4, "H2O": 0.6},
    basis="mole",
    pressure=2.0e6,
    temperature=3000.0,
)

eq = Equilibrium(
    gas,
    mode="tp",
    pressure=2.0e6,
    temperature=3000.0,
)
```

### Common properties

```python
print(eq.success)
print(eq.message)
print(eq.iterations)

print(eq.temperature)
print(eq.pressure)
print(eq.density)

print(eq.mole_fractions)
print(eq.mass_fractions)
print(eq.normalized_mole_fractions)
print(eq.normalized_mass_fractions)

print(eq.enthalpy)
print(eq.internal_energy)
print(eq.entropy)

print(eq.specific_heat_cp)
print(eq.specific_heat_cp_frozen)
print(eq.specific_heat_cp_equilibrium)

print(eq.specific_heat_ratio)
print(eq.specific_heat_ratio_frozen)
print(eq.specific_heat_ratio_equilibrium)

print(eq.speed_of_sound)
print(eq.speed_of_sound_frozen)
print(eq.speed_of_sound_equilibrium)

print(eq.dynamic_viscosity)
print(eq.thermal_conductivity)
print(eq.prandtl)
```

### CombustionGas output

`Equilibrium` can generate a `CombustionGas` object from its equilibrium composition.

```python
gas = eq.combustion_gas

print(gas.specific_heat_cp)
print(gas.dynamic_viscosity)
print(gas.thermal_conductivity)
```

You can control trace-species filtering:

```python
composition = eq.combustion_gas_composition(
    trace=1e-8,
    max_species=25,
)
```

### Frozen vs equilibrium properties

`Equilibrium` distinguishes between frozen-composition and equilibrium-composition properties.

Examples:

```python
print(eq.specific_heat_cp_frozen)
print(eq.specific_heat_cp_equilibrium)

print(eq.specific_heat_ratio_frozen)
print(eq.specific_heat_ratio_equilibrium)

print(eq.speed_of_sound_frozen)
print(eq.speed_of_sound_equilibrium)

print(eq.conductivity_frozen)
print(eq.conductivity_reaction)
print(eq.conductivity_equilibrium)
```

This is useful for propulsion and nozzle calculations where frozen and shifting-equilibrium assumptions may give different results.

---

## `Material`

`Material` provides temperature-dependent isotropic engineering material properties from ThermoProp's built-in material database.

### Constructor

```python
Material(
    material,
    temperature=298.15,
    allow_extrapolation=True,
)
```

Example:

```python
from thermoprop import Material

copper = Material(
    "c101",
    temperature=300,
)

print(copper.density)
print(copper.thermal_conductivity)
```

### Supported properties

Depending on the material, available properties may include:

* Density
* Specific volume
* Yield strength
* Ultimate strength
* Tensile strength
* Elastic modulus
* Young's modulus
* Torsional modulus
* Shear modulus
* Poisson ratio
* Thermal conductivity
* Specific heat
* Coefficient of thermal expansion
* Thermal diffusivity
* Melting point
* Freezing temperature
* Electrical resistivity

### Property lookup

```python
mat = Material("in718", temperature=300)

print(mat.yield_strength)
print(mat.thermal_conductivity)

print(mat.get("yield_strength", temperature=900))
print(mat.units("yield_strength"))
print(mat.temperature_range("yield_strength"))
```

### Curve access

```python
T, y = mat.curve("yield_strength")
```

### State update

```python
mat.temperature = 900
print(mat.yield_strength)
```

or:

```python
mat.set_state(temperature=900)
```

### Available materials

```python
from thermoprop import materials

print(materials())
```

---

# Built-In Databases

## `SpeciesDatabase`

`SpeciesDatabase` is ThermoProp's unified species-name and backend-mapping database.

It maps ThermoProp species names to backend-specific names for:

* CoolProp
* PYroMat
* NASA CEA
* RocketProps

Use it indirectly through wrappers or directly for discovery.

```python
from thermoprop import species
from thermoprop import supported_species
from thermoprop import species_aliases

print(species())
print(supported_species("Fluid"))
print(supported_species("IdealGas"))
print(supported_species("Propellant"))
print(supported_species("CombustionGas"))
print(species_aliases())
```

### Runtime species aliases

```python
from thermoprop import add_species_alias
from thermoprop import Fluid

add_species_alias("my-water", "Water")

water = Fluid(
    "my-water",
    pressure=101325,
    temperature=300,
)
```

Convenience aliases are also available:

```python
from thermoprop import aliases
from thermoprop import add_alias

print(aliases())
add_alias("my-air", "Air")
```

---

## `MaterialDatabase`

`MaterialDatabase` stores material identity records, aliases, metadata, and temperature-dependent property curves.

Use it directly or through `Material`.

```python
from thermoprop import materials
from thermoprop import material_aliases
from thermoprop import add_material_alias

print(materials())
print(material_aliases())

add_material_alias("chamber-alloy", "Inconel 718")
```

---

## `CEA`

`CEA` is a package-level instance of `CEADatabase`.

It provides direct access to parsed NASA CEA / CEAM thermochemical and transport data.

### Discovery

```python
from thermoprop import CEA

print(CEA.names)
print(CEA.gas_species)
print(CEA.condensed_species)
print(CEA.reactant_names)
print(CEA.transport_names)
```

### Search

```python
print(CEA.find_species("H2O"))
print(CEA.find_transport_species("CO2"))
```

### Species data

```python
print(CEA.molecular_weight("CO2"))
print(CEA.molar_mass("CO2"))
print(CEA.elemental_composition("CO2"))
print(CEA.temperature_ranges("CO2"))
```

### Thermodynamic properties

```python
cp, h, s0 = CEA.thermo_molar("CO2", 3000.0)

print(cp)
print(h)
print(s0)
```

Mass-specific helpers are also available:

```python
print(CEA.cp_mass("CO2", 3000.0))
print(CEA.enthalpy_mass("CO2", 3000.0))
print(CEA.entropy_mass_standard("CO2", 3000.0))
```

### Transport properties

```python
print(CEA.viscosity("CO2", 1000.0))
print(CEA.conductivity("CO2", 1000.0))
```

### Mixture helpers

```python
x = [0.7, 0.3]
species_names = ["CO2", "H2O"]

w = CEA.mole_to_mass(species_names, x)
print(w)

x2 = CEA.mass_to_mole(species_names, w)
print(x2)
```

---

# Combustion Workflow

ThermoProp separates combustion setup, equilibrium solving, and product-property evaluation.

```text
Propellant -> Reactants -> Equilibrium -> CombustionGas
```

## Step 1: Define propellant states

```python
from thermoprop import Propellant

fuel = Propellant(
    "RP-1",
    temperature=298.15,
    pressure=2.0e6,
)

oxidizer = Propellant(
    "LOX",
    temperature=90.17,
    pressure=2.0e6,
)
```

## Step 2: Build reactants

```python
from thermoprop import Reactants

reactants = Reactants(
    fuels=[fuel],
    oxidizers=[oxidizer],
    mixture_ratio=2.5,
)
```

## Step 3: Solve equilibrium

```python
from thermoprop import Equilibrium

eq = Equilibrium(
    reactants,
    mode="hp",
    pressure=2.0e6,
)
```

## Step 4: Evaluate gas properties

```python
gas = eq.combustion_gas

print(eq.temperature)
print(eq.mole_fractions)
print(gas.specific_heat_cp)
print(gas.dynamic_viscosity)
```

---

# Property Discovery

Most wrappers expose property and flash-input discovery methods.

## Supported properties

```python
from thermoprop import Fluid
from thermoprop import IdealGas
from thermoprop import CombustionGas
from thermoprop import Propellant
from thermoprop import Material
from thermoprop import Equilibrium

print(Fluid.supported_properties())
print(IdealGas.supported_properties())
print(CombustionGas.supported_properties())
print(Propellant.supported_properties())
print(Material.supported_properties())
print(Equilibrium.supported_properties())
```

## Supported flash inputs

```python
print(Fluid.supported_flash_inputs())
print(IdealGas.supported_flash_inputs())
print(CombustionGas.supported_flash_inputs())
print(Propellant.supported_flash_inputs())
print(Material.supported_flash_inputs())
print(Equilibrium.supported_flash_inputs())
```

## Available species

```python
print(Fluid.get_available_fluids())
print(IdealGas.get_available_gases())
print(CombustionGas.get_available_species())
print(Propellant.get_available_propellants())
```

## Available materials

```python
print(Material.get_available_materials())
print(Material.get_available_properties())
```

---

# Thermodynamic Reference States

ThermoProp provides a unified API across multiple thermodynamic backends.

However, the underlying libraries and databases may use different thermodynamic reference states.

This affects absolute values of:

* Enthalpy
* Internal energy
* Entropy
* Gibbs energy
* Helmholtz energy

For example, a CoolProp `Fluid`, a PYroMat `IdealGas`, a NASA CEA `CombustionGas`, and a RocketProps / CEA `Propellant` may report different absolute enthalpy values even for physically similar states.

This is expected.

Within a single backend, property differences are generally meaningful:

* Delta enthalpy
* Delta internal energy
* Delta entropy
* Heat capacity
* Density
* Speed of sound
* Transport properties

When combining results from multiple wrappers, establish a consistent thermodynamic reference basis if absolute values are required.

---

# Units

ThermoProp's public API uses SI units.

Common units include:

| Quantity             | Unit                                |
| -------------------- | ----------------------------------- |
| Pressure             | Pa                                  |
| Temperature          | K                                   |
| Density              | kg/m³                               |
| Specific volume      | m³/kg                               |
| Enthalpy             | J/kg                                |
| Internal energy      | J/kg                                |
| Entropy              | J/kg-K                              |
| Specific heat        | J/kg-K                              |
| Dynamic viscosity    | Pa-s                                |
| Kinematic viscosity  | m²/s                                |
| Thermal conductivity | W/m-K                               |
| Surface tension      | N/m                                 |
| Molar mass           | kg/mol                              |
| Molecular weight     | kg/kmol, numerically equal to g/mol |
| Material strength    | Pa                                  |

---

# Limitations

## General

ThermoProp wraps and combines several independent property sources.

Property availability depends on:

* The selected wrapper
* The selected species or material
* Backend support
* Temperature range
* Pressure range
* Phase
* Availability of NASA CEA transport data

Use runtime introspection methods such as `supported_properties()`, `supported_flash_inputs()`, and `data_sources` where available.

---

## `Fluid`

`Fluid` is limited by CoolProp support.

Limitations include:

* Supported fluids are limited to CoolProp-compatible fluids.
* Mixture behavior follows CoolProp capabilities and limitations.
* Some advanced properties may be unavailable for some fluids or states.
* Two-phase and mixture flashes may be backend-limited.

---

## `IdealGas`

`IdealGas` assumes ideal-gas behavior.

Limitations include:

* Not intended for dense gases.
* Not intended for near-critical states.
* Not a real-fluid equation-of-state wrapper.
* Transport-property availability depends on CEA data or fallback correlations.
* Absolute thermodynamic reference states follow PYroMat conventions.

---

## `CombustionGas`

`CombustionGas` uses gas-phase NASA CEA species.

Limitations include:

* Only gas-phase CEA product species are accepted.
* Condensed species and CEA reactant cards are not valid `CombustionGas` species.
* It does not solve chemical equilibrium by itself.
* Use `Equilibrium` when product composition should be determined from reactants.
* Transport data may be estimated for species without explicit CEAM transport fits.
* Temperature must remain within the common valid NASA polynomial range for the selected species.

---

## `Propellant`

`Propellant` combines RocketProps and NASA CEA data.

Limitations include:

* Property availability depends on backend support for the selected species.
* RocketProps-backed properties are primarily liquid engineering correlations.
* CEA-backed properties depend on available CEA species, reactant, or condensed-phase entries.
* Not every propellant has all thermodynamic, transport, reference, or critical properties.
* Mixture propellants are only supported where the underlying backend supports them.

---

## `Reactants`

`Reactants` is a reactant-mixture definition object.

Limitations include:

* Inputs must be `Propellant` objects.
* Fuels and oxidizers are grouped separately.
* Mixture ratio is oxidizer-to-fuel mass ratio.
* Reactant enthalpy and elemental composition require valid propellant thermochemical data.

---

## `Equilibrium`

`Equilibrium` assumes chemical equilibrium.

Limitations include:

* Does not model finite-rate chemistry.
* Does not model transient reaction kinetics.
* Does not model mixing, diffusion, ignition delay, or combustion instability.
* Results depend on the selected candidate product species.
* HP equilibrium uses the reactant enthalpy basis supplied by the input reactants.
* Condensed product phases are not currently treated as full equilibrium products in the gas-product solver.

---

## `Material`

`Material` provides temperature-dependent isotropic property curves.

Limitations include:

* No anisotropic material support.
* No composite material support.
* No pressure dependence.
* No stress-strain curves.
* No creep modeling.
* No fatigue data.
* No fracture-mechanics data.
* No plasticity model.
* Property values are interpolated from stored curves or constants.

---

# Documentation

Full documentation:

https://saakethramoju.github.io/softwares/thermoprop/

Source code:

https://github.com/saakethramoju/ThermoProp

PyPI:

https://pypi.org/project/thermoprop/

---

# Acknowledgments

ThermoProp incorporates, depends on, or adapts data from:

* CoolProp
* PYroMat
* RocketProps
* NASA CEA / CEAM
* MatProtLib
* NumPy
* SciPy

ThermoProp's engineering material database was adapted from material property data compiled and distributed through the MatProtLib project.

Special thanks to Tyson Tran and the MatProtLib project for making engineering material datasets publicly available.

The author also gratefully acknowledges the NASA Glenn Research Center and the NASA CEA development team for making thermochemical and transport datasets publicly available.

---

# License

ThermoProp is released under the GNU General Public License v3.0.

See:

* `LICENSE`
* `THIRD_PARTY_LICENSES.md`
