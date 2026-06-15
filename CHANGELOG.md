# Changelog

## 1.0.1

### Improved

* Added shared public-API helpers for wrapper property introspection.
* Added shared fraction-vector validation for `Fluid`, `IdealGas`, and `CombustionGas`.
* Added missing `Equilibrium.supported_properties()`, `Equilibrium.show_supported_properties()`, and `Equilibrium.supports_property()` support through the shared API mixin.
* Added snake_case `Equilibrium.combustion_gas` and `Equilibrium.combustion_gas_composition()` aliases while preserving the existing `CombustionGas` aliases.
* Added lightweight docstrings for internal CEA equilibrium option/result dataclasses.
* Clarified the `Equilibrium.py` module layout documentation.
* Cleaned the release source tree by excluding local virtual environments, cached bytecode, previous distributions, Git internals, and the stale development `uv.lock`.

### Fixed

* Fixed the local smoke-test import path to use `from thermoprop import *` instead of `from src.thermoprop import *`.
* Relaxed overly strict dependency lower bounds to improve installability while preserving the features used by ThermoProp.

### Notes

* No equilibrium solver equations or thermodynamic/transport property formulas were intentionally changed in this cleanup release.

## 1.0.0

### Added

#### NASA CEA / CEAM Integration

* Added `CEADatabase`, a native NASA CEA / CEAM thermodynamic and transport database interface.

* Added support for:

  * NASA-9 thermodynamic polynomials
  * NASA CEA transport-property correlations
  * Species thermodynamic lookup
  * Species transport-property lookup
  * Species elemental composition lookup
  * Species molecular-weight lookup
  * Species discovery and inspection utilities
  * Strict NASA CEA species naming and resolution

* Added support for:

  * Gas-phase product species
  * Condensed-phase species
  * Predefined CEA reactants
  * Binary transport interaction coefficients
  * Mixture molecular-weight calculations
  * Mixture composition conversions

---

#### Combustion Products

* Added the `CombustionGas` wrapper for fixed-composition combustion-product gas mixtures.

* Added support for:

  * Pure combustion-product species
  * Multi-species combustion-gas mixtures
  * Mass-fraction compositions
  * Mole-fraction compositions
  * NASA CEA thermodynamic property evaluation
  * NASA CEA transport-property evaluation
  * CEA mixture transport-property mixing
  * Estimated transport-property fallbacks when explicit transport data are unavailable

* Added support for:

  * Density
  * Enthalpy
  * Internal energy
  * Entropy
  * Gibbs free energy
  * Helmholtz free energy
  * Specific heats
  * Specific heat ratio
  * Speed of sound
  * Dynamic viscosity
  * Thermal conductivity
  * Prandtl number

---

#### Chemical Equilibrium

* Added `Reactants` for defining reactant mixtures independently of equilibrium calculations.

* Added support for:

  * Reactant mass-fraction mixtures
  * Reactant mole-fraction mixtures
  * Reactant elemental composition tracking
  * Reactant thermodynamic reference-state evaluation
  * Reactant heat-of-formation calculations
  * Mixture molecular-weight calculations

* Added `Equilibrium`, a native Gibbs free-energy minimization equilibrium solver.

* Added support for:

  * NASA CEA species databases
  * Arbitrary reactant mixtures
  * Element conservation constraints
  * Equilibrium-composition prediction
  * Equilibrium thermodynamic properties
  * Frozen-composition property evaluation
  * Equilibrium combustion-gas generation

---

#### Species and Material Databases

* Added `SpeciesDatabase`, a unified immutable species database covering:

  * CoolProp fluids
  * PYroMat species
  * NASA CEA species
  * NASA CEA reactants
  * RocketProps propellants

* Added `MaterialDatabase`, a unified engineering-material database containing:

  * Material property data
  * Material metadata
  * Material lookup utilities

* Added package-level discovery utilities:

  * `species()`
  * `supported_species()`
  * `materials()`
  * `supported_materials()`

---

### Improved

#### IdealGas

* Reworked transport-property support to use NASA CEA transport data when available.

* Retained Sutherland-law viscosity as a fallback when transport data are unavailable.

* Added support for:

  * Thermal conductivity
  * Prandtl number
  * Improved mixture viscosity calculations
  * Improved mixture transport-property calculations

* Improved consistency with NASA CEA thermodynamic and transport-property evaluations.

---

#### Propellant

* Expanded `Propellant` beyond RocketProps-only fluids.

* Added support for NASA CEA species and reactants.

* Added automatic backend selection between RocketProps and NASA CEA.

* Added support for:

  * Heat of formation
  * Standard entropy
  * Enthalpy
  * Internal energy
  * Molecular-weight evaluation
  * Elemental composition lookup
  * Reference-state thermodynamic data

* Improved support for gaseous propellants and CEA reactant definitions.

---

#### Architecture

* Removed legacy registry architecture:

  * `FluidRegistry`
  * `MaterialRegistry`
  * `CombustionRegistry`

* Migrated all wrappers to database-driven name resolution.

* Standardized species management through `SpeciesDatabase`.

* Standardized material management through `MaterialDatabase`.

* Moved CEA databases into the ThermoProp package structure for reliable installation and distribution.

* Added validation for invalid mixture compositions, including negative mass and mole fractions.

* Added internal caching of material-property curves.

* Added internal caching of CEA species-discovery and lookup operations.

* Improved property-evaluation performance throughout ThermoProp.

* Improved consistency between:

  * `Fluid`
  * `IdealGas`
  * `CombustionGas`
  * `Propellant`
  * `Material`

---

### Documentation

* Added documentation for:

  * CEADatabase
  * CombustionGas
  * Reactants
  * Equilibrium
  * SpeciesDatabase
  * MaterialDatabase

* Added examples covering:

  * Combustion-gas evaluation
  * Chemical-equilibrium calculations
  * Reactant-mixture construction
  * Species discovery
  * Material-property lookup
  * NASA CEA transport-property usage

* Updated README and package documentation to reflect support for:

  * Real fluids
  * Ideal gases
  * Rocket propellants
  * Combustion gases
  * Chemical equilibrium
  * Engineering materials

---

### Acknowledgments

* ThermoProp's engineering material database was adapted from material property data compiled and distributed through the MatProtLib project.

* Special thanks to Tyson Tran for making these engineering material datasets publicly available.

* NASA CEA thermodynamic and transport datasets were adapted from the NASA CEA / CEAM databases.

## 0.3.3

### Added

* Added advanced thermodynamic property support to `Fluid`, including:

  * Thermal expansion coefficient
  * Isothermal compressibility
  * Helmholtz free energy
  * Gibbs free energy
  * Joule-Thomson coefficient

* Added thermodynamic partial-derivative support to `Fluid` through:

  * `partial_derivative()`
  * `dhdT_const_p`
  * `dhdP_const_t`
  * `drhodT_const_p`
  * `drhodP_const_t`
  * `dTdP_const_h`

* Added advanced thermodynamic property support to `IdealGas`, including:

  * Thermal expansion coefficient
  * Isothermal compressibility
  * Helmholtz free energy
  * Joule-Thomson coefficient

* Added selected analytic thermodynamic partial derivatives to `IdealGas`.

* Added thermal diffusivity support to `Material`.

* Added `thermal_expansion_coefficient` alias support to `Material`.

### Improved

* Expanded CoolProp-backed property coverage through additional thermodynamic derivative access.
* Improved support for advanced thermodynamic analysis and equation-of-state diagnostics.
* Improved API consistency between `Fluid`, `IdealGas`, `Propellant`, and `Material`.
* Standardized advanced thermodynamic property naming across ThermoProp wrappers.
* Improved discoverability of backend-supported thermodynamic properties.

### Documentation

* Added documentation for advanced thermodynamic properties.
* Added thermodynamic partial-derivative examples.
* Added Joule-Thomson coefficient examples.
* Added thermal expansion coefficient documentation.
* Added thermal diffusivity documentation for `Material`.
* Updated README feature summaries for `Fluid`, `IdealGas`, and `Material`.
* Expanded wrapper capability documentation and property reference examples.

## 0.3.2

### Added

* Added flash-input introspection utilities across ThermoProp wrappers:

  * `supported_flash_pairs()`
  * `available_flash_pairs()`
  * `supported_flash_inputs()`
  * `available_flash_inputs()`

* Added flash-input validation support to `IdealGas`.

* Added supported flash-input discovery utilities to:

  * `Fluid`
  * `IdealGas`
  * `Propellant`

* Added approximate ideal-gas Prandtl number support to `IdealGas`.

* Added approximate ideal-gas thermal conductivity support to `IdealGas`.

* Added Wilke mixture-viscosity support for ideal-gas mixtures.

* Added thermal conductivity alias support through `thermal_conductivity`.

### Improved

* Improved API consistency across `Fluid`, `IdealGas`, `Propellant`, and `Material`.
* Standardized wrapper introspection utilities across ThermoProp backends.
* Improved validation of user-specified state inputs for `IdealGas`.
* Improved discoverability of supported flash-input combinations and wrapper capabilities.
* Extended `IdealGas` transport-property support with approximate Prandtl number estimation.
* Extended `IdealGas` transport-property support with approximate thermal conductivity estimation based on viscosity, specific heat, and Prandtl number.
* Improved ideal-gas mixture transport-property calculations through Wilke viscosity mixing.
* Expanded Sutherland-law viscosity support to ideal-gas mixture calculations.

### Documentation

* Added flash-input introspection examples.
* Added supported state-input discovery examples.
* Added wrapper capability inspection examples.
* Updated API reference documentation for wrapper introspection utilities.
* Updated IdealGas documentation to describe approximate Prandtl number, viscosity, and thermal conductivity support.
* Updated IdealGas documentation to describe transport-property limitations and supported gases.

## 0.3.1

### Added

* Added property introspection utilities across ThermoProp wrappers:

  * `supported_properties()`
  * `show_supported_properties()`
  * `supports_property()`
* Added property introspection support to:

  * `Fluid`
  * `IdealGas`
  * `Propellant`
  * `Material`

### Improved

* Improved API consistency across ThermoProp wrappers.
* Standardized property discovery and wrapper introspection utilities.
* Improved discoverability of supported properties without requiring documentation lookup.

### Documentation

* Added property introspection examples.
* Added wrapper capability discovery examples.
* Updated API reference documentation for `Fluid`, `IdealGas`, `Propellant`, and `Material`.

## 0.3.0

### Added

* Added the `Material` wrapper for isotropic engineering material properties.
* Added `MaterialRegistry` for material name management and alias support.
* Added support for user-defined material aliases through `MaterialRegistry`.
* Added built-in temperature-dependent engineering material property database.
* Added support for:

  * Aluminum 6061
  * Aluminum 7075
  * Copper C101
  * Copper C11000
  * Copper C17200
  * GRCop-42
  * GRCop-84
  * 1018 Carbon Steel
  * 1045 Carbon Steel
  * 3140 Low-Alloy Steel
  * 4140 Steel
  * Stainless Steel 303
  * Stainless Steel 304
  * Stainless Steel 316
  * A286 Steel
  * Inconel 625
  * Inconel 718
  * Graphite
* Added temperature-dependent lookup support for:

  * Yield strength
  * Ultimate strength
  * Elastic modulus
  * Torsional modulus
  * Density
  * Poisson ratio
  * Thermal conductivity
  * Specific heat
  * Coefficient of thermal expansion
  * Melting point
  * Electrical resistivity
* Added material property alias support.
* Added material curve inspection and property availability utilities.

### Improved

* Extended ThermoProp beyond thermodynamic properties to include engineering material properties.
* Improved API consistency across `Fluid`, `IdealGas`, `Propellant`, and `Material`.
* Unified registry-based name handling through `FluidRegistry` and `MaterialRegistry`.
* Improved package organization and material property lookup infrastructure.

### Documentation

* Added Material documentation and usage examples.
* Added MaterialRegistry documentation and examples.
* Added material property lookup examples.
* Added supported materials documentation.
* Added acknowledgments for the MatProtLib project.
* Expanded third-party licensing documentation to include MatProtLib.
* Updated package description to reflect support for engineering materials.

### Acknowledgments

* ThermoProp's engineering material property database was adapted from material property data compiled and distributed through the MatProtLib project.
* Special thanks to Tyson Tran for making these engineering material datasets publicly available.

## 0.2.1

### Added

* Added user-configurable alias management through `FluidRegistry`.
* Added support for custom fluid aliases.
* Added support for custom propellant aliases.

### Improved

* Unified backend access through `FluidRegistry`.
* Improved API consistency between `Fluid`, `IdealGas`, and `Propellant`.
* Simplified wrapper name handling by centralizing backend mappings in `FluidRegistry`.
* Expanded third-party licensing documentation.

### Documentation

* Added FluidRegistry documentation and examples.
* Added custom alias management examples.
* Added backend lookup and registry inspection examples.
* Updated README examples and package documentation.

## 0.2.0

### Added

* Added RocketProps integration.
* Added the `Propellant` wrapper for liquid rocket propellant properties.
* Added support for liquid rocket propellant aliases through `FluidRegistry`.
* Added common propellant lookup support, including LOX, RP-1, Methane, Hydrogen, MMH, UDMH, N2O4, MON blends, Aerozine-50, and others.
* Added RocketProps dependency to ThermoProp.
* Added backend identification properties to ThermoProp wrappers.

### Documentation

* Updated README to include RocketProps and Propellant usage examples.
* Updated package description to reflect support for real fluids, ideal gases, and liquid rocket propellants.

## 0.1.6

### Documentation

* Added documentation describing thermodynamic reference-state differences between ThermoProp backends.
* Clarified that absolute enthalpy, internal energy, and entropy values may differ between wrappers due to backend-specific reference conventions.
* Added guidance for users combining results from multiple ThermoProp wrappers.

## 0.1.5

* Documented how to update Fluid and IdealGas state properties.
* Clarified that IdealGas can be initialized from temperature alone.
* Documented the current IdealGas viscosity limitations.

## 0.1.4

* Improved README with API overview and examples.
* Added pressure-enthalpy example.
* Added common property example.
* Added package keywords and classifiers.
* Added changelog.

## 0.1.3

* Added automated GitHub-to-PyPI publishing workflow.
* Improved README examples.
* Updated PyPI package page documentation.

## 0.1.2

* Relaxed CoolProp dependency to support CoolProp >= 6.8.0.
* Added licensing information.

## 0.1.1

* Added README.
* Added project URLs.
* Added license metadata.

## 0.1.0

* Initial release.
