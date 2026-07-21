# Changelog

## 2.1.0

### Added

- Added `Rocket`, a thin CEA-style theoretical rocket-performance wrapper around `Reactants`, `Equilibrium`, and `CombustionGas`.
- Added infinite-area chamber and automatic finite-area combustor selection through `contraction_ratio`.
- Added absolute-pressure, subsonic-area-ratio, and supersonic-area-ratio nozzle stations.
- Added equilibrium expansion, frozen-at-chamber IAC expansion, and frozen-at-throat expansion.
- Added numeric `frozen_at=A/At` support for freezing composition at any supersonic nozzle area ratio, including automatic freeze-station solving and `rocket.freeze_station` access.
- Added direct chamber, injector, throat, exit, grouped-station, pressure lookup, and area-ratio lookup interfaces.
- Added characteristic velocity, mass flux, Mach number, area ratio, pressure ratio, thrust coefficient, matched-pressure specific impulse, and vacuum specific impulse outputs.
- Added a complete CEA-style text report with reactants, station properties, performance, and composition tables.

## 2.0.2

- 'prantdl' added to 'Propellant'.

## 2.0.1

### Fixed

- Fixed `Propellant.minimum_pressure` and `maximum_pressure` so RocketProps saturation-table limits are no longer reported as general compressed-liquid pressure limits.
- Added `saturation_pressure_range`, `minimum_saturation_pressure`, and `maximum_saturation_pressure` to `Propellant`.
- Added compressed-liquid pressure correction to the direct CEA enthalpy path when a RocketProps liquid model is available.
- Fixed `Fluid.minimum_pressure` to use CoolProp minimum-pressure metadata instead of substituting triple-point pressure.

## 2.0.0

### Added

* Added detailed docstrings across the public API so users can rely on `help()`, IDE tooltips, and future generated API documentation before a separate documentation site exists.
* Added detailed documentation for package-level helpers, wrapper properties, flash-input discovery methods, CEA database accessors, material database helpers, equilibrium result accessors, and public exception classes.
* Added a README section explaining the 2.0.0 documentation-first publishing status and how the README, changelog, third-party license file, publishing checklist, and docstrings fit together as the initial documentation set.
* Added `PUBLISHING.md` with release validation, wheel-inspection, and upload commands for 2.0.0.

### Changed

* Bumped the package version to `2.0.0` in `pyproject.toml` and `thermoprop.__version__`.
* Updated the package description to emphasize that ThermoProp is now prepared as a documented public package.
* Expanded short property and helper docstrings into multi-paragraph descriptions with units, state-update behavior, backend limitations, and error behavior where applicable.

### Packaging

* Reviewed the publish-facing metadata in `pyproject.toml`, including name, version, description, Python requirement, dependencies, classifiers, URLs, package data inclusion, build backend, and third-party credit metadata.
* Added `license-files = ["LICENSE"]` so built wheels include the GPL license file under the dist-info license directory.
* Expanded the `uv_build` source distribution include list so `CHANGELOG.md`, `LICENSE`, `THIRD_PARTY_LICENSES.md`, `PUBLISHING.md`, examples, and packaged CEA/material data are included in the source archive.

### Notes

* No equilibrium solver equations, CEA thermodynamic-data evaluation, CEA transport equations, CoolProp/PYroMat/RocketProps backend behavior, or material interpolation behavior were intentionally changed for this release.
* ThermoProp still has no separate official documentation site generated from the repository, so the README and docstrings are intentionally verbose.

## 1.0.2

### Added

* Added native `mode="sp"` to `Equilibrium` for assigned entropy/pressure equilibrium states. SP now solves temperature and equilibrium composition together in a CEA-style reduced Gibbs matrix, shares the TP/HP condensed-phase active-set logic, and keeps the older TP entropy-root implementation as an internal fallback for difficult phase-boundary cases.
* Added batched `update()` methods across the public wrapper API: `Fluid`, `IdealGas`, `Propellant`, `CombustionGas`, `Material`, `Reactants`, and `Equilibrium`.
* Added explicit `Equilibrium.solve()` and `Equilibrium.is_stale` for iterative solvers that want to batch changes before solving.
* Added shared state-update helpers so wrapper APIs can distinguish omitted arguments from intentionally supplied `None` values.

### Improved

* `Reactants.update()` rebuilds feed entries once after multiple changes, and can be called with no arguments to refresh entries after contained propellant/gas states change.
* `Equilibrium.update(..., solve=False)` lets transient and steady-state workflows change reactants, pressure, temperature, mode, and solver options without immediately running the equilibrium solve.
* State-only wrapper updates reuse existing backend metadata and caches where possible instead of forcing object reconstruction.
* Existing simple property setters remain available and continue to update immediately for backward compatibility.

### Notes

* TP and HP public behavior are intended to remain unchanged. SP is now a native constant-pressure thermal solve rather than a public TP-root wrapper.

## 1.0.1

### Improved

* Added support for very high and low mixture ratio values.
* Moved the large species and material registries out of Python source and into packaged JSON data files with thin typed loader modules.
* Preserved eager public wrapper imports so `from thermoprop import Propellant, Equilibrium` continues to return classes, not modules.
* Added clearer `list_species()`, `list_materials()`, and `list_supported_species()` discovery aliases while preserving existing `species()` and `materials()` functions.
* Added shared formatting helpers for wrapper `__str__` output, including consistent `N/A` handling for optional or non-finite properties.
* Added a lightweight `examples/smoke_api.py` discovery smoke test that does not require CoolProp, PYroMat, or RocketProps.
* Added shared public-API helpers for wrapper property introspection.
* Added shared fraction-vector validation for `Fluid`, `IdealGas`, and `CombustionGas`.
* Added missing `Equilibrium.supported_properties()`, `Equilibrium.show_supported_properties()`, and `Equilibrium.supports_property()` support through the shared API mixin.
* Added snake_case `Equilibrium.combustion_gas` and `Equilibrium.combustion_gas_composition()` aliases while preserving the existing `CombustionGas` aliases.
* Added lightweight docstrings for internal CEA equilibrium option/result dataclasses.
* Clarified the `Equilibrium.py` module layout documentation.
* Cleaned the release source tree by excluding local virtual environments, cached bytecode, previous distributions, Git internals, and the stale development `uv.lock`.
* Added vectorized CEA thermo array evaluation for solver internals and gas wrappers.
* Vectorized CEA mixture transport denominator assembly and reaction-conductivity pair accumulation.
* Split Equilibrium input normalization and solver dispatch into `CEAEquilibrium.facade`, keeping `Equilibrium.py` focused on the public wrapper API.

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
