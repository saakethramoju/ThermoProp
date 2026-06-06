# Changelog

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
