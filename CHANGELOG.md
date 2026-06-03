# Changelog

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
