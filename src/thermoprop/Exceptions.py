"""ThermoProp-specific exception types.

The wrappers in this package call several independent backends.  These exception
classes provide a stable ThermoProp-level hierarchy so user code can catch setup,
state, range, database, transport, and equilibrium failures without depending on
backend-specific exception classes.
"""

from __future__ import annotations


class ThermoPropError(Exception):
    """Base class for all ThermoProp-defined exceptions.

    Catch this class when application code wants to handle any expected
    ThermoProp failure category while still allowing unrelated Python exceptions
    to propagate.  More specific subclasses are raised whenever the package can
    identify whether the failure came from configuration, state flashing, range
    checking, database lookup, transport data, or equilibrium convergence.
    """


class ThermoPropConfigurationError(ThermoPropError, ValueError):
    """Raised when user-supplied model configuration is invalid.

    Examples include unsupported composition bases, ambiguous aliases, invalid
    species or material names, malformed reactant groups, and mutually exclusive
    constructor options.  The error generally means the model setup should be
    corrected before trying another thermodynamic state.
    """


class ThermoPropStateError(ThermoPropError, ValueError):
    """Raised when a thermodynamic state specification is invalid.

    This covers unsupported flash pairs, missing required state variables,
    non-finite state values, physically inconsistent state combinations, and
    attempts to assign solved-output properties as independent inputs in a mode
    where they are not valid inputs.
    """


class ThermoPropFlashError(ThermoPropStateError):
    """Raised when a backend cannot solve the requested thermodynamic flash.

    The species name and state pair may be valid, but the selected backend could
    still fail because the state is outside its numerical domain, the phase is
    not supported, or the nonlinear flash did not converge.
    """


class ThermoPropRangeError(ThermoPropStateError):
    """Raised when a requested property evaluation is outside a valid range.

    ThermoProp uses this for temperature, pressure, material-curve, CEA
    polynomial, and transport-fit limits when returning a value would be
    misleading.  Callers may catch it to try a different backend, enable allowed
    extrapolation, or clamp inputs explicitly in their own model.
    """


class PropertyUnavailableError(ThermoPropError, AttributeError):
    """Raised when a wrapper cannot provide a requested public property.

    A property can be unavailable because the active species lacks backend data,
    the current phase does not define the quantity, the selected material has no
    curve for that property, or an advanced backend property is not implemented
    for the current state.
    """


class SpeciesLookupError(ThermoPropConfigurationError):
    """Raised when a species, fluid, gas, propellant, or reactant name is unknown.

    ThermoProp applies canonical-name resolution and runtime aliases before
    raising this exception.  The message normally includes a hint to use the
    relevant discovery helper, such as ``list_species()`` or
    ``supported_species("Fluid")``.
    """


class MaterialLookupError(ThermoPropConfigurationError):
    """Raised when a material name, alias, or material property is unknown.

    This is separate from ``SpeciesLookupError`` so applications that combine
    fluid and material models can report database problems in the correct domain.
    The message usually includes available material or property names.
    """


class ThermoPropDatabaseError(ThermoPropError):
    """Raised when packaged ThermoProp database files are missing or inconsistent.

    This category is intended for installation, packaging, or data-version
    problems rather than ordinary user input mistakes.  Examples include missing
    CEA NPZ files, malformed JSON registries, or a registry entry that points to
    unavailable property data.
    """


class ThermoDataError(ThermoPropDatabaseError):
    """Raised when thermodynamic data cannot be loaded or evaluated reliably.

    The error is used for NASA CEA thermodynamic polynomial data, species records,
    temperature intervals, elemental compositions, or derived thermodynamic
    quantities when the database content is incomplete or inconsistent.
    """


class TransportDataError(ThermoPropDatabaseError):
    """Raised when transport-property data cannot be loaded or evaluated reliably.

    Transport data include species viscosity fits, conductivity fits, and binary
    interaction coefficients used in CEA/CEAM mixture transport calculations.
    This exception distinguishes transport-data problems from thermodynamic-data
    and nonlinear-equilibrium failures.
    """


class EquilibriumError(ThermoPropError):
    """Base class for ThermoProp chemical-equilibrium failures.

    Catch this class around ``Equilibrium`` construction or ``solve()`` calls when
    application code wants to handle both setup and convergence failures from the
    CEA-style equilibrium solver.
    """


class EquilibriumSetupError(EquilibriumError, ThermoPropConfigurationError):
    """Raised when an equilibrium problem cannot be assembled.

    Examples include empty reactant compositions, invalid mode names, missing
    pressure/temperature/entropy inputs for the chosen mode, element sets with no
    compatible product species, or invalid condensed/ion candidate options.
    """


class EquilibriumConvergenceError(EquilibriumError):
    """Raised when the equilibrium nonlinear solve fails to converge.

    The problem setup was accepted, but the TP, HP, SP, or condensed-phase active
    set iteration did not satisfy its convergence criteria.  Callers may inspect
    the message and retry with a different guess temperature, candidate set,
    tolerance, or temperature bound.
    """
