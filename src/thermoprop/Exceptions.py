"""
ThermoProp-specific exception types.

These classes are used for higher-level ThermoProp 
failure categories that users may want to catch explicitly.
"""

from __future__ import annotations


class ThermoPropError(Exception):
    """Base class for all ThermoProp-specific errors."""


class ThermoPropConfigurationError(ThermoPropError, ValueError):
    """Invalid setup, unsupported input combination, or invalid wrapper configuration."""


class ThermoPropStateError(ThermoPropError, ValueError):
    """Invalid, incomplete, or inconsistent thermodynamic state."""


class ThermoPropFlashError(ThermoPropStateError):
    """Unsupported or failed thermodynamic flash pair."""


class ThermoPropRangeError(ThermoPropStateError):
    """A state is outside the valid data or polynomial range."""


class PropertyUnavailableError(ThermoPropError, AttributeError):
    """A requested property is not available for this object/backend/state."""


class SpeciesLookupError(ThermoPropConfigurationError):
    """A species, alias, backend species name, or CEA name could not be resolved."""


class MaterialLookupError(ThermoPropConfigurationError):
    """A material name, alias, or material property could not be resolved."""


class ThermoPropDatabaseError(ThermoPropError):
    """A required database file, table, coefficient set, or record is invalid or missing."""


class ThermoDataError(ThermoPropDatabaseError):
    """Thermodynamic coefficient data are missing, invalid, or unusable."""


class TransportDataError(PropertyUnavailableError):
    """Transport data are not available for the requested species/state."""


class EquilibriumError(ThermoPropError):
    """Base class for equilibrium setup and solve failures."""


class EquilibriumSetupError(EquilibriumError, ThermoPropConfigurationError):
    """Equilibrium inputs are invalid before the solve starts."""


class EquilibriumConvergenceError(EquilibriumError, RuntimeError):
    """An equilibrium solve failed to converge or failed during iteration."""
