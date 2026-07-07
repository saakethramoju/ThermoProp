"""Public ThermoProp API.

ThermoProp exposes its main engineering-property wrappers directly at package
level so users can write concise imports such as::

    from thermoprop import Fluid, Propellant, Reactants, Equilibrium

The package-level helpers are intentionally small discovery and alias-management
functions.  They delegate to :class:`SpeciesDatabase` and :class:`MaterialDatabase`
so command-line sessions, examples, tests, and future documentation generators
can discover supported names without constructing property objects first.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .Fluid import Fluid
from .IdealGas import IdealGas
from .Propellant import Propellant
from .CombustionGas import CombustionGas
from .Material import Material
from .Reactants import Reactants
from .Equilibrium import Equilibrium

from .CEADatabase import CEA
from .SpeciesDatabase import SpeciesDatabase
from .MaterialDatabase import MaterialDatabase

from .Exceptions import (
    ThermoPropError,
    ThermoPropConfigurationError,
    ThermoPropStateError,
    ThermoPropFlashError,
    ThermoPropRangeError,
    PropertyUnavailableError,
    SpeciesLookupError,
    MaterialLookupError,
    ThermoPropDatabaseError,
    ThermoDataError,
    TransportDataError,
    EquilibriumError,
    EquilibriumSetupError,
    EquilibriumConvergenceError,
)


def list_species() -> list[str]:
    """Return all canonical species names known to ThermoProp.

    The list spans the installed species registry, including names that may map
    to CoolProp, PYroMat, NASA CEA/CEAM, RocketProps, or multiple backends.  It is
    useful for discovery, validation, and generating user-facing option lists.
    Runtime aliases are resolved by lookup functions but are not returned as
    canonical species names here.
    """
    return SpeciesDatabase.species()


def species() -> list[str]:
    """Backward-compatible alias for :func:`list_species`.

    This shorter name remains available for existing scripts.  New examples and
    documentation should prefer :func:`list_species` because it reads more clearly
    and is less likely to collide with a local variable named ``species``.
    """
    return list_species()


def supported_species(wrapper: str | None = None) -> list[str]:
    """Return species supported by a specific ThermoProp wrapper.

    Parameters
    ----------
    wrapper:
        Optional wrapper name such as ``"Fluid"``, ``"IdealGas"``,
        ``"Propellant"``, or ``"CombustionGas"``.  If omitted, all canonical
        ThermoProp species names are returned.

    Returns
    -------
    list[str]
        Canonical ThermoProp species names whose registry records support the
        requested wrapper/backend path.
    """
    if wrapper is None:
        return SpeciesDatabase.species()

    return SpeciesDatabase.supported_species(wrapper)


def list_supported_species(wrapper: str | None = None) -> list[str]:
    """Readable alias for :func:`supported_species`.

    Use this name when an API, notebook, or user interface benefits from a more
    explicit verb phrase.  It accepts the same optional wrapper name and returns
    the same canonical species list.
    """
    return supported_species(wrapper)


def species_aliases() -> dict[str, str]:
    """Return all built-in and runtime species aliases.

    The returned dictionary maps alias strings to canonical ThermoProp species
    names.  It includes aliases loaded from packaged registry data plus aliases
    registered during the current Python process with :func:`add_species_alias`.
    """
    return SpeciesDatabase.aliases()


def add_species_alias(alias: str, species_name: str) -> None:
    """Register a runtime species alias for the current Python process.

    The alias is validated against canonical ThermoProp species names and
    existing aliases so lookups remain unambiguous.  Package data files are not
    modified; the alias exists only in memory for the active process.
    """
    SpeciesDatabase.add_alias(alias, species_name)


def list_materials() -> list[str]:
    """Return all canonical material names known to ThermoProp.

    The list comes from the packaged material registry used by the
    :class:`Material` wrapper.  It is suitable for documentation, interactive
    discovery, model validation, and user-interface option lists.
    """
    return MaterialDatabase.materials()


def materials() -> list[str]:
    """Backward-compatible alias for :func:`list_materials`.

    Existing scripts can keep using ``materials()``.  New examples should prefer
    ``list_materials()`` for clarity and to avoid collisions with local variables.
    """
    return list_materials()


def supported_materials() -> list[str]:
    """Return all canonical materials supported by the :class:`Material` wrapper.

    This currently matches :func:`list_materials`, but the separate function name
    keeps the species and material discovery APIs parallel and leaves room for
    future wrapper-specific filtering.
    """
    return list_materials()


def material_aliases() -> dict[str, str]:
    """Return all built-in and runtime material aliases.

    The returned dictionary maps alias strings to canonical material names.  It
    includes packaged aliases plus aliases registered in the active Python
    process with :func:`add_material_alias`.
    """
    return MaterialDatabase.aliases()


def add_material_alias(alias: str, material_name: str) -> None:
    """Register a runtime material alias for the current Python process.

    The alias is validated against canonical material names and existing aliases.
    Package JSON data are not modified; the alias is only an in-memory convenience
    for project-specific naming conventions.
    """
    MaterialDatabase.add_alias(alias, material_name)


# Convenience aliases.
aliases = species_aliases
add_alias = add_species_alias


__all__ = [
    "__version__",
    "Fluid",
    "IdealGas",
    "Propellant",
    "CombustionGas",
    "Reactants",
    "Equilibrium",
    "Material",
    "CEA",
    "SpeciesDatabase",
    "MaterialDatabase",
    "EquilibriumConvergenceError",
    "EquilibriumSetupError",
    "EquilibriumError",
    "TransportDataError",
    "ThermoDataError",
    "ThermoPropDatabaseError",
    "MaterialLookupError",
    "SpeciesLookupError",
    "PropertyUnavailableError",
    "ThermoPropRangeError",
    "ThermoPropFlashError",
    "ThermoPropStateError",
    "ThermoPropConfigurationError",
    "ThermoPropError",
    "list_species",
    "species",
    "supported_species",
    "list_supported_species",
    "species_aliases",
    "add_species_alias",
    "list_materials",
    "materials",
    "supported_materials",
    "material_aliases",
    "add_material_alias",
    "aliases",
    "add_alias",
]
