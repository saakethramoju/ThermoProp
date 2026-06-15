"""Public ThermoProp API.

The package exposes the main user-facing wrappers directly so the long-standing
import style remains valid::

    from thermoprop import Propellant, Reactants, Equilibrium

Discovery helpers such as :func:`list_species` and :func:`list_materials` use the
lightweight registry loaders added for 1.0.1.
"""

from __future__ import annotations

__version__ = "1.0.2"

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


def list_species() -> list[str]:
    """Return all ThermoProp-supported fluid/gas/propellant species."""
    return SpeciesDatabase.species()


def species() -> list[str]:
    """Backward-compatible alias for :func:`list_species`."""
    return list_species()


def supported_species(wrapper: str | None = None) -> list[str]:
    """Return species supported by a ThermoProp species wrapper.

    Parameters
    ----------
    wrapper:
        Optional wrapper name such as ``"Fluid"``, ``"IdealGas"``,
        ``"Propellant"``, or ``"CombustionGas"``. If omitted, all registered
        ThermoProp species are returned.
    """
    if wrapper is None:
        return SpeciesDatabase.species()

    return SpeciesDatabase.supported_species(wrapper)


def list_supported_species(wrapper: str | None = None) -> list[str]:
    """Readable alias for :func:`supported_species`."""
    return supported_species(wrapper)


def species_aliases() -> dict[str, str]:
    """Return all registered ThermoProp species aliases."""
    return SpeciesDatabase.aliases()


def add_species_alias(alias: str, species_name: str) -> None:
    """Add a runtime ThermoProp species alias."""
    SpeciesDatabase.add_alias(alias, species_name)


def list_materials() -> list[str]:
    """Return all ThermoProp-supported materials."""
    return MaterialDatabase.materials()


def materials() -> list[str]:
    """Backward-compatible alias for :func:`list_materials`."""
    return list_materials()


def supported_materials() -> list[str]:
    """Return all ThermoProp-supported materials."""
    return list_materials()


def material_aliases() -> dict[str, str]:
    """Return all registered ThermoProp material aliases."""
    return MaterialDatabase.aliases()


def add_material_alias(alias: str, material_name: str) -> None:
    """Add a runtime ThermoProp material alias."""
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
