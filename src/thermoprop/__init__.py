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


def species() -> list[str]:
    """Return all ThermoProp-supported fluid/gas/propellant species."""
    return SpeciesDatabase.species()


def supported_species(wrapper: str | None = None) -> list[str]:
    """Return species supported by a ThermoProp species wrapper."""
    if wrapper is None:
        return SpeciesDatabase.species()

    return SpeciesDatabase.supported_species(wrapper)


def species_aliases() -> dict[str, str]:
    """Return all registered ThermoProp species aliases."""
    return SpeciesDatabase.aliases()


def add_species_alias(alias: str, species_name: str) -> None:
    """Add a runtime ThermoProp species alias."""
    SpeciesDatabase.add_alias(alias, species_name)


def materials() -> list[str]:
    """Return all ThermoProp-supported materials."""
    return MaterialDatabase.materials()


def supported_materials() -> list[str]:
    """Return all ThermoProp-supported materials."""
    return MaterialDatabase.materials()


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
    "species",
    "supported_species",
    "species_aliases",
    "add_species_alias",
    "materials",
    "supported_materials",
    "material_aliases",
    "add_material_alias",
    "aliases",
    "add_alias",
]