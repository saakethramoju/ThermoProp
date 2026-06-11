from .Fluid import Fluid
from .IdealGas import IdealGas
from .Propellant import Propellant
from .CombustionGas import CombustionGas
from .Material import Material
from .MaterialRegistry import MaterialRegistry

from SpeciesDatabase import SpeciesDatabase


def species() -> list[str]:
    """
    Return all ThermoProp-supported species.
    """
    return SpeciesDatabase.species()


def supported_species(wrapper: str | None = None) -> list[str]:
    """
    Return species supported by a wrapper.

    Parameters
    ----------
    wrapper : str | None
        Wrapper name:

        - "Fluid"
        - "IdealGas"
        - "Propellant"
        - "CombustionGas"

        If None, all ThermoProp species are returned.
    """
    if wrapper is None:
        return SpeciesDatabase.species()

    return SpeciesDatabase.supported_species(wrapper)


def aliases() -> dict[str, str]:
    """
    Return all registered ThermoProp aliases.
    """
    return SpeciesDatabase.aliases()


def add_alias(alias: str, species_name: str) -> None:
    """
    Add a runtime ThermoProp alias.

    Parameters
    ----------
    alias : str
        Alias to register.

    species_name : str
        ThermoProp species name.
    """
    SpeciesDatabase.add_alias(alias, species_name)


__all__ = [
    "Fluid",
    "IdealGas",
    "Propellant",
    "CombustionGas",
    "Material",
    "MaterialRegistry",
    "species",
    "supported_species",
    "aliases",
    "add_alias",
]