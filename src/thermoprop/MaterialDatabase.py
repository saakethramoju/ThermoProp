"""Engineering-material registry and tabulated property database.

Material records and curves live in ``thermoprop/data/*.json`` so material data
can be updated without editing a multi-thousand-line Python module. The public
API and internal lookup behavior are preserved for the ``Material`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from types import MappingProxyType
from typing import Any
from .Exceptions import MaterialLookupError, ThermoPropConfigurationError


@dataclass(frozen=True, slots=True)
class MaterialRecord:
    """Immutable ThermoProp material identity record."""

    name: str
    category: str


def curve(T: list[float], Y: list[float], units: str) -> dict[str, object]:
    """Return a temperature-dependent property curve."""
    return {"temperature": T, "value": Y, "units": units}


def constant(value: float, units: str, T: float = 298.15) -> dict[str, object]:
    """Return a one-point property curve for source constants."""
    return {"temperature": [T], "value": [value], "units": units}


def _load_json(filename: str) -> Any:
    """Load a package data JSON file from ``thermoprop.data``."""
    with resources.files("thermoprop.data").joinpath(filename).open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


MATERIAL_DATA: dict[str, dict[str, object]] = _load_json("material_data.json")
"""Material property-curve data keyed by canonical material name."""

MATERIAL_DATABASE = MappingProxyType(
    {
        name: MaterialRecord(**record)
        for name, record in _load_json("material_records.json").items()
    }
)
"""Read-only material identity registry keyed by canonical material name."""

_DEFAULT_ALIASES: dict[str, str] = _load_json("material_aliases.json")
_USER_ALIASES: dict[str, str] = {}

_property_payload = _load_json("material_properties.json")
SUPPORTED_MATERIAL_PROPERTIES: tuple[str, ...] = tuple(_property_payload["supported"])
PROPERTY_ALIASES: dict[str, str] = dict(_property_payload["aliases"])


def _normalize_key(value: str) -> str:
    """Return compact lookup key for material names, aliases, and properties."""
    return "".join(c.lower() for c in str(value) if c.isalnum())


def _build_name_lookup() -> dict[str, str]:
    """Build normalized material-name lookup table."""
    lookup = {_normalize_key(name): name for name in MATERIAL_DATABASE}
    lookup.update({_normalize_key(alias): name for alias, name in _DEFAULT_ALIASES.items()})
    lookup.update({_normalize_key(alias): name for alias, name in _USER_ALIASES.items()})
    return lookup


_NAME_LOOKUP: dict[str, str] = _build_name_lookup()

_NORMALIZED_PROPERTY_ALIASES: dict[str, str] = {
    _normalize_key(alias): canonical
    for alias, canonical in PROPERTY_ALIASES.items()
}


class MaterialDatabase:
    """ThermoProp material registry and tabulated material-property database.

    The database stores canonical material names, aliases, categories, and
    temperature-dependent property curves used by the ``Material`` wrapper.
    Runtime aliases can be added without mutating the built-in JSON data.
    """

    # ---------------- Public user API ---------------- #

    @classmethod
    def materials(cls) -> list[str]:
        """Return all ThermoProp-supported material names."""
        return sorted(MATERIAL_DATABASE.keys())

    @classmethod
    def list_materials(cls) -> list[str]:
        """Readable alias for :meth:`materials`."""
        return cls.materials()

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Return built-in and runtime material aliases."""
        out = dict(_DEFAULT_ALIASES)
        out.update(_USER_ALIASES)
        return dict(sorted(out.items()))

    @classmethod
    def add_alias(cls, alias: str, material_name: str) -> None:
        """Add a runtime alias that maps to a ThermoProp material name."""
        global _NAME_LOOKUP

        alias_key = _normalize_key(alias)

        if not alias_key:
            raise ThermoPropConfigurationError("Alias cannot be empty.")

        canonical = cls._name(material_name)

        if alias_key in {_normalize_key(name) for name in MATERIAL_DATABASE}:
            raise ThermoPropConfigurationError(
                f"Alias {alias!r} conflicts with an existing material name."
            )

        existing = _NAME_LOOKUP.get(alias_key)

        if existing is not None:
            if existing == canonical:
                return

            raise ThermoPropConfigurationError(
                f"Alias {alias!r} already maps to {existing!r} and cannot be modified."
            )

        _USER_ALIASES[str(alias)] = canonical
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def supports(cls, value: str) -> bool:
        """Return True when *value* resolves to a material."""
        return cls._supports(value)

    # ---------------- Internal wrapper API ---------------- #

    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return _normalize_key(value)

    @classmethod
    def _normalize_property(cls, value: str) -> str:
        key = _normalize_key(value)

        try:
            return _NORMALIZED_PROPERTY_ALIASES[key]
        except KeyError:
            raise MaterialLookupError(
                f"Unknown material property: {value!r}. "
                f"Supported properties: {sorted(set(PROPERTY_ALIASES.values()))}"
            ) from None

    @classmethod
    def _name(cls, value: str) -> str:
        key = cls._normalize_name(value)

        try:
            return _NAME_LOOKUP[key]
        except KeyError:
            raise MaterialLookupError(
                f"Unknown material name: {value!r}. "
                f"Available materials: {cls.materials()}"
            ) from None

    @classmethod
    def _record(cls, value: str) -> MaterialRecord:
        return MATERIAL_DATABASE[cls._name(value)]

    @classmethod
    def _category(cls, value: str) -> str:
        return cls._record(value).category

    @classmethod
    def _supports(cls, value: str) -> bool:
        try:
            cls._name(value)
            return True
        except MaterialLookupError:
            return False

    @classmethod
    def _data(cls, value: str) -> dict[str, Any]:
        name = cls._name(value)

        try:
            return MATERIAL_DATA[name]
        except KeyError:
            raise MaterialLookupError(
                f"Material {name!r} exists in MaterialDatabase, "
                "but has no data block in MATERIAL_DATA."
            ) from None

    @classmethod
    def _properties(cls) -> list[str]:
        return list(SUPPORTED_MATERIAL_PROPERTIES)

    @classmethod
    def _property_aliases(cls) -> dict[str, str]:
        return dict(sorted(PROPERTY_ALIASES.items()))

    @classmethod
    def _show_materials(cls) -> list[str]:
        names = cls.materials()
        for name in names:
            print(name)
        return names

    @classmethod
    def _show_aliases(cls) -> dict[str, str]:
        aliases = cls.aliases()

        if not aliases:
            return aliases

        width = max(len(alias) for alias in aliases)
        print("Material Aliases")
        print("-" * (width + 20))

        for alias, name in aliases.items():
            print(f"{alias:<{width}} -> {name}")

        return aliases

    @classmethod
    def _show_properties(cls) -> list[str]:
        properties = cls._properties()
        for prop in properties:
            print(prop)
        return properties
