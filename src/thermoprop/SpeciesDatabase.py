"""Species registry and backend-name resolver for ThermoProp.

The species and alias records live in ``thermoprop/data/*.json`` so updates to
backend mappings do not require editing thousands of lines of Python source.
This module remains the public, typed resolver used by Fluid, IdealGas,
Propellant, CombustionGas, Reactants, and Equilibrium.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
import json
from types import MappingProxyType
from typing import Any

try:
    from .CEADatabase import CEA
except Exception:  # pragma: no cover - keep discovery usable without CEA data
    CEA = None


@dataclass(frozen=True, slots=True)
class SpeciesRecord:
    """Immutable ThermoProp species entry and backend-specific names."""

    name: str
    coolprop: str | None = None
    pyromat: str | None = None  # no ig. prefix
    cea: str | None = None
    rocketprops: str | None = None
    coolprop_surrogate: bool = False
    notes: str | None = None


def _load_json(filename: str) -> Any:
    """Load a package data JSON file from ``thermoprop.data``."""
    with resources.files("thermoprop.data").joinpath(filename).open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


SPECIES_DATABASE = MappingProxyType(
    {
        name: SpeciesRecord(**record)
        for name, record in _load_json("species_records.json").items()
    }
)
"""Read-only mapping of canonical ThermoProp species names to records."""

DEFAULT_ALIASES = MappingProxyType(_load_json("species_aliases.json"))
"""Read-only mapping of built-in aliases to canonical ThermoProp species."""

_USER_ALIASES: dict[str, str] = {}


def _normalize_key(value: str) -> str:
    """Return compact lookup key for species names and aliases."""
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _looks_like_cea_gas(name: str | None) -> bool:
    """Heuristic fallback used when the CEA database is not importable."""
    if name is None:
        return False
    text = str(name)
    if "(" in text and ")" in text:
        return False
    if text.upper() in {"RP-1", "JP-4", "JP-5", "IRFNA"}:
        return False
    return True


def _build_alias_lookup() -> dict[str, str]:
    """Build normalized alias lookup table including runtime aliases."""
    lookup = {_normalize_key(alias): name for alias, name in DEFAULT_ALIASES.items()}
    lookup.update({_normalize_key(alias): name for alias, name in _USER_ALIASES.items()})
    return lookup


def _build_backend_lookup(field: str) -> dict[str, str]:
    """Build backend-name -> canonical-name lookup for a record field."""
    lookup: dict[str, str] = {}
    for name, record in SPECIES_DATABASE.items():
        value = getattr(record, field)
        if value is not None and value not in lookup:
            lookup[value] = name
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()
_COOLPROP_LOOKUP = _build_backend_lookup("coolprop")
_PYROMAT_LOOKUP = _build_backend_lookup("pyromat")
_CEA_LOOKUP = _build_backend_lookup("cea")
_ROCKETPROPS_LOOKUP = _build_backend_lookup("rocketprops")


class SpeciesDatabase:
    """ThermoProp species registry and cross-backend name resolver.

    The registry separates convenient ThermoProp names from backend-specific
    names. For example, one canonical ThermoProp species can map to different
    names in CoolProp, PYroMat, NASA CEA, and RocketProps.
    """

    _WRAPPER_ALIASES = {
        "fluid": "fluid",
        "realfluid": "fluid",
        "real-fluid": "fluid",
        "coolprop": "fluid",
        "cool-prop": "fluid",
        "idealgas": "idealgas",
        "ideal-gas": "idealgas",
        "ideal_gas": "idealgas",
        "gas": "idealgas",
        "pyromat": "idealgas",
        "pyro-mat": "idealgas",
        "propellant": "propellant",
        "propellants": "propellant",
        "rocketprops": "propellant",
        "rocket-props": "propellant",
        "combustiongas": "combustiongas",
        "combustion-gas": "combustiongas",
        "combustion_gas": "combustiongas",
        "cea": "combustiongas",
        "nasa-cea": "combustiongas",
    }

    _BACKEND_FIELDS = {
        "fluid": "coolprop",
        "idealgas": "pyromat",
        "propellant": ("rocketprops", "cea"),
        "combustiongas": "cea",
    }

    # ---------------- Public user API ---------------- #

    @classmethod
    def species(cls) -> list[str]:
        """Return every ThermoProp canonical species name."""
        return sorted(SPECIES_DATABASE.keys())

    @classmethod
    def list_species(cls) -> list[str]:
        """Readable alias for :meth:`species`."""
        return cls.species()

    @classmethod
    def supported_species(cls, wrapper: str) -> list[str]:
        """Return ThermoProp species supported by a wrapper.

        ``wrapper`` may be one of ``"Fluid"``, ``"IdealGas"``,
        ``"Propellant"``, or ``"CombustionGas"``. Backend aliases such as
        ``"CoolProp"``, ``"PYroMat"``, ``"RocketProps"``, and ``"CEA"`` are
        accepted too.
        """
        wrapper = cls._normalize_wrapper(wrapper)

        if wrapper == "combustiongas":
            return sorted(
                name
                for name, record in SPECIES_DATABASE.items()
                if record.cea is not None and cls._is_combustion_gas_species(record.cea)
            )

        field = cls._BACKEND_FIELDS[wrapper]

        if isinstance(field, tuple):
            return sorted(
                name
                for name, record in SPECIES_DATABASE.items()
                if any(getattr(record, item) is not None for item in field)
            )

        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if getattr(record, field) is not None
        )

    @classmethod
    def list_supported_species(cls, wrapper: str) -> list[str]:
        """Readable alias for :meth:`supported_species`."""
        return cls.supported_species(wrapper)

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Return built-in plus runtime aliases."""
        out = dict(DEFAULT_ALIASES)
        out.update(_USER_ALIASES)
        return dict(sorted(out.items()))

    @classmethod
    def add_alias(cls, alias: str, thermoprop_name: str) -> None:
        """Add a runtime-only user alias for an existing ThermoProp species."""
        global _ALIAS_LOOKUP

        alias = str(alias).strip()
        if not alias:
            raise ValueError("Alias cannot be empty.")

        canonical = cls._name(thermoprop_name)
        key = _normalize_key(alias)

        if key in _ALIAS_LOOKUP:
            raise ValueError(
                f"Alias {alias!r} already resolves to {_ALIAS_LOOKUP[key]!r}."
            )

        _USER_ALIASES[alias] = canonical
        _ALIAS_LOOKUP = _build_alias_lookup()

    @classmethod
    def record(cls, value: str) -> dict[str, Any]:
        """Return a public dictionary copy of a species record."""
        return asdict(cls._record(value))

    @classmethod
    def supports(cls, value: str, wrapper: str) -> bool:
        """Return True when *value* is supported by *wrapper*."""
        return cls._supports_wrapper(value, wrapper)

    @classmethod
    def backend_name(cls, value: str, backend: str, *, include_prefix: bool = False) -> str:
        """Return backend-specific name for a ThermoProp species."""
        return cls._backend_name(value, backend, include_prefix=include_prefix)

    # ---------------- Internal lookup API for ThermoProp wrappers ---------------- #

    @staticmethod
    def _normalize_name(value: str) -> str:
        return _normalize_key(value)

    @classmethod
    def _normalize_wrapper(cls, wrapper: str) -> str:
        key = cls._normalize_name(wrapper)

        try:
            return cls._WRAPPER_ALIASES[key]
        except KeyError:
            raise ValueError(
                f"Unknown wrapper: {wrapper!r}. Expected one of "
                "'Fluid', 'IdealGas', 'Propellant', or 'CombustionGas'."
            ) from None

    @classmethod
    def _name(cls, value: str) -> str:
        value = str(value).strip()

        if value in SPECIES_DATABASE:
            return value

        key = _normalize_key(value)

        if key in _ALIAS_LOOKUP:
            return _ALIAS_LOOKUP[key]

        # Strict ThermoProp names are case-sensitive, but this small fallback
        # allows exact backend-like CEA names generated outside the alias table.
        for name in SPECIES_DATABASE:
            if _normalize_key(name) == key:
                return name

        raise ValueError(f"Unknown ThermoProp species name or alias: {value!r}")

    @classmethod
    def _record(cls, value: str) -> SpeciesRecord:
        return SPECIES_DATABASE[cls._name(value)]

    @classmethod
    def _backend_name(cls, value: str, backend: str, *, include_prefix: bool = False) -> str:
        backend_key = cls._normalize_name(backend)

        if backend_key in {"fluid", "coolprop", "cool-prop"}:
            return cls._coolprop_name(value)

        if backend_key in {"idealgas", "ideal-gas", "ideal_gas", "pyromat", "pyro-mat"}:
            return cls._pyromat_name(value, include_prefix=include_prefix)

        if backend_key in {"combustiongas", "combustion-gas", "combustion_gas", "cea", "nasa-cea"}:
            return cls._cea_name(value)

        if backend_key in {"propellant", "rocketprops", "rocket-props"}:
            return cls._rocketprops_name(value)

        raise ValueError(f"Unknown backend: {backend!r}")

    @classmethod
    def _coolprop_name(cls, value: str) -> str:
        record = cls._record(value)
        if record.coolprop is None:
            raise ValueError(f"{record.name!r} is not supported by Fluid/CoolProp.")
        return record.coolprop

    @classmethod
    def _pyromat_name(cls, value: str, *, include_prefix: bool = False) -> str:
        record = cls._record(value)
        if record.pyromat is None:
            raise ValueError(f"{record.name!r} is not supported by IdealGas/PYroMat.")
        return f"ig.{record.pyromat}" if include_prefix else record.pyromat

    @classmethod
    def _cea_name(cls, value: str) -> str:
        record = cls._record(value)
        if record.cea is None:
            raise ValueError(f"{record.name!r} is not supported by CEA.")
        return record.cea

    @classmethod
    def _rocketprops_name(cls, value: str) -> str:
        record = cls._record(value)
        if record.rocketprops is None:
            raise ValueError(f"{record.name!r} is not supported by RocketProps.")
        return record.rocketprops

    @classmethod
    def _propellant_name(cls, value: str) -> str:
        return cls._rocketprops_name(value)

    @classmethod
    def _supports_wrapper(cls, value: str, wrapper: str) -> bool:
        wrapper = cls._normalize_wrapper(wrapper)

        try:
            record = cls._record(value)
        except ValueError:
            return False

        if wrapper == "fluid":
            return record.coolprop is not None

        if wrapper == "idealgas":
            return record.pyromat is not None

        if wrapper == "propellant":
            return record.rocketprops is not None or record.cea is not None

        if wrapper == "combustiongas":
            return record.cea is not None and cls._is_combustion_gas_species(record.cea)

        return False

    @classmethod
    def _supported_by(cls, value: str) -> dict[str, bool]:
        return {
            "Fluid": cls._supports_wrapper(value, "Fluid"),
            "IdealGas": cls._supports_wrapper(value, "IdealGas"),
            "Propellant": cls._supports_wrapper(value, "Propellant"),
            "CombustionGas": cls._supports_wrapper(value, "CombustionGas"),
        }

    @classmethod
    def _is_combustion_gas_species(cls, cea_name: str) -> bool:
        """Return True if a CEA entry is a gas/product thermo species."""
        if CEA is None:
            return _looks_like_cea_gas(cea_name)

        try:
            return bool(CEA.has_thermo(cea_name) and CEA.is_gas(cea_name))
        except Exception:
            return _looks_like_cea_gas(cea_name)


# Compatibility name for wrappers that import SpeciesRegistry.
SpeciesRegistry = SpeciesDatabase
