from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class CombustionRecord:
    """Canonical combustion/propellant entry and backend-specific names."""

    name: str
    rocketprops: str | None = None
    cea: str | None = None
    cea_reactant: str | None = None


COMBUSTION_DATABASE = MappingProxyType({
    # ---------- Common propellants also useful as CEA gas species ----------
    "Hydrogen": CombustionRecord("Hydrogen", rocketprops="PH2", cea="H2", cea_reactant="H2(L)"),
    "Methane": CombustionRecord("Methane", rocketprops="Methane", cea="CH4", cea_reactant="CH4(L)"),
    "Oxygen": CombustionRecord("Oxygen", rocketprops="LOX", cea="O2", cea_reactant="O2(L)"),
    "Water": CombustionRecord("Water", rocketprops="Water", cea="H2O", cea_reactant="H2O(L)"),
    "Ammonia": CombustionRecord("Ammonia", rocketprops="NH3", cea="NH3", cea_reactant="NH3(L)"),
    "n-Propane": CombustionRecord("n-Propane", rocketprops="Propane", cea="C3H8", cea_reactant="C3H8(L)"),
    "Methanol": CombustionRecord("Methanol", rocketprops="Methanol", cea="CH3OH", cea_reactant="CH3OH(L)"),
    "Ethanol": CombustionRecord("Ethanol", rocketprops="Ethanol", cea="C2H5OH", cea_reactant="C2H5OH(L)"),
    "NitrousOxide": CombustionRecord("NitrousOxide", rocketprops="N2O", cea="N2O", cea_reactant="N2O"),
    "Fluorine": CombustionRecord("Fluorine", rocketprops="F2", cea="F2", cea_reactant="F2(L)"),

    # ---------- RocketProps propellants / named mixtures ----------
    "RP1": CombustionRecord("RP1", rocketprops="RP1", cea_reactant="RP-1"),
    "A50": CombustionRecord("A50", rocketprops="A50"),
    "CLF5": CombustionRecord("CLF5", rocketprops="CLF5", cea="CLF5", cea_reactant="CLF5"),
    "F2": CombustionRecord("F2", rocketprops="F2", cea="F2", cea_reactant="F2(L)"),
    "H2O2": CombustionRecord("H2O2", rocketprops="H2O2", cea="H2O2", cea_reactant="H2O2(L)"),
    "IRFNA": CombustionRecord("IRFNA", rocketprops="IRFNA", cea_reactant="IRFNA"),
    "MHF3": CombustionRecord("MHF3", rocketprops="MHF3"),
    "MMH": CombustionRecord("MMH", rocketprops="MMH", cea_reactant="CH6N2(L)"),
    "MON10": CombustionRecord("MON10", rocketprops="MON10"),
    "MON25": CombustionRecord("MON25", rocketprops="MON25"),
    "MON30": CombustionRecord("MON30", rocketprops="MON30"),
    "N2H4": CombustionRecord("N2H4", rocketprops="N2H4", cea="N2H4", cea_reactant="N2H4(L)"),
    "N2O4": CombustionRecord("N2O4", rocketprops="N2O4", cea="N2O4", cea_reactant="N2O4(L)"),
    "PH2": CombustionRecord("PH2", rocketprops="PH2", cea_reactant="H2(L)"),
    "UDMH": CombustionRecord("UDMH", rocketprops="UDMH", cea_reactant="C2H8N2(L),UDMH"),
})


# General combustion-gas aliases. These are for CEA gas/species lookup in
# CombustionGas, not for CoolProp/PYroMat Fluid or IdealGas lookup.
ALIASES: dict[str, str] = {
    "h2": "Hydrogen",
    "hydrogen": "Hydrogen",
    "gh2": "Hydrogen",
    "gaseous hydrogen": "Hydrogen",
    "hydrogen gas": "Hydrogen",

    "ch4": "Methane",
    "methane": "Methane",
    "gch4": "Methane",
    "gaseous methane": "Methane",
    "methane gas": "Methane",

    "o2": "Oxygen",
    "oxygen": "Oxygen",
    "gox": "Oxygen",
    "gaseous oxygen": "Oxygen",
    "oxygen gas": "Oxygen",

    "h2o": "Water",
    "water": "Water",
    "steam": "Water",

    "nh3": "Ammonia",
    "ammonia": "Ammonia",

    "c3h8": "n-Propane",
    "propane": "n-Propane",
    "n-propane": "n-Propane",

    "ch3oh": "Methanol",
    "methanol": "Methanol",
    "methyl alcohol": "Methanol",

    "c2h5oh": "Ethanol",
    "ethanol": "Ethanol",
    "ethyl alcohol": "Ethanol",

    "n2o": "NitrousOxide",
    "nitrous": "NitrousOxide",
    "nitrous oxide": "NitrousOxide",
    "nitrous-oxide": "NitrousOxide",

    "f2": "Fluorine",
    "fluorine": "Fluorine",
    "fluorine gas": "Fluorine",
}


# Propellant aliases are intentionally separate from gas/species aliases.
# For example, "rp-1" should mean RocketProps RP1 / CEA reactant RP-1 here.
PROPELLANT_ALIASES: dict[str, str] = {
    "rp-1": "RP1",
    "rp1": "RP1",
    "rp 1": "RP1",
    "rpa1": "RP1",
    "kerosene": "RP1",
    "jet-a": "RP1",
    "jeta": "RP1",
    "rocket propellant 1": "RP1",
    "rocket-propellant-1": "RP1",

    "lox": "Oxygen",
    "o2": "Oxygen",
    "oxygen": "Oxygen",
    "liquid oxygen": "Oxygen",
    "gaseous oxygen": "Oxygen",

    "h2": "Hydrogen",
    "lh2": "Hydrogen",
    "gh2": "Hydrogen",
    "ph2": "Hydrogen",
    "hydrogen": "Hydrogen",
    "liquid hydrogen": "Hydrogen",
    "gaseous hydrogen": "Hydrogen",

    "ch4": "Methane",
    "methane": "Methane",
    "lch4": "Methane",
    "lng": "Methane",
    "lco": "Methane",
    "liquid methane": "Methane",
    "gaseous methane": "Methane",

    "n2o": "NitrousOxide",
    "nitrous oxide": "NitrousOxide",
    "nitrous-oxide": "NitrousOxide",

    "nh3": "Ammonia",
    "ammonia": "Ammonia",
    "liquid ammonia": "Ammonia",

    "propane": "n-Propane",
    "c3h8": "n-Propane",
    "liquid propane": "n-Propane",

    "h2o": "Water",
    "water": "Water",

    "n2o4": "N2O4",
    "nto": "N2O4",
    "nitrogen tetroxide": "N2O4",
    "nitrogen-tetroxide": "N2O4",

    "mmh": "MMH",
    "udmh": "UDMH",
    "n2h4": "N2H4",
    "hydrazine": "N2H4",

    "a50": "A50",
    "aerozine50": "A50",
    "aerozine-50": "A50",

    "h2o2": "H2O2",
    "peroxide": "H2O2",
    "hydrogen peroxide": "H2O2",

    "mon10": "MON10",
    "mon25": "MON25",
    "mon30": "MON30",
    "mon-10": "MON10",
    "mon-25": "MON25",
    "mon-30": "MON30",

    "f2": "F2",
    "fluorine": "F2",
    "clf5": "CLF5",
    "chlorine pentafluoride": "CLF5",

    "irfna": "IRFNA",
    "red fuming nitric acid": "IRFNA",
    "inhibited red fuming nitric acid": "IRFNA",

    "mhf3": "MHF3",
}


def _normalize_key(value: str) -> str:
    """Return the compact key used for alias and combustion lookup."""
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _build_name_lookup() -> dict[str, str]:
    """Build the normalized-name lookup table once for fast registry queries."""
    lookup = {
        _normalize_key(name): name
        for name in COMBUSTION_DATABASE
    }

    lookup.update({
        _normalize_key(alias): name
        for alias, name in ALIASES.items()
    })

    return lookup


def _build_propellant_lookup() -> dict[str, str]:
    """Build the normalized lookup table for RocketProps propellants."""
    lookup = {
        _normalize_key(name): name
        for name, record in COMBUSTION_DATABASE.items()
        if record.rocketprops is not None
    }

    lookup.update({
        _normalize_key(alias): name
        for alias, name in PROPELLANT_ALIASES.items()
    })

    return lookup


_NAME_LOOKUP = _build_name_lookup()
_PROPELLANT_LOOKUP = _build_propellant_lookup()


class classproperty(property):
    """Small descriptor for read-only class-level properties."""

    def __get__(self, obj, owner):
        return self.fget(owner)


class CombustionRegistry:
    """
    User-facing combustion and propellant registry for ThermoProp.

    This registry owns the mappings needed by:

        Propellant    -> RocketProps liquid propellant names
        CombustionGas -> NASA CEA / CEAM gas/product species names
        CEA reactants -> NASA CEA / CEAM reactant names

    It is intentionally separate from FluidRegistry so Fluid and IdealGas do not
    inherit propellant-specific behavior. IdealGas can still keep its own CEA
    species mapping in FluidRegistry for transport properties.
    """

    _BACKEND_ALIASES = {
        "rocketprops": "rocketprops",
        "rocket-props": "rocketprops",
        "rp": "rocketprops",
        "propellant": "rocketprops",
        "propellants": "rocketprops",

        "cea": "cea",
        "ceam": "cea",
        "nasa": "cea",
        "nasa-cea": "cea",
        "nasacea": "cea",
        "combustiongas": "cea",
        "combustion-gas": "cea",
        "gas": "cea",
        "species": "cea",

        "cea-reactant": "cea_reactant",
        "cea_reactant": "cea_reactant",
        "ceareactant": "cea_reactant",
        "reactant": "cea_reactant",
        "reactants": "cea_reactant",
    }

    @staticmethod
    def normalize_name(value: str) -> str:
        """Normalize a user name, alias, or canonical name for lookup."""
        return _normalize_key(value)

    @classmethod
    def normalize_backend(cls, backend: str) -> str:
        """Normalize a backend name."""
        lookup = cls.normalize_name(backend)

        try:
            return cls._BACKEND_ALIASES[lookup]
        except KeyError:
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected one of: "
                "'rocketprops', 'propellant', 'cea', or 'cea_reactant'."
            )

    @classmethod
    def name(cls, value: str) -> str:
        """Return the canonical combustion registry name for a name or alias."""
        lookup = cls.normalize_name(value)

        try:
            return _NAME_LOOKUP[lookup]
        except KeyError:
            raise ValueError(f"Unknown combustion species name: {value!r}")

    @classmethod
    def propellant_registry_name(cls, value: str) -> str:
        """Return the canonical registry name used for RocketProps lookup."""
        lookup = cls.normalize_name(value)

        try:
            return _PROPELLANT_LOOKUP[lookup]
        except KeyError:
            raise ValueError(f"Unknown RocketProps propellant name: {value!r}")

    @classmethod
    def record(cls, value: str) -> CombustionRecord:
        """Return the full registry record for a user name or gas/species alias."""
        return COMBUSTION_DATABASE[cls.name(value)]

    @classmethod
    def propellant_record(cls, value: str) -> CombustionRecord:
        """Return the full registry record for a user propellant name or alias."""
        return COMBUSTION_DATABASE[cls.propellant_registry_name(value)]

    @classmethod
    def backend_name(cls, value: str, backend: str) -> str:
        """Return the backend-specific name for a user name or alias."""
        backend = cls.normalize_backend(backend)

        if backend == "rocketprops":
            return cls.propellant_name(value)

        if backend == "cea_reactant":
            return cls.cea_reactant_name(value)

        return cls.cea_name(value)

    @classmethod
    def propellant_name(cls, value: str) -> str:
        """Return the RocketProps backend name for a user propellant name or alias."""
        record = cls.propellant_record(value)

        if record.rocketprops is None:
            raise ValueError(f"{record.name!r} is not supported by RocketProps.")

        return record.rocketprops

    @classmethod
    def cea_name(cls, value: str) -> str:
        """Return the NASA CEA / CEAM species name for a user name or alias."""
        record = cls.record(value)

        if record.cea is None:
            raise ValueError(f"{record.name!r} is not supported by NASA CEA data.")

        return record.cea

    @classmethod
    def cea_reactant_name(cls, value: str) -> str:
        """Return the NASA CEA / CEAM reactant name for a propellant alias."""
        record = cls.propellant_record(value)

        if record.cea_reactant is None:
            raise ValueError(
                f"{record.name!r} does not have a NASA CEA reactant mapping."
            )

        return record.cea_reactant

    @classmethod
    def supports(cls, value: str, backend: str) -> bool:
        """Return True if ``value`` is supported by the selected backend."""
        backend = cls.normalize_backend(backend)

        if backend == "rocketprops":
            return cls.supports_propellant(value)

        if backend == "cea_reactant":
            return cls.supports_cea_reactant(value)

        return cls.supports_cea(value)

    @classmethod
    def supports_propellant(cls, value: str) -> bool:
        """Return True if the propellant alias has a RocketProps mapping."""
        try:
            cls.propellant_name(value)
            return True
        except ValueError:
            return False

    @classmethod
    def supports_cea(cls, value: str) -> bool:
        """Return True if the species has a NASA CEA / CEAM mapping."""
        try:
            return cls.record(value).cea is not None
        except ValueError:
            return False

    @classmethod
    def supports_cea_reactant(cls, value: str) -> bool:
        """Return True if the propellant has a NASA CEA / CEAM reactant mapping."""
        try:
            return cls.propellant_record(value).cea_reactant is not None
        except ValueError:
            return False

    @classmethod
    def add_alias(cls, alias: str, name: str) -> None:
        """Add a CombustionGas/CEA-species alias."""
        global _NAME_LOOKUP
        ALIASES[alias] = cls.name(name)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def add_propellant_alias(cls, alias: str, name: str) -> None:
        """Add a RocketProps-specific propellant alias."""
        global _PROPELLANT_LOOKUP
        PROPELLANT_ALIASES[alias] = cls.propellant_registry_name(name)
        _PROPELLANT_LOOKUP = _build_propellant_lookup()

    @classmethod
    def add_backend_alias(cls, alias: str, name: str, backend: str) -> None:
        """Add an alias for a specific combustion backend."""
        backend = cls.normalize_backend(backend)

        if backend == "rocketprops" or backend == "cea_reactant":
            cls.add_propellant_alias(alias, name)
            return

        cls.add_alias(alias, name)

    @classmethod
    def remove_alias(cls, alias: str) -> None:
        """Remove a CombustionGas/CEA-species alias and refresh the lookup cache."""
        global _NAME_LOOKUP
        ALIASES.pop(alias, None)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def remove_propellant_alias(cls, alias: str) -> None:
        """Remove a RocketProps-specific Propellant alias and refresh the lookup cache."""
        global _PROPELLANT_LOOKUP
        PROPELLANT_ALIASES.pop(alias, None)
        _PROPELLANT_LOOKUP = _build_propellant_lookup()

    @classmethod
    def remove_backend_alias(cls, alias: str, backend: str) -> None:
        """Remove an alias from a specific backend alias table."""
        backend = cls.normalize_backend(backend)

        if backend == "rocketprops" or backend == "cea_reactant":
            cls.remove_propellant_alias(alias)
            return

        cls.remove_alias(alias)

    @classmethod
    def describe(cls, value: str) -> dict[str, str | None | bool]:
        """Return a compact description of a combustion gas/species entry."""
        record = cls.record(value)

        return {
            "input": value,
            "name": record.name,
            "rocketprops": record.rocketprops,
            "cea": record.cea,
            "cea_reactant": record.cea_reactant,
            "supports_propellant": cls.supports_propellant(value),
            "supports_cea": record.cea is not None,
            "supports_cea_reactant": cls.supports_cea_reactant(value),
        }

    @classmethod
    def describe_propellant(cls, value: str) -> dict[str, str | bool | None]:
        """Return a compact description of a Propellant/RocketProps lookup."""
        record = cls.propellant_record(value)

        return {
            "input": value,
            "name": record.name,
            "rocketprops": record.rocketprops,
            "cea_reactant": record.cea_reactant,
            "supports_propellant": record.rocketprops is not None,
            "supports_cea_reactant": record.cea_reactant is not None,
        }

    @classmethod
    def supported_names(cls, backend: str) -> list[str]:
        """Return canonical registry names supported by the selected backend."""
        backend = cls.normalize_backend(backend)

        if backend == "rocketprops":
            return cls.propellant_supported_names

        if backend == "cea_reactant":
            return cls.cea_reactant_supported_names

        return cls.cea_supported_names

    @classproperty
    def names(cls) -> list[str]:
        """Return all canonical combustion registry names."""
        return sorted(COMBUSTION_DATABASE.keys())

    @classproperty
    def propellant_supported_names(cls) -> list[str]:
        """Return canonical names with RocketProps support."""
        return sorted(
            name
            for name, record in COMBUSTION_DATABASE.items()
            if record.rocketprops is not None
        )

    @classproperty
    def cea_supported_names(cls) -> list[str]:
        """Return canonical names with NASA CEA / CEAM gas/species support."""
        return sorted(
            name
            for name, record in COMBUSTION_DATABASE.items()
            if record.cea is not None
        )

    @classproperty
    def cea_reactant_supported_names(cls) -> list[str]:
        """Return canonical names with NASA CEA / CEAM reactant mappings."""
        return sorted(
            name
            for name, record in COMBUSTION_DATABASE.items()
            if record.cea_reactant is not None
        )

    @classproperty
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the CombustionGas/CEA-species alias table."""
        return dict(sorted(ALIASES.items()))

    @classproperty
    def propellant_aliases(cls) -> dict[str, str]:
        """Return a copy of the RocketProps-specific Propellant alias table."""
        return dict(sorted(PROPELLANT_ALIASES.items()))

    @classmethod
    def show_species(cls) -> list[str]:
        """Print and return all canonical combustion registry names."""
        for name in cls.names:
            print(name)

        return cls.names

    @classmethod
    def show_supported(cls, backend: str) -> list[str]:
        """Print and return canonical names supported by the selected backend."""
        names = cls.supported_names(backend)

        for name in names:
            print(name)

        return names

    @classmethod
    def show_aliases(cls) -> dict[str, str]:
        """Print and return CombustionGas/CEA-species aliases."""
        aliases = cls.aliases

        if not aliases:
            return aliases

        width = max(len(alias) for alias in aliases)

        print("CombustionGas / CEA Species Aliases")
        print("-" * (width + 32))

        for alias, name in aliases.items():
            print(f"{alias:<{width}} -> {name}")

        return aliases

    @classmethod
    def show_propellant_aliases(cls) -> dict[str, str]:
        """Print and return RocketProps-specific Propellant aliases."""
        aliases = cls.propellant_aliases

        if not aliases:
            return aliases

        width = max(len(alias) for alias in aliases)

        print("Propellant Aliases")
        print("-" * (width + 20))

        for alias, name in aliases.items():
            backend = cls.propellant_name(name)

            try:
                cea_reactant = cls.cea_reactant_name(name)
                print(f"{alias:<{width}} -> {name} ({backend}, CEA: {cea_reactant})")
            except ValueError:
                print(f"{alias:<{width}} -> {name} ({backend})")

        return aliases

    @classmethod
    def show_cea_reactants(cls) -> list[str]:
        """Print and return canonical propellant names with CEA reactant mappings."""
        names = cls.cea_reactant_supported_names

        for name in names:
            print(f"{name} -> {COMBUSTION_DATABASE[name].cea_reactant}")

        return names

    @classmethod
    def show_backend_names(cls, value: str) -> dict[str, str | bool | None]:
        """Return combustion-side backend mappings for a name or alias."""
        record = cls.record(value)

        return {
            "input": value,
            "canonical": record.name,
            "rocketprops": record.rocketprops,
            "cea": record.cea,
            "cea_reactant": record.cea_reactant,
            "supports_propellant": cls.supports_propellant(value),
            "supports_cea": record.cea is not None,
            "supports_cea_reactant": cls.supports_cea_reactant(value),
        }

    @classmethod
    def show_propellant_backend_names(cls, value: str) -> dict[str, str | bool | None]:
        """Return propellant-side RocketProps and CEA reactant mappings."""
        record = cls.propellant_record(value)

        return {
            "input": value,
            "canonical": record.name,
            "rocketprops": record.rocketprops,
            "cea_reactant": record.cea_reactant,
            "supports_propellant": record.rocketprops is not None,
            "supports_cea_reactant": record.cea_reactant is not None,
        }
