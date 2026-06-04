# MaterialRegistry.py

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class MaterialRecord:
    """Canonical material entry."""

    name: str
    category: str


MATERIAL_DATABASE = MappingProxyType({
    # ---------- Aluminum alloys ----------
    "Aluminum 6061": MaterialRecord("Aluminum 6061", "Aluminum Alloy"),
    "Aluminum 7075": MaterialRecord("Aluminum 7075", "Aluminum Alloy"),

    # ---------- Copper alloys ----------
    "Copper C101": MaterialRecord("Copper C101", "Copper Alloy"),
    "Copper C11000": MaterialRecord("Copper C11000", "Copper Alloy"),
    "Copper C17200": MaterialRecord("Copper C17200", "Copper Alloy"),
    "GRCop-42": MaterialRecord("GRCop-42", "Copper Alloy"),
    "GRCop-84": MaterialRecord("GRCop-84", "Copper Alloy"),

    # ---------- Carbon and low-alloy steels ----------
    "1018 Carbon Steel": MaterialRecord("1018 Carbon Steel", "Steel"),
    "1045 Carbon Steel": MaterialRecord("1045 Carbon Steel", "Steel"),
    "3140 Low-Alloy Steel": MaterialRecord("3140 Low-Alloy Steel", "Steel"),
    "4140 Steel": MaterialRecord("4140 Steel", "Steel"),

    # ---------- Stainless steels ----------
    "Stainless Steel 303": MaterialRecord("Stainless Steel 303", "Stainless Steel"),
    "Stainless Steel 304": MaterialRecord("Stainless Steel 304", "Stainless Steel"),
    "Stainless Steel 316": MaterialRecord("Stainless Steel 316", "Stainless Steel"),
    "A286 Steel": MaterialRecord("A286 Steel", "Stainless Steel"),

    # ---------- Nickel superalloys ----------
    "Inconel 625": MaterialRecord("Inconel 625", "Nickel Superalloy"),
    "Inconel 718": MaterialRecord("Inconel 718", "Nickel Superalloy"),

    # ---------- Ceramics / non-metals ----------
    "Graphite": MaterialRecord("Graphite", "Ceramic / Non-Metal"),
})


ALIASES: dict[str, str] = {
    # ---------- Aluminum ----------
    "6061": "Aluminum 6061",
    "al6061": "Aluminum 6061",
    "aluminum6061": "Aluminum 6061",
    "6061t6": "Aluminum 6061",

    "7075": "Aluminum 7075",
    "al7075": "Aluminum 7075",
    "aluminum7075": "Aluminum 7075",
    "7075t6": "Aluminum 7075",

    # ---------- Copper ----------
    "c101": "Copper C101",
    "copper101": "Copper C101",
    "copperc101": "Copper C101",

    "c11000": "Copper C11000",
    "copper11000": "Copper C11000",
    "copperc11000": "Copper C11000",

    "c17200": "Copper C17200",
    "copper17200": "Copper C17200",
    "copperc17200": "Copper C17200",
    "berylliumcopper": "Copper C17200",
    "becu": "Copper C17200",

    "grcop42": "GRCop-42",
    "grcop-42": "GRCop-42",
    "grcop 42": "GRCop-42",

    "grcop84": "GRCop-84",
    "grcop-84": "GRCop-84",
    "grcop 84": "GRCop-84",

    # ---------- Steels ----------
    "1018": "1018 Carbon Steel",
    "1018steel": "1018 Carbon Steel",
    "1018carbon": "1018 Carbon Steel",
    "1018carbonsteel": "1018 Carbon Steel",

    "1045": "1045 Carbon Steel",
    "1045steel": "1045 Carbon Steel",
    "1045carbon": "1045 Carbon Steel",
    "1045carbonsteel": "1045 Carbon Steel",

    "3140": "3140 Low-Alloy Steel",
    "3140steel": "3140 Low-Alloy Steel",
    "3140lowalloy": "3140 Low-Alloy Steel",
    "3140lowalloysteel": "3140 Low-Alloy Steel",

    "4140": "4140 Steel",
    "4140steel": "4140 Steel",

    # ---------- Stainless ----------
    "303": "Stainless Steel 303",
    "303ss": "Stainless Steel 303",
    "ss303": "Stainless Steel 303",
    "stainless303": "Stainless Steel 303",

    "304": "Stainless Steel 304",
    "304ss": "Stainless Steel 304",
    "ss304": "Stainless Steel 304",
    "stainless304": "Stainless Steel 304",

    "316": "Stainless Steel 316",
    "316ss": "Stainless Steel 316",
    "ss316": "Stainless Steel 316",
    "stainless316": "Stainless Steel 316",

    "a286": "A286 Steel",
    "a286steel": "A286 Steel",
    "alloya286": "A286 Steel",

    # ---------- Inconel ----------
    "625": "Inconel 625",
    "in625": "Inconel 625",
    "inc625": "Inconel 625",
    "inconel625": "Inconel 625",
    "alloy625": "Inconel 625",

    "718": "Inconel 718",
    "in718": "Inconel 718",
    "inc718": "Inconel 718",
    "inconel718": "Inconel 718",
    "alloy718": "Inconel 718",

    # ---------- Graphite ----------
    "graphite": "Graphite",
    "carbon": "Graphite",
}


SUPPORTED_MATERIAL_PROPERTIES: tuple[str, ...] = (
    "yield_strength",
    "ultimate_strength",
    "elastic_modulus",
    "torsional_modulus",
    "density",
    "poisson_ratio",
    "thermal_conductivity",
    "specific_heat",
    "coefficient_of_thermal_expansion",
    "melting_point",
    "electrical_resistivity",
)


PROPERTY_ALIASES: dict[str, str] = {
    "yield": "yield_strength",
    "yieldstrength": "yield_strength",
    "yield_strength": "yield_strength",

    "uts": "ultimate_strength",
    "ultimate": "ultimate_strength",
    "ultimate_strength": "ultimate_strength",
    "tensile_strength": "ultimate_strength",

    "e": "elastic_modulus",
    "youngs": "elastic_modulus",
    "youngs_modulus": "elastic_modulus",
    "young_modulus": "elastic_modulus",
    "elastic_modulus": "elastic_modulus",

    "g": "torsional_modulus",
    "shear": "torsional_modulus",
    "shear_modulus": "torsional_modulus",
    "torsional_modulus": "torsional_modulus",

    "rho": "density",
    "density": "density",

    "nu": "poisson_ratio",
    "poisson": "poisson_ratio",
    "poisson_ratio": "poisson_ratio",

    "k": "thermal_conductivity",
    "conductivity": "thermal_conductivity",
    "thermal_conductivity": "thermal_conductivity",

    "cp": "specific_heat",
    "specific_heat": "specific_heat",

    "cte": "coefficient_of_thermal_expansion",
    "thermal_expansion": "coefficient_of_thermal_expansion",
    "coefficient_thermal_expansion": "coefficient_of_thermal_expansion",
    "coefficient_of_thermal_expansion": "coefficient_of_thermal_expansion",

    "tmelt": "melting_point",
    "melting": "melting_point",
    "melting_point": "melting_point",

    "resistivity": "electrical_resistivity",
    "electrical_resistivity": "electrical_resistivity",
}


def _normalize_key(value: str) -> str:
    """Return compact lookup key for material names, aliases, and properties."""
    return "".join(c.lower() for c in str(value) if c.isalnum())


def _build_name_lookup() -> dict[str, str]:
    """Build normalized material-name lookup table."""
    lookup = {_normalize_key(name): name for name in MATERIAL_DATABASE}
    lookup.update({_normalize_key(alias): name for alias, name in ALIASES.items()})
    return lookup


_NAME_LOOKUP = _build_name_lookup()


class classproperty(property):
    """Small descriptor for read-only class-level properties."""

    def __get__(self, obj, owner):
        return self.fget(owner)


class MaterialRegistry:
    """User-facing isotropic material registry for ThermoProp."""

    @staticmethod
    def normalize_name(value: str) -> str:
        """Normalize a user name, alias, or canonical name for lookup."""
        return _normalize_key(value)


    @staticmethod
    def normalize_property(value: str) -> str:
        """Normalize a material property name or alias."""
        key = _normalize_key(value)

        normalized_aliases = {
            _normalize_key(alias): canonical
            for alias, canonical in PROPERTY_ALIASES.items()
        }

        try:
            return normalized_aliases[key]
        except KeyError:
            raise ValueError(
                f"Unknown material property: {value!r}. "
                f"Supported properties: {sorted(set(PROPERTY_ALIASES.values()))}"
            ) from None

    @classmethod
    def name(cls, value: str) -> str:
        """Return canonical ThermoProp material name for a user name or alias."""
        key = cls.normalize_name(value)
        try:
            return _NAME_LOOKUP[key]
        except KeyError:
            raise ValueError(
                f"Unknown material name: {value!r}. "
                f"Available materials: {cls.names}"
            ) from None

    @classmethod
    def record(cls, value: str) -> MaterialRecord:
        """Return full material registry record for a user name or alias."""
        return MATERIAL_DATABASE[cls.name(value)]

    @classmethod
    def category(cls, value: str) -> str:
        """Return material category."""
        return cls.record(value).category

    @classmethod
    def supports(cls, value: str) -> bool:
        """Return True if material name or alias is registered."""
        try:
            cls.name(value)
            return True
        except ValueError:
            return False

    @classmethod
    def add_alias(cls, alias: str, name: str) -> None:
        """Add a runtime material alias."""
        global _NAME_LOOKUP
        ALIASES[alias] = cls.name(name)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def remove_alias(cls, alias: str) -> None:
        """Remove a runtime/user alias and refresh the lookup cache."""
        global _NAME_LOOKUP
        ALIASES.pop(alias, None)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def describe(cls, value: str) -> dict[str, str]:
        """Return compact description of a material lookup."""
        record = cls.record(value)
        return {"input": value, "name": record.name, "category": record.category}

    @classproperty
    def names(cls) -> list[str]:
        """Return all canonical material names."""
        return sorted(MATERIAL_DATABASE.keys())

    @classproperty
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the material alias table."""
        return dict(sorted(ALIASES.items()))

    @classproperty
    def properties(cls) -> list[str]:
        """Return canonical supported material property names."""
        return list(SUPPORTED_MATERIAL_PROPERTIES)

    @classproperty
    def property_aliases(cls) -> dict[str, str]:
        """Return fixed material property aliases."""
        return dict(sorted(PROPERTY_ALIASES.items()))

    @classmethod
    def show_materials(cls) -> list[str]:
        """Print and return all canonical material names."""
        for name in cls.names:
            print(name)
        return cls.names

    @classmethod
    def show_aliases(cls) -> dict[str, str]:
        """Print and return material aliases."""
        aliases = cls.aliases
        if not aliases:
            return aliases
        width = max(len(alias) for alias in aliases)
        print("Material Aliases")
        print("-" * (width + 20))
        for alias, name in aliases.items():
            print(f"{alias:<{width}} -> {name}")
        return aliases

    @classmethod
    def show_properties(cls) -> list[str]:
        """Print and return supported material properties."""
        for prop in cls.properties:
            print(prop)
        return cls.properties
