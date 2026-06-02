from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType



@dataclass(frozen=True)
class SpeciesRecord:
    """Canonical species entry and backend-specific names."""

    name: str
    coolprop: str | None = None
    pyromat: str | None = None  # no "ig." prefix


SPECIES_DATABASE = MappingProxyType({
    # ---------- Common gases / shared ----------
    "Air": SpeciesRecord("Air", coolprop="Air", pyromat="air"),
    "Argon": SpeciesRecord("Argon", coolprop="Argon", pyromat="Ar"),
    "CarbonDioxide": SpeciesRecord("CarbonDioxide", coolprop="CarbonDioxide", pyromat="CO2"),
    "CarbonMonoxide": SpeciesRecord("CarbonMonoxide", coolprop="CarbonMonoxide", pyromat="CO"),
    "Helium": SpeciesRecord("Helium", coolprop="Helium", pyromat="He"),
    "Hydrogen": SpeciesRecord("Hydrogen", coolprop="Hydrogen", pyromat="H2"),
    "Methane": SpeciesRecord("Methane", coolprop="Methane", pyromat="CH4"),
    "Nitrogen": SpeciesRecord("Nitrogen", coolprop="Nitrogen", pyromat="N2"),
    "Oxygen": SpeciesRecord("Oxygen", coolprop="Oxygen", pyromat="O2"),
    "Water": SpeciesRecord("Water", coolprop="Water", pyromat="H2O"),

    # ---------- Common fluids with PYroMat mappings ----------
    "Ammonia": SpeciesRecord("Ammonia", coolprop="Ammonia", pyromat="NH3"),
    "Ethane": SpeciesRecord("Ethane", coolprop="Ethane", pyromat="C2H6"),
    "Ethylene": SpeciesRecord("Ethylene", coolprop="Ethylene", pyromat="C2H4"),
    "n-Propane": SpeciesRecord("n-Propane", coolprop="n-Propane", pyromat="C3H8"),
    "n-Butane": SpeciesRecord("n-Butane", coolprop="n-Butane", pyromat="C4H10"),
    "IsoButane": SpeciesRecord("IsoButane", coolprop="IsoButane", pyromat=None),
    "Benzene": SpeciesRecord("Benzene", coolprop="Benzene", pyromat="C6H6"),
    "Toluene": SpeciesRecord("Toluene", coolprop="Toluene", pyromat="C7H8"),
    "Methanol": SpeciesRecord("Methanol", coolprop="Methanol", pyromat="CH4O"),
    "Ethanol": SpeciesRecord("Ethanol", coolprop="Ethanol", pyromat="C2H6O"),
    "NitrousOxide": SpeciesRecord("NitrousOxide", coolprop="NitrousOxide", pyromat="N2O"),
    "HydrogenChloride": SpeciesRecord("HydrogenChloride", coolprop="HydrogenChloride", pyromat="HCl"),
    "HydrogenSulfide": SpeciesRecord("HydrogenSulfide", coolprop="HydrogenSulfide", pyromat="H2S"),
    "SulfurDioxide": SpeciesRecord("SulfurDioxide", coolprop="SulfurDioxide", pyromat="SO2"),
    "SulfurHexafluoride": SpeciesRecord("SulfurHexafluoride", coolprop="SulfurHexafluoride", pyromat="SF6"),
    "Neon": SpeciesRecord("Neon", coolprop="Neon", pyromat="Ne"),
    "Krypton": SpeciesRecord("Krypton", coolprop="Krypton", pyromat="Kr"),
    "Xenon": SpeciesRecord("Xenon", coolprop="Xenon", pyromat="Xe"),

    # ---------- CoolProp pure / pseudo-pure fluids ----------
    "1-Butene": SpeciesRecord("1-Butene", coolprop="1-Butene", pyromat=None),
    "Acetone": SpeciesRecord("Acetone", coolprop="Acetone", pyromat=None),
    "CarbonylSulfide": SpeciesRecord("CarbonylSulfide", coolprop="CarbonylSulfide", pyromat=None),
    "CycloHexane": SpeciesRecord("CycloHexane", coolprop="CycloHexane", pyromat=None),
    "CycloPropane": SpeciesRecord("CycloPropane", coolprop="CycloPropane", pyromat=None),
    "Cyclopentane": SpeciesRecord("Cyclopentane", coolprop="Cyclopentane", pyromat=None),
    "D4": SpeciesRecord("D4", coolprop="D4", pyromat=None),
    "D5": SpeciesRecord("D5", coolprop="D5", pyromat=None),
    "D6": SpeciesRecord("D6", coolprop="D6", pyromat=None),
    "Deuterium": SpeciesRecord("Deuterium", coolprop="Deuterium", pyromat=None),
    "Dichloroethane": SpeciesRecord("Dichloroethane", coolprop="Dichloroethane", pyromat=None),
    "DiethylEther": SpeciesRecord("DiethylEther", coolprop="DiethylEther", pyromat=None),
    "DimethylCarbonate": SpeciesRecord("DimethylCarbonate", coolprop="DimethylCarbonate", pyromat=None),
    "DimethylEther": SpeciesRecord("DimethylEther", coolprop="DimethylEther", pyromat=None),
    "EthylBenzene": SpeciesRecord("EthylBenzene", coolprop="EthylBenzene", pyromat=None),
    "EthyleneOxide": SpeciesRecord("EthyleneOxide", coolprop="EthyleneOxide", pyromat=None),
    "Fluorine": SpeciesRecord("Fluorine", coolprop="Fluorine", pyromat=None),
    "HFE143m": SpeciesRecord("HFE143m", coolprop="HFE143m", pyromat=None),
    "HeavyWater": SpeciesRecord("HeavyWater", coolprop="HeavyWater", pyromat=None),
    "IsoButene": SpeciesRecord("IsoButene", coolprop="IsoButene", pyromat=None),
    "Isohexane": SpeciesRecord("Isohexane", coolprop="Isohexane", pyromat=None),
    "Isopentane": SpeciesRecord("Isopentane", coolprop="Isopentane", pyromat=None),
    "MD2M": SpeciesRecord("MD2M", coolprop="MD2M", pyromat=None),
    "MD3M": SpeciesRecord("MD3M", coolprop="MD3M", pyromat=None),
    "MD4M": SpeciesRecord("MD4M", coolprop="MD4M", pyromat=None),
    "MDM": SpeciesRecord("MDM", coolprop="MDM", pyromat=None),
    "MM": SpeciesRecord("MM", coolprop="MM", pyromat=None),
    "MethylLinoleate": SpeciesRecord("MethylLinoleate", coolprop="MethylLinoleate", pyromat=None),
    "MethylLinolenate": SpeciesRecord("MethylLinolenate", coolprop="MethylLinolenate", pyromat=None),
    "MethylOleate": SpeciesRecord("MethylOleate", coolprop="MethylOleate", pyromat=None),
    "MethylPalmitate": SpeciesRecord("MethylPalmitate", coolprop="MethylPalmitate", pyromat=None),
    "MethylStearate": SpeciesRecord("MethylStearate", coolprop="MethylStearate", pyromat=None),
    "Neopentane": SpeciesRecord("Neopentane", coolprop="Neopentane", pyromat=None),
    "Novec649": SpeciesRecord("Novec649", coolprop="Novec649", pyromat=None),
    "OrthoDeuterium": SpeciesRecord("OrthoDeuterium", coolprop="OrthoDeuterium", pyromat=None),
    "OrthoHydrogen": SpeciesRecord("OrthoHydrogen", coolprop="OrthoHydrogen", pyromat=None),
    "ParaDeuterium": SpeciesRecord("ParaDeuterium", coolprop="ParaDeuterium", pyromat=None),
    "ParaHydrogen": SpeciesRecord("ParaHydrogen", coolprop="ParaHydrogen", pyromat=None),
    "Propylene": SpeciesRecord("Propylene", coolprop="Propylene", pyromat=None),
    "Propyne": SpeciesRecord("Propyne", coolprop="Propyne", pyromat=None),

    # ---------- Refrigerants / pseudo-pure ----------
    "R11": SpeciesRecord("R11", coolprop="R11", pyromat=None),
    "R113": SpeciesRecord("R113", coolprop="R113", pyromat=None),
    "R114": SpeciesRecord("R114", coolprop="R114", pyromat=None),
    "R115": SpeciesRecord("R115", coolprop="R115", pyromat=None),
    "R116": SpeciesRecord("R116", coolprop="R116", pyromat=None),
    "R12": SpeciesRecord("R12", coolprop="R12", pyromat=None),
    "R123": SpeciesRecord("R123", coolprop="R123", pyromat=None),
    "R1233zd(E)": SpeciesRecord("R1233zd(E)", coolprop="R1233zd(E)", pyromat=None),
    "R1234yf": SpeciesRecord("R1234yf", coolprop="R1234yf", pyromat=None),
    "R1234ze(E)": SpeciesRecord("R1234ze(E)", coolprop="R1234ze(E)", pyromat=None),
    "R1234ze(Z)": SpeciesRecord("R1234ze(Z)", coolprop="R1234ze(Z)", pyromat=None),
    "R124": SpeciesRecord("R124", coolprop="R124", pyromat=None),
    "R1243zf": SpeciesRecord("R1243zf", coolprop="R1243zf", pyromat=None),
    "R125": SpeciesRecord("R125", coolprop="R125", pyromat=None),
    "R13": SpeciesRecord("R13", coolprop="R13", pyromat=None),
    "R1336mzz(E)": SpeciesRecord("R1336mzz(E)", coolprop="R1336mzz(E)", pyromat=None),
    "R134a": SpeciesRecord("R134a", coolprop="R134a", pyromat=None),
    "R13I1": SpeciesRecord("R13I1", coolprop="R13I1", pyromat=None),
    "R14": SpeciesRecord("R14", coolprop="R14", pyromat=None),
    "R141b": SpeciesRecord("R141b", coolprop="R141b", pyromat=None),
    "R142b": SpeciesRecord("R142b", coolprop="R142b", pyromat=None),
    "R143a": SpeciesRecord("R143a", coolprop="R143a", pyromat=None),
    "R152A": SpeciesRecord("R152A", coolprop="R152A", pyromat=None),
    "R161": SpeciesRecord("R161", coolprop="R161", pyromat=None),
    "R21": SpeciesRecord("R21", coolprop="R21", pyromat=None),
    "R218": SpeciesRecord("R218", coolprop="R218", pyromat=None),
    "R22": SpeciesRecord("R22", coolprop="R22", pyromat=None),
    "R227EA": SpeciesRecord("R227EA", coolprop="R227EA", pyromat=None),
    "R23": SpeciesRecord("R23", coolprop="R23", pyromat=None),
    "R236EA": SpeciesRecord("R236EA", coolprop="R236EA", pyromat=None),
    "R236FA": SpeciesRecord("R236FA", coolprop="R236FA", pyromat=None),
    "R245ca": SpeciesRecord("R245ca", coolprop="R245ca", pyromat=None),
    "R245fa": SpeciesRecord("R245fa", coolprop="R245fa", pyromat=None),
    "R32": SpeciesRecord("R32", coolprop="R32", pyromat=None),
    "R365MFC": SpeciesRecord("R365MFC", coolprop="R365MFC", pyromat=None),
    "R40": SpeciesRecord("R40", coolprop="R40", pyromat=None),
    "R404A": SpeciesRecord("R404A", coolprop="R404A", pyromat=None),
    "R407C": SpeciesRecord("R407C", coolprop="R407C", pyromat=None),
    "R41": SpeciesRecord("R41", coolprop="R41", pyromat=None),
    "R410A": SpeciesRecord("R410A", coolprop="R410A", pyromat=None),
    "R507A": SpeciesRecord("R507A", coolprop="R507A", pyromat=None),
    "RC318": SpeciesRecord("RC318", coolprop="RC318", pyromat=None),
    "SES36": SpeciesRecord("SES36", coolprop="SES36", pyromat=None),

    # ---------- More hydrocarbons / aromatics ----------
    "cis-2-Butene": SpeciesRecord("cis-2-Butene", coolprop="cis-2-Butene", pyromat=None),
    "trans-2-Butene": SpeciesRecord("trans-2-Butene", coolprop="trans-2-Butene", pyromat=None),
    "m-Xylene": SpeciesRecord("m-Xylene", coolprop="m-Xylene", pyromat=None),
    "o-Xylene": SpeciesRecord("o-Xylene", coolprop="o-Xylene", pyromat=None),
    "p-Xylene": SpeciesRecord("p-Xylene", coolprop="p-Xylene", pyromat=None),
    "n-Decane": SpeciesRecord("n-Decane", coolprop="n-Decane", pyromat=None),
    "n-Dodecane": SpeciesRecord("n-Dodecane", coolprop="n-Dodecane", pyromat=None),
    "n-Heptane": SpeciesRecord("n-Heptane", coolprop="n-Heptane", pyromat=None),
    "n-Hexane": SpeciesRecord("n-Hexane", coolprop="n-Hexane", pyromat=None),
    "n-Nonane": SpeciesRecord("n-Nonane", coolprop="n-Nonane", pyromat=None),
    "n-Octane": SpeciesRecord("n-Octane", coolprop="n-Octane", pyromat=None),
    "n-Pentane": SpeciesRecord("n-Pentane", coolprop="n-Pentane", pyromat=None),
    "n-Undecane": SpeciesRecord("n-Undecane", coolprop="n-Undecane", pyromat=None),
})


ALIASES: dict[str, str] = {
    "air": "Air",

    "ar": "Argon",
    "argon": "Argon",

    "co2": "CarbonDioxide",
    "carbon dioxide": "CarbonDioxide",
    "carbon-dioxide": "CarbonDioxide",

    "co": "CarbonMonoxide",
    "carbon monoxide": "CarbonMonoxide",
    "carbon-monoxide": "CarbonMonoxide",

    "he": "Helium",
    "helium": "Helium",

    "h2": "Hydrogen",
    "hydrogen": "Hydrogen",
    "gh2": "Hydrogen",
    "lh2": "Hydrogen",

    "ch4": "Methane",
    "methane": "Methane",
    "lng": "Methane",

    "n2": "Nitrogen",
    "gn2": "Nitrogen",
    "ln2": "Nitrogen",
    "nitrogen": "Nitrogen",

    "o2": "Oxygen",
    "go2": "Oxygen",
    "lox": "Oxygen",
    "gox": "Oxygen",
    "oxygen": "Oxygen",

    "h2o": "Water",
    "steam": "Water",
    "water": "Water",

    "nh3": "Ammonia",
    "ammonia": "Ammonia",

    "c2h6": "Ethane",
    "ethane": "Ethane",

    "c2h4": "Ethylene",
    "ethylene": "Ethylene",

    "c3h8": "n-Propane",
    "propane": "n-Propane",
    "n-propane": "n-Propane",
    "r290": "n-Propane",

    "c4h10": "n-Butane",
    "butane": "n-Butane",
    "n-butane": "n-Butane",

    "isobutane": "IsoButane",
    "iso-butane": "IsoButane",
    "r600a": "IsoButane",

    "n2o": "NitrousOxide",
    "nitrous oxide": "NitrousOxide",
    "nitrous-oxide": "NitrousOxide",

    "rp-1": "n-Dodecane",
    "rp1": "n-Dodecane",
    "jeta": "n-Dodecane",
    "jet-a": "n-Dodecane",
    "kerosene": "n-Dodecane",

    "octane": "n-Octane",
    "decane": "n-Decane",
    "dodecane": "n-Dodecane",
    "heptane": "n-Heptane",
    "hexane": "n-Hexane",
    "nonane": "n-Nonane",
    "pentane": "n-Pentane",
    "undecane": "n-Undecane",
}


def _normalize_key(value: str) -> str:
    """Return the compact key used for alias and species lookup."""
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
        for name in SPECIES_DATABASE
    }

    lookup.update({
        _normalize_key(alias): name
        for alias, name in ALIASES.items()
    })

    return lookup


_NAME_LOOKUP = _build_name_lookup()


class classproperty(property):
    """Small descriptor for read-only class-level properties."""

    def __get__(self, obj, owner):
        return self.fget(owner)


class FluidRegistry:
    """Registry that maps user-facing names to CoolProp and PYroMat names."""

    @staticmethod
    def normalize_name(value: str) -> str:
        """Normalize a user name, alias, or canonical name for lookup."""
        return _normalize_key(value)

    @classmethod
    def name(cls, value: str) -> str:
        """Return the canonical registry name for a user name or alias."""
        lookup = cls.normalize_name(value)

        try:
            return _NAME_LOOKUP[lookup]
        except KeyError:
            raise ValueError(f"Unknown fluid/species name: {value!r}")

    @classmethod
    def record(cls, value: str) -> SpeciesRecord:
        """Return the full registry record for a user name or alias."""
        return SPECIES_DATABASE[cls.name(value)]

    @classmethod
    def coolprop_name(cls, value: str) -> str:
        """Return the CoolProp backend name for a user name or alias."""
        record = cls.record(value)

        if record.coolprop is None:
            raise ValueError(
                f"{record.name!r} is not supported by CoolProp."
            )

        return record.coolprop

    @classmethod
    def pyromat_name(
        cls,
        value: str,
        include_prefix: bool = False,
    ) -> str:
        """Return the PYroMat species name for a user name or alias."""
        record = cls.record(value)

        if record.pyromat is None:
            raise ValueError(
                f"{record.name!r} is not supported by PYroMat."
            )

        if include_prefix:
            return f"ig.{record.pyromat}"

        return record.pyromat

    @classmethod
    def supports_coolprop(cls, value: str) -> bool:
        """Return True if the species has a CoolProp backend mapping."""
        try:
            return cls.record(value).coolprop is not None
        except ValueError:
            return False

    @classmethod
    def supports_pyromat(cls, value: str) -> bool:
        """Return True if the species has a PYroMat backend mapping."""
        try:
            return cls.record(value).pyromat is not None
        except ValueError:
            return False

    @classmethod
    def supports_both(cls, value: str) -> bool:
        """Return True if the species is available in both backends."""
        return (
            cls.supports_coolprop(value)
            and cls.supports_pyromat(value)
        )

    @classmethod
    def add_alias(cls, alias: str, name: str) -> None:
        """Add a user alias and refresh the lookup cache."""
        global _NAME_LOOKUP
        ALIASES[alias] = cls.name(name)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def remove_alias(cls, alias: str) -> None:
        """Remove a user alias and refresh the lookup cache."""
        global _NAME_LOOKUP
        ALIASES.pop(alias, None)
        _NAME_LOOKUP = _build_name_lookup()

    @classproperty
    def names(cls) -> list[str]:
        """Return all canonical registry names."""
        return sorted(SPECIES_DATABASE.keys())

    @classproperty
    def coolprop_supported_names(cls) -> list[str]:
        """Return canonical names with CoolProp support."""
        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if record.coolprop is not None
        )

    @classproperty
    def pyromat_supported_names(cls) -> list[str]:
        """Return canonical names with PYroMat support."""
        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if record.pyromat is not None
        )

    @classproperty
    def supports_both_names(cls) -> list[str]:
        """Return canonical names supported by both backends."""
        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if (
                record.coolprop is not None
                and record.pyromat is not None
            )
        )

    @classproperty
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the user alias table."""
        return dict(sorted(ALIASES.items()))

    @classmethod
    def show_species(cls) -> list[str]:
        """Print and return all canonical registry names."""
        for name in cls.names:
            print(name)

        return cls.names
