from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

@dataclass(frozen=True)
class SpeciesRecord:
    """Canonical species entry and backend-specific names."""

    name: str
    coolprop: str | None = None
    pyromat: str | None = None  # no "ig." prefix
    cea: str | None = None

SPECIES_DATABASE = MappingProxyType({
    # ---------- Common gases / shared ----------
    "Air": SpeciesRecord("Air", coolprop="Air", pyromat="air", cea="Air"),
    "Argon": SpeciesRecord("Argon", coolprop="Argon", pyromat="Ar", cea="Ar"),
    "CarbonDioxide": SpeciesRecord("CarbonDioxide", coolprop="CarbonDioxide", pyromat="CO2", cea="CO2"),
    "CarbonMonoxide": SpeciesRecord("CarbonMonoxide", coolprop="CarbonMonoxide", pyromat="CO", cea="CO"),
    "Helium": SpeciesRecord("Helium", coolprop="Helium", pyromat="He", cea="He"),
    "Hydrogen": SpeciesRecord("Hydrogen", coolprop="Hydrogen", pyromat="H2", cea="H2"),
    "Methane": SpeciesRecord("Methane", coolprop="Methane", pyromat="CH4", cea="CH4"),
    "Nitrogen": SpeciesRecord("Nitrogen", coolprop="Nitrogen", pyromat="N2", cea="N2"),
    "Oxygen": SpeciesRecord("Oxygen", coolprop="Oxygen", pyromat="O2", cea="O2"),
    "Water": SpeciesRecord("Water", coolprop="Water", pyromat="H2O", cea="H2O"),

    # ---------- Common fluids with PYroMat mappings ----------
    "Ammonia": SpeciesRecord("Ammonia", coolprop="Ammonia", pyromat="NH3", cea="NH3"),
    "Ethane": SpeciesRecord("Ethane", coolprop="Ethane", pyromat="C2H6", cea="C2H6"),
    "Ethylene": SpeciesRecord("Ethylene", coolprop="Ethylene", pyromat="C2H4", cea="C2H4"),
    "n-Propane": SpeciesRecord("n-Propane", coolprop="n-Propane", pyromat="C3H8", cea="C3H8"),
    "n-Butane": SpeciesRecord("n-Butane", coolprop="n-Butane", pyromat="C4H10", cea="C4H10"),
    "IsoButane": SpeciesRecord("IsoButane", coolprop="IsoButane", pyromat=None),
    "Benzene": SpeciesRecord("Benzene", coolprop="Benzene", pyromat="C6H6", cea="C6H6"),
    "Toluene": SpeciesRecord("Toluene", coolprop="Toluene", pyromat="C7H8", cea="C7H8"),
    "Methanol": SpeciesRecord("Methanol", coolprop="Methanol", pyromat="CH4O", cea="CH3OH"),
    "Ethanol": SpeciesRecord("Ethanol", coolprop="Ethanol", pyromat="C2H6O", cea="C2H5OH"),
    "NitrousOxide": SpeciesRecord("NitrousOxide", coolprop="NitrousOxide", pyromat="N2O", cea="N2O"),
    "HydrogenChloride": SpeciesRecord("HydrogenChloride", coolprop="HydrogenChloride", pyromat="HCl", cea="HCl"),
    "HydrogenSulfide": SpeciesRecord("HydrogenSulfide", coolprop="HydrogenSulfide", pyromat="H2S", cea="H2S"),
    "SulfurDioxide": SpeciesRecord("SulfurDioxide", coolprop="SulfurDioxide", pyromat="SO2", cea="SO2"),
    "SulfurHexafluoride": SpeciesRecord("SulfurHexafluoride", coolprop="SulfurHexafluoride", pyromat="SF6", cea="SF6"),
    "Neon": SpeciesRecord("Neon", coolprop="Neon", pyromat="Ne", cea="Ne"),
    "Krypton": SpeciesRecord("Krypton", coolprop="Krypton", pyromat="Kr", cea="Kr"),
    "Xenon": SpeciesRecord("Xenon", coolprop="Xenon", pyromat="Xe", cea="Xe"),

    # ---------- CoolProp pure / pseudo-pure fluids ----------
    "1-Butene": SpeciesRecord("1-Butene", coolprop="1-Butene", pyromat=None),
    "Acetone": SpeciesRecord("Acetone", coolprop="Acetone", pyromat=None, cea="CH3COCH3"),
    "CarbonylSulfide": SpeciesRecord("CarbonylSulfide", coolprop="CarbonylSulfide", pyromat=None, cea="COS"),
    "CycloHexane": SpeciesRecord("CycloHexane", coolprop="CycloHexane", pyromat=None, cea="C6H12"),
    "CycloPropane": SpeciesRecord("CycloPropane", coolprop="CycloPropane", pyromat=None, cea="C3H6,cyclo-"),
    "Cyclopentane": SpeciesRecord("Cyclopentane", coolprop="Cyclopentane", pyromat=None, cea="C5H10,cyclo-"),
    "D4": SpeciesRecord("D4", coolprop="D4", pyromat=None),
    "D5": SpeciesRecord("D5", coolprop="D5", pyromat=None),
    "D6": SpeciesRecord("D6", coolprop="D6", pyromat=None),
    "Deuterium": SpeciesRecord("Deuterium", coolprop="Deuterium", pyromat=None, cea="D2"),
    "Dichloroethane": SpeciesRecord("Dichloroethane", coolprop="Dichloroethane", pyromat=None),
    "DiethylEther": SpeciesRecord("DiethylEther", coolprop="DiethylEther", pyromat=None),
    "DimethylCarbonate": SpeciesRecord("DimethylCarbonate", coolprop="DimethylCarbonate", pyromat=None),
    "DimethylEther": SpeciesRecord("DimethylEther", coolprop="DimethylEther", pyromat=None, cea="CH3OCH3"),
    "EthylBenzene": SpeciesRecord("EthylBenzene", coolprop="EthylBenzene", pyromat=None, cea="C6H5C2H5"),
    "EthyleneOxide": SpeciesRecord("EthyleneOxide", coolprop="EthyleneOxide", pyromat=None, cea="C2H4O,ethylen-o"),
    "Fluorine": SpeciesRecord("Fluorine", coolprop="Fluorine", pyromat=None, cea="F2"),
    "HFE143m": SpeciesRecord("HFE143m", coolprop="HFE143m", pyromat=None),
    "HeavyWater": SpeciesRecord("HeavyWater", coolprop="HeavyWater", pyromat=None, cea="D2O"),
    "IsoButene": SpeciesRecord("IsoButene", coolprop="IsoButene", pyromat=None, cea="C4H8,isobutene"),
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
    "OrthoHydrogen": SpeciesRecord("OrthoHydrogen", coolprop="OrthoHydrogen", pyromat=None, cea="H2"),
    "ParaDeuterium": SpeciesRecord("ParaDeuterium", coolprop="ParaDeuterium", pyromat=None),
    "ParaHydrogen": SpeciesRecord("ParaHydrogen", coolprop="ParaHydrogen", pyromat=None, cea="H2"),
    "Propylene": SpeciesRecord("Propylene", coolprop="Propylene", pyromat=None, cea="C3H6,propylene"),
    "Propyne": SpeciesRecord("Propyne", coolprop="Propyne", pyromat=None, cea="C3H4,propyne"),

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
    "n-Decane": SpeciesRecord("n-Decane", coolprop="n-Decane", pyromat=None, cea="C10H22"),
    "n-Dodecane": SpeciesRecord("n-Dodecane", coolprop="n-Dodecane", pyromat=None, cea="C12H26"),
    "n-Heptane": SpeciesRecord("n-Heptane", coolprop="n-Heptane", pyromat=None, cea="C7H16,n-heptane"),
    "n-Hexane": SpeciesRecord("n-Hexane", coolprop="n-Hexane", pyromat=None, cea="C6H14,n-hexane"),
    "n-Nonane": SpeciesRecord("n-Nonane", coolprop="n-Nonane", pyromat=None, cea="C9H20"),
    "n-Octane": SpeciesRecord("n-Octane", coolprop="n-Octane", pyromat=None, cea="C8H18,isooctane"),
    "n-Pentane": SpeciesRecord("n-Pentane", coolprop="n-Pentane", pyromat=None, cea="C5H12,n-pentane"),
    "n-Undecane": SpeciesRecord("n-Undecane", coolprop="n-Undecane", pyromat=None, cea="C11H24"),

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

# Additional CEA/PYroMat-friendly gas aliases. These preserve the canonical
# ThermoProp names while allowing common chemical formulas and phase words.
ALIASES.update({
    "gaseous oxygen": "Oxygen",
    "liquid oxygen": "Oxygen",
    "oxygen gas": "Oxygen",
    "oxygen liquid": "Oxygen",
    "gaseous hydrogen": "Hydrogen",
    "liquid hydrogen": "Hydrogen",
    "hydrogen gas": "Hydrogen",
    "hydrogen liquid": "Hydrogen",
    "gaseous nitrogen": "Nitrogen",
    "liquid nitrogen": "Nitrogen",
    "nitrogen gas": "Nitrogen",
    "nitrogen liquid": "Nitrogen",
    "gaseous methane": "Methane",
    "liquid methane": "Methane",
    "l-methane": "Methane",
    "gch4": "Methane",
    "gox": "Oxygen",
    "gh2": "Hydrogen",
    "gn2": "Nitrogen",
    "gch4": "Methane",
    "argon gas": "Argon",
    "helium gas": "Helium",
    "neon gas": "Neon",
    "krypton gas": "Krypton",
    "xenon gas": "Xenon",
    "n2o gas": "NitrousOxide",
    "nitrous": "NitrousOxide",
    "carbonmonoxide": "CarbonMonoxide",
    "carbondioxide": "CarbonDioxide",
    "sulfur dioxide": "SulfurDioxide",
    "sulfur-dioxide": "SulfurDioxide",
    "sulfurdioxide": "SulfurDioxide",
    "sulfur hexafluoride": "SulfurHexafluoride",
    "sulfur-hexafluoride": "SulfurHexafluoride",
    "sulfurhexafluoride": "SulfurHexafluoride",
    "hcl": "HydrogenChloride",
    "hydrogen chloride": "HydrogenChloride",
    "hydrogen-chloride": "HydrogenChloride",
    "h2s": "HydrogenSulfide",
    "hydrogen sulfide": "HydrogenSulfide",
    "hydrogen-sulfide": "HydrogenSulfide",
    "so2": "SulfurDioxide",
    "sf6": "SulfurHexafluoride",
    "f2": "Fluorine",
    "fluorine": "Fluorine",
    "fluorine gas": "Fluorine",
    "d2": "Deuterium",
    "deuterium": "Deuterium",
    "d2o": "HeavyWater",
    "heavy water": "HeavyWater",
    "heavy-water": "HeavyWater",
    "methanol": "Methanol",
    "ch3oh": "Methanol",
    "methyl alcohol": "Methanol",
    "ethanol": "Ethanol",
    "c2h5oh": "Ethanol",
    "ethyl alcohol": "Ethanol",
    "benzene": "Benzene",
    "c6h6": "Benzene",
    "toluene": "Toluene",
    "c7h8": "Toluene",
    "acetone": "Acetone",
    "dimethyl ether": "DimethylEther",
    "dme": "DimethylEther",
    "ethylene oxide": "EthyleneOxide",
    "propylene": "Propylene",
    "propyne": "Propyne",
    "cyclohexane": "CycloHexane",
    "cyclopentane": "Cyclopentane",
    "cyclopropane": "CycloPropane",
})

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
    """
    User-facing fluid/species registry for ThermoProp.

    The registry gives users simple names such as ``"water"``, ``"gn2"``,
    ``"rp-1"``, or ``"lox"`` and maps them to the correct backend names for:

        Fluid      -> CoolProp
        IdealGas   -> PYroMat
        CEA        -> NASA CEA / CEAM data

    CEA mappings are kept because IdealGas can use CEA species names for
    transport-property lookup. Propellant and combustion-gas mappings belong
    in CombustionRegistry.
    """

    _BACKEND_ALIASES = {
        "coolprop": "coolprop",
        "cool-prop": "coolprop",
        "cp": "coolprop",
        "fluid": "coolprop",

        "pyromat": "pyromat",
        "pyro-mat": "pyromat",
        "pm": "pyromat",
        "idealgas": "pyromat",
        "ideal-gas": "pyromat",
        "ideal_gas": "pyromat",

        "cea": "cea",
        "ceam": "cea",
        "nasa": "cea",
        "nasa-cea": "cea",
        "nasacea": "cea",
        "chem-equilibrium": "cea",
        "chemical-equilibrium": "cea",

    }

    @staticmethod
    def normalize_name(value: str) -> str:
        """Normalize a user name, alias, or canonical name for lookup."""
        return _normalize_key(value)

    @classmethod
    def normalize_backend(cls, backend: str) -> str:
        """
        Normalize a backend name.

        Accepted examples include ``"coolprop"``, ``"cp"``, ``"fluid"``,
        ``"pyromat"``, ``"pm"``, and ``"idealgas"``.
        """
        lookup = cls.normalize_name(backend)

        try:
            return cls._BACKEND_ALIASES[lookup]
        except KeyError:
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected one of: "
                "'coolprop', 'pyromat', or 'cea'."
            )

    @classmethod
    def name(cls, value: str) -> str:
        """Return the canonical ThermoProp registry name for a user name or alias."""
        lookup = cls.normalize_name(value)

        try:
            return _NAME_LOOKUP[lookup]
        except KeyError:
            raise ValueError(f"Unknown fluid/species name: {value!r}")

    @classmethod
    def record(cls, value: str) -> SpeciesRecord:
        """Return the full registry record for a user name or general alias."""
        return SPECIES_DATABASE[cls.name(value)]

    @classmethod
    def backend_name(cls, value: str, backend: str, include_prefix: bool = False) -> str:
        """
        Return the backend-specific name for a user name or alias.

        Parameters
        ----------
        value:
            User name, alias, or canonical registry name.
        backend:
            Backend selector. Accepted examples include ``"coolprop"``,
            ``"pyromat"`` and ``"cea"``.
        include_prefix:
            If True and backend is PYroMat, return names with the ``"ig."``
            prefix, such as ``"ig.N2"``.
        """
        backend = cls.normalize_backend(backend)

        if backend == "coolprop":
            return cls.coolprop_name(value)

        if backend == "pyromat":
            return cls.pyromat_name(value, include_prefix=include_prefix)

        return cls.cea_name(value)

    @classmethod
    def coolprop_name(cls, value: str) -> str:
        """Return the CoolProp backend name for a user name or alias."""
        record = cls.record(value)

        if record.coolprop is None:
            raise ValueError(f"{record.name!r} is not supported by CoolProp.")

        return record.coolprop

    @classmethod
    def pyromat_name(cls, value: str, include_prefix: bool = False) -> str:
        """Return the PYroMat species name for a user name or alias."""
        record = cls.record(value)

        if record.pyromat is None:
            raise ValueError(f"{record.name!r} is not supported by PYroMat.")

        if include_prefix:
            return f"ig.{record.pyromat}"

        return record.pyromat

    @classmethod
    def cea_name(cls, value: str) -> str:
        """Return the NASA CEA / CEAM species name for a user name or alias."""
        record = cls.record(value)

        if record.cea is None:
            raise ValueError(f"{record.name!r} is not supported by NASA CEA data.")

        return record.cea

    @classmethod
    def supports(cls, value: str, backend: str) -> bool:
        """Return True if ``value`` is supported by the selected backend."""
        backend = cls.normalize_backend(backend)

        if backend == "coolprop":
            return cls.supports_coolprop(value)

        if backend == "pyromat":
            return cls.supports_pyromat(value)

        return cls.supports_cea(value)

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
    def supports_cea(cls, value: str) -> bool:
        """Return True if the species has a NASA CEA / CEAM mapping."""
        try:
            return cls.record(value).cea is not None
        except ValueError:
            return False

    @classmethod
    def supports_both(cls, value: str) -> bool:
        """Return True if the species is available in both CoolProp and PYroMat."""
        return cls.supports_coolprop(value) and cls.supports_pyromat(value)

    @classmethod
    def add_alias(cls, alias: str, name: str) -> None:
        """
        Add a general alias for Fluid and IdealGas lookups.

        Examples
        --------
        FluidRegistry.add_alias("my-water", "Water")
        FluidRegistry.add_alias("my-rp1-surrogate", "n-Dodecane")
        """
        global _NAME_LOOKUP
        ALIASES[alias] = cls.name(name)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def add_backend_alias(cls, alias: str, name: str, backend: str) -> None:
        """
        Add an alias for a specific backend.

        This is a convenience wrapper around :meth:`add_alias`.
        """
        backend = cls.normalize_backend(backend)

        cls.add_alias(alias, name)

    @classmethod
    def remove_alias(cls, alias: str) -> None:
        """Remove a general Fluid/IdealGas alias and refresh the lookup cache."""
        global _NAME_LOOKUP
        ALIASES.pop(alias, None)
        _NAME_LOOKUP = _build_name_lookup()

    @classmethod
    def remove_backend_alias(cls, alias: str, backend: str) -> None:
        """Remove an alias from a specific backend alias table."""
        backend = cls.normalize_backend(backend)

        cls.remove_alias(alias)

    @classmethod
    def describe(cls, value: str) -> dict[str, str | None | bool]:
        """
        Return a compact description of a general registry entry.

        Use this when users want to check what a name maps to for Fluid and
        IdealGas.
        """
        record = cls.record(value)

        return {
            "input": value,
            "name": record.name,
            "coolprop": record.coolprop,
            "pyromat": record.pyromat,
            "cea": record.cea,
            "supports_coolprop": record.coolprop is not None,
            "supports_pyromat": record.pyromat is not None,
            "supports_cea": record.cea is not None,
        }

    @classmethod
    def supported_names(cls, backend: str) -> list[str]:
        """Return canonical registry names supported by the selected backend."""
        backend = cls.normalize_backend(backend)

        if backend == "coolprop":
            return cls.coolprop_supported_names

        if backend == "pyromat":
            return cls.pyromat_supported_names

        return cls.cea_supported_names

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
    def cea_supported_names(cls) -> list[str]:
        """Return canonical names with NASA CEA / CEAM support."""
        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if record.cea is not None
        )

    @classproperty
    def supports_both_names(cls) -> list[str]:
        """Return canonical names supported by both CoolProp and PYroMat."""
        return sorted(
            name
            for name, record in SPECIES_DATABASE.items()
            if record.coolprop is not None and record.pyromat is not None
        )

    @classproperty
    def aliases(cls) -> dict[str, str]:
        """Return a copy of the general Fluid/IdealGas alias table."""
        return dict(sorted(ALIASES.items()))

    @classmethod
    def show_species(cls) -> list[str]:
        """Print and return all canonical registry names."""
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
        """Print and return general Fluid/IdealGas aliases."""
        aliases = cls.aliases

        if not aliases:
            return aliases

        width = max(len(alias) for alias in aliases)

        print("Fluid / IdealGas Aliases")
        print("-" * (width + 24))

        for alias, name in aliases.items():
            print(f"{alias:<{width}} -> {name}")

        return aliases

    @classmethod
    def show_backend_names(cls, value: str) -> dict:
        """Return Fluid / IdealGas backend mappings for a name or alias."""
        record = cls.record(value)

        return {
            "input": value,
            "canonical": record.name,
            "coolprop": record.coolprop,
            "pyromat": record.pyromat,
            "cea": record.cea,
        }

