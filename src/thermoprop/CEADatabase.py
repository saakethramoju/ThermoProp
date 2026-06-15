from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CEASpecies:
    name: str
    index: int
    molar_mass: float
    has_thermo: bool
    has_transport: bool
    is_reactant: bool
    elemental_composition: dict[str, float]
    heat_of_formation_molar: float | None
    temperature_ranges: list[tuple[float, float]]
    is_condensed: bool
    is_gas: bool


class CEADatabase:
    """
    Strict-name interface to parser-generated NASA CEA / CEAM data.

    CEADatabase stores thermodynamic, transport, elemental-composition, phase, and
    reactant metadata parsed from CEA-style source data.

    This database intentionally does not apply user aliases. Names must match the
    parsed CEA database exactly, except for leading or trailing whitespace. Use
    SpeciesDatabase for user-friendly ThermoProp aliases and cross-backend name
    resolution.

    Data provided
    -------------

    CEADatabase exposes:

    - strict CEA names
    - gas product species
    - condensed species
    - predefined reactant cards
    - elemental composition
    - molecular weight
    - heat of formation
    - NASA-9 thermodynamic coefficients
    - standard-state Cp, h, and s
    - transport data when available

    Name inspection
    ---------------

    Use:

        CEA.names
        CEA.gas_species
        CEA.condensed_species
        CEA.reactant_names
        CEA.transport_names
        CEA.find_species("text")
        CEA.find_transport_species("text")

    Thermodynamic convention
    ------------------------

    CEA thermodynamic functions use the CEA reference basis and NASA polynomial
    conventions. Absolute enthalpy and entropy should be interpreted consistently
    with the selected thermodynamic model.

    Public API units are SI unless a method name explicitly indicates molar or
    kmol-based CEA units.
    """

    _RU = 8.31446261815324
    _RU_KMOL = 8314.46261815324

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            data_dir = _ROOT / "cea_data"

        self.data_dir = Path(data_dir)

        self.thermo_path = self.data_dir / "thermo_ceam.npz"
        self.thermo_index_path = self.data_dir / "thermo_name_index.json"
        self.transport_path = self.data_dir / "trans_ceam.npz"
        self.transport_index_path = self.data_dir / "trans_name_index.json"

        self._thermo = self._load_npz_required(self.thermo_path)
        self._thermo_index = self._load_json_required(self.thermo_index_path)

        if self.transport_path.exists() and self.transport_index_path.exists():
            self._transport = self._load_npz_required(self.transport_path)
            self._transport_index = self._load_json_required(self.transport_index_path)
        else:
            self._transport = None
            self._transport_index = {}

        self._names_cache: list[str] | None = None
        self._transport_names_cache: list[str] | None = None
        self._product_names_cache: list[str] | None = None
        self._reactant_names_cache: list[str] | None = None
        self._gas_species_cache: list[str] | None = None
        self._condensed_species_cache: list[str] | None = None
        self._all_elements_cache: set[str] | None = None

    @staticmethod
    def _load_npz_required(path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"CEA database file not found: {path}")

        npz = np.load(path, allow_pickle=False)
        data = {key: npz[key] for key in npz.files}
        npz.close()
        return data

    @staticmethod
    def _load_json_required(path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"CEA index file not found: {path}")

        with open(path, "r") as f:
            return {str(k): int(v) for k, v in json.load(f).items()}

    def resolve_name(self, name: str) -> str:
        """
        Return the exact CEA database name.

        No aliases are applied. For example, "Jet-A" will only work if "Jet-A"
        exists directly in the parsed CEA database.
        """
        raw = str(name).strip()

        if raw in self._thermo_index:
            return raw

        raise ValueError(
            f"Unknown CEA species or reactant: {name!r}. "
            "CEA names are strict. Use CEA.show_species(), CEA.names, "
            "or CEA.find_species(...) to inspect available names."
        )

    @property
    def names(self) -> list[str]:
        """Return every strict CEA database name."""
        if self._names_cache is None:
            self._names_cache = sorted(self._thermo_index.keys())

        return list(self._names_cache)

    @property
    def species_names(self) -> list[str]:
        """Alias-free list of every parsed CEA name."""
        return self.names

    @property
    def transport_names(self) -> list[str]:
        """Return every strict CEA name with transport data."""
        if self._transport_names_cache is None:
            self._transport_names_cache = sorted(self._transport_index.keys())

        return list(self._transport_names_cache)

    @property
    def product_names(self) -> list[str]:
        """Return species with NASA polynomial thermo data."""
        if self._product_names_cache is None:
            self._product_names_cache = sorted(
                name for name in self.names if self.has_thermo(name)
            )

        return list(self._product_names_cache)

    @property
    def reactant_names(self) -> list[str]:
        """Return predefined CEA reactant cards."""
        if self._reactant_names_cache is None:
            self._reactant_names_cache = sorted(
                name for name in self.names if self.is_reactant(name)
            )

        return list(self._reactant_names_cache)

    @property
    def product_species(self) -> list[str]:
        return self.product_names

    @property
    def gas_species(self) -> list[str]:
        if self._gas_species_cache is None:
            self._gas_species_cache = sorted(
                name for name in self.product_names if self.is_gas(name)
            )

        return list(self._gas_species_cache)

    @property
    def condensed_species(self) -> list[str]:
        if self._condensed_species_cache is None:
            self._condensed_species_cache = sorted(
                name for name in self.names
                if self.is_condensed(name)
            )

        return list(self._condensed_species_cache)

    @property
    def predefined_reactants(self) -> list[str]:
        return self.reactant_names

    @property
    def all_elements(self) -> set[str]:
        if self._all_elements_cache is None:
            elements: set[str] = set()

            for name in self.names:
                elements.update(self.element_set(name))

            self._all_elements_cache = elements

        return set(self._all_elements_cache)

    def find_species(self, text: str, *, case_sensitive: bool = False) -> list[str]:
        """
        Search available CEA names by substring.

        This does not resolve aliases. It only helps inspect strict database names.
        """
        query = str(text)

        if case_sensitive:
            return sorted(name for name in self.names if query in name)

        query = query.lower()
        return sorted(name for name in self.names if query in name.lower())

    def find_transport_species(self, text: str, *, case_sensitive: bool = False) -> list[str]:
        query = str(text)

        if case_sensitive:
            return sorted(name for name in self.transport_names if query in name)

        query = query.lower()
        return sorted(name for name in self.transport_names if query in name.lower())

    def has_species(self, name: str) -> bool:
        return str(name).strip() in self._thermo_index

    def has_transport(self, name: str) -> bool:
        name = self.resolve_name(name)
        return name in self._transport_index

    def index(self, name: str) -> int:
        name = self.resolve_name(name)
        return int(self._thermo_index[name])

    def transport_index(self, name: str) -> int:
        raw = str(name).strip()

        if raw in self._transport_index:
            return int(self._transport_index[raw])

        name = self.resolve_name(raw)

        try:
            return int(self._transport_index[name])
        except KeyError:
            raise ValueError(f"CEA transport data are not available for {name!r}")

    def raw(self, key: str, name: str) -> Any:
        return self._thermo[key][self.index(name)]

    def raw_by_index(self, key: str, index: int) -> Any:
        return self._thermo[key][int(index)]

    def species(self, name: str) -> CEASpecies:
        name = self.resolve_name(name)
        idx = self.index(name)

        return CEASpecies(
            name=name,
            index=idx,
            molar_mass=self.molar_mass(name),
            has_thermo=self.has_thermo(name),
            has_transport=self.has_transport(name),
            is_reactant=self.is_reactant(name),
            elemental_composition=self.elemental_composition(name),
            heat_of_formation_molar=self.heat_of_formation_molar(name),
            temperature_ranges=self.temperature_ranges(name),
            is_condensed=self.is_condensed(name),
            is_gas=self.is_gas(name),
        )

    def describe(self, name: str) -> dict[str, Any]:
        s = self.species(name)

        return {
            "name": s.name,
            "index": s.index,
            "molar_mass": s.molar_mass,
            "has_thermo": s.has_thermo,
            "has_transport": s.has_transport,
            "is_reactant": s.is_reactant,
            "elemental_composition": s.elemental_composition,
            "heat_of_formation_molar": s.heat_of_formation_molar,
            "temperature_ranges": s.temperature_ranges,
            "is_condensed": s.is_condensed,
            "is_gas": s.is_gas,
        }

    def molar_mass(self, name: str) -> float:
        return float(self.raw("mw", name)) / 1000.0

    def molecular_weight(self, name: str) -> float:
        return float(self.raw("mw", name))

    def heat_of_formation_molar(self, name: str) -> float | None:
        value = float(self.raw("hf298", name))
        return value if np.isfinite(value) else None

    def elemental_composition(self, name: str) -> dict[str, float]:
        symbols = self.raw("element_symbols", name)
        counts = self.raw("element_counts", name)

        out: dict[str, float] = {}

        for symbol, count in zip(symbols, counts):
            symbol = str(symbol).strip()

            if not symbol or not np.isfinite(count):
                continue

            out[symbol] = float(count)

        return out

    def n_intervals(self, name: str) -> int:
        return int(self.raw("n_intervals", name))

    def has_thermo(self, name: str) -> bool:
        idx = self.index(name)
        n = int(self._thermo["n_intervals"][idx])

        if n <= 0:
            return False

        coeffs = self._thermo["coeffs"][idx]
        return not bool(np.isnan(coeffs).all())

    def is_reactant(self, name: str) -> bool:
        name = self.resolve_name(name)

        comment = self.reactant_comment(name).lower()

        if "react" in comment:
            return True

        return not self.has_thermo(name)
            
    def reactant_comment(self, name: str) -> str:
        idx = self.index(name)
        return str(self._thermo["comments"][idx]).strip()

    @staticmethod
    def _phase_label_from_name(name: str) -> str | None:
        raw = str(name).strip()

        start = raw.find("(")
        if start < 0:
            return None

        end = raw.find(")", start + 1)
        if end < 0:
            return None

        return raw[start + 1:end].strip()

    @staticmethod
    def _is_condensed_phase_label(label: str | None) -> bool:
        if not label:
            return False

        known_labels = {
            "L", "l", "liq", "liquid",
            "cr", "cr1", "cr2",
            "gr", "graphite",
            "s", "solid",
            "a", "b", "c", "d", "e",
            "alpha", "beta", "gamma", "delta",
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        }

        return label in known_labels

    def phase_label(self, name: str) -> str | None:
        name = self.resolve_name(name)
        return self._phase_label_from_name(name)
        
    def is_condensed_phase_name(self, name: str) -> bool:
        name = self.resolve_name(name)
        return self._is_condensed_phase_label(
            self.phase_label(name)
        )

    def is_condensed(self, name: str) -> bool:
        name = self.resolve_name(name)
        return self.is_condensed_phase_name(name)

    def is_gas(self, name: str) -> bool:
        name = self.resolve_name(name)
        return not self.is_condensed(name)

    def element_set(self, name: str) -> set[str]:
        return set(self.elemental_composition(name).keys())

    def temperature_ranges(self, name: str) -> list[tuple[float, float]]:
        idx = self.index(name)
        n = int(self._thermo["n_intervals"][idx])

        if n <= 0:
            return []

        ranges = self._thermo["t_ranges"][idx, :n]

        return [
            (float(tmin), float(tmax))
            for tmin, tmax in ranges
            if np.isfinite(tmin) and np.isfinite(tmax)
        ]

    def temperature_limits(self, names: list[str]) -> tuple[float, float]:
        mins = []
        maxs = []

        for name in names:
            ranges = self.temperature_ranges(name)

            if not ranges:
                raise ValueError(f"{name!r} has no valid NASA polynomial ranges.")

            mins.append(min(r[0] for r in ranges))
            maxs.append(max(r[1] for r in ranges))

        return max(mins), min(maxs)

    def interval_index(
        self,
        name: str,
        temperature: float,
        *,
        extrapolation_margin: float = 50.0,
    ) -> int:
        name = self.resolve_name(name)
        idx = self.index(name)
        n = int(self._thermo["n_intervals"][idx])
        T = float(temperature)

        if n <= 0:
            raise ValueError(f"{name!r} has no valid NASA polynomial ranges.")

        ranges = self._thermo["t_ranges"][idx, :n]

        for j in range(n):
            Tmin, Tmax = ranges[j]
            if Tmin <= T <= Tmax:
                return int(j)

        # CEA behavior: warn out of range, but still evaluate the polynomial.
        # Use the nearest interval, but keep the actual requested T.
        first_Tmin = float(ranges[0, 0])
        if first_Tmin - 25.0 <= T < first_Tmin:
            return 0

        last_Tmax = float(ranges[n - 1, 1])
        if last_Tmax < T <= last_Tmax + 25.0:
            return int(n - 1)

        # CEA behavior: use nearest interval outside range.
        first_Tmin = float(ranges[0, 0])
        if T < first_Tmin:
            return 0

        last_Tmax = float(ranges[n - 1, 1])
        if T > last_Tmax:
            return int(n - 1)

        raise ValueError(
            f"No CEA polynomial interval for {name!r} at T={T:.6g} K."
        )



    def nasa9_coefficients(self, name: str, temperature: float) -> np.ndarray:
        name = self.resolve_name(name)

        if not self.has_thermo(name):
            raise ValueError(
                f"{name!r} has no NASA-9 thermo polynomial data. "
                "It is probably a CEA reactant definition."
            )

        idx = self.index(name)
        j = self.interval_index(name, temperature)
        return np.asarray(self._thermo["coeffs"][idx, j], dtype=float)

    def thermo_molar(self, name: str, temperature: float) -> tuple[float, float, float]:
        T = float(temperature)
        coeffs = self.nasa9_coefficients(name, T)

        if len(coeffs) == 10:
            a0, a1, a2, a3, a4, a5, a6, unused, b1, b2 = coeffs
        elif len(coeffs) == 9:
            a0, a1, a2, a3, a4, a5, a6, b1, b2 = coeffs
        else:
            raise ValueError(
                f"{name!r} has {len(coeffs)} thermo coefficients; expected 9 or 10."
            )

        lnT = np.log(T)

        cp_over_R = (
            a0 / T**2
            + a1 / T
            + a2
            + a3 * T
            + a4 * T**2
            + a5 * T**3
            + a6 * T**4
        )

        h_over_RT = (
            -a0 / T**2
            + a1 * lnT / T
            + a2
            + a3 * T / 2.0
            + a4 * T**2 / 3.0
            + a5 * T**3 / 4.0
            + a6 * T**4 / 5.0
            + b1 / T
        )

        s_over_R = (
            -a0 / (2.0 * T**2)
            - a1 / T
            + a2 * lnT
            + a3 * T
            + a4 * T**2 / 2.0
            + a5 * T**3 / 3.0
            + a6 * T**4 / 4.0
            + b2
        )

        cp = cp_over_R * self._RU_KMOL
        h = h_over_RT * self._RU_KMOL * T
        s0 = s_over_R * self._RU_KMOL

        return float(cp), float(h), float(s0)

    def cp_molar(self, name: str, temperature: float) -> float:
        return self.thermo_molar(name, temperature)[0]

    def enthalpy_molar(self, name: str, temperature: float) -> float:
        return self.thermo_molar(name, temperature)[1]

    def entropy_molar_standard(self, name: str, temperature: float) -> float:
        return self.thermo_molar(name, temperature)[2]

    def thermo_mass(self, name: str, temperature: float) -> tuple[float, float, float]:
        mw = self.molecular_weight(name)
        cp, h, s0 = self.thermo_molar(name, temperature)
        return cp / mw, h / mw, s0 / mw

    def cp_mass(self, name: str, temperature: float) -> float:
        return self.thermo_mass(name, temperature)[0]

    def enthalpy_mass(self, name: str, temperature: float) -> float:
        return self.thermo_mass(name, temperature)[1]

    def entropy_mass_standard(self, name: str, temperature: float) -> float:
        return self.thermo_mass(name, temperature)[2]

    def _require_transport_loaded(self) -> None:
        if self._transport is None:
            raise FileNotFoundError(
                "CEA transport files were not found. Expected "
                f"{self.transport_path} and {self.transport_index_path}."
            )
        

    def transport_key(self, name1: str, name2: str) -> str:
        name1 = self.resolve_name(name1)
        name2 = self.resolve_name(name2)

        key12 = f"{name1}|{name2}"
        key21 = f"{name2}|{name1}"

        if key12 in self._transport_index:
            return key12

        if key21 in self._transport_index:
            return key21

        raise ValueError(
            f"CEA binary transport data are not available for "
            f"{name1!r} and {name2!r}."
        )

    def has_binary_transport(self, name1: str, name2: str) -> bool:
        try:
            self.transport_key(name1, name2)
            return True
        except Exception:
            return False

    def binary_transport_fit(
        self,
        name1: str,
        name2: str,
        temperature: float,
    ) -> float:
        key = self.transport_key(name1, name2)
        return self.transport_fit(key, temperature, "viscosity")

    def binary_viscosity_interaction(
        self,
        name1: str,
        name2: str,
        temperature: float,
    ) -> float:
        """
        Return CEA binary eta_ij interaction parameter [Pa-s].

        CEA stores binary interaction parameters with the same fit form and
        viscosity units as pure viscosity: micropoise.
        """
        eta_micro_poise = self.binary_transport_fit(
            name1,
            name2,
            temperature,
        )

        return eta_micro_poise * 1e-7

    def transport_temperature_ranges(
        self,
        name: str,
        kind: str,
    ) -> list[tuple[float, float]]:
        self._require_transport_loaded()

        raw = str(name).strip()
        tidx = self.transport_index(raw)
        kind = kind.lower()

        if kind in {"viscosity", "mu"}:
            n = int(self._transport["n_v"][tidx])
            ranges = self._transport["v_ranges"][tidx, :n]
        elif kind in {"conductivity", "thermal_conductivity", "k"}:
            n = int(self._transport["n_c"][tidx])
            ranges = self._transport["c_ranges"][tidx, :n]
        else:
            raise ValueError("kind must be 'viscosity' or 'conductivity'.")

        return [
            (float(tmin), float(tmax))
            for tmin, tmax in ranges
            if np.isfinite(tmin) and np.isfinite(tmax)
        ]

    def transport_interval_index(
        self,
        name: str,
        temperature: float,
        kind: str,
        *,
        extrapolation_margin: float = 50.0,
    ) -> int:
        ranges = self.transport_temperature_ranges(name, kind)

        T = float(temperature)
        raw = str(name).strip()

        for j, (Tmin, Tmax) in enumerate(ranges):
            if Tmin <= T <= Tmax:
                return j

        first_Tmin = float(ranges[0][0])

        if first_Tmin - extrapolation_margin <= T < first_Tmin:
            return 0

        last_Tmax = float(ranges[-1][1])

        if last_Tmax < T <= last_Tmax + extrapolation_margin:
            return len(ranges) - 1

        raise ValueError(
            f"No CEA {kind} transport interval for {raw!r} at T={T:.6g} K."
        )
    
    
    def transport_fit(self, name: str, temperature: float, kind: str) -> float:
        self._require_transport_loaded()

        tidx = self.transport_index(name)
        j = self.transport_interval_index(name, temperature, kind)
        T = float(temperature)
        kind = kind.lower()

        if kind in {"viscosity", "mu"}:
            A, B, C, D = self._transport["v_coeffs"][tidx, j]
        elif kind in {"conductivity", "thermal_conductivity", "k"}:
            A, B, C, D = self._transport["c_coeffs"][tidx, j]
        else:
            raise ValueError("kind must be 'viscosity' or 'conductivity'.")

        return float(np.exp(A * np.log(T) + B / T + C / T**2 + D))

    def viscosity(self, name: str, temperature: float) -> float:
        mu_micro_poise = self.transport_fit(name, temperature, "viscosity")
        return mu_micro_poise * 1e-7

    def conductivity(self, name: str, temperature: float) -> float:
        k_micro_w_cm_k = self.transport_fit(name, temperature, "conductivity")
        return k_micro_w_cm_k * 1e-4

    def mole_to_mass(
        self,
        species_names: list[str],
        mole_fractions: list[float] | np.ndarray,
    ) -> np.ndarray:
        species_names = [self.resolve_name(name) for name in species_names]
        x = np.asarray(mole_fractions, dtype=float)

        if not np.isclose(np.sum(x), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0.")

        mw = np.asarray(
            [self.molecular_weight(name) for name in species_names],
            dtype=float,
        )

        return x * mw / np.dot(x, mw)

    def mass_to_mole(
        self,
        species_names: list[str],
        mass_fractions: list[float] | np.ndarray,
    ) -> np.ndarray:
        species_names = [self.resolve_name(name) for name in species_names]
        w = np.asarray(mass_fractions, dtype=float)

        if not np.isclose(np.sum(w), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0.")

        mw = np.asarray(
            [self.molecular_weight(name) for name in species_names],
            dtype=float,
        )

        inv = w / mw
        return inv / np.sum(inv)

    def mixture_molar_mass(
        self,
        species_names: list[str],
        mole_fractions: list[float] | np.ndarray,
    ) -> float:
        species_names = [self.resolve_name(name) for name in species_names]
        x = np.asarray(mole_fractions, dtype=float)

        if not np.isclose(np.sum(x), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0.")

        mw_kg_per_kmol = np.asarray(
            [self.molecular_weight(name) for name in species_names],
            dtype=float,
        )

        return float(np.dot(x, mw_kg_per_kmol)) / 1000.0

    def show_species(self) -> list[str]:
        for name in self.names:
            print(name)
        return self.names

    def show_products(self) -> list[str]:
        for name in self.product_names:
            print(name)
        return self.product_names

    def show_product_species(self) -> list[str]:
        return self.show_products()

    def show_gas_species(self) -> list[str]:
        for name in self.gas_species:
            print(name)
        return self.gas_species

    def show_condensed_species(self) -> list[str]:
        for name in self.condensed_species:
            print(name)
        return self.condensed_species

    def show_reactants(self) -> list[str]:
        for name in self.reactant_names:
            print(name)
        return self.reactant_names

    def show_transport_species(self) -> list[str]:
        for name in self.transport_names:
            print(name)
        return self.transport_names


CEA = CEADatabase()