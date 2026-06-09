from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Tuple, Union

import numpy as np

from .CombustionRegistry import CombustionRegistry


# Folder containing the parser-generated CEA/CEAM data files.
CEA_DATA_FOLDER = "cea_data"

_ROOT = Path(__file__).resolve().parents[2]
_CEA_DATA_DIR = _ROOT / CEA_DATA_FOLDER

_THERMO_PATH = _CEA_DATA_DIR / "thermo_ceam.npz"
_THERMO_INDEX_PATH = _CEA_DATA_DIR / "thermo_name_index.json"
_TRANS_PATH = _CEA_DATA_DIR / "trans_ceam.npz"
_TRANS_INDEX_PATH = _CEA_DATA_DIR / "trans_name_index.json"

if not _THERMO_PATH.exists() or not _THERMO_INDEX_PATH.exists():
    raise FileNotFoundError(
        "CEA thermo data files were not found. Expected "
        f"{_THERMO_PATH} and {_THERMO_INDEX_PATH}."
    )

_CEA_THERMO_FILE = np.load(_THERMO_PATH, allow_pickle=False)
_CEA_THERMO = {key: _CEA_THERMO_FILE[key] for key in _CEA_THERMO_FILE.files}
_CEA_THERMO_FILE.close()

with open(_THERMO_INDEX_PATH, "r") as f:
    _CEA_THERMO_INDEX = json.load(f)

if _TRANS_PATH.exists() and _TRANS_INDEX_PATH.exists():
    _CEA_TRANS_FILE = np.load(_TRANS_PATH, allow_pickle=False)
    _CEA_TRANS = {key: _CEA_TRANS_FILE[key] for key in _CEA_TRANS_FILE.files}
    _CEA_TRANS_FILE.close()

    with open(_TRANS_INDEX_PATH, "r") as f:
        _CEA_TRANS_INDEX = json.load(f)
else:
    _CEA_TRANS = None
    _CEA_TRANS_INDEX = {}


class CombustionGas:
    """
    Fixed-composition ideal-gas mixture wrapper using NASA CEA / CEAM data.

    This class evaluates thermodynamic and transport properties for a known gas
    composition at a specified pressure and temperature. It does not solve
    equilibrium chemistry. Equilibrium and frozen-flow models should compute or
    provide the composition, then use this wrapper for gas properties.
    """

    _BACKEND_NAME = "NASA CEA / CEAM"

    _RU = 8.31446261815324
    _RU_KMOL = 8314.46261815324
    _P_REF = 100000.0

    _UNSUPPORTED_PROPERTIES = {
        "quality",
        "surface_tension",
        "vapor_pressure",
        "saturation_pressure",
        "saturation_temperature",
        "heat_of_vaporization",
        "critical_pressure",
        "critical_temperature",
        "critical_density",
        "freezing_temperature",
        "boiling_temperature",
    }

    _FLASH_INPUTS = {
        frozenset(("pressure", "temperature")),
    }

    def __init__(
        self,
        composition: Union[str, Dict[str, float]],
        basis: str = "mole",
        pressure: float | None = None,
        temperature: float | None = None,
        quality: float | None = None,
    ):
        if quality is not None:
            raise ValueError("CombustionGas does not support vapor quality.")

        if pressure is None or temperature is None:
            raise ValueError("CombustionGas requires pressure and temperature.")

        if basis not in ("mole", "mass"):
            raise ValueError("basis must be 'mole' or 'mass'.")

        self._pressure = float(pressure)
        self._temperature = float(temperature)

        self._species_names: List[str] = []
        self._display_names: List[str] = []
        self._thermo_indices: List[int] = []
        self._transport_indices: List[int | None] = []

        if isinstance(composition, str):
            species_name, display_name, thermo_index, transport_index = self._resolve_species(composition)
            self._species_names = [species_name]
            self._display_names = [display_name]
            self._thermo_indices = [thermo_index]
            self._transport_indices = [transport_index]
            self._mole_fractions = np.array([1.0], dtype=float)
            self._mass_fractions = np.array([1.0], dtype=float)
            self._mixture = False

        elif isinstance(composition, dict):
            if not composition:
                raise ValueError("composition cannot be empty.")

            tmp: dict[str, tuple[float, str, int, int | None, list[str]]] = {}

            for user_name, frac in composition.items():
                species_name, display_name, thermo_index, transport_index = self._resolve_species(user_name)
                total, _, _, _, labels = tmp.get(
                    species_name,
                    (0.0, display_name, thermo_index, transport_index, []),
                )
                tmp[species_name] = (
                    total + float(frac),
                    display_name,
                    thermo_index,
                    transport_index,
                    labels + [display_name],
                )

            fractions = np.array([item[0] for item in tmp.values()], dtype=float)

            if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{basis.capitalize()} fractions must sum to 1.0.")

            self._species_names = list(tmp.keys())
            self._display_names = [", ".join(sorted(set(item[4]))) for item in tmp.values()]
            self._thermo_indices = [item[2] for item in tmp.values()]
            self._transport_indices = [item[3] for item in tmp.values()]

            if basis == "mole":
                self._mole_fractions = fractions
                self._mass_fractions = self.mole_to_mass(self._species_names, fractions)
            else:
                self._mass_fractions = fractions
                self._mole_fractions = self.mass_to_mole(self._species_names, fractions)

            self._mixture = len(self._species_names) > 1

        else:
            raise TypeError("composition must be a species name or a dict of fractions.")

        self._M = np.array([self._mw_from_index(i) for i in self._thermo_indices], dtype=float)
        self._minimum_temperature, self._maximum_temperature = self._temperature_limits()

        self._validate_temperature()

    @classmethod
    def _resolve_species(cls, value: str) -> tuple[str, str, int, int | None]:
        """Resolve a user species name/alias to CEA species and table indices."""
        raw_name = str(value).strip()

        try:
            reactant_name = CombustionRegistry.cea_reactant_name(raw_name)
        except Exception:
            reactant_name = None

        try:
            species_name = CombustionRegistry.cea_name(raw_name)
            display_name = CombustionRegistry.name(raw_name)
        except Exception:
            species_name = raw_name
            display_name = species_name

        if species_name not in _CEA_THERMO_INDEX and reactant_name is not None:
            raise ValueError(
                f"{value!r} maps to CEA reactant {reactant_name!r}, not a "
                "gas-phase CEA product species. CombustionGas is a fixed-"
                "composition gas-property wrapper, so it needs product species "
                "such as 'H2O', 'CO2', 'CO', 'H2', 'O2', 'OH', or a dict of "
                "their fractions."
            )

        if species_name not in _CEA_THERMO_INDEX:
            raise ValueError(
                f"{value!r} could not be resolved to a CEA thermo species. "
                "Use a supported CombustionRegistry CEA alias or a direct CEA product "
                "species name."
            )

        thermo_index = int(_CEA_THERMO_INDEX[species_name])
        n_intervals = int(_CEA_THERMO["n_intervals"][thermo_index])

        if n_intervals <= 0 or np.isnan(_CEA_THERMO["coeffs"][thermo_index]).all():
            elements = cls._elemental_composition_from_index(thermo_index)
            extra = ""
            if reactant_name is not None:
                extra = f" It also maps to CEA reactant {reactant_name!r}."
            raise ValueError(
                f"{species_name!r} is present in the CEA data, but it has no NASA-9 "
                "polynomial intervals. It is a CEA reactant definition rather "
                "than a gas-phase thermodynamic species. "
                f"Elemental composition: {elements}.{extra} "
                "CombustionGas only accepts product/species entries with polynomial data."
            )

        transport_index = None
        if species_name in _CEA_TRANS_INDEX:
            transport_index = int(_CEA_TRANS_INDEX[species_name])

        return species_name, display_name, thermo_index, transport_index

    @staticmethod
    def _elemental_composition_from_index(index: int) -> dict[str, float]:
        symbols = _CEA_THERMO["element_symbols"][index]
        counts = _CEA_THERMO["element_counts"][index]

        return {
            str(symbol): float(count)
            for symbol, count in zip(symbols, counts)
            if str(symbol) and np.isfinite(count)
        }

    @staticmethod
    def _mw_from_index(index: int) -> float:
        """Return molecular weight [kg/kmol], numerically equal to g/mol."""
        return float(_CEA_THERMO["mw"][index])

    @classmethod
    def _species_molar_mass(cls, species_name: str) -> float:
        """Return molar mass [kg/mol] for a direct CEA species name."""
        if species_name not in _CEA_THERMO_INDEX:
            raise ValueError(f"Unknown CEA species: {species_name!r}")
        return cls._mw_from_index(int(_CEA_THERMO_INDEX[species_name])) / 1000.0

    @property
    def pressure(self) -> float:
        return self._pressure

    @pressure.setter
    def pressure(self, value: float):
        self._pressure = float(value)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = float(value)
        self._validate_temperature()

    @property
    def pressure_temperature(self) -> Tuple[float, float]:
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature).")
        self._pressure = float(values[0])
        self._temperature = float(values[1])
        self._validate_temperature()

    @property
    def name(self) -> str:
        return ", ".join(self._display_names)

    @property
    def backend(self) -> str:
        return self._BACKEND_NAME

    @property
    def species(self) -> List[str]:
        return list(self._species_names)

    @property
    def phase(self) -> str:
        return "Ideal Gas"

    @property
    def is_mixture(self) -> bool:
        return self._mixture

    @property
    def mole_fractions(self) -> dict[str, float]:
        return {
            name: float(x)
            for name, x in zip(self._species_names, self._mole_fractions)
        }

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        if len(self._species_names) == 1:
            raise ValueError("Cannot change mole fractions for a pure gas.")
        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0.")
        self._mole_fractions = np.asarray(value, dtype=float)
        self._mass_fractions = self._mole_fractions * self._M / np.dot(self._mole_fractions, self._M)

    @property
    def mass_fractions(self) -> dict[str, float]:
        return {
            name: float(w)
            for name, w in zip(self._species_names, self._mass_fractions)
        }

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        if len(self._species_names) == 1:
            raise ValueError("Cannot change mass fractions for a pure gas.")
        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0.")
        self._mass_fractions = np.asarray(value, dtype=float)
        inv = self._mass_fractions / self._M
        self._mole_fractions = inv / inv.sum()

    @property
    def molar_mass(self) -> float:
        """Mixture molar mass [kg/mol]."""
        mw_kg_per_kmol = float(np.dot(self._mole_fractions, self._M))
        return mw_kg_per_kmol / 1000.0

    @property
    def gas_constant(self) -> float:
        """Mixture gas constant [J/kg-K]."""
        return self._RU / self.molar_mass

    @property
    def universal_gas_constant(self) -> float:
        return self._RU

    @property
    def compressibility(self) -> float:
        return 1.0

    @property
    def density(self) -> float:
        return self.pressure / (self.gas_constant * self.temperature)

    @density.setter
    def density(self, value: float):
        self._pressure = float(value) * self.gas_constant * self.temperature

    @property
    def specific_volume(self) -> float:
        return 1.0 / self.density

    @property
    def quality(self) -> float:
        raise NotImplementedError("CombustionGas does not support vapor quality.")

    @quality.setter
    def quality(self, value: float):
        raise ValueError("CombustionGas does not support vapor quality.")

    def _validate_temperature(self) -> None:
        if self.temperature < self.minimum_temperature or self.temperature > self.maximum_temperature:
            raise ValueError(
                f"Temperature {self.temperature:.6g} K is outside the common valid "
                f"CEA polynomial range [{self.minimum_temperature:.6g}, "
                f"{self.maximum_temperature:.6g}] K for this composition."
            )

    def _temperature_limits(self) -> tuple[float, float]:
        mins = []
        maxs = []

        for idx in self._thermo_indices:
            n = int(_CEA_THERMO["n_intervals"][idx])
            ranges = _CEA_THERMO["t_ranges"][idx, :n]
            mins.append(float(np.nanmin(ranges[:, 0])))
            maxs.append(float(np.nanmax(ranges[:, 1])))

        return max(mins), min(maxs)

    @staticmethod
    def _interval_index(thermo_index: int, temperature: float) -> int:
        n = int(_CEA_THERMO["n_intervals"][thermo_index])
        T = float(temperature)

        for j in range(n):
            Tmin, Tmax = _CEA_THERMO["t_ranges"][thermo_index, j]
            if Tmin <= T <= Tmax:
                return j

        name = str(_CEA_THERMO["names"][thermo_index])
        raise ValueError(f"No CEA polynomial interval for {name!r} at T={T:.6g} K.")

    @classmethod
    def _species_thermo(cls, thermo_index: int, temperature: float) -> tuple[float, float, float]:
        T = float(temperature)
        j = cls._interval_index(thermo_index, T)

        a = _CEA_THERMO["coeffs"][thermo_index, j]
        a0, a1, a2, a3, a4, a5, a6, b1, b2 = a
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

        cp = cp_over_R * cls._RU_KMOL
        h = h_over_RT * cls._RU_KMOL * T
        s0 = s_over_R * cls._RU_KMOL

        return float(cp), float(h), float(s0)

    def _pure_cp_mass(self) -> np.ndarray:
        return np.array(
            [self._species_thermo(i, self.temperature)[0] / self._M[k] for k, i in enumerate(self._thermo_indices)],
            dtype=float,
        )

    def _pure_h_mass(self) -> np.ndarray:
        return np.array(
            [self._species_thermo(i, self.temperature)[1] / self._M[k] for k, i in enumerate(self._thermo_indices)],
            dtype=float,
        )

    def _pure_s0_mass(self) -> np.ndarray:
        return np.array(
            [self._species_thermo(i, self.temperature)[2] / self._M[k] for k, i in enumerate(self._thermo_indices)],
            dtype=float,
        )

    @property
    def specific_heat_cp(self) -> float:
        return float(np.dot(self._mass_fractions, self._pure_cp_mass()))

    @property
    def specific_heat_cv(self) -> float:
        return self.specific_heat_cp - self.gas_constant

    @property
    def specific_heat(self) -> float:
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        cv = self.specific_heat_cv
        return None if cv == 0.0 else self.specific_heat_cp / cv

    @property
    def enthalpy(self) -> float:
        return float(np.dot(self._mass_fractions, self._pure_h_mass()))

    @property
    def internal_energy(self) -> float:
        return self.enthalpy - self.gas_constant * self.temperature

    @property
    def entropy(self) -> float:
        p_i = self._mole_fractions * self.pressure
        p_i = np.maximum(p_i, np.finfo(float).tiny)

        pressure_correction_molar = -self._RU_KMOL * np.log(p_i / self._P_REF)

        pure_s0_molar = np.array(
            [self._species_thermo(i, self.temperature)[2] for i in self._thermo_indices],
            dtype=float,
        )

        pure_s_mass_at_pi = (pure_s0_molar + pressure_correction_molar) / self._M
        return float(np.dot(self._mass_fractions, pure_s_mass_at_pi))

    @property
    def gibbs_energy(self) -> float:
        return self.enthalpy - self.temperature * self.entropy

    @property
    def free_energy(self) -> float:
        return self.helmholtz_energy

    @property
    def helmholtz_energy(self) -> float:
        return self.internal_energy - self.temperature * self.entropy

    @property
    def speed_of_sound(self) -> float:
        return float(np.sqrt(self.specific_heat_ratio * self.gas_constant * self.temperature))

    @staticmethod
    def _transport_interval_index(transport_index: int, temperature: float, kind: str) -> int:
        if _CEA_TRANS is None:
            raise FileNotFoundError(
                "CEA transport files were not found. Expected "
                f"{_TRANS_PATH} and {_TRANS_INDEX_PATH}."
            )

        T = float(temperature)

        if kind == "viscosity":
            n = int(_CEA_TRANS["n_v"][transport_index])
            ranges = _CEA_TRANS["v_ranges"][transport_index]
        elif kind == "conductivity":
            n = int(_CEA_TRANS["n_c"][transport_index])
            ranges = _CEA_TRANS["c_ranges"][transport_index]
        else:
            raise ValueError(f"Unknown transport kind: {kind!r}")

        for j in range(n):
            Tmin, Tmax = ranges[j]
            if Tmin <= T <= Tmax:
                return j

        raise ValueError(
            f"No CEA {kind} transport interval for transport index "
            f"{transport_index} at T={T:.6g} K."
        )

    @classmethod
    def _transport_fit(cls, transport_index: int, temperature: float, kind: str) -> float:
        j = cls._transport_interval_index(transport_index, temperature, kind)
        T = float(temperature)

        if kind == "viscosity":
            A, B, C, D = _CEA_TRANS["v_coeffs"][transport_index, j]
        elif kind == "conductivity":
            A, B, C, D = _CEA_TRANS["c_coeffs"][transport_index, j]
        else:
            raise ValueError(f"Unknown transport kind: {kind!r}")

        return float(np.exp(A * np.log(T) + B / T + C / T**2 + D))

    def _require_transport(self, species_index: int, kind: str) -> int:
        transport_index = self._transport_indices[species_index]

        if transport_index is None:
            name = self._species_names[species_index]
            raise NotImplementedError(
                f"CEA {kind} transport data are not available for {name!r}."
            )

        return int(transport_index)

    def _pure_viscosities(self) -> np.ndarray:
        values = []

        for k in range(len(self._species_names)):
            idx = self._require_transport(k, "viscosity")
            mu_micro_poise = self._transport_fit(idx, self.temperature, "viscosity")
            values.append(mu_micro_poise * 1e-7)

        return np.asarray(values, dtype=float)

    def _pure_conductivities(self) -> np.ndarray:
        values = []

        for k in range(len(self._species_names)):
            idx = self._require_transport(k, "conductivity")
            k_micro_w_cm_k = self._transport_fit(idx, self.temperature, "conductivity")
            values.append(k_micro_w_cm_k * 1e-4)

        return np.asarray(values, dtype=float)

    @staticmethod
    def _wilke_phi(property_values: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
        phi = np.zeros((len(property_values), len(property_values)))

        for i in range(len(property_values)):
            for j in range(len(property_values)):
                phi[i, j] = (
                    (1.0 + np.sqrt(property_values[i] / property_values[j]) * (molecular_weights[j] / molecular_weights[i]) ** 0.25) ** 2
                    / np.sqrt(8.0 * (1.0 + molecular_weights[i] / molecular_weights[j]))
                )

        return phi

    @staticmethod
    def _wilke_mix(mole_fractions: np.ndarray, values: np.ndarray, molecular_weights: np.ndarray) -> float:
        phi = CombustionGas._wilke_phi(values, molecular_weights)

        return float(
            sum(
                mole_fractions[i] * values[i] / sum(mole_fractions[j] * phi[i, j] for j in range(len(values)))
                for i in range(len(values))
            )
        )

    @property
    def dynamic_viscosity(self) -> float:
        mu = self._pure_viscosities()

        if not self._mixture:
            return float(mu[0])

        return self._wilke_mix(self._mole_fractions, mu, self._M)

    @property
    def conductivity(self) -> float:
        k = self._pure_conductivities()

        if not self._mixture:
            return float(k[0])

        return self._wilke_mix(self._mole_fractions, k, self._M)

    @property
    def thermal_conductivity(self) -> float:
        return self.conductivity

    @property
    def kinematic_viscosity(self) -> float:
        return self.dynamic_viscosity / self.density

    @property
    def prandtl(self) -> float:
        k = self.thermal_conductivity

        if k is None or k == 0.0:
            return None

        return self.specific_heat_cp * self.dynamic_viscosity / k

    @property
    def thermal_expansion_coefficient(self) -> float:
        return 1.0 / self.temperature

    @property
    def isothermal_compressibility(self) -> float:
        return 1.0 / self.pressure

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        of = of.lower()
        wrt = with_respect_to.lower()
        const = constant.lower()

        R = self.gas_constant
        T = self.temperature
        rho = self.density
        cp = self.specific_heat_cp
        cv = self.specific_heat_cv

        if (of, wrt, const) == ("hmass", "t", "p"):
            return cp

        if (of, wrt, const) == ("umass", "t", "dmass"):
            return cv

        if (of, wrt, const) == ("dmass", "p", "t"):
            return 1.0 / (R * T)

        if (of, wrt, const) == ("dmass", "t", "p"):
            return -rho / T

        if (of, wrt, const) == ("t", "p", "hmass"):
            return 0.0

        if (of, wrt, const) == ("hmass", "p", "t"):
            return 0.0

        raise NotImplementedError(
            f"CombustionGas partial derivative d({of})/d({wrt})|{const} is not implemented."
        )

    @property
    def dhdT_const_p(self) -> float:
        return self.partial_derivative("Hmass", "T", "P")

    @property
    def dhdp_const_T(self) -> float:
        return self.partial_derivative("Hmass", "P", "T")

    @property
    def drhodT_const_p(self) -> float:
        return self.partial_derivative("Dmass", "T", "P")

    @property
    def drhodp_const_T(self) -> float:
        return self.partial_derivative("Dmass", "P", "T")

    @property
    def dTdp_const_h(self) -> float:
        return self.partial_derivative("T", "P", "Hmass")

    @property
    def joule_thomson_coefficient(self) -> float:
        return self.dTdp_const_h

    @property
    def minimum_temperature(self) -> float:
        return self._minimum_temperature

    @property
    def maximum_temperature(self) -> float:
        return self._maximum_temperature

    @property
    def minimum_pressure(self) -> float:
        return 1e-30

    @property
    def maximum_pressure(self) -> float:
        return np.inf

    def _safe(self, value, fmt=".3e"):
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def __str__(self):
        def format_dict(d: dict, decimals=5):
            return {k: round(v, decimals) for k, v in d.items()}

        rows = [
            ("Gas(es)", ", ".join(self._species_names)),
            ("Backend", self.backend),
            ("Mole fractions", format_dict(self.mole_fractions, 5)),
            ("Mass fractions", format_dict(self.mass_fractions, 5)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Compressibility Z", self._safe(self.compressibility, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Cp [J/kg-K]", self._safe(self.specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]", self._safe(self.specific_heat_cv, ".3f")),
            ("Specific heat ratio", self._safe(self.specific_heat_ratio, ".5f")),
            ("Gas constant [J/kg-K]", self._safe(self.gas_constant, ".3f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.thermal_conductivity, ".3f")),
            ("Prandtl number", self._safe(self.prandtl, ".5f")),
            ("Speed of sound [m/s]", self._safe(self.speed_of_sound, ".3f")),
        ]

        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)

    def __repr__(self) -> str:
        species_str = ", ".join(self._species_names)
        return (
            f"{self.__class__.__name__}(species=[{species_str}], "
            f"pressure={self.pressure:.3e} Pa, "
            f"temperature={self.temperature:.2f} K)"
        )

    @classmethod
    def get_available_species(cls) -> list[str]:
        """Return direct CEA species names with NASA-9 polynomial data."""
        names = []

        for name, i in _CEA_THERMO_INDEX.items():
            idx = int(i)
            if int(_CEA_THERMO["n_intervals"][idx]) > 0 and not np.isnan(_CEA_THERMO["coeffs"][idx]).all():
                names.append(name)

        return sorted(names)

    @classmethod
    def show_available_species(cls) -> list[str]:
        species = cls.get_available_species()

        for name in species:
            print(name)

        return species

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
        )

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @staticmethod
    def mole_to_mass(species_names: List[str], mole_fractions: List[float]):
        if not np.isclose(sum(mole_fractions), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0.")
        x = np.asarray(mole_fractions, dtype=float)
        M = np.array([CombustionGas._species_molar_mass(name) * 1000.0 for name in species_names])
        return x * M / np.dot(x, M)

    @staticmethod
    def mass_to_mole(species_names: List[str], mass_fractions: List[float]):
        if not np.isclose(sum(mass_fractions), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0.")
        w = np.asarray(mass_fractions, dtype=float)
        M = np.array([CombustionGas._species_molar_mass(name) * 1000.0 for name in species_names])
        inv = w / M
        return inv / inv.sum()

    @classmethod
    def supported_properties(cls) -> list[str]:
        unsupported = getattr(cls, "_UNSUPPORTED_PROPERTIES", set())

        return sorted(
            name
            for name, value in vars(cls).items()
            if isinstance(value, property)
            and not name.startswith("_")
            and name not in unsupported
        )

    @classmethod
    def show_supported_properties(cls) -> list[str]:
        properties = cls.supported_properties()

        for prop in properties:
            print(prop)

        return properties

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        return property_name in cls.supported_properties()