from __future__ import annotations

from typing import Tuple
from pathlib import Path
import json

import numpy as np

from .CombustionRegistry import CombustionRegistry


CEA_DATA_FOLDER = "cea_data"

_CEA_DATA_DIR = Path(__file__).resolve().parents[2] / CEA_DATA_FOLDER
_CEA_THERMO_PATH = _CEA_DATA_DIR / "thermo_ceam.npz"
_CEA_THERMO_INDEX_PATH = _CEA_DATA_DIR / "thermo_name_index.json"


def _load_cea_thermo_data():
    """Load CEA reactant/species data once at module import.

    The Propellant wrapper still uses RocketProps for liquid correlations.
    CEA data are only used for reactant bookkeeping properties needed by
    combustion calculations, such as elemental composition and heat of
    formation.
    """
    if not _CEA_THERMO_PATH.exists() or not _CEA_THERMO_INDEX_PATH.exists():
        return None, {}

    thermo = np.load(_CEA_THERMO_PATH, allow_pickle=False)

    with open(_CEA_THERMO_INDEX_PATH, "r") as f:
        index = json.load(f)

    return thermo, index


_CEA_THERMO, _CEA_THERMO_INDEX = _load_cea_thermo_data()


class Propellant:
    """
    RocketProps-backed liquid propellant property wrapper.

    This class intentionally stays close to what RocketProps is designed to do:
    liquid rocket propellant engineering properties. It is not a thermodynamic
    flash solver and does not attempt to calculate enthalpy, internal energy,
    entropy, vapor-state properties, or two-phase states.

    Notes
    -----
    RocketProps is mainly useful for liquid propellant properties such as:

        density
        compressed-liquid density
        dynamic viscosity
        compressed-liquid dynamic viscosity
        vapor pressure
        saturation temperature
        heat of vaporization
        surface tension
        heat capacity
        thermal conductivity
        critical properties
        normal boiling point
        freezing point

    Supported state inputs:

        Propellant(..., temperature=...)
        Propellant(..., temperature=..., pressure=...)

    Temperature-only states use saturated-liquid correlations.
    Temperature-pressure states use compressed-liquid correlations where
    RocketProps provides them.

    Public API units are SI:

        pressure: Pa
        temperature: K
        density: kg/m^3
        dynamic_viscosity: Pa-s
        conductivity: W/m-K
        surface_tension: N/m
        specific_heat_cp: J/kg-K
        heat_of_vaporization: J/kg
    """

    _BACKEND_NAME = "RocketProps"

    _UNSUPPORTED_PROPERTIES = {
        "enthalpy",
        "internal_energy",
        "entropy",
        "specific_heat_cv",
        "specific_heat_ratio",
        "speed_of_sound",
        "thermal_expansion_coefficient",
        "isothermal_compressibility",
        "joule_thomson_coefficient",
        "partial_derivative",
        "helmholtz_energy",
        "gibbs_energy",
        "gas_constant",
        "universal_gas_constant",
        "prandtl",
    }

    _FLASH_INPUTS = {
        frozenset(("temperature",)),
        frozenset(("pressure", "temperature")),
    }

    _PSIA_TO_PA = 6894.757293168361
    _PA_TO_PSIA = 1.0 / _PSIA_TO_PA

    _BTU_PER_LBM_TO_J_PER_KG = 2326.0
    _BTU_PER_LBM_R_TO_J_PER_KG_K = 4186.8
    _BTU_PER_HR_FT_R_TO_W_PER_M_K = 1.730735
    _LBF_PER_IN_TO_N_PER_M = 175.126835

    def __init__(
        self,
        propellant: str,
        temperature: float,
        pressure: float | None = None,
    ):
        """
        Initialize a liquid propellant property state.

        Parameters
        ----------
        propellant:
            RocketProps propellant name or a common alias, such as "rp1",
            "lox", "mmh", "n2o4", "MON25", or "A50".
        temperature:
            Propellant temperature in K.
        pressure:
            Optional pressure in Pa. If omitted, saturated-liquid properties are
            used. If provided, compressed-liquid properties are used where
            RocketProps supports them.

        Raises
        ------
        ValueError
            If the specified pressure is below vapor pressure at the specified
            temperature. In that case, the state is not a stable liquid state,
            and this wrapper intentionally refuses to extrapolate.
        """
        self._propellant_name = self._normalize_name(propellant)
        self._backend = self._get_backend(self._propellant_name)

        self._cea_reactant_name, self._cea_reactant_index = self._cea_reactant_lookup(propellant)

        self._temperature = float(temperature)
        self._pressure = None if pressure is None else float(pressure)

        self._validate_liquid_state()

    # ---------------- Unit conversion helpers ---------------- #

    @staticmethod
    def _degR_from_K(temperature: float) -> float:
        """Convert K to degR."""
        return float(temperature) * 9.0 / 5.0

    @staticmethod
    def _K_from_degR(temperature: float) -> float:
        """Convert degR to K."""
        return float(temperature) * 5.0 / 9.0

    @classmethod
    def _psia_from_Pa(cls, pressure: float) -> float:
        """Convert Pa to psia."""
        return float(pressure) * cls._PA_TO_PSIA

    @classmethod
    def _Pa_from_psia(cls, pressure: float) -> float:
        """Convert psia to Pa."""
        return float(pressure) * cls._PSIA_TO_PA

    @classmethod
    def _normalize_name(cls, propellant: str) -> str:
        """Return the RocketProps backend name for a user propellant name."""
        return CombustionRegistry.propellant_name(propellant)

    @staticmethod
    def _get_backend(propellant: str):
        """Load a RocketProps propellant object."""
        try:
            from rocketprops.rocket_prop import get_prop
        except ImportError as exc:
            raise ImportError(
                "Propellant requires RocketProps. Install it with "
                "`pip install rocketprops`."
            ) from exc

        backend = get_prop(propellant)

        if backend is None:
            raise ValueError(f"Unknown RocketProps propellant: {propellant!r}")

        return backend

    @staticmethod
    def _cea_reactant_lookup(propellant: str) -> tuple[str | None, int | None]:
        """Return cached CEA reactant name and database row for this propellant."""
        try:
            reactant_name = CombustionRegistry.cea_reactant_name(propellant)
        except Exception:
            return None, None

        if _CEA_THERMO is None:
            return reactant_name, None

        index = _CEA_THERMO_INDEX.get(reactant_name)

        if index is None:
            return reactant_name, None

        return reactant_name, int(index)

    def _cea_value(self, key: str):
        """Return one raw CEA database value for the cached reactant row."""
        if self._cea_reactant_index is None or _CEA_THERMO is None:
            return None

        return _CEA_THERMO[key][self._cea_reactant_index]

    def _call(self, *names: str, default=None):
        """Return the first available backend attribute or no-argument method."""
        for name in names:
            attr = getattr(self._backend, name, None)

            if attr is None:
                continue

            try:
                return attr() if callable(attr) else attr
            except TypeError:
                continue

        return default

    def _call_at_temperature(self, *names: str, default=None):
        """Call the first available RocketProps temperature-based method."""
        TdegR = self._degR_from_K(self.temperature)

        for name in names:
            fn = getattr(self._backend, name, None)

            if fn is None:
                continue

            try:
                return fn(TdegR)
            except TypeError:
                continue

        return default

    def _call_compressed(self, *names: str, default=None):
        """Call the first available RocketProps compressed-liquid method."""
        if self.pressure is None:
            return default

        TdegR = self._degR_from_K(self.temperature)
        Ppsia = self._psia_from_Pa(self.pressure)

        for name in names:
            fn = getattr(self._backend, name, None)

            if fn is None:
                continue

            try:
                return fn(TdegR, Ppsia)
            except TypeError:
                continue

        return default

    def _validate_liquid_state(self) -> None:
        """
        Refuse states that are below vapor pressure.

        RocketProps liquid correlations should not be used to represent a vapor
        or superheated state. Temperature-only construction is allowed because
        it means saturated liquid at vapor_pressure.
        """
        if self.pressure is None:
            return

        pvap = self.vapor_pressure

        if pvap is None:
            return

        if self.pressure < pvap:
            raise ValueError(
                f"{self._propellant_name}: pressure={self.pressure:.6g} Pa is "
                f"below vapor pressure={pvap:.6g} Pa at "
                f"temperature={self.temperature:.6g} K. "
                "RocketProps liquid correlations are not valid for this state."
            )

    def _unsupported(self, property_name: str):
        """Raise a clear error for properties RocketProps is not used to compute."""
        raise NotImplementedError(
            f"Propellant.{property_name} is not supported. "
            "RocketProps is used here for liquid propellant property "
            "correlations, not as a thermodynamic flash solver."
        )

    # ---------------- State setters ---------------- #

    @property
    def pressure(self) -> float | None:
        """
        Absolute pressure in Pa.

        If pressure is None, properties are evaluated as saturated-liquid
        properties at the current temperature.
        """
        return self._pressure

    @pressure.setter
    def pressure(self, value: float | None):
        self._pressure = None if value is None else float(value)
        self._validate_liquid_state()

    @property
    def temperature(self) -> float:
        """Absolute temperature in K."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = float(value)
        self._validate_liquid_state()

    @property
    def pressure_temperature(self) -> Tuple[float | None, float]:
        """Return (pressure [Pa] or None, temperature [K])."""
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float | None, float]):
        """Update state from pressure and temperature."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature)")

        self._pressure = None if values[0] is None else float(values[0])
        self._temperature = float(values[1])
        self._validate_liquid_state()

    # ---------------- Package-consistent API properties ---------------- #

    @property
    def name(self) -> str:
        """Canonical RocketProps propellant name."""
        return self.propellant

    @property
    def backend(self) -> str:
        """Name of the wrapped property backend."""
        return self._BACKEND_NAME

    @property
    def propellant(self) -> str:
        """Canonical RocketProps propellant name."""
        return self._propellant_name

    @property
    def species(self) -> list[str]:
        """Return the propellant name as a one-item species list."""
        return [self.propellant]

    @property
    def phase(self) -> str:
        """
        Thermodynamic phase label.

        Propellant only exposes liquid RocketProps correlations, so this is
        always "Liquid" for valid states.
        """
        return "Liquid"

    @property
    def phase_model(self) -> str:
        """
        Return the RocketProps liquid-property model currently being used.

        Temperature-only states use saturated-liquid correlations.
        Temperature-pressure states use compressed-liquid correlations where
        RocketProps supports them.
        """
        if self.pressure is None:
            return "Saturated Liquid"
        return "Compressed Liquid"

    @property
    def quality(self) -> float:
        """
        Vapor quality.

        This wrapper only represents liquid-property states, so quality is
        always 0.0. RocketProps is not being used here as a two-phase solver.
        """
        return 0.0

    @quality.setter
    def quality(self, value: float):
        raise ValueError("Propellant only supports liquid-property states.")

    @property
    def thermal_expansion_coefficient(self) -> None:
        """Volumetric thermal expansion coefficient is not supported."""
        return self._unsupported("thermal_expansion_coefficient")

    @property
    def isothermal_compressibility(self):
        return self._unsupported("isothermal_compressibility")

    @property
    def joule_thomson_coefficient(self):
        return self._unsupported("joule_thomson_coefficient")

    @property
    def helmholtz_energy(self):
        return self._unsupported("helmholtz_energy")

    @property
    def gibbs_energy(self):
        return self._unsupported("gibbs_energy")

    @property
    def gas_constant(self):
        return self._unsupported("gas_constant")

    @property
    def universal_gas_constant(self):
        return self._unsupported("universal_gas_constant")

    @property
    def prandtl(self):
        return self._unsupported("prandtl")

    @property
    def enthalpy(self) -> None:
        """Enthalpy is not supported by this RocketProps wrapper."""
        return self._unsupported("enthalpy")

    @property
    def internal_energy(self) -> None:
        """Internal energy is not supported by this RocketProps wrapper."""
        return self._unsupported("internal_energy")

    @property
    def entropy(self) -> None:
        """Entropy is not supported by this RocketProps wrapper."""
        return self._unsupported("entropy")

    @property
    def specific_heat_cv(self) -> None:
        """Cv is not supported by this RocketProps wrapper."""
        return self._unsupported("specific_heat_cv")

    @property
    def specific_heat_ratio(self) -> None:
        """Specific heat ratio is not supported by this RocketProps wrapper."""
        return self._unsupported("specific_heat_ratio")

    @property
    def speed_of_sound(self) -> None:
        """Speed of sound is not supported by this RocketProps wrapper."""
        return self._unsupported("speed_of_sound")

    # ---------------- Liquid propellant properties ---------------- #

    @property
    def density(self) -> float:
        """
        Liquid density in kg/m^3.

        Uses compressed-liquid density when pressure is provided and supported;
        otherwise falls back to saturated-liquid density at temperature.
        """
        value = self._call_compressed("SG_compressed", default=None)

        if value is None:
            value = self._call_at_temperature("SGLiqAtTdegR", "SGAtTdegR", default=None)

        if value is None:
            return None

        return float(value) * 1000.0

    @property
    def specific_volume(self) -> float:
        """Liquid specific volume in m^3/kg."""
        rho = self.density

        if rho is None or rho == 0:
            return None

        return 1.0 / rho

    @property
    def dynamic_viscosity(self) -> float:
        """
        Liquid dynamic viscosity in Pa-s.

        Uses compressed-liquid viscosity when pressure is provided and supported;
        otherwise falls back to saturated-liquid viscosity at temperature.
        """
        value = self._call_compressed("Visc_compressed", default=None)

        if value is None:
            value = self._call_at_temperature("ViscAtTdegR", "ViscAtT", default=None)

        if value is None:
            return None

        return float(value) * 0.1

    @property
    def kinematic_viscosity(self) -> float:
        """Liquid kinematic viscosity in m^2/s."""
        mu = self.dynamic_viscosity
        rho = self.density

        if mu is None or rho is None or rho == 0:
            return None

        return mu / rho

    @property
    def vapor_pressure(self) -> float:
        """Saturation pressure in Pa at the current temperature."""
        value = self._call_at_temperature("PvapAtTdegR", "PvapAtT", default=None)

        if value is None:
            return None

        return self._Pa_from_psia(value)

    @property
    def saturation_pressure(self) -> float:
        """Alias for vapor_pressure in Pa."""
        return self.vapor_pressure

    @property
    def saturation_temperature(self) -> float:
        """Saturation temperature in K at the current pressure."""
        if self.pressure is None:
            return self.temperature

        Ppsia = self._psia_from_Pa(self.pressure)

        for name in ("TdegRAtPsat", "TsatAtP"):
            fn = getattr(self._backend, name, None)

            if fn is None:
                continue

            try:
                return self._K_from_degR(fn(Ppsia))
            except Exception:
                continue

        return None

    @property
    def heat_of_vaporization(self) -> float:
        """Heat of vaporization in J/kg at current temperature."""
        value = self._call_at_temperature("HvapAtTdegR", "HvapAtT", default=None)

        if value is None:
            return None

        return float(value) * self._BTU_PER_LBM_TO_J_PER_KG

    @property
    def surface_tension(self) -> float:
        """Liquid surface tension in N/m."""
        value = self._call_at_temperature("SurfAtTdegR", "SurfAtT", default=None)

        if value is None:
            return None

        return float(value) * self._LBF_PER_IN_TO_N_PER_M

    @property
    def specific_heat_cp(self) -> float:
        """Liquid constant-pressure heat capacity in J/kg-K."""
        value = self._call_at_temperature("CpAtTdegR", "CpAtT", default=None)

        if value is None:
            return None

        return float(value) * self._BTU_PER_LBM_R_TO_J_PER_KG_K

    @property
    def specific_heat(self) -> float:
        """Backward-compatible alias for Cp."""
        return self.specific_heat_cp

    @property
    def conductivity(self) -> float:
        """Liquid thermal conductivity in W/m-K."""
        value = self._call_at_temperature("CondAtTdegR", "CondAtT", default=None)

        if value is None:
            return None

        return float(value) * self._BTU_PER_HR_FT_R_TO_W_PER_M_K

    @property
    def thermal_conductivity(self) -> float:
        """Backward-compatible alias for conductivity."""
        return self.conductivity

    @property
    def saturated_liquid_compressibility_factor(self) -> float:
        """
        Saturated-liquid compressibility factor when RocketProps provides it.

        This is not a general CoolProp-style real-fluid compressibility
        calculation. The longer name avoids confusion with Fluid.compressibility.
        """
        value = self._call_at_temperature("ZLiqAtTdegR", "ZLiqAtT", default=None)

        if value is None:
            return None

        return float(value)

    @property
    def compressibility(self) -> float:
        """Alias for saturated-liquid compressibility factor."""
        return self.saturated_liquid_compressibility_factor

    # ---------------- CEA reactant/reference properties ---------------- #

    @property
    def cea_reactant(self) -> str | None:
        """NASA CEA / CEAM reactant name used for combustion calculations."""
        return self._cea_reactant_name

    @property
    def elemental_composition(self) -> dict[str, float] | None:
        """CEA reactant elemental composition as ``{symbol: atom_count}``.

        For example, RP-1 is stored by CEA as a normalized pseudo-formula
        similar to ``C1 H1.95``. This is combustion bookkeeping data, not a
        full molecular structure.
        """
        symbols = self._cea_value("element_symbols")
        counts = self._cea_value("element_counts")

        if symbols is None or counts is None:
            return None

        composition = {}

        for symbol, count in zip(symbols, counts):
            symbol = str(symbol).strip()

            if not symbol or not np.isfinite(count):
                continue

            composition[symbol] = float(count)

        return composition

    @property
    def cea_molar_mass(self) -> float | None:
        """CEA reactant molar mass in kg/mol.

        This can differ from ``molar_mass`` for pseudo-propellants like RP-1.
        RocketProps reports a liquid surrogate molecular weight, while CEA may
        use a normalized reactant formula such as ``C1 H1.95``.
        """
        value = self._cea_value("mw")

        if value is None or not np.isfinite(value):
            return None

        return float(value) / 1000.0

    @property
    def heat_of_formation_molar(self) -> float | None:
        """CEA reactant heat of formation at the reference state in J/mol."""
        value = self._cea_value("hf298")

        if value is None or not np.isfinite(value):
            return None

        return float(value)

    @property
    def heat_of_formation(self) -> float | None:
        """CEA reactant heat of formation at the reference state in J/kg."""
        h_molar = self.heat_of_formation_molar
        mw = self.cea_molar_mass

        if h_molar is None or mw is None or mw == 0.0:
            return None

        return h_molar / mw

    @property
    def enthalpy_of_formation(self) -> float | None:
        return self.heat_of_formation

    @property
    def reference_temperature(self) -> float | None:
        """CEA reactant reference temperature in K."""
        ranges = self._cea_value("t_ranges")

        if ranges is None:
            return None

        value = float(ranges[0, 0])

        if not np.isfinite(value):
            return None

        return value

    # ---------------- Static/reference properties ---------------- #

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        value = self._call("MolWt", "MolecularWt", "MolarMass", default=None)

        if value is None:
            return None

        return float(value) / 1000.0

    @property
    def critical_pressure(self) -> float:
        """Critical pressure in Pa."""
        value = self._call("Pc", "Pcrit", "P_crit", default=None)

        if value is None:
            return None

        return self._Pa_from_psia(value)

    @property
    def critical_temperature(self) -> float:
        """Critical temperature in K."""
        value = self._call("Tc", "Tcrit", "T_crit", default=None)

        if value is None:
            return None

        return self._K_from_degR(value)

    @property
    def critical_density(self) -> float:
        """Critical density in kg/m^3 when available."""
        value = self._call("SGc", "rhoc", "rho_crit", default=None)

        if value is None:
            return None

        return float(value) * 1000.0

    @property
    def freezing_temperature(self) -> float:
        """Freezing temperature in K when available."""
        value = self._call("Tfreeze", "Tfrz", "T_freeze", default=None)

        if value is None:
            return None

        return self._K_from_degR(value)

    @property
    def boiling_temperature(self) -> float:
        """Normal boiling temperature in K when available."""
        value = self._call("Tnbp", "Tboil", "T_boil", default=None)

        if value is None:
            return None

        return self._K_from_degR(value)

    @property
    def minimum_temperature(self) -> float:
        """Minimum valid correlation temperature in K when available."""
        data_range = getattr(self._backend, "T_data_range", None)

        if data_range is None:
            return self.freezing_temperature

        try:
            return self._K_from_degR(data_range()[0])
        except Exception:
            return self.freezing_temperature

    @property
    def maximum_temperature(self) -> float:
        """Maximum valid correlation temperature in K when available."""
        data_range = getattr(self._backend, "T_data_range", None)

        if data_range is None:
            return self.critical_temperature

        try:
            return self._K_from_degR(data_range()[1])
        except Exception:
            return self.critical_temperature

    @property
    def minimum_pressure(self) -> float:
        """Minimum valid compressed-liquid pressure in Pa when available."""
        data_range = getattr(self._backend, "P_data_range", None)

        if data_range is None:
            return 0.0

        try:
            return self._Pa_from_psia(data_range()[0])
        except Exception:
            return 0.0

    @property
    def maximum_pressure(self) -> float:
        """Maximum valid compressed-liquid pressure in Pa when available."""
        data_range = getattr(self._backend, "P_data_range", None)

        if data_range is None:
            return float("inf")

        try:
            return self._Pa_from_psia(data_range()[1])
        except Exception:
            return float("inf")

    @property
    def is_mixture(self) -> bool:
        """Return True for common RocketProps named mixture families."""
        name = self._propellant_name.upper()
        return name.startswith("MON") or name in {"A50", "M20", "MHF3"}

    # ---------------- String output ---------------- #

    def _safe(self, value, fmt=".3e"):
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def _safe_property(self, property_name: str, fmt=".3e"):
        """
        Safely format a property for string output.

        Unsupported properties display as N/A instead of raising while printing.
        Direct user access still raises NotImplementedError.
        """
        try:
            return self._safe(getattr(self, property_name), fmt)
        except NotImplementedError:
            return "N/A"

    def __str__(self):
        rows = [
            ("Propellant", self.propellant),
            ("Backend", self.backend),
            ("CEA reactant", self._safe(self.cea_reactant) if self.cea_reactant is not None else "N/A"),
            ("Phase", self.phase),
            ("Phase model", self.phase_model),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e") if self.pressure is not None else "Saturation"),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Specific volume [m³/kg]", self._safe(self.specific_volume, ".3e")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]", self._safe_property("internal_energy", ".3e")),
            ("Enthalpy [J/kg]", self._safe_property("enthalpy", ".3e")),
            ("Entropy [J/kg-K]", self._safe_property("entropy", ".3e")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Kinematic viscosity [m²/s]", self._safe(self.kinematic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Surface tension [N/m]", self._safe(self.surface_tension, ".3e")),
            ("Vapor pressure [Pa]", self._safe(self.vapor_pressure, ".3e")),
            ("Saturation temperature [K]", self._safe(self.saturation_temperature, ".2f")),
            ("Heat of vaporization [J/kg]", self._safe(self.heat_of_vaporization, ".3e")),
            ("Cp [J/kg-K]", self._safe(self.specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]", self._safe_property("specific_heat_cv", ".3f")),
            ("Specific heat ratio", self._safe_property("specific_heat_ratio", ".5f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("CEA molar mass [kg/mol]", self._safe(self.cea_molar_mass, ".6f")),
            ("CEA Hf [J/kg]", self._safe(self.heat_of_formation, ".3e")),
            ("CEA Tref [K]", self._safe(self.reference_temperature, ".2f")),
            ("Speed of sound [m/s]", self._safe_property("speed_of_sound", ".3f")),
        ]

        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)

    def __repr__(self) -> str:
        pressure = "None" if self.pressure is None else f"{self.pressure:.3e}"
        return (
            f"{self.__class__.__name__}(propellant={self.propellant!r}, "
            f"temperature={self.temperature:.2f} K, "
            f"pressure={pressure} Pa)"
        )

    # ---------------- Utilities ---------------- #

    @staticmethod
    def get_available_propellants() -> list[str]:
        """Return canonical registry names with RocketProps support."""
        return sorted(CombustionRegistry.propellant_supported_names)

    @staticmethod
    def show_available_propellants() -> list[str]:
        """Print and return common RocketProps propellant names."""
        names = Propellant.get_available_propellants()

        for name in names:
            print(name)

        return names

    @staticmethod
    def get_available_fluids() -> list[str]:
        """Return available RocketProps propellant names.

        Fluid-style alias for API consistency with Fluid.
        """
        return Propellant.get_available_propellants()

    @staticmethod
    def show_available_fluids() -> list[str]:
        """Print and return available RocketProps propellant names.

        Fluid-style alias for API consistency with Fluid.
        """
        return Propellant.show_available_propellants()

    @classmethod
    def show_aliases(cls) -> dict[str, str]:
        """Print and return RocketProps-specific propellant aliases."""
        aliases = CombustionRegistry.propellant_aliases

        if not aliases:
            return aliases

        width = max(len(alias) for alias in aliases)

        print("Propellant Aliases")
        print("-" * (width + 20))

        for alias, backend in sorted(aliases.items()):
            print(f"{alias:<{width}} -> {backend}")

        return dict(aliases)

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        """Return supported propellant state input combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
        )

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        """Return supported propellant state input combinations."""
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        """Return supported two-property propellant state input combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
            if len(inputs) == 2
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        """Return supported two-property propellant state input combinations."""
        return cls.available_flash_pairs()

    @classmethod
    def supported_properties(cls) -> list[str]:
        """Return public properties intentionally supported by this wrapper."""
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
        """Print and return public properties intentionally supported by this wrapper."""
        properties = cls.supported_properties()

        for prop in properties:
            print(prop)

        return properties

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        """Return True if this wrapper intentionally supports property_name."""
        return property_name in cls.supported_properties()