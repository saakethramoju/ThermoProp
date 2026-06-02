from typing import List, Union, Dict, Tuple

import numpy as np
from scipy.optimize import root_scalar
from functools import lru_cache

import CoolProp.CoolProp as CP

from .FluidRegistry import FluidRegistry


class Fluid:
    """
    High-level wrapper around CoolProp AbstractState objects with added
    convenience methods for initialization, state updates, and property access.

    Supports both pure fluids and mixtures by mole or mass fractions.

    Notes
    -----
    Uses full property names:

        Fluid(..., pressure=..., temperature=...)
        Fluid(..., pressure=..., enthalpy=...)
        Fluid(..., pressure=..., quality=...)
        Fluid(..., temperature=..., quality=...)
        Fluid(..., density=..., internal_energy=...)
        Fluid(..., pressure=..., density=...)
        Fluid(..., pressure=..., internal_energy=...)
        Fluid(..., temperature=..., density=...)
        Fluid(..., density=..., enthalpy=...)
        Fluid(..., temperature=..., enthalpy=...)

    Pair setters/getters:

        pressure_temperature
        pressure_enthalpy
        pressure_quality
        temperature_quality
        density_internal_energy
        pressure_density
        pressure_internal_energy
        temperature_density
        density_enthalpy
        temperature_enthalpy
    """

    _PHASE_NAMES = {
        getattr(CP, "iphase_unknown", -999): "Unknown",
        getattr(CP, "iphase_liquid", -999): "Liquid",
        getattr(CP, "iphase_supercritical", -999): "Supercritical",
        getattr(CP, "iphase_supercritical_gas", -999): "SupercriticalGas",
        getattr(CP, "iphase_supercritical_liquid", -999): "SupercriticalLiquid",
        getattr(CP, "iphase_gas", -999): "Gas",
        getattr(CP, "iphase_twophase", -999): "TwoPhase",
        getattr(CP, "iphase_critical_point", -999): "CriticalPoint",
    }

    _FLASH_PAIRS = {
        frozenset(("pressure", "temperature")): (
            "PT_INPUTS",
            ("pressure", "temperature"),
        ),
        frozenset(("pressure", "enthalpy")): (
            "HmassP_INPUTS",
            ("enthalpy", "pressure"),
        ),
        frozenset(("pressure", "quality")): (
            "PQ_INPUTS",
            ("pressure", "quality"),
        ),
        frozenset(("temperature", "quality")): (
            "QT_INPUTS",
            ("quality", "temperature"),
        ),
        frozenset(("density", "internal_energy")): (
            "DmassUmass_INPUTS",
            ("density", "internal_energy"),
        ),
        frozenset(("pressure", "density")): (
            "DmassP_INPUTS",
            ("density", "pressure"),
        ),
        frozenset(("pressure", "internal_energy")): (
            "PUmass_INPUTS",
            ("pressure", "internal_energy"),
        ),
        frozenset(("temperature", "density")): (
            "DmassT_INPUTS",
            ("density", "temperature"),
        ),
        frozenset(("density", "enthalpy")): (
            "DmassHmass_INPUTS",
            ("density", "enthalpy"),
        ),
        frozenset(("temperature", "enthalpy")): (
            "HmassT_INPUTS",
            ("enthalpy", "temperature"),
        ),
    }

    def __init__(
        self,
        fluid: Union[str, Dict[str, float]],
        basis: str = "mass",
        pressure: float = None,
        enthalpy: float = None,
        temperature: float = None,
        quality: float = None,
        density: float = None,
        internal_energy: float = None,
    ):
        """
        Initialize a Fluid state.
        """
        valid_fluids = Fluid.get_available_fluids()

        self._fluids: List[str] = []
        self._display_names: List[str] = []

        if isinstance(fluid, str):
            backend, display = Fluid._normalize_name(fluid)
            if backend not in valid_fluids:
                raise ValueError(
                    f"Invalid fluid '{fluid}'. "
                    f"Use Fluid.show_available_fluids() to check valid names."
                )
            self._fluids = [backend]
            self._display_names = [display]
            self._mole_fractions = np.array([1.0])
            self._mass_fractions = np.array([1.0])
            self._mixture = False

        elif isinstance(fluid, dict):
            if len(fluid) == 1:
                f, frac = next(iter(fluid.items()))
                if not np.isclose(frac, 1.0, atol=1e-12):
                    raise ValueError(f"Single-component dict must have fraction = 1.0, got {frac}")
                backend, display = Fluid._normalize_name(f)
                if backend not in valid_fluids:
                    raise ValueError(
                        f"Invalid fluid '{f}'. "
                        f"Use Fluid.show_available_fluids() to check valid names."
                    )
                self._fluids = [backend]
                self._display_names = [display]
                self._mole_fractions = np.array([1.0])
                self._mass_fractions = np.array([1.0])
                self._mixture = False

            else:
                tmp: Dict[str, Tuple[float, List[str]]] = {}
                for user_name, frac in fluid.items():
                    backend, display = Fluid._normalize_name(user_name)
                    if backend not in valid_fluids:
                        raise ValueError(
                            f"Invalid fluid '{user_name}' (backend '{backend}' not found). "
                            f"Use Fluid.show_available_fluids() to check valid names."
                        )
                    total, names = tmp.get(backend, (0.0, []))
                    tmp[backend] = (total + float(frac), names + [display])

                self._fluids = list(tmp.keys())
                fractions = np.array([v[0] for v in tmp.values()], dtype=float)
                self._display_names = [", ".join(sorted(set(names))) for _, names in tmp.values()]

                if basis == "mole":
                    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                        raise ValueError("Mole fractions must sum to 1.0")
                    self._mole_fractions = fractions
                    self._mass_fractions = Fluid.mole_to_mass(self._fluids, fractions)
                elif basis == "mass":
                    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                        raise ValueError("Mass fractions must sum to 1.0")
                    self._mass_fractions = fractions
                    self._mole_fractions = Fluid.mass_to_mole(self._fluids, fractions)
                else:
                    raise ValueError("basis must be 'mole' or 'mass'")

                self._mixture = len(self._fluids) > 1

        else:
            raise TypeError("fluid must be a string (pure) or dict (mixture)")

        self._P = None
        self._h = None
        self._last_state_values: dict | None = None
        self._fluid_string = "&".join(self._fluids)
        self._backend = self._build_state()
        self._pyfluid = self._backend

        flash_values = {
            "pressure": pressure,
            "enthalpy": enthalpy,
            "temperature": temperature,
            "quality": quality,
            "density": density,
            "internal_energy": internal_energy,
        }

        provided = {
            key: value
            for key, value in flash_values.items()
            if value is not None
        }

        if len(provided) != 2:
            raise LookupError(
                "Please provide exactly two thermodynamic properties. "
                "Supported names are pressure, temperature, enthalpy, "
                "quality, density, and internal_energy."
            )

        self._set_state_from_named_pair(provided)

    # ---------------- Core ---------------- #
    def _build_state(self):
        """Create and configure a CoolProp AbstractState."""
        state = CP.AbstractState("HEOS", self._fluid_string)
        if self._mixture:
            state.set_mass_fractions([float(x) for x in self._mass_fractions])
        return state

    def _sync_from_backend(self):
        """Synchronize cached pressure and enthalpy from the backend state."""
        self._P = float(self._backend.p())
        self._h = float(self._backend.hmass())
        self._pyfluid = self._backend

    def _set_state_from_named_pair(self, values: dict) -> None:
        self._last_state_values = dict(values)
        keys = frozenset(values.keys())

        if keys not in Fluid._FLASH_PAIRS:
            raise ValueError(
                f"Unsupported flash pair: {sorted(keys)}. "
                f"Supported pairs are: {self.available_flash_pairs()}."
            )

        if keys == frozenset(("pressure", "enthalpy")) and self._mixture:
            pressure = float(values["pressure"])
            enthalpy = float(values["enthalpy"])

            T, Q = Fluid.get_temperature_and_quality(
                self._backend,
                pressure,
                enthalpy,
            )

            if 0.0 < Q < 1.0:
                self._update_state(CP.PQ_INPUTS, pressure, Q)
            else:
                self._update_state(CP.PT_INPUTS, pressure, T)

            self._sync_from_backend()
            return

        input_pair_name, order = Fluid._FLASH_PAIRS[keys]

        if not hasattr(CP, input_pair_name):
            raise ValueError(
                f"CoolProp does not expose {input_pair_name} in this installation."
            )

        input_pair = getattr(CP, input_pair_name)
        value1 = values[order[0]]
        value2 = values[order[1]]

        self._update_state(input_pair, value1, value2)
        self._sync_from_backend()



    def set_pyfluid(self):
        """Rebuild backend CoolProp state using current pressure and enthalpy."""
        if self._mixture:
            T, Q = Fluid.get_temperature_and_quality(self._backend, self._P, self._h)
            if 0.0 < Q < 1.0:
                self._update_state(CP.PQ_INPUTS, self._P, Q)
            else:
                self._update_state(CP.PT_INPUTS, self._P, T)
        else:
            self._update_state(CP.HmassP_INPUTS, self._h, self._P)

    # ---------------- Internal state helpers ---------------- #
    def _enthalpy_from_pressure_temperature(self, pressure: float, temperature: float) -> float:
        self._set_state_from_named_pair(
            {
                "pressure": pressure,
                "temperature": temperature,
            }
        )
        return float(self._backend.hmass())

    def _enthalpy_from_pressure_quality(self, pressure: float, quality: float) -> float:
        self._set_state_from_named_pair(
            {
                "pressure": pressure,
                "quality": quality,
            }
        )
        return float(self._backend.hmass())

    def _state_from_temperature_quality(self, temperature: float, quality: float) -> Tuple[float, float]:
        self._set_state_from_named_pair(
            {
                "temperature": temperature,
                "quality": quality,
            }
        )
        return float(self._backend.p()), float(self._backend.hmass())

    def _state_from_density_internal_energy(self, density: float, internal_energy: float) -> None:
        self._set_state_from_named_pair(
            {
                "density": density,
                "internal_energy": internal_energy,
            }
        )

    def _keyed_output(self, key, default=None):
        try:
            return float(self._backend.keyed_output(key))
        except Exception:
            return default

    def _trivial_output(self, key, default=None):
        try:
            return float(self._backend.trivial_keyed_output(key))
        except Exception:
            return default



    def _update_state(self, input_pair, value1: float, value2: float):
        """Update CoolProp state and keep the old internal alias current."""

        try:
            self._backend.update(input_pair, float(value1), float(value2))

        except Exception:
            if self._mixture:
                try:
                    self._backend.specify_phase(CP.iphase_gas)
                    self._backend.update(input_pair, float(value1), float(value2))
                    self._backend.unspecify_phase()
                except Exception:
                    self._backend.unspecify_phase()
                    raise
            else:
                raise

        self._pyfluid = self._backend
    # ---------------- Fractions ---------------- #
    @property
    def mole_fractions(self) -> dict:
        """Return mole fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mole_fractions)}

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        """Update mole fractions. Fractions must sum to 1."""
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mole fractions for a pure fluid")

        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")

        self._mole_fractions = np.array(value, dtype=float)
        self._mass_fractions = Fluid.mole_to_mass(self._fluids, value)

        self._backend = self._build_state()
        self._pyfluid = self._backend

        if self._last_state_values is not None:
            self._set_state_from_named_pair(self._last_state_values)

    @property
    def mass_fractions(self) -> dict:
        """Return mass fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mass_fractions)}

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        """Update mass fractions. Fractions must sum to 1."""
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mass fractions for a pure fluid")

        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")

        self._mass_fractions = np.array(value, dtype=float)
        self._mole_fractions = Fluid.mass_to_mole(self._fluids, value)

        self._backend = self._build_state()
        self._pyfluid = self._backend

        if self._last_state_values is not None:
            self._set_state_from_named_pair(self._last_state_values)

    # ---------------- State setters ---------------- #
    @property
    def pressure(self) -> float:
        """Absolute pressure in Pa."""
        return self._P

    @pressure.setter
    def pressure(self, value: float):
        self.pressure_enthalpy = (float(value), self.enthalpy)

    @property
    def enthalpy(self) -> float:
        """Mass-specific enthalpy in J/kg."""
        return self._h

    @enthalpy.setter
    def enthalpy(self, value: float):
        self.pressure_enthalpy = (self.pressure, float(value))

    @property
    def pressure_enthalpy(self) -> Tuple[float, float]:
        """Return (pressure [Pa], enthalpy [J/kg])."""
        return self.pressure, self.enthalpy

    @pressure_enthalpy.setter
    def pressure_enthalpy(self, values: Tuple[float, float]):
        """Update state from pressure and enthalpy."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_enthalpy must be set with (pressure, enthalpy)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    @property
    def pressure_temperature(self) -> Tuple[float, float]:
        """Return (pressure [Pa], temperature [K])."""
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float, float]):
        """Update state from pressure and temperature."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "temperature": float(values[1]),
            }
        )

    @property
    def pressure_quality(self) -> Tuple[float, float]:
        """Return (pressure [Pa], quality [-])."""
        return self.pressure, self.quality

    @pressure_quality.setter
    def pressure_quality(self, values: Tuple[float, float]):
        """Update state from pressure and quality."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_quality must be set with (pressure, quality)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "quality": float(values[1]),
            }
        )

    @property
    def temperature_quality(self) -> Tuple[float, float]:
        """Return (temperature [K], quality [-])."""
        return self.temperature, self.quality

    @temperature_quality.setter
    def temperature_quality(self, values: Tuple[float, float]):
        """Update state from temperature and quality."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_quality must be set with (temperature, quality)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "quality": float(values[1]),
            }
        )

    @property
    def density_internal_energy(self) -> Tuple[float, float]:
        """Return (density [kg/m^3], internal_energy [J/kg])."""
        return self.density, self.internal_energy

    @density_internal_energy.setter
    def density_internal_energy(self, values: Tuple[float, float]):
        """Update state from density and internal energy."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_internal_energy must be set with (density, internal_energy)")

        self._set_state_from_named_pair(
            {
                "density": float(values[0]),
                "internal_energy": float(values[1]),
            }
        )

    @property
    def pressure_density(self) -> Tuple[float, float]:
        """Return (pressure [Pa], density [kg/m^3])."""
        return self.pressure, self.density

    @pressure_density.setter
    def pressure_density(self, values: Tuple[float, float]):
        """Update state from pressure and density."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_density must be set with (pressure, density)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "density": float(values[1]),
            }
        )

    @property
    def pressure_internal_energy(self) -> Tuple[float, float]:
        """Return (pressure [Pa], internal_energy [J/kg])."""
        return self.pressure, self.internal_energy

    @pressure_internal_energy.setter
    def pressure_internal_energy(self, values: Tuple[float, float]):
        """Update state from pressure and internal energy."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_internal_energy must be set with (pressure, internal_energy)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "internal_energy": float(values[1]),
            }
        )

    @property
    def temperature_density(self) -> Tuple[float, float]:
        """Return (temperature [K], density [kg/m^3])."""
        return self.temperature, self.density

    @temperature_density.setter
    def temperature_density(self, values: Tuple[float, float]):
        """Update state from temperature and density."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_density must be set with (temperature, density)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "density": float(values[1]),
            }
        )

    @property
    def density_enthalpy(self) -> Tuple[float, float]:
        """Return (density [kg/m^3], enthalpy [J/kg])."""
        return self.density, self.enthalpy

    @density_enthalpy.setter
    def density_enthalpy(self, values: Tuple[float, float]):
        """Update state from density and enthalpy."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_enthalpy must be set with (density, enthalpy)")

        self._set_state_from_named_pair(
            {
                "density": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    @property
    def temperature_enthalpy(self) -> Tuple[float, float]:
        """Return (temperature [K], enthalpy [J/kg])."""
        return self.temperature, self.enthalpy

    @temperature_enthalpy.setter
    def temperature_enthalpy(self, values: Tuple[float, float]):
        """Update state from temperature and enthalpy."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_enthalpy must be set with (temperature, enthalpy)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    # ---------------- Thermo properties ---------------- #
    @property
    def species(self) -> List[str]:
        return self._display_names

    @property
    def temperature(self) -> float:
        """Absolute temperature in K."""
        return float(self._backend.T())

    @temperature.setter
    def temperature(self, value: float):
        """
        Update temperature while holding pressure constant.

        Requires pressure to already be defined.
        """
        if self._P is None:
            raise ValueError(
                "Cannot set temperature without pressure. "
                "Set pressure first."
            )

        self.pressure_temperature = (self.pressure, float(value))

    @property
    def phase(self) -> str:
        """Thermodynamic phase name, mapped to the old pyfluids-style names."""

        if self._mixture:
            try:
                Z = self.compressibility
                if Z is not None and Z > 0.2:
                    return "Gas"
            except Exception:
                pass

        try:
            return Fluid._PHASE_NAMES.get(int(self._backend.phase()), "Unknown")
        except Exception:
            return "Unknown"

    @property
    def compressibility(self) -> float:
        """Compressibility factor Z."""
        return self._keyed_output(CP.iZ)

    @property
    def conductivity(self) -> float:
        """Thermal conductivity in W/m-K."""
        try:
            return float(self._backend.conductivity())
        except Exception:
            return None

    @property
    def critical_pressure(self) -> float:
        """Critical pressure in Pa."""
        try:
            return float(self._backend.p_critical())
        except Exception:
            return self._trivial_output(CP.iP_critical)

    @property
    def critical_temperature(self) -> float:
        """Critical temperature in K."""
        try:
            return float(self._backend.T_critical())
        except Exception:
            return self._trivial_output(CP.iT_critical)

    @property
    def density(self) -> float:
        """Mass density in kg/m^3."""
        return float(self._backend.rhomass())

    @density.setter
    def density(self, value: float):
        """
        Update density while holding internal energy constant.
        """
        self.density_internal_energy = (float(value), self.internal_energy)

    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity in Pa-s."""
        try:
            return float(self._backend.viscosity())
        except Exception:
            return None

    @property
    def entropy(self) -> float:
        """Mass-specific entropy in J/kg-K."""
        return float(self._backend.smass())

    @property
    def freezing_temperature(self) -> float:
        """Freezing/melting temperature in K when available; otherwise Tmin."""
        return self.minimum_temperature

    @property
    def internal_energy(self) -> float:
        """Mass-specific internal energy in J/kg."""
        return float(self._backend.umass())

    @internal_energy.setter
    def internal_energy(self, value: float):
        """
        Update internal energy while holding density constant.
        """
        self.density_internal_energy = (self.density, float(value))

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity in m^2/s."""
        mu = self.dynamic_viscosity
        rho = self.density
        if mu is None or rho is None or rho == 0:
            return None
        return mu / rho

    @property
    def maximum_pressure(self) -> float:
        """Maximum valid pressure in Pa."""
        try:
            return float(self._backend.pmax())
        except Exception:
            return self._trivial_output(CP.iP_max)

    @property
    def maximum_temperature(self) -> float:
        """Maximum valid temperature in K."""
        try:
            return float(self._backend.Tmax())
        except Exception:
            return self._trivial_output(CP.iT_max)

    @property
    def minimum_pressure(self) -> float:
        """Minimum valid pressure in Pa."""
        try:
            return float(self._backend.p_triple())
        except Exception:
            return self._trivial_output(CP.iP_triple)

    @property
    def minimum_temperature(self) -> float:
        """Minimum valid temperature in K."""
        try:
            return float(self._backend.Tmin())
        except Exception:
            return self._trivial_output(CP.iT_min)

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        return float(self._backend.molar_mass())

    @property
    def prandtl(self) -> float:
        """Prandtl number."""
        return self._keyed_output(CP.iPrandtl)

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound in m/s."""
        try:
            return float(self._backend.speed_sound())
        except Exception:
            return None

    @property
    def specific_heat_cp(self) -> float:
        """Cp in J/kg-K."""
        try:
            return float(self._backend.cpmass())
        except Exception:
            return None

    @property
    def specific_heat_cv(self) -> float:
        """Cv in J/kg-K."""
        try:
            return float(self._backend.cvmass())
        except Exception:
            return None

    @property
    def specific_heat(self) -> float:
        """Backward-compatible alias for Cp."""
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        """
        Specific heat ratio gamma = Cp/Cv.
        """
        try:
            cp = self.specific_heat_cp
            cv = self.specific_heat_cv

            if cp is None or cv is None or cv == 0.0:
                return None

            return cp / cv

        except Exception:
            return None

    @property
    def specific_volume(self) -> float:
        """Specific volume in m^3/kg."""
        rho = self.density
        if rho is None or rho == 0:
            return None
        return 1.0 / rho

    @property
    def surface_tension(self) -> float:
        """Surface tension in N/m when available."""
        try:
            return float(self._backend.surface_tension())
        except Exception:
            return None

    @property
    def triple_pressure(self) -> float:
        """Triple point pressure in Pa."""
        try:
            return float(self._backend.p_triple())
        except Exception:
            return self._trivial_output(CP.iP_triple)

    @property
    def triple_temperature(self) -> float:
        """Triple point temperature in K."""
        try:
            return float(self._backend.Ttriple())
        except Exception:
            return self._trivial_output(CP.iT_triple)

    @property
    def is_mixture(self) -> bool:
        """Return True if this fluid is a mixture, False if pure."""
        return self._mixture

    @property
    def quality(self) -> float:
        """
        Vapor quality from 0 to 1.

        For gas mixtures, return 1.0 if phase is gas-like.
        """

        ph = self.phase

        if ph == "TwoPhase":
            try:
                return float(self._backend.Q())
            except Exception:
                return float("nan")

        if ph in ("Gas", "Supercritical", "SupercriticalGas"):
            return 1.0

        if ph in ("Liquid", "SupercriticalLiquid"):
            return 0.0

        return float("nan")

    @quality.setter
    def quality(self, value: float):
        """
        Update vapor quality while holding pressure constant.

        Requires pressure to already be defined.
        """
        if self._P is None:
            raise ValueError(
                "Cannot set quality without pressure. "
                "Set pressure first."
            )

        Q = float(value)

        if not (0.0 <= Q <= 1.0):
            raise ValueError("Quality must be between 0 and 1.")

        self.pressure_quality = (self.pressure, Q)


    @property
    def saturation_pressure(self) -> float:
        """
        Saturation pressure in Pa for current temperature,
        only if temperature <= critical temperature.
        """
        Tc = self.critical_temperature

        if Tc is not None and self.temperature > Tc:
            return None

        try:
            tmp = self._build_state()
            tmp.update(CP.QT_INPUTS, 1.0, self.temperature)
            return float(tmp.p())

        except Exception:
            return None

    @property
    def saturation_temperature(self) -> float:
        """Saturation temperature in K for current pressure, only if pressure <= Pc."""
        pc = self.critical_pressure
        if pc is not None and self.pressure > pc:
            return None
        try:
            tmp = self._build_state()
            tmp.update(CP.PQ_INPUTS, self._P, 1.0)
            return float(tmp.T())
        except Exception:
            return None

    # ---------------- String output ---------------- #
    def _safe(self, value, fmt=".3e"):
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def __str__(self):
        def format_dict(d: dict, decimals=3):
            return {k: round(v, decimals) for k, v in d.items()}

        rows = [
            ("Fluid(s)", ", ".join(self._display_names)),
            ("Mole fractions", format_dict(self.mole_fractions, 3)),
            ("Mass fractions", format_dict(self.mass_fractions, 3)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Saturation temperature [K]", self._safe(self.saturation_temperature, ".2f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("Speed of sound [m/s]", self._safe(self.speed_of_sound, ".6f"))
        ]
        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)

    def __repr__(self) -> str:
        species_str = ", ".join(self._display_names)
        return (
            f"{self.__class__.__name__}(species=[{species_str}], "
            f"pressure={self._P:.3e} Pa, "
            f"enthalpy={self._h:.3e} J/kg, "
            f"temperature={self.temperature:.2f} K)"
        )

    # ---------------- Utilities ---------------- #
    @classmethod
    def _normalize_name(cls, user_name: str) -> Tuple[str, str]:
        """
        Return (backend_name, display_name). If user_name is an alias, map it to
        the CoolProp backend name but preserve user input for display.
        """
        backend = FluidRegistry.coolprop_name(user_name)
        display = FluidRegistry.name(user_name)
        return backend, display


    @staticmethod
    @lru_cache(maxsize=None)
    def _molar_mass_of(fluid: str) -> float:
        """Return pure-fluid molar mass in kg/mol."""
        return float(CP.PropsSI("M", fluid))

    @staticmethod
    def mole_to_mass(fluids: List[str], mole_fractions: List[float]):
        """Convert mole fractions to mass fractions."""
        if not np.isclose(sum(mole_fractions), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")
        mole_fractions = np.asarray(mole_fractions, dtype=float)
        molar_masses = np.array([Fluid._molar_mass_of(f) for f in fluids])
        m_bar = np.dot(mole_fractions, molar_masses)
        return mole_fractions * molar_masses / m_bar

    @staticmethod
    def mass_to_mole(fluids: List[str], mass_fractions: List[float]):
        """Convert mass fractions to mole fractions."""
        if not np.isclose(sum(mass_fractions), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")
        mass_fractions = np.asarray(mass_fractions, dtype=float)
        molar_masses = np.array([Fluid._molar_mass_of(f) for f in fluids])
        inv = mass_fractions / molar_masses
        return inv / inv.sum()

    @staticmethod
    def get_temperature_and_quality(fluid, pressure: float, target_enthalpy: float) -> Tuple[float, float]:
        """
        Given a CoolProp AbstractState, pressure, and enthalpy, return (T, Q).

        For mixtures this avoids relying on direct P-H flashes through the dome.
        It mirrors the old pyfluids workaround: compare target enthalpy to the
        saturated liquid/vapor enthalpies, then solve T at fixed P outside dome.
        """
        try:
            fluid.update(CP.PQ_INPUTS, pressure, 0.0)
            h_liquid = float(fluid.hmass())
            T_sat = float(fluid.T())

            fluid.update(CP.PQ_INPUTS, pressure, 1.0)
            h_vapor = float(fluid.hmass())
        except Exception:
            h_liquid = None
            h_vapor = None
            T_sat = None

        h = float(target_enthalpy)

        if h_liquid is not None and h_vapor is not None and h_liquid <= h <= h_vapor:
            denom = h_vapor - h_liquid
            Q = 0.0 if abs(denom) < 1e-15 else (h - h_liquid) / denom
            return T_sat, float(Q)

        def residual(T):
            try:
                fluid.update(CP.PT_INPUTS, pressure, T)
                return float(fluid.hmass()) - h
            except Exception:
                return np.nan

        try:
            Tmin = float(fluid.Tmin())
        except Exception:
            Tmin = 1.0
        try:
            Tmax = float(fluid.Tmax())
        except Exception:
            Tmax = 5000.0

        Ts = np.linspace(Tmin * 1.000001, Tmax * 0.999999, 300)
        vals = []
        for T in Ts:
            r = residual(T)
            vals.append(r if np.isfinite(r) else np.nan)

        bracket = None
        for T1, T2, r1, r2 in zip(Ts[:-1], Ts[1:], vals[:-1], vals[1:]):
            if not (np.isfinite(r1) and np.isfinite(r2)):
                continue
            if r1 == 0:
                bracket = (T1, T1)
                break
            if r1 * r2 <= 0:
                bracket = (T1, T2)
                break

        if bracket is None:
            raise ValueError(
                f"Could not find a valid temperature bracket for pressure={pressure:.6g} Pa, "
                f"enthalpy={h:.6g} J/kg over T=[{Tmin:.6g}, {Tmax:.6g}] K."
            )

        if bracket[0] == bracket[1]:
            T = bracket[0]
        else:
            sol = root_scalar(residual, method="brentq", bracket=bracket)
            T = float(sol.root)

        if h_liquid is not None and h < h_liquid:
            Q = 0.0
        elif h_vapor is not None and h > h_vapor:
            Q = 1.0
        else:
            Q = float("nan")

        return T, Q

    @staticmethod
    def show_available_fluids():
        """Print and return available CoolProp fluid names."""
        fluids = Fluid.get_available_fluids()
        for f in fluids:
            print(f)
        return fluids

    @staticmethod
    def get_available_fluids():
        """Return available CoolProp fluid names."""
        return sorted(
            FluidRegistry.coolprop_name(name)
            for name in FluidRegistry.coolprop_supported_names
        )

    @staticmethod
    def available_flash_pairs():
        return sorted(
            "-".join(sorted(pair))
            for pair in Fluid._FLASH_PAIRS
        )

