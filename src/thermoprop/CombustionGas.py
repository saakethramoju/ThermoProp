from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import root_scalar

from .CEADatabase import CEA
from .SpeciesDatabase import SpeciesDatabase
from .ReferenceState import normalize_reference_target
from ._api import PropertyIntrospectionMixin
from ._formatting import format_optional, rounded_dict, format_rows
from ._validation import validate_fraction_vector

class CombustionGas(PropertyIntrospectionMixin):
    """
    NASA CEA / CEAM gas-mixture wrapper with ThermoProp's common property API.

    CombustionGas evaluates gas-phase thermodynamic and transport properties for
    CEA product species and mixtures. Thermodynamic properties are evaluated from
    CEA NASA-9 polynomial data. Transport properties use generated CEA / CEAM
    transport data when available.

    Supported state inputs
    ----------------------

    One thermal state input may be supplied:

        CombustionGas(..., temperature=T)
        CombustionGas(..., enthalpy=h)
        CombustionGas(..., internal_energy=u)

    Pressure may be supplied with the thermal state:

        CombustionGas(..., pressure=P, temperature=T)
        CombustionGas(..., pressure=P, enthalpy=h)
        CombustionGas(..., pressure=P, internal_energy=u)

    Density may be used with one closure variable:

        CombustionGas(..., pressure=P, density=rho)
        CombustionGas(..., density=rho, temperature=T)
        CombustionGas(..., density=rho, enthalpy=h)
        CombustionGas(..., density=rho, internal_energy=u)

    Mixtures
    --------

    Mixtures are passed as dictionaries:

        CombustionGas({"CO2": 0.4, "H2O": 0.6}, basis="mole", pressure=P, temperature=T)

    The basis may be "mass" or "mole".

    Limitations
    -----------

    CombustionGas accepts only gas-phase CEA product species with NASA-9 polynomial
    thermodynamic data. CEA reactant cards and condensed species should be handled
    with Propellant or Equilibrium instead.

    CombustionGas does not model real-fluid phase behavior or vapor quality.

    Reference states
    ----------------

    CEA defines its own thermodynamic reference state. Absolute enthalpy, internal
    energy, entropy, Gibbs energy, and Helmholtz/free energy should not be compared
    directly with other wrappers unless a common reference is selected.

    Use set_reference="Fluid", "IdealGas", or "Propellant" to apply constant
    offsets at 298.15 K and 101325 Pa.

    This aligns only the reference values. It does not change CEA thermodynamic
    curves, mixture behavior, transport properties, or ideal-gas assumptions.

    Public API units are SI.
    """

    _BACKEND_NAME = "NASA CEA / CEAM"

    _RU = 8.31446261815324
    _RU_KMOL = 8314.46261815324
    _P_REF = 100000.0

    _REFERENCE_TEMPERATURE = 298.15
    _REFERENCE_PRESSURE = 101325.0
    _REFERENCE_CACHE: dict[tuple, tuple[float, float, float]] = {}


    _UNSUPPORTED_PROPERTIES = {
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
        frozenset(("temperature",)),
        frozenset(("enthalpy",)),
        frozenset(("internal_energy",)),
        frozenset(("pressure", "density")),
        frozenset(("pressure", "temperature")),
        frozenset(("pressure", "enthalpy")),
        frozenset(("pressure", "internal_energy")),
        frozenset(("density", "temperature")),
        frozenset(("density", "enthalpy")),
        frozenset(("density", "internal_energy")),
    }

    def __init__(
        self,
        fluid: Union[str, Dict[str, float]],
        basis: str = "mass",
        pressure: float | None = None,
        enthalpy: float | None = None,
        temperature: float | None = None,
        internal_energy: float | None = None,
        density: float | None = None,
        quality: float | None = None,
        set_reference: str | None = None,
    ):
        if quality is not None:
            raise ValueError("CombustionGas does not support vapor quality.")

        self._reference_target = self._normalize_reference_target(set_reference)
        self._reference_offsets: tuple[float, float, float] | None = None

        if basis not in ("mole", "mass"):
            raise ValueError("basis must be 'mole' or 'mass'.")

        self._species_names: List[str] = []
        self._display_names: List[str] = []
        self._thermo_indices: List[int] = []
        self._transport_indices: List[int | None] = []
        self._property_cache: dict[str, object] = {}

        if isinstance(fluid, str):
            species_name, display_name, thermo_index, transport_index = self._resolve_species(fluid)
            self._species_names = [species_name]
            self._display_names = [display_name]
            self._thermo_indices = [thermo_index]
            self._transport_indices = [transport_index]
            self._mole_fractions = np.array([1.0], dtype=float)
            self._mass_fractions = np.array([1.0], dtype=float)
            self._mixture = False

        elif isinstance(fluid, dict):
            if not fluid:
                raise ValueError("composition cannot be empty.")

            tmp: dict[str, tuple[float, str, int, int | None, list[str]]] = {}

            for user_name, frac in fluid.items():
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
            fractions = self._validate_fractions(
                fractions,
                f"{basis.capitalize()} fractions",
            )

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
            raise TypeError("fluid must be a species name or a dict of fractions.")

        self._M = np.array([CEA.molecular_weight(name) for name in self._species_names], dtype=float)
        self._minimum_temperature, self._maximum_temperature = self._temperature_limits()

        self._pressure: float | None = None
        self._enthalpy: float | None = None
        self._temperature: float | None = None
        self._last_state_values: dict | None = None

        self._set_state(
            pressure=pressure,
            temperature=temperature,
            enthalpy=enthalpy,
            internal_energy=internal_energy,
            density=density,
        )

    @staticmethod
    def _validate_fractions(fractions, label: str, *, atol: float = 1e-6) -> np.ndarray:
        """Validate a mass- or mole-fraction vector and return it as an array."""
        return validate_fraction_vector(fractions, label, atol=atol)


    # ---------------- Reference-state matching ---------------- #
    @classmethod
    def _normalize_reference_target(cls, value):
        return normalize_reference_target(value, cls.__name__)

    @property
    def reference(self) -> str:
        return self._reference_target or self.__class__.__name__

    @property
    def set_reference(self) -> str:
        return self.reference

    def _composition_cache_key(self) -> tuple:
        return tuple(
            sorted(
                (name, round(float(w), 15))
                for name, w in zip(self._display_names, self._mass_fractions)
            )
        )

    def _composition_argument(self) -> str | dict[str, float]:
        if len(self._display_names) == 1:
            return self._display_names[0]
        return {name: float(w) for name, w in zip(self._display_names, self._mass_fractions)}

    def _reference_cache_key(self) -> tuple:
        return (
            "CombustionGas",
            self._reference_target,
            self._composition_cache_key(),
            self._REFERENCE_TEMPERATURE,
            self._REFERENCE_PRESSURE,
        )

    def _raw_entropy_at(self, temperature: float, pressure: float) -> float:
        p_i = self._mole_fractions * float(pressure)
        p_i = np.maximum(p_i, np.finfo(float).tiny)
        pressure_correction_molar = -self._RU_KMOL * np.log(p_i / self._P_REF)
        pure_s0_molar = np.array(
            [CEA.entropy_molar_standard(name, temperature) for name in self._species_names],
            dtype=float,
        )
        pure_s_mass_at_pi = (pure_s0_molar + pressure_correction_molar) / self._M
        return float(np.dot(self._mass_fractions, pure_s_mass_at_pi))

    def _raw_reference_properties(self) -> tuple[float, float, float]:
        T = self._REFERENCE_TEMPERATURE
        P = self._REFERENCE_PRESSURE
        h = self._enthalpy_from_temperature(T)
        u = self._internal_energy_from_temperature(T)
        s = self._raw_entropy_at(T, P)
        return h, u, s

    def _target_reference_properties(self) -> tuple[float, float, float]:
        target = self._reference_target
        if target is None:
            return self._raw_reference_properties()
        fluid = self._composition_argument()
        T = self._REFERENCE_TEMPERATURE
        P = self._REFERENCE_PRESSURE
        if target == "Fluid":
            from .Fluid import Fluid
            obj = Fluid(fluid, basis="mass", pressure=P, temperature=T, set_reference=None)
        elif target == "IdealGas":
            from .IdealGas import IdealGas
            obj = IdealGas(fluid, basis="mass", pressure=P, temperature=T, set_reference=None)
        elif target == "Propellant":
            from .Propellant import Propellant
            if not isinstance(fluid, str):
                raise ValueError("set_reference='Propellant' is only supported for pure CombustionGas species.")
            obj = Propellant(fluid, pressure=P, temperature=T, set_reference=None)
        else:
            raise ValueError(f"Unsupported reference target: {target!r}")
        return float(obj.enthalpy), float(obj.internal_energy), float(obj.entropy)

    def _get_reference_offsets(self) -> tuple[float, float, float]:
        if self._reference_target is None:
            return 0.0, 0.0, 0.0
        if self._reference_offsets is not None:
            return self._reference_offsets
        key = self._reference_cache_key()
        cached = self._REFERENCE_CACHE.get(key)
        if cached is not None:
            self._reference_offsets = cached
            return cached
        raw_h, raw_u, raw_s = self._raw_reference_properties()
        ref_h, ref_u, ref_s = self._target_reference_properties()
        offsets = (ref_h - raw_h, ref_u - raw_u, ref_s - raw_s)
        self._REFERENCE_CACHE[key] = offsets
        self._reference_offsets = offsets
        return offsets

    def _clear_reference_cache(self) -> None:
        self._reference_offsets = None

    def _to_raw_basis(self, name: str, value: float) -> float:
        if self._reference_target is None:
            return float(value)
        dh, du, _ = self._get_reference_offsets()
        if name == "enthalpy":
            return float(value) - dh
        if name == "internal_energy":
            return float(value) - du
        return float(value)

    def _from_raw_basis(self, name: str, value: float | None) -> float | None:
        if value is None:
            return None
        if self._reference_target is None:
            return float(value)
        dh, du, ds = self._get_reference_offsets()
        T = self._temperature
        if name == "enthalpy":
            return float(value) + dh
        if name == "internal_energy":
            return float(value) + du
        if name == "entropy":
            return float(value) + ds
        if name == "gibbs_energy":
            return float(value) + dh - T * ds
        if name in {"free_energy", "helmholtz_energy"}:
            return float(value) + du - T * ds
        return float(value)

    @classmethod
    def _resolve_species(cls, value: str) -> tuple[str, str, int, int | None]:
        """Resolve a user species name/alias to a CEA gas species and table indices."""
        raw_name = str(value).strip()

        try:
            species_name = SpeciesDatabase._cea_name(raw_name)
            display_name = SpeciesDatabase._name(raw_name)
        except Exception:
            try:
                species_name = CEA.resolve_name(raw_name)
            except Exception:
                species_name = raw_name
            display_name = species_name

        if not CEA.has_species(species_name):
            raise ValueError(
                f"{value!r} could not be resolved to a CEA thermo species. "
                "Use a supported SpeciesDatabase alias or a direct CEA product "
                "species name."
            )

        species_name = CEA.resolve_name(species_name)
        thermo_index = CEA.index(species_name)

        if not CEA.has_thermo(species_name):
            elements = CEA.elemental_composition(species_name)
            raise ValueError(
                f"{species_name!r} is present in the CEA data, but it has no NASA-9 "
                "polynomial intervals. It is a CEA reactant/reference definition "
                "rather than a gas-phase thermodynamic species. "
                f"Elemental composition: {elements}. "
                "CombustionGas only accepts product/species entries with polynomial data."
            )

        if not CEA.is_gas(species_name):
            raise ValueError(
                f"{species_name!r} is a CEA condensed/reference entry, not a gas species. "
                "CombustionGas only accepts gas-phase CEA product species."
            )

        transport_index = None

        if CEA.has_transport(species_name):
            transport_index = CEA.transport_index(species_name)

        return species_name, display_name, thermo_index, transport_index

    # ---------------- State setting / flashing ---------------- #

    def _set_state(
        self,
        pressure: float | None = None,
        temperature: float | None = None,
        enthalpy: float | None = None,
        internal_energy: float | None = None,
        density: float | None = None,
    ) -> None:
        self._clear_property_cache()

        self._last_state_values = {
            "pressure": pressure,
            "temperature": temperature,
            "enthalpy": enthalpy,
            "internal_energy": internal_energy,
            "density": density,
        }
        self._last_state_values = {
            key: value
            for key, value in self._last_state_values.items()
            if value is not None
        }

        provided = frozenset(self._last_state_values)

        if enthalpy is not None:
            enthalpy = self._to_raw_basis("enthalpy", enthalpy)

        if internal_energy is not None:
            internal_energy = self._to_raw_basis("internal_energy", internal_energy)

        if provided not in self._FLASH_INPUTS:
            raise LookupError(
                "Unsupported combustion-gas state input combination. "
                f"Supported inputs are: {self.available_flash_inputs()}."
            )

        if provided == frozenset(("pressure", "density")):
            self._pressure = float(pressure)
            self._temperature = self._pressure / (float(density) * self.gas_constant)
            self._validate_temperature()
            self._enthalpy = self._enthalpy_from_temperature(self._temperature)
            return

        if temperature is not None:
            self._temperature = float(temperature)
            self._validate_temperature()
            self._enthalpy = self._enthalpy_from_temperature(self._temperature)

        elif enthalpy is not None:
            self._enthalpy = float(enthalpy)
            self._temperature = self._temperature_from_enthalpy(self._enthalpy)
            self._validate_temperature()

        elif internal_energy is not None:
            self._temperature = self._temperature_from_internal_energy(float(internal_energy))
            self._validate_temperature()
            self._enthalpy = self._enthalpy_from_temperature(self._temperature)

        if pressure is not None:
            self._pressure = float(pressure)

        if density is not None:
            pressure_from_density = float(density) * self.gas_constant * self._temperature

            if self._pressure is None:
                self._pressure = pressure_from_density
            else:
                if not np.isclose(self._pressure, pressure_from_density, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        "Provided pressure and density are inconsistent with the "
                        "ideal-gas equation of state at the solved temperature. "
                        f"pressure={self._pressure:.6g}, "
                        f"density*R*temperature={pressure_from_density:.6g}"
                    )

    def _require_pressure(self, property_name: str = "This property"):
        if self._pressure is None:
            raise ValueError(f"{property_name} requires pressure. Set gas.pressure first.")

    def _cache_get(self, property_name: str):
        return self._property_cache.get(property_name, None)

    def _cache_set(self, property_name: str, value):
        self._property_cache[property_name] = value
        return value

    def _clear_property_cache(self) -> None:
        self._property_cache.clear()

    def _enthalpy_from_temperature(self, temperature: float) -> float:
        _, h_mass, _, _ = CEA.thermo_mass_array(
            self._species_names,
            temperature,
            on_error="raise",
        )
        return float(np.dot(self._mass_fractions, h_mass))

    def _internal_energy_from_temperature(self, temperature: float) -> float:
        return self._enthalpy_from_temperature(temperature) - self.gas_constant * float(temperature)

    def _temperature_from_enthalpy(self, enthalpy_target: float) -> float:
        def residual(temperature):
            return self._enthalpy_from_temperature(temperature) - enthalpy_target

        return self._solve_temperature_from_residual(
            residual,
            "enthalpy",
            enthalpy_target,
        )

    def _temperature_from_internal_energy(self, internal_energy_target: float) -> float:
        def residual(temperature):
            return self._internal_energy_from_temperature(temperature) - internal_energy_target

        return self._solve_temperature_from_residual(
            residual,
            "internal_energy",
            internal_energy_target,
        )

    def _solve_temperature_from_residual(
        self,
        residual,
        variable_name: str,
        target_value: float,
    ) -> float:
        minimum_temperature = self.minimum_temperature
        maximum_temperature = self.maximum_temperature

        temperatures = np.linspace(minimum_temperature, maximum_temperature, 400)
        residuals = np.array([residual(temperature) for temperature in temperatures])

        for temperature, residual_value in zip(temperatures, residuals):
            if abs(residual_value) < 1e-8:
                return float(temperature)

        for temperature_1, temperature_2, residual_1, residual_2 in zip(
            temperatures[:-1],
            temperatures[1:],
            residuals[:-1],
            residuals[1:],
        ):
            if (
                np.isfinite(residual_1)
                and np.isfinite(residual_2)
                and residual_1 * residual_2 <= 0
            ):
                sol = root_scalar(
                    residual,
                    bracket=(temperature_1, temperature_2),
                    method="brentq",
                )
                return float(sol.root)

        raise ValueError(
            f"Could not solve combustion-gas temperature from "
            f"{variable_name}={target_value:.6g} J/kg "
            f"over temperature=[{minimum_temperature:.3f}, {maximum_temperature:.3f}] K."
        )

    def _partial_pressures(self) -> np.ndarray:
        self._require_pressure("Partial pressures")
        return self._mole_fractions * self._pressure

    # ---------------- Core package-style API ---------------- #

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

    # ---------------- Fractions ---------------- #

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

        self._mole_fractions = self._validate_fractions(value, "Mole fractions")
        self._mass_fractions = self._mole_fractions * self._M / np.dot(self._mole_fractions, self._M)
        self._clear_reference_cache()
        self._clear_property_cache()

        if self._last_state_values is not None:
            self._set_state(**self._last_state_values)

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

        self._mass_fractions = self._validate_fractions(value, "Mass fractions")
        inv = self._mass_fractions / self._M
        self._mole_fractions = inv / inv.sum()
        self._clear_reference_cache()
        self._clear_property_cache()

        if self._last_state_values is not None:
            self._set_state(**self._last_state_values)

    # ---------------- State properties ---------------- #

    @property
    def pressure(self) -> float | None:
        return self._pressure

    @pressure.setter
    def pressure(self, value: float):
        self._pressure = float(value)
        self._clear_property_cache()

    @property
    def enthalpy(self) -> float:
        return self._from_raw_basis("enthalpy", self._enthalpy)

    @enthalpy.setter
    def enthalpy(self, value: float):
        self._clear_property_cache()
        self._enthalpy = self._to_raw_basis("enthalpy", value)
        self._temperature = self._temperature_from_enthalpy(self._enthalpy)
        self._validate_temperature()

    @property
    def internal_energy(self) -> float:
        cached = self._cache_get("internal_energy")
        if cached is not None:
            return cached

        return self._cache_set(
            "internal_energy",
            self._from_raw_basis("internal_energy", self._internal_energy_from_temperature(self._temperature)),
        )

    @internal_energy.setter
    def internal_energy(self, value: float):
        self._clear_property_cache()
        raw_value = self._to_raw_basis("internal_energy", value)
        self._temperature = self._temperature_from_internal_energy(raw_value)
        self._validate_temperature()
        self._enthalpy = self._enthalpy_from_temperature(self._temperature)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._clear_property_cache()
        self._temperature = float(value)
        self._validate_temperature()
        self._enthalpy = self._enthalpy_from_temperature(self._temperature)

    @property
    def density(self) -> float:
        cached = self._cache_get("density")
        if cached is not None:
            return cached

        self._require_pressure("Density")
        return self._cache_set(
            "density",
            self._pressure / (self.gas_constant * self._temperature),
        )

    @density.setter
    def density(self, value: float):
        if self._temperature is None:
            raise ValueError("Cannot set density without temperature.")
        self._pressure = float(value) * self.gas_constant * self._temperature
        self._clear_property_cache()

    @property
    def pressure_temperature(self) -> Tuple[float | None, float]:
        return self._pressure, self._temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float | None, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature)")
        self._set_state(pressure=values[0], temperature=values[1])

    @property
    def pressure_enthalpy(self) -> Tuple[float | None, float]:
        return self._pressure, self.enthalpy

    @pressure_enthalpy.setter
    def pressure_enthalpy(self, values: Tuple[float | None, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_enthalpy must be set with (pressure, enthalpy)")
        self._set_state(pressure=values[0], enthalpy=values[1])

    @property
    def pressure_internal_energy(self) -> Tuple[float | None, float]:
        return self._pressure, self.internal_energy

    @pressure_internal_energy.setter
    def pressure_internal_energy(self, values: Tuple[float | None, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_internal_energy must be set with (pressure, internal_energy)")
        self._set_state(pressure=values[0], internal_energy=values[1])

    @property
    def density_temperature(self) -> Tuple[float, float]:
        return self.density, self._temperature

    @density_temperature.setter
    def density_temperature(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_temperature must be set with (density, temperature)")
        self._set_state(density=values[0], temperature=values[1])

    @property
    def density_enthalpy(self) -> Tuple[float, float]:
        return self.density, self.enthalpy

    @density_enthalpy.setter
    def density_enthalpy(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_enthalpy must be set with (density, enthalpy)")
        self._set_state(density=values[0], enthalpy=values[1])

    @property
    def density_internal_energy(self) -> Tuple[float, float]:
        return self.density, self.internal_energy

    @density_internal_energy.setter
    def density_internal_energy(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_internal_energy must be set with (density, internal_energy)")
        self._set_state(density=values[0], internal_energy=values[1])

    @property
    def pressure_density(self) -> Tuple[float, float]:
        return self._pressure, self.density

    @pressure_density.setter
    def pressure_density(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_density must be set with (pressure, density)")
        self._set_state(pressure=values[0], density=values[1])

    @property
    def HP(self) -> Tuple[float, float | None]:
        return self.enthalpy, self._pressure

    @HP.setter
    def HP(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("HP must be set with (enthalpy, pressure)")
        self._set_state(enthalpy=values[0], pressure=values[1])

    @property
    def TP(self) -> Tuple[float, float | None]:
        return self._temperature, self._pressure

    @TP.setter
    def TP(self, values: Tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("TP must be set with (temperature, pressure)")
        self._set_state(temperature=values[0], pressure=values[1])

    # ---------------- Thermo properties ---------------- #

    @property
    def molar_mass(self) -> float:
        """Mixture molar mass [kg/mol]."""
        cached = self._cache_get("molar_mass")
        if cached is not None:
            return cached

        return self._cache_set(
            "molar_mass",
            float(1.0 / np.sum(self._mass_fractions / (self._M / 1000.0))),
        )

    @property
    def gas_constant(self) -> float:
        """Mixture gas constant [J/kg-K]."""
        cached = self._cache_get("gas_constant")
        if cached is not None:
            return cached

        return self._cache_set("gas_constant", self._RU / self.molar_mass)

    @property
    def universal_gas_constant(self) -> float:
        return self._RU

    @property
    def compressibility(self) -> float:
        return 1.0

    @property
    def specific_volume(self) -> float:
        cached = self._cache_get("specific_volume")
        if cached is not None:
            return cached

        return self._cache_set("specific_volume", 1.0 / self.density)

    @property
    def thermal_expansion_coefficient(self) -> float:
        return 1.0 / self.temperature

    @property
    def isothermal_compressibility(self) -> float:
        self._require_pressure("Isothermal compressibility")
        return 1.0 / self.pressure

    @property
    def quality(self) -> float:
        return 1.0

    @quality.setter
    def quality(self, value: float):
        raise ValueError("CombustionGas does not support vapor quality.")

    @property
    def specific_heat_cp(self) -> float:
        cached = self._cache_get("specific_heat_cp")
        if cached is not None:
            return cached

        return self._cache_set(
            "specific_heat_cp",
            float(np.dot(self._mass_fractions, self._pure_cp_mass())),
        )

    @property
    def specific_heat_cv(self) -> float:
        cached = self._cache_get("specific_heat_cv")
        if cached is not None:
            return cached

        return self._cache_set("specific_heat_cv", self.specific_heat_cp - self.gas_constant)

    @property
    def specific_heat(self) -> float:
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        cached = self._cache_get("specific_heat_ratio")
        if cached is not None:
            return cached

        cv = self.specific_heat_cv
        return self._cache_set("specific_heat_ratio", None if cv == 0.0 else self.specific_heat_cp / cv)

    @property
    def entropy(self) -> float:
        cached = self._cache_get("entropy")
        if cached is not None:
            return cached

        self._require_pressure("Entropy")

        p_i = self._mole_fractions * self.pressure
        p_i = np.maximum(p_i, np.finfo(float).tiny)

        pressure_correction_molar = -self._RU_KMOL * np.log(p_i / self._P_REF)

        _, _, pure_s0_molar, _ = CEA.thermo_molar_array(
            self._species_names,
            self.temperature,
            on_error="raise",
        )

        pure_s_mass_at_pi = (pure_s0_molar + pressure_correction_molar) / self._M
        return self._cache_set("entropy", self._from_raw_basis("entropy", float(np.dot(self._mass_fractions, pure_s_mass_at_pi))))

    @property
    def gibbs_energy(self) -> float:
        cached = self._cache_get("gibbs_energy")
        if cached is not None:
            return cached

        self._require_pressure("Gibbs energy")
        return self._cache_set("gibbs_energy", self.enthalpy - self.temperature * self.entropy)

    @property
    def free_energy(self) -> float:
        return self.helmholtz_energy

    @property
    def helmholtz_energy(self) -> float:
        cached = self._cache_get("helmholtz_energy")
        if cached is not None:
            return cached

        self._require_pressure("Helmholtz energy")
        return self._cache_set("helmholtz_energy", self.internal_energy - self.temperature * self.entropy)

    @property
    def speed_of_sound(self) -> float:
        cached = self._cache_get("speed_of_sound")
        if cached is not None:
            return cached

        return self._cache_set(
            "speed_of_sound",
            float(np.sqrt(self.specific_heat_ratio * self.gas_constant * self.temperature)),
        )

    def _validate_temperature(self) -> None:
        if self._temperature is None:
            return

        if self.temperature < self.minimum_temperature or self.temperature > self.maximum_temperature:
            raise ValueError(
                f"Temperature {self.temperature:.6g} K is outside the common valid "
                f"CEA polynomial range [{self.minimum_temperature:.6g}, "
                f"{self.maximum_temperature:.6g}] K for this composition."
            )

    def _temperature_limits(self) -> tuple[float, float]:
        tmin, tmax = CEA.temperature_limits(self._species_names)

        return max(50.0, tmin - 50.0), tmax

    @staticmethod
    def _interval_index(thermo_index: int, temperature: float) -> int:
        name = str(CEA.raw_by_index("names", thermo_index))
        return CEA.interval_index(name, temperature)

    @classmethod
    def _species_thermo(cls, thermo_index: int, temperature: float) -> tuple[float, float, float]:
        name = str(CEA.raw_by_index("names", thermo_index))
        return CEA.thermo_molar(name, temperature)

    def _pure_cp_mass(self) -> np.ndarray:
        cached = self._cache_get("_pure_cp_mass")
        if cached is not None:
            return cached

        cp_mass, _, _, _ = CEA.thermo_mass_array(
            self._species_names,
            self.temperature,
            on_error="raise",
        )
        return self._cache_set("_pure_cp_mass", cp_mass)

    def _pure_h_mass(self) -> np.ndarray:
        cached = self._cache_get("_pure_h_mass")
        if cached is not None:
            return cached

        _, h_mass, _, _ = CEA.thermo_mass_array(
            self._species_names,
            self.temperature,
            on_error="raise",
        )
        return self._cache_set("_pure_h_mass", h_mass)

    def _pure_s0_mass(self) -> np.ndarray:
        cached = self._cache_get("_pure_s0_mass")
        if cached is not None:
            return cached

        _, _, s0_mass, _ = CEA.thermo_mass_array(
            self._species_names,
            self.temperature,
            on_error="raise",
        )
        return self._cache_set("_pure_s0_mass", s0_mass)

    # ---------------- Transport properties ---------------- #

    @staticmethod
    def _transport_interval_index(transport_index: int, temperature: float, kind: str) -> int:
        name = str(CEA.transport_names[transport_index])
        return CEA.transport_interval_index(name, temperature, kind)

    @classmethod
    def _transport_fit(cls, transport_index: int, temperature: float, kind: str) -> float:
        name = str(CEA.transport_names[transport_index])
        return CEA.transport_fit(name, temperature, kind)

    def _require_transport(self, species_index: int, kind: str) -> int:
        transport_index = self._transport_indices[species_index]

        if transport_index is None:
            name = self._species_names[species_index]
            raise NotImplementedError(
                f"CEA {kind} transport data are not available for {name!r}."
            )

        return int(transport_index)

    def _estimated_viscosity(self, species_name: str) -> float:
        """
        Estimate pure-species gas viscosity [Pa-s] when CEA transport data are
        unavailable.
        """
        T = float(self.temperature)
        M = CEA.molecular_weight(species_name)

        omega = np.log(50.0 * M**4.6 / T**1.4)
        omega = max(float(omega), 1.0)

        viscns = 2.67e-8

        return float(viscns * np.sqrt(M * T) / omega)

    def _estimated_conductivity(
        self,
        species_name: str,
        viscosity: float | None = None,
    ) -> float:
        """
        Estimate pure-species frozen thermal conductivity [W/m-K] when CEA
        transport data are unavailable.
        """
        if viscosity is None:
            viscosity = self._estimated_viscosity(species_name)

        T = float(self.temperature)
        M = CEA.molecular_weight(species_name)

        cp_molar = CEA.thermo_molar(species_name, T)[0]
        cp_over_R = cp_molar / self._RU_KMOL

        return float(
            viscosity
            * self._RU_KMOL
            * (0.00375 + 0.00132 * (cp_over_R - 2.5))
            / M
        )

    @property
    def estimated_transport_species(self) -> list[str]:
        return [
            name
            for name in self._species_names
            if not CEA.has_transport(name)
        ]

    def _pure_viscosities(self) -> np.ndarray:
        cached = self._cache_get("_pure_viscosities")
        if cached is not None:
            return cached

        values, valid = CEA.viscosity_array(
            self._species_names,
            self.temperature,
            on_error="nan",
        )
        missing = (~valid) | (~np.isfinite(values)) | (values <= 0.0)

        if np.any(missing):
            values = np.array(values, dtype=float, copy=True)
            values[missing] = np.array(
                [self._estimated_viscosity(self._species_names[i]) for i in np.nonzero(missing)[0]],
                dtype=float,
            )

        return self._cache_set("_pure_viscosities", np.asarray(values, dtype=float))

    def _pure_conductivities(self) -> np.ndarray:
        cached = self._cache_get("_pure_conductivities")
        if cached is not None:
            return cached

        viscosities = self._pure_viscosities()
        values, valid = CEA.conductivity_array(
            self._species_names,
            self.temperature,
            on_error="nan",
        )
        missing = (~valid) | (~np.isfinite(values)) | (values <= 0.0)

        if np.any(missing):
            values = np.array(values, dtype=float, copy=True)
            values[missing] = np.array(
                [
                    self._estimated_conductivity(
                        self._species_names[i],
                        viscosity=float(viscosities[i]),
                    )
                    for i in np.nonzero(missing)[0]
                ],
                dtype=float,
            )

        return self._cache_set("_pure_conductivities", np.asarray(values, dtype=float))

    def _estimated_binary_viscosity_interaction(
        self,
        i: int,
        j: int,
        pure_viscosities: np.ndarray,
    ) -> float:
        Mi = float(self._M[i])
        Mj = float(self._M[j])
        etai = float(pure_viscosities[i])
        etaj = float(pure_viscosities[j])

        ratio = np.sqrt(Mj / Mi)

        etaij = 5.656854 * etai * np.sqrt(Mj / (Mi + Mj))
        etaij = etaij / (1.0 + np.sqrt(ratio * etai / etaj))**2

        return float(etaij)

    def _binary_viscosity_interaction_matrix(self) -> np.ndarray:
        cached = self._cache_get("_binary_viscosity_interaction_matrix")
        if cached is not None:
            return cached

        pure_viscosities = self._pure_viscosities()
        eta = CEA.binary_viscosity_interaction_matrix(
            self._species_names,
            self.temperature,
            pure_viscosities=pure_viscosities,
            molecular_weights=self._M,
        )
        return self._cache_set("_binary_viscosity_interaction_matrix", eta)

    def _cea_phi_matrix(self) -> np.ndarray:
        cached = self._cache_get("_cea_phi_matrix")
        if cached is not None:
            return cached

        eta_i = self._pure_viscosities()
        eta_ij = self._binary_viscosity_interaction_matrix()
        M_i = self._M[:, None]
        M_j = self._M[None, :]

        phi = 2.0 * M_j * eta_i[:, None] / (eta_ij * (M_i + M_j))
        np.fill_diagonal(phi, 0.0)

        return self._cache_set("_cea_phi_matrix", phi)

    def _cea_psi_matrix(self) -> np.ndarray:
        cached = self._cache_get("_cea_psi_matrix")
        if cached is not None:
            return cached

        phi = self._cea_phi_matrix()
        M_i = self._M[:, None]
        M_j = self._M[None, :]

        correction = (
            1.0
            + 2.41
            * (M_i - M_j)
            * (M_i - 0.142 * M_j)
            / (M_i + M_j) ** 2
        )

        psi = phi * correction
        np.fill_diagonal(psi, 0.0)

        return self._cache_set("_cea_psi_matrix", psi)

    @staticmethod
    def _cea_mix(
        mole_fractions: np.ndarray,
        values: np.ndarray,
        interaction_matrix: np.ndarray,
    ) -> float:
        mole_fractions = np.asarray(mole_fractions, dtype=float)
        values = np.asarray(values, dtype=float)
        interaction_matrix = np.asarray(interaction_matrix, dtype=float)
        denominator = mole_fractions + interaction_matrix @ mole_fractions

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = mole_fractions * values / denominator

        return float(np.sum(terms))

    @staticmethod
    def _wilke_phi(property_values: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
        values_i = property_values[:, None]
        values_j = property_values[None, :]
        weights_i = molecular_weights[:, None]
        weights_j = molecular_weights[None, :]

        return (
            (1.0 + np.sqrt(values_i / values_j) * (weights_j / weights_i) ** 0.25) ** 2
            / np.sqrt(8.0 * (1.0 + weights_i / weights_j))
        )

    @staticmethod
    def _wilke_mix(mole_fractions: np.ndarray, values: np.ndarray, molecular_weights: np.ndarray) -> float:
        mole_fractions = np.asarray(mole_fractions, dtype=float)
        values = np.asarray(values, dtype=float)
        phi = CombustionGas._wilke_phi(values, molecular_weights)
        denominator = phi @ mole_fractions

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = mole_fractions * values / denominator

        return float(np.sum(terms))

    @property
    def dynamic_viscosity(self) -> float:
        cached = self._cache_get("dynamic_viscosity")
        if cached is not None:
            return cached

        mu = self._pure_viscosities()

        if not self._mixture:
            return self._cache_set("dynamic_viscosity", float(mu[0]))

        return self._cache_set(
            "dynamic_viscosity",
            self._cea_mix(self._mole_fractions, mu, self._cea_phi_matrix()),
        )

    @property
    def conductivity(self) -> float:
        cached = self._cache_get("conductivity")
        if cached is not None:
            return cached

        k = self._pure_conductivities()

        if not self._mixture:
            return self._cache_set("conductivity", float(k[0]))

        return self._cache_set(
            "conductivity",
            self._cea_mix(self._mole_fractions, k, self._cea_psi_matrix()),
        )

    @property
    def thermal_conductivity(self) -> float:
        return self.conductivity

    @property
    def kinematic_viscosity(self) -> float:
        cached = self._cache_get("kinematic_viscosity")
        if cached is not None:
            return cached

        return self._cache_set("kinematic_viscosity", self.dynamic_viscosity / self.density)

    @property
    def prandtl(self) -> float:
        cached = self._cache_get("prandtl")
        if cached is not None:
            return cached

        k = self.thermal_conductivity

        if k is None or k == 0.0:
            return None

        return self._cache_set("prandtl", self.specific_heat_cp * self.dynamic_viscosity / k)

    # ---------------- Derivatives ---------------- #

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        of = of.lower()
        wrt = with_respect_to.lower()
        const = constant.lower()

        R = self.gas_constant
        T = self.temperature
        P = self.pressure
        rho = self.density if self.pressure is not None else None
        cp = self.specific_heat_cp
        cv = self.specific_heat_cv

        if P is None:
            raise ValueError("CombustionGas partial derivatives require pressure.")

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

    # ---------------- String output ---------------- #

    def _safe(self, value, fmt=".3e"):
        return format_optional(value, fmt)

    def __str__(self):
        pressure = self.pressure
        temperature = self.temperature
        density = self.density if self._pressure is not None else None
        internal_energy = self.internal_energy
        enthalpy = self.enthalpy
        entropy = self.entropy if self._pressure is not None else None
        specific_heat_cp = self.specific_heat_cp
        specific_heat_cv = self.specific_heat_cv
        specific_heat_ratio = self.specific_heat_ratio
        gas_constant = self.gas_constant
        molar_mass = self.molar_mass
        dynamic_viscosity = self.dynamic_viscosity
        thermal_conductivity = self.thermal_conductivity
        prandtl = self.prandtl if self._pressure is not None else self.prandtl
        speed_of_sound = self.speed_of_sound

        rows = [
            ("Gas(es)", ", ".join(self._species_names)),
            ("Backend", self.backend),
            ("Mole fractions", rounded_dict(self.mole_fractions, 5)),
            ("Mass fractions", rounded_dict(self.mass_fractions, 5)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(pressure, ".3e")),
            ("Temperature [K]", self._safe(temperature, ".2f")),
            ("Density [kg/m³]", self._safe(density, ".3f") if self._pressure is not None else "N/A"),
            ("Compressibility Z", self._safe(self.compressibility, ".3f")),
            ("Internal energy [J/kg]", self._safe(internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(entropy, ".3e") if self._pressure is not None else "N/A"),
            ("Cp [J/kg-K]", self._safe(specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]", self._safe(specific_heat_cv, ".3f")),
            ("Specific heat ratio", self._safe(specific_heat_ratio, ".5f")),
            ("Gas constant [J/kg-K]", self._safe(gas_constant, ".3f")),
            ("Molar mass [kg/mol]", self._safe(molar_mass, ".6f")),
            ("Dynamic viscosity [Pa·s]", self._safe(dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(thermal_conductivity, ".3f")),
            ("Prandtl number", self._safe(prandtl, ".5f")),
            ("Speed of sound [m/s]", self._safe(speed_of_sound, ".3f")),
        ]

        if self.estimated_transport_species:
            rows.append(("Estimated transport", self.estimated_transport_species))

        return format_rows(rows)

    def __repr__(self) -> str:
        species_str = ", ".join(self._species_names)
        pressure_str = "None" if self._pressure is None else f"{self._pressure:.3e}"
        return (
            f"{self.__class__.__name__}(species=[{species_str}], "
            f"pressure={pressure_str} Pa, "
            f"enthalpy={self.enthalpy:.3e} J/kg, "
            f"temperature={self.temperature:.2f} K)"
        )

    # ---------------- Utilities ---------------- #

    @classmethod
    def get_available_species(cls) -> list[str]:
        """Return ThermoProp species supported by CombustionGas."""
        return SpeciesDatabase.supported_species("CombustionGas")

    @classmethod
    def show_available_species(cls) -> list[str]:
        species = cls.get_available_species()

        for name in species:
            print(name)

        return species

    @staticmethod
    def get_available_gases() -> list[str]:
        """Return ThermoProp species supported by CombustionGas."""
        return CombustionGas.get_available_species()

    @staticmethod
    def show_available_gases() -> list[str]:
        """Print and return ThermoProp species supported by CombustionGas."""
        return CombustionGas.show_available_species()

    @staticmethod
    def get_available_fluids() -> list[str]:
        """Fluid-style alias for API consistency."""
        return CombustionGas.get_available_species()

    @staticmethod
    def show_available_fluids() -> list[str]:
        """Fluid-style alias for API consistency."""
        return CombustionGas.show_available_species()

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
        )

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
            if len(inputs) == 2
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        return cls.available_flash_pairs()

    @staticmethod
    def mole_to_mass(species_names: List[str], mole_fractions: List[float]):
        mole_fractions = CombustionGas._validate_fractions(mole_fractions, "Mole fractions")
        return CEA.mole_to_mass(species_names, mole_fractions)

    @staticmethod
    def mass_to_mole(species_names: List[str], mass_fractions: List[float]):
        mass_fractions = CombustionGas._validate_fractions(mass_fractions, "Mass fractions")
        return CEA.mass_to_mole(species_names, mass_fractions)
