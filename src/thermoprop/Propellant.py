from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import warnings

from scipy.integrate import quad, IntegrationWarning

from .Fluid import Fluid
from .CEADatabase import CEA
from .SpeciesDatabase import SpeciesDatabase
from .ReferenceState import normalize_reference_target
from ._api import PropertyIntrospectionMixin
from ._formatting import format_optional, rounded_dict, format_rows
from ._state_api import UNSET, is_provided, provided_items

class Propellant(PropertyIntrospectionMixin):
    """
    Combined RocketProps / CEA propellant and reactant property wrapper.

    Propellant resolves names through SpeciesDatabase first. When the
    registry maps a name to RocketProps, RocketProps is used for liquid
    engineering properties. When the registry maps a name to a CEA reactant, or
    when the input is an exact strict CEA database name, CEA is used for
    reference/thermochemical data.

    If both backends are available, liquid states use both backends:
    RocketProps supplies liquid engineering correlations, while CEA supplies
    reactant/reference data and condensed NASA-polynomial thermodynamics when
    that liquid/condensed entry has polynomial data. Gas states use the CEA gas
    species when one is available.

    If a registry entry has RocketProps support but no CEA mapping, CEA is not
    queried. This keeps RocketProps-only species and mixtures independent from
    strict CEA names.

    Supported state inputs:

        Propellant(..., temperature=...)
        Propellant(..., temperature=..., pressure=...)

    Optional quality is accepted only as an inlet-enthalpy correction, not as a
    full two-phase equation of state:

        Propellant(..., pressure=..., quality=...)
        Propellant(..., temperature=..., quality=...)
        Propellant(..., temperature=..., pressure=..., quality=...)

    For quality-corrected states, all ordinary properties remain RocketProps /
    CEA single-phase estimates at the resolved saturation state. Only enthalpy
    is corrected using the CoolProp enthalpy difference between the requested
    two-phase state and saturated liquid at the same pressure or temperature:

        h = h_propellant_liquid + (h_fluid_quality - h_fluid_liquid)

    This keeps the reactant enthalpy on the Propellant/CEA reference basis
    while accounting for inlet vapor quality.

    Public API units are SI.
    """

    _BACKEND_NAME = "RocketProps + CEA"

    _UNSUPPORTED_PROPERTIES = {
        "thermal_expansion_coefficient",
        "isothermal_compressibility",
        "joule_thomson_coefficient",
        "partial_derivative",
        "universal_gas_constant",
        "prandtl",
    }

    _FLASH_INPUTS = {
        frozenset(("temperature",)),
        frozenset(("pressure", "temperature")),
        frozenset(("pressure", "quality")),
        frozenset(("temperature", "quality")),
        frozenset(("pressure", "temperature", "quality")),
    }

    _P0_CEA = 101325.0
    _PSIA_TO_PA = 6894.757293168361
    _PA_TO_PSIA = 1.0 / _PSIA_TO_PA

    _BTU_PER_LBM_TO_J_PER_KG = 2326.0
    _BTU_PER_LBM_R_TO_J_PER_KG_K = 4186.8
    _BTU_PER_HR_FT_R_TO_W_PER_M_K = 1.730735
    _LBF_PER_IN_TO_N_PER_M = 175.126835
    _RU = 8.31446261815324
    _CACHE_MISS = object()

    _REFERENCE_TEMPERATURE = 298.15
    _REFERENCE_PRESSURE = 101325.0
    _REFERENCE_CACHE: dict[tuple, tuple[float, float, float]] = {}

    _CEA_REFERENCE_HINTS = {
        "Ammonia": "NH3(L)",
        "NH3": "NH3(L)",
        "CLF5": "CLF5",
        "Ethanol": "C2H5OH(L)",
        "F2": "F2(L)",
        "Fluorine": "F2(L)",
        "H2O2": "H2O2(L)",
        "Hydrogen": "H2(L)",
        "PH2": "H2(L)",
        "IRFNA": "IRFNA",
        "MMH": "CH6N2(L)",
        "Methane": "CH4(L)",
        "Methanol": "CH3OH(L)",
        "N2H4": "N2H4(L)",
        "N2O4": "N2O4(L)",
        "NitrousOxide": "N2O",
        "N2O": "N2O",
        "Oxygen": "O2(L)",
        "LOX": "O2(L)",
        "RP1": "RP-1",
        "UDMH": "C2H8N2(L),UDMH",
        "Water": "H2O(L)",
        "n-Propane": "C3H8(L)",
        "Propane": "C3H8(L)",
    }


    def __init__(
        self,
        propellant: str,
        temperature: float | None = None,
        pressure: float | None = None,
        quality: float | None = None,
        set_reference: str | None = None,
    ):
        self._reference_target = self._normalize_reference_target(set_reference)
        self._reference_offsets: tuple[float, float, float] | None = None

        quality = self._validate_quality(quality)

        if temperature is None:
            if quality is None:
                raise TypeError("Propellant requires temperature, or pressure + quality.")

            if pressure is None:
                raise ValueError("Propellant pressure-quality states require pressure.")

            temperature = self._temperature_from_fluid_quality(
                propellant,
                pressure=float(pressure),
                quality=quality,
            )

        elif quality is not None and pressure is None:
            pressure = self._pressure_from_fluid_quality(
                propellant,
                temperature=float(temperature),
                quality=quality,
            )

        self._input_name = str(propellant)
        self._temperature = float(temperature)
        self._pressure = None if pressure is None else float(pressure)
        self._quality_override = quality

        self._registry_name: str | None = None
        self._rocketprops_name: str | None = None
        self._cea_species_name: str | None = None
        self._cea_species_index: int | None = None
        self._cea_reactant_name: str | None = None
        self._cea_reactant_index: int | None = None
        self._cea_name: str | None = None
        self._cea_index: int | None = None
        self._backend = None
        self._data_sources: dict[str, str] = {}
        self._property_cache: dict[str, Any] = {}

        self._resolve_backends(propellant)

        if (
            self._rocketprops_name is None
            and self._cea_species_name is None
            and self._cea_reactant_name is None
        ):
            raise ValueError(
                f"Unknown propellant or CEA species: {propellant!r}. "
                "Use Propellant.show_available_propellants(), "
                "Propellant.show_available_cea_species(), or "
                "SpeciesDatabase.supported_species('Propellant') to inspect names."
            )

        if self._rocketprops_name is not None:
            self._backend = self._get_rocketprops_backend(self._rocketprops_name)

        self._update_active_cea_name()
        self._validate_liquid_state()


    # ---------------- Quality enthalpy correction helpers ---------------- #

    @staticmethod
    def _validate_quality(value: float | None) -> float | None:
        if value is None:
            return None

        q = float(value)

        if not np.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError("quality must be between 0 and 1.")

        return q

    @staticmethod
    def _temperature_from_fluid_quality(
        propellant: str,
        *,
        pressure: float,
        quality: float,
    ) -> float:

        fluid = Fluid(propellant, pressure=pressure, quality=quality)
        return float(fluid.temperature)

    @staticmethod
    def _pressure_from_fluid_quality(
        propellant: str,
        *,
        temperature: float,
        quality: float,
    ) -> float:

        fluid = Fluid(propellant, temperature=temperature, quality=quality)
        return float(fluid.pressure)

    @property
    def has_quality_enthalpy_correction(self) -> bool:
        return self._quality_override is not None and self._quality_override > 0.0

    @property
    def enthalpy_correction(self) -> float | None:
        cached = self._cache_get("enthalpy_correction")
        if cached is not self._CACHE_MISS:
            return cached

        if not self.has_quality_enthalpy_correction:
            return self._cache_set("enthalpy_correction", 0.0)

        try:
            value = self._fluid_quality_enthalpy_correction()
        except Exception:
            return self._cache_set("enthalpy_correction", None)

        return self._cache_set(
            "enthalpy_correction",
            self._source(
                "enthalpy_correction",
                value,
                "CoolProp Δh",
            ),
        )

    def _fluid_quality_enthalpy_correction(self) -> float | None:
        if self._quality_override is None:
            return 0.0

        fluid_name = self._composition_argument()
        q = float(self._quality_override)

        if self.pressure is not None:
            mix = Fluid(fluid_name, pressure=self.pressure, quality=q)
            liquid = Fluid(fluid_name, pressure=self.pressure, quality=0.0)
        else:
            mix = Fluid(fluid_name, temperature=self.temperature, quality=q)
            liquid = Fluid(fluid_name, temperature=self.temperature, quality=0.0)

        return float(mix.enthalpy - liquid.enthalpy)


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
        return ((self.propellant, 1.0),)

    def _composition_argument(self) -> str:
        return self.propellant

    def _reference_cache_key(self) -> tuple:
        return (
            "Propellant",
            self._reference_target,
            self._composition_cache_key(),
            self._REFERENCE_TEMPERATURE,
            self._REFERENCE_PRESSURE,
        )

    def _raw_property_value(self, property_name: str):
        old_target = self._reference_target
        try:
            self._reference_target = None
            return getattr(self, property_name)
        finally:
            self._reference_target = old_target

    def _raw_reference_properties(self) -> tuple[float, float, float]:
        obj = self.__class__(
            self._composition_argument(),
            pressure=self._REFERENCE_PRESSURE,
            temperature=self._REFERENCE_TEMPERATURE,
            set_reference=None,
        )
        return float(obj.enthalpy), float(obj.internal_energy), float(obj.entropy)

    def _target_reference_properties(self) -> tuple[float, float, float]:
        target = self._reference_target
        if target is None:
            return self._raw_reference_properties()
        fluid = self._composition_argument()
        T = self._REFERENCE_TEMPERATURE
        P = self._REFERENCE_PRESSURE
        if target == "Fluid":
            obj = Fluid(fluid, pressure=P, temperature=T, set_reference=None)
        elif target == "IdealGas":
            from .IdealGas import IdealGas
            obj = IdealGas(fluid, pressure=P, temperature=T, set_reference=None)
        elif target == "CombustionGas":
            from .CombustionGas import CombustionGas
            obj = CombustionGas(fluid, pressure=P, temperature=T, set_reference=None)
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

    def _from_raw_basis(self, name: str, value: float | None) -> float | None:
        if value is None:
            return None
        if self._reference_target is None:
            return float(value)
        dh, du, ds = self._get_reference_offsets()
        if name == "enthalpy":
            return float(value) + dh
        if name == "internal_energy":
            return float(value) + du
        if name == "entropy":
            return float(value) + ds
        if name == "gibbs_energy":
            return float(value) + dh - self.temperature * ds
        if name in {"free_energy", "helmholtz_energy"}:
            return float(value) + du - self.temperature * ds
        return float(value)

    # ---------------- Resolution ---------------- #

    def _resolve_backends(self, propellant: str) -> None:
        """Resolve RocketProps and CEA names.

        The registry no longer stores CEA reactant cards. It only supplies
        durable cross-backend species names and RocketProps names. Liquid or
        reference CEA rows are discovered here from the input name, registry
        name, RocketProps name, CEA species name, exact CEA names, and a small
        legacy hint table for propellants whose CEA reference name is not
        directly recoverable from the RocketProps name.
        """
        raw = str(propellant).strip()

        try:
            record = SpeciesDatabase._record(raw)
            self._registry_name = record.name
            self._rocketprops_name = record.rocketprops

            if record.cea is not None:
                self._set_cea_species_name_if_present(record.cea)

            self._set_discovered_cea_reference_name(
                raw,
                self._registry_name,
                self._rocketprops_name,
                self._cea_species_name,
            )

            if self._rocketprops_name is not None or self._cea_species_name is not None or self._cea_reactant_name is not None:
                return

        except Exception:
            pass

        if CEA.has_species(raw):
            self._set_exact_cea_name(raw)
            return

        try:
            record = SpeciesDatabase._record(raw)
            self._registry_name = record.name

            if record.cea is not None:
                self._set_cea_species_name_if_present(record.cea)

            try:
                self._rocketprops_name = SpeciesDatabase._rocketprops_name(raw)
            except Exception:
                pass

            self._set_discovered_cea_reference_name(
                raw,
                self._registry_name,
                self._rocketprops_name,
                self._cea_species_name,
            )

            if self._rocketprops_name is not None or self._cea_species_name is not None or self._cea_reactant_name is not None:
                return

        except Exception:
            pass

        try:
            self._rocketprops_name = SpeciesDatabase._rocketprops_name(raw)
            self._set_discovered_cea_reference_name(raw, self._rocketprops_name)
            return
        except Exception:
            pass

        try:
            from rocketprops.rocket_prop import get_prop
        except ImportError:
            return

        backend = get_prop(raw)
        if backend is not None:
            self._rocketprops_name = raw
            self._set_discovered_cea_reference_name(raw)

    @classmethod
    def _normalize_cea_search_key(cls, value: str) -> str:
        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    @classmethod
    def _cea_reference_candidates(cls, *values: str | None) -> list[str]:
        """Return likely CEA condensed/reference names for the supplied names."""
        candidates: list[str] = []

        def add(value):
            if value is None:
                return
            value = str(value).strip()
            if not value or value in candidates:
                return
            candidates.append(value)

        for value in values:
            if value is None:
                continue

            raw = str(value).strip()
            upper = raw.upper()

            add(raw)
            add(upper)

            hint = cls._CEA_REFERENCE_HINTS.get(raw) or cls._CEA_REFERENCE_HINTS.get(upper)
            add(hint)

            if upper == "RP1":
                add("RP-1")
            if upper == "LOX":
                add("O2(L)")
            if upper in {"PH2", "LH2"}:
                add("H2(L)")

            if raw and not raw.endswith(")"):
                add(f"{raw}(L)")
                add(f"{upper}(L)")

        return candidates

    def _set_discovered_cea_reference_name(self, *values: str | None) -> None:
        """Find a CEA condensed/reference row without registry reactant cards."""
        if self._cea_reactant_name is not None:
            return

        candidates = self._cea_reference_candidates(*values)

        for candidate in candidates:
            if candidate is not None and CEA.has_species(candidate):
                name = CEA.resolve_name(candidate)
                if not (CEA.has_thermo(name) and CEA.is_gas(name)):
                    self._set_cea_reactant_name_if_present(name)
                    return

        try:
            reactant_names = list(CEA.reactant_names)
        except Exception:
            reactant_names = []

        candidate_keys = {
            self._normalize_cea_search_key(candidate)
            for candidate in candidates
            if candidate is not None
        }

        for name in reactant_names:
            key = self._normalize_cea_search_key(name)
            if key in candidate_keys:
                self._set_cea_reactant_name_if_present(name)
                return

        # Last pass: allow named CEA cards like C2H8N2(L),UDMH to match UDMH.
        for name in reactant_names:
            key = self._normalize_cea_search_key(name)
            for candidate_key in candidate_keys:
                if candidate_key and candidate_key in key:
                    self._set_cea_reactant_name_if_present(name)
                    return

    def _set_cea_species_name_if_present(self, name: str) -> None:
        if CEA.has_species(name):
            self._cea_species_name = CEA.resolve_name(name)
            self._cea_species_index = CEA.index(self._cea_species_name)

    def _set_cea_reactant_name_if_present(self, name: str) -> None:
        if CEA.has_species(name):
            self._cea_reactant_name = CEA.resolve_name(name)
            self._cea_reactant_index = CEA.index(self._cea_reactant_name)

    def _set_exact_cea_name(self, name: str) -> None:
        name = CEA.resolve_name(name)
        if CEA.has_thermo(name) and CEA.is_gas(name):
            self._cea_species_name = name
            self._cea_species_index = CEA.index(name)
        else:
            self._cea_reactant_name = name
            self._cea_reactant_index = CEA.index(name)

        self._cea_name = name
        self._cea_index = CEA.index(name)

    def _update_active_cea_name(self) -> None:
        """Select the CEA row that matches the current phase.

        For liquid states, keep the CEA reactant/condensed row active so CEA
        reference data such as elemental composition, molar mass, and heat of
        formation are accounted for alongside RocketProps liquid correlations.
        For vapor states, switch to the CEA gas species when one is available.
        """
        if self._backend is not None:
            if self._active_is_liquid_model_raw():
                active = self._cea_reactant_name or self._cea_species_name
            else:
                active = self._cea_species_name or self._cea_reactant_name
        else:
            # Exact CEA-only input: keep the input phase. If both gas and
            # reactant names were resolved through the registry, use the phase
            # implied by pressure/vapor pressure only when RocketProps exists.
            active = self._cea_name or self._cea_species_name or self._cea_reactant_name

        self._cea_name = active
        self._cea_index = CEA.index(active) if active is not None else None

    def _active_is_liquid_model_raw(self) -> bool:
        if self._quality_override is not None:
            return True

        if self._backend is None:
            return False

        if self.pressure is None:
            tc = self.critical_temperature

            if tc is not None and self.temperature >= tc:
                return False

            return True

        pvap = self._rocketprops_vapor_pressure_no_source()

        if pvap is None:
            return True

        return self.pressure >= pvap

    @property
    def _active_is_liquid_model(self) -> bool:
        return self._active_is_liquid_model_raw()

    @staticmethod
    def _get_rocketprops_backend(propellant: str):
        try:
            from rocketprops.rocket_prop import get_prop
        except ImportError as exc:
            raise ImportError(
                "Propellant requires RocketProps for RocketProps-backed "
                "propellants. Install it with `pip install rocketprops`."
            ) from exc

        backend = get_prop(propellant)

        if backend is None:
            raise ValueError(f"Unknown RocketProps propellant: {propellant!r}")

        return backend

    # ---------------- Unit conversion helpers ---------------- #

    @staticmethod
    def _degR_from_K(temperature: float) -> float:
        return float(temperature) * 9.0 / 5.0

    @staticmethod
    def _K_from_degR(temperature: float) -> float:
        return float(temperature) * 5.0 / 9.0

    @classmethod
    def _psia_from_Pa(cls, pressure: float) -> float:
        return float(pressure) * cls._PA_TO_PSIA

    @classmethod
    def _Pa_from_psia(cls, pressure: float) -> float:
        return float(pressure) * cls._PSIA_TO_PA

    # ---------------- Source tracking helpers ---------------- #

    @property
    def data_sources(self) -> dict[str, str]:
        """Return the source used by each property that has been evaluated."""
        return dict(self._data_sources)

    def property_source(self, property_name: str) -> str | None:
        """Return the backend source used for one property, if evaluated."""
        return self._data_sources.get(property_name)

    def _source(self, property_name: str, value: Any, source: str):
        if value is not None:
            self._data_sources[property_name] = source
        return value

    def _cache_get(self, property_name: str):
        return self._property_cache.get(property_name, self._CACHE_MISS)

    def _cache_set(self, property_name: str, value: Any):
        self._property_cache[property_name] = value
        return value

    def _clear_property_cache(self) -> None:
        self._property_cache.clear()
        self._data_sources.clear()

    @staticmethod
    def _require_number(value: Any, message: str) -> float:
        """Return a finite float or raise a clear error."""
        if value is None:
            raise ValueError(message)

        if isinstance(value, complex):
            if abs(value.imag) > 1e-12:
                raise ValueError(message)
            value = value.real

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(message)

        return value

    @staticmethod
    def _real_or_none(value: Any) -> float | None:
        """Return a finite real float, or None for invalid/complex backend values."""
        if value is None:
            return None

        if isinstance(value, complex):
            if abs(value.imag) > 1e-12:
                return None
            value = value.real

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(value):
            return None

        return value

    # ---------------- Backend call helpers ---------------- #

    def _call(self, *names: str, default=None):
        if self._backend is None:
            return default

        for name in names:
            attr = getattr(self._backend, name, None)

            if attr is None:
                continue

            try:
                return attr() if callable(attr) else attr
            except TypeError:
                continue
            except Exception:
                continue

        return default

    def _call_at_temperature(self, *names: str, default=None):
        if self._backend is None:
            return default

        TdegR = self._degR_from_K(self.temperature)

        for name in names:
            fn = getattr(self._backend, name, None)

            if fn is None:
                continue

            try:
                return fn(TdegR)
            except TypeError:
                continue
            except Exception:
                continue

        return default

    def _call_compressed(self, *names: str, default=None):
        if self._backend is None or self.pressure is None:
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
            except Exception:
                continue

        return default

    def _cea_value(self, key: str):
        if self._cea_index is None:
            return None
        return CEA.raw_by_index(key, self._cea_index)

    def _cea_call(self, method: str, *args, default=None):
        if self._cea_name is None:
            return default

        fn = getattr(CEA, method)

        try:
            return fn(self._cea_name, *args)
        except Exception:
            return default

    def _rocketprops_vapor_pressure_no_source(self) -> float | None:
        """RocketProps vapor pressure without updating source bookkeeping."""
        value = self._real_or_none(
            self._call_at_temperature("PvapAtTdegR", "PvapAtT", default=None)
        )

        if value is None:
            return None

        return self._Pa_from_psia(value)

    def _validate_liquid_state(self) -> None:
        self._update_active_cea_name()

        if self._backend is None or self.pressure is None:
            return

        if self._quality_override is not None:
            return

        pvap = self.vapor_pressure

        if pvap is None:
            return

        if self.pressure < pvap and self._cea_species_name is None:
            raise ValueError(
                f"{self._rocketprops_name}: pressure={self.pressure:.6g} Pa is "
                f"below vapor pressure={pvap:.6g} Pa at "
                f"temperature={self.temperature:.6g} K, and no CEA gas species "
                "is available for automatic vapor-phase fallback."
            )

    def _unsupported(self, property_name: str):
        raise NotImplementedError(
            f"Propellant.{property_name} is not supported by RocketProps or CEA "
            "for this wrapper."
        )

    def update(
        self,
        propellant=UNSET,
        *,
        temperature=UNSET,
        pressure=UNSET,
        quality=UNSET,
        set_reference=UNSET,
    ):
        """Update the propellant identity and/or state in place.

        State-only updates reuse the existing RocketProps/CEA metadata and clear
        cached properties once.  Changing the propellant name or reference target
        rebuilds the object while preserving the current state unless replacement
        state inputs are supplied.
        """

        structural = is_provided(propellant) or is_provided(set_reference)

        if structural:
            new_propellant = self._input_name if not is_provided(propellant) else propellant
            new_temperature = self.temperature if not is_provided(temperature) else temperature
            new_pressure = self.pressure if not is_provided(pressure) else pressure
            new_quality = self._quality_override if not is_provided(quality) else quality
            new_reference = self._reference_target if not is_provided(set_reference) else set_reference

            rebuilt = self.__class__(
                new_propellant,
                temperature=new_temperature,
                pressure=new_pressure,
                quality=new_quality,
                set_reference=new_reference,
            )
            self.__dict__.update(rebuilt.__dict__)
            return self

        if is_provided(pressure) and is_provided(temperature):
            self.pressure_temperature = (pressure, temperature)
        else:
            if is_provided(temperature):
                self.temperature = temperature
            if is_provided(pressure):
                self.pressure = pressure

        if is_provided(quality):
            self.quality = quality

        return self


    # ---------------- State setters ---------------- #

    @property
    def pressure(self) -> float | None:
        return self._pressure

    @pressure.setter
    def pressure(self, value: float | None):
        self._pressure = None if value is None else float(value)
        self._clear_property_cache()
        self._validate_liquid_state()

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = float(value)
        self._clear_property_cache()
        self._validate_liquid_state()

    @property
    def pressure_temperature(self) -> Tuple[float | None, float]:
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float | None, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature)")

        self._pressure = None if values[0] is None else float(values[0])
        self._temperature = float(values[1])
        self._clear_property_cache()
        self._validate_liquid_state()

    # ---------------- General API properties ---------------- #

    @property
    def name(self) -> str:
        return self.propellant

    @property
    def backend(self) -> str:
        if self._rocketprops_name is not None and self._cea_name is not None:
            return "RocketProps + CEA"
        if self._rocketprops_name is not None:
            return "RocketProps"
        return "CEA"

    @property
    def propellant(self) -> str:
        if self._registry_name is not None:
            return self._registry_name
        if self._cea_name is not None:
            return self._cea_name
        return str(self._rocketprops_name)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def registry_name(self) -> str | None:
        return self._registry_name

    @property
    def rocketprops_name(self) -> str | None:
        return self._rocketprops_name

    @property
    def cea_name(self) -> str | None:
        """Active CEA name for the current state/phase."""
        return self._cea_name

    @property
    def cea_species(self) -> str | None:
        """CEA gas/product species name, when available."""
        return self._cea_species_name

    @property
    def cea_reactant(self) -> str | None:
        """CEA condensed/reference reactant name, when available."""
        return self._cea_reactant_name

    @property
    def species(self) -> list[str]:
        return [self.propellant]

    @property
    def has_rocketprops(self) -> bool:
        return self._rocketprops_name is not None

    @property
    def has_cea(self) -> bool:
        return self._cea_name is not None

    @property
    def has_cea_reference_data(self) -> bool:
        return self._cea_name is not None

    @property
    def has_cea_species_thermo(self) -> bool:
        return self._cea_species_name is not None and CEA.has_thermo(self._cea_species_name)

    @property
    def has_cea_reactant_thermo(self) -> bool:
        return self._cea_reactant_name is not None and CEA.has_thermo(self._cea_reactant_name)

    @property
    def has_cea_thermo(self) -> bool:
        if self._cea_name is None:
            return False
        return CEA.has_thermo(self._cea_name)

    @property
    def has_cea_transport(self) -> bool:
        if self._cea_name is None:
            return False
        try:
            return CEA.has_transport(self._cea_name)
        except Exception:
            return False

    @property
    def phase(self) -> str:
        if self._quality_override is not None:
            return "Liquid reactant (quality enthalpy correction)"

        if self._backend is not None and self._active_is_liquid_model:
            return "Liquid"
        if self._cea_name is None:
            return "Unknown"
        if CEA.is_reactant(self._cea_name):
            return "Reference Reactant"
        if CEA.is_condensed(self._cea_name):
            label = CEA.phase_label(self._cea_name)
            return f"Condensed ({label})" if label else "Condensed"
        return "Gas"

    @property
    def phase_model(self) -> str:
        if self._quality_override is not None:
            return "RocketProps liquid + CEA reference + CoolProp Δh"

        if self._backend is not None and self._active_is_liquid_model:
            rp_model = (
                "RocketProps saturated liquid table"
                if self.pressure is None
                else "RocketProps compressed liquid table"
            )

            if self._cea_name is None:
                return rp_model

            if self.has_cea_thermo:
                return f"{rp_model} + CEA condensed NASA-9 polynomial"

            return f"{rp_model} + CEA reactant/reference data"

        if self.has_cea_thermo:
            return "CEA NASA-9 polynomial"

        return "CEA reactant/reference data"

    @property
    def quality(self) -> float | None:
        if self._quality_override is not None:
            return self._quality_override
        if self._backend is not None and self._active_is_liquid_model:
            return 0.0
        if self._backend is not None and not self._active_is_liquid_model:
            return 1.0
        return None

    @quality.setter
    def quality(self, value: float):
        self._quality_override = self._validate_quality(value)

        if self._quality_override is not None and self.pressure is None:
            self._pressure = self._pressure_from_fluid_quality(
                self._composition_argument(),
                temperature=self.temperature,
                quality=self._quality_override,
            )

        self._clear_property_cache()
        self._validate_liquid_state()

    # ---------------- Unsupported placeholders ---------------- #

    @property
    def thermal_expansion_coefficient(self):
        return self._unsupported("thermal_expansion_coefficient")

    @property
    def isothermal_compressibility(self):
        return self._unsupported("isothermal_compressibility")

    @property
    def joule_thomson_coefficient(self):
        return self._unsupported("joule_thomson_coefficient")

    @property
    def universal_gas_constant(self):
        return self._RU

    @property
    def prandtl(self):
        return self._unsupported("prandtl")

    @property
    def specific_heat_cv(self) -> float | None:
        cached = self._cache_get("specific_heat_cv")
        if cached is not self._CACHE_MISS:
            return cached

        if not (
            self._cea_name is not None
            and self.has_cea_thermo
            and CEA.is_gas(self._cea_name)
        ):
            return self._cache_set("specific_heat_cv", None)

        cp = self.specific_heat_cp
        R = self.gas_constant

        if cp is None or R is None:
            return self._cache_set("specific_heat_cv", None)

        cv = cp - R

        if cv <= 0.0:
            return self._cache_set("specific_heat_cv", None)

        return self._cache_set("specific_heat_cv", self._source("specific_heat_cv", cv, "CEA ideal gas"))


    @property
    def specific_heat_ratio(self) -> float | None:
        cached = self._cache_get("specific_heat_ratio")
        if cached is not self._CACHE_MISS:
            return cached

        cp = self.specific_heat_cp
        cv = self.specific_heat_cv

        if cp is None or cv is None or cv <= 0.0:
            return self._cache_set("specific_heat_ratio", None)

        return self._cache_set("specific_heat_ratio", self._source("specific_heat_ratio", cp / cv, "CEA ideal gas"))


    @property
    def speed_of_sound(self) -> float | None:
        cached = self._cache_get("speed_of_sound")
        if cached is not self._CACHE_MISS:
            return cached

        if not (
            self._cea_name is not None
            and self.has_cea_thermo
            and CEA.is_gas(self._cea_name)
        ):
            return self._cache_set("speed_of_sound", None)

        gamma = self.specific_heat_ratio
        R = self.gas_constant

        if gamma is None or R is None:
            return self._cache_set("speed_of_sound", None)

        value = gamma * R * self.temperature

        if value <= 0.0:
            return self._cache_set("speed_of_sound", None)

        return self._cache_set("speed_of_sound", self._source("speed_of_sound", float(np.sqrt(value)), "CEA ideal gas"))
    # ---------------- Thermodynamic/reference properties ---------------- #

    @property
    def gibbs_energy(self):
        h = self.enthalpy
        s = self.entropy

        if h is None or s is None:
            return None

        return h - self.temperature * s


    @property
    def helmholtz_energy(self):
        u = self.internal_energy
        s = self.entropy

        if u is None or s is None:
            return None

        return u - self.temperature * s


    @property
    def free_energy(self):
        return self.helmholtz_energy

    @property
    def molar_mass(self) -> float | None:
        """Engineering molar mass in kg/mol when physically meaningful.

        RocketProps-backed propellants use the RocketProps molecular weight.
        CEA gas/thermo species use the CEA molecular weight. CEA-only
        reactant/reference entries such as JP-4 expose their normalized
        empirical-formula molecular weight through ``cea_formula_molar_mass``
        instead, because it is not the real fluid molecular weight.
        """
        cached = self._cache_get("molar_mass")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._real_or_none(self._call("MolWt", "MolecularWt", "MolarMass", default=None))
        if value is not None:
            return self._cache_set(
                "molar_mass",
                self._source("molar_mass", value / 1000.0, "RocketProps"),
            )

        if (
            self._cea_name is not None
            and self.has_cea_thermo
            and CEA.is_gas(self._cea_name)
        ):
            value = self._cea_call("molar_mass", default=None)
            return self._cache_set("molar_mass", self._source("molar_mass", value, "CEA"))

        return self._cache_set("molar_mass", None)

    @property
    def molecular_weight(self) -> float | None:
        mw = self.molar_mass
        if mw is None:
            return None
        return mw * 1000.0

    @property
    def cea_formula_molar_mass(self) -> float | None:
        """CEA molecular weight in kg/mol for the active CEA row.

        For gas/product species, this is the actual molecular weight. For
        CEA reactant/reference entries such as RP-1, JP-4, and Jet-A, this is
        the molecular weight of the normalized empirical formula used by CEA
        for stoichiometry, not necessarily the real propellant molecular weight.
        """
        cached = self._cache_get("cea_formula_molar_mass")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._cea_call("molar_mass", default=None)
        return self._cache_set(
            "cea_formula_molar_mass",
            self._source("cea_formula_molar_mass", value, "CEA"),
        )

    @property
    def cea_molar_mass(self) -> float | None:
        """Backward-compatible alias for ``cea_formula_molar_mass``."""
        value = self.cea_formula_molar_mass
        if value is not None:
            self._data_sources["cea_molar_mass"] = self.property_source("cea_formula_molar_mass") or "CEA"
        return value

    @property
    def gas_constant(self) -> float | None:
        cached = self._cache_get("gas_constant")
        if cached is not self._CACHE_MISS:
            return cached

        if (
            self._cea_name is not None
            and self.has_cea_thermo
            and CEA.is_gas(self._cea_name)
        ):
            mw = self.cea_formula_molar_mass
            if mw is None or mw == 0.0:
                return self._cache_set("gas_constant", None)
            return self._cache_set("gas_constant", self._source("gas_constant", self._RU / mw, "CEA"))

        if self._rocketprops_name is not None:
            mw = self.molar_mass
            if mw is None or mw == 0.0:
                return self._cache_set("gas_constant", None)
            source = self.property_source("molar_mass") or "RocketProps"
            return self._cache_set("gas_constant", self._source("gas_constant", self._RU / mw, source))

        return self._cache_set("gas_constant", None)

    @property
    def elemental_composition(self) -> dict[str, float] | None:
        value = self._cea_call("elemental_composition", default=None)
        return self._source("elemental_composition", value, "CEA")

    @property
    def heat_of_formation_molar(self) -> float | None:
        value = self._cea_call("heat_of_formation_molar", default=None)
        return self._source("heat_of_formation_molar", value, "CEA")

    @property
    def heat_of_formation(self) -> float | None:
        cached = self._cache_get("heat_of_formation")
        if cached is not self._CACHE_MISS:
            return cached

        h_molar = self.heat_of_formation_molar
        mw = self.cea_formula_molar_mass

        if h_molar is None or mw is None or mw == 0.0:
            return self._cache_set("heat_of_formation", None)

        return self._cache_set("heat_of_formation", self._source("heat_of_formation", h_molar / mw, "CEA"))

    @property
    def enthalpy_of_formation(self) -> float | None:
        return self.heat_of_formation

    @property
    def specific_heat_cp(self) -> float | None:
        cached = self._cache_get("specific_heat_cp")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._real_or_none(
            self._call_at_temperature("CpAtTdegR", "CpAtT", default=None)
        ) if self._active_is_liquid_model else None

        if value is not None:
            return self._cache_set(
                "specific_heat_cp",
                self._source(
                    "specific_heat_cp",
                    value * self._BTU_PER_LBM_R_TO_J_PER_KG_K,
                    "RocketProps",
                ),
            )

        value = self._cea_call("cp_mass", self.temperature, default=None)
        return self._cache_set("specific_heat_cp", self._source("specific_heat_cp", value, "CEA"))

    @property
    def specific_heat(self) -> float | None:
        return self.specific_heat_cp

    @property
    def enthalpy(self) -> float | None:
        """Specific enthalpy in J/kg.

        For CEA thermo species, this is the NASA-polynomial enthalpy. For
        RocketProps liquid reactants with only CEA reference data, this anchors
        the liquid enthalpy to the CEA heat of formation and integrates the exact
        thermodynamic liquid relation along the path:

            (Tref, Pref) -> (T, Pref) -> (T, P)

        using RocketProps liquid-property correlations:

            dh = Cp dT + [v - T (dv/dT)_P] dP

        The pressure correction is evaluated numerically from the
        RocketProps liquid-density model. Accuracy therefore depends on
        the underlying RocketProps correlations and should be viewed as
        an engineering estimate rather than a validated liquid equation
        of state.
        """
        cached = self._cache_get("enthalpy")
        if cached is not self._CACHE_MISS:
            return self._from_raw_basis("enthalpy", cached)

        value = self._cea_call("enthalpy_mass", self.temperature, default=None)
        if value is not None and self.has_cea_thermo:
            return self._from_raw_basis("enthalpy", self._cache_set("enthalpy", self._source("enthalpy", value, "CEA")))

        if not self._active_is_liquid_model or self._backend is None:
            return self._from_raw_basis("enthalpy", self._cache_set("enthalpy", None))

        hf = self.heat_of_formation
        tref = self.reference_temperature

        if hf is None or tref is None:
            return self._from_raw_basis("enthalpy", self._cache_set("enthalpy", None))

        pref = self.vapor_pressure_at_temperature(tref)
        if pref is None or not np.isfinite(pref) or pref <= 0.0:
            pref = self._P0_CEA

        pressure = self.pressure if self.pressure is not None else self.vapor_pressure
        if pressure is None or not np.isfinite(pressure) or pressure <= 0.0:
            pressure = pref

        h_temperature, _ = quad(
            lambda T: self._require_number(
                self._cp_at_temperature(T, pref),
                "Could not evaluate RocketProps Cp during liquid enthalpy integration.",
            ),
            tref,
            self.temperature,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            h_pressure, _ = quad(
                lambda P: self._dh_dp_liquid(self.temperature, P),
                pref,
                pressure,
            )

        value = float(hf + h_temperature + h_pressure)
        source = "CEA/RP"

        correction = self.enthalpy_correction
        if correction is not None and correction != 0.0:
            value += correction
            source = "CEA/RP + CoolProp Δh"

        return self._from_raw_basis(
            "enthalpy",
            self._cache_set(
                "enthalpy",
                self._source(
                    "enthalpy",
                    value,
                    source,
                ),
            ),
        )

    @property
    def internal_energy(self) -> float | None:
        cached = self._cache_get("internal_energy")
        if cached is not self._CACHE_MISS:
            return self._from_raw_basis("internal_energy", cached)

        h = self._raw_property_value("enthalpy")

        if h is None:
            return self._from_raw_basis("internal_energy", self._cache_set("internal_energy", None))

        if self._cea_name is not None and CEA.is_gas(self._cea_name):
            R = self.gas_constant
            if R is not None:
                return self._from_raw_basis("internal_energy", self._cache_set("internal_energy", self._source("internal_energy", h - R * self.temperature, "CEA")))

        rho = self.density
        pressure = self.pressure

        if pressure is not None and rho is not None and rho != 0.0:
            return self._from_raw_basis(
                "internal_energy",
                self._cache_set(
                    "internal_energy",
                    self._source(
                        "internal_energy",
                        h - pressure / rho,
                        self.property_source("enthalpy") or "CEA/RP",
                    ),
                ),
            )

        return self._from_raw_basis("internal_energy", self._cache_set("internal_energy", None))

    def _cp_at_temperature(self, temperature: float, pressure: float | None = None) -> float | None:
        """RocketProps liquid Cp at a temporary temperature and optional pressure.

        RocketProps Cp is temperature based, but the temporary pressure is still
        accepted so enthalpy integration follows a clear constant-pressure path.
        """
        T_old = self._temperature
        P_old = self._pressure

        try:
            self._temperature = float(temperature)
            if pressure is not None:
                self._pressure = float(pressure)

            value = self._real_or_none(
                self._call_at_temperature("CpAtTdegR", "CpAtT", default=None)
            )

            if value is None:
                return None

            return value * self._BTU_PER_LBM_R_TO_J_PER_KG_K

        finally:
            self._temperature = T_old
            self._pressure = P_old

    def vapor_pressure_at_temperature(self, temperature: float) -> float | None:
        """RocketProps vapor pressure in Pa at a temporary temperature."""
        T_old = self._temperature

        try:
            self._temperature = float(temperature)
            return self._rocketprops_vapor_pressure_no_source()

        finally:
            self._temperature = T_old

    def _specific_volume_at(self, temperature: float, pressure: float) -> float | None:
        """Liquid specific volume in m^3/kg at temporary T and P."""
        T_old = self._temperature
        P_old = self._pressure

        try:
            self._temperature = float(temperature)
            self._pressure = float(pressure)

            value = self._call_compressed("SG_compressed", default=None)

            if value is None:
                value = self._call_at_temperature("SGLiqAtTdegR", "SGAtTdegR", default=None)

            value = self._real_or_none(value)

            if value is None:
                return None

            rho = value * 1000.0

            if rho == 0.0:
                return None

            return 1.0 / rho

        finally:
            self._temperature = T_old
            self._pressure = P_old

    def _dvdT_constP(self, temperature: float, pressure: float) -> float:
        """Numerical derivative (dv/dT)_P for RocketProps liquid density."""
        dT = max(1e-3, abs(float(temperature)) * 1e-5)

        v1 = self._specific_volume_at(float(temperature) - dT, pressure)
        v2 = self._specific_volume_at(float(temperature) + dT, pressure)

        if v1 is None or v2 is None:
            raise ValueError("Could not evaluate liquid specific volume derivative.")

        return (v2 - v1) / (2.0 * dT)

    def _dh_dp_liquid(self, temperature: float, pressure: float) -> float:
        """Liquid (dh/dP)_T = v - T(dv/dT)_P in J/kg/Pa."""
        v = self._specific_volume_at(temperature, pressure)

        if v is None:
            raise ValueError("Could not evaluate liquid specific volume.")

        dvdT = self._dvdT_constP(temperature, pressure)

        return float(v - temperature * dvdT)

    @property
    def standard_entropy(self) -> float | None:
        cached = self._cache_get("standard_entropy")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._cea_call("entropy_mass_standard", self.temperature, default=None)
        return self._cache_set("standard_entropy", self._source("standard_entropy", value, "CEA"))

    @property
    def entropy(self) -> float | None:
        cached = self._cache_get("entropy")
        if cached is not self._CACHE_MISS:
            return self._from_raw_basis("entropy", cached)

        value = self.standard_entropy

        if value is None:
            return self._from_raw_basis("entropy", self._cache_set("entropy", None))

        if (
            self.pressure is not None
            and self._cea_name is not None
            and CEA.has_thermo(self._cea_name)
            and CEA.is_gas(self._cea_name)
        ):
            mw = self.cea_formula_molar_mass

            if mw is None or mw == 0.0:
                return self._from_raw_basis("entropy", self._cache_set("entropy", None))

            R_cea = self._RU / mw
            value = value - R_cea * np.log(self.pressure / self._P0_CEA)

        return self._from_raw_basis("entropy", self._cache_set("entropy", self._source("entropy", value, "CEA")))

    @property
    def reference_temperature(self) -> float | None:
        cached = self._cache_get("reference_temperature")
        if cached is not self._CACHE_MISS:
            return cached

        if self._cea_name is None or self._cea_index is None:
            return self._cache_set("reference_temperature", None)

        if CEA.is_reactant(self._cea_name):
            ranges = CEA.raw_by_index("t_ranges", self._cea_index)

            if ranges is not None:
                value = float(ranges[0, 0])

                if np.isfinite(value):
                    return self._cache_set("reference_temperature", self._source("reference_temperature", value, "CEA"))

        return self._cache_set("reference_temperature", self._source("reference_temperature", 298.15, "CEA"))

    @property
    def cea_polynomial_temperature_range(self) -> tuple[float, float] | None:
        cached = self._cache_get("cea_polynomial_temperature_range")
        if cached is not self._CACHE_MISS:
            return cached

        if self._cea_name is None or not CEA.has_thermo(self._cea_name):
            return self._cache_set("cea_polynomial_temperature_range", None)

        ranges = CEA.temperature_ranges(self._cea_name)

        if not ranges:
            return self._cache_set("cea_polynomial_temperature_range", None)

        value = (
            self._source("cea_polynomial_minimum_temperature", min(r[0] for r in ranges), "CEA"),
            self._source("cea_polynomial_maximum_temperature", max(r[1] for r in ranges), "CEA"),
        )
        return self._cache_set("cea_polynomial_temperature_range", value)

    @property
    def minimum_temperature(self) -> float | None:
        data_range = getattr(self._backend, "T_data_range", None) if self._backend is not None else None

        if data_range is not None:
            try:
                value = self._real_or_none(data_range()[0])
                if value is not None:
                    return self._source("minimum_temperature", self._K_from_degR(value), "RocketProps")
            except Exception:
                pass

        if self._cea_index is not None:
            ranges = CEA.raw_by_index("t_ranges", self._cea_index)
            if ranges is not None:
                vals = [float(row[0]) for row in ranges if np.isfinite(row[0])]
                if vals:
                    return self._source("minimum_temperature", min(vals), "CEA")

        return self.freezing_temperature

    @property
    def maximum_temperature(self) -> float | None:
        data_range = getattr(self._backend, "T_data_range", None) if self._backend is not None else None

        if data_range is not None:
            try:
                value = self._real_or_none(data_range()[1])
                if value is not None:
                    return self._source("maximum_temperature", self._K_from_degR(value), "RocketProps")
            except Exception:
                pass

        if self._cea_index is not None:
            ranges = CEA.raw_by_index("t_ranges", self._cea_index)
            if ranges is not None:
                vals = [float(row[1]) for row in ranges if np.isfinite(row[1])]
                if vals:
                    return self._source("maximum_temperature", max(vals), "CEA")

        return self.critical_temperature

    # ---------------- Liquid / transport properties ---------------- #

    @property
    def density(self) -> float | None:
        cached = self._cache_get("density")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._call_compressed("SG_compressed", default=None) if self._active_is_liquid_model else None

        if value is None and self._active_is_liquid_model:
            value = self._call_at_temperature("SGLiqAtTdegR", "SGAtTdegR", default=None)

        value = self._real_or_none(value)

        if value is not None:
            return self._cache_set("density", self._source("density", value * 1000.0, "RocketProps"))

        if (
            self.pressure is not None
            and self._cea_name is not None
            and CEA.has_thermo(self._cea_name)
            and CEA.is_gas(self._cea_name)
        ):
            R = self.gas_constant
            if R is not None and R != 0.0:
                return self._cache_set("density", self._source("density", self.pressure / (R * self.temperature), "CEA ideal gas"))

        return self._cache_set("density", None)

    @property
    def specific_volume(self) -> float | None:
        cached = self._cache_get("specific_volume")
        if cached is not self._CACHE_MISS:
            return cached

        rho = self.density

        if rho is None or rho == 0:
            return self._cache_set("specific_volume", None)

        return self._cache_set("specific_volume", self._source("specific_volume", 1.0 / rho, self.property_source("density") or "Unknown"))

    @property
    def dynamic_viscosity(self) -> float | None:
        cached = self._cache_get("dynamic_viscosity")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._call_compressed("Visc_compressed", default=None) if self._active_is_liquid_model else None

        if value is None and self._active_is_liquid_model:
            value = self._call_at_temperature("ViscAtTdegR", "ViscAtT", default=None)

        value = self._real_or_none(value)

        if value is not None:
            return self._cache_set("dynamic_viscosity", self._source("dynamic_viscosity", value * 0.1, "RocketProps"))

        value = self._real_or_none(self._cea_call("viscosity", self.temperature, default=None))
        return self._cache_set("dynamic_viscosity", self._source("dynamic_viscosity", value, "CEA"))

    @property
    def kinematic_viscosity(self) -> float | None:
        cached = self._cache_get("kinematic_viscosity")
        if cached is not self._CACHE_MISS:
            return cached

        mu = self.dynamic_viscosity
        rho = self.density

        if mu is None or rho is None or rho == 0:
            return self._cache_set("kinematic_viscosity", None)

        return self._cache_set(
            "kinematic_viscosity",
            self._source(
                "kinematic_viscosity",
                mu / rho,
                f"{self.property_source('dynamic_viscosity')} + {self.property_source('density')}",
            ),
        )

    @property
    def conductivity(self) -> float | None:
        cached = self._cache_get("conductivity")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._call_at_temperature("CondAtTdegR", "CondAtT", default=None) if self._active_is_liquid_model else None

        value = self._real_or_none(value)

        if value is not None:
            return self._cache_set(
                "conductivity",
                self._source(
                    "conductivity",
                    value * self._BTU_PER_HR_FT_R_TO_W_PER_M_K,
                    "RocketProps",
                ),
            )

        value = self._real_or_none(self._cea_call("conductivity", self.temperature, default=None))
        return self._cache_set("conductivity", self._source("conductivity", value, "CEA"))

    @property
    def thermal_conductivity(self) -> float | None:
        return self.conductivity

    @property
    def vapor_pressure(self) -> float | None:
        cached = self._cache_get("vapor_pressure")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._rocketprops_vapor_pressure_no_source()
        return self._cache_set("vapor_pressure", self._source("vapor_pressure", value, "RocketProps"))

    @property
    def saturation_pressure(self) -> float | None:
        return self.vapor_pressure

    @property
    def saturation_temperature(self) -> float | None:
        cached = self._cache_get("saturation_temperature")
        if cached is not self._CACHE_MISS:
            return cached

        if self._backend is None:
            return self._cache_set("saturation_temperature", None)

        if self.pressure is None:
            return self._cache_set("saturation_temperature", self.temperature)

        Ppsia = self._psia_from_Pa(self.pressure)

        for name in ("TdegRAtPsat", "TsatAtP"):
            fn = getattr(self._backend, name, None)

            if fn is None:
                continue

            try:
                value = self._real_or_none(fn(Ppsia))
                if value is not None:
                    return self._cache_set("saturation_temperature", self._source("saturation_temperature", self._K_from_degR(value), "RocketProps"))
            except Exception:
                continue

        return self._cache_set("saturation_temperature", None)

    @property
    def heat_of_vaporization(self) -> float | None:
        cached = self._cache_get("heat_of_vaporization")
        if cached is not self._CACHE_MISS:
            return cached

        if not self._active_is_liquid_model:
            return self._cache_set("heat_of_vaporization", None)
        value = self._call_at_temperature("HvapAtTdegR", "HvapAtT", default=None)

        value = self._real_or_none(value)

        if value is None:
            return self._cache_set("heat_of_vaporization", None)

        return self._cache_set(
            "heat_of_vaporization",
            self._source(
                "heat_of_vaporization",
                value * self._BTU_PER_LBM_TO_J_PER_KG,
                "RocketProps",
            ),
        )

    @property
    def surface_tension(self) -> float | None:
        cached = self._cache_get("surface_tension")
        if cached is not self._CACHE_MISS:
            return cached

        if not self._active_is_liquid_model:
            return self._cache_set("surface_tension", None)
        value = self._call_at_temperature("SurfAtTdegR", "SurfAtT", default=None)

        value = self._real_or_none(value)

        if value is None:
            return self._cache_set("surface_tension", None)

        return self._cache_set("surface_tension", self._source("surface_tension", value * self._LBF_PER_IN_TO_N_PER_M, "RocketProps"))

    @property
    def saturated_liquid_compressibility_factor(self) -> float | None:
        cached = self._cache_get("saturated_liquid_compressibility_factor")
        if cached is not self._CACHE_MISS:
            return cached

        value = self._call_at_temperature("ZLiqAtTdegR", "ZLiqAtT", default=None)

        value = self._real_or_none(value)

        if value is None:
            return self._cache_set("saturated_liquid_compressibility_factor", None)

        return self._cache_set("saturated_liquid_compressibility_factor", self._source("saturated_liquid_compressibility_factor", value, "RocketProps"))

    @property
    def compressibility(self) -> float | None:
        return self.saturated_liquid_compressibility_factor

    # ---------------- Static RocketProps properties ---------------- #

    @property
    def critical_pressure(self) -> float | None:
        value = self._real_or_none(self._call("Pc", "Pcrit", "P_crit", default=None))

        if value is None:
            return None

        return self._source("critical_pressure", self._Pa_from_psia(value), "RocketProps")

    @property
    def critical_temperature(self) -> float | None:
        value = self._real_or_none(self._call("Tc", "Tcrit", "T_crit", default=None))

        if value is None:
            return None

        return self._source("critical_temperature", self._K_from_degR(value), "RocketProps")

    @property
    def critical_density(self) -> float | None:
        value = self._real_or_none(self._call("SGc", "rhoc", "rho_crit", default=None))

        if value is None:
            return None

        return self._source("critical_density", value * 1000.0, "RocketProps")

    @property
    def freezing_temperature(self) -> float | None:
        value = self._real_or_none(self._call("Tfreeze", "Tfrz", "T_freeze", default=None))

        if value is None:
            return None

        return self._source("freezing_temperature", self._K_from_degR(value), "RocketProps")

    @property
    def boiling_temperature(self) -> float | None:
        value = self._real_or_none(self._call("Tnbp", "Tboil", "T_boil", default=None))

        if value is None:
            return None

        return self._source("boiling_temperature", self._K_from_degR(value), "RocketProps")

    @property
    def minimum_pressure(self) -> float:
        data_range = getattr(self._backend, "P_data_range", None) if self._backend is not None else None

        if data_range is None:
            return 0.0

        try:
            value = self._real_or_none(data_range()[0])
            if value is not None:
                return self._source("minimum_pressure", self._Pa_from_psia(value), "RocketProps")
        except Exception:
            return 0.0

    @property
    def maximum_pressure(self) -> float:
        data_range = getattr(self._backend, "P_data_range", None) if self._backend is not None else None

        if data_range is None:
            return float("inf")

        try:
            value = self._real_or_none(data_range()[1])
            if value is not None:
                return self._source("maximum_pressure", self._Pa_from_psia(value), "RocketProps")
        except Exception:
            return float("inf")

    @property
    def is_mixture(self) -> bool:
        if self._rocketprops_name is None:
            return False
        name = self._rocketprops_name.upper()
        return name.startswith("MON") or name in {"A50", "M20", "MHF3"}

    # ---------------- String output ---------------- #

    def _safe(self, value, fmt=".3e"):
        return format_optional(value, fmt)
    def _safe_property(self, property_name: str, fmt=".3e"):
        try:
            return self._safe(getattr(self, property_name), fmt)
        except NotImplementedError:
            return "N/A"

    def _source_label(self, property_name: str) -> str:
        source = self.property_source(property_name)
        return f" [{source}]" if source else ""

    def __str__(self):
        quality_corrected = self._quality_override is not None

        internal_energy = self.internal_energy
        enthalpy = self.enthalpy
        enthalpy_correction = self.enthalpy_correction
        vapor_pressure = self.vapor_pressure
        saturation_temperature = self.saturation_temperature
        heat_of_vaporization = self.heat_of_vaporization
        molar_mass = self.molar_mass
        cea_formula_molar_mass = self.cea_formula_molar_mass
        heat_of_formation = self.heat_of_formation
        reference_temperature = self.reference_temperature
        cea_thermo_range = self.cea_polynomial_temperature_range
        gas_constant = self.gas_constant
        elemental_composition = self.elemental_composition

        if quality_corrected:
            density = None
            specific_volume = None
            standard_entropy = None
            entropy = None
            dynamic_viscosity = None
            kinematic_viscosity = None
            conductivity = None
            surface_tension = None
            specific_heat_cp = None
            specific_heat_cv = None
            specific_heat_ratio = None
            speed_of_sound = None
        else:
            density = self.density
            specific_volume = self.specific_volume
            standard_entropy = self.standard_entropy
            entropy = self.entropy
            dynamic_viscosity = self.dynamic_viscosity
            kinematic_viscosity = self.kinematic_viscosity
            conductivity = self.conductivity
            surface_tension = self.surface_tension
            specific_heat_cp = self.specific_heat_cp
            specific_heat_cv = self.specific_heat_cv
            specific_heat_ratio = self.specific_heat_ratio
            speed_of_sound = self.speed_of_sound

        def label(property_name: str) -> str:
            if quality_corrected and property_name in {
                "density",
                "specific_volume",
                "standard_entropy",
                "entropy",
                "dynamic_viscosity",
                "kinematic_viscosity",
                "conductivity",
                "surface_tension",
                "specific_heat_cp",
                "specific_heat_cv",
                "specific_heat_ratio",
                "speed_of_sound",
            }:
                return ""

            return self._source_label(property_name)

        rows = [
            ("Propellant", self.propellant),
            ("Input name", self.input_name),
            ("Registry name", self.registry_name or "N/A"),
            ("Backend", self.backend),
            ("RocketProps name", self.rocketprops_name or "N/A"),
            ("Active CEA name", self.cea_name or "N/A"),
            ("CEA species", self.cea_species or "N/A"),
            ("CEA reactant", self.cea_reactant or "N/A"),
            ("Has RocketProps", self.has_rocketprops),
            ("Has CEA", self.has_cea),
            ("Has CEA reference data", self.has_cea_reference_data),
            ("Has active CEA thermo", self.has_cea_thermo),
            ("Has CEA species thermo", self.has_cea_species_thermo),
            ("Has CEA reactant thermo", self.has_cea_reactant_thermo),
            ("Has active CEA transport", self.has_cea_transport),
            ("Phase", self.phase),
            ("Phase model", self.phase_model),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e") if self.pressure is not None else "Saturation/None"),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]" + label("density"), self._safe(density, ".3f")),
            ("Specific volume [m³/kg]" + label("specific_volume"), self._safe(specific_volume, ".3e")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]" + self._source_label("internal_energy"), self._safe(internal_energy, ".3e")),
            ("Enthalpy [J/kg]" + self._source_label("enthalpy"), self._safe(enthalpy, ".3e")),
            ("Enthalpy correction [J/kg]" + self._source_label("enthalpy_correction"), self._safe(enthalpy_correction, ".3e")),
            ("Standard entropy [J/kg-K]" + label("standard_entropy"), self._safe(standard_entropy, ".3e")),
            ("Entropy [J/kg-K]" + label("entropy"), self._safe(entropy, ".3e")),
            ("Dynamic viscosity [Pa·s]" + label("dynamic_viscosity"), self._safe(dynamic_viscosity, ".3e")),
            ("Kinematic viscosity [m²/s]" + label("kinematic_viscosity"), self._safe(kinematic_viscosity, ".3e")),
            ("Conductivity [W/m-K]" + label("conductivity"), self._safe(conductivity, ".3f")),
            ("Surface tension [N/m]" + label("surface_tension"), self._safe(surface_tension, ".3e")),
            ("Vapor pressure [Pa]" + self._source_label("vapor_pressure"), self._safe(vapor_pressure, ".3e")),
            ("Saturation temperature [K]" + self._source_label("saturation_temperature"), self._safe(saturation_temperature, ".2f")),
            ("Heat of vaporization [J/kg]" + self._source_label("heat_of_vaporization"), self._safe(heat_of_vaporization, ".3e")),
            ("Cp [J/kg-K]" + label("specific_heat_cp"), self._safe(specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]" + label("specific_heat_cv"), self._safe(specific_heat_cv, ".3f")),
            ("Specific heat ratio" + label("specific_heat_ratio"), self._safe(specific_heat_ratio, ".5f")),
            ("Molar mass [kg/mol]" + self._source_label("molar_mass"), self._safe(molar_mass, ".6f")),
            ("CEA formula MW [kg/mol]" + self._source_label("cea_formula_molar_mass"), self._safe(cea_formula_molar_mass, ".6f")),
            ("CEA Hf [J/kg]" + self._source_label("heat_of_formation"), self._safe(heat_of_formation, ".3e")),
            ("CEA reference T [K]" + self._source_label("reference_temperature"), self._safe(reference_temperature, ".2f")),
            ("CEA thermo range [K]", cea_thermo_range or "N/A"),
            ("Gas constant [J/kg-K]" + self._source_label("gas_constant"), self._safe(gas_constant, ".3f")),
            ("Elemental composition" + self._source_label("elemental_composition"), elemental_composition or "N/A"),
            ("Speed of sound [m/s]" + label("speed_of_sound"), self._safe(speed_of_sound, ".3f")),
        ]

        return format_rows(rows)

    def __repr__(self) -> str:
        pressure = "None" if self.pressure is None else f"{self.pressure:.3e}"
        return (
            f"{self.__class__.__name__}(propellant={self.propellant!r}, "
            f"temperature={self.temperature:.2f} K, "
            f"pressure={pressure} Pa, quality={self.quality!r}, "
            f"backend={self.backend!r})"
        )

    # ---------------- Utilities ---------------- #

    @staticmethod
    def get_available_propellants() -> list[str]:
        names = set(SpeciesDatabase.supported_species("Propellant"))
        names.update(CEA.reactant_names)
        names.update(CEA.names)
        return sorted(names)

    @staticmethod
    def show_available_propellants() -> list[str]:
        names = Propellant.get_available_propellants()
        for name in names:
            print(name)
        return names

    @staticmethod
    def get_available_cea_species() -> list[str]:
        return CEA.names

    @staticmethod
    def show_available_cea_species() -> list[str]:
        return CEA.show_species()

    @staticmethod
    def get_available_cea_reactants() -> list[str]:
        return CEA.reactant_names

    @staticmethod
    def show_available_cea_reactants() -> list[str]:
        return CEA.show_reactants()

    @staticmethod
    def get_available_rocketprops() -> list[str]:
        return sorted(
            name
            for name in SpeciesDatabase.species()
            if SpeciesDatabase._record(name).rocketprops is not None
        )

    @staticmethod
    def show_available_rocketprops() -> list[str]:
        names = Propellant.get_available_rocketprops()
        for name in names:
            print(name)
        return names

    @staticmethod
    def get_available_fluids() -> list[str]:
        return Propellant.get_available_propellants()

    @staticmethod
    def show_available_fluids() -> list[str]:
        return Propellant.show_available_propellants()

    @classmethod
    def show_aliases(cls) -> dict[str, str]:
        return {}

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        return sorted("-".join(sorted(inputs)) for inputs in cls._FLASH_INPUTS)

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
