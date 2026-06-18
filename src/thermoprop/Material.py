# Material.py

from __future__ import annotations

from typing import Tuple

import numpy as np

from .MaterialDatabase import MaterialDatabase
from ._api import PropertyIntrospectionMixin
from ._formatting import format_optional, rounded_dict, format_rows
from ._state_api import UNSET, is_provided, provided_items


class Material(PropertyIntrospectionMixin):
    """
    ThermoProp material-property lookup wrapper.

    Material provides temperature-dependent solid material properties from
    MaterialDatabase with a ThermoProp-style API. It is intended for structural,
    thermal, and heat-transfer calculations involving metals and engineering
    materials.

    State
    -----

    Material properties are temperature-dependent only:

        Material("in718", temperature=300.0)

    Pressure is not used. The pressure attribute is always None, and setting
    pressure raises an error.

    Interpolation
    -------------

    Properties are linearly interpolated from tabulated temperature curves. If a
    property has a single stored value, that value is treated as constant.

    By default, allow_extrapolation=True clamps values outside the available
    temperature range to the nearest endpoint. Set allow_extrapolation=False to
    raise an error outside the stored range.

    Examples
    --------

        mat = Material("in718", temperature=300.0)

        rho = mat.density
        k = mat.thermal_conductivity
        cp = mat.specific_heat
        alpha = mat.thermal_diffusivity

        mat.temperature = 900.0
        sy = mat.yield_strength

        sy_600 = mat.get("yield_strength", temperature=600.0)

    Available data
    --------------

    Use available_properties and available_property_units to inspect stored
    properties for the selected material.

    Public API units are SI.
    """
    _BACKEND_NAME = "ThermoProp MaterialDatabase"

    _UNSUPPORTED_PROPERTIES = {
        "enthalpy",
        "internal_energy",
        "entropy",
        "quality",
        "dynamic_viscosity",
        "kinematic_viscosity",
        "speed_of_sound",
        "specific_heat_cv",
        "specific_heat_ratio",
        "isothermal_compressibility",
        "joule_thomson_coefficient",
        "partial_derivative",
        "helmholtz_energy",
        "gibbs_energy",
        "gas_constant",
        "universal_gas_constant",
        "prandtl",
        "surface_tension",
        "vapor_pressure",
        "saturation_pressure",
        "saturation_temperature",
        "heat_of_vaporization",
    }

    _FLASH_INPUTS = {
        frozenset(("temperature",)),
        frozenset(("pressure", "temperature")),
    }

    def __init__(
        self,
        material: str,
        temperature: float = 298.15,
        allow_extrapolation: bool = True,
    ):
        self._material_name = self._normalize_name(material)

        try:
            self._data = MaterialDatabase._data(self._material_name)
        except KeyError:
            raise ValueError(
                f"Material {self._material_name!r} exists in MaterialDatabase, "
                "but has no data block in MATERIAL_DATA."
            )

        self._temperature = float(temperature)
        self.allow_extrapolation = bool(allow_extrapolation)
        self._curve_cache: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def cache_key(self) -> tuple:
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.material,
            float(self.temperature),
            bool(self.allow_extrapolation),
        )

    # ---------------- Core package-style API ---------------- #

    @property
    def name(self) -> str:
        return self.material

    @property
    def backend(self) -> str:
        return self._BACKEND_NAME

    @property
    def material(self) -> str:
        return self._material_name

    @property
    def species(self) -> list[str]:
        return [self.material]

    @property
    def category(self) -> str:
        return self._data.get("category", MaterialDatabase._category(self.material))

    @property
    def default_condition(self) -> str:
        return self._data.get("default_condition", "")

    @property
    def phase(self) -> str:
        return "Solid"

    @property
    def is_mixture(self) -> bool:
        return False

    # ---------------- State ---------------- #

    @property
    def temperature(self) -> float:
        """Absolute temperature in K."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = float(value)

    @property
    def pressure(self) -> None:
        return None

    @pressure.setter
    def pressure(self, value):
        raise ValueError("Material properties are only temperature-dependent.")

    @property
    def pressure_temperature(self) -> Tuple[None, float]:
        return None, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[None, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature).")

        pressure, temperature = values

        if pressure is not None:
            raise ValueError("Material properties are pressure-independent. Use pressure=None.")

        self.temperature = temperature



    def update(
        self,
        material=UNSET,
        *,
        temperature=UNSET,
        pressure=UNSET,
        allow_extrapolation=UNSET,
    ):
        """Update material selection or temperature in place.

        Materials are temperature-dependent in ThermoProp.  Passing a non-None
        pressure is rejected to match the existing pressure setter behavior.
        """

        if is_provided(pressure) and pressure is not None:
            raise ValueError("Material properties are only temperature-dependent.")

        if is_provided(material):
            new_temperature = self.temperature if not is_provided(temperature) else temperature
            new_allow = self.allow_extrapolation if not is_provided(allow_extrapolation) else allow_extrapolation
            rebuilt = self.__class__(
                material,
                temperature=new_temperature,
                allow_extrapolation=new_allow,
            )
            self.__dict__.update(rebuilt.__dict__)
            return self

        if is_provided(allow_extrapolation):
            self.allow_extrapolation = bool(allow_extrapolation)

        if is_provided(temperature):
            self.temperature = temperature

        return self

    def set_state(self, *, temperature: float):
        self.update(temperature=temperature)
        return self

    # ---------------- Data access ---------------- #

    @property
    def available_properties(self) -> list[str]:
        return sorted(self._data.get("properties", {}).keys())

    @property
    def available_property_units(self) -> dict[str, str]:
        return {
            name: prop.get("units", "")
            for name, prop in self._data.get("properties", {}).items()
        }

    def has_property(self, property_name: str) -> bool:
        try:
            prop_name = MaterialDatabase._normalize_property(property_name)
        except ValueError:
            return False

        return prop_name in self._data.get("properties", {})

    def _get_property_data(self, property_name: str) -> tuple[str, dict]:
        try:
            prop_name = MaterialDatabase._normalize_property(property_name)
        except ValueError as exc:
            raise AttributeError(str(exc)) from None

        properties = self._data.get("properties", {})

        if prop_name not in properties:
            raise AttributeError(
                f"Material {self.material!r} has no property {prop_name!r}. "
                f"Available properties: {self.available_properties}"
            )

        return prop_name, properties[prop_name]

    def _get_curve_arrays(self, property_name: str) -> tuple[str, np.ndarray, np.ndarray]:
        prop_name, prop = self._get_property_data(property_name)

        cached = self._curve_cache.get(prop_name)
        if cached is not None:
            return cached

        temperatures = np.asarray(prop["temperature"], dtype=float)
        values = np.asarray(prop["value"], dtype=float)

        if temperatures.size == 0:
            raise ValueError(
                f"Material {self.material!r} property {prop_name!r} has an empty curve."
            )

        if temperatures.size != values.size:
            raise ValueError(
                f"Material {self.material!r} property {prop_name!r} has mismatched "
                f"temperature/value arrays: {temperatures.size} temperatures, "
                f"{values.size} values."
            )

        order = np.argsort(temperatures)
        cached = prop_name, temperatures[order], values[order]
        self._curve_cache[prop_name] = cached
        return cached

    def temperature_range(self, property_name: str) -> tuple[float, float]:
        _, temperatures, _ = self._get_curve_arrays(property_name)
        return float(temperatures[0]), float(temperatures[-1])

    def minimum_temperature(self, property_name: str) -> float:
        return self.temperature_range(property_name)[0]

    def maximum_temperature(self, property_name: str) -> float:
        return self.temperature_range(property_name)[1]

    def get(
        self,
        property_name: str,
        temperature: float | None = None,
        allow_extrapolation: bool | None = None,
    ) -> float:
        """
        Return a material property at temperature.

        Missing properties raise AttributeError.

        By default, temperatures outside the stored curve range raise ValueError.
        Set allow_extrapolation=True to allow np.interp endpoint clamping.
        """
        prop_name, temperatures, values = self._get_curve_arrays(property_name)

        T = self.temperature if temperature is None else float(temperature)
        allow = self.allow_extrapolation if allow_extrapolation is None else bool(allow_extrapolation)

        if temperatures.size == 1:
            return float(values[0])

        Tmin = float(temperatures[0])
        Tmax = float(temperatures[-1])

        if not allow and (T < Tmin or T > Tmax):
            raise ValueError(
                f"{prop_name!r} for {self.material!r} is only available from "
                f"{Tmin:.6g} K to {Tmax:.6g} K. Got {T:.6g} K. "
                "Pass allow_extrapolation=True to clamp to the nearest endpoint."
            )

        return float(np.interp(T, temperatures, values))


    def units(self, property_name: str) -> str:
        _, prop = self._get_property_data(property_name)
        return prop.get("units", "")

    def curve(self, property_name: str) -> tuple[np.ndarray, np.ndarray]:
        _, temperatures, values = self._get_curve_arrays(property_name)
        return temperatures, values

    # ---------------- Mechanical properties ---------------- #

    @property
    def yield_strength(self) -> float:
        return self.get("yield_strength")

    @property
    def ultimate_strength(self) -> float:
        return self.get("ultimate_strength")

    @property
    def tensile_strength(self) -> float:
        return self.ultimate_strength

    @property
    def elastic_modulus(self) -> float:
        return self.get("elastic_modulus")

    @property
    def youngs_modulus(self) -> float:
        return self.elastic_modulus

    @property
    def young_modulus(self) -> float:
        return self.elastic_modulus

    @property
    def torsional_modulus(self) -> float:
        return self.get("torsional_modulus")

    @property
    def shear_modulus(self) -> float:
        return self.torsional_modulus

    @property
    def poisson_ratio(self) -> float:
        return self.get("poisson_ratio")

    # ---------------- Thermal / electrical properties ---------------- #

    @property
    def density(self) -> float:
        return self.get("density")

    @property
    def specific_volume(self) -> float:
        rho = self.density
        if rho == 0:
            return None
        return 1.0 / rho

    @property
    def thermal_conductivity(self) -> float:
        return self.get("thermal_conductivity")

    @property
    def conductivity(self) -> float:
        return self.thermal_conductivity

    @property
    def specific_heat(self) -> float:
        return self.get("specific_heat")

    @property
    def specific_heat_cp(self) -> float:
        return self.specific_heat

    @property
    def coefficient_of_thermal_expansion(self) -> float:
        return self.get("coefficient_of_thermal_expansion")
        
    @property
    def thermal_expansion_coefficient(self) -> float:
        return self.coefficient_of_thermal_expansion

    @property
    def cte(self) -> float:
        return self.coefficient_of_thermal_expansion
        
    @property
    def thermal_diffusivity(self) -> float:
        """
        Thermal diffusivity [m²/s].

        alpha = k / (rho Cp)
        """
        rho = self.density
        Cp = self.specific_heat_cp
        k = self.thermal_conductivity

        if rho is None or Cp is None or k is None or rho == 0.0 or Cp == 0.0:
            return None

        return k / (rho * Cp)

    @property
    def melting_point(self) -> float:
        return self.get("melting_point")

    @property
    def freezing_temperature(self) -> float:
        return self.melting_point

    @property
    def electrical_resistivity(self) -> float:
        return self.get("electrical_resistivity")

    # ---------------- Unsupported Fluid-like properties ---------------- #

    def _unsupported(self, property_name: str):
        raise NotImplementedError(
            f"Material.{property_name} is not supported. "
            "Material only provides temperature-dependent isotropic solid "
            "property curves from MaterialDatabase.py."
        )

    @property
    def enthalpy(self):
        return self._unsupported("enthalpy")

    @property
    def internal_energy(self):
        return self._unsupported("internal_energy")

    @property
    def entropy(self):
        return self._unsupported("entropy")

    @property
    def quality(self):
        return self._unsupported("quality")

    @quality.setter
    def quality(self, value):
        raise ValueError("Material does not support vapor quality.")

    @property
    def dynamic_viscosity(self):
        return self._unsupported("dynamic_viscosity")

    @property
    def kinematic_viscosity(self):
        return self._unsupported("kinematic_viscosity")

    @property
    def speed_of_sound(self):
        return self._unsupported("speed_of_sound")

    @property
    def specific_heat_cv(self):
        return self._unsupported("specific_heat_cv")

    @property
    def specific_heat_ratio(self):
        return self._unsupported("specific_heat_ratio")

    # ---------------- String output ---------------- #

    def _safe(self, value, fmt=".3e"):
        return format_optional(value, fmt)
    def _safe_property(self, property_name: str, fmt=".3e"):
        try:
            return self._safe(getattr(self, property_name), fmt)
        except (AttributeError, NotImplementedError, ValueError):
            return "N/A"

    def __str__(self):
        rows = [
            ("Material", self.material),
            ("Backend", self.backend),
            ("Category", self.category),
            ("Default condition", self.default_condition),
            ("Phase", self.phase),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe_property("density", ".3f")),
            ("Specific volume [m³/kg]", self._safe_property("specific_volume", ".3e")),
            ("Yield strength [Pa]", self._safe_property("yield_strength", ".3e")),
            ("Ultimate strength [Pa]", self._safe_property("ultimate_strength", ".3e")),
            ("Elastic modulus [Pa]", self._safe_property("elastic_modulus", ".3e")),
            ("Torsional modulus [Pa]", self._safe_property("torsional_modulus", ".3e")),
            ("Poisson ratio", self._safe_property("poisson_ratio", ".5f")),
            ("Thermal conductivity [W/m-K]", self._safe_property("thermal_conductivity", ".3f")),
            ("Specific heat [J/kg-K]", self._safe_property("specific_heat", ".3f")),
            ("CTE [1/K]", self._safe_property("coefficient_of_thermal_expansion", ".3e")),
            ("Melting point [K]", self._safe_property("melting_point", ".2f")),
            ("Electrical resistivity [Ohm-m]", self._safe_property("electrical_resistivity", ".3e")),
        ]

        return format_rows(rows)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(material={self.material!r}, "
            f"temperature={self.temperature:.2f} K)"
        )

    # ---------------- Utilities ---------------- #

    @classmethod
    def _normalize_name(cls, user_name: str) -> str:
        return MaterialDatabase._name(user_name)

    @staticmethod
    def get_available_materials() -> list[str]:
        """Return available ThermoProp material names."""
        return MaterialDatabase.materials()

    @staticmethod
    def show_available_materials() -> list[str]:
        names = Material.get_available_materials()
        for name in names:
            print(name)
        return names

    @staticmethod
    def get_available_properties() -> list[str]:
        """Return normalized material-property names known by the registry."""
        return sorted(MaterialDatabase._properties())

    @staticmethod
    def show_available_properties() -> list[str]:
        properties = Material.get_available_properties()
        for prop in properties:
            print(prop)
        return properties

    @classmethod
    def show_aliases(cls) -> dict[str, str]:
        return MaterialDatabase._show_aliases()
        

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        """Return supported material state input combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
        )

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        """Return supported material state input combinations."""
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        """Return supported two-property material state input combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
            if len(inputs) == 2
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        """Return supported two-property material state input combinations."""
        return cls.available_flash_pairs()
