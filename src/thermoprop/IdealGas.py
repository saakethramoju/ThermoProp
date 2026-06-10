from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import root_scalar
import pyromat as pm

from .FluidRegistry import FluidRegistry
from CEADatabase import CEA


class IdealGas:
    """
    PYroMat ideal-gas wrapper with a Fluid-like API.

    Thermodynamic properties are evaluated with PYroMat. Pure-species
    transport properties use generated NASA CEA / CEAM transport data when
    available, with Sutherland viscosity retained as a fallback.

    Supports thermal state from:
        temperature
        enthalpy
        internal_energy

    Pressure is optional.

    Density can be used only with another closure:
        density + pressure         -> temperature
        density + temperature      -> pressure
        density + enthalpy         -> temperature, pressure
        density + internal_energy  -> temperature, pressure

        
    --- IMPORTANT!! ---
    PYroMat defines its own enthalpy and internal energy reference states.

    Absolute enthalpy and internal energy values should not be directly compared
    with those from other thermodynamic libraries unless a common reference basis
    has been established.
    """

    _BACKEND_NAME = "PYroMat"



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

    _SUTHERLAND_VISCOSITY = {
        "ig.air": {"mu0": 1.716e-5, "T0": 273.0, "S": 111.0},
        "ig.Ar": {"mu0": 2.125e-5, "T0": 273.0, "S": 114.0},
        "ig.CO2": {"mu0": 1.370e-5, "T0": 273.0, "S": 222.0},
        "ig.CO": {"mu0": 1.657e-5, "T0": 273.0, "S": 136.0},
        "ig.N2": {"mu0": 1.663e-5, "T0": 273.0, "S": 107.0},
        "ig.O2": {"mu0": 1.919e-5, "T0": 273.0, "S": 139.0},
        "ig.H2": {"mu0": 8.411e-6, "T0": 273.0, "S": 97.0},
        "ig.H2O": {"mu0": 1.12e-5, "T0": 350.0, "S": 1064.0},
    }

    _RU = 8.31446261815324  # J/mol-K

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
    ):
        self._configure_units()

        if quality is not None:
            raise ValueError("IdealGas does not support vapor quality.")

        self._species_ids: List[str] = []
        self._display_names: List[str] = []

        if isinstance(fluid, str):
            sid, display = self._normalize_name(fluid)
            self._species_ids = [sid]
            self._display_names = [display]
            self._mole_fractions = np.array([1.0])
            self._mass_fractions = np.array([1.0])
            self._mixture = False

        elif isinstance(fluid, dict):
            if basis not in ("mass", "mole"):
                raise ValueError("basis must be 'mass' or 'mole'")

            tmp: Dict[str, Tuple[float, List[str]]] = {}

            for user_name, frac in fluid.items():
                sid, display = self._normalize_name(user_name)
                total, names = tmp.get(sid, (0.0, []))
                tmp[sid] = (total + float(frac), names + [display])

            self._species_ids = list(tmp.keys())
            fractions = np.array([v[0] for v in tmp.values()], dtype=float)
            self._display_names = [
                ", ".join(sorted(set(v[1])))
                for v in tmp.values()
            ]

            if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{basis.capitalize()} fractions must sum to 1.0")

            if basis == "mole":
                self._mole_fractions = fractions
                self._mass_fractions = self.mole_to_mass(
                    self._species_ids,
                    fractions,
                )
            else:
                self._mass_fractions = fractions
                self._mole_fractions = self.mass_to_mole(
                    self._species_ids,
                    fractions,
                )

            self._mixture = len(self._species_ids) > 1

        else:
            raise TypeError("fluid must be a string or a dict mixture")

        # Cache each species' strict CEA name once during construction.
        # CEADatabase owns the generated CEAM files and transport evaluation.
        self._cea_transport_names = self._build_cea_transport_names()

        self._species = [pm.get(sid) for sid in self._species_ids]
        self._M = self._molar_masses()
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

    @property
    def name(self) -> str:
        return ", ".join(self._display_names)
    
    @property
    def backend(self) -> str:
        """Name of the thermodynamic property backend."""
        return self._BACKEND_NAME
    # ---------------- Units ---------------- #

    @staticmethod
    def _configure_units():
        pm.config["unit_pressure"] = "Pa"
        pm.config["unit_temperature"] = "K"
        pm.config["unit_energy"] = "J"
        pm.config["unit_matter"] = "kg"
        pm.config["unit_volume"] = "m3"
        pm.config["unit_molar"] = "mol"

    # ---------------- State setting / flashing ---------------- #

    def _set_state(
        self,
        pressure: float | None = None,
        temperature: float | None = None,
        enthalpy: float | None = None,
        internal_energy: float | None = None,
        density: float | None = None,
    ) -> None:
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

        if provided not in self._FLASH_INPUTS:
            raise LookupError(
                "Unsupported ideal-gas state input combination. "
                f"Supported inputs are: {self.available_flash_inputs()}."
            )

        if provided == frozenset(("pressure", "density")):
            self._pressure = float(pressure)
            self._temperature = self._pressure / (float(density) * self.gas_constant)
            self._enthalpy = self._enthalpy_from_temperature(self._temperature)
            return

        if temperature is not None:
            self._temperature = float(temperature)
            self._enthalpy = self._enthalpy_from_temperature(self._temperature)

        elif enthalpy is not None:
            self._enthalpy = float(enthalpy)
            self._temperature = self._temperature_from_enthalpy(self._enthalpy)

        elif internal_energy is not None:
            self._temperature = self._temperature_from_internal_energy(float(internal_energy))
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

    # ---------------- Internal helpers ---------------- #

    def _require_pressure(self, property_name: str = "This property"):
        if self._pressure is None:
            raise ValueError(f"{property_name} requires pressure. Set gas.pressure first.")

    def _mix_mass_weighted(
        self,
        method: str,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
    ):
        vals = []

        for sp in self._species:
            fn = getattr(sp, method)
            kwargs = {}

            if temperature is not None:
                kwargs["T"] = temperature

            if pressure is not None:
                kwargs["p"] = pressure

            vals.append(float(np.asarray(fn(**kwargs)).squeeze()))

        return float(np.dot(self._mass_fractions, vals))

    def _enthalpy_from_temperature(self, temperature: float) -> float:
        return self._mix_mass_weighted("h", temperature=temperature)

    def _internal_energy_from_temperature(self, temperature: float) -> float:
        return self._mix_mass_weighted("e", temperature=temperature)

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
            f"Could not solve ideal-gas temperature from "
            f"{variable_name}={target_value:.6g} J/kg "
            f"over temperature=[{minimum_temperature:.3f}, {maximum_temperature:.3f}] K."
        )

    def _partial_pressures(self) -> np.ndarray:
        self._require_pressure("Partial pressures")
        return self._mole_fractions * self._pressure

    # ---------------- Fractions ---------------- #

    @property
    def mole_fractions(self) -> dict:
        return {
            name: float(x)
            for name, x in zip(self._display_names, self._mole_fractions)
        }

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        if len(self._species_ids) == 1:
            raise ValueError("Cannot change mole fractions for a pure gas")

        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")

        self._mole_fractions = np.array(value, dtype=float)
        self._mass_fractions = self._mole_fractions * self._M / np.dot(self._mole_fractions, self._M)

        if self._last_state_values is not None:
            self._set_state(**self._last_state_values)

    @property
    def mass_fractions(self) -> dict:
        return {
            name: float(x)
            for name, x in zip(self._display_names, self._mass_fractions)
        }

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        if len(self._species_ids) == 1:
            raise ValueError("Cannot change mass fractions for a pure gas")

        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")

        self._mass_fractions = np.array(value, dtype=float)
        inv = self._mass_fractions / self._M
        self._mole_fractions = inv / inv.sum()

        if self._last_state_values is not None:
            self._set_state(**self._last_state_values)

    # ---------------- State setters ---------------- #

    @property
    def pressure(self) -> float | None:
        return self._pressure

    @pressure.setter
    def pressure(self, value: float):
        self._pressure = float(value)

    @property
    def enthalpy(self) -> float:
        return self._enthalpy

    @enthalpy.setter
    def enthalpy(self, value: float):
        self._enthalpy = float(value)
        self._temperature = self._temperature_from_enthalpy(self._enthalpy)

    @property
    def internal_energy(self) -> float:
        return self._internal_energy_from_temperature(self._temperature)

    @internal_energy.setter
    def internal_energy(self, value: float):
        self._temperature = self._temperature_from_internal_energy(float(value))
        self._enthalpy = self._enthalpy_from_temperature(self._temperature)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = float(value)
        self._enthalpy = self._enthalpy_from_temperature(self._temperature)

    @property
    def density(self) -> float:
        self._require_pressure("Density")
        return self._pressure / (self.gas_constant * self._temperature)

    @density.setter
    def density(self, value: float):
        if self._temperature is None:
            raise ValueError("Cannot set density without temperature.")
        self._pressure = float(value) * self.gas_constant * self._temperature

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
        return self._pressure, self._enthalpy

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
        return self.density, self._enthalpy

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

    # Backward-compatible aliases
    @property
    def HP(self) -> Tuple[float, float | None]:
        return self._enthalpy, self._pressure

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

    def _unsupported(self, property_name: str):
        raise NotImplementedError(
            f"IdealGas.{property_name} is not supported by this wrapper."
        )


    @property
    def species(self) -> List[str]:
        return self._display_names

    @property
    def phase(self) -> str:
        return "Ideal Gas"

    @property
    def compressibility(self) -> float:
        return 1.0

    @property
    def specific_volume(self) -> float:
        return 1.0 / self.density
            
    @property
    def thermal_expansion_coefficient(self) -> float:
        """Volumetric thermal expansion coefficient beta [1/K]."""
        return 1.0 / self.temperature

    @property
    def isothermal_compressibility(self) -> float:
        """Isothermal compressibility [1/Pa]."""
        self._require_pressure("Isothermal compressibility")
        return 1.0 / self.pressure

    @property
    def helmholtz_energy(self) -> float:
        """Mass-specific Helmholtz free energy [J/kg]."""
        return self.free_energy

    @property
    def molar_mass(self) -> float:
        return float(1.0 / np.sum(self._mass_fractions / self._M))

    @property
    def gas_constant(self) -> float:
        return self._RU / self.molar_mass

    @property
    def specific_heat_cp(self) -> float:
        return self._mix_mass_weighted("cp", temperature=self._temperature)

    @property
    def specific_heat_cv(self) -> float:
        return self._mix_mass_weighted("cv", temperature=self._temperature)

    @property
    def specific_heat(self) -> float:
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        cp = self.specific_heat_cp
        cv = self.specific_heat_cv
        return None if cv == 0 else cp / cv

    @property
    def free_energy(self) -> float:
        self._require_pressure("Free energy")
        try:
            return self._mix_mass_weighted(
                "f",
                temperature=self._temperature,
                pressure=self._pressure,
            )
        except Exception:
            return self.internal_energy - self._temperature * self.entropy

    @property
    def gibbs_energy(self) -> float:
        self._require_pressure("Gibbs energy")
        try:
            if not self._mixture:
                return self._mix_mass_weighted(
                    "g",
                    temperature=self._temperature,
                    pressure=self._pressure,
                )

            vals = []
            for wi, sp, pi in zip(
                self._mass_fractions,
                self._species,
                self._partial_pressures(),
            ):
                vals.append(wi * float(np.asarray(sp.g(T=self._temperature, p=pi)).squeeze()))
            return float(sum(vals))

        except Exception:
            return self.enthalpy - self._temperature * self.entropy

    @property
    def entropy(self) -> float:
        self._require_pressure("Entropy")

        if not self._mixture:
            return self._mix_mass_weighted(
                "s",
                temperature=self._temperature,
                pressure=self._pressure,
            )

        vals = []
        for wi, sp, pi in zip(
            self._mass_fractions,
            self._species,
            self._partial_pressures(),
        ):
            vals.append(wi * float(np.asarray(sp.s(T=self._temperature, p=pi)).squeeze()))
        return float(sum(vals))

    @property
    def quality(self) -> float:
        return 1.0

    @quality.setter
    def quality(self, value: float):
        raise ValueError("IdealGas does not support vapor quality.")

    @property
    def speed_of_sound(self) -> float:
        return float(np.sqrt(self.specific_heat_ratio * self.gas_constant * self._temperature))
        
    def _build_cea_transport_names(self) -> list[str | None]:
        """Return cached strict CEA names for species with transport data.

        A value of None means the species was not found in CEADatabase's
        transport table. Runtime property calls can then fall back to the older
        approximate methods without repeatedly checking the registry.
        """
        names: list[str | None] = []

        for display_name in self._display_names:
            try:
                cea_name = FluidRegistry.cea_name(display_name)

                if CEA.has_transport(cea_name):
                    names.append(CEA.resolve_name(cea_name))
                else:
                    names.append(None)

            except Exception:
                names.append(None)

        return names

    def _species_viscosity_sutherland(self, species_id: str) -> float:
        """Return pure-species dynamic viscosity [Pa-s] from Sutherland's law.

        This is retained as a fallback for gases that do not have CEADatabase
        transport data, preserving the previous IdealGas behavior.
        """
        if species_id not in self._SUTHERLAND_VISCOSITY:
            raise NotImplementedError(
                f"Neither CEA nor Sutherland viscosity data are available for {self.name}."
            )

        data = self._SUTHERLAND_VISCOSITY[species_id]

        mu0 = data["mu0"]
        T0 = data["T0"]
        S = data["S"]
        T = self.temperature

        return float(mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S))

    def _species_viscosity(self, species_id: str, species_position: int | None = None) -> float:
        """Return pure-species dynamic viscosity [Pa-s].

        CEADatabase transport data are preferred. Sutherland's law is only used as
        a fallback so existing supported gases keep working if CEA data are
        unavailable for a species or temperature.
        """
        if species_position is not None:
            cea_name = self._cea_transport_names[species_position]

            if cea_name is not None:
                try:
                    return CEA.viscosity(cea_name, self.temperature)
                except Exception:
                    pass

        return self._species_viscosity_sutherland(species_id)

    def _species_conductivity(self, species_position: int) -> float:
        """Return pure-species thermal conductivity [W/m-K] from CEA data."""
        cea_name = self._cea_transport_names[species_position]

        if cea_name is None:
            raise NotImplementedError(
                f"CEA thermal conductivity data are not available for {self._display_names[species_position]}."
            )

        return CEA.conductivity(cea_name, self.temperature)

    @property
    def _eucken_prandtl(self) -> float | None:
        """Fallback approximate ideal-gas Prandtl number.

        This preserves the older IdealGas behavior for species or mixtures that
        do not have CEA conductivity support.
        """
        gamma = self.specific_heat_ratio

        if gamma is None or abs(9.0 * gamma - 5.0) < 1e-15:
            return None

        return 4.0 * gamma / (9.0 * gamma - 5.0)

    def _mixture_viscosity_wilke(self) -> float:
        """Return ideal-gas-mixture viscosity [Pa-s] using Wilke's rule."""
        x = self._mole_fractions
        M = self._M

        mu = np.array(
            [
                self._species_viscosity(sid, species_position=i)
                for i, sid in enumerate(self._species_ids)
            ],
            dtype=float,
        )

        phi = np.zeros((len(mu), len(mu)))

        for i in range(len(mu)):
            for j in range(len(mu)):
                phi[i, j] = (
                    (1.0 + np.sqrt(mu[i] / mu[j]) * (M[j] / M[i]) ** 0.25) ** 2
                    / np.sqrt(8.0 * (1.0 + M[i] / M[j]))
                )

        return float(
            sum(
                x[i] * mu[i] / sum(x[j] * phi[i, j] for j in range(len(mu)))
                for i in range(len(mu))
            )
        )

    @property
    def dynamic_viscosity(self) -> float:
        """
        Dynamic viscosity [Pa-s].

        Pure gases use CEADatabase transport fits when available.
        Mixtures use Wilke's rule with those pure-species viscosities.
        """
        if self._mixture:
            return self._mixture_viscosity_wilke()

        return self._species_viscosity(self._species_ids[0], 0)

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        return self.dynamic_viscosity / self.density

    @property
    def prandtl(self) -> float:
        """Prandtl number [-].

        For pure gases with CEA conductivity data, this is calculated from
        Pr = Cp * mu / k. Otherwise it falls back to the older Eucken-style
        approximation so existing IdealGas behavior is preserved.
        """
        if not self._mixture:
            try:
                k = self._species_conductivity(0)

                if k is not None and k != 0.0:
                    return self.specific_heat_cp * self.dynamic_viscosity / k
            except Exception:
                pass

        return self._eucken_prandtl

    @property
    def conductivity(self) -> float:
        """Thermal conductivity [W/m-K].

        Pure gases use CEADatabase transport fits when available.
        If CEA conductivity is unavailable, this preserves the previous
        approximation k = Cp * mu / Pr.
        """
        if not self._mixture:
            try:
                return self._species_conductivity(0)
            except Exception:
                pass

        Pr = self._eucken_prandtl

        if Pr is None or Pr == 0.0:
            return None

        return self.specific_heat_cp * self.dynamic_viscosity / Pr

    @property
    def thermal_conductivity(self) -> float:
        """Alias for conductivity."""
        return self.conductivity

    @property
    def minimum_pressure(self) -> float:
        return 1e-9

    @property
    def maximum_pressure(self) -> float:
        return np.inf

    @property
    def minimum_temperature(self) -> float:
        return self._minimum_temperature

    @property
    def maximum_temperature(self) -> float:
        return self._maximum_temperature

    @property
    def is_mixture(self) -> bool:
        return self._mixture
    

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        """
        Return selected ideal-gas first partial derivatives.

        Supported variables:
            T, P, Dmass, Hmass, Umass
        """
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
            raise ValueError("Ideal-gas partial derivatives require pressure.")

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
            f"IdealGas partial derivative d({of})/d({wrt})|{const} is not implemented."
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
            ("Gas(es)", ", ".join(self._display_names)),
            ("Mole fractions", format_dict(self.mole_fractions, 3)),
            ("Mass fractions", format_dict(self.mass_fractions, 3)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f") if self._pressure is not None else "N/A"),
            ("Compressibility Z", self._safe(self.compressibility, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e") if self._pressure is not None else "N/A"),
            ("Cp [J/kg-K]", self._safe(self.specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]", self._safe(self.specific_heat_cv, ".3f")),
            ("Specific heat ratio", self._safe(self.specific_heat_ratio, ".5f")),
            ("Gas constant [J/kg-K]", self._safe(self.gas_constant, ".3f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("Speed of sound [m/s]", self._safe(self.speed_of_sound, ".3f")),
        ]

        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)

    def __repr__(self) -> str:
        species_str = ", ".join(self._display_names)
        pressure_str = "None" if self._pressure is None else f"{self._pressure:.3e}"
        return (
            f"{self.__class__.__name__}(species=[{species_str}], "
            f"pressure={pressure_str} Pa, "
            f"enthalpy={self._enthalpy:.3e} J/kg, "
            f"temperature={self.temperature:.2f} K)"
        )

    # ---------------- Utilities ---------------- #

    @classmethod
    def _normalize_name(cls, user_name: str) -> Tuple[str, str]:
        """Return the PYroMat species ID and display name for a user name."""
        sid = FluidRegistry.pyromat_name(user_name, include_prefix=True)
        display = FluidRegistry.name(user_name)

        try:
            pm.get(sid)
        except Exception:
            raise ValueError(
                f"Invalid ideal gas '{user_name}'. "
                f"Use IdealGas.show_available_gases() to check valid names."
            )

        return sid, display

    @staticmethod
    def get_available_gases() -> List[str]:
        """Return available PYroMat ideal-gas names."""
        return sorted(FluidRegistry.pyromat_supported_names)

    @staticmethod
    def show_available_gases() -> List[str]:
        """Print and return available PYroMat ideal-gas names."""
        gases = IdealGas.get_available_gases()

        for gas in gases:
            print(gas)

        return gases

    @staticmethod
    def get_available_fluids() -> List[str]:
        """Return available PYroMat ideal-gas names.

        Fluid-style alias for API consistency with Fluid.
        """
        return IdealGas.get_available_gases()

    @staticmethod
    def show_available_fluids() -> List[str]:
        """Print and return available PYroMat ideal-gas names.

        Fluid-style alias for API consistency with Fluid.
        """
        return IdealGas.show_available_gases()

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        """Return supported ideal-gas state input combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
        )

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        """Return supported ideal-gas state input combinations."""
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        """Return supported two-property ideal-gas flash combinations."""
        return sorted(
            "-".join(sorted(inputs))
            for inputs in cls._FLASH_INPUTS
            if len(inputs) == 2
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        """Return supported two-property ideal-gas flash combinations."""
        return cls.available_flash_pairs()

    @staticmethod
    def _molar_mass_of(species_id: str) -> float:
        if not species_id.startswith("ig."):
            species_id = f"ig.{species_id}"
        return float(np.asarray(pm.get(species_id).mw()).squeeze())

    def _molar_masses(self) -> np.ndarray:
        return np.array(
            [self._molar_mass_of(sid) for sid in self._species_ids],
            dtype=float,
        )

    def _temperature_limits(self) -> Tuple[float, float]:
        mins = []
        maxs = []

        for sp in self._species:
            try:
                Tlim = sp.Tlim()
                mins.append(float(Tlim[0]))
                maxs.append(float(Tlim[1]))
            except Exception:
                mins.append(200.0)
                maxs.append(6000.0)

        return max(mins), min(maxs)

    @staticmethod
    def mole_to_mass(species_ids: List[str], mole_fractions: List[float]):
        if not np.isclose(sum(mole_fractions), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")
        x = np.asarray(mole_fractions, dtype=float)
        M = np.array([IdealGas._molar_mass_of(sid) for sid in species_ids])
        return x * M / np.dot(x, M)

    @staticmethod
    def mass_to_mole(species_ids: List[str], mass_fractions: List[float]):
        if not np.isclose(sum(mass_fractions), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")
        w = np.asarray(mass_fractions, dtype=float)
        M = np.array([IdealGas._molar_mass_of(sid) for sid in species_ids])
        inv = w / M
        return inv / inv.sum()

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
