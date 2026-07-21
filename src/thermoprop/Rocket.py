"""CEA-style theoretical rocket-performance wrapper.

``Rocket`` is intentionally a thin layer over :class:`Reactants`,
:class:`Equilibrium`, and :class:`CombustionGas`.  ``Reactants`` owns the
propellant definition and feed conditions.  ``Equilibrium`` owns the HP and SP
chemistry solves.  This module only adds the one-dimensional rocket-flow
closures needed to arrange those states into chamber, throat, and requested
nozzle stations.

All public inputs and outputs use SI units.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Real
from typing import Any

from scipy.optimize import brentq

from .CombustionGas import CombustionGas
from .Equilibrium import Equilibrium
from .Exceptions import ThermoPropConfigurationError, ThermoPropStateError
from .Reactants import Reactants
from ._formatting import format_optional, format_rows
from ._state_api import UNSET, is_provided


_STANDARD_GRAVITY = 9.80665
_ROOT_RELATIVE_TOLERANCE = 1.0e-10
_PRESSURE_MATCH_RELATIVE_TOLERANCE = 2.0e-8
_AREA_MATCH_RELATIVE_TOLERANCE = 2.0e-8


def _as_values(value, name: str) -> list[float]:
    """Normalize a scalar or iterable of scalars to a validated float list."""
    if value is None:
        return []

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a number or an iterable of numbers.")

    if isinstance(value, Iterable):
        values = [float(item) for item in value]
    else:
        values = [float(value)]

    for item in values:
        if not math.isfinite(item):
            raise ValueError(f"{name} values must be finite.")

    return values


def _normalize_frozen_at(value: str | float | None) -> str | float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError(
            "frozen_at must be None, 'chamber', 'throat', or a supersonic area ratio."
        )

    if isinstance(value, Real):
        ratio = float(value)
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError(
                "A numeric frozen_at value must be a finite supersonic area ratio "
                "greater than or equal to 1."
            )
        if math.isclose(
            ratio,
            1.0,
            rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
            abs_tol=1.0e-12,
        ):
            return "throat"
        return ratio

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": None,
        "equilibrium": None,
        "chamber": "chamber",
        "combustion": "chamber",
        "combustor_end": "chamber",
        "combustor": "chamber",
        "inf": "chamber",
        "infinite_area": "chamber",
        "infinite_area_chamber": "chamber",
        "throat": "throat",
    }

    if key not in aliases:
        raise ValueError(
            "frozen_at must be None, 'chamber', 'throat', or a numeric "
            "supersonic area ratio. "
            "For a finite-area combustor, 'chamber' means the combustor-end station."
        )

    return aliases[key]


def _clone_station(
    station: "RocketStation",
    *,
    name: str,
    kind: str,
    branch: str | None,
    requested_pressure: float | None = None,
    requested_area_ratio: float | None = None,
) -> "RocketStation":
    return RocketStation(
        name=name,
        thermo=station.thermo,
        kind=kind,
        branch=branch,
        chemistry=station.chemistry,
        frozen_from=station.frozen_from,
        velocity=station.velocity,
        mach=station.mach,
        mass_flux=station.mass_flux,
        area_ratio=station.area_ratio,
        pressure_ratio=station.pressure_ratio,
        requested_pressure=requested_pressure,
        requested_area_ratio=requested_area_ratio,
        is_freeze_station=station.is_freeze_station,
    )


class RocketStation:
    """Thermodynamic and flow data at one rocket station.

    Users do not construct this object directly.  ``Rocket`` returns stations
    through ``chamber``, ``throat``, ``exit``, and the grouped station lists.
    Thermodynamic properties are delegated to the underlying ``Equilibrium`` or
    ``CombustionGas`` object, so normal access remains concise::

        rocket.throat.temperature
        rocket.exit.mass_fractions
        rocket.exit.speed_of_sound

    The wrapper adds rocket-flow quantities such as velocity, Mach number,
    mass flux, pressure ratio, area ratio, thrust coefficient, and specific
    impulse.
    """

    def __init__(
        self,
        *,
        name: str,
        thermo: Equilibrium | CombustionGas,
        kind: str,
        branch: str | None,
        chemistry: str,
        frozen_from: str | None,
        velocity: float,
        mach: float,
        mass_flux: float,
        area_ratio: float | None,
        pressure_ratio: float | None,
        requested_pressure: float | None = None,
        requested_area_ratio: float | None = None,
        is_freeze_station: bool = False,
    ):
        self.name = str(name)
        self.thermo = thermo
        self.kind = str(kind)
        self.branch = branch
        self.chemistry = str(chemistry)
        self.frozen_from = frozen_from
        self.velocity = float(velocity)
        self.mach = float(mach)
        self.mass_flux = float(mass_flux)
        self.area_ratio = None if area_ratio is None else float(area_ratio)
        self.pressure_ratio = None if pressure_ratio is None else float(pressure_ratio)
        self.requested_pressure = (
            None if requested_pressure is None else float(requested_pressure)
        )
        self.requested_area_ratio = (
            None if requested_area_ratio is None else float(requested_area_ratio)
        )
        self.is_freeze_station = bool(is_freeze_station)
        self._characteristic_velocity: float | None = None

    def _set_characteristic_velocity(self, value: float) -> None:
        self._characteristic_velocity = float(value)

    def __getattr__(self, name: str):
        # ``thermo`` is assigned before delegation can occur.  This guard keeps
        # error messages sensible during object construction or unpickling.
        thermo = self.__dict__.get("thermo")
        if thermo is None:
            raise AttributeError(name)
        return getattr(thermo, name)

    @property
    def label(self) -> str:
        """Short station label used in reports."""
        if self.kind == "pressure":
            return f"P={self.pressure:.4g} Pa"
        if self.kind == "area_ratio":
            prefix = "SUB" if self.branch == "subsonic" else "SUP"
            return f"{prefix} A/At={self.area_ratio:.5g}"
        return self.name

    @property
    def area_per_mass_flow(self) -> float | None:
        """Area per unit mass flow, in m²/(kg/s)."""
        if self.mass_flux <= 0.0:
            return None
        return 1.0 / self.mass_flux

    @property
    def characteristic_velocity(self) -> float | None:
        """Rocket characteristic velocity, in m/s."""
        return self._characteristic_velocity

    @property
    def cstar(self) -> float | None:
        """Alias for :attr:`characteristic_velocity`."""
        return self.characteristic_velocity

    @property
    def thrust_coefficient(self) -> float | None:
        """Matched-pressure thrust coefficient, ``velocity / cstar``."""
        if self._characteristic_velocity is None:
            return None
        return self.velocity / self._characteristic_velocity

    @property
    def cf(self) -> float | None:
        """Alias for :attr:`thrust_coefficient`."""
        return self.thrust_coefficient

    @property
    def vacuum_effective_exhaust_velocity(self) -> float | None:
        """Effective exhaust velocity for zero ambient pressure, in m/s."""
        if self.mass_flux <= 0.0:
            return None
        return self.velocity + self.pressure / self.mass_flux

    @property
    def vacuum_thrust_coefficient(self) -> float | None:
        """Vacuum thrust coefficient for this station."""
        if self._characteristic_velocity is None:
            return None
        effective_velocity = self.vacuum_effective_exhaust_velocity
        if effective_velocity is None:
            return None
        return effective_velocity / self._characteristic_velocity

    @property
    def cf_vac(self) -> float | None:
        """Alias for :attr:`vacuum_thrust_coefficient`."""
        return self.vacuum_thrust_coefficient

    @property
    def specific_impulse(self) -> float:
        """Matched-pressure specific impulse, in seconds."""
        return self.velocity / _STANDARD_GRAVITY

    @property
    def isp(self) -> float:
        """Alias for :attr:`specific_impulse`."""
        return self.specific_impulse

    @property
    def vacuum_specific_impulse(self) -> float | None:
        """Vacuum specific impulse, in seconds."""
        effective_velocity = self.vacuum_effective_exhaust_velocity
        if effective_velocity is None:
            return None
        return effective_velocity / _STANDARD_GRAVITY

    @property
    def isp_vac(self) -> float | None:
        """Alias for :attr:`vacuum_specific_impulse`."""
        return self.vacuum_specific_impulse

    def as_dict(self, trace: float = 1.0e-12) -> dict[str, Any]:
        """Return thermodynamic, flow, performance, and composition data."""
        if hasattr(self.thermo, "as_dict"):
            thermo_data = self.thermo.as_dict(trace=trace)
        else:
            thermo_data = {
                "pressure": self.pressure,
                "temperature": self.temperature,
                "density": self.density,
                "enthalpy": self.enthalpy,
                "internal_energy": self.internal_energy,
                "entropy": self.entropy,
                "molecular_weight": self.molecular_weight,
                "specific_heat_cp": self.specific_heat_cp,
                "specific_heat_cv": self.specific_heat_cv,
                "specific_heat_ratio": self.specific_heat_ratio,
                "speed_of_sound": self.speed_of_sound,
                "mole_fractions": {
                    name: value
                    for name, value in self.mole_fractions.items()
                    if value >= trace
                },
                "mass_fractions": {
                    name: value
                    for name, value in self.mass_fractions.items()
                    if value >= trace
                },
            }
        return {
            "name": self.name,
            "kind": self.kind,
            "branch": self.branch,
            "chemistry": self.chemistry,
            "frozen_from": self.frozen_from,
            "velocity": self.velocity,
            "mach": self.mach,
            "mass_flux": self.mass_flux,
            "area_per_mass_flow": self.area_per_mass_flow,
            "area_ratio": self.area_ratio,
            "pressure_ratio": self.pressure_ratio,
            "is_freeze_station": self.is_freeze_station,
            "characteristic_velocity": self.characteristic_velocity,
            "thrust_coefficient": self.thrust_coefficient,
            "vacuum_thrust_coefficient": self.vacuum_thrust_coefficient,
            "specific_impulse": self.specific_impulse,
            "vacuum_specific_impulse": self.vacuum_specific_impulse,
            "thermo": thermo_data,
        }

    def __repr__(self) -> str:
        return (
            f"RocketStation(name={self.name!r}, pressure={self.pressure:.6g}, "
            f"temperature={self.temperature:.6g}, mach={self.mach:.6g}, "
            f"area_ratio={self.area_ratio!r}, chemistry={self.chemistry!r}, "
            f"is_freeze_station={self.is_freeze_station!r})"
        )


class _ExpansionModel:
    """Internal equilibrium or fixed-composition isentropic expansion model."""

    def __init__(
        self,
        *,
        reactants: Reactants,
        entropy: float,
        guess_temperature: float,
        chemistry: str,
        freeze_state: Equilibrium | None = None,
        frozen_from: str | None = None,
    ):
        self.reactants = reactants
        self.entropy = float(entropy)
        self.guess_temperature = float(guess_temperature)
        self.chemistry = chemistry
        self.frozen_from = frozen_from
        self._cache: dict[float, Equilibrium | CombustionGas] = {}

        if chemistry == "frozen":
            if freeze_state is None:
                raise ThermoPropStateError(
                    "A freeze state is required for frozen rocket expansion."
                )
            if freeze_state.condensed_species:
                names = ", ".join(freeze_state.condensed_species)
                raise ThermoPropStateError(
                    "Frozen rocket expansion currently requires a gas-only freeze "
                    f"composition; condensed species were present: {names}."
                )
            self._composition = dict(freeze_state.gas_mole_fractions)
            # Re-evaluate the same fixed gas at the freeze state so its entropy is
            # exactly on CombustionGas's fixed-composition reference.
            gas = CombustionGas(
                self._composition,
                basis="mole",
                pressure=freeze_state.pressure,
                temperature=freeze_state.temperature,
            )
            self.entropy = gas.entropy
            self.guess_temperature = gas.temperature
        else:
            self._composition = None

    @staticmethod
    def _key(pressure: float) -> float:
        # Pressure roots revisit nearly identical floating-point values.  A
        # rounded key avoids repeating expensive equilibrium solves without
        # affecting engineering precision.
        return round(float(pressure), 7)

    def state(self, pressure: float) -> Equilibrium | CombustionGas:
        pressure = float(pressure)
        key = self._key(pressure)
        state = self._cache.get(key)
        if state is not None:
            return state

        if self.chemistry == "equilibrium":
            state = Equilibrium(
                self.reactants,
                mode="sp",
                pressure=pressure,
                entropy=self.entropy,
                guess_temperature=self.guess_temperature,
            )
        else:
            state = CombustionGas(
                self._composition,
                basis="mole",
                pressure=pressure,
                entropy=self.entropy,
            )

        self._cache[key] = state
        return state


class Rocket:
    """Solve a CEA-style theoretical rocket problem from ``Reactants``.

    Parameters
    ----------
    reactants:
        Complete ThermoProp ``Reactants`` definition.  Fuel and oxidizer names,
        blend fractions, mixture ratio, temperatures, pressures, inerts, and
        igniters remain owned by this object.
    chamber_pressure:
        Assigned combustion pressure in Pa.  For the normal infinite-area
        chamber model this is the infinite-area chamber pressure.  When
        ``contraction_ratio`` is provided, CEA's finite-area combustor model is
        selected and this assigned pressure is the injector-face pressure.
    exit_pressures:
        Optional absolute nozzle-station pressures in Pa.  A scalar or iterable
        may be supplied.
    subsonic_area_ratios:
        Optional subsonic ``A/At`` values.  A scalar or iterable may be supplied.
    supersonic_area_ratios:
        Optional supersonic ``A/At`` values.  A scalar or iterable may be supplied.
    frozen_at:
        ``None`` for equilibrium expansion, ``"chamber"`` to freeze composition
        at the chamber/combustor-end station, ``"throat"`` to equilibrate to
        the throat and freeze all downstream supersonic stations, or a numeric
        supersonic ``A/At`` value to freeze at that nozzle station.  A numeric
        value of ``1.0`` is equivalent to freezing at the throat.
    contraction_ratio:
        Optional finite-area combustor ``Ac/At``.  Supplying it automatically
        selects the FAC model; omitting it selects the IAC model.

    Notes
    -----
    ``Rocket`` adds only one-dimensional continuity, momentum, energy, choking,
    and area-ratio closures.  It deliberately reuses ``Equilibrium`` for HP
    combustion and equilibrium SP expansion, and ``CombustionGas`` for the
    fixed-composition SP states downstream of a freeze station.
    """

    def __init__(
        self,
        reactants: Reactants,
        chamber_pressure: float,
        *,
        exit_pressures=None,
        subsonic_area_ratios=None,
        supersonic_area_ratios=None,
        frozen_at: str | float | None = None,
        contraction_ratio: float | None = None,
    ):
        if not isinstance(reactants, Reactants):
            raise TypeError("reactants must be a ThermoProp Reactants object.")

        self._reactants = reactants
        self._chamber_pressure = float(chamber_pressure)
        self._exit_pressures = _as_values(exit_pressures, "exit_pressures")
        self._subsonic_area_ratios = _as_values(
            subsonic_area_ratios,
            "subsonic_area_ratios",
        )
        self._supersonic_area_ratios = _as_values(
            supersonic_area_ratios,
            "supersonic_area_ratios",
        )
        self._frozen_at = _normalize_frozen_at(frozen_at)
        self._contraction_ratio = (
            None if contraction_ratio is None else float(contraction_ratio)
        )

        self._injector: RocketStation | None = None
        self._infinite_area_chamber: RocketStation | None = None
        self._chamber: RocketStation | None = None
        self._throat: RocketStation | None = None
        self._freeze_station: RocketStation | None = None
        self._pressure_stations: list[RocketStation] = []
        self._subsonic_stations: list[RocketStation] = []
        self._supersonic_stations: list[RocketStation] = []
        self._characteristic_velocity: float | None = None
        self._reference_pressure: float | None = None
        self._total_enthalpy: float | None = None
        self._dirty = True

        self._validate_inputs()
        self.solve()

    def _validate_inputs(self) -> None:
        if not math.isfinite(self._chamber_pressure) or self._chamber_pressure <= 0.0:
            raise ValueError("chamber_pressure must be a finite positive pressure in Pa.")

        for pressure in self._exit_pressures:
            if pressure <= 0.0:
                raise ValueError("exit_pressures must contain positive pressures in Pa.")

        for ratio in self._subsonic_area_ratios:
            if ratio < 1.0:
                raise ValueError("subsonic_area_ratios must be greater than or equal to 1.")

        for ratio in self._supersonic_area_ratios:
            if ratio < 1.0:
                raise ValueError("supersonic_area_ratios must be greater than or equal to 1.")

        if self._contraction_ratio is not None:
            if not math.isfinite(self._contraction_ratio) or self._contraction_ratio <= 1.0:
                raise ValueError("contraction_ratio must be greater than 1 for FAC.")
            if self._frozen_at == "chamber":
                raise ThermoPropConfigurationError(
                    "FAC with frozen_at='chamber' is not supported because the frozen "
                    "throat mass flux must be coupled back into the finite-area combustor "
                    "closure. Use frozen_at=None, frozen_at='throat', or a numeric "
                    "supersonic area ratio."
                )
            too_large = [
                value
                for value in self._subsonic_area_ratios
                if value > self._contraction_ratio * (1.0 + _AREA_MATCH_RELATIVE_TOLERANCE)
            ]
            if too_large:
                raise ValueError(
                    "FAC subsonic_area_ratios must not exceed contraction_ratio; "
                    f"received {too_large}."
                )

    @property
    def reactants(self) -> Reactants:
        return self._reactants

    @property
    def chamber_pressure(self) -> float:
        """Return the assigned constructor pressure in Pa.

        In FAC mode this is the assigned injector-face pressure.  Use
        ``rocket.chamber.pressure`` for the solved combustor-end pressure.
        """
        return self._chamber_pressure

    @chamber_pressure.setter
    def chamber_pressure(self, value: float) -> None:
        self.update(chamber_pressure=value)

    @property
    def assigned_pressure(self) -> float:
        """Alias for the pressure assigned to the rocket problem."""
        return self._chamber_pressure

    @property
    def exit_pressures(self) -> list[float]:
        return list(self._exit_pressures)

    @property
    def subsonic_area_ratios(self) -> list[float]:
        return list(self._subsonic_area_ratios)

    @property
    def supersonic_area_ratios(self) -> list[float]:
        return list(self._supersonic_area_ratios)

    @property
    def frozen_at(self) -> str | float | None:
        return self._frozen_at

    @property
    def contraction_ratio(self) -> float | None:
        return self._contraction_ratio

    @property
    def model(self) -> str:
        return "FAC" if self.is_fac else "IAC"

    @property
    def is_fac(self) -> bool:
        return self._contraction_ratio is not None

    @property
    def is_frozen(self) -> bool:
        return self._frozen_at is not None

    @property
    def is_stale(self) -> bool:
        return self._dirty

    @property
    def injector(self) -> RocketStation | None:
        """Injector-face station for FAC, otherwise ``None``."""
        return self._injector

    @property
    def infinite_area_chamber(self) -> RocketStation:
        """Infinite-area chamber station used as the ``cstar`` reference."""
        if self._infinite_area_chamber is None:
            raise ThermoPropStateError("Rocket has not been solved.")
        return self._infinite_area_chamber

    @property
    def chamber(self) -> RocketStation:
        """Physical chamber station: IAC chamber or FAC combustor end."""
        if self._chamber is None:
            raise ThermoPropStateError("Rocket has not been solved.")
        return self._chamber

    @property
    def combustor_end(self) -> RocketStation | None:
        """FAC combustor-end station, otherwise ``None``."""
        return self._chamber if self.is_fac else None

    @property
    def throat(self) -> RocketStation:
        if self._throat is None:
            raise ThermoPropStateError("Rocket has not been solved.")
        return self._throat

    @property
    def freeze_station(self) -> RocketStation | None:
        """Equilibrium station whose composition is frozen downstream.

        Returns ``None`` for a fully equilibrium rocket.  For chamber and throat
        freezing this returns the corresponding physical station.  For a
        numeric ``frozen_at`` value it returns the automatically solved
        supersonic area-ratio station, even when that ratio was not included in
        ``supersonic_area_ratios``.
        """
        return self._freeze_station

    @property
    def pressure_stations(self) -> list[RocketStation]:
        return list(self._pressure_stations)

    @property
    def subsonic_stations(self) -> list[RocketStation]:
        return list(self._subsonic_stations)

    @property
    def supersonic_stations(self) -> list[RocketStation]:
        return list(self._supersonic_stations)

    @property
    def exits(self) -> list[RocketStation]:
        """All user-requested pressure and area-ratio stations."""
        return [
            *self._pressure_stations,
            *self._subsonic_stations,
            *self._supersonic_stations,
        ]

    @property
    def exit(self) -> RocketStation:
        """Return the last user-requested nozzle station."""
        exits = self.exits
        if not exits:
            raise ThermoPropStateError(
                "No exit stations were requested. Supply exit_pressures, "
                "subsonic_area_ratios, or supersonic_area_ratios."
            )
        return exits[-1]

    @property
    def stations(self) -> list[RocketStation]:
        """Stations in normal report order."""
        main: list[RocketStation] = []
        if self._injector is not None:
            main.append(self._injector)
        main.extend([self.chamber, self.throat])

        pressure_and_subsonic = [
            *self._pressure_stations,
            *self._subsonic_stations,
        ]
        supersonic = list(self._supersonic_stations)

        freeze_station = self._freeze_station
        if (
            freeze_station is not None
            and freeze_station not in main
            and freeze_station not in pressure_and_subsonic
            and freeze_station not in supersonic
        ):
            freeze_ratio = freeze_station.area_ratio
            insertion_index = len(supersonic)
            if freeze_ratio is not None:
                for index, station in enumerate(supersonic):
                    if (
                        station.area_ratio is not None
                        and station.area_ratio > freeze_ratio
                    ):
                        insertion_index = index
                        break
            supersonic.insert(insertion_index, freeze_station)

        return [*main, *pressure_and_subsonic, *supersonic]

    @property
    def all_stations(self) -> list[RocketStation]:
        """All stations, including the fictitious FAC infinite-area state."""
        if not self.is_fac:
            return self.stations
        stations = self.stations
        return [stations[0], self.infinite_area_chamber, *stations[1:]]

    @property
    def reference_pressure(self) -> float:
        """Infinite-area pressure used in the characteristic-velocity definition."""
        if self._reference_pressure is None:
            raise ThermoPropStateError("Rocket has not been solved.")
        return self._reference_pressure

    @property
    def characteristic_velocity(self) -> float:
        """Characteristic velocity ``cstar``, in m/s."""
        if self._characteristic_velocity is None:
            raise ThermoPropStateError("Rocket has not been solved.")
        return self._characteristic_velocity

    @property
    def cstar(self) -> float:
        return self.characteristic_velocity

    @property
    def throat_mass_flux(self) -> float:
        """Throat mass flux, in kg/(m² s)."""
        return self.throat.mass_flux

    @property
    def mass_flow_per_throat_area(self) -> float:
        return self.throat_mass_flux

    @property
    def isp(self) -> float:
        return self.exit.isp

    @property
    def isp_vac(self) -> float | None:
        return self.exit.isp_vac

    @property
    def cf(self) -> float | None:
        return self.exit.cf

    @property
    def cf_vac(self) -> float | None:
        return self.exit.cf_vac

    @staticmethod
    def _root_xtol(pressure_scale: float) -> float:
        return max(1.0e-7, abs(float(pressure_scale)) * 1.0e-11)

    def _flow_values(
        self,
        thermo: Equilibrium | CombustionGas,
    ) -> tuple[float, float, float]:
        if self._total_enthalpy is None:
            raise ThermoPropStateError("Rocket total enthalpy is not available.")

        kinetic_energy = 2.0 * (self._total_enthalpy - thermo.enthalpy)
        if kinetic_energy < 0.0 and abs(kinetic_energy) < max(
            1.0e-5,
            abs(self._total_enthalpy) * 1.0e-11,
        ):
            kinetic_energy = 0.0
        if kinetic_energy < 0.0:
            raise ThermoPropStateError(
                "Nozzle state enthalpy exceeds the rocket total enthalpy. "
                "The requested pressure is not on the downstream expansion branch."
            )

        velocity = math.sqrt(kinetic_energy)
        speed_of_sound = float(thermo.speed_of_sound)
        mach = velocity / speed_of_sound
        mass_flux = float(thermo.density) * velocity
        return velocity, mach, mass_flux

    def _make_station(
        self,
        *,
        name: str,
        thermo: Equilibrium | CombustionGas,
        kind: str,
        branch: str | None,
        chemistry: str,
        frozen_from: str | None,
        area_ratio: float | None = None,
        requested_pressure: float | None = None,
        requested_area_ratio: float | None = None,
        zero_velocity: bool = False,
        is_freeze_station: bool = False,
    ) -> RocketStation:
        if zero_velocity:
            velocity = 0.0
            mach = 0.0
            mass_flux = 0.0
        else:
            velocity, mach, mass_flux = self._flow_values(thermo)

        reference_pressure = self._reference_pressure
        pressure_ratio = (
            None
            if reference_pressure is None
            else reference_pressure / float(thermo.pressure)
        )

        station = RocketStation(
            name=name,
            thermo=thermo,
            kind=kind,
            branch=branch,
            chemistry=chemistry,
            frozen_from=frozen_from,
            velocity=velocity,
            mach=mach,
            mass_flux=mass_flux,
            area_ratio=area_ratio,
            pressure_ratio=pressure_ratio,
            requested_pressure=requested_pressure,
            requested_area_ratio=requested_area_ratio,
            is_freeze_station=is_freeze_station,
        )
        return station

    def _find_low_supersonic_pressure(
        self,
        model: _ExpansionModel,
        high_pressure: float,
        residual,
    ) -> float:
        low = high_pressure * 0.5
        minimum = max(1.0e-3, self.reference_pressure * 1.0e-12)
        last_error: Exception | None = None

        for _ in range(80):
            try:
                if residual(low) >= 0.0:
                    return low
            except Exception as exc:  # try a less aggressive pressure first
                last_error = exc
            low *= 0.5
            if low <= minimum:
                break

        message = "Could not bracket the requested supersonic rocket station."
        if last_error is not None:
            message += f" Last property error: {last_error}"
        raise ThermoPropStateError(message)

    def _solve_throat(
        self,
        model: _ExpansionModel,
        *,
        upper_pressure: float,
    ) -> RocketStation:
        upper = float(upper_pressure) * (1.0 - 1.0e-9)

        def residual(pressure: float) -> float:
            thermo = model.state(pressure)
            velocity, mach, _ = self._flow_values(thermo)
            return velocity - thermo.speed_of_sound

        lower = upper * 0.25
        for _ in range(60):
            try:
                if residual(lower) > 0.0:
                    break
            except Exception:
                pass
            lower *= 0.5
        else:
            raise ThermoPropStateError("Could not bracket the choked throat pressure.")

        pressure = brentq(
            residual,
            lower,
            upper,
            xtol=self._root_xtol(upper_pressure),
            rtol=_ROOT_RELATIVE_TOLERANCE,
            maxiter=100,
        )
        thermo = model.state(pressure)
        station = self._make_station(
            name="Throat",
            thermo=thermo,
            kind="throat",
            branch="throat",
            chemistry=model.chemistry,
            frozen_from=model.frozen_from,
            area_ratio=1.0,
        )

        if abs(station.mach - 1.0) > 2.0e-5:
            raise ThermoPropStateError(
                f"Throat solve did not converge to Mach 1; obtained {station.mach:.8g}."
            )
        return station

    def _solve_area_station(
        self,
        ratio: float,
        *,
        branch: str,
        model: _ExpansionModel,
        allow_fac_chamber_match: bool = True,
    ) -> RocketStation:
        ratio = float(ratio)
        throat = self.throat

        if math.isclose(ratio, 1.0, rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE, abs_tol=1.0e-12):
            return _clone_station(
                throat,
                name=f"{branch.title()} A/At=1",
                kind="area_ratio",
                branch=branch,
                requested_area_ratio=ratio,
            )

        def residual(pressure: float) -> float:
            thermo = model.state(pressure)
            _, _, mass_flux = self._flow_values(thermo)
            return throat.mass_flux / mass_flux - ratio

        if branch == "subsonic":
            lower = throat.pressure * (1.0 + 1.0e-9)
            upper_station = self.chamber

            if allow_fac_chamber_match and self.is_fac and math.isclose(
                ratio,
                self.contraction_ratio,
                rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
                abs_tol=1.0e-12,
            ):
                return _clone_station(
                    upper_station,
                    name=f"Subsonic A/At={ratio:g}",
                    kind="area_ratio",
                    branch="subsonic",
                    requested_area_ratio=ratio,
                )

            upper = upper_station.pressure * (1.0 - 1.0e-9)
            if residual(upper) < 0.0:
                raise ThermoPropStateError(
                    f"Subsonic area ratio {ratio:g} is outside the physical chamber-to-throat branch."
                )
        else:
            upper = throat.pressure * (1.0 - 1.0e-9)
            lower = self._find_low_supersonic_pressure(model, upper, residual)

        pressure = brentq(
            residual,
            lower,
            upper,
            xtol=self._root_xtol(self.reference_pressure),
            rtol=_ROOT_RELATIVE_TOLERANCE,
            maxiter=120,
        )
        thermo = model.state(pressure)
        station = self._make_station(
            name=f"{branch.title()} A/At={ratio:g}",
            thermo=thermo,
            kind="area_ratio",
            branch=branch,
            chemistry=model.chemistry,
            frozen_from=model.frozen_from,
            requested_area_ratio=ratio,
        )
        station.area_ratio = throat.mass_flux / station.mass_flux

        if not math.isclose(
            station.area_ratio,
            ratio,
            rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
            abs_tol=2.0e-8,
        ):
            raise ThermoPropStateError(
                f"Area-ratio solve requested {ratio:g} but obtained {station.area_ratio:g}."
            )
        return station

    def _solve_pressure_station(
        self,
        pressure: float,
        *,
        equilibrium_model: _ExpansionModel,
        frozen_model: _ExpansionModel | None,
    ) -> RocketStation:
        pressure = float(pressure)
        chamber_pressure = self.chamber.pressure
        throat_pressure = self.throat.pressure

        if pressure > chamber_pressure * (1.0 + _PRESSURE_MATCH_RELATIVE_TOLERANCE):
            raise ValueError(
                f"Exit pressure {pressure:g} Pa exceeds the physical chamber pressure "
                f"{chamber_pressure:g} Pa."
            )

        if math.isclose(
            pressure,
            chamber_pressure,
            rel_tol=_PRESSURE_MATCH_RELATIVE_TOLERANCE,
            abs_tol=self._root_xtol(chamber_pressure),
        ):
            return _clone_station(
                self.chamber,
                name=f"P={pressure:g} Pa",
                kind="pressure",
                branch="chamber",
                requested_pressure=pressure,
            )

        if math.isclose(
            pressure,
            throat_pressure,
            rel_tol=_PRESSURE_MATCH_RELATIVE_TOLERANCE,
            abs_tol=self._root_xtol(throat_pressure),
        ):
            return _clone_station(
                self.throat,
                name=f"P={pressure:g} Pa",
                kind="pressure",
                branch="throat",
                requested_pressure=pressure,
            )

        if (
            isinstance(self.frozen_at, float)
            and self.freeze_station is not None
            and math.isclose(
                pressure,
                self.freeze_station.pressure,
                rel_tol=_PRESSURE_MATCH_RELATIVE_TOLERANCE,
                abs_tol=self._root_xtol(self.freeze_station.pressure),
            )
        ):
            return _clone_station(
                self.freeze_station,
                name=f"P={pressure:g} Pa",
                kind="pressure",
                branch="supersonic",
                requested_pressure=pressure,
            )

        branch = "subsonic" if pressure > throat_pressure else "supersonic"
        model = equilibrium_model
        if frozen_model is not None:
            if self.frozen_at == "chamber":
                model = frozen_model
            elif self.frozen_at == "throat" and branch == "supersonic":
                model = frozen_model
            elif isinstance(self.frozen_at, float):
                freeze_station = self.freeze_station
                if freeze_station is None:
                    raise ThermoPropStateError(
                        "Numeric frozen expansion is missing its freeze station."
                    )
                if pressure < freeze_station.pressure * (
                    1.0 - _PRESSURE_MATCH_RELATIVE_TOLERANCE
                ):
                    model = frozen_model

        thermo = model.state(pressure)
        station = self._make_station(
            name=f"P={pressure:g} Pa",
            thermo=thermo,
            kind="pressure",
            branch=branch,
            chemistry=model.chemistry,
            frozen_from=model.frozen_from,
            requested_pressure=pressure,
        )
        station.area_ratio = self.throat.mass_flux / station.mass_flux
        return station

    def _iac_reference(self, pressure: float) -> tuple[Equilibrium, _ExpansionModel, RocketStation]:
        chamber_equilibrium = Equilibrium(
            self.reactants,
            mode="hp",
            pressure=pressure,
        )
        self._total_enthalpy = chamber_equilibrium.enthalpy
        self._reference_pressure = float(pressure)

        equilibrium_model = _ExpansionModel(
            reactants=self.reactants,
            entropy=chamber_equilibrium.entropy,
            guess_temperature=chamber_equilibrium.temperature,
            chemistry="equilibrium",
        )
        chamber_station = self._make_station(
            name="Chamber",
            thermo=chamber_equilibrium,
            kind="chamber",
            branch=None,
            chemistry="equilibrium",
            frozen_from=None,
            zero_velocity=True,
        )
        return chamber_equilibrium, equilibrium_model, chamber_station

    def _solve_numeric_freeze(
        self,
        equilibrium_model: _ExpansionModel,
    ) -> _ExpansionModel:
        """Solve an equilibrium supersonic freeze station and build its frozen model."""
        if not isinstance(self.frozen_at, float):
            raise ThermoPropStateError(
                "A numeric freeze station was requested without a numeric frozen_at value."
            )

        ratio = self.frozen_at
        freeze_station = self._solve_area_station(
            ratio,
            branch="supersonic",
            model=equilibrium_model,
        )
        freeze_station.name = f"Freeze A/At={ratio:g}"
        freeze_station.kind = "freeze"
        freeze_station.requested_area_ratio = None
        freeze_station.is_freeze_station = True
        self._freeze_station = freeze_station

        if not isinstance(freeze_station.thermo, Equilibrium):
            raise ThermoPropStateError(
                "A numeric freeze station must be solved from an equilibrium state."
            )

        return _ExpansionModel(
            reactants=self.reactants,
            entropy=freeze_station.entropy,
            guess_temperature=freeze_station.temperature,
            chemistry="frozen",
            freeze_state=freeze_station.thermo,
            frozen_from=f"A/At={ratio:g}",
        )

    def _solve_iac(self) -> tuple[_ExpansionModel, _ExpansionModel | None]:
        chamber_equilibrium, equilibrium_model, chamber_station = self._iac_reference(
            self.chamber_pressure
        )
        self._infinite_area_chamber = chamber_station
        self._chamber = chamber_station

        if self.frozen_at == "chamber":
            chamber_station.is_freeze_station = True
            self._freeze_station = chamber_station
            throat_model = _ExpansionModel(
                reactants=self.reactants,
                entropy=chamber_equilibrium.entropy,
                guess_temperature=chamber_equilibrium.temperature,
                chemistry="frozen",
                freeze_state=chamber_equilibrium,
                frozen_from="chamber",
            )
        else:
            throat_model = equilibrium_model

        self._throat = self._solve_throat(
            throat_model,
            upper_pressure=self.reference_pressure,
        )

        frozen_model = None
        if self.frozen_at == "chamber":
            frozen_model = throat_model
        elif self.frozen_at == "throat":
            if not isinstance(self.throat.thermo, Equilibrium):
                raise ThermoPropStateError("Throat freeze requires an equilibrium throat state.")
            self.throat.is_freeze_station = True
            self._freeze_station = self.throat
            frozen_model = _ExpansionModel(
                reactants=self.reactants,
                entropy=self.throat.entropy,
                guess_temperature=self.throat.temperature,
                chemistry="frozen",
                freeze_state=self.throat.thermo,
                frozen_from="throat",
            )
        elif isinstance(self.frozen_at, float):
            frozen_model = self._solve_numeric_freeze(equilibrium_model)

        return equilibrium_model, frozen_model

    def _fac_result_at_reference_pressure(
        self,
        reference_pressure: float,
        contraction_ratio: float,
    ) -> tuple[
        tuple[Equilibrium, _ExpansionModel, RocketStation, RocketStation, float],
        RocketStation,
    ]:
        chamber_equilibrium, equilibrium_model, infinite_station = self._iac_reference(
            reference_pressure
        )
        throat = self._solve_throat(
            equilibrium_model,
            upper_pressure=reference_pressure,
        )

        # Temporarily expose the throat and an infinite-area upper station so the
        # generic subsonic area solver can be reused.  The physical FAC chamber is
        # replaced immediately after this calculation.
        old_chamber = self._chamber
        old_throat = self._throat
        self._chamber = infinite_station
        self._throat = throat
        chamber_end = self._solve_area_station(
            contraction_ratio,
            branch="subsonic",
            model=equilibrium_model,
            allow_fac_chamber_match=False,
        )
        self._chamber = old_chamber
        self._throat = old_throat

        chamber_end.name = "Combustor End"
        chamber_end.kind = "chamber"
        chamber_end.requested_area_ratio = None
        chamber_end.area_ratio = float(contraction_ratio)

        momentum_pressure = (
            chamber_end.pressure
            + chamber_end.density * chamber_end.velocity * chamber_end.velocity
        )
        return (
            chamber_equilibrium,
            equilibrium_model,
            infinite_station,
            throat,
            momentum_pressure,
        ), chamber_end

    def _solve_fac(self) -> tuple[_ExpansionModel, _ExpansionModel | None]:
        assigned_injector_pressure = self.chamber_pressure
        contraction_ratio = self.contraction_ratio
        injector_equilibrium = Equilibrium(
            self.reactants,
            mode="hp",
            pressure=assigned_injector_pressure,
        )
        injector_total_enthalpy = injector_equilibrium.enthalpy

        def residual(reference_pressure: float) -> float:
            self._total_enthalpy = injector_total_enthalpy
            result, _ = self._fac_result_at_reference_pressure(
                reference_pressure,
                contraction_ratio,
            )
            return result[-1] - assigned_injector_pressure

        upper = assigned_injector_pressure * (1.0 - 1.0e-10)
        lower = assigned_injector_pressure * 0.5
        for _ in range(50):
            if residual(lower) < 0.0:
                break
            lower *= 0.5
        else:
            raise ThermoPropStateError(
                "Could not bracket the finite-area combustor reference pressure."
            )

        reference_pressure = brentq(
            residual,
            lower,
            upper,
            xtol=self._root_xtol(assigned_injector_pressure),
            rtol=_ROOT_RELATIVE_TOLERANCE,
            maxiter=80,
        )

        self._total_enthalpy = injector_total_enthalpy
        result, chamber_end = self._fac_result_at_reference_pressure(
            reference_pressure,
            contraction_ratio,
        )
        (
            infinite_equilibrium,
            equilibrium_model,
            infinite_station,
            equilibrium_throat,
            momentum_pressure,
        ) = result

        if not math.isclose(
            momentum_pressure,
            assigned_injector_pressure,
            rel_tol=2.0e-8,
            abs_tol=self._root_xtol(assigned_injector_pressure),
        ):
            raise ThermoPropStateError(
                "Finite-area combustor momentum closure did not converge."
            )

        self._reference_pressure = reference_pressure
        self._injector = self._make_station(
            name="Injector",
            thermo=injector_equilibrium,
            kind="injector",
            branch=None,
            chemistry="equilibrium",
            frozen_from=None,
            zero_velocity=True,
        )
        self._infinite_area_chamber = infinite_station
        self._infinite_area_chamber.name = "Infinite-Area Reference"
        self._infinite_area_chamber.kind = "reference"
        self._chamber = chamber_end

        if self.frozen_at == "chamber":
            if not isinstance(chamber_end.thermo, Equilibrium):
                raise ThermoPropStateError("FAC chamber freeze requires equilibrium chamber data.")
            throat_model = _ExpansionModel(
                reactants=self.reactants,
                entropy=chamber_end.entropy,
                guess_temperature=chamber_end.temperature,
                chemistry="frozen",
                freeze_state=chamber_end.thermo,
                frozen_from="chamber",
            )
            self._throat = self._solve_throat(
                throat_model,
                upper_pressure=chamber_end.pressure,
            )
        else:
            self._throat = equilibrium_throat
            throat_model = equilibrium_model

        frozen_model = None
        if self.frozen_at == "chamber":
            frozen_model = throat_model
        elif self.frozen_at == "throat":
            if not isinstance(self.throat.thermo, Equilibrium):
                raise ThermoPropStateError("Throat freeze requires an equilibrium throat state.")
            self.throat.is_freeze_station = True
            self._freeze_station = self.throat
            frozen_model = _ExpansionModel(
                reactants=self.reactants,
                entropy=self.throat.entropy,
                guess_temperature=self.throat.temperature,
                chemistry="frozen",
                freeze_state=self.throat.thermo,
                frozen_from="throat",
            )
        elif isinstance(self.frozen_at, float):
            frozen_model = self._solve_numeric_freeze(equilibrium_model)

        return equilibrium_model, frozen_model

    def solve(self):
        """Rebuild reactant aggregates and solve the complete rocket problem."""
        self._validate_inputs()
        self.reactants.update()

        self._injector = None
        self._infinite_area_chamber = None
        self._chamber = None
        self._throat = None
        self._freeze_station = None
        self._pressure_stations = []
        self._subsonic_stations = []
        self._supersonic_stations = []
        self._characteristic_velocity = None
        self._reference_pressure = None
        self._total_enthalpy = None

        if self.is_fac:
            equilibrium_model, frozen_model = self._solve_fac()
        else:
            equilibrium_model, frozen_model = self._solve_iac()

        self._characteristic_velocity = self.reference_pressure / self.throat.mass_flux

        # Characteristic velocity is a property of the complete rocket problem;
        # attach it to every station after the choked mass flux is available.
        for station in [
            self._injector,
            self._infinite_area_chamber,
            self._chamber,
            self._throat,
            self._freeze_station,
        ]:
            if station is not None:
                station._set_characteristic_velocity(self.characteristic_velocity)

        for pressure in self._exit_pressures:
            station = self._solve_pressure_station(
                pressure,
                equilibrium_model=equilibrium_model,
                frozen_model=frozen_model,
            )
            station._set_characteristic_velocity(self.characteristic_velocity)
            if isinstance(self.frozen_at, float) and station.is_freeze_station:
                self._freeze_station = station
            self._pressure_stations.append(station)

        subsonic_model = equilibrium_model
        if self.frozen_at == "chamber" and frozen_model is not None:
            subsonic_model = frozen_model

        for ratio in self._subsonic_area_ratios:
            station = self._solve_area_station(
                ratio,
                branch="subsonic",
                model=subsonic_model,
            )
            station._set_characteristic_velocity(self.characteristic_velocity)
            self._subsonic_stations.append(station)

        for ratio in self._supersonic_area_ratios:
            if (
                isinstance(self.frozen_at, float)
                and self.freeze_station is not None
                and math.isclose(
                    ratio,
                    self.frozen_at,
                    rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
                    abs_tol=2.0e-8,
                )
            ):
                station = _clone_station(
                    self.freeze_station,
                    name=f"Supersonic A/At={ratio:g}",
                    kind="area_ratio",
                    branch="supersonic",
                    requested_area_ratio=ratio,
                )
                self._freeze_station = station
            else:
                supersonic_model = equilibrium_model
                if frozen_model is not None:
                    if self.frozen_at in {"chamber", "throat"}:
                        supersonic_model = frozen_model
                    elif isinstance(self.frozen_at, float) and ratio > self.frozen_at:
                        supersonic_model = frozen_model

                station = self._solve_area_station(
                    ratio,
                    branch="supersonic",
                    model=supersonic_model,
                )
            station._set_characteristic_velocity(self.characteristic_velocity)
            self._supersonic_stations.append(station)

        self._dirty = False
        return self

    def update(
        self,
        reactants=UNSET,
        chamber_pressure=UNSET,
        *,
        exit_pressures=UNSET,
        subsonic_area_ratios=UNSET,
        supersonic_area_ratios=UNSET,
        frozen_at=UNSET,
        contraction_ratio=UNSET,
        solve: bool = True,
    ):
        """Update public rocket inputs and optionally solve once at the end."""
        if is_provided(reactants):
            if not isinstance(reactants, Reactants):
                raise TypeError("reactants must be a ThermoProp Reactants object.")
            self._reactants = reactants

        if is_provided(chamber_pressure):
            self._chamber_pressure = float(chamber_pressure)

        if is_provided(exit_pressures):
            self._exit_pressures = _as_values(exit_pressures, "exit_pressures")

        if is_provided(subsonic_area_ratios):
            self._subsonic_area_ratios = _as_values(
                subsonic_area_ratios,
                "subsonic_area_ratios",
            )

        if is_provided(supersonic_area_ratios):
            self._supersonic_area_ratios = _as_values(
                supersonic_area_ratios,
                "supersonic_area_ratios",
            )

        if is_provided(frozen_at):
            self._frozen_at = _normalize_frozen_at(frozen_at)

        if is_provided(contraction_ratio):
            self._contraction_ratio = (
                None if contraction_ratio is None else float(contraction_ratio)
            )

        self._validate_inputs()
        self._dirty = True
        if solve:
            return self.solve()
        return self

    def at_pressure(self, pressure: float) -> RocketStation:
        """Return a requested pressure station by absolute pressure in Pa."""
        pressure = float(pressure)
        for station in self._pressure_stations:
            if math.isclose(
                station.requested_pressure,
                pressure,
                rel_tol=_PRESSURE_MATCH_RELATIVE_TOLERANCE,
                abs_tol=self._root_xtol(pressure),
            ):
                return station
        raise KeyError(f"No requested rocket station has pressure {pressure:g} Pa.")

    def at_area_ratio(self, area_ratio: float, *, branch: str = "supersonic") -> RocketStation:
        """Return an area-ratio station on the selected branch.

        In addition to user-requested stations, this returns an automatically
        solved numeric freeze station when its ``A/At`` matches ``area_ratio``.
        """
        branch_key = str(branch).strip().lower()
        if branch_key in {"sup", "super", "supersonic"}:
            stations = self._supersonic_stations
            branch_key = "supersonic"
        elif branch_key in {"sub", "subsonic"}:
            stations = self._subsonic_stations
            branch_key = "subsonic"
        else:
            raise ValueError("branch must be 'subsonic' or 'supersonic'.")

        area_ratio = float(area_ratio)
        if (
            branch_key == "supersonic"
            and self.freeze_station is not None
            and self.freeze_station.area_ratio is not None
            and math.isclose(
                self.freeze_station.area_ratio,
                area_ratio,
                rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
                abs_tol=2.0e-8,
            )
        ):
            return self.freeze_station

        for station in stations:
            if math.isclose(
                station.requested_area_ratio,
                area_ratio,
                rel_tol=_AREA_MATCH_RELATIVE_TOLERANCE,
                abs_tol=2.0e-8,
            ):
                return station
        raise KeyError(
            f"No requested {branch_key} rocket station has A/At={area_ratio:g}."
        )

    def as_dict(self, trace: float = 1.0e-12) -> dict[str, Any]:
        """Return all rocket inputs, major results, and station dictionaries."""
        return {
            "model": self.model,
            "reactants": {
                "mixture_ratio": self.reactants.mixture_ratio,
                "mass_fractions": self.reactants.mass_fractions,
                "mole_fractions": self.reactants.mole_fractions,
            },
            "assigned_pressure": self.chamber_pressure,
            "contraction_ratio": self.contraction_ratio,
            "frozen_at": self.frozen_at,
            "freeze_station": (
                None
                if self.freeze_station is None
                else self.freeze_station.as_dict(trace=trace)
            ),
            "reference_pressure": self.reference_pressure,
            "characteristic_velocity": self.characteristic_velocity,
            "throat_mass_flux": self.throat_mass_flux,
            "stations": [station.as_dict(trace=trace) for station in self.stations],
        }

    @staticmethod
    def _format_group(values: dict[str, float]) -> str:
        if not values:
            return "None"
        return ", ".join(
            f"{name} ({100.0 * fraction:.4g}%)"
            for name, fraction in values.items()
        )

    @staticmethod
    def _table(stations: list[RocketStation], rows: list[tuple[str, Any]]) -> str:
        if not stations:
            return ""
        label_width = max(29, max(len(label) for label, _ in rows))
        column_width = max(15, max(len(station.label) + 2 for station in stations))
        header = " " * label_width + "".join(
            f"{station.label:>{column_width}}" for station in stations
        )
        line = "-" * len(header)
        body = [header, line]
        for label, getter in rows:
            values = []
            for station in stations:
                try:
                    value = getter(station)
                except Exception:
                    value = None
                values.append(f"{format_optional(value, '.6g'):>{column_width}}")
            body.append(f"{label:<{label_width}}" + "".join(values))
        return "\n".join(body)

    def report(
        self,
        *,
        fractions: str = "mole",
        trace: float = 1.0e-5,
        max_species: int | None = 30,
        include_composition: bool = True,
    ) -> str:
        """Return a full human-readable CEA-style rocket report.

        ``fractions`` may be ``"mole"`` or ``"mass"``.  Species below ``trace``
        at every reported station are omitted.  ``max_species`` limits the rows
        after sorting by the largest fraction found at any station.
        """
        fraction_key = str(fractions).strip().lower()
        if fraction_key not in {"mole", "mass"}:
            raise ValueError("fractions must be 'mole' or 'mass'.")
        trace = float(trace)
        if trace < 0.0:
            raise ValueError("trace must be nonnegative.")

        model_description = (
            "Finite-area combustor"
            if self.is_fac
            else "Infinite-area combustion chamber"
        )
        chemistry_description = (
            "Equilibrium"
            if self.frozen_at is None
            else (
                f"Frozen at supersonic A/At={self.frozen_at:g}"
                if isinstance(self.frozen_at, float)
                else f"Frozen at {self.frozen_at}"
            )
        )

        lines = [
            "THERMOPROP ROCKET PERFORMANCE",
            "=" * 29,
            "",
            "REACTANTS",
            "---------",
            format_rows(
                [
                    ("Mixture ratio O/F", format_optional(self.reactants.mixture_ratio, ".8g")),
                    ("Fuels", self._format_group(self.reactants.fuel_mass_fractions)),
                    ("Oxidizers", self._format_group(self.reactants.oxidizer_mass_fractions)),
                    ("Inerts", self._format_group(self.reactants.inert_mass_fractions)),
                    ("Igniters", self._format_group(self.reactants.igniter_mass_fractions)),
                ]
            ),
            "",
            "ROCKET PROBLEM",
            "--------------",
            format_rows(
                [
                    ("Model", model_description),
                    ("Assigned pressure [Pa]", format_optional(self.chamber_pressure, ".9g")),
                    ("Contraction ratio Ac/At", format_optional(self.contraction_ratio, ".8g")),
                    ("Expansion chemistry", chemistry_description),
                    ("C* reference pressure [Pa]", format_optional(self.reference_pressure, ".9g")),
                    ("Characteristic velocity [m/s]", format_optional(self.cstar, ".9g")),
                    ("Throat mass flux [kg/m^2-s]", format_optional(self.throat_mass_flux, ".9g")),
                ]
            ),
            "",
            "THERMODYNAMIC AND FLOW PROPERTIES",
            "---------------------------------",
            self._table(
                self.stations,
                [
                    ("Chemistry", lambda s: s.chemistry),
                    ("Frozen from", lambda s: s.frozen_from),
                    ("Pressure [Pa]", lambda s: s.pressure),
                    ("Temperature [K]", lambda s: s.temperature),
                    ("Density [kg/m^3]", lambda s: s.density),
                    ("Enthalpy [J/kg]", lambda s: s.enthalpy),
                    ("Entropy [J/kg-K]", lambda s: s.entropy),
                    ("Molecular weight [kg/kmol]", lambda s: s.molecular_weight),
                    ("Specific heat ratio", lambda s: s.specific_heat_ratio),
                    ("Speed of sound [m/s]", lambda s: s.speed_of_sound),
                    ("Velocity [m/s]", lambda s: s.velocity),
                    ("Mach number", lambda s: s.mach),
                    ("Mass flux [kg/m^2-s]", lambda s: s.mass_flux),
                    ("Pressure ratio Pinf/P", lambda s: s.pressure_ratio),
                    ("Area ratio A/At", lambda s: s.area_ratio),
                ],
            ),
        ]

        if self.exits:
            lines.extend(
                [
                    "",
                    "PERFORMANCE PARAMETERS",
                    "----------------------",
                    self._table(
                        self.exits,
                        [
                            ("Thrust coefficient", lambda s: s.cf),
                            ("Vacuum thrust coefficient", lambda s: s.cf_vac),
                            ("Specific impulse [s]", lambda s: s.isp),
                            ("Vacuum specific impulse [s]", lambda s: s.isp_vac),
                        ],
                    ),
                ]
            )

        if include_composition:
            fraction_property = (
                "mole_fractions" if fraction_key == "mole" else "mass_fractions"
            )
            compositions: list[dict[str, float]] = [
                dict(getattr(station, fraction_property)) for station in self.stations
            ]
            species = set().union(*(composition.keys() for composition in compositions))
            ranked = sorted(
                species,
                key=lambda name: max(
                    composition.get(name, 0.0) for composition in compositions
                ),
                reverse=True,
            )
            ranked = [
                name
                for name in ranked
                if max(composition.get(name, 0.0) for composition in compositions) >= trace
            ]
            if max_species is not None:
                ranked = ranked[: int(max_species)]

            if ranked:
                rows = [
                    (
                        species_name,
                        lambda station, name=species_name: getattr(
                            station,
                            fraction_property,
                        ).get(name, 0.0),
                    )
                    for species_name in ranked
                ]
                lines.extend(
                    [
                        "",
                        f"{fraction_key.upper()} FRACTIONS",
                        "-" * (len(fraction_key) + 10),
                        self._table(self.stations, rows),
                    ]
                )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        return (
            f"Rocket(model={self.model!r}, chamber_pressure={self.chamber_pressure:.6g}, "
            f"contraction_ratio={self.contraction_ratio!r}, frozen_at={self.frozen_at!r}, "
            f"stations={len(self.stations)})"
        )
