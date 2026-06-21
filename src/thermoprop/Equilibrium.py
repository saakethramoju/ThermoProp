"""
Equilibrium.py

Public API wrapper for the modular CEA-style equilibrium package.

Directory layout expected:

src/thermoprop/
    Equilibrium.py
    CEADatabase.py
    Reactants.py
    CombustionGas.py
    CEAEquilibrium/
        state.py
        species.py
        thermo.py
        matrix.py
        tp_solver.py
        hp_solver.py
        condensed.py
        properties.py
        transport.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .Reactants import Reactants
from .CombustionGas import CombustionGas
from ._api import PropertyIntrospectionMixin
from ._formatting import format_optional, rounded_dict, format_rows
from ._state_api import UNSET, is_provided, provided_items


from .CEAEquilibrium.state import FeedState, EquilibriumState, EquilibriumResults
from .CEAEquilibrium.facade import (
    EquilibriumConfig,
    EquilibriumSolveSummary,
    resolve_reactants,
    run_equilibrium_solve,
    solver_options,
)
from .CEAEquilibrium.tp_solver import solve_tp
from .CEAEquilibrium.properties import (
    build_results,
    enthalpy as state_enthalpy,
    entropy as state_entropy,
    internal_energy as state_internal_energy,
    gibbs_energy as state_gibbs_energy,
    helmholtz_energy as state_helmholtz_energy,
    density as state_density,
    specific_volume as state_specific_volume,
    gas_constant as state_gas_constant,
    molecular_weight as state_molecular_weight,
    molecular_weight_all_species,
    mole_fractions as state_mole_fractions,
    gas_mole_fractions as state_gas_mole_fractions,
    mass_fractions as state_mass_fractions,
    gas_mass_fractions as state_gas_mass_fractions,
    speed_of_sound_frozen,
    speed_of_sound_equilibrium,
    frozen_mixture_derivatives,
)


class Equilibrium(PropertyIntrospectionMixin):
    """
    Chemical equilibrium solver using Gibbs free-energy minimization.

    The Equilibrium class computes the thermodynamic equilibrium composition
    of a reacting mixture at a specified pressure and thermodynamic state.
    Supported modes are:

    * TP (constant temperature, constant pressure)
    * HP (constant enthalpy, constant pressure)

    The solver minimizes the total Gibbs free energy subject to elemental
    conservation constraints and returns an equilibrium composition that
    can be used directly by CombustionGas, nozzle calculations, transport
    property calculations, and rocket performance analyses.

    Notes
    -----
    CEA-style heat capacities
    =========================

    NASA CEA reports three different heat capacities that are often confused:

    1. Thermodynamic equilibrium Cp
    --------------------------------

    Reported in the main thermodynamic properties table.

    This is the true equilibrium heat capacity:

        Cp = (∂h/∂T)_P

    where chemical equilibrium is maintained during the temperature change.
    As temperature increases, species are allowed to dissociate, recombine,
    condense, or vaporize as required by equilibrium.

    Therefore this quantity contains both:

        * sensible heating effects
        * chemical reaction effects

    and is typically the largest Cp reported by CEA.

    This is the Cp normally used in equilibrium thermodynamic calculations
    and corresponds to the derivative of equilibrium enthalpy with respect
    to temperature.

    2. Transport equilibrium Cp
    ----------------------------

    Reported in the TRANSPORT PROPERTIES section under:

        WITH EQUILIBRIUM REACTIONS

    Transport properties (viscosity, conductivity, Prandtl number, etc.)
    are only defined for the gas phase.

    CEA therefore removes condensed species and computes properties using
    only the gas mixture. Equilibrium chemistry is still allowed, so gas
    composition may change with temperature.

    Consequently:

        Cp_transport_eq < Cp_thermodynamic_eq

    whenever condensed species are present.

    This value should be used when computing gas-phase transport properties
    such as thermal conductivity and Prandtl number.

    3. Transport frozen Cp
    -----------------------

    Reported in the TRANSPORT PROPERTIES section under:

        WITH FROZEN REACTIONS

    The gas composition is held fixed and no chemical re-equilibration is
    allowed.

    In this case:

        Cp_frozen = Σ Yi * Cpi

    for the fixed gas composition.

    Because reaction energy effects are excluded:

        Cp_transport_frozen <= Cp_transport_eq

    Typical ordering is:

        Cp_transport_frozen
            <= Cp_transport_eq
            <= Cp_thermodynamic_eq

    Condensed species
    =================

    The equilibrium composition may contain condensed species (graphite,
    liquid metals, condensed oxides, etc.).

    Thermodynamic properties such as:

        h, u, s, g, Cp

    are generally evaluated using the complete equilibrium mixture.

    Transport properties such as:

        μ, k, Pr

    should generally be evaluated using only the gas-phase species, matching
    NASA CEA transport-property conventions.

    References
    ----------
    Gordon, S., and McBride, B. J.,
    "Computer Program for Calculation of Complex Chemical Equilibrium
    Compositions and Applications", NASA RP-1311.

    McBride, B. J., and Gordon, S.,
    NASA CEA Users Manual.
    """
    _BACKEND_NAME = "ThermoProp CEA-style Equilibrium"

    def __init__(
        self,
        reactants: Reactants | CombustionGas | dict[str, float],
        *,
        mode: str = "hp",
        temperature: float | None = None,
        pressure: float | None = None,
        basis: str = "mass",
        guess_temperature: float = 3800.0,
        candidates: list[str] | None = None,
        include_condensed: bool = True,
        include_ions: bool = False,
        include_electron: bool = False,
        combustion_gas_trace: float = 1e-12,
        combustion_gas_max_species: int | None = None,
        max_iterations: int = 120,
        max_outer_iterations: int = 30,
        verbose: bool = False,
        equilibrium_derivative_temperature_step: float = 1.0,
    ):
        self._input = reactants
        self._mode = mode.lower().strip()
        self._temperature_input = None if temperature is None else float(temperature)
        self._pressure = None if pressure is None else float(pressure)
        self._basis = basis
        self._guess_temperature = float(guess_temperature)
        self._candidates = candidates

        self._include_condensed = bool(include_condensed)
        self._include_ions = bool(include_ions)
        self._include_electron = bool(include_electron)

        self._combustion_gas_trace = float(combustion_gas_trace)
        self._combustion_gas_max_species = combustion_gas_max_species
        self._max_iterations = int(max_iterations)
        self._max_outer_iterations = int(max_outer_iterations)
        self._verbose = bool(verbose)
        self._equilibrium_derivative_temperature_step = float(
            equilibrium_derivative_temperature_step
        )

        self._reactants = resolve_reactants(
            reactants,
            basis=basis,
            temperature=temperature,
            pressure=pressure,
        )

        self._feed: FeedState | None = None
        self._solve_result = None
        self._state: EquilibriumState | None = None
        self._results: EquilibriumResults | None = None
        self._summary: EquilibriumSolveSummary | None = None
        self._gas_cache: CombustionGas | None = None
        self._cea_extended_range_hp_warning: bool = False
        self._dirty: bool = True

        self.solve()

    def _solver_config(self) -> EquilibriumConfig:
        """Collect current public inputs into an immutable solver config."""
        return EquilibriumConfig(
            mode=self._mode,
            pressure=self._pressure,
            temperature_input=self._temperature_input,
            basis=self._basis,
            guess_temperature=self._guess_temperature,
            candidates=self._candidates,
            include_condensed=self._include_condensed,
            include_ions=self._include_ions,
            include_electron=self._include_electron,
            combustion_gas_trace=self._combustion_gas_trace,
            combustion_gas_max_species=self._combustion_gas_max_species,
            max_iterations=self._max_iterations,
            max_outer_iterations=self._max_outer_iterations,
            verbose=self._verbose,
            equilibrium_derivative_temperature_step=self._equilibrium_derivative_temperature_step,
        )

    def _solver_options(self):
        """Backward-compatible internal hook used by neighbor TP solves."""
        return solver_options(self._solver_config())

    def _solve(self) -> None:
        self._gas_cache = None

        run = run_equilibrium_solve(
            config=self._solver_config(),
            original_input=self._input,
            reactants=self._reactants,
            tp_neighbor_solver=self._tp_neighbor_state,
        )

        self._feed = run.feed
        self._solve_result = run.solve_result
        self._state = run.state
        self._results = run.results
        self._summary = run.summary
        self._cea_extended_range_hp_warning = run.cea_extended_range_hp_warning

    def solve(self):
        """Solve or re-solve the current equilibrium state in place.

        This method is intentionally explicit so iterative callers can batch
        input changes with ``update(..., solve=False)`` and solve exactly once.
        The constructor and property setters still solve immediately for backward
        compatibility with ThermoProp 1.0.x usage.
        """

        self._reactants = resolve_reactants(
            self._input,
            basis=self._basis,
            temperature=self._temperature_input,
            pressure=self._pressure,
        )
        self._solve()
        self._dirty = False
        return self

    def update(
        self,
        reactants=UNSET,
        *,
        mode=UNSET,
        temperature=UNSET,
        pressure=UNSET,
        basis=UNSET,
        guess_temperature=UNSET,
        candidates=UNSET,
        include_condensed=UNSET,
        include_ions=UNSET,
        include_electron=UNSET,
        combustion_gas_trace=UNSET,
        combustion_gas_max_species=UNSET,
        max_iterations=UNSET,
        max_outer_iterations=UNSET,
        verbose=UNSET,
        equilibrium_derivative_temperature_step=UNSET,
        solve: bool = True,
    ):
        """Update equilibrium inputs and optionally re-solve in place.

        Parameters are the same public inputs accepted by the constructor.
        Use ``solve=False`` inside residual loops to change several inputs and
        call :meth:`solve` yourself when the state is ready.  With the default
        ``solve=True`` this method preserves the old immediate-update behavior.
        """

        if is_provided(reactants):
            self._input = reactants

        if is_provided(mode):
            self._mode = str(mode).lower().strip()

        if is_provided(temperature):
            if self._mode == "tp":
                self._temperature_input = None if temperature is None else float(temperature)
            else:
                if temperature is not None:
                    self._guess_temperature = float(temperature)

        if is_provided(pressure):
            self._pressure = None if pressure is None else float(pressure)

        if is_provided(basis):
            self._basis = basis

        if is_provided(guess_temperature):
            self._guess_temperature = float(guess_temperature)

        if is_provided(candidates):
            self._candidates = candidates

        if is_provided(include_condensed):
            self._include_condensed = bool(include_condensed)

        if is_provided(include_ions):
            self._include_ions = bool(include_ions)

        if is_provided(include_electron):
            self._include_electron = bool(include_electron)

        if is_provided(combustion_gas_trace):
            self._combustion_gas_trace = float(combustion_gas_trace)

        if is_provided(combustion_gas_max_species):
            self._combustion_gas_max_species = combustion_gas_max_species

        if is_provided(max_iterations):
            self._max_iterations = int(max_iterations)

        if is_provided(max_outer_iterations):
            self._max_outer_iterations = int(max_outer_iterations)

        if is_provided(verbose):
            self._verbose = bool(verbose)

        if is_provided(equilibrium_derivative_temperature_step):
            self._equilibrium_derivative_temperature_step = float(
                equilibrium_derivative_temperature_step
            )

        self._reactants = resolve_reactants(
            self._input,
            basis=self._basis,
            temperature=self._temperature_input,
            pressure=self._pressure,
        )
        self._gas_cache = None
        self._dirty = True

        if solve:
            return self.solve()

        return self

    @property
    def is_stale(self) -> bool:
        """Whether inputs have changed since the last completed solve."""

        return self._dirty

    def _tp_neighbor_state(
        self,
        base_state: EquilibriumState,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> EquilibriumState:
        """
        Fast TP neighbor solve using the same active species set.
        Used for equilibrium derivative properties.
        """
        T = base_state.temperature if temperature is None else float(temperature)
        P = base_state.pressure if pressure is None else float(pressure)

        state = base_state.copy()
        state.temperature = T
        state.pressure = P
        state.converged = False

        tp_options, _, _, _ = self._solver_options()
        tp_options.verbose = False

        result = solve_tp(state, options=tp_options)

        if not result.success:
            raise RuntimeError(f"Neighbor TP solve failed: {result.message}")

        return result.state


    @staticmethod
    def _object_cache_key(value) -> tuple:
        cache_key = getattr(value, "cache_key", None)

        if callable(cache_key):
            try:
                return ("cache_key", cache_key())
            except Exception:
                pass

        return ("object", type(value).__name__, id(value))

    def cache_key(self) -> tuple:
        """Stable state fingerprint for FullFlow ``Lookup`` caching."""

        try:
            species_state = tuple(
                sorted(
                    (name, round(float(n), 18))
                    for name, n in self.species_moles.items()
                    if abs(float(n)) > self._combustion_gas_trace
                )
            )
        except Exception:
            species_state = ()

        return (
            "Equilibrium",
            self._object_cache_key(self._input),
            self._mode,
            None if self._pressure is None else round(float(self._pressure), 12),
            None if self._temperature_input is None else round(float(self._temperature_input), 12),
            self._basis,
            None if self._candidates is None else tuple(self._candidates),
            self._include_condensed,
            self._include_ions,
            self._include_electron,
            round(float(self._combustion_gas_trace), 30),
            self._combustion_gas_max_species,
            species_state,
        )

    @property
    def backend(self) -> str:
        return self._BACKEND_NAME

    @property
    def name(self) -> str:
        return "Equilibrium combustion products"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def input(self):
        return self._input

    @property
    def reactants(self):
        return self._reactants

    @property
    def feed(self) -> FeedState:
        return self._feed

    @property
    def state(self) -> EquilibriumState:
        return self._state

    @property
    def results(self) -> EquilibriumResults:
        return self._results

    @property
    def summary(self) -> EquilibriumSolveSummary:
        return self._summary

    @property
    def success(self) -> bool:
        return self._summary.success

    @property
    def message(self) -> str:
        return self._summary.message

    @property
    def iterations(self) -> int:
        return self._summary.iterations

    @property
    def outer_iterations(self) -> int:
        return self._summary.outer_iterations

    @property
    def inserted_condensed_species(self) -> list[str]:
        return list(self._summary.inserted_condensed_species)

    @property
    def removed_condensed_species(self) -> list[str]:
        return list(self._summary.removed_condensed_species)

    @property
    def pressure(self) -> float:
        return self._state.pressure

    @pressure.setter
    def pressure(self, value: float) -> None:
        self.update(pressure=value, solve=True)

    @property
    def temperature(self) -> float:
        return self._state.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.update(temperature=value, solve=True)

    @property
    def pressure_temperature(self) -> tuple[float, float]:
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: tuple[float, float]) -> None:
        self.update(pressure=values[0], temperature=values[1], mode="tp", solve=True)

    @property
    def TP(self) -> tuple[float, float]:
        return self.temperature, self.pressure

    @TP.setter
    def TP(self, values: tuple[float, float]) -> None:
        self.update(temperature=values[0], pressure=values[1], mode="tp", solve=True)

    @property
    def HP(self) -> tuple[float, float]:
        return self.enthalpy, self.pressure

    @HP.setter
    def HP(self, values: tuple[float, float]) -> None:
        raise ValueError(
            "HP equilibrium enthalpy is fixed by the reactants. "
            "Change reactants or pressure instead."
        )

    @property
    def species(self) -> list[str]:
        return list(self._state.species.names)

    @property
    def elements(self) -> list[str]:
        return list(self._state.species.elements)

    @property
    def gas_species(self) -> list[str]:
        return [
            name for name, mask in zip(self.species, self._state.species.gas_mask)
            if mask
        ]

    @property
    def condensed_species(self) -> list[str]:
        return [
            name for name, mask in zip(self.species, self._state.species.condensed_mask)
            if mask
        ]

    @property
    def ion_species(self) -> list[str]:
        return [
            name for name, mask in zip(self.species, self._state.species.ion_mask)
            if mask
        ]

    @property
    def species_moles(self) -> dict[str, float]:
        return {
            name: float(n)
            for name, n in zip(self._state.species.names, self._state.n)
        }
        
    @property
    def species_moles_trace(self) -> dict[str, float]:
        return {
            name: float(n)
            for name, n in zip(self._state.species.names, self._state.n)
            if float(n) > self._combustion_gas_trace
        }

    @property
    def moles(self) -> dict[str, float]:
        return self.species_moles

    @property
    def total_gas_moles(self) -> float:
        return float(self._state.total_gas_moles)

    @property
    def mole_fractions(self) -> dict[str, float]:
        return state_mole_fractions(
            self._state,
            trace=self._combustion_gas_trace,
        )
    @property
    def gas_mole_fractions(self) -> dict[str, float]:
        return state_gas_mole_fractions(
            self._state,
            trace=self._combustion_gas_trace,
        )
    
    @property
    def mass_fractions(self) -> dict[str, float]:
        return state_mass_fractions(
            self._state,
            trace=self._combustion_gas_trace,
        )

    @property
    def composition(self) -> dict[str, float]:
        """Full equilibrium mass-fraction composition with strict CEA names.

        This includes gas and condensed species exactly as CEA represents them.
        Phase-specific entries such as ``H2O`` and ``H2O(L)`` remain separate.
        Use ``gas_composition`` or ``combustion_gas.composition`` when a
        normalized gas-only composition is needed for gas-property chaining.
        """
        return self.mass_fractions

    @property
    def gas_mass_fractions(self) -> dict[str, float]:
        return state_gas_mass_fractions(
            self._state,
            trace=self._combustion_gas_trace,
        )

    @property
    def gas_composition(self) -> dict[str, float]:
        """Gas-only mass-fraction composition with strict CEA gas names."""
        return dict(self.combustion_gas.mass_fractions)

    @property
    def normalized_mole_fractions(self) -> dict[str, float]:
        return self.combustion_gas_composition()

    @property
    def normalized_mass_fractions(self) -> dict[str, float]:
        return dict(self.combustion_gas.mass_fractions)

    def combustion_gas_composition(
        self,
        trace: float | None = None,
        max_species: int | None = None,
    ) -> dict[str, float]:
        """Return normalized gas-only mole fractions for `CombustionGas`."""
        if trace is None:
            trace = self._combustion_gas_trace
        if max_species is None:
            max_species = self._combustion_gas_max_species

        items = [
            (name, x)
            for name, x in self.gas_mole_fractions.items()
            if x > trace
        ]
        items.sort(key=lambda item: item[1], reverse=True)

        if max_species is not None:
            items = items[: int(max_species)]

        total = sum(x for _, x in items)

        if total <= 0.0:
            gas = self.gas_mole_fractions
            name, _ = max(gas.items(), key=lambda item: item[1])
            return {name: 1.0}

        return {name: x / total for name, x in items}

    def CombustionGas_composition(
        self,
        trace: float | None = None,
        max_species: int | None = None,
    ) -> dict[str, float]:
        """Backward-compatible alias for `combustion_gas_composition`."""
        return self.combustion_gas_composition(trace=trace, max_species=max_species)

    @property
    def combustion_gas(self) -> CombustionGas:
        """Return a gas-only `CombustionGas` view of the equilibrium products."""
        if self._gas_cache is None:
            self._gas_cache = CombustionGas(
                self.combustion_gas_composition(),
                basis="mole",
                pressure=self.pressure,
                temperature=self.temperature,
            )
        return self._gas_cache

    @property
    def CombustionGas(self) -> CombustionGas:
        """Backward-compatible alias for `combustion_gas`."""
        return self.combustion_gas

    @property
    def combustiongas(self) -> CombustionGas:
        """Lowercase alias for ``combustion_gas``."""
        return self.combustion_gas

    @property
    def gas(self) -> CombustionGas:
        return self.combustion_gas

    @property
    def fluid(self) -> dict[str, float]:
        """Gas-only composition dictionary for fluid-property chaining."""
        return self.gas_composition

    @property
    def density(self) -> float:
        return self._results.density

    @density.setter
    def density(self, value: float) -> None:
        self._pressure = float(value) * self.gas_constant * self.temperature
        self._solve()

    @property
    def specific_volume(self) -> float:
        return 1.0 / self.density

    @property
    def enthalpy(self) -> float:
        return self._results.enthalpy

    @property
    def entropy(self) -> float:
        return self._results.entropy

    @property
    def internal_energy(self) -> float:
        return self._results.internal_energy

    @property
    def gibbs_energy(self) -> float:
        return state_gibbs_energy(self._state)

    @property
    def helmholtz_energy(self) -> float:
        return state_helmholtz_energy(self._state)

    @property
    def free_energy(self) -> float:
        return self.helmholtz_energy

    @property
    def gas_constant(self) -> float:
        return state_gas_constant(self._state)

    @property
    def universal_gas_constant(self) -> float:
        return 8.31446261815324

    @property
    def molecular_weight(self) -> float:
        return molecular_weight_all_species(self._state)
        
    @property
    def molecular_weight_gas(self) -> float:
        return state_molecular_weight(self._state)

    @property
    def molecular_weight_all_species(self) -> float:
        return molecular_weight_all_species(self._state)
        
    @property
    def moles_inverse(self) -> float:
        return 8314.46261815324 / self.gas_constant

    @property
    def molar_mass(self) -> float:
        return self.molecular_weight / 1000.0

    @property
    def specific_heat_cp_frozen(self) -> float:
        return self._results.cp_frozen

    @property
    def specific_heat_cv_frozen(self) -> float:
        return self._results.cv_frozen

    @property
    def specific_heat_cp_equilibrium(self) -> float:
        return self._results.cp_equilibrium

    @property
    def specific_heat_cv_equilibrium(self) -> float:
        return self._results.cv_equilibrium

    @property
    def specific_heat_cp(self) -> float:
        return self.specific_heat_cp_equilibrium

    @property
    def specific_heat_cv(self) -> float:
        return self.specific_heat_cv_equilibrium

    @property
    def specific_heat(self) -> float:
        return self.specific_heat_cp

    @property
    def cp_frozen(self) -> float:
        return self.specific_heat_cp_frozen

    @property
    def cp_equilibrium(self) -> float:
        return self.specific_heat_cp_equilibrium

    @property
    def cp_reaction(self) -> float:
        return self.cp_equilibrium - self.cp_frozen

    @property
    def cv_frozen(self) -> float:
        return self.specific_heat_cv_frozen

    @property
    def cv_equilibrium(self) -> float:
        return self.specific_heat_cv_equilibrium

    @property
    def cp_transport_frozen(self) -> float | None:
        return self._results.cp_transport_frozen

    @property
    def cp_transport_equilibrium(self) -> float | None:
        return self._results.cp_transport_equilibrium

    @property
    def gamma_frozen(self) -> float:
        return self._results.gamma_frozen

    @property
    def gamma_equilibrium(self) -> float:
        return self._results.gamma_equilibrium

    @property
    def specific_heat_ratio_frozen(self) -> float:
        return self.gamma_frozen

    @property
    def specific_heat_ratio_equilibrium(self) -> float:
        return self.gamma_equilibrium

    @property
    def specific_heat_ratio(self) -> float:
        return self.gamma_equilibrium

    @property
    def gamma(self) -> float:
        return self.gamma_equilibrium

    @property
    def dynamic_viscosity_frozen(self) -> float | None:
        return self._results.viscosity_frozen

    @property
    def dynamic_viscosity_equilibrium(self) -> float | None:
        return self._results.viscosity_equilibrium

    @property
    def dynamic_viscosity(self) -> float | None:
        return self.dynamic_viscosity_equilibrium

    @property
    def viscosity_frozen(self) -> float | None:
        return self.dynamic_viscosity_frozen

    @property
    def viscosity_equilibrium(self) -> float | None:
        return self.dynamic_viscosity_equilibrium

    @property
    def viscosity(self) -> float | None:
        return self.dynamic_viscosity

    @property
    def kinematic_viscosity(self) -> float | None:
        mu = self.dynamic_viscosity
        if mu is None:
            return None
        return mu / self.density

    @property
    def thermal_conductivity_frozen(self) -> float | None:
        return self._results.conductivity_frozen

    @property
    def thermal_conductivity_equilibrium(self) -> float | None:
        return self._results.conductivity_equilibrium

    @property
    def thermal_conductivity(self) -> float | None:
        return self.thermal_conductivity_equilibrium

    @property
    def conductivity_frozen(self) -> float | None:
        return self.thermal_conductivity_frozen

    @property
    def conductivity_equilibrium(self) -> float | None:
        return self.thermal_conductivity_equilibrium

    @property
    def conductivity(self) -> float | None:
        return self.thermal_conductivity
        
    @property
    def conductivity_reaction(self) -> float | None:
        return self._results.conductivity_reaction
        
    @property
    def thermal_conductivity_reaction(self) -> float | None:
        return self.conductivity_reaction

    @property
    def prandtl_frozen(self) -> float | None:
        return self._results.prandtl_frozen

    @property
    def prandtl_equilibrium(self) -> float | None:
        return self._results.prandtl_equilibrium

    @property
    def prandtl(self) -> float | None:
        return self.prandtl_equilibrium

    @property
    def speed_of_sound_frozen(self) -> float:
        return speed_of_sound_frozen(self._state)

    @property
    def speed_of_sound_equilibrium(self) -> float:
        return speed_of_sound_equilibrium(
            self._state,
            gamma_equilibrium=self.gamma_equilibrium,
        )

    @property
    def speed_of_sound(self) -> float:
        return self.speed_of_sound_equilibrium

    @property
    def phase(self) -> str:
        if self.condensed_species:
            return "Equilibrium Gas + Condensed"
        return "Equilibrium Gas"

    @property
    def compressibility(self) -> float:
        return 1.0

    @property
    def is_mixture(self) -> bool:
        return True

    @property
    def element_matrix(self) -> np.ndarray:
        return self._state.species.A.copy()

    @property
    def element_vector(self) -> np.ndarray:
        return self._state.element_totals.copy()

    @property
    def element_moles(self) -> np.ndarray:
        return self._state.species.A @ self._state.n

    @property
    def element_error(self) -> np.ndarray:
        return self.element_moles - self.element_vector

    @property
    def max_element_error(self) -> float:
        return float(np.max(np.abs(self.element_error)))

    @property
    def max_element_relative_error(self) -> float:
        scale = np.maximum(np.abs(self.element_vector), 1e-300)
        return float(np.max(np.abs(self.element_error / scale)))

    @property
    def enthalpy_error(self) -> float | None:
        return self._summary.enthalpy_error

    @property
    def residual_norm(self) -> float | None:
        return self._summary.residual_norm

    @property
    def thermal_expansion_coefficient(self) -> float:
        return 1.0 / self.temperature

    @property
    def isothermal_compressibility(self) -> float:
        return 1.0 / self.pressure

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        return self.combustion_gas.partial_derivative(of, with_respect_to, constant)

    @property
    def dhdT_const_p(self) -> float:
        return self.specific_heat_cp_equilibrium

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

    def as_dict(self, trace: float = 1e-12) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "outer_iterations": self.outer_iterations,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "density": self.density,
            "specific_volume": self.specific_volume,
            "enthalpy": self.enthalpy,
            "entropy": self.entropy,
            "internal_energy": self.internal_energy,
            "gibbs_energy": self.gibbs_energy,
            "helmholtz_energy": self.helmholtz_energy,
            "specific_heat_cp": self.specific_heat_cp,
            "specific_heat_cv": self.specific_heat_cv,
            "specific_heat_cp_frozen": self.specific_heat_cp_frozen,
            "specific_heat_cv_frozen": self.specific_heat_cv_frozen,
            "specific_heat_cp_equilibrium": self.specific_heat_cp_equilibrium,
            "specific_heat_cv_equilibrium": self.specific_heat_cv_equilibrium,
            "specific_heat_ratio": self.specific_heat_ratio,
            "specific_heat_ratio_frozen": self.specific_heat_ratio_frozen,
            "specific_heat_ratio_equilibrium": self.specific_heat_ratio_equilibrium,
            "gas_constant": self.gas_constant,
            "molecular_weight": self.molecular_weight,
            "molecular_weight_all_species": self.molecular_weight_all_species,
            "composition": self.composition,
            "mole_fractions": state_mole_fractions(self._state, trace=trace),
            "gas_mole_fractions": state_gas_mole_fractions(self._state, trace=trace),
            "gas_mass_fractions": state_gas_mass_fractions(self._state, trace=trace),
            "gas_composition": self.gas_composition,
            "mass_fractions": state_mass_fractions(self._state, trace=trace),
            "species_moles": {
                name: value
                for name, value in self.species_moles.items()
                if value > trace
            },
            "gas_species": self.gas_species,
            "condensed_species": self.condensed_species,
            "ion_species": self.ion_species,
            "inserted_condensed_species": self.inserted_condensed_species,
            "removed_condensed_species": self.removed_condensed_species,
            "dynamic_viscosity_frozen": self.dynamic_viscosity_frozen,
            "dynamic_viscosity_equilibrium": self.dynamic_viscosity_equilibrium,
            "thermal_conductivity_frozen": self.thermal_conductivity_frozen,
            "thermal_conductivity_reaction": self.conductivity_reaction,
            "thermal_conductivity_equilibrium": self.thermal_conductivity_equilibrium,
            "prandtl_frozen": self.prandtl_frozen,
            "prandtl_equilibrium": self.prandtl_equilibrium,
            "speed_of_sound_frozen": self.speed_of_sound_frozen,
            "speed_of_sound_equilibrium": self.speed_of_sound_equilibrium,
            "element_error": self.element_error,
            "max_element_error": self.max_element_error,
            "max_element_relative_error": self.max_element_relative_error,
            "enthalpy_error": self.enthalpy_error,
            "residual_norm": self.residual_norm,
        }

    def _safe(self, value, fmt=".6g") -> str:
        return format_optional(value, fmt, missing="None")

    def __str__(self) -> str:
        rows = [
            ("Mode", self.mode.upper()),
            ("Backend", self.backend),
            ("Success", self.success),
            ("Message", self.message),
            ("Iterations", self.iterations),
            ("Outer iterations", self.outer_iterations),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".6e")),
            ("Temperature [K]", self._safe(self.temperature, ".3f")),
            ("Density [kg/m^3]", self._safe(self.density, ".6g")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".6e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".6e")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".6e")),
            ("Cp eq [J/kg-K]", self._safe(self.specific_heat_cp_equilibrium, ".6g")),
            ("Cp frozen [J/kg-K]", self._safe(self.specific_heat_cp_frozen, ".6g")),
            ("Cp transport eq [J/kg-K]", self._safe(self.cp_transport_equilibrium, ".6g")),
            ("Cp transport frozen [J/kg-K]", self._safe(self.cp_transport_frozen, ".6g")),
            ("Cv eq [J/kg-K]", self._safe(self.specific_heat_cv_equilibrium, ".6g")),
            ("Cv frozen [J/kg-K]", self._safe(self.specific_heat_cv_frozen, ".6g")),
            ("Gamma eq", self._safe(self.gamma_equilibrium, ".6g")),
            ("Gamma frozen", self._safe(self.gamma_frozen, ".6g")),
            ("Gas constant [J/kg-K]", self._safe(self.gas_constant, ".6g")),
            ("M, (1/n) [kg/kmol]", self._safe(self.moles_inverse, ".6g")),
            ("Molecular weight [kg/kmol]", self._safe(self.molecular_weight, ".6g")),
            ("Viscosity eq [Pa*s]", self._safe(self.dynamic_viscosity_equilibrium, ".6e")),
            ("Conductivity eq [W/m-K]", self._safe(self.thermal_conductivity_equilibrium, ".6g")),
            ("Prandtl eq", self._safe(self.prandtl_equilibrium, ".6g")),
            ("Speed of sound eq [m/s]", self._safe(self.speed_of_sound_equilibrium, ".6g")),
            ("Max element error", self._safe(self.max_element_error, ".6e")),
        ]

        if self.enthalpy_error is not None:
            rows.append(("Enthalpy error [J/kg]", self._safe(self.enthalpy_error, ".6e")))

        if self.condensed_species:
            rows.append(("Condensed species", self.condensed_species))

        rows.append(("Equilibrium mole fractions", ""))

        return format_rows(rows) + "\n\n" + self._format_species_table()
    

    def _format_species_table(self, trace: float | None = None, max_species: int | None = 25) -> str:
        if trace is None:
            trace = self._combustion_gas_trace

        items = []

        for name, x in state_mole_fractions(self._state, trace=0.0).items():
            if x <= 0.0:
                continue

            if x < trace:
                continue

            if name in self.condensed_species:
                phase = "condensed"
            elif name in self.ion_species:
                phase = "ion"
            else:
                phase = "gas"

            items.append((name, phase, x))

        items.sort(key=lambda item: item[2], reverse=True)

        if max_species is not None:
            items = items[:max_species]

        if not items:
            return "Species mole fractions: none"

        name_width = max(len("Species"), max(len(name) for name, _, _ in items))
        phase_width = max(len("Phase"), max(len(phase) for _, phase, _ in items))

        lines = [
            f"{'Species':<{name_width}}   {'Phase':<{phase_width}}   Mole Fraction",
            "-" * (name_width + phase_width + 20),
        ]

        for name, phase, x in items:
            lines.append(f"  {name:<{name_width}}   {phase:<{phase_width}}   {x:.6f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode={self.mode!r}, "
            f"pressure={self.pressure:.6g}, "
            f"temperature={self.temperature:.6g}, "
            f"success={self.success})"
        )

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        return ["enthalpy-pressure", "pressure-temperature"]

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        return cls.available_flash_pairs()