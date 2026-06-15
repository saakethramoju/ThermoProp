"""
Equilibrium.py

Public API wrapper for the modular CEA-style equilibrium package.

Directory layout expected:

src/
    Equilibrium.py
    CEADatabase.py
    Reactants.py
    CombustionGas.py
    Equilibrium/
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

from dataclasses import dataclass
from typing import Any

import numpy as np

from .CEADatabase import CEA
from .Reactants import Reactants
from .CombustionGas import CombustionGas


from .CEAEquilibrium.state import FeedState, EquilibriumState, EquilibriumResults
from .CEAEquilibrium.species import CHARGE_ELEMENT
from .CEAEquilibrium.condensed import (
    CondensedOptions,
    CondensedSolveResult,
    solve_with_condensed_phases_tp,
    solve_with_condensed_phases_hp,
)
from .CEAEquilibrium.tp_solver import TPSolverOptions, solve_tp
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
    speed_of_sound_frozen,
    speed_of_sound_equilibrium,
    frozen_mixture_derivatives,
)
from .CEAEquilibrium.transport import (
    TransportOptions,
    build_transport_values,
)


@dataclass(slots=True)
class EquilibriumSolveSummary:
    success: bool
    message: str
    mode: str
    iterations: int
    outer_iterations: int
    inserted_condensed_species: list[str]
    removed_condensed_species: list[str]
    max_element_error: float | None = None
    enthalpy_error: float | None = None
    residual_norm: float | None = None


class _DictReactants:
    """
    Adapter for raw composition dictionaries.

    Dict interpretation:
        {"CO2": 0.5, "H2O": 0.5}

    basis:
        "mole" or "mass"

    For HP, temperature is required so reactant enthalpy is defined.
    """

    def __init__(
        self,
        composition: dict[str, float],
        *,
        basis: str = "mole",
        temperature: float | None = None,
        pressure: float | None = None,
    ):
        if not composition:
            raise ValueError("Reactant composition dictionary cannot be empty.")

        self.composition = {
            CEA.resolve_name(name) if hasattr(CEA, "resolve_name") else name: float(value)
            for name, value in composition.items()
            if float(value) > 0.0
        }

        if not self.composition:
            raise ValueError("Reactant composition dictionary must contain positive amounts.")

        self.basis = basis.lower()
        self.temperature = temperature
        self.pressure = pressure

        if self.basis not in {"mole", "mass"}:
            raise ValueError("dict reactant basis must be 'mole' or 'mass'.")

        total = sum(self.composition.values())
        raw = {name: value / total for name, value in self.composition.items()}

        if self.basis == "mole":
            mole_fractions = raw
        else:
            denom = 0.0
            for name, y in raw.items():
                denom += y / CEA.molar_mass(name)
            mole_fractions = {
                name: (y / CEA.molar_mass(name)) / denom
                for name, y in raw.items()
            }

        self.mole_fractions = mole_fractions

        mw = sum(
            x * CEA.molar_mass(name)
            for name, x in mole_fractions.items()
        )

        self.total_mass = 1.0
        self.total_moles = 1.0 / mw
        self.total_kmoles = self.total_moles / 1000.0
        self.molecular_weight = mw
        self.molecular_weight_kg_per_kmol = mw * 1000.0

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        totals: dict[str, float] = {}

        for name, x in self.mole_fractions.items():
            n_mol_per_kg = x * self.total_moles
            n_kmol_per_kg = n_mol_per_kg / 1000.0

            comp = CEA.elemental_composition(name)

            for element, count in comp.items():
                totals[element] = totals.get(element, 0.0) + n_kmol_per_kg * float(count)

        return dict(sorted(totals.items()))

    @property
    def reactant_enthalpy(self) -> float:
        if self.temperature is None:
            raise ValueError("HP equilibrium with dict reactants requires temperature.")

        h = 0.0

        for name, x in self.mole_fractions.items():
            n_kmol_per_kg = x * self.total_kmoles
            h_kmol = CEA.thermo_molar(name, self.temperature)[1]
            h += n_kmol_per_kg * h_kmol

        return float(h)

    @property
    def reactant_internal_energy(self) -> float:
        return self.reactant_enthalpy - self.total_kmoles * 8314.46261815324 * float(self.temperature)

    def element_vector(self, elements: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        if elements is None:
            elements = sorted(self.element_moles_per_kg)

        b = np.array(
            [self.element_moles_per_kg.get(element, 0.0) for element in elements],
            dtype=float,
        )

        return elements, b


class _CombustionGasReactants:
    def __init__(self, gas: CombustionGas):
        self.gas = gas
        self.total_mass = 1.0

        mole_fractions = dict(gas.mole_fractions)

        if not mole_fractions:
            raise ValueError("CombustionGas composition cannot be empty.")

        total_x = sum(float(x) for x in mole_fractions.values())

        if total_x <= 0.0:
            raise ValueError("CombustionGas mole fractions must sum to a positive value.")

        self.mole_fractions = {
            name: float(x) / total_x
            for name, x in mole_fractions.items()
            if float(x) > 0.0
        }

        self.molecular_weight = sum(
            self.mole_fractions[name] * CEA.molar_mass(name)
            for name in self.mole_fractions
        )

        self.molecular_weight_kg_per_kmol = self.molecular_weight * 1000.0
        self.total_moles = 1.0 / self.molecular_weight
        self.total_kmoles = self.total_moles / 1000.0

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        totals: dict[str, float] = {}

        for name, x in self.mole_fractions.items():
            n_kmol_per_kg = x * self.total_kmoles
            comp = CEA.elemental_composition(name)

            for element, count in comp.items():
                totals[element] = totals.get(element, 0.0) + n_kmol_per_kg * float(count)

        return dict(sorted(totals.items()))

    @property
    def reactant_enthalpy(self) -> float:
        return float(self.gas.enthalpy)

    @property
    def reactant_internal_energy(self) -> float:
        return float(self.gas.internal_energy)

    def element_vector(self, elements: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        if elements is None:
            elements = sorted(self.element_moles_per_kg)

        b = np.array(
            [self.element_moles_per_kg.get(element, 0.0) for element in elements],
            dtype=float,
        )

        return elements, b


class Equilibrium:
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
        basis: str = "mole",
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

        self._reactants = self._resolve_reactants(
            reactants,
            basis=basis,
            temperature=temperature,
            pressure=pressure,
        )

        self._feed: FeedState | None = None
        self._solve_result: CondensedSolveResult | None = None
        self._state: EquilibriumState | None = None
        self._results: EquilibriumResults | None = None
        self._summary: EquilibriumSolveSummary | None = None
        self._gas_cache: CombustionGas | None = None

        self._solve()

    @staticmethod
    def _resolve_reactants(
        reactants,
        *,
        basis: str,
        temperature: float | None,
        pressure: float | None,
    ):
        if isinstance(reactants, Reactants):
            return reactants

        if isinstance(reactants, CombustionGas):
            return _CombustionGasReactants(reactants)

        if isinstance(reactants, dict):
            return _DictReactants(
                reactants,
                basis=basis,
                temperature=temperature,
                pressure=pressure,
            )

        if reactants.__class__.__name__ == "Reactants":
            return reactants

        if reactants.__class__.__name__ == "CombustionGas":
            return _CombustionGasReactants(reactants)

        raise TypeError("reactants must be Reactants, CombustionGas, or dict.")

    def _validate(self) -> None:
        if self._pressure is None or self._pressure <= 0.0:
            raise ValueError("Equilibrium requires positive pressure [Pa].")

        if self._mode == "tp":
            if self._temperature_input is None:
                raise ValueError("TP equilibrium requires temperature [K].")
            if self._temperature_input <= 0.0:
                raise ValueError("temperature must be positive.")

        elif self._mode == "hp":
            if isinstance(self._input, dict) and self._temperature_input is None:
                raise ValueError("HP equilibrium with dict reactants requires temperature.")
            if self._guess_temperature <= 0.0:
                raise ValueError("guess_temperature must be positive.")

        else:
            raise ValueError("mode must be 'tp' or 'hp'.")

    def _build_feed(self) -> FeedState:
        element_moles = dict(self._reactants.element_moles_per_kg)
        elements = sorted(e for e in element_moles if e != CHARGE_ELEMENT)

        b = np.array(
            [element_moles.get(element, 0.0) for element in elements],
            dtype=float,
        )

        # Reactants.element_moles_per_kg is mol element / kg mixture.
        # The CEAEquilibrium solver uses kmol element / kg mixture.
        if isinstance(self._input, Reactants) or self._input.__class__.__name__ == "Reactants":
            b = b / 1000.0
        if self._include_ions:
            elements_with_charge = list(elements) + [CHARGE_ELEMENT]
            b = np.append(b, 0.0)
        else:
            elements_with_charge = elements

        return FeedState(
            element_totals=b,
            elements=elements_with_charge,
            enthalpy=getattr(self._reactants, "reactant_enthalpy", None),
            internal_energy=self._safe_reactant_internal_energy(),
            temperature=self._temperature_input,
            pressure=self._pressure,
            source=self._reactants,
        )
        
    def _safe_reactant_internal_energy(self) -> float | None:
        try:
            return float(self._reactants.reactant_internal_energy)
        except Exception:
            return None

    def _solver_options(self):
        from .CEAEquilibrium.tp_solver import TPSolverOptions
        from .CEAEquilibrium.hp_solver import HPSolverOptions

        tp_options = TPSolverOptions(
            max_iterations=self._max_iterations,
            species_trace=self._combustion_gas_trace,
            verbose=self._verbose,
        )

        hp_options = HPSolverOptions(
            max_iterations=self._max_iterations,
            species_trace=self._combustion_gas_trace,
            verbose=self._verbose,
        )

        condensed_options = CondensedOptions(
            enabled=self._include_condensed,
            max_outer_iterations=self._max_outer_iterations,
            include_ions=self._include_ions,
            include_electron=self._include_electron,
            verbose=self._verbose,
        )

        transport_options = TransportOptions(
            trace=self._combustion_gas_trace,
            max_species=self._combustion_gas_max_species,
            equilibrium_derivative_temperature_step=self._equilibrium_derivative_temperature_step,
        )

        return tp_options, hp_options, condensed_options, transport_options

    def _solve(self) -> None:
        self._validate()

        self._gas_cache = None
        self._feed = self._build_feed()

        elements_for_species = [
            e for e in self._feed.elements if e != CHARGE_ELEMENT
        ]

        tp_options, hp_options, condensed_options, transport_options = self._solver_options()

        if self._mode == "tp":
            solve_result = solve_with_condensed_phases_tp(
                elements=elements_for_species,
                element_totals=self._feed.element_totals,
                temperature=float(self._temperature_input),
                pressure=float(self._pressure),
                candidates=self._candidates,
                tp_options=tp_options,
                condensed_options=condensed_options,
            )

        else:
            if self._feed.enthalpy is None:
                raise ValueError("HP equilibrium requires reactant enthalpy.")

            solve_result = solve_with_condensed_phases_hp(
                elements=elements_for_species,
                element_totals=self._feed.element_totals,
                pressure=float(self._pressure),
                target_enthalpy=float(self._feed.enthalpy),
                guess_temperature=self._guess_temperature,
                candidates=self._candidates,
                hp_options=hp_options,
                condensed_options=condensed_options,
            )

        self._solve_result = solve_result

        if not solve_result.success:
            raise RuntimeError(f"Equilibrium solve failed: {solve_result.message}")

        self._state = solve_result.state

        transport_values = build_transport_values(
            self._state,
            tp_neighbor_solver=self._tp_neighbor_state,
            options=transport_options,
        )

        self._results = build_results(
            self._state,
            tp_neighbor_solver=self._tp_neighbor_state,
            equilibrium_derivative_step=self._equilibrium_derivative_temperature_step,
            transport_values=transport_values,
        )

        last = solve_result.last_solver_result

        self._summary = EquilibriumSolveSummary(
            success=solve_result.success,
            message=solve_result.message,
            mode=self._mode,
            iterations=solve_result.inner_iterations,
            outer_iterations=solve_result.outer_iterations,
            inserted_condensed_species=list(solve_result.inserted_species),
            removed_condensed_species=list(solve_result.removed_species),
            max_element_error=getattr(last, "max_element_error", None),
            enthalpy_error=getattr(last, "enthalpy_error", None),
            residual_norm=getattr(last, "residual_norm", None),
        )

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
        self._pressure = float(value)
        self._solve()

    @property
    def temperature(self) -> float:
        return self._state.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if self._mode == "tp":
            self._temperature_input = float(value)
        else:
            self._guess_temperature = float(value)
        self._solve()

    @property
    def pressure_temperature(self) -> tuple[float, float]:
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: tuple[float, float]) -> None:
        self._pressure = float(values[0])
        self._temperature_input = float(values[1])
        self._mode = "tp"
        self._solve()

    @property
    def TP(self) -> tuple[float, float]:
        return self.temperature, self.pressure

    @TP.setter
    def TP(self, values: tuple[float, float]) -> None:
        self._temperature_input = float(values[0])
        self._pressure = float(values[1])
        self._mode = "tp"
        self._solve()

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
    def normalized_mole_fractions(self) -> dict[str, float]:
        return self.CombustionGas_composition()

    @property
    def normalized_mass_fractions(self) -> dict[str, float]:
        return dict(self.CombustionGas.mass_fractions)

    def CombustionGas_composition(
        self,
        trace: float | None = None,
        max_species: int | None = None,
    ) -> dict[str, float]:
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

    @property
    def CombustionGas(self) -> CombustionGas:
        if self._gas_cache is None:
            self._gas_cache = CombustionGas(
                self.CombustionGas_composition(),
                basis="mole",
                pressure=self.pressure,
                temperature=self.temperature,
            )
        return self._gas_cache

    @property
    def gas(self) -> CombustionGas:
        return self.CombustionGas

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
        return self.CombustionGas.partial_derivative(of, with_respect_to, constant)

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
            "mole_fractions": state_mole_fractions(self._state, trace=trace),
            "gas_mole_fractions": state_gas_mole_fractions(self._state, trace=trace),
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
        if value is None:
            return "None"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

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

        width = max(len(key) for key, _ in rows)
        header = "\n".join(f"{key:<{width}} : {value}" for key, value in rows)

        return header + "\n\n" + self._format_species_table()
    

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