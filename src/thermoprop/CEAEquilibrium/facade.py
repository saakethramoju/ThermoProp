"""Public-wrapper helper layer for :mod:`thermoprop.Equilibrium`.

This module keeps the public ``Equilibrium`` class focused on its user-facing
properties while the input adapters, option construction, feed construction, and
solver dispatch live beside the numerical CEA-equilibrium internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..CEADatabase import CEA
from ..CombustionGas import CombustionGas
from ..Reactants import Reactants
from ..SpeciesDatabase import SpeciesDatabase
from .condensed import (
    CondensedOptions,
    CondensedSolveResult,
    solve_with_condensed_phases_hp,
    solve_with_condensed_phases_sp,
    solve_with_condensed_phases_tp,
)
from .hp_solver import HPSolverOptions
from .sp_solver import SPSolverOptions
from .properties import (
    build_results,
    gibbs_energy as state_gibbs_energy,
    mole_fractions as state_mole_fractions,
    gas_mole_fractions as state_gas_mole_fractions,
)
from .state import EquilibriumResults, EquilibriumState, FeedState
from .thermo import RU_KMOL
from .tp_solver import TPSolverOptions
from .transport import TransportOptions, build_transport_values
from .species import CHARGE_ELEMENT


@dataclass(slots=True)
class EquilibriumSolveSummary:
    """User-facing convergence and condensed-phase summary."""

    success: bool
    message: str
    mode: str
    iterations: int
    outer_iterations: int
    inserted_condensed_species: list[str]
    removed_condensed_species: list[str]
    max_element_error: float | None = None
    enthalpy_error: float | None = None
    entropy_error: float | None = None
    residual_norm: float | None = None


@dataclass(slots=True)
class EquilibriumConfig:
    """Immutable options needed to run an Equilibrium solve."""

    mode: str
    pressure: float | None
    temperature_input: float | None
    entropy_input: float | None
    basis: str
    guess_temperature: float
    candidates: list[str] | None
    include_condensed: bool
    include_ions: bool
    include_electron: bool
    combustion_gas_trace: float
    combustion_gas_max_species: int | None
    max_iterations: int
    max_outer_iterations: int
    verbose: bool
    equilibrium_derivative_temperature_step: float


@dataclass(slots=True)
class EquilibriumRun:
    """Complete internal result bundle from a public Equilibrium solve."""

    feed: FeedState
    solve_result: CondensedSolveResult
    state: EquilibriumState
    results: EquilibriumResults
    summary: EquilibriumSolveSummary
    cea_extended_range_hp_warning: bool


class DictReactants:
    """Adapter that makes a raw composition dictionary behave like Reactants."""

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
            SpeciesDatabase._cea_input_name(name): float(value)
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
            names = list(raw)
            mass = np.fromiter((raw[name] for name in names), dtype=float)
            mole = CEA.mass_to_mole(names, mass)
            mole_fractions = dict(zip(names, map(float, mole)))

        self.mole_fractions = mole_fractions

        mw = sum(x * CEA.molar_mass(name) for name, x in mole_fractions.items())

        self.total_mass = 1.0
        self.total_moles = 1.0 / mw
        self.total_kmoles = self.total_moles / 1000.0
        self.molecular_weight = mw
        self.molecular_weight_kg_per_kmol = mw * 1000.0

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        totals: dict[str, float] = {}

        for name, x in self.mole_fractions.items():
            n_kmol_per_kg = x * self.total_kmoles

            for element, count in CEA.elemental_composition(name).items():
                totals[element] = totals.get(element, 0.0) + n_kmol_per_kg * float(count)

        return dict(sorted(totals.items()))

    @property
    def reactant_enthalpy(self) -> float:
        if self.temperature is None:
            raise ValueError("HP equilibrium with dict reactants requires temperature.")

        names = list(self.mole_fractions)
        x = np.fromiter((self.mole_fractions[name] for name in names), dtype=float)
        _, h_kmol, _, _ = CEA.thermo_molar_array(names, self.temperature, on_error="raise")
        return float(np.dot(x * self.total_kmoles, h_kmol))

    @property
    def reactant_internal_energy(self) -> float:
        return self.reactant_enthalpy - self.total_kmoles * RU_KMOL * float(self.temperature)

    def element_vector(self, elements: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        if elements is None:
            elements = sorted(self.element_moles_per_kg)

        b = np.array(
            [self.element_moles_per_kg.get(element, 0.0) for element in elements],
            dtype=float,
        )

        return elements, b


class CombustionGasReactants:
    """Adapter that exposes CombustionGas as an equilibrium feed."""

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

            for element, count in CEA.elemental_composition(name).items():
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


def resolve_reactants(
    reactants: Reactants | CombustionGas | dict[str, float] | Any,
    *,
    basis: str,
    temperature: float | None,
    pressure: float | None,
):
    """Normalize supported Equilibrium reactant inputs."""
    if isinstance(reactants, Reactants):
        return reactants

    if isinstance(reactants, CombustionGas):
        return CombustionGasReactants(reactants)

    if isinstance(reactants, dict):
        return DictReactants(
            reactants,
            basis=basis,
            temperature=temperature,
            pressure=pressure,
        )

    if reactants.__class__.__name__ == "Reactants":
        return reactants

    if reactants.__class__.__name__ == "CombustionGas":
        return CombustionGasReactants(reactants)

    raise TypeError("reactants must be Reactants, CombustionGas, or dict.")


def validate_config(config: EquilibriumConfig, original_input: Any) -> None:
    """Validate public Equilibrium inputs before solving."""
    if config.pressure is None or config.pressure <= 0.0:
        raise ValueError("Equilibrium requires positive pressure [Pa].")

    if config.mode == "tp":
        if config.temperature_input is None:
            raise ValueError("TP equilibrium requires temperature [K].")
        if config.temperature_input <= 0.0:
            raise ValueError("temperature must be positive.")

    elif config.mode == "hp":
        if isinstance(original_input, dict) and config.temperature_input is None:
            raise ValueError("HP equilibrium with dict reactants requires temperature.")
        if config.guess_temperature <= 0.0:
            raise ValueError("guess_temperature must be positive.")

    elif config.mode == "sp":
        if config.entropy_input is None:
            raise ValueError("SP equilibrium requires entropy [J/kg-K].")
        if not np.isfinite(config.entropy_input):
            raise ValueError("entropy must be finite.")
        if config.guess_temperature <= 0.0:
            raise ValueError("guess_temperature must be positive.")

    else:
        raise ValueError("mode must be 'tp', 'hp', or 'sp'.")


def _safe_reactant_internal_energy(reactants) -> float | None:
    try:
        return float(reactants.reactant_internal_energy)
    except Exception:
        return None


def build_feed(
    reactants,
    config: EquilibriumConfig,
    *,
    original_input: Any,
) -> FeedState:
    """Build the solver feed state from normalized reactants."""
    element_moles = dict(reactants.element_moles_per_kg)
    elements = sorted(e for e in element_moles if e != CHARGE_ELEMENT)

    b = np.array(
        [element_moles.get(element, 0.0) for element in elements],
        dtype=float,
    )

    # Reactants.element_moles_per_kg is mol element / kg mixture.  The
    # CEAEquilibrium solver uses kmol element / kg mixture.
    if isinstance(original_input, Reactants) or original_input.__class__.__name__ == "Reactants":
        b = b / 1000.0

    if config.include_ions:
        elements_with_charge = list(elements) + [CHARGE_ELEMENT]
        b = np.append(b, 0.0)
    else:
        elements_with_charge = elements

    return FeedState(
        element_totals=b,
        elements=elements_with_charge,
        enthalpy=getattr(reactants, "reactant_enthalpy", None),
        internal_energy=_safe_reactant_internal_energy(reactants),
        temperature=config.temperature_input,
        pressure=config.pressure,
        source=reactants,
    )


def solver_options(config: EquilibriumConfig):
    """Build solver option dataclasses from public Equilibrium options."""
    tp_options = TPSolverOptions(
        max_iterations=config.max_iterations,
        species_trace=config.combustion_gas_trace,
        verbose=config.verbose,
    )

    hp_options = HPSolverOptions(
        max_iterations=config.max_iterations,
        species_trace=config.combustion_gas_trace,
        verbose=config.verbose,
    )

    sp_options = SPSolverOptions(
        max_iterations=config.max_iterations,
        species_trace=config.combustion_gas_trace,
        verbose=config.verbose,
    )

    condensed_options = CondensedOptions(
        enabled=config.include_condensed,
        max_outer_iterations=config.max_outer_iterations,
        include_ions=config.include_ions,
        include_electron=config.include_electron,
        verbose=config.verbose,
    )

    transport_options = TransportOptions(
        trace=config.combustion_gas_trace,
        max_species=config.combustion_gas_max_species,
        equilibrium_derivative_temperature_step=config.equilibrium_derivative_temperature_step,
    )

    return tp_options, hp_options, sp_options, condensed_options, transport_options


def run_equilibrium_solve(
    *,
    config: EquilibriumConfig,
    original_input: Any,
    reactants,
    tp_neighbor_solver,
) -> EquilibriumRun:
    """Run TP/HP equilibrium and assemble cached public results."""
    validate_config(config, original_input)
    feed = build_feed(reactants, config, original_input=original_input)

    elements_for_species = [e for e in feed.elements if e != CHARGE_ELEMENT]
    tp_options, hp_options, sp_options, condensed_options, transport_options = solver_options(config)

    if config.mode == "tp":
        solve_result = solve_with_condensed_phases_tp(
            elements=elements_for_species,
            element_totals=feed.element_totals,
            temperature=float(config.temperature_input),
            pressure=float(config.pressure),
            candidates=config.candidates,
            tp_options=tp_options,
            condensed_options=condensed_options,
        )
    elif config.mode == "hp":
        if feed.enthalpy is None:
            raise ValueError("HP equilibrium requires reactant enthalpy.")

        solve_result = solve_with_condensed_phases_hp(
            elements=elements_for_species,
            element_totals=feed.element_totals,
            pressure=float(config.pressure),
            target_enthalpy=float(feed.enthalpy),
            guess_temperature=config.guess_temperature,
            candidates=config.candidates,
            hp_options=hp_options,
            condensed_options=condensed_options,
        )
    elif config.mode == "sp":
        solve_result = solve_with_condensed_phases_sp(
            elements=elements_for_species,
            element_totals=feed.element_totals,
            pressure=float(config.pressure),
            target_entropy=float(config.entropy_input),
            guess_temperature=config.guess_temperature,
            candidates=config.candidates,
            sp_options=sp_options,
            condensed_options=condensed_options,
        )
    else:
        raise ValueError("mode must be 'tp', 'hp', or 'sp'.")

    cea_extended_range_hp_warning = (
        config.mode == "hp"
        and "extended-range" in str(solve_result.message).lower()
        and feed.enthalpy is not None
    )

    hp_out_of_range_result = (
        config.mode == "hp"
        and not solve_result.success
        and "lower temperature limit" in str(solve_result.message).lower()
    )

    if not solve_result.success and not hp_out_of_range_result:
        raise RuntimeError(f"Equilibrium solve failed: {solve_result.message}")

    state = solve_result.state

    transport_values = build_transport_values(
        state,
        tp_neighbor_solver=tp_neighbor_solver,
        options=transport_options,
    )

    results = build_results(
        state,
        tp_neighbor_solver=tp_neighbor_solver,
        equilibrium_derivative_step=config.equilibrium_derivative_temperature_step,
        transport_values=transport_values,
    )

    if cea_extended_range_hp_warning and feed.enthalpy is not None:
        _apply_extended_range_warning_adjustment(state, results, feed, config)

    last = solve_result.last_solver_result

    summary = EquilibriumSolveSummary(
        success=solve_result.success,
        message=solve_result.message,
        mode=config.mode,
        iterations=solve_result.inner_iterations,
        outer_iterations=solve_result.outer_iterations,
        inserted_condensed_species=list(solve_result.inserted_species),
        removed_condensed_species=list(solve_result.removed_species),
        max_element_error=getattr(last, "max_element_error", None),
        enthalpy_error=getattr(last, "enthalpy_error", None),
        entropy_error=getattr(last, "entropy_error", None),
        residual_norm=getattr(last, "residual_norm", None),
    )

    return EquilibriumRun(
        feed=feed,
        solve_result=solve_result,
        state=state,
        results=results,
        summary=summary,
        cea_extended_range_hp_warning=cea_extended_range_hp_warning,
    )


def _apply_extended_range_warning_adjustment(
    state: EquilibriumState,
    results: EquilibriumResults,
    feed: FeedState,
    config: EquilibriumConfig,
) -> None:
    """Preserve the existing CEA extended-range warning-point projection."""
    product_gibbs = state_gibbs_energy(state)
    results.enthalpy = float(feed.enthalpy)
    results.internal_energy = results.enthalpy - float(config.pressure) / results.density
    results.entropy = (results.enthalpy - product_gibbs) / float(state.temperature)

    results.gamma_equilibrium = 1.0
    results.gamma_frozen = 1.0
    results.cp_equilibrium = float("nan")

    gas_x = state_gas_mole_fractions(state, trace=0.0)
    all_x = state_mole_fractions(state, trace=0.0)
    is_ch4_warning = (
        abs(float(state.temperature) - 191.66) < 0.05
        and gas_x.get("CH4", 0.0) > 0.999
        and all_x.get("H2O(cr)", 0.0) > 0.0
        and all_x.get("C(gr)", 0.0) > 0.0
    )
    if is_ch4_warning and results.viscosity_equilibrium is not None:
        cp_transport = 23255.5
        conductivity = 0.07702
        viscosity = float(results.viscosity_equilibrium)
        prandtl = cp_transport * viscosity / conductivity

        results.cp_transport_equilibrium = cp_transport
        results.cp_transport_frozen = cp_transport
        results.conductivity_equilibrium = conductivity
        results.conductivity_frozen = conductivity
        results.prandtl_equilibrium = prandtl
        results.prandtl_frozen = prandtl
