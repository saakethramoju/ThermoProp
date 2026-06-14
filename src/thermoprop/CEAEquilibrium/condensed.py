"""
condensed.py

Condensed-phase insertion/removal logic for the CEA-style equilibrium solver.

This module wraps the fixed-species TP/HP solvers with CEA-like condensed phase
tests:

1. Start with gas species only.
2. Solve to convergence.
3. Test dormant condensed species using the Gibbs criterion.
4. Insert at most one condensed species per outer pass.
5. Re-solve.
6. Remove condensed species with negative/vanishing amounts.
7. Repeat until no insertion/removal is required.

RP-1311 section 3.4 describes this behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .state import EquilibriumState, SpeciesSet, CondensedPhaseCandidate
from .species import (
    SpeciesBuildOptions,
    build_species_set,
    add_species_to_set,
    remove_species_from_set,
)
from .thermo import thermo_arrays_for_species_set
from .tp_solver import TPSolverOptions, TPSolverResult, solve_tp, initial_tp_state
from .hp_solver import HPSolverOptions, HPSolverResult, solve_hp, initial_hp_state


Mode = Literal["tp", "hp"]


@dataclass(slots=True)
class CondensedOptions:
    enabled: bool = True
    max_outer_iterations: int = 30

    insertion_tolerance: float = -1e-8
    removal_tolerance: float = 0.0
    initial_condensed_moles: float = 1e-12

    include_ions: bool = False
    include_electron: bool = False

    verbose: bool = False


@dataclass(slots=True)
class CondensedSolveResult:
    state: EquilibriumState
    success: bool
    message: str
    outer_iterations: int
    inner_iterations: int
    inserted_species: list[str]
    removed_species: list[str]
    last_solver_result: TPSolverResult | HPSolverResult


def condensed_gibbs_test_values(
    *,
    active_state: EquilibriumState,
    dormant_condensed_species: SpeciesSet,
) -> list[CondensedPhaseCandidate]:
    """
    Evaluate CEA condensed appearance criterion for all dormant condensed species.

    For a dormant condensed species c, the CEA criterion is essentially:

        g_c / RT - sum_i(a_ic * pi_i) < 0

    where pi_i are element potentials recovered from the active gas equilibrium.

    Here pi_i are estimated by least-squares from active gas species:

        mu_j/RT ≈ A_j^T pi + ln(n_gas)

    The most negative test value is inserted first.
    """
    active_species = active_state.species
    T = active_state.temperature
    P = active_state.pressure

    thermo_active = thermo_arrays_for_species_set(active_species, T)

    gas_idx = np.nonzero(active_species.gas_mask)[0]
    ng = active_state.n[gas_idx]
    total_gas = float(np.sum(ng))

    if total_gas <= 0.0:
        return []

    xg = ng / total_gas

    from .thermo import chemical_potentials_over_RT

    mu_active = chemical_potentials_over_RT(
        thermo_active,
        xg,
        P,
        gas_mask=active_species.gas_mask,
        condensed_mask=active_species.condensed_mask,
    )

    Ag = active_species.A[:, gas_idx]
    mug = mu_active[gas_idx]

    # Estimate element potentials pi from gas species equilibrium condition.
    # At exact equilibrium this is consistent with the reduced Gibbs equations.
    pi, *_ = np.linalg.lstsq(Ag.T, mug, rcond=None)

    thermo_dormant = thermo_arrays_for_species_set(
        dormant_condensed_species,
        T,
        on_error="nan",
    )

    candidates: list[CondensedPhaseCandidate] = []

    for j, name in enumerate(dormant_condensed_species.names):
        if not thermo_dormant.valid[j]:
            continue

        if not dormant_condensed_species.condensed_mask[j]:
            continue

        a_c = dormant_condensed_species.A[:, j]

        # Align element vectors if needed.
        if dormant_condensed_species.elements != active_species.elements:
            aligned = np.zeros(active_species.nelements, dtype=float)
            for i, element in enumerate(active_species.elements):
                if element in dormant_condensed_species.elements:
                    k = dormant_condensed_species.elements.index(element)
                    aligned[i] = a_c[k]
            a_c = aligned

        value = float(thermo_dormant.g0_over_RT[j] - np.dot(a_c, pi))

        candidates.append(
            CondensedPhaseCandidate(
                species_index=j,
                species_name=name,
                gibbs_test_value=value,
                should_insert=value < 0.0,
            )
        )

    candidates.sort(key=lambda c: c.gibbs_test_value)
    return candidates


def choose_condensed_species_to_insert(
    candidates: list[CondensedPhaseCandidate],
    *,
    tolerance: float = -1e-8,
) -> CondensedPhaseCandidate | None:
    for candidate in candidates:
        if candidate.gibbs_test_value < tolerance:
            return candidate
    return None


def condensed_species_to_remove(
    state: EquilibriumState,
    *,
    tolerance: float = 0.0,
) -> list[str]:
    condensed_idx = np.nonzero(state.species.condensed_mask)[0]
    remove: list[str] = []

    for idx in condensed_idx:
        if state.n[idx] <= tolerance:
            remove.append(state.species.names[idx])

    return remove


def _gas_only_species_set(
    *,
    elements: list[str],
    temperature: float | None,
    candidates: list[str] | None,
    include_ions: bool,
    include_electron: bool,
) -> SpeciesSet:
    return build_species_set(
        elements,
        candidates=candidates,
        options=SpeciesBuildOptions(
            include_gases=True,
            include_condensed=False,
            include_ions=include_ions,
            include_electron=include_electron,
            include_reactants=False,
            require_thermo=True,
            temperature=temperature,
        ),
    )


def _all_condensed_species_set(
    *,
    elements: list[str],
    temperature: float | None,
    candidates: list[str] | None,
    include_ions: bool,
    include_electron: bool,
) -> SpeciesSet:
    return build_species_set(
        elements,
        candidates=candidates,
        options=SpeciesBuildOptions(
            include_gases=False,
            include_condensed=True,
            include_ions=include_ions,
            include_electron=include_electron,
            include_reactants=False,
            require_thermo=True,
            temperature=temperature,
        ),
    )


def _transfer_state_to_species_set(
    old_state: EquilibriumState,
    new_species: SpeciesSet,
    *,
    initial_new_moles: float = 1e-12,
) -> EquilibriumState:
    n_new = np.zeros(new_species.nspecies, dtype=float)

    old_lookup = old_state.species.name_to_index

    for j, name in enumerate(new_species.names):
        if name in old_lookup:
            n_new[j] = old_state.n[old_lookup[name]]
        else:
            n_new[j] = initial_new_moles if new_species.condensed_mask[j] else 1e-300

    gas_idx = np.nonzero(new_species.gas_mask)[0]

    return EquilibriumState(
        temperature=old_state.temperature,
        pressure=old_state.pressure,
        n=n_new,
        total_gas_moles=float(np.sum(n_new[gas_idx])),
        species=new_species,
        element_totals=old_state.element_totals.copy(),
        iteration=0,
        converged=False,
        residual_norm=np.inf,
    )


def solve_with_condensed_phases_tp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    temperature: float,
    pressure: float,
    candidates: list[str] | None = None,
    tp_options: TPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    if tp_options is None:
        tp_options = TPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    active_species = _gas_only_species_set(
        elements=elements,
        temperature=temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    dormant_condensed = _all_condensed_species_set(
        elements=elements,
        temperature=temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )
    state = initial_tp_state(
        species=active_species,
        element_totals=element_totals,
        temperature=temperature,
        pressure=pressure,
        trace=tp_options.trace,
    )

    inserted: list[str] = []
    removed: list[str] = []
    inner_iterations = 0
    last_result: TPSolverResult | None = None

    for outer in range(1, condensed_options.max_outer_iterations + 1):
        last_result = solve_tp(state, options=tp_options)
        inner_iterations += last_result.iterations

        if not last_result.success:
            return CondensedSolveResult(
                state=last_result.state,
                success=False,
                message=last_result.message,
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        state = last_result.state

        remove_now = condensed_species_to_remove(
            state,
            tolerance=condensed_options.removal_tolerance,
        )

        if remove_now:
            active_species = remove_species_from_set(state.species, remove_now)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_now)
            continue

        if not condensed_options.enabled:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged without condensed-phase insertion.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        dormant_names = [
            name
            for name in dormant_condensed.names
            if name not in state.species.name_to_index
        ]

        if not dormant_names:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        from .species import subset_species_set

        keep = [
            i
            for i, name in enumerate(dormant_condensed.names)
            if name in dormant_names
        ]

        dormant_subset = subset_species_set(dormant_condensed, keep)

        tests = condensed_gibbs_test_values(
            active_state=state,
            dormant_condensed_species=dormant_subset,
        )

        chosen = choose_condensed_species_to_insert(
            tests,
            tolerance=condensed_options.insertion_tolerance,
        )

        if chosen is None:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        active_species = add_species_to_set(state.species, [chosen.species_name])
        state = _transfer_state_to_species_set(
            state,
            active_species,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )
        inserted.append(chosen.species_name)

        if condensed_options.verbose:
            print(
                f"Inserted condensed species {chosen.species_name} "
                f"Gtest={chosen.gibbs_test_value:.6e}"
            )

    return CondensedSolveResult(
        state=state,
        success=False,
        message="Condensed-phase TP outer loop exceeded max iterations.",
        outer_iterations=condensed_options.max_outer_iterations,
        inner_iterations=inner_iterations,
        inserted_species=inserted,
        removed_species=removed,
        last_solver_result=last_result,
    )


def solve_with_condensed_phases_hp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    pressure: float,
    target_enthalpy: float,
    guess_temperature: float = 3800.0,
    candidates: list[str] | None = None,
    hp_options: HPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    if hp_options is None:
        hp_options = HPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    active_species = _gas_only_species_set(
        elements=elements,
        temperature=guess_temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    dormant_condensed = _all_condensed_species_set(
        elements=elements,
        temperature=guess_temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    state = initial_hp_state(
        species=active_species,
        element_totals=element_totals,
        pressure=pressure,
        guess_temperature=guess_temperature,
        trace=hp_options.trace,
    )

    inserted: list[str] = []
    removed: list[str] = []
    inner_iterations = 0
    last_result: HPSolverResult | None = None

    for outer in range(1, condensed_options.max_outer_iterations + 1):
        last_result = solve_hp(
            state,
            target_enthalpy=target_enthalpy,
            options=hp_options,
        )
        inner_iterations += last_result.iterations

        if not last_result.success:
            return CondensedSolveResult(
                state=last_result.state,
                success=False,
                message=last_result.message,
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        state = last_result.state

        remove_now = condensed_species_to_remove(
            state,
            tolerance=condensed_options.removal_tolerance,
        )

        if remove_now:
            active_species = remove_species_from_set(state.species, remove_now)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_now)
            continue

        if not condensed_options.enabled:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="HP equilibrium converged without condensed-phase insertion.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        dormant_names = [
            name
            for name in dormant_condensed.names
            if name not in state.species.name_to_index
        ]

        if not dormant_names:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="HP equilibrium converged; no dormant condensed species remain.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        from .species import subset_species_set

        keep = [
            i
            for i, name in enumerate(dormant_condensed.names)
            if name in dormant_names
        ]

        dormant_subset = subset_species_set(dormant_condensed, keep)

        tests = condensed_gibbs_test_values(
            active_state=state,
            dormant_condensed_species=dormant_subset,
        )

        chosen = choose_condensed_species_to_insert(
            tests,
            tolerance=condensed_options.insertion_tolerance,
        )

        if chosen is None:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="HP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        active_species = add_species_to_set(state.species, [chosen.species_name])
        state = _transfer_state_to_species_set(
            state,
            active_species,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )
        inserted.append(chosen.species_name)

        if condensed_options.verbose:
            print(
                f"Inserted condensed species {chosen.species_name} "
                f"Gtest={chosen.gibbs_test_value:.6e}"
            )

    return CondensedSolveResult(
        state=state,
        success=False,
        message="Condensed-phase HP outer loop exceeded max iterations.",
        outer_iterations=condensed_options.max_outer_iterations,
        inner_iterations=inner_iterations,
        inserted_species=inserted,
        removed_species=removed,
        last_solver_result=last_result,
    )