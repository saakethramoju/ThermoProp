"""
tp_solver.py

Constant-temperature / constant-pressure equilibrium solver.

This module solves the CEA/RP-1311 reduced Gibbs TP problem for a fixed
species set.

Condensed-phase insertion/removal is intentionally handled outside this solver
by condensed.py. This solver assumes the supplied SpeciesSet is the active set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Exceptions import EquilibriumSetupError
from .state import EquilibriumState, SpeciesSet
from .thermo import thermo_arrays_for_species_set
from .matrix import (
    build_tp_matrix,
    solve_matrix_system,
    unpack_tp_solution,
    apply_correction,
    cea_damping_factor,
    residual_norm,
    max_species_correction,
)


@dataclass(slots=True)
class TPSolverOptions:
    """Numerical controls for fixed-temperature, fixed-pressure solves."""
    max_iterations: int = 80
    trace: float = 1e-300
    species_trace: float = 1e-12

    element_tolerance: float = 1e-8
    correction_tolerance: float = 5e-6

    size: float = 18.420681

    verbose: bool = False


@dataclass(slots=True)
class TPSolverResult:
    """Result bundle returned by the fixed-species TP solver."""
    state: EquilibriumState
    success: bool
    message: str
    iterations: int
    max_element_error: float
    max_correction: float
    residual_norm: float
    element_potentials: np.ndarray | None = None


def initial_tp_state(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    temperature: float,
    pressure: float,
    total_gas_moles_guess: float | None = None,
    trace: float = 1e-300,
) -> EquilibriumState:
    """
    CEA-style crude initial estimate.

    For first point, CEA uses a simple gas mole estimate and zero condensed
    species. Here we distribute gas moles uniformly among active gases and set
    condensed species to zero.
    """
    element_totals = np.asarray(element_totals, dtype=float)

    n = np.zeros(species.nspecies, dtype=float)

    gas_idx = np.nonzero(species.gas_mask)[0]
    condensed_idx = np.nonzero(species.condensed_mask)[0]

    if total_gas_moles_guess is None:
        total_gas_moles_guess = max(float(np.sum(element_totals)) / 2.0, 0.1)

    total_gas_moles_guess = max(float(total_gas_moles_guess), trace * len(gas_idx))

    n[gas_idx] = total_gas_moles_guess / max(len(gas_idx), 1)

    if len(condensed_idx):
        n[condensed_idx] = 0.0

    return EquilibriumState(
        temperature=float(temperature),
        pressure=float(pressure),
        n=n,
        total_gas_moles=float(np.sum(n[gas_idx])),
        species=species,
        element_totals=element_totals,
        iteration=0,
        converged=False,
        residual_norm=np.inf,
    )


def solve_tp(
    state: EquilibriumState,
    *,
    options: TPSolverOptions | None = None,
) -> TPSolverResult:
    """
    Solve TP equilibrium for the active SpeciesSet.

    Parameters
    ----------
    state:
        Initial equilibrium state.

    options:
        Solver controls.

    Returns
    -------
    TPSolverResult
    """
    if options is None:
        options = TPSolverOptions()

    current = state.copy()
    species = current.species

    gas_idx = np.nonzero(species.gas_mask)[0]

    if len(gas_idx) == 0:
        raise EquilibriumSetupError("TP solver requires at least one gas species.")

    n = np.maximum(current.n.astype(float), 0.0)
    n[gas_idx] = np.maximum(n[gas_idx], options.trace)

    for iteration in range(1, options.max_iterations + 1):
        thermo = thermo_arrays_for_species_set(
            species,
            current.temperature,
            on_error="raise",
        )

        system = build_tp_matrix(
            species=species,
            n=n,
            element_totals=current.element_totals,
            thermo=thermo,
            pressure=current.pressure,
            trace=options.trace,
        )

        raw = solve_matrix_system(system)

        correction = unpack_tp_solution(
            raw,
            species=species,
            mu_over_RT=system.mu_over_RT,
        )

        damping = cea_damping_factor(
            species=species,
            n=n,
            correction=correction,
            size=options.size,
        )

        n_new = apply_correction(
            species=species,
            n=n,
            correction=correction,
            damping=damping,
            trace=options.trace,
        )

        n_new[gas_idx] = np.maximum(n_new[gas_idx], options.trace)

        # Remove tiny negative condensed amounts caused by linear correction.
        condensed_idx = np.nonzero(species.condensed_mask)[0]
        if len(condensed_idx):
            tiny_negative = (
                (n_new[condensed_idx] < 0.0)
                & (np.abs(n_new[condensed_idx]) < options.species_trace)
            )
            if np.any(tiny_negative):
                local = condensed_idx[tiny_negative]
                n_new[local] = 0.0

        current.n = n_new
        current.total_gas_moles = float(np.sum(n_new[gas_idx]))
        current.iteration = iteration
        current.residual_norm = residual_norm(system, raw)

        element_error = species.A @ n_new - current.element_totals
        max_element_error = float(np.max(np.abs(element_error)))

        max_corr = max(
            max_species_correction(species, correction),
            abs(correction.dln_total_gas_moles),
        )

        if options.verbose:
            print(
                f"TP {iteration:3d} "
                f"alpha={damping:.3e} "
                f"elem={max_element_error:.3e} "
                f"corr={max_corr:.3e} "
                f"ngas={current.total_gas_moles:.6e}"
            )

        n = n_new

        if _tp_converged(
            current=current,
            correction=correction,
            max_element_error=max_element_error,
            max_correction=max_corr,
            options=options,
        ):
            current.converged = True
            return TPSolverResult(
                state=current,
                success=True,
                message="TP equilibrium converged.",
                iterations=iteration,
                max_element_error=max_element_error,
                max_correction=max_corr,
                residual_norm=current.residual_norm,
                element_potentials=correction.element_potentials.copy(),
            )

    current.converged = False
    return TPSolverResult(
        state=current,
        success=False,
        message="TP equilibrium did not converge within max_iterations.",
        iterations=options.max_iterations,
        max_element_error=float(np.max(np.abs(species.A @ n - current.element_totals))),
        max_correction=max_corr if "max_corr" in locals() else np.inf,
        residual_norm=current.residual_norm,
        element_potentials=correction.element_potentials.copy() if "correction" in locals() else None,
    )


def _tp_converged(
    *,
    current: EquilibriumState,
    correction,
    max_element_error: float,
    max_correction: float,
    options: TPSolverOptions,
) -> bool:
    """
    CEA-inspired TP convergence criteria.

    CEA checks composition corrections, total mole correction, and elemental
    balance. Here we use the same concepts while keeping the implementation
    explicit and SI/kmol based.
    """
    if max_element_error > options.element_tolerance:
        return False

    if current.residual_norm > 1e-10:
        return False

    if abs(correction.dln_total_gas_moles) > options.correction_tolerance:
        return False

    gas_idx = np.nonzero(current.species.gas_mask)[0]
    ng = current.n[gas_idx]
    total_gas = float(np.sum(ng))

    if total_gas > 0.0:
        weighted_gas_correction = np.max(
            np.abs(ng * correction.dln_gas_moles)
        )
        if weighted_gas_correction > options.correction_tolerance:
            return False

    condensed_idx = np.nonzero(current.species.condensed_mask)[0]
    if len(condensed_idx):
        if np.max(np.abs(correction.condensed_corrections)) > options.correction_tolerance:
            return False
    return True



def solve_tp_from_scratch(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    temperature: float,
    pressure: float,
    options: TPSolverOptions | None = None,
) -> TPSolverResult:
    if options is None:
        options = TPSolverOptions()

    state = initial_tp_state(
        species=species,
        element_totals=element_totals,
        temperature=temperature,
        pressure=pressure,
        trace=options.trace,
    )

    return solve_tp(state, options=options)
