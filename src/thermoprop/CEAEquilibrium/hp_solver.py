"""
hp_solver.py

Constant-enthalpy / constant-pressure equilibrium solver.

This module solves the CEA/RP-1311 reduced Gibbs HP problem for a fixed
species set.

Condensed-phase insertion/removal is handled outside this solver by
condensed.py. This solver assumes the supplied SpeciesSet is the active set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import EquilibriumState, SpeciesSet
from .thermo import (
    thermo_arrays_for_species_set,
    mixture_enthalpy,
)
from .matrix import (
    build_hp_matrix,
    solve_matrix_system,
    unpack_hp_solution,
    apply_correction,
    cea_damping_factor,
    residual_norm,
    max_species_correction,
)


@dataclass(slots=True)
class HPSolverOptions:
    max_iterations: int = 120
    trace: float = 1e-300
    species_trace: float = 1e-12

    element_tolerance: float = 1e-8
    enthalpy_tolerance: float = 1e-3
    correction_tolerance: float = 5e-6
    temperature_correction_tolerance: float = 1e-4

    min_temperature: float = 200.0
    max_temperature: float = 20000.0

    size: float = 18.420681
    verbose: bool = False


@dataclass(slots=True)
class HPSolverResult:
    state: EquilibriumState
    success: bool
    message: str
    iterations: int
    max_element_error: float
    enthalpy_error: float
    max_correction: float
    temperature_correction: float
    residual_norm: float


def initial_hp_state(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    pressure: float,
    guess_temperature: float = 3800.0,
    total_gas_moles_guess: float | None = None,
    trace: float = 1e-300,
) -> EquilibriumState:
    """
    CEA-style crude initial HP estimate.

    CEA uses 3800 K unless a better estimate is supplied. Condensed species
    start at zero. Gas species are distributed uniformly.
    """
    element_totals = np.asarray(element_totals, dtype=float)

    T = float(guess_temperature)

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
        temperature=T,
        pressure=float(pressure),
        n=n,
        total_gas_moles=float(np.sum(n[gas_idx])),
        species=species,
        element_totals=element_totals,
        iteration=0,
        converged=False,
        residual_norm=np.inf,
    )


def solve_hp(
    state: EquilibriumState,
    *,
    target_enthalpy: float,
    options: HPSolverOptions | None = None,
) -> HPSolverResult:
    """
    Solve HP equilibrium for the active SpeciesSet.

    Parameters
    ----------
    state:
        Initial equilibrium state.

    target_enthalpy:
        Reactant enthalpy [J/kg].

    options:
        Solver controls.
    """
    if options is None:
        options = HPSolverOptions()

    current = state.copy()
    species = current.species

    gas_idx = np.nonzero(species.gas_mask)[0]

    if len(gas_idx) == 0:
        raise RuntimeError("HP solver requires at least one gas species.")

    n = np.maximum(current.n.astype(float), 0.0)
    n[gas_idx] = np.maximum(n[gas_idx], options.trace)

    T = float(
        np.clip(
            current.temperature,
            options.min_temperature,
            options.max_temperature,
        )
    )

    max_corr = np.inf
    dlnT_abs = np.inf
    h_error = np.inf
    max_element_error = np.inf

    for iteration in range(1, options.max_iterations + 1):
        T = float(np.clip(T, options.min_temperature, options.max_temperature))
        current.temperature = T

        thermo = thermo_arrays_for_species_set(
            species,
            current.temperature,
            on_error="nan",
        )

        invalid = ~thermo.valid
        if np.any(invalid):
            n[invalid] = 0.0
            gas_idx = np.nonzero(species.gas_mask)[0]
            n[gas_idx] = np.maximum(n[gas_idx], options.trace)

        system = build_hp_matrix(
            species=species,
            n=n,
            element_totals=current.element_totals,
            thermo=thermo,
            pressure=current.pressure,
            target_enthalpy=float(target_enthalpy),
            trace=options.trace,
        )

        raw = solve_matrix_system(system)

        correction = unpack_hp_solution(
            raw,
            species=species,
            thermo=thermo,
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

        condensed_idx = np.nonzero(species.condensed_mask)[0]
        if len(condensed_idx):
            tiny_negative = (
                (n_new[condensed_idx] < 0.0)
                & (np.abs(n_new[condensed_idx]) < options.species_trace)
            )
            if np.any(tiny_negative):
                n_new[condensed_idx[tiny_negative]] = 0.0

        dlnT = 0.0 if correction.dln_temperature is None else correction.dln_temperature
        T_new = T * np.exp(np.clip(damping * dlnT, -5.0, 5.0))
        T_new = float(np.clip(T_new, options.min_temperature, options.max_temperature))

        current.n = n_new
        current.temperature = T_new
        current.total_gas_moles = float(np.sum(n_new[gas_idx]))
        current.iteration = iteration
        current.residual_norm = residual_norm(system)

        thermo_new = thermo_arrays_for_species_set(
            species,
            current.temperature,
            on_error="nan",
        )
        h_products = mixture_enthalpy(n_new, thermo_new)
        h_error = h_products - float(target_enthalpy)

        element_error = species.A @ n_new - current.element_totals
        max_element_error = float(np.max(np.abs(element_error)))

        max_corr = max(
            max_species_correction(species, correction),
            abs(correction.dln_total_gas_moles),
        )

        dlnT_abs = abs(dlnT)

        if options.verbose:
            print(
                f"HP {iteration:3d} "
                f"alpha={damping:.3e} "
                f"T={current.temperature:.3f} "
                f"elem={max_element_error:.3e} "
                f"h_err={h_error:.3e} "
                f"corr={max_corr:.3e} "
                f"dlnT={dlnT_abs:.3e} "
                f"ngas={current.total_gas_moles:.6e}"
            )

        n = n_new
        T = T_new

        if _hp_converged(
            current=current,
            correction=correction,
            max_element_error=max_element_error,
            enthalpy_error=h_error,
            max_correction=max_corr,
            options=options,
        ):
            current.converged = True
            return HPSolverResult(
                state=current,
                success=True,
                message="HP equilibrium converged.",
                iterations=iteration,
                max_element_error=max_element_error,
                enthalpy_error=h_error,
                max_correction=max_corr,
                temperature_correction=dlnT_abs,
                residual_norm=current.residual_norm,
            )

    current.converged = False

    return HPSolverResult(
        state=current,
        success=False,
        message="HP equilibrium did not converge within max_iterations.",
        iterations=options.max_iterations,
        max_element_error=max_element_error,
        enthalpy_error=h_error,
        max_correction=max_corr,
        temperature_correction=dlnT_abs,
        residual_norm=current.residual_norm,
    )


def _hp_converged(
    *,
    current: EquilibriumState,
    correction,
    max_element_error: float,
    enthalpy_error: float,
    max_correction: float,
    options: HPSolverOptions,
) -> bool:
    if max_element_error > options.element_tolerance:
        return False

    if abs(enthalpy_error) > options.enthalpy_tolerance:
        return False

    if abs(correction.dln_total_gas_moles) > options.correction_tolerance:
        return False

    if correction.dln_temperature is None:
        return False

    if abs(correction.dln_temperature) > options.temperature_correction_tolerance:
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


def solve_hp_from_scratch(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    pressure: float,
    target_enthalpy: float,
    guess_temperature: float = 3800.0,
    options: HPSolverOptions | None = None,
) -> HPSolverResult:
    if options is None:
        options = HPSolverOptions()

    state = initial_hp_state(
        species=species,
        element_totals=element_totals,
        pressure=pressure,
        guess_temperature=guess_temperature,
        trace=options.trace,
    )

    return solve_hp(
        state,
        target_enthalpy=target_enthalpy,
        options=options,
    )