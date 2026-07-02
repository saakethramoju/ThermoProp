"""
sp_solver.py

Native constant-entropy / constant-pressure equilibrium solver.

This module solves the CEA/RP-1311 reduced Gibbs SP problem for a fixed
active species set.  It is the constant-entropy sibling of the HP solver:
composition and temperature are corrected simultaneously in one Newton matrix
instead of wrapping TP equilibrium in an outer temperature root.

Condensed-phase insertion/removal is handled outside this solver by
condensed.py. This solver assumes the supplied SpeciesSet is the active set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Exceptions import EquilibriumSetupError
from .state import EquilibriumState, SpeciesSet
from .thermo import (
    thermo_arrays_for_species_set,
    mixture_entropy,
)
from .matrix import (
    build_sp_matrix,
    solve_matrix_system,
    unpack_sp_solution,
    apply_correction,
    cea_damping_factor,
    residual_norm,
    max_species_correction,
)


@dataclass(slots=True)
class SPSolverOptions:
    """Numerical controls for native fixed-entropy, fixed-pressure solves.

    Entropy is specified on ThermoProp's public mass basis, J/kg-K.  Internally
    the SP matrix uses CEA's gas-constant-normalized kmol/kg basis, but the
    conversion is local to the matrix builder.
    """

    max_iterations: int = 120
    max_bracket_iterations: int = 80

    trace: float = 1e-300
    species_trace: float = 1e-12

    element_tolerance: float = 1e-8
    entropy_tolerance: float = 1e-3
    correction_tolerance: float = 5e-6
    temperature_correction_tolerance: float = 5e-4

    min_temperature: float = 100.0
    max_temperature: float = 20000.0

    size: float = 18.420681
    verbose: bool = False


@dataclass(slots=True)
class SPSolverResult:
    """Result bundle returned by the fixed-species SP solver."""

    state: EquilibriumState
    success: bool
    message: str
    iterations: int
    max_element_error: float
    entropy_error: float
    max_correction: float
    temperature_correction: float
    residual_norm: float
    element_potentials: np.ndarray | None = None


def initial_sp_state(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    pressure: float,
    guess_temperature: float = 3800.0,
    total_gas_moles_guess: float | None = None,
    trace: float = 1e-300,
) -> EquilibriumState:
    """Create a crude initial state for native SP equilibrium.

    The structure is intentionally identical to the HP initial state.  SP and
    HP are both constant-pressure thermal solves with unknown temperature, so
    the same CEA-style gas mole estimate is appropriate.
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


def solve_sp(
    state: EquilibriumState,
    *,
    target_entropy: float,
    options: SPSolverOptions | None = None,
) -> SPSolverResult:
    """Solve native SP equilibrium for the active SpeciesSet.

    Parameters
    ----------
    state:
        Initial equilibrium state.
    target_entropy:
        Assigned mixture entropy [J/kg-K].
    options:
        Numerical controls.

    Returns
    -------
    SPSolverResult
    """
    if options is None:
        options = SPSolverOptions()

    if not np.isfinite(target_entropy):
        raise EquilibriumSetupError("SP solver requires a finite target entropy [J/kg-K].")

    current = state.copy()
    species = current.species

    gas_idx = np.nonzero(species.gas_mask)[0]

    if len(gas_idx) == 0:
        raise EquilibriumSetupError("SP solver requires at least one gas species.")

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
    s_error = np.inf
    max_element_error = np.inf
    correction = None

    for iteration in range(1, options.max_iterations + 1):
        T = float(np.clip(T, options.min_temperature, options.max_temperature))
        current.temperature = T

        thermo = thermo_arrays_for_species_set(
            species,
            current.temperature,
            on_error="raise",
        )

        system = build_sp_matrix(
            species=species,
            n=n,
            element_totals=current.element_totals,
            thermo=thermo,
            pressure=current.pressure,
            target_entropy=float(target_entropy),
            trace=options.trace,
        )

        raw = solve_matrix_system(system)

        correction = unpack_sp_solution(
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
        T_new_raw = T * np.exp(np.clip(damping * dlnT, -5.0, 5.0))

        if not np.isfinite(T_new_raw):
            T_new_raw = T

        T_new = float(
            np.clip(
                T_new_raw,
                options.min_temperature,
                options.max_temperature,
            )
        )

        current.n = n_new
        current.temperature = T_new
        current.total_gas_moles = float(np.sum(n_new[gas_idx]))
        current.iteration = iteration
        current.residual_norm = residual_norm(system, raw)

        try:
            thermo_new = thermo_arrays_for_species_set(
                species,
                current.temperature,
                on_error="raise",
            )
        except Exception as exc:
            current.converged = False
            return SPSolverResult(
                state=current,
                success=False,
                message=f"SP active species thermo invalid at updated temperature: {exc}",
                iterations=iteration,
                max_element_error=np.inf,
                entropy_error=np.inf,
                max_correction=max_corr,
                temperature_correction=dlnT_abs,
                residual_norm=current.residual_norm,
                element_potentials=correction.element_potentials.copy(),
            )

        s_products = mixture_entropy(
            n_new,
            thermo_new,
            current.pressure,
            gas_mask=species.gas_mask,
            trace=options.trace,
        )
        s_error = float(target_entropy) - s_products

        element_error = species.A @ n_new - current.element_totals
        max_element_error = float(np.max(np.abs(element_error)))

        max_corr = max(
            max_species_correction(species, correction),
            abs(correction.dln_total_gas_moles),
        )

        dlnT_abs = abs(dlnT)

        if options.verbose:
            print(
                f"SP {iteration:3d} "
                f"alpha={damping:.3e} "
                f"T={current.temperature:.3f} "
                f"elem={max_element_error:.3e} "
                f"s_err={s_error:.3e} "
                f"corr={max_corr:.3e} "
                f"dlnT={dlnT:.3e} "
                f"ngas={current.total_gas_moles:.6e}"
            )

        n = n_new
        T = T_new

        if _sp_converged(
            current=current,
            correction=correction,
            max_element_error=max_element_error,
            entropy_error=s_error,
            max_correction=max_corr,
            options=options,
        ):
            current.converged = True
            return SPSolverResult(
                state=current,
                success=True,
                message="SP equilibrium converged.",
                iterations=iteration,
                max_element_error=max_element_error,
                entropy_error=s_error,
                max_correction=max_corr,
                temperature_correction=dlnT_abs,
                residual_norm=current.residual_norm,
                element_potentials=correction.element_potentials.copy(),
            )

    current.converged = False

    if options.verbose:
        print(
            "SP FAILED",
            "T =", current.temperature,
            "max_elem =", max_element_error,
            "s_err =", s_error,
            "corr =", max_corr,
        )

    return SPSolverResult(
        state=current,
        success=False,
        message="SP equilibrium did not converge within max_iterations.",
        iterations=options.max_iterations,
        max_element_error=max_element_error,
        entropy_error=s_error,
        max_correction=max_corr,
        temperature_correction=dlnT_abs,
        residual_norm=current.residual_norm,
        element_potentials=correction.element_potentials.copy() if correction is not None else None,
    )


def _sp_converged(
    *,
    current: EquilibriumState,
    correction,
    max_element_error: float,
    entropy_error: float,
    max_correction: float,
    options: SPSolverOptions,
) -> bool:
    if max_element_error > options.element_tolerance:
        return False

    if current.residual_norm > 1e-10:
        return False

    if abs(entropy_error) > options.entropy_tolerance:
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


def solve_sp_from_scratch(
    *,
    species: SpeciesSet,
    element_totals: np.ndarray,
    pressure: float,
    target_entropy: float,
    guess_temperature: float = 3800.0,
    options: SPSolverOptions | None = None,
) -> SPSolverResult:
    if options is None:
        options = SPSolverOptions()

    state = initial_sp_state(
        species=species,
        element_totals=element_totals,
        pressure=pressure,
        guess_temperature=guess_temperature,
        trace=options.trace,
    )

    return solve_sp(
        state,
        target_entropy=target_entropy,
        options=options,
    )
