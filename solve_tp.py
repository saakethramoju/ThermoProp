from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from EquilibriumState import EquilibriumState


@dataclass(frozen=True)
class TPSolveResult:
    success: bool
    message: str
    iterations: int
    max_element_error: float
    max_mole_correction: float
    max_total_mole_correction: float
    state: EquilibriumState


def solve_tp(
    state: EquilibriumState,
    *,
    element_tol: float = 1e-8,
    correction_tol: float = 1e-8,
    max_iterations: int = 200,
    trace_moles: float = 1e-300,
    verbose: bool = False,
) -> TPSolveResult:
    """
    CEA-style gas-only TP equilibrium solve.

    This solves the reduced Gibbs Newton system for assigned temperature and
    pressure. Condensed species are not included yet.

    Mole units are mol species / kg reactant mixture.
    """
    A = state.A
    b = state.b
    g0_RT = state.standard_gibbs_over_RT
    lnP = np.log(state.pressure / 100000.0)

    ne, ns = A.shape

    solved_state = state.copy()
    n = np.maximum(solved_state.moles.astype(float), trace_moles)

    max_mole_correction = np.inf
    max_total_mole_correction = np.inf

    message = "maximum iterations exceeded"

    for iteration in range(1, max_iterations + 1):
        ntot = float(np.sum(n))

        if ntot <= 0.0:
            raise RuntimeError("Total moles became nonpositive during TP solve.")

        x = n / ntot

        mu_RT = (
            g0_RT
            + np.log(np.maximum(x, trace_moles))
            + lnP
        )

        element_current = A @ n
        element_error = element_current - b

        K = A @ (n[:, None] * A.T)
        c = element_current

        matrix = np.zeros((ne + 1, ne + 1), dtype=float)
        rhs = np.zeros(ne + 1, dtype=float)

        matrix[:ne, :ne] = K
        matrix[:ne, ne] = c
        matrix[ne, :ne] = c
        matrix[ne, ne] = 0.0

        rhs[:ne] = b - element_current + A @ (n * mu_RT)
        rhs[ne] = float(np.sum(n * mu_RT))

        try:
            correction = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

        element_potentials = correction[:ne]
        dln_total_moles = float(correction[ne])

        dln_moles = (
            -mu_RT
            + A.T @ element_potentials
            + dln_total_moles
        )

        max_mole_correction = float(np.max(np.abs(dln_moles)))
        max_total_mole_correction = abs(dln_total_moles)

        alpha = 1.0

        if max_mole_correction > 2.0:
            alpha = min(alpha, 2.0 / max_mole_correction)

        if max_total_mole_correction > 0.4:
            alpha = min(alpha, 0.4 / max_total_mole_correction)

        n_new = n * np.exp(np.clip(alpha * dln_moles, -700.0, 700.0))
        n_new = np.maximum(n_new, trace_moles)

        solved_state.update_moles(n_new)

        max_element_error = solved_state.max_element_error

        if verbose:
            print(
                f"{iteration:4d} "
                f"alpha={alpha:.3e} "
                f"max|element error|={max_element_error:.3e} "
                f"max|dln n_j|={max_mole_correction:.3e} "
                f"|dln n|={max_total_mole_correction:.3e}"
            )

        n = n_new

        if (
            max_element_error < element_tol
            and max_mole_correction < correction_tol
            and max_total_mole_correction < correction_tol
        ):
            message = "converged"
            return TPSolveResult(
                success=True,
                message=message,
                iterations=iteration,
                max_element_error=max_element_error,
                max_mole_correction=max_mole_correction,
                max_total_mole_correction=max_total_mole_correction,
                state=solved_state,
            )

    solved_state.update_moles(n)

    return TPSolveResult(
        success=False,
        message=message,
        iterations=max_iterations,
        max_element_error=solved_state.max_element_error,
        max_mole_correction=max_mole_correction,
        max_total_mole_correction=max_total_mole_correction,
        state=solved_state,
    )