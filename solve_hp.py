from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from EquilibriumState import EquilibriumState


P_REF = 100000.0


@dataclass(frozen=True)
class HPSolveResult:
    success: bool
    message: str
    iterations: int
    max_element_error: float
    enthalpy_error: float
    max_mole_correction: float
    max_total_mole_correction: float
    temperature_correction: float
    state: EquilibriumState


def solve_hp(
    state: EquilibriumState,
    *,
    element_tol: float = 1e-8,
    enthalpy_tol: float = 1e-3,
    correction_tol: float = 1e-8,
    max_iterations: int = 200,
    trace_moles: float = 1e-300,
    min_temperature: float = 200.0,
    max_temperature: float = 20000.0,
    verbose: bool = False,
) -> HPSolveResult:
    """
    CEA-style gas-only HP equilibrium solve.

    Known:
        pressure
        reactant enthalpy
        reactant element amounts

    Unknown:
        product moles
        product temperature

    Mole units:
        mol species / kg reactant mixture
    """
    A = state.A
    b = state.b
    pressure = state.pressure
    target_enthalpy = state.products.reactants.reactant_enthalpy

    ne, ns = A.shape

    solved_state = state.copy()
    n = np.maximum(solved_state.moles.astype(float), trace_moles)
    T = float(solved_state.temperature)

    max_mole_correction = np.inf
    max_total_mole_correction = np.inf
    dlnT = np.inf
    enthalpy_error = np.inf

    message = "maximum iterations exceeded"

    for iteration in range(1, max_iterations + 1):
        T = float(np.clip(T, min_temperature, max_temperature))

        solved_state.temperature = T
        solved_state.products.temperature = T

        g0_RT = solved_state.standard_gibbs_over_RT
        h_kmol = solved_state.products.standard_enthalpies_molar
        cp_kmol = solved_state.products.standard_cps_molar

        h_mol = h_kmol / 1000.0
        cp_mol = cp_kmol / 1000.0
        h_RT = h_kmol / (8.31446261815324e3 * T)

        lnP = np.log(pressure / P_REF)

        ntot = float(np.sum(n))

        if ntot <= 0.0:
            raise RuntimeError("Total moles became nonpositive during HP solve.")

        x = n / ntot

        mu_RT = (
            g0_RT
            + np.log(np.maximum(x, trace_moles))
            + lnP
        )

        element_current = A @ n
        element_error = element_current - b

        mixture_enthalpy = float(np.sum(n * h_mol))
        enthalpy_error = mixture_enthalpy - target_enthalpy

        K = A @ (n[:, None] * A.T)
        c = element_current
        q = A @ (n * h_RT)

        h_element = A @ (n * h_mol)
        h_total = mixture_enthalpy
        h_temperature = float(np.sum(n * h_mol * h_RT) + np.sum(n * cp_mol * T))

        matrix = np.zeros((ne + 2, ne + 2), dtype=float)
        rhs = np.zeros(ne + 2, dtype=float)

        matrix[:ne, :ne] = K
        matrix[:ne, ne] = c
        matrix[:ne, ne + 1] = q

        matrix[ne, :ne] = c
        matrix[ne, ne] = 0.0
        matrix[ne, ne + 1] = float(np.sum(n * h_RT))

        matrix[ne + 1, :ne] = h_element
        matrix[ne + 1, ne] = h_total
        matrix[ne + 1, ne + 1] = h_temperature

        rhs[:ne] = b - element_current + A @ (n * mu_RT)
        rhs[ne] = float(np.sum(n * mu_RT))
        rhs[ne + 1] = target_enthalpy - mixture_enthalpy + float(np.sum(n * h_mol * mu_RT))

        try:
            correction = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

        element_potentials = correction[:ne]
        dln_total_moles = float(correction[ne])
        dlnT = float(correction[ne + 1])

        dln_moles = (
            -mu_RT
            + A.T @ element_potentials
            + dln_total_moles
            + h_RT * dlnT
        )

        max_mole_correction = float(np.max(np.abs(dln_moles)))
        max_total_mole_correction = abs(dln_total_moles)
        temperature_correction = abs(dlnT)

        alpha = 1.0

        if max_mole_correction > 2.0:
            alpha = min(alpha, 2.0 / max_mole_correction)

        if max_total_mole_correction > 0.4:
            alpha = min(alpha, 0.4 / max_total_mole_correction)

        if temperature_correction > 0.2:
            alpha = min(alpha, 0.2 / temperature_correction)

        n = n * np.exp(np.clip(alpha * dln_moles, -700.0, 700.0))
        n = np.maximum(n, trace_moles)

        T = T * np.exp(np.clip(alpha * dlnT, -5.0, 5.0))
        T = float(np.clip(T, min_temperature, max_temperature))

        solved_state.temperature = T
        solved_state.products.temperature = T
        solved_state.update_moles(n)

        max_element_error = solved_state.max_element_error
        enthalpy_error = solved_state.enthalpy - target_enthalpy

        if verbose:
            print(
                f"{iteration:4d} "
                f"alpha={alpha:.3e} "
                f"T={T:.3f} "
                f"max|element error|={max_element_error:.3e} "
                f"h_error={enthalpy_error:.3e} "
                f"max|dln n_j|={max_mole_correction:.3e} "
                f"|dln n|={max_total_mole_correction:.3e} "
                f"|dlnT|={temperature_correction:.3e}"
            )

        if (
            max_element_error < element_tol
            and abs(enthalpy_error) < enthalpy_tol
            and max_mole_correction < correction_tol
            and max_total_mole_correction < correction_tol
            and temperature_correction < correction_tol
        ):
            message = "converged"
            return HPSolveResult(
                success=True,
                message=message,
                iterations=iteration,
                max_element_error=max_element_error,
                enthalpy_error=enthalpy_error,
                max_mole_correction=max_mole_correction,
                max_total_mole_correction=max_total_mole_correction,
                temperature_correction=temperature_correction,
                state=solved_state,
            )

    return HPSolveResult(
        success=False,
        message=message,
        iterations=max_iterations,
        max_element_error=solved_state.max_element_error,
        enthalpy_error=enthalpy_error,
        max_mole_correction=max_mole_correction,
        max_total_mole_correction=max_total_mole_correction,
        temperature_correction=abs(dlnT),
        state=solved_state,
    )