"""
matrix.py

Reduced CEA/RP-1311 Gibbs-iteration matrix assembly.

This module builds the linear Newton systems used by TP and HP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import SpeciesSet
from .thermo import ThermoArrays, chemical_potentials_over_RT


@dataclass(slots=True)
class MatrixSystem:
    matrix: np.ndarray
    rhs: np.ndarray
    mu_over_RT: np.ndarray
    element_current: np.ndarray
    total_gas_moles: float


@dataclass(slots=True)
class MatrixSolution:
    element_potentials: np.ndarray
    condensed_corrections: np.ndarray
    dln_total_gas_moles: float
    dln_temperature: float | None
    dln_gas_moles: np.ndarray
    raw_solution: np.ndarray


def _split_counts(species: SpeciesSet) -> tuple[int, int, np.ndarray, np.ndarray]:
    gas_idx = np.nonzero(species.gas_mask)[0]
    condensed_idx = np.nonzero(species.condensed_mask)[0]
    return len(gas_idx), len(condensed_idx), gas_idx, condensed_idx


def _safe_gas_mole_fractions(
    n: np.ndarray,
    species: SpeciesSet,
    *,
    trace: float = 1e-300,
) -> tuple[np.ndarray, float]:
    gas_idx = np.nonzero(species.gas_mask)[0]
    ng = np.maximum(n[gas_idx], 0.0)
    ntot = float(np.sum(ng))

    if ntot <= 0.0:
        xg = np.full(len(gas_idx), 1.0 / max(len(gas_idx), 1))
        ntot = trace * max(len(gas_idx), 1)
    else:
        xg = ng / ntot

    return np.maximum(xg, trace), ntot


def build_tp_matrix(
    *,
    species: SpeciesSet,
    n: np.ndarray,
    element_totals: np.ndarray,
    thermo: ThermoArrays,
    pressure: float,
    trace: float = 1e-300,
) -> MatrixSystem:
    """
    Build reduced Gibbs TP system.

    Unknown order:
        [pi_i..., dn_condensed_j..., dln_n]

    where pi_i are dimensionless element multipliers.
    """
    n = np.asarray(n, dtype=float)
    b = np.asarray(element_totals, dtype=float)

    A = species.A
    gas_idx = np.nonzero(species.gas_mask)[0]
    condensed_idx = np.nonzero(species.condensed_mask)[0]

    ne = A.shape[0]
    nc = len(condensed_idx)
    size = ne + nc + 1

    xg, total_gas = _safe_gas_mole_fractions(n, species, trace=trace)

    mu = chemical_potentials_over_RT(
        thermo,
        xg,
        pressure,
        gas_mask=species.gas_mask,
        condensed_mask=species.condensed_mask,
        trace=trace,
    )

    ng = n[gas_idx]
    Ag = A[:, gas_idx]

    element_current = A @ n
    gas_element_current = Ag @ ng

    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    # Eq. 2.24 reduced element equations
    matrix[:ne, :ne] = Ag @ (ng[:, None] * Ag.T)

    if nc:
        Ac = A[:, condensed_idx]
        matrix[:ne, ne:ne + nc] = Ac
        matrix[ne:ne + nc, :ne] = Ac.T

    matrix[:ne, -1] = gas_element_current
    matrix[-1, :ne] = gas_element_current

    rhs[:ne] = (
        b
        - element_current
        + Ag @ (ng * mu[gas_idx])
    )

    # Eq. 2.25 condensed equations
    if nc:
        rhs[ne:ne + nc] = -mu[condensed_idx]

    # Eq. 2.26 total gas mole equation
    matrix[-1, -1] = 0.0
    rhs[-1] = float(np.sum(ng * mu[gas_idx]))

    return MatrixSystem(
        matrix=matrix,
        rhs=rhs,
        mu_over_RT=mu,
        element_current=element_current,
        total_gas_moles=total_gas,
    )


def solve_matrix_system(system: MatrixSystem) -> np.ndarray:
    try:
        return np.linalg.solve(system.matrix, system.rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system.matrix, system.rhs, rcond=None)[0]


def unpack_tp_solution(
    raw: np.ndarray,
    *,
    species: SpeciesSet,
    mu_over_RT: np.ndarray,
) -> MatrixSolution:
    ne = species.nelements
    condensed_idx = np.nonzero(species.condensed_mask)[0]
    gas_idx = np.nonzero(species.gas_mask)[0]
    nc = len(condensed_idx)

    pi = raw[:ne]
    dn_condensed = raw[ne:ne + nc]
    dln_n = float(raw[-1])

    dln_gas = (
        -mu_over_RT[gas_idx]
        + species.A[:, gas_idx].T @ pi
        + dln_n
    )

    return MatrixSolution(
        element_potentials=pi,
        condensed_corrections=dn_condensed,
        dln_total_gas_moles=dln_n,
        dln_temperature=None,
        dln_gas_moles=dln_gas,
        raw_solution=raw,
    )


def build_hp_matrix(
    *,
    species: SpeciesSet,
    n: np.ndarray,
    element_totals: np.ndarray,
    thermo: ThermoArrays,
    pressure: float,
    target_enthalpy: float,
    trace: float = 1e-300,
) -> MatrixSystem:
    """
    Build reduced Gibbs HP system.

    Unknown order:
        [pi_i..., dn_condensed_j..., dln_n, dln_T]
    """
    n = np.asarray(n, dtype=float)
    b = np.asarray(element_totals, dtype=float)

    A = species.A
    gas_idx = np.nonzero(species.gas_mask)[0]
    condensed_idx = np.nonzero(species.condensed_mask)[0]

    ne = A.shape[0]
    nc = len(condensed_idx)
    size = ne + nc + 2

    xg, total_gas = _safe_gas_mole_fractions(n, species, trace=trace)

    mu = chemical_potentials_over_RT(
        thermo,
        xg,
        pressure,
        gas_mask=species.gas_mask,
        condensed_mask=species.condensed_mask,
        trace=trace,
    )

    ng = n[gas_idx]
    Ag = A[:, gas_idx]

    h_RT = thermo.h0_over_RT
    h = thermo.enthalpy_molar
    cp = thermo.specific_heat_cp_molar
    T = thermo.temperature

    element_current = A @ n
    gas_element_current = Ag @ ng

    H = float(np.sum(n * h))
    H_error = float(target_enthalpy - H)

    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    # Eq. 2.24, with temperature column
    matrix[:ne, :ne] = Ag @ (ng[:, None] * Ag.T)

    if nc:
        Ac = A[:, condensed_idx]
        matrix[:ne, ne:ne + nc] = Ac
        matrix[ne:ne + nc, :ne] = Ac.T

    matrix[:ne, ne + nc] = gas_element_current
    matrix[:ne, ne + nc + 1] = Ag @ (ng * h_RT[gas_idx])

    rhs[:ne] = (
        b
        - element_current
        + Ag @ (ng * mu[gas_idx])
    )

    # Eq. 2.25 condensed equations
    if nc:
        matrix[ne:ne + nc, ne + nc + 1] = h_RT[condensed_idx]
        rhs[ne:ne + nc] = -mu[condensed_idx]

    # Eq. 2.26 total gas mole equation
    row_n = ne + nc
    matrix[row_n, :ne] = gas_element_current
    matrix[row_n, ne + nc] = 0.0
    matrix[row_n, ne + nc + 1] = float(np.sum(ng * h_RT[gas_idx]))
    rhs[row_n] = float(np.sum(ng * mu[gas_idx]))

    # Eq. 2.27 enthalpy equation
    row_h = ne + nc + 1

    matrix[row_h, :ne] = A @ (n * h)
    if nc:
        matrix[row_h, ne:ne + nc] = h[condensed_idx]

    matrix[row_h, ne + nc] = H

    matrix[row_h, ne + nc + 1] = float(
        np.sum(n * cp * T)
        + np.sum(n * h * h_RT)
    )

    rhs[row_h] = H_error + float(np.sum(n * h * mu))

    return MatrixSystem(
        matrix=matrix,
        rhs=rhs,
        mu_over_RT=mu,
        element_current=element_current,
        total_gas_moles=total_gas,
    )


def unpack_hp_solution(
    raw: np.ndarray,
    *,
    species: SpeciesSet,
    thermo: ThermoArrays,
    mu_over_RT: np.ndarray,
) -> MatrixSolution:
    ne = species.nelements
    condensed_idx = np.nonzero(species.condensed_mask)[0]
    gas_idx = np.nonzero(species.gas_mask)[0]
    nc = len(condensed_idx)

    pi = raw[:ne]
    dn_condensed = raw[ne:ne + nc]
    dln_n = float(raw[ne + nc])
    dln_T = float(raw[ne + nc + 1])

    dln_gas = (
        -mu_over_RT[gas_idx]
        + species.A[:, gas_idx].T @ pi
        + dln_n
        + thermo.h0_over_RT[gas_idx] * dln_T
    )

    return MatrixSolution(
        element_potentials=pi,
        condensed_corrections=dn_condensed,
        dln_total_gas_moles=dln_n,
        dln_temperature=dln_T,
        dln_gas_moles=dln_gas,
        raw_solution=raw,
    )


def apply_correction(
    *,
    species: SpeciesSet,
    n: np.ndarray,
    correction: MatrixSolution,
    damping: float,
    trace: float = 1e-300,
) -> np.ndarray:
    n_new = np.array(n, dtype=float, copy=True)

    gas_idx = np.nonzero(species.gas_mask)[0]
    condensed_idx = np.nonzero(species.condensed_mask)[0]

    n_new[gas_idx] *= np.exp(
        np.clip(damping * correction.dln_gas_moles, -700.0, 700.0)
    )

    if len(condensed_idx):
        n_new[condensed_idx] += damping * correction.condensed_corrections

    n_new[gas_idx] = np.maximum(n_new[gas_idx], trace)

    return n_new


def cea_damping_factor(
    *,
    species: SpeciesSet,
    n: np.ndarray,
    correction: MatrixSolution,
    size: float = 18.420681,
) -> float:
    """
    CEA-style correction limiter.

    Limits:
    - |dln gas species| <= 2 for significant species
    - |dln total gas moles| <= 0.4
    - |dln T| <= 0.4 when present
    - small species cannot jump above about 1e-4 mole fraction
    """
    gas_idx = np.nonzero(species.gas_mask)[0]
    ng = n[gas_idx]
    total_gas = float(np.sum(ng))

    alpha = 1.0

    if total_gas > 0.0:
        xg = ng / total_gas
        significant = np.log(np.maximum(xg, 1e-300)) > -size
    else:
        significant = np.ones_like(ng, dtype=bool)

    if np.any(significant):
        max_dln = float(np.max(np.abs(correction.dln_gas_moles[significant])))
        if max_dln > 2.0:
            alpha = min(alpha, 2.0 / max_dln)

    dln_n = abs(correction.dln_total_gas_moles)
    if dln_n > 0.4:
        alpha = min(alpha, 0.4 / dln_n)

    if correction.dln_temperature is not None:
        dln_T = abs(correction.dln_temperature)
        if dln_T > 0.4:
            alpha = min(alpha, 0.4 / dln_T)

    if total_gas > 0.0:
        small = ~significant
        grow = correction.dln_gas_moles - correction.dln_total_gas_moles > 0.0
        limited = small & grow

        if np.any(limited):
            x_small = np.maximum(xg[limited], 1e-300)
            numerator = np.log(1e-4) - np.log(x_small)
            denominator = (
                correction.dln_gas_moles[limited]
                - correction.dln_total_gas_moles
            )
            valid = denominator > 0.0
            if np.any(valid):
                alpha = min(alpha, float(np.min(numerator[valid] / denominator[valid])))

    return float(max(0.0, min(1.0, alpha)))


def residual_norm(system: MatrixSystem) -> float:
    return float(np.linalg.norm(system.rhs, ord=np.inf))


def max_species_correction(species: SpeciesSet, correction: MatrixSolution) -> float:
    values = []

    if correction.dln_gas_moles.size:
        values.append(float(np.max(np.abs(correction.dln_gas_moles))))

    if correction.condensed_corrections.size:
        values.append(float(np.max(np.abs(correction.condensed_corrections))))

    if not values:
        return 0.0

    return max(values)