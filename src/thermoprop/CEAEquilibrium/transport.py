"""
transport.py

Frozen and equilibrium transport-property helpers for the CEA-style
equilibrium solver.

The frozen transport path intentionally delegates to ThermoProp's existing
CombustionGas implementation because that class already wraps CEADatabase
species transport fits and CEA-style gas mixture rules.

Equilibrium thermal conductivity adds a reaction-conductivity contribution
using a Brokaw/Svehla-style linear reaction system, matching the structure of
NASA RP-1311 Chapter 5.

All public outputs are SI:
    viscosity              Pa*s
    thermal conductivity   W/m-K
    Prandtl number         dimensionless
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..CEADatabase import CEA
from ..CombustionGas import CombustionGas
from .state import EquilibriumState
from .properties import (
    cp_frozen,
    equilibrium_cp_finite_difference,
    mass_fractions,
)


RU = 8.31446261815324


@dataclass(slots=True)
class TransportOptions:
    trace: float = 1e-12
    max_species: int | None = 50
    equilibrium_derivative_temperature_step: float = 1.0
    include_reaction_conductivity: bool = True


@dataclass(slots=True)
class TransportResult:
    viscosity_frozen: float | None
    conductivity_frozen: float | None
    prandtl_frozen: float | None

    viscosity_equilibrium: float | None
    conductivity_equilibrium: float | None
    prandtl_equilibrium: float | None

    conductivity_reaction: float | None


def transport_composition(
    state: EquilibriumState,
    *,
    trace: float = 1e-12,
    max_species: int | None = 50,
) -> dict[str, float]:
    """
    Gas-only mole-fraction composition for transport.

    CEA transport is only defined for gases. Condensed species are excluded.
    """
    gas_idx = np.nonzero(state.species.gas_mask)[0]
    ng = state.n[gas_idx]
    total = float(np.sum(ng))

    if total <= 0.0:
        return {}

    items: list[tuple[str, float]] = []

    for idx, n_i in zip(gas_idx, ng):
        x = float(n_i / total)
        if x > trace:
            items.append((state.species.names[idx], x))

    items.sort(key=lambda item: item[1], reverse=True)

    if max_species is not None:
        items = items[: int(max_species)]

    total_kept = sum(x for _, x in items)

    if total_kept <= 0.0:
        idx = int(gas_idx[np.argmax(ng)])
        return {state.species.names[idx]: 1.0}

    return {
        name: x / total_kept
        for name, x in items
    }


def make_combustion_gas_for_transport(
    state: EquilibriumState,
    *,
    options: TransportOptions | None = None,
) -> CombustionGas:
    if options is None:
        options = TransportOptions()

    composition = transport_composition(
        state,
        trace=options.trace,
        max_species=options.max_species,
    )

    if not composition:
        raise RuntimeError("Cannot build transport gas; no gas species are present.")

    return CombustionGas(
        composition,
        basis="mole",
        pressure=state.pressure,
        temperature=state.temperature,
    )


def frozen_transport(
    state: EquilibriumState,
    *,
    options: TransportOptions | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Frozen viscosity, conductivity, and Prandtl number.

    Uses CombustionGas's existing CEA transport implementation.
    """
    if options is None:
        options = TransportOptions()

    try:
        gas = make_combustion_gas_for_transport(state, options=options)
    except Exception:
        return None, None, None

    try:
        mu = gas.dynamic_viscosity
    except Exception:
        mu = None

    try:
        k = gas.conductivity
    except Exception:
        k = None

    if mu is None or k is None or k == 0.0:
        pr = None
    else:
        pr = cp_frozen(state) * mu / k

    return mu, k, pr


def _transport_species_arrays(
    state: EquilibriumState,
    *,
    options: TransportOptions,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return names, x, molecular weights [kg/mol], and h [J/mol] for gas species
    used in transport calculations.
    """
    composition = transport_composition(
        state,
        trace=options.trace,
        max_species=options.max_species,
    )

    names = list(composition)
    x = np.array([composition[name] for name in names], dtype=float)
    x = x / np.sum(x)

    M = np.array([CEA.molar_mass(name) for name in names], dtype=float)

    h = np.array(
        [CEA.thermo_molar(name, state.temperature)[1] / 1000.0 for name in names],
        dtype=float,
    )

    return names, x, M, h


def _element_matrix_for_names(
    names: list[str],
    elements: list[str],
) -> np.ndarray:
    A = np.zeros((len(elements), len(names)), dtype=float)

    for j, name in enumerate(names):
        comp = CEA.elemental_composition(name)
        for i, element in enumerate(elements):
            A[i, j] = float(comp.get(element, 0.0))

    return A


def _nullspace(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)

    if singular_values.size == 0:
        rank = 0
    else:
        scale = max(matrix.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tolerance * scale))

    return vh[rank:, :]


def _binary_viscosity_interaction_matrix(
    gas: CombustionGas,
    names: list[str],
    temperature: float,
) -> np.ndarray:
    """
    Get eta_ij interaction matrix.

    Prefer CombustionGas private implementation if available. Otherwise fall
    back to CEADatabase pair interactions when exposed.
    """
    for attr in (
        "_binary_viscosity_interaction_matrix",
        "binary_viscosity_interaction_matrix",
    ):
        if hasattr(gas, attr):
            method = getattr(gas, attr)
            try:
                return np.asarray(method(), dtype=float)
            except TypeError:
                try:
                    return np.asarray(method(names, temperature), dtype=float)
                except Exception:
                    pass
            except Exception:
                pass

    ns = len(names)
    eta = np.zeros((ns, ns), dtype=float)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                try:
                    eta[i, j] = float(CEA.viscosity(ni, temperature))
                except Exception:
                    eta[i, j] = np.nan
                continue

            value = None

            for func_name in (
                "binary_viscosity_interaction",
                "viscosity_interaction",
            ):
                if hasattr(CEA, func_name):
                    try:
                        value = float(getattr(CEA, func_name)(ni, nj, temperature))
                        break
                    except Exception:
                        pass

            if value is None:
                try:
                    mui = float(CEA.viscosity(ni, temperature))
                    muj = float(CEA.viscosity(nj, temperature))
                    Mi = float(CEA.molar_mass(ni))
                    Mj = float(CEA.molar_mass(nj))
                    value = (
                        (1.0 + np.sqrt(mui / muj) * (Mj / Mi) ** 0.25) ** 2
                        / np.sqrt(8.0 * (1.0 + Mi / Mj))
                    )
                except Exception:
                    value = np.nan

            eta[i, j] = value

    return eta


def reaction_conductivity(
    state: EquilibriumState,
    *,
    options: TransportOptions | None = None,
) -> float | None:
    """
    Reaction contribution to equilibrium thermal conductivity [W/m-K].

    This follows the same structure as RP-1311 Eq. 5.8:
        lambda_reaction = R * sum_i(DeltaH_i/RT * z_i)

    where z_i comes from a reaction-diffusion linear system over independent
    gas reactions.

    The independent reaction basis is obtained as the nullspace of the element
    matrix for the gas species retained for transport.
    """
    if options is None:
        options = TransportOptions()

    if not options.include_reaction_conductivity:
        return 0.0

    try:
        gas = make_combustion_gas_for_transport(state, options=options)
        names, x, M, h = _transport_species_arrays(state, options=options)
    except Exception:
        return None

    if len(names) <= 1:
        return 0.0

    elements = [
        e
        for e in state.species.elements
        if e != "E"
    ]

    A = _element_matrix_for_names(names, elements)
    alpha = _nullspace(A)

    nr = alpha.shape[0]

    if nr == 0:
        return 0.0

    eta_ij = _binary_viscosity_interaction_matrix(
        gas,
        names,
        state.temperature,
    )

    T = float(state.temperature)
    RT = RU * T
    astar = 1.1

    x_safe = np.maximum(x, 1e-300)

    G = np.zeros((nr, nr), dtype=float)

    ns = len(names)

    for k in range(ns - 1):
        for l in range(k + 1, ns):
            eta_kl = float(eta_ij[k, l])

            if eta_kl <= 0.0 or not np.isfinite(eta_kl):
                continue

            diffusion_factor = (
                5.0
                * M[k]
                * M[l]
                / (3.0 * astar * eta_kl * (M[k] + M[l]))
            )

            delta = alpha[:, k] / x_safe[k] - alpha[:, l] / x_safe[l]
            G += diffusion_factor * x[k] * x[l] * np.outer(delta, delta)

    delta_h_over_RT = alpha @ h / RT

    try:
        z = np.linalg.solve(G, delta_h_over_RT)
    except np.linalg.LinAlgError:
        try:
            z = np.linalg.lstsq(G, delta_h_over_RT, rcond=None)[0]
        except Exception:
            return None

    value = RU * float(np.dot(delta_h_over_RT, z))

    if not np.isfinite(value):
        return None

    return max(0.0, value)


def equilibrium_transport(
    state: EquilibriumState,
    *,
    tp_neighbor_solver=None,
    options: TransportOptions | None = None,
) -> TransportResult:
    """
    Compute frozen and equilibrium transport properties.

    Viscosity has no reaction contribution in CEA, so equilibrium viscosity is
    the same as frozen viscosity.

    Conductivity equilibrium = frozen conductivity + reaction conductivity.

    Prandtl numbers use frozen/equilibrium Cp consistently.
    """
    if options is None:
        options = TransportOptions()

    mu_f, k_f, pr_f = frozen_transport(state, options=options)

    if k_f is None:
        k_re = None
        k_eq = None
    else:
        k_re = reaction_conductivity(state, options=options)
        if k_re is None:
            k_eq = None
        else:
            k_eq = k_f + k_re

    mu_eq = mu_f

    if tp_neighbor_solver is not None:
        try:
            cp_eq = equilibrium_cp_finite_difference(
                state,
                tp_neighbor_solver=tp_neighbor_solver,
                dT=options.equilibrium_derivative_temperature_step,
            )
        except Exception:
            cp_eq = cp_frozen(state)
    else:
        cp_eq = cp_frozen(state)

    if mu_eq is None or k_eq is None or k_eq == 0.0:
        pr_eq = None
    else:
        pr_eq = cp_eq * mu_eq / k_eq

    return TransportResult(
        viscosity_frozen=mu_f,
        conductivity_frozen=k_f,
        prandtl_frozen=pr_f,
        viscosity_equilibrium=mu_eq,
        conductivity_equilibrium=k_eq,
        prandtl_equilibrium=pr_eq,
        conductivity_reaction=k_re,
    )


def build_transport_values(
    state: EquilibriumState,
    *,
    tp_neighbor_solver=None,
    options: TransportOptions | None = None,
) -> dict[str, float | None]:
    result = equilibrium_transport(
        state,
        tp_neighbor_solver=tp_neighbor_solver,
        options=options,
    )

    return {
        "viscosity_frozen": result.viscosity_frozen,
        "conductivity_frozen": result.conductivity_frozen,
        "prandtl_frozen": result.prandtl_frozen,
        "viscosity_equilibrium": result.viscosity_equilibrium,
        "conductivity_equilibrium": result.conductivity_equilibrium,
        "prandtl_equilibrium": result.prandtl_equilibrium,
        "conductivity_reaction": result.conductivity_reaction,
    }