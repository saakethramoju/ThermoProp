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

from .thermo import thermo_arrays_for_species_set, RU_KMOL

RU = 8.31446261815324


@dataclass(slots=True)
class TransportOptions:
    """Options for CEA-style frozen/equilibrium transport evaluation."""
    trace: float = 1e-12
    max_species: int | None = None #50
    equilibrium_derivative_temperature_step: float = 1.0
    include_reaction_conductivity: bool = True


@dataclass(slots=True)
class TransportResult:
    """Transport-property values returned by the equilibrium wrapper."""
    cp_frozen: float | None
    viscosity_frozen: float | None
    conductivity_frozen: float | None
    prandtl_frozen: float | None

    cp_equilibrium: float | None
    viscosity_equilibrium: float | None
    conductivity_equilibrium: float | None
    prandtl_equilibrium: float | None

    conductivity_reaction: float | None


def transport_composition(
    state: EquilibriumState,
    *,
    trace: float = 1e-12,
    max_species: int | None = None,
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

def gas_mass_fraction(state: EquilibriumState) -> float:
    gas_mask = state.species.gas_mask
    n = state.n
    mw = state.species.molecular_weights * 1000.0  # kg/kmol

    gas_mass = float(np.sum(n[gas_mask] * mw[gas_mask]))
    total_mass = float(np.sum(n * mw))

    if total_mass <= 0.0:
        return 1.0

    return gas_mass / total_mass

def gas_only_cp_frozen(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    gas_mask = state.species.gas_mask

    cp_per_kg_total = float(
        np.sum(state.n[gas_mask] * thermo.specific_heat_cp_molar[gas_mask])
    )

    return cp_per_kg_total / gas_mass_fraction(state)

def gas_only_cp_reaction(
    state: EquilibriumState,
    *,
    options: TransportOptions | None = None,
) -> float:
    if options is None:
        options = TransportOptions()

    try:
        names, x, M, h = _transport_species_arrays(state, options=options)
    except Exception:
        return 0.0

    if len(names) <= 1:
        return 0.0

    elements = [
        e for e in state.species.elements
        if e != "E"
    ]

    A = _element_matrix_for_names(names, elements)
    alpha = _nullspace(A)

    nr = alpha.shape[0]

    if nr == 0:
        return 0.0

    x_safe = np.maximum(x, 1e-300)

    D = np.zeros((nr, nr), dtype=float)

    ns = len(names)

    for k in range(ns - 1):
        for l in range(k + 1, ns):
            delta = alpha[:, k] / x_safe[k] - alpha[:, l] / x_safe[l]
            D += x[k] * x[l] * np.outer(delta, delta)

    T = float(state.temperature)
    delta_h_over_RT = alpha @ h / (RU * T)

    try:
        X = np.linalg.solve(D, delta_h_over_RT)
    except np.linalg.LinAlgError:
        try:
            X = np.linalg.lstsq(D, delta_h_over_RT, rcond=None)[0]
        except Exception:
            return 0.0

    Rmix_total_basis = float(state.total_gas_moles) * RU_KMOL
    value_total_basis = Rmix_total_basis * float(np.dot(delta_h_over_RT, X))
    value = value_total_basis / gas_mass_fraction(state)

    if not np.isfinite(value):
        return 0.0

    return max(0.0, value)



def _estimated_viscosity(name: str, T: float) -> float:
    M = CEA.molecular_weight(name)
    omega = np.log(50.0 * M**4.6 / T**1.4)
    omega = max(float(omega), 1.0)
    return float(2.67e-8 * np.sqrt(M * T) / omega)


def _estimated_conductivity(name: str, T: float, viscosity: float) -> float:
    M = CEA.molecular_weight(name)
    cp_molar = CEA.thermo_molar(name, T)[0]
    cp_over_R = cp_molar / RU_KMOL

    return float(
        viscosity
        * RU_KMOL
        * (0.00375 + 0.00132 * (cp_over_R - 2.5))
        / M
    )


def _estimated_binary_eta(i: int, j: int, names: list[str], mu_i: np.ndarray) -> float:
    Mi = CEA.molecular_weight(names[i])
    Mj = CEA.molecular_weight(names[j])
    etai = float(mu_i[i])
    etaj = float(mu_i[j])

    ratio = np.sqrt(Mj / Mi)

    etaij = 5.656854 * etai * np.sqrt(Mj / (Mi + Mj))
    etaij = etaij / (1.0 + np.sqrt(ratio * etai / etaj))**2

    return float(etaij)


def _cea_mix(x: np.ndarray, values: np.ndarray, interaction: np.ndarray) -> float:
    total = 0.0
    ns = len(values)

    for i in range(ns):
        denom = x[i]

        for j in range(ns):
            if i == j:
                continue
            denom += x[j] * interaction[i, j]

        total += x[i] * values[i] / denom

    return float(total)


def frozen_transport(
    state: EquilibriumState,
    *,
    options: TransportOptions | None = None,
) -> tuple[float | None, float | None, float | None]:
    if options is None:
        options = TransportOptions()

    composition = transport_composition(
        state,
        trace=options.trace,
        max_species=options.max_species,
    )

    if not composition:
        return None, None, None

    names = list(composition)
    x = np.array([composition[name] for name in names], dtype=float)
    x = x / np.sum(x)

    T = float(state.temperature)
    ns = len(names)

    mu_i = np.zeros(ns, dtype=float)
    k_i = np.zeros(ns, dtype=float)
    M = np.array([CEA.molecular_weight(name) for name in names], dtype=float)

    for i, name in enumerate(names):
        try:
            mu_i[i] = CEA.viscosity(name, T)
        except Exception:
            mu_i[i] = _estimated_viscosity(name, T)

        try:
            k_i[i] = CEA.conductivity(name, T)
        except Exception:
            k_i[i] = _estimated_conductivity(name, T, mu_i[i])

    if (
        np.any(~np.isfinite(mu_i))
        or np.any(~np.isfinite(k_i))
        or np.any(mu_i <= 0.0)
        or np.any(k_i <= 0.0)
    ):
        return None, None, None

    eta_ij = np.zeros((ns, ns), dtype=float)

    for i in range(ns):
        for j in range(ns):
            if i == j:
                eta_ij[i, j] = mu_i[i]
                continue

            try:
                eta_ij[i, j] = CEA.binary_viscosity_interaction(
                    names[i],
                    names[j],
                    T,
                )
            except Exception:
                eta_ij[i, j] = _estimated_binary_eta(i, j, names, mu_i)

    M_i = M[:, None]
    M_j = M[None, :]

    phi = 2.0 * M_j * mu_i[:, None] / (eta_ij * (M_i + M_j))
    np.fill_diagonal(phi, 0.0)

    psi = phi * (
        1.0
        + 2.41
        * (M_i - M_j)
        * (M_i - 0.142 * M_j)
        / (M_i + M_j) ** 2
    )
    np.fill_diagonal(psi, 0.0)

    mu = _cea_mix(x, mu_i, phi)
    k = _cea_mix(x, k_i, psi)

    cp_transport_frozen = gas_only_cp_frozen(state)

    if k <= 0.0 or not np.isfinite(k):
        pr = None
    else:
        pr = cp_transport_frozen * mu / k

    return float(mu), float(k), pr



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
    gas,
    names: list[str],
    temperature: float,
) -> np.ndarray:
    """
    Get eta_ij interaction matrix.

    Prefer CombustionGas private implementation if available. Otherwise fall
    back to CEADatabase pair interactions when exposed.
    """
    if gas is not None:
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
        gas = make_combustion_gas_for_transport(
            state,
            options=options,
        )

        names, x, M, h = _transport_species_arrays(
            state,
            options=options,
        )
    except Exception:
        return 0.0

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

    cp_fr_transport = gas_only_cp_frozen(state)
    cp_eq_transport = cp_fr_transport + gas_only_cp_reaction(
        state,
        options=options,
    )

    if k_f is None:
        k_re = None
        k_eq = None
    else:
        k_re = reaction_conductivity(state, options=options)
        if k_re is None:
            k_re = 0.0

        k_eq = k_f + k_re

    mu_eq = mu_f

    cp_re_transport = gas_only_cp_reaction(state, options=options)
    cp_eq = cp_fr_transport + cp_re_transport

    if mu_f is None or k_f is None or k_f == 0.0:
        pr_f = None
    else:
        pr_f = cp_fr_transport * mu_f / k_f

    if mu_eq is None or k_eq is None or k_eq == 0.0:
        pr_eq = None
    else:
        pr_eq = cp_eq_transport * mu_eq / k_eq

    return TransportResult(
        cp_frozen=cp_fr_transport,
        cp_equilibrium=cp_eq_transport,
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
    empty = {
        "cp_transport_frozen": None,
        "viscosity_frozen": None,
        "conductivity_frozen": None,
        "prandtl_frozen": None,
        "cp_transport_equilibrium": None,
        "viscosity_equilibrium": None,
        "conductivity_equilibrium": None,
        "prandtl_equilibrium": None,
        "conductivity_reaction": None,
    }

    try:
        result = equilibrium_transport(
            state,
            tp_neighbor_solver=tp_neighbor_solver,
            options=options,
        )
    except Exception:
        return empty

    return {
        "cp_transport_frozen": result.cp_frozen,
        "viscosity_frozen": result.viscosity_frozen,
        "conductivity_frozen": result.conductivity_frozen,
        "prandtl_frozen": result.prandtl_frozen,
        "cp_transport_equilibrium": result.cp_equilibrium,
        "viscosity_equilibrium": result.viscosity_equilibrium,
        "conductivity_equilibrium": result.conductivity_equilibrium,
        "prandtl_equilibrium": result.prandtl_equilibrium,
        "conductivity_reaction": result.conductivity_reaction,
    }
