"""
properties.py

Mixture thermodynamic properties for the CEA-style equilibrium solver.

This module evaluates final equilibrium/frozen thermodynamic properties from an
EquilibriumState. It does not solve the equilibrium composition.

Internal convention
-------------------
Species amounts:
    n[j] = kmol species j / kg mixture

ThermoArrays:
    h, cp, s are J/kmol, J/kmol-K, J/kmol-K

Public properties:
    SI mass basis, e.g. J/kg, J/kg-K, kg/m^3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import EquilibriumState, EquilibriumResults
from .thermo import (
    RU_KMOL,
    thermo_arrays_for_species_set,
    mixture_enthalpy,
    mixture_entropy,
    mixture_internal_energy,
    mixture_gibbs_energy,
    frozen_specific_heat_cp,
    frozen_specific_heat_cv,
    density_ideal_mixture,
    specific_volume_ideal_mixture,
    mixture_gas_constant,
    gas_molecular_weight,
    mass_fraction_dict,
    mole_fraction_dict,
    gas_mole_fraction_dict,
)


@dataclass(slots=True)
class MixtureDerivatives:
    """First-derivative set used for gamma and speed of sound."""
    dlnv_dlnT_const_p: float
    dlnv_dlnp_const_T: float
    dlnv_dlnp_const_s: float
    gamma_s: float


def frozen_mixture_derivatives(state: EquilibriumState) -> MixtureDerivatives:
    """
    Frozen ideal-mixture derivatives.

    For frozen ideal gas composition:
        (d ln V / d ln T)_P = 1
        (d ln V / d ln P)_T = -1
    """
    gamma = frozen_gamma(state)

    return MixtureDerivatives(
        dlnv_dlnT_const_p=1.0,
        dlnv_dlnp_const_T=-1.0,
        dlnv_dlnp_const_s=-1.0 / gamma,
        gamma_s=gamma,
    )


def finite_difference_equilibrium_derivatives(
    state: EquilibriumState,
    *,
    tp_neighbor_solver,
    dT: float = 1.0,
    dP_fraction: float = 1e-4,
) -> MixtureDerivatives:
    """
    Numerical equilibrium derivatives.

    Parameters
    ----------
    state:
        Base equilibrium state.

    tp_neighbor_solver:
        Callable with signature:
            tp_neighbor_solver(base_state, temperature=None, pressure=None)
        returning an EquilibriumState re-equilibrated at TP.

    dT:
        Temperature perturbation [K].

    dP_fraction:
        Relative pressure perturbation.

    Notes
    -----
    This is a practical fallback. A stricter CEA clone should replace this with
    the analytical derivative systems from RP-1311 section 2.5.
    """
    T0 = state.temperature
    P0 = state.pressure

    plus_T = tp_neighbor_solver(
        state,
        temperature=T0 + dT,
        pressure=P0,
    )
    minus_T = tp_neighbor_solver(
        state,
        temperature=max(T0 - dT, 1e-6),
        pressure=P0,
    )

    dlnv_dlnT = (
        np.log(specific_volume(plus_T))
        - np.log(specific_volume(minus_T))
    ) / (
        np.log(plus_T.temperature)
        - np.log(minus_T.temperature)
    )

    plus_P = tp_neighbor_solver(
        state,
        temperature=T0,
        pressure=P0 * (1.0 + dP_fraction),
    )
    minus_P = tp_neighbor_solver(
        state,
        temperature=T0,
        pressure=P0 * (1.0 - dP_fraction),
    )

    dlnv_dlnp_T = (
        np.log(specific_volume(plus_P))
        - np.log(specific_volume(minus_P))
    ) / (
        np.log(plus_P.pressure)
        - np.log(minus_P.pressure)
    )

    cp_eq = equilibrium_cp_finite_difference(
        state,
        tp_neighbor_solver=tp_neighbor_solver,
        dT=dT,
    )

    Rmix = gas_constant(state)

    dlnv_dlnp_s = dlnv_dlnp_T + Rmix * dlnv_dlnT**2 / cp_eq
    gamma_s = -1.0 / dlnv_dlnp_s

    return MixtureDerivatives(
        dlnv_dlnT_const_p=float(dlnv_dlnT),
        dlnv_dlnp_const_T=float(dlnv_dlnp_T),
        dlnv_dlnp_const_s=float(dlnv_dlnp_s),
        gamma_s=float(gamma_s),
    )


def enthalpy(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return mixture_enthalpy(state.n, thermo)


def entropy(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return mixture_entropy(
        state.n,
        thermo,
        state.pressure,
        gas_mask=state.species.gas_mask,
    )


def internal_energy(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return mixture_internal_energy(
        state.n,
        thermo,
        gas_mask=state.species.gas_mask,
    )


def gibbs_energy(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return mixture_gibbs_energy(
        state.n,
        thermo,
        state.pressure,
        gas_mask=state.species.gas_mask,
    )


def helmholtz_energy(state: EquilibriumState) -> float:
    return internal_energy(state) - state.temperature * entropy(state)


def density(state: EquilibriumState) -> float:
    return density_ideal_mixture(
        state.n,
        state.temperature,
        state.pressure,
        gas_mask=state.species.gas_mask,
    )


def specific_volume(state: EquilibriumState) -> float:
    return specific_volume_ideal_mixture(
        state.n,
        state.temperature,
        state.pressure,
        gas_mask=state.species.gas_mask,
    )


def gas_constant(state: EquilibriumState) -> float:
    return mixture_gas_constant(
        state.n,
        gas_mask=state.species.gas_mask,
    )


def molecular_weight(state: EquilibriumState) -> float:
    """
    Gas molecular weight [kg/kmol].

    This follows CEA's gas-EOS molecular weight convention. Condensed species
    affect total mixture mass but not gas mole count.
    """
    return gas_molecular_weight(
        state.n,
        state.species.molecular_weights,
        gas_mask=state.species.gas_mask,
    )


def molecular_weight_all_species(state: EquilibriumState) -> float:
    """
    Conventional all-species mole-weighted molecular weight [kg/kmol].
    """
    total = float(np.sum(state.n))

    if total <= 0.0:
        return np.nan

    mw_kg_per_kmol = state.species.molecular_weights * 1000.0
    return float(np.sum(state.n * mw_kg_per_kmol) / total)


def cp_frozen(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return frozen_specific_heat_cp(state.n, thermo)


def cv_frozen(state: EquilibriumState) -> float:
    thermo = thermo_arrays_for_species_set(state.species, state.temperature)
    return frozen_specific_heat_cv(
        state.n,
        thermo,
        gas_mask=state.species.gas_mask,
    )


def frozen_gamma(state: EquilibriumState) -> float:
    cp = cp_frozen(state)
    cv = cv_frozen(state)

    if cv <= 0.0:
        return np.nan

    return float(cp / cv)


def speed_of_sound_frozen(state: EquilibriumState) -> float:
    gamma = frozen_gamma(state)
    R = gas_constant(state)

    if gamma <= 0.0 or R <= 0.0:
        return np.nan

    return float(np.sqrt(gamma * R * state.temperature))


def equilibrium_cp_finite_difference(
    state: EquilibriumState,
    *,
    tp_neighbor_solver,
    dT: float = 1.0,
) -> float:
    """
    Equilibrium Cp [J/kg-K] from TP finite difference.

    Cp_eq = (dh/dT)_P with composition re-equilibrated.

    This is numerically robust but not as exact as CEA's analytical derivative
    matrix. It can be replaced later without changing the public API.
    """
    T0 = state.temperature
    P0 = state.pressure

    plus = tp_neighbor_solver(
        state,
        temperature=T0 + dT,
        pressure=P0,
    )

    if T0 - dT > 1e-6:
        minus = tp_neighbor_solver(
            state,
            temperature=T0 - dT,
            pressure=P0,
        )
        return float((enthalpy(plus) - enthalpy(minus)) / (2.0 * dT))

    return float((enthalpy(plus) - enthalpy(state)) / dT)


def equilibrium_cv_from_derivatives(
    state: EquilibriumState,
    *,
    cp_equilibrium: float,
    derivatives: MixtureDerivatives,
) -> float:
    """
    Compute Cv_eq from Cp_eq and log-volume derivatives.

    Based on RP-1311 thermodynamic derivative relations.
    """
    Rmix = gas_constant(state)

    denom = -derivatives.dlnv_dlnp_const_T

    if denom <= 0.0:
        return np.nan

    cv = cp_equilibrium + (
        Rmix
        * derivatives.dlnv_dlnT_const_p**2
        / derivatives.dlnv_dlnp_const_T
    )

    return float(cv)


def equilibrium_gamma_from_derivatives(
    derivatives: MixtureDerivatives,
) -> float:
    return derivatives.gamma_s


def speed_of_sound_equilibrium(
    state: EquilibriumState,
    *,
    gamma_equilibrium: float,
) -> float:
    R = gas_constant(state)

    if gamma_equilibrium <= 0.0 or R <= 0.0:
        return np.nan

    return float(np.sqrt(gamma_equilibrium * R * state.temperature))


def mole_fractions(
    state: EquilibriumState,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    return mole_fraction_dict(
        state.species.names,
        state.n,
        trace=trace,
    )


def gas_mole_fractions(
    state: EquilibriumState,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    return gas_mole_fraction_dict(
        state.species.names,
        state.n,
        gas_mask=state.species.gas_mask,
        trace=trace,
    )


def mass_fractions(
    state: EquilibriumState,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    return mass_fraction_dict(
        state.species.names,
        state.n,
        state.species.molecular_weights,
        trace=trace,
    )


def gas_mass_fractions(
    state: EquilibriumState,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    names = [
        name
        for name, gas in zip(state.species.names, state.species.gas_mask)
        if gas
    ]
    n = state.n[state.species.gas_mask]
    mw = state.species.molecular_weights[state.species.gas_mask]
    return mass_fraction_dict(
        names,
        n,
        mw,
        trace=trace,
    )


def condensed_mass_fraction(state: EquilibriumState) -> float:
    y = mass_fraction_dict(
        state.species.names,
        state.n,
        state.species.molecular_weights,
        trace=0.0,
    )

    total = 0.0
    for name, is_condensed in zip(state.species.names, state.species.condensed_mask):
        if is_condensed:
            total += y.get(name, 0.0)

    return float(total)


def build_results(
    state: EquilibriumState,
    *,
    tp_neighbor_solver=None,
    equilibrium_derivative_step: float = 1.0,
    transport_values: dict[str, float | None] | None = None,
) -> EquilibriumResults:
    """
    Build an EquilibriumResults object from a converged state.

    If tp_neighbor_solver is provided, equilibrium Cp/Cv/gamma are evaluated.
    Otherwise equilibrium values fall back to frozen values.
    """
    h = enthalpy(state)
    s = entropy(state)
    u = internal_energy(state)
    rho = density(state)

    cpf = cp_frozen(state)
    cvf = cv_frozen(state)
    gammaf = cpf / cvf if cvf > 0.0 else np.nan

    if tp_neighbor_solver is not None:
        cpe = equilibrium_cp_finite_difference(
            state,
            tp_neighbor_solver=tp_neighbor_solver,
            dT=max(
                0.05,
                1e-4 * state.temperature,
            )
        )
        derivs = finite_difference_equilibrium_derivatives(
            state,
            tp_neighbor_solver=tp_neighbor_solver,
            dT=max(
                0.05,
                1e-4 * state.temperature,
            )
        )
        cve = equilibrium_cv_from_derivatives(
            state,
            cp_equilibrium=cpe,
            derivatives=derivs,
        )
        gammae = equilibrium_gamma_from_derivatives(derivs)
    else:
        cpe = cpf
        cve = cvf
        gammae = gammaf

    transport_values = transport_values or {}

    return EquilibriumResults(
        state=state,
        enthalpy=h,
        entropy=s,
        internal_energy=u,
        density=rho,
        cp_frozen=cpf,
        cv_frozen=cvf,
        cp_equilibrium=cpe,
        cv_equilibrium=cve,
        gamma_frozen=gammaf,
        gamma_equilibrium=gammae,
        cp_transport_frozen=transport_values.get("cp_transport_frozen"),
        cp_transport_equilibrium=transport_values.get("cp_transport_equilibrium"),
        viscosity_frozen=transport_values.get("viscosity_frozen"),
        conductivity_frozen=transport_values.get("conductivity_frozen"),
        prandtl_frozen=transport_values.get("prandtl_frozen"),
        viscosity_equilibrium=transport_values.get("viscosity_equilibrium"),
        conductivity_equilibrium=transport_values.get("conductivity_equilibrium"),
        prandtl_equilibrium=transport_values.get("prandtl_equilibrium"),
        conductivity_reaction=transport_values.get("conductivity_reaction"),
    )


def has_condensed_species(state: EquilibriumState) -> bool:
    condensed_idx = np.nonzero(state.species.condensed_mask)[0]
    if len(condensed_idx) == 0:
        return False
    return bool(np.any(state.n[condensed_idx] > 0.0))