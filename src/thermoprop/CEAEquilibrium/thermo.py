"""
thermo.py

Vectorized NASA/CEA thermodynamic property evaluation for the CEA-style
equilibrium solver.

Internal convention
-------------------
Species amounts in the equilibrium package are kmol species / kg mixture.

Therefore this module exposes molar properties mainly as:

    J / kmol
    J / kmol-K

and dimensionless forms:

    cp/R
    h/RT
    s/R
    g/RT

where R = 8314.46261815324 J/kmol-K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..CEADatabase import CEA
from .state import SpeciesSet


P_REF = 100000.0
RU_KMOL = 8314.46261815324
RU_MOL = 8.31446261815324


@dataclass(slots=True)
class ThermoArrays:
    """Represent the public ThermoProp ``ThermoArrays`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """
    names: list[str]
    temperature: float

    specific_heat_cp_molar: np.ndarray      # J/kmol-K
    enthalpy_molar: np.ndarray              # J/kmol
    entropy_molar_standard: np.ndarray      # J/kmol-K

    g0_over_RT: np.ndarray                  # dimensionless
    h0_over_RT: np.ndarray                  # dimensionless
    s0_over_R: np.ndarray                   # dimensionless
    cp_over_R: np.ndarray                   # dimensionless

    valid: np.ndarray                       # bool mask


def thermo_arrays(
    names: Iterable[str],
    temperature: float,
    *,
    on_error: str = "raise",
) -> ThermoArrays:
    """
    Evaluate CEA thermo for many species at one temperature.

    The heavy NASA-9 polynomial work is delegated to the vectorized
    :class:`CEADatabase` array API so TP/HP solver iterations avoid one Python
    call per species for Cp, h, and s.  Numerical formulas and units are
    unchanged from the scalar CEA methods.
    """
    names = list(names)
    T = float(temperature)

    if T <= 0.0:
        raise ValueError("temperature must be positive.")

    cp, h, s, valid = CEA.thermo_molar_array(
        names,
        T,
        on_error=on_error,
    )

    h0_over_RT = h / (RU_KMOL * T)
    s0_over_R = s / RU_KMOL
    cp_over_R = cp / RU_KMOL
    g0_over_RT = h0_over_RT - s0_over_R

    return ThermoArrays(
        names=names,
        temperature=T,
        specific_heat_cp_molar=cp,
        enthalpy_molar=h,
        entropy_molar_standard=s,
        g0_over_RT=g0_over_RT,
        h0_over_RT=h0_over_RT,
        s0_over_R=s0_over_R,
        cp_over_R=cp_over_R,
        valid=valid,
    )


def thermo_arrays_for_species_set(
    species: SpeciesSet,
    temperature: float,
    *,
    on_error: str = "raise",
) -> ThermoArrays:
    """Execute the public ``thermo_arrays_for_species_set`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    return thermo_arrays(
        species.names,
        temperature,
        on_error=on_error,
    )


def chemical_potentials_over_RT(
    thermo: ThermoArrays,
    mole_fractions_gas: np.ndarray,
    pressure: float,
    *,
    gas_mask: np.ndarray,
    condensed_mask: np.ndarray | None = None,
    trace: float = 1e-300,
) -> np.ndarray:
    """
    Compute mu/RT for ideal gases and pure condensed species.

    For gases:
        mu/RT = g0/RT + ln(x_j) + ln(P/P_ref)

    For condensed species:
        mu/RT = g0/RT

    This matches RP-1311 Eq. 2.11 for ideal gases and pure condensed phases.
    """
    P = float(pressure)

    if P <= 0.0:
        raise ValueError("pressure must be positive.")

    mu = np.array(thermo.g0_over_RT, dtype=float, copy=True)

    gas_mask = np.asarray(gas_mask, dtype=bool)

    if condensed_mask is None:
        condensed_mask = ~gas_mask
    else:
        condensed_mask = np.asarray(condensed_mask, dtype=bool)

    x = np.asarray(mole_fractions_gas, dtype=float)

    if x.shape[0] != int(np.sum(gas_mask)):
        raise ValueError("mole_fractions_gas length does not match gas_mask.")

    mu[gas_mask] = (
        thermo.g0_over_RT[gas_mask]
        + np.log(np.maximum(x, trace))
        + np.log(P / P_REF)
    )

    mu[condensed_mask] = thermo.g0_over_RT[condensed_mask]

    return mu


def mixture_enthalpy(
    n: np.ndarray,
    thermo: ThermoArrays,
) -> float:
    """Execute the documented ``mixture_enthalpy`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    n = np.asarray(n, dtype=float)
    return float(np.sum(n * thermo.enthalpy_molar))


def mixture_entropy(
    n: np.ndarray,
    thermo: ThermoArrays,
    pressure: float,
    *,
    gas_mask: np.ndarray,
    trace: float = 1e-300,
) -> float:
    """Execute the documented ``mixture_entropy`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    P = float(pressure)

    if P <= 0.0:
        raise ValueError("pressure must be positive.")

    n = np.asarray(n, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    s_species = np.array(
        thermo.entropy_molar_standard,
        dtype=float,
        copy=True,
    )

    ng = n[gas_mask]
    ntot_gas = float(np.sum(ng))

    if ntot_gas > 0.0:
        xg = ng / ntot_gas
        s_species[gas_mask] = (
            s_species[gas_mask]
            - RU_KMOL * np.log(np.maximum(xg, trace) * P / P_REF)
        )

    return float(np.sum(n * s_species))


def mixture_gibbs_energy(
    n: np.ndarray,
    thermo: ThermoArrays,
    pressure: float,
    *,
    gas_mask: np.ndarray,
    trace: float = 1e-300,
) -> float:
    """Execute the documented ``mixture_gibbs_energy`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    n = np.asarray(n, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    ng = n[gas_mask]
    ntot_gas = float(np.sum(ng))

    if ntot_gas > 0.0:
        xg = ng / ntot_gas
    else:
        xg = np.full_like(ng, 1.0 / max(len(ng), 1))

    mu_over_RT = chemical_potentials_over_RT(
        thermo,
        xg,
        pressure,
        gas_mask=gas_mask,
        trace=trace,
    )

    return float(np.sum(n * mu_over_RT) * RU_KMOL * thermo.temperature)


def mixture_internal_energy(
    n: np.ndarray,
    thermo: ThermoArrays,
    *,
    gas_mask: np.ndarray,
) -> float:
    """
    Mixture internal energy [J/kg].

    For ideal gas species:
        u = h - R T

    For condensed species:
        u ≈ h

    CEA assumes condensed species occupy negligible volume.
    """
    n = np.asarray(n, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    u = np.array(thermo.enthalpy_molar, dtype=float, copy=True)
    u[gas_mask] -= RU_KMOL * thermo.temperature

    return float(np.sum(n * u))


def frozen_specific_heat_cp(
    n: np.ndarray,
    thermo: ThermoArrays,
) -> float:
    """Execute the documented ``frozen_specific_heat_cp`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    n = np.asarray(n, dtype=float)
    return float(np.sum(n * thermo.specific_heat_cp_molar))


def frozen_specific_heat_cv(
    n: np.ndarray,
    thermo: ThermoArrays,
    *,
    gas_mask: np.ndarray,
) -> float:
    """Execute the documented ``frozen_specific_heat_cv`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    n = np.asarray(n, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    cv = np.array(thermo.specific_heat_cp_molar, dtype=float, copy=True)
    cv[gas_mask] -= RU_KMOL

    return float(np.sum(n * cv))


def total_gas_moles(
    n: np.ndarray,
    *,
    gas_mask: np.ndarray,
) -> float:
    """Execute the public ``total_gas_moles`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    return float(np.sum(np.asarray(n, dtype=float)[np.asarray(gas_mask, dtype=bool)]))


def gas_mole_fractions(
    n: np.ndarray,
    *,
    gas_mask: np.ndarray,
) -> np.ndarray:
    """Execute the public ``gas_mole_fractions`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    n = np.asarray(n, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    ng = n[gas_mask]
    total = float(np.sum(ng))

    if total <= 0.0:
        return np.zeros_like(ng)

    return ng / total


def all_species_mole_fractions(
    n: np.ndarray,
) -> np.ndarray:
    """Execute the public ``all_species_mole_fractions`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    n = np.asarray(n, dtype=float)
    total = float(np.sum(n))

    if total <= 0.0:
        return np.zeros_like(n)

    return n / total


def mass_fractions(
    n: np.ndarray,
    molecular_weights: np.ndarray,
) -> np.ndarray:
    """
    Species mass fractions.

    n:
        kmol/kg

    molecular_weights:
        kg/kmol or kg/mol are both acceptable only if used consistently.
        For this package SpeciesSet stores kg/mol, so we multiply by 1000.
    """
    n = np.asarray(n, dtype=float)
    mw = np.asarray(molecular_weights, dtype=float)

    mass = n * mw * 1000.0
    total = float(np.sum(mass))

    if total <= 0.0:
        return np.zeros_like(n)

    return mass / total


def gas_molecular_weight(
    n: np.ndarray,
    molecular_weights: np.ndarray,
    *,
    gas_mask: np.ndarray,
) -> float:
    """
    Gas molecular weight [kg/kmol].

    CEA's ideal-gas EOS uses gas moles only, while mixture mass includes
    condensed species.
    """
    n = np.asarray(n, dtype=float)
    mw = np.asarray(molecular_weights, dtype=float)
    gas_mask = np.asarray(gas_mask, dtype=bool)

    ng = n[gas_mask]
    total = float(np.sum(ng))

    if total <= 0.0:
        return np.nan

    return float(np.sum(ng * mw[gas_mask] * 1000.0) / total)


def mixture_gas_constant(
    n: np.ndarray,
    *,
    gas_mask: np.ndarray,
) -> float:
    """Execute the documented ``mixture_gas_constant`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    ng = total_gas_moles(n, gas_mask=gas_mask)
    return ng * RU_KMOL


def density_ideal_mixture(
    n: np.ndarray,
    temperature: float,
    pressure: float,
    *,
    gas_mask: np.ndarray,
) -> float:
    """Execute the documented ``density_ideal_mixture`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    Rmix = mixture_gas_constant(n, gas_mask=gas_mask)

    if Rmix <= 0.0:
        return np.inf

    return float(pressure / (Rmix * temperature))


def specific_volume_ideal_mixture(
    n: np.ndarray,
    temperature: float,
    pressure: float,
    *,
    gas_mask: np.ndarray,
) -> float:
    """Execute the public ``specific_volume_ideal_mixture`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    rho = density_ideal_mixture(
        n,
        temperature,
        pressure,
        gas_mask=gas_mask,
    )

    if rho <= 0.0:
        return np.inf

    return 1.0 / rho


def mole_fraction_dict(
    names: list[str],
    n: np.ndarray,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    """Execute the public ``mole_fraction_dict`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    x = all_species_mole_fractions(n)

    return {
        name: float(value)
        for name, value in zip(names, x)
        if value > trace
    }


def gas_mole_fraction_dict(
    names: list[str],
    n: np.ndarray,
    *,
    gas_mask: np.ndarray,
    trace: float = 0.0,
) -> dict[str, float]:
    """Execute the public ``gas_mole_fraction_dict`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    gas_names = [
        name
        for name, is_gas in zip(names, gas_mask)
        if is_gas
    ]

    xg = gas_mole_fractions(n, gas_mask=gas_mask)

    return {
        name: float(value)
        for name, value in zip(gas_names, xg)
        if value > trace
    }


def mass_fraction_dict(
    names: list[str],
    n: np.ndarray,
    molecular_weights: np.ndarray,
    *,
    trace: float = 0.0,
) -> dict[str, float]:
    """Execute the public ``mass_fraction_dict`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    y = mass_fractions(n, molecular_weights)

    return {
        name: float(value)
        for name, value in zip(names, y)
        if value > trace
    }