"""
state.py

Internal equilibrium data structures.

These classes are intentionally lightweight and contain
no solver logic.

All equilibrium calculations operate on these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SpeciesSet:
    """
    Static species information.

    This object never changes during a solve.
    """

    # species names
    names: list[str]

    # molecular weights [kg/mol]
    molecular_weights: np.ndarray

    # elemental composition matrix
    #
    # shape:
    #     (nelements, nspecies)
    #
    # A[i,j] = number of atoms of element i
    #          in species j
    #
    A: np.ndarray

    # element names corresponding to rows of A
    elements: list[str]

    # species flags
    gas_mask: np.ndarray
    condensed_mask: np.ndarray
    ion_mask: np.ndarray

    # convenience lookup
    name_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:

        ns = len(self.names)

        if self.molecular_weights.shape != (ns,):
            raise ValueError(
                "molecular_weights shape mismatch."
            )

        if self.A.shape[1] != ns:
            raise ValueError(
                "Element matrix shape mismatch."
            )

        if self.gas_mask.shape != (ns,):
            raise ValueError(
                "gas_mask shape mismatch."
            )

        if self.condensed_mask.shape != (ns,):
            raise ValueError(
                "condensed_mask shape mismatch."
            )

        if self.ion_mask.shape != (ns,):
            raise ValueError(
                "ion_mask shape mismatch."
            )

        if not self.name_to_index:
            self.name_to_index = {
                name: i
                for i, name in enumerate(self.names)
            }

    @property
    def nspecies(self) -> int:
        return len(self.names)

    @property
    def nelements(self) -> int:
        return len(self.elements)


@dataclass(slots=True)
class FeedState:
    """
    Reactant information supplied to the equilibrium solver.

    All input types:

        Reactants
        CombustionGas
        dict

    are converted into this structure.
    """

    # element totals

    # kmol element / kg mixture
    element_totals: np.ndarray

    # element ordering
    elements: list[str]

    # J/kg
    enthalpy: float | None = None

    # J/kg
    internal_energy: float | None = None

    # K
    temperature: float | None = None

    # Pa
    pressure: float | None = None

    # optional metadata
    source: Any = None

    @property
    def nelements(self) -> int:
        return len(self.elements)


@dataclass(slots=True)
class EquilibriumState:
    """
    Mutable equilibrium solution state.

    This is the object updated during Newton iterations.
    """

    # thermodynamic state

    temperature: float
    pressure: float

    # species amounts

    # kmol species / kg mixture
    n: np.ndarray

    # total gas kmol/kg
    total_gas_moles: float

    # reference species set
    species: SpeciesSet

    # element totals
    element_totals: np.ndarray

    # iteration bookkeeping

    iteration: int = 0

    converged: bool = False

    residual_norm: float = np.inf

    def copy(self) -> "EquilibriumState":
        return EquilibriumState(
            temperature=self.temperature,
            pressure=self.pressure,
            n=self.n.copy(),
            total_gas_moles=self.total_gas_moles,
            species=self.species,
            element_totals=self.element_totals.copy(),
            iteration=self.iteration,
            converged=self.converged,
            residual_norm=self.residual_norm,
        )

    @property
    def nspecies(self) -> int:
        return self.species.nspecies

    @property
    def nelements(self) -> int:
        return self.species.nelements

    @property
    def gas_moles(self) -> np.ndarray:
        return self.n[self.species.gas_mask]

    @property
    def condensed_moles(self) -> np.ndarray:
        return self.n[self.species.condensed_mask]

    @property
    def ion_moles(self) -> np.ndarray:
        return self.n[self.species.ion_mask]

    @property
    def mole_fractions_gas(self) -> np.ndarray:

        gas = self.gas_moles

        total = np.sum(gas)

        if total <= 0.0:
            return np.zeros_like(gas)

        return gas / total

    @property
    def molecular_weight_gas(self) -> float:
        """
        Gas molecular weight.

        Equivalent to CEA equation (2.3a)
        for gaseous species only.
        """

        gas_mask = self.species.gas_mask

        n = self.n[gas_mask]

        total = np.sum(n)

        if total <= 0.0:
            return np.nan

        mw = self.species.molecular_weights[gas_mask]

        return float(
            np.sum(n * mw) / total
        )

    @property
    def element_residual(self) -> np.ndarray:
        """
        Current elemental imbalance.

        A*n - b
        """

        return (
            self.species.A @ self.n
            - self.element_totals
        )


@dataclass(slots=True)
class NewtonCorrection:
    """
    Result of one Newton solve.
    """

    dx: np.ndarray

    residual: np.ndarray

    residual_norm: float

    success: bool

    message: str = ""


@dataclass(slots=True)
class CondensedPhaseCandidate:
    """
    Candidate condensed phase considered
    during phase insertion/removal logic.
    """

    species_index: int

    species_name: str

    gibbs_test_value: float

    should_insert: bool


@dataclass(slots=True)
class TransportState:
    """
    Cached transport data used by
    equilibrium transport calculations.
    """

    names: list[str]

    mole_fractions: np.ndarray

    molecular_weights: np.ndarray

    viscosity_species: np.ndarray

    conductivity_species: np.ndarray


@dataclass(slots=True)
class EquilibriumResults:
    """
    Final immutable results object.

    Used internally before constructing
    the public Equilibrium API.
    """

    state: EquilibriumState

    enthalpy: float

    entropy: float

    internal_energy: float

    density: float

    cp_frozen: float
    cv_frozen: float

    cp_equilibrium: float
    cv_equilibrium: float

    gamma_frozen: float
    gamma_equilibrium: float

    viscosity_frozen: float | None = None
    conductivity_frozen: float | None = None
    prandtl_frozen: float | None = None

    viscosity_equilibrium: float | None = None
    conductivity_equilibrium: float | None = None
    prandtl_equilibrium: float | None = None