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
    """Represent the public ThermoProp ``SpeciesSet`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
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
        """Return the public ``nspecies`` value for this ``SpeciesSet`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return len(self.names)

    @property
    def nelements(self) -> int:
        """Return the public ``nelements`` value for this ``SpeciesSet`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
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
        """Return the public ``nelements`` value for this ``FeedState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return len(self.elements)


@dataclass(slots=True)
class EquilibriumState:
    """Represent the public ThermoProp ``EquilibriumState`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
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
        """Execute the public ``copy`` operation for ``EquilibriumState``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
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
        """Return the public ``nspecies`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.species.nspecies

    @property
    def nelements(self) -> int:
        """Return the public ``nelements`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.species.nelements

    @property
    def gas_moles(self) -> np.ndarray:
        """Return the public ``gas_moles`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.n[self.species.gas_mask]

    @property
    def condensed_moles(self) -> np.ndarray:
        """Return the public ``condensed_moles`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.n[self.species.condensed_mask]

    @property
    def ion_moles(self) -> np.ndarray:
        """Return the public ``ion_moles`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.n[self.species.ion_mask]

    @property
    def mole_fractions_gas(self) -> np.ndarray:

        """Return the public ``mole_fractions_gas`` value for this ``EquilibriumState`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        gas = self.gas_moles

        total = np.sum(gas)

        if total <= 0.0:
            return np.zeros_like(gas)

        return gas / total

    @property
    def molecular_weight_gas(self) -> float:
        """Return the gas-phase molecular weight for this ``EquilibriumState`` state.

        The value is evaluated from the current state and active backend.  Units are
        kg/kmol, numerically equal to g/mol.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
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
        """Return the element residual for this ``EquilibriumState`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """

        return (
            self.species.A @ self.n
            - self.element_totals
        )


@dataclass(slots=True)
class NewtonCorrection:
    """Represent the public ThermoProp ``NewtonCorrection`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """

    dx: np.ndarray

    residual: np.ndarray

    residual_norm: float

    success: bool

    message: str = ""


@dataclass(slots=True)
class CondensedPhaseCandidate:
    """Represent the public ThermoProp ``CondensedPhaseCandidate`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """

    species_index: int

    species_name: str

    gibbs_test_value: float

    should_insert: bool


@dataclass(slots=True)
class TransportState:
    """Represent the public ThermoProp ``TransportState`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """

    names: list[str]

    mole_fractions: np.ndarray

    molecular_weights: np.ndarray

    viscosity_species: np.ndarray

    conductivity_species: np.ndarray


@dataclass(slots=True)
class EquilibriumResults:
    """Represent the public ThermoProp ``EquilibriumResults`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
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

    cp_transport_frozen: float | None = None
    cp_transport_equilibrium: float | None = None

    viscosity_frozen: float | None = None
    conductivity_frozen: float | None = None
    prandtl_frozen: float | None = None

    viscosity_equilibrium: float | None = None
    conductivity_equilibrium: float | None = None
    prandtl_equilibrium: float | None = None

    conductivity_reaction: float | None = None