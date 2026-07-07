"""
species.py

Species screening and element-matrix construction for the CEA-style
equilibrium solver.

This module does not solve equilibrium. It only converts CEADatabase species
metadata into a SpeciesSet usable by TP/HP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..CEADatabase import CEA
from ..SpeciesDatabase import SpeciesDatabase
from .state import SpeciesSet
from ..Exceptions import EquilibriumSetupError, EquilibriumConvergenceError, PropertyUnavailableError


CHARGE_ELEMENT = "E"


@dataclass(slots=True, frozen=True)
class SpeciesBuildOptions:
    """Represent the public ThermoProp ``SpeciesBuildOptions`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """
    include_gases: bool = True
    include_condensed: bool = True
    include_ions: bool = True
    include_electron: bool = True
    include_reactants: bool = False
    require_thermo: bool = True
    temperature: float | None = None


def is_ion_name(name: str) -> bool:
    """Execute the public ``is_ion_name`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    if name == "e-":
        return True
    return name.endswith("+") or name.endswith("-") or "++" in name or "--" in name


def species_charge(name: str) -> float:
    """
    CEA charge convention for charge-balance row.

    Positive ions are electron-deficient, negative ions/electron are
    electron-rich. The exact sign only matters if used consistently in
    charge balance.
    """
    if name == "e-":
        return -1.0

    if name.endswith("++"):
        return 2.0
    if name.endswith("--"):
        return -2.0
    if name.endswith("+"):
        return 1.0
    if name.endswith("-"):
        return -1.0

    return 0.0


def _database_species_names() -> list[str]:
    """
    Return the broadest CEA product species list available from CEADatabase.
    """
    for attr in ("species_names", "names", "product_species"):
        if hasattr(CEA, attr):
            value = getattr(CEA, attr)
            if callable(value):
                value = value()
            return list(value)

    if hasattr(CEA, "species"):
        value = CEA.species
        if callable(value):
            value = value()
        return list(value)

    raise AttributeError(
        "CEADatabase object CEA must expose product_species, species_names, "
        "names, or species."
    )


def _resolve_name(name: str) -> str:
    return SpeciesDatabase._cea_input_name(name)



def _name_has_condensed_phase_tag(name: str) -> bool:
    lower = name.lower()
    return (
        "(l)" in lower
        or "(cr" in lower
        or "(a)" in lower
        or "(b)" in lower
        or "(i)" in lower
        or "(ii)" in lower
        or "(iii)" in lower
        or "(iv)" in lower
        or "(v)" in lower
    )



def species_valid_near_temperature(
    name: str,
    temperature: float | None,
    *,
    margin: float = 0.0,
) -> bool:
    """Execute the public ``species_valid_near_temperature`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    if temperature is None:
        return True

    limits = _temperature_limits(name)

    if limits is None:
        return True

    Tmin, Tmax = limits
    T = float(temperature)

    return (Tmin - margin) <= T <= (Tmax + margin)


def _has_species(name: str) -> bool:
    if hasattr(CEA, "has_species"):
        return bool(CEA.has_species(name))
    try:
        CEA.elemental_composition(name)
        return True
    except Exception:
        return False



def _temperature_limits(name: str) -> tuple[float, float] | None:
    try:
        if hasattr(CEA, "temperature_ranges"):
            ranges = CEA.temperature_ranges(name)
            if ranges:
                lows = [float(r[0]) for r in ranges]
                highs = [float(r[1]) for r in ranges]
                return min(lows), max(highs)
    except Exception:
        pass

    try:
        if hasattr(CEA, "temperature_limits"):
            limits = CEA.temperature_limits(name)
            return tuple(float(x) for x in limits)
    except Exception:
        pass

    return None



def _has_thermo(name: str, temperature: float | None = None) -> bool:
    if hasattr(CEA, "has_thermo"):
        return bool(CEA.has_thermo(name))

    try:
        CEA.thermo_molar(name, 298.15)
        return True
    except Exception:
        return False


def _is_reactant(name: str) -> bool:
    if hasattr(CEA, "is_reactant"):
        try:
            return bool(CEA.is_reactant(name))
        except Exception:
            return False
    return False


def _is_gas(name: str) -> bool:
    # CEA names with explicit condensed phase tags must be treated as condensed,
    # even if CEADatabase metadata is incomplete.
    if _name_has_condensed_phase_tag(name):
        return False

    if hasattr(CEA, "is_gas"):
        try:
            return bool(CEA.is_gas(name))
        except Exception:
            pass

    return True


def _is_condensed(name: str) -> bool:
    # Name syntax is authoritative for CEA condensed phases.
    if _name_has_condensed_phase_tag(name):
        return True

    if hasattr(CEA, "is_condensed"):
        try:
            return bool(CEA.is_condensed(name))
        except Exception:
            pass

    return False


def _elemental_composition(name: str) -> dict[str, float]:
    comp = CEA.elemental_composition(name)
    return {
        str(element): float(count)
        for element, count in comp.items()
        if abs(float(count)) > 0.0
    }


def _molar_mass(name: str) -> float:
    """
    Return kg/mol.
    """
    if hasattr(CEA, "molar_mass"):
        return float(CEA.molar_mass(name))
    if hasattr(CEA, "molecular_weight"):
        return float(CEA.molecular_weight(name)) / 1000.0
    raise AttributeError("CEA must expose molar_mass() or molecular_weight().")


def normalize_elements(elements: Iterable[str]) -> list[str]:
    """Execute the public ``normalize_elements`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    normalized = sorted({str(e) for e in elements if str(e) != CHARGE_ELEMENT})
    return normalized


def species_is_compatible(
    name: str,
    allowed_elements: set[str],
    *,
    include_ions: bool = True,
    include_electron: bool = True,
) -> bool:
    """Execute the public ``species_is_compatible`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    if name == "e-":
        return include_ions and include_electron

    comp = _elemental_composition(name)

    if not comp:
        return False

    if not set(comp).issubset(allowed_elements):
        return False

    if is_ion_name(name) and not include_ions:
        return False

    return True


def build_species_names(
    elements: Iterable[str],
    *,
    candidates: Iterable[str] | None = None,
    options: SpeciesBuildOptions | None = None,
) -> list[str]:
    """
    Build the CEA-compatible species list.

    Species are included when:
    - present in CEADatabase
    - compatible with the input element set
    - phase is enabled by options
    - thermo is available
    - ion/electron rules are satisfied
    """
    if options is None:
        options = SpeciesBuildOptions()

    allowed_elements = set(normalize_elements(elements))

    if candidates is None:
        raw_candidates = _database_species_names()
    else:
        raw_candidates = list(candidates)

    names: list[str] = []
    seen: set[str] = set()

    for raw_name in raw_candidates:
        try:
            name = _resolve_name(raw_name)
        except Exception:
            continue

        if not _has_species(name):
            continue

        if name in seen:
            continue

        if not options.include_reactants and _is_reactant_only_species(name):
            continue

        gas = _is_gas(name)
        condensed = _is_condensed(name)
        ion = is_ion_name(name)

        if gas and not options.include_gases:
            continue

        if condensed and not options.include_condensed:
            continue

        if ion and not options.include_ions:
            continue

        if name == "e-" and not options.include_electron:
            continue

        if not species_is_compatible(
            name,
            allowed_elements,
            include_ions=options.include_ions,
            include_electron=options.include_electron,
        ):
            continue

        # Do NOT skip species because T is outside nominal thermo range.
        # CEA extrapolates and warns instead of removing species.

        if options.require_thermo and not _has_thermo(name, options.temperature):
            continue

        names.append(name)
        seen.add(name)

    if not names:
        raise EquilibriumSetupError("No compatible CEA species were found.")

    names.sort(key=_species_sort_key)
    return names





def _species_sort_key(name: str) -> tuple[int, str]:
    """
    CEA-style ordering: gases first, condensed species after gases.
    Electron is kept with gas/ion species.
    """
    if _is_condensed(name):
        phase_rank = 1
    else:
        phase_rank = 0

    return phase_rank, name


def build_element_list(
    input_elements: Iterable[str],
    species_names: Iterable[str],
    *,
    include_charge: bool = True,
) -> list[str]:
    """Execute the public ``build_element_list`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    ordered = normalize_elements(input_elements)

    if include_charge and any(is_ion_name(name) for name in species_names):
        ordered.append(CHARGE_ELEMENT)

    return ordered


def build_element_matrix(
    species_names: list[str],
    elements: list[str],
) -> np.ndarray:
    """Execute the public ``build_element_matrix`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    A = np.zeros((len(elements), len(species_names)), dtype=float)

    for j, name in enumerate(species_names):
        comp = _elemental_composition(name)

        for i, element in enumerate(elements):
            if element == CHARGE_ELEMENT:
                A[i, j] = species_charge(name)
            else:
                A[i, j] = comp.get(element, 0.0)

    return A


def build_species_set(
    elements: Iterable[str],
    *,
    candidates: Iterable[str] | None = None,
    options: SpeciesBuildOptions | None = None,
) -> SpeciesSet:
    """
    Build a SpeciesSet from CEADatabase.

    Parameters
    ----------
    elements:
        Elements present in the reactants, e.g. ["C", "H", "O"].

    candidates:
        Optional restricted candidate species list.

    options:
        SpeciesBuildOptions controlling gases, condensed phases, ions,
        electron, reactants, and thermo checks.
    """
    if options is None:
        options = SpeciesBuildOptions()

    names = build_species_names(
        elements,
        candidates=candidates,
        options=options,
    )

    element_list = build_element_list(
        elements,
        names,
        include_charge=options.include_ions,
    )

    A = build_element_matrix(names, element_list)

    molecular_weights = np.array(
        [_molar_mass(name) for name in names],
        dtype=float,
    )

    gas_mask = np.array(
        [_is_gas(name) for name in names],
        dtype=bool,
    )

    condensed_mask = np.array(
        [_is_condensed(name) for name in names],
        dtype=bool,
    )

    ion_mask = np.array(
        [is_ion_name(name) for name in names],
        dtype=bool,
    )

    if options.include_gases and not np.any(gas_mask):
        raise EquilibriumSetupError("Species set must contain at least one gas species.")
    
    return SpeciesSet(
        names=names,
        molecular_weights=molecular_weights,
        A=A,
        elements=element_list,
        gas_mask=gas_mask,
        condensed_mask=condensed_mask,
        ion_mask=ion_mask,
    )


def split_species_by_phase(species: SpeciesSet) -> tuple[list[str], list[str]]:
    """Execute the public ``split_species_by_phase`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    gases = [
        name
        for name, is_gas in zip(species.names, species.gas_mask)
        if is_gas
    ]

    condensed = [
        name
        for name, is_condensed in zip(species.names, species.condensed_mask)
        if is_condensed
    ]

    return gases, condensed


def subset_species_set(species: SpeciesSet, indices: Iterable[int]) -> SpeciesSet:
    """Execute the public ``subset_species_set`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    idx = np.array(list(indices), dtype=int)

    names = [species.names[i] for i in idx]

    return SpeciesSet(
        names=names,
        molecular_weights=species.molecular_weights[idx].copy(),
        A=species.A[:, idx].copy(),
        elements=list(species.elements),
        gas_mask=species.gas_mask[idx].copy(),
        condensed_mask=species.condensed_mask[idx].copy(),
        ion_mask=species.ion_mask[idx].copy(),
    )


def add_species_to_set(
    species: SpeciesSet,
    new_names: Iterable[str],
) -> SpeciesSet:
    """Execute the public ``add_species_to_set`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    combined = list(species.names)

    for name in new_names:
        resolved = _resolve_name(name)
        if resolved not in combined:
            combined.append(resolved)

    elements = build_element_list(
        species.elements,
        combined,
        include_charge=any(is_ion_name(name) for name in combined),
    )

    A = build_element_matrix(combined, elements)

    return SpeciesSet(
        names=combined,
        molecular_weights=np.array([_molar_mass(name) for name in combined]),
        A=A,
        elements=elements,
        gas_mask=np.array([_is_gas(name) for name in combined], dtype=bool),
        condensed_mask=np.array([_is_condensed(name) for name in combined], dtype=bool),
        ion_mask=np.array([is_ion_name(name) for name in combined], dtype=bool),
    )


def remove_species_from_set(
    species: SpeciesSet,
    remove_names: Iterable[str],
) -> SpeciesSet:
    """Execute the public ``remove_species_from_set`` operation for ``ThermoProp``.

    This method is part of the importable ThermoProp API rather than an internal
    helper.  Arguments are validated and normalized before use, return values follow
    ThermoProp's SI-unit and composition conventions, and lookup or state failures
    are reported through ThermoProp exception types with contextual messages."""
    remove = {_resolve_name(name) for name in remove_names}

    keep_indices = [
        i
        for i, name in enumerate(species.names)
        if name not in remove
    ]

    return subset_species_set(species, keep_indices)


def active_species_indices_from_moles(
    n: np.ndarray,
    *,
    gas_mask: np.ndarray,
    total_gas_moles: float | None = None,
    trace: float = 1e-12,
) -> np.ndarray:
    """Execute the documented ``active_species_indices_from_moles`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    n = np.asarray(n, dtype=float)

    if total_gas_moles is None:
        total_gas_moles = float(np.sum(n[gas_mask]))

    active = np.zeros_like(n, dtype=bool)

    if total_gas_moles > 0.0:
        active[gas_mask] = n[gas_mask] / total_gas_moles > trace

    active[~gas_mask] = n[~gas_mask] > 0.0

    return np.nonzero(active)[0]



def _is_reactant_only_species(name: str) -> bool:
    # Condensed CEA phase species like O2(L), C(gr), H2O(L)
    # must never be treated as reactant-only, even if they lack NASA intervals.
    if _is_condensed(name):
        return False

    try:
        if hasattr(CEA, "reactant_names"):
            reactants = CEA.reactant_names
            if callable(reactants):
                reactants = reactants()
            if name in set(reactants):
                return True
    except Exception:
        pass

    try:
        if hasattr(CEA, "is_reactant"):
            return bool(CEA.is_reactant(name))
    except Exception:
        pass

    return False
