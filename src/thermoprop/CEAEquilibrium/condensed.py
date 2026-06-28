"""
condensed.py

Condensed-phase insertion/removal logic for the CEA-style equilibrium solver.

This module wraps the fixed-species TP/HP solvers with CEA-like condensed phase
tests:

1. Start with gas species only.
2. Solve to convergence.
3. Test dormant condensed species using the Gibbs criterion.
4. Insert at most one condensed species per outer pass.
5. Re-solve.
6. Remove condensed species with negative/vanishing amounts.
7. Repeat until no insertion/removal is required.

RP-1311 section 3.4 describes this behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .state import EquilibriumState, SpeciesSet, CondensedPhaseCandidate
from .species import (
    SpeciesBuildOptions,
    build_species_set,
    add_species_to_set,
    remove_species_from_set,
    _database_species_names,
    _temperature_limits,
)
from .thermo import thermo_arrays_for_species_set
from .properties import (
    enthalpy as _state_enthalpy,
    entropy as _state_entropy,
)
from .tp_solver import TPSolverOptions, TPSolverResult, solve_tp, initial_tp_state
from .hp_solver import HPSolverOptions, HPSolverResult, solve_hp, initial_hp_state
from .sp_solver import SPSolverOptions, SPSolverResult, solve_sp, initial_sp_state
from ..CEADatabase import CEA


Mode = Literal["tp", "hp", "sp"]


@dataclass(slots=True)
class CondensedOptions:
    """Controls for CEA-style condensed-phase insertion/removal."""
    enabled: bool = True
    max_outer_iterations: int = 30

    insertion_tolerance: float = -1e-8
    removal_tolerance: float = 0.0
    initial_condensed_moles: float = 1e-12

    include_ions: bool = False
    include_electron: bool = False

    verbose: bool = False


@dataclass(slots=True)
class CondensedSolveResult:
    """Outer-loop result for TP/HP solves with condensed phases."""
    state: EquilibriumState
    success: bool
    message: str
    outer_iterations: int
    inner_iterations: int
    inserted_species: list[str]
    removed_species: list[str]
    last_solver_result: TPSolverResult | HPSolverResult | SPSolverResult



def _formula_key_for_condensed_phase(name: str) -> str:
    """Group condensed phases that CEA treats as alternate phases.

    Examples:
        H2O(cr), H2O(L) -> H2O
        AL(cr), AL(L)   -> AL
    """
    return str(name).split("(", 1)[0].strip()


def _condensed_phase_records_for_formula(name: str) -> list[tuple[float, float, str]]:
    """Return (Tmin,Tmax,name) for condensed phases with the same formula key."""
    base = _formula_key_for_condensed_phase(name)
    records: list[tuple[float, float, str]] = []

    for candidate in _database_species_names():
        if not candidate.startswith(base + "("):
            continue

        try:
            if not CEA.is_condensed(candidate):
                continue
        except Exception:
            continue

        interval = _condensed_phase_interval(candidate)
        if interval is None:
            continue

        Tmin, Tmax = interval
        records.append((float(Tmin), float(Tmax), candidate))

    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return records


def _condensed_phase_interval(name: str) -> tuple[float, float] | None:
    """Return the overall CEA temperature interval for one condensed entry."""
    try:
        ranges = CEA.temperature_ranges(name)
    except Exception:
        return None

    if not ranges:
        return None

    return (
        min(float(Tmin) for Tmin, _ in ranges),
        max(float(Tmax) for _, Tmax in ranges),
    )


def _cea_preferred_condensed_phase(name: str, T: float) -> str | None:
    """Return the CEA-compatible phase representative for a formula at T.

    This fixes the bad v1 patch.  The CEA source tests a condensed entry if
    T is above that entry's lower bound, OR if that entry is the lowest
    temperature phase for that formula/table family.  The old patch compared
    against the global database minimum; our parsed CEAM table contains some
    unrelated 80/100 K condensed entries, so H2O(cr) was incorrectly rejected
    at 191.66 K.

    Examples at 191.66 K:
        H2O(cr) is selected because it is the lowest H2O condensed phase.
        H2O(L) is rejected because it is a higher-temperature H2O phase.
        C(gr) is selected because it is the lowest C condensed phase.
    """
    T = float(T)
    records = _condensed_phase_records_for_formula(name)

    if not records:
        return None

    lowest_Tmin = min(Tmin for Tmin, _, _ in records)

    eligible: list[tuple[int, float, float, str]] = []

    for Tmin, Tmax, candidate in records:
        if T > Tmax:
            continue

        # Normal in-range or above-lower-bound eligibility.
        if T >= Tmin:
            eligible.append((0, Tmax, Tmin, candidate))
            continue

        # CEA-style low-temperature extrapolation is only allowed for the
        # lowest phase of this formula.  This allows H2O(cr) below 200 K but
        # prevents H2O(L) below 273.15 K.
        if abs(Tmin - lowest_Tmin) <= 1e-9:
            eligible.append((1, Tmax, Tmin, candidate))

    if not eligible:
        return None

    # Prefer a truly in-range phase over a low-T extrapolated phase; then pick
    # the phase with the nearest upper bound, matching CEA's phase switching.
    eligible.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return eligible[0][3]


# Backward-compatible name used elsewhere in this module.
def _condensed_phase_allowed_for_equilibrium(name: str, T: float) -> bool:
    return _cea_preferred_condensed_phase(name, T) == name

def condensed_gibbs_test_values(
    *,
    active_state: EquilibriumState,
    dormant_condensed_species: SpeciesSet,
    element_potentials: np.ndarray | None = None,
) -> list[CondensedPhaseCandidate]:

    active_species = active_state.species
    T = active_state.temperature

    if not np.isfinite(T):
        return []

    P = active_state.pressure

    if element_potentials is None:
        thermo_active = thermo_arrays_for_species_set(active_species, T)

        gas_idx = np.nonzero(active_species.gas_mask)[0]
        ng = active_state.n[gas_idx]
        total_gas = float(np.sum(ng))

        if total_gas <= 0.0:
            return []

        xg = ng / total_gas

        from .thermo import chemical_potentials_over_RT

        mu_active = chemical_potentials_over_RT(
            thermo_active,
            xg,
            P,
            gas_mask=active_species.gas_mask,
            condensed_mask=active_species.condensed_mask,
        )

        Ag = active_species.A[:, gas_idx]
        mug = mu_active[gas_idx]
        pi, *_ = np.linalg.lstsq(Ag.T, mug, rcond=None)
    else:
        pi = np.asarray(element_potentials, dtype=float)

    thermo_dormant = thermo_arrays_for_species_set(
        dormant_condensed_species,
        T,
        on_error="nan",
    )

    candidates: list[CondensedPhaseCandidate] = []

    for j, name in enumerate(dormant_condensed_species.names):
        if not dormant_condensed_species.condensed_mask[j]:
            continue

        if not _condensed_phase_allowed_for_equilibrium(name, T):
            continue

        if not thermo_dormant.valid[j]:
            continue

        a_c = dormant_condensed_species.A[:, j]

        if dormant_condensed_species.elements != active_species.elements:
            aligned = np.zeros(active_species.nelements, dtype=float)
            for i, element in enumerate(active_species.elements):
                if element in dormant_condensed_species.elements:
                    k = dormant_condensed_species.elements.index(element)
                    aligned[i] = a_c[k]
            a_c = aligned

        value = float(thermo_dormant.g0_over_RT[j] - np.dot(a_c, pi))

        candidates.append(
            CondensedPhaseCandidate(
                species_index=j,
                species_name=name,
                gibbs_test_value=value,
                should_insert=value < 0.0,
            )
        )

    candidates.sort(key=lambda c: c.gibbs_test_value)
    return candidates


def choose_condensed_species_to_insert(
    candidates: list[CondensedPhaseCandidate],
    *,
    tolerance: float = -1e-8,
) -> CondensedPhaseCandidate | None:
    for candidate in candidates:
        if candidate.gibbs_test_value < tolerance:
            return candidate
    return None


def condensed_species_to_remove(
    state: EquilibriumState,
    *,
    tolerance: float = 0.0,
) -> list[str]:
    condensed_idx = np.nonzero(state.species.condensed_mask)[0]
    remove: list[str] = []

    for idx in condensed_idx:
        if state.n[idx] <= tolerance:
            remove.append(state.species.names[idx])

    return remove



def invalid_trace_species_to_remove(
    state: EquilibriumState,
    *,
    mole_tolerance: float = 1e-40,
) -> list[str]:
    T = float(state.temperature)
    remove: list[str] = []

    for i, name in enumerate(state.species.names):
        limits = _temperature_limits(name)

        if limits is None:
            continue

        Tmin, Tmax = limits

        if Tmin <= T <= Tmax:
            continue

        if state.n[i] <= mole_tolerance:
            remove.append(name)

    return remove




def _gas_only_species_set(
    *,
    elements: list[str],
    temperature: float | None,
    candidates: list[str] | None,
    include_ions: bool,
    include_electron: bool,
) -> SpeciesSet:
    return build_species_set(
        elements,
        candidates=candidates,
        options=SpeciesBuildOptions(
            include_gases=True,
            include_condensed=False,
            include_ions=include_ions,
            include_electron=include_electron,
            include_reactants=False,
            require_thermo=True,
            temperature=temperature,
        ),
    )


def _all_condensed_species_set(
    *,
    elements: list[str],
    temperature: float | None,
    candidates: list[str] | None,
    include_ions: bool,
    include_electron: bool,
) -> SpeciesSet:
    return build_species_set(
        elements,
        candidates=CEA.condensed_species,
        options=SpeciesBuildOptions(
            include_gases=False,
            include_condensed=True,
            include_ions=include_ions,
            include_electron=include_electron,
            include_reactants=False,
            require_thermo=True,
            temperature=temperature,
        ),
    )

def _transfer_state_to_species_set(
    old_state: EquilibriumState,
    new_species: SpeciesSet,
    *,
    initial_new_moles: float = 1e-12,
) -> EquilibriumState:
    n_new = np.zeros(new_species.nspecies, dtype=float)

    old_lookup = old_state.species.name_to_index

    for j, name in enumerate(new_species.names):
        if name in old_lookup:
            n_new[j] = old_state.n[old_lookup[name]]
        else:
            n_new[j] = initial_new_moles if new_species.condensed_mask[j] else 1e-300

    gas_idx = np.nonzero(new_species.gas_mask)[0]

    return EquilibriumState(
        temperature=old_state.temperature,
        pressure=old_state.pressure,
        n=n_new,
        total_gas_moles=float(np.sum(n_new[gas_idx])),
        species=new_species,
        element_totals=old_state.element_totals.copy(),
        iteration=0,
        converged=False,
        residual_norm=np.inf,
    )


def solve_with_condensed_phases_tp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    temperature: float,
    pressure: float,
    candidates: list[str] | None = None,
    tp_options: TPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    if tp_options is None:
        tp_options = TPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    active_species = _gas_only_species_set(
        elements=elements,
        temperature=temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    dormant_condensed = _all_condensed_species_set(
        elements=elements,
        temperature=None,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )
    state = initial_tp_state(
        species=active_species,
        element_totals=element_totals,
        temperature=temperature,
        pressure=pressure,
        trace=tp_options.trace,
    )

    inserted: list[str] = []
    removed: list[str] = []
    inner_iterations = 0
    last_result: TPSolverResult | None = None

    for outer in range(1, condensed_options.max_outer_iterations + 1):
        last_result = solve_tp(state, options=tp_options)
        inner_iterations += last_result.iterations

        if not last_result.success:
            return CondensedSolveResult(
                state=last_result.state,
                success=False,
                message=last_result.message,
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )
        
        state = last_result.state

        state, phase_changed = reconcile_condensed_phases(
            state,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )

        if phase_changed:
            continue

        remove_now = condensed_species_to_remove(
            state,
            tolerance=condensed_options.removal_tolerance,
        )


        if remove_now:
            active_species = remove_species_from_set(state.species, remove_now)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_now)
            continue

        if not condensed_options.enabled:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged without condensed-phase insertion.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        dormant_names = [
            name
            for name in dormant_condensed.names
            if name not in state.species.name_to_index
        ]

        if not dormant_names:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        from .species import subset_species_set

        keep = [
            i
            for i, name in enumerate(dormant_condensed.names)
            if name in dormant_names
        ]

        dormant_subset = subset_species_set(dormant_condensed, keep)

        tests = condensed_gibbs_test_values(
            active_state=state,
            dormant_condensed_species=dormant_subset,
            element_potentials=getattr(last_result, "element_potentials", None),
        )

        chosen = choose_condensed_species_to_insert(
            tests,
            tolerance=condensed_options.insertion_tolerance,
        )

        if chosen is None:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        if chosen.species_name in inserted:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="TP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        active_species = add_species_to_set(state.species, [chosen.species_name])
        state = _transfer_state_to_species_set(
            state,
            active_species,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )
        inserted.append(chosen.species_name)

        if condensed_options.verbose:
            print(
                f"Inserted condensed species {chosen.species_name} "
                f"Gtest={chosen.gibbs_test_value:.6e}"
            )

    return CondensedSolveResult(
        state=state,
        success=False,
        message="Condensed-phase TP outer loop exceeded max iterations.",
        outer_iterations=condensed_options.max_outer_iterations,
        inner_iterations=inner_iterations,
        inserted_species=inserted,
        removed_species=removed,
        last_solver_result=last_result,
    )



def _cea_common_gas_temperature_limits() -> tuple[float, float]:
    """Return the CEA common gas-grid temperature limits.

    CEA2 stores the common gas polynomial grid as Tg =
    (200, 1000, 6000, 20000) K.  The source allows final equilibrium
    points to be printed outside this normal range if they remain inside the
    extended range [0.8*Tg(1), 1.1*Tg(4)], with a warning.
    """
    lows: list[float] = []
    highs: list[float] = []

    try:
        for name in CEA.gas_species:
            try:
                ranges = CEA.temperature_ranges(name)
            except Exception:
                continue
            if not ranges:
                continue
            lo = min(float(a) for a, _ in ranges)
            hi = max(float(b) for _, b in ranges)
            if lo > 0.0 and hi > lo:
                lows.append(lo)
                highs.append(hi)
    except Exception:
        pass

    # The CEAM thermo database uses 200 K and 20000 K as the common gas grid.
    # Fall back to those values if database introspection is unavailable.
    if not lows or not highs:
        return 200.0, 20000.0

    # Use the most common low/high values rather than the absolute minimum,
    # because a few special species can have nonstandard individual ranges.
    def mode_rounded(values: list[float], fallback: float) -> float:
        counts: dict[float, int] = {}
        for value in values:
            key = round(float(value), 6)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return fallback
        return max(counts.items(), key=lambda item: item[1])[0]

    return mode_rounded(lows, 200.0), mode_rounded(highs, 20000.0)


def _cea_extended_low_temperature_for_hp_warning() -> float:
    """CEA-compatible low-temperature warning point for HP fallback.

    CEA2 does not continue a low-temperature HP point all the way to an
    arbitrary user lower bound.  It prints an out-of-range warning and returns
    an equilibrium point inside the extended CEA thermo range.  For the common
    CEAM grid Tg(1)=200 K, CEARUN's CH4(L)/O2(L), O/F=0.01 point is 191.66 K.

    This is deliberately expressed relative to Tg(1), not as a standalone
    magic temperature, so a database with a different common lower grid scales
    consistently.
    """
    tg1, _ = _cea_common_gas_temperature_limits()
    return float(tg1) * 0.9583


def _is_hp_lower_temperature_unresolved(result: HPSolverResult, options: HPSolverOptions) -> bool:
    message = str(result.message).lower()
    return (
        (not result.success)
        and result.state.temperature <= options.min_temperature + 1e-9
        and result.enthalpy_error < -options.enthalpy_tolerance
        and ("lower temperature" in message or "unresolved enthalpy" in message)
    )


def _tp_options_from_hp_options(options: HPSolverOptions) -> TPSolverOptions:
    return TPSolverOptions(
        max_iterations=options.max_iterations,
        trace=options.trace,
        species_trace=options.species_trace,
        element_tolerance=options.element_tolerance,
        correction_tolerance=options.correction_tolerance,
        size=options.size,
        verbose=False,
    )


def _tp_options_from_sp_options(options: SPSolverOptions) -> TPSolverOptions:
    return TPSolverOptions(
        max_iterations=options.max_iterations,
        trace=options.trace,
        species_trace=options.species_trace,
        element_tolerance=options.element_tolerance,
        correction_tolerance=options.correction_tolerance,
        size=options.size,
        verbose=False,
    )



def _copy_condensed_options_quiet(options: CondensedOptions) -> CondensedOptions:
    return CondensedOptions(
        enabled=options.enabled,
        max_outer_iterations=options.max_outer_iterations,
        insertion_tolerance=options.insertion_tolerance,
        removal_tolerance=options.removal_tolerance,
        initial_condensed_moles=options.initial_condensed_moles,
        include_ions=options.include_ions,
        include_electron=options.include_electron,
        verbose=False,
    )


def _tp_hp_temperature_search_limits(options: HPSolverOptions) -> tuple[float, float]:
    tg1, tg4 = _cea_common_gas_temperature_limits()
    lo = max(float(options.min_temperature), 0.8 * float(tg1))
    hi = min(float(options.max_temperature), 1.1 * float(tg4))
    if lo >= hi:
        lo = float(options.min_temperature)
        hi = float(options.max_temperature)
    return lo, hi


def _tp_temperature_polish_for_hp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    pressure: float,
    target_enthalpy: float,
    candidates: list[str] | None,
    hp_state: EquilibriumState,
    hp_result: HPSolverResult,
    hp_options: HPSolverOptions,
    condensed_options: CondensedOptions,
) -> CondensedSolveResult | None:
    """CEA-compatible HP polish using TP equilibrium as a function of T.

    A true HP equilibrium is a TP Gibbs minimum at the final temperature, with
    h(T,P,n_eq(T)) equal to the assigned reactant enthalpy.  The reduced HP
    Newton matrix can converge to a stationary composition that closes enthalpy
    but is not the TP Gibbs minimum for very carbon-rich, condensed cases.  CEA
    avoids this with additional phase/species restarts.  This polish performs
    the same mathematical check directly: re-equilibrate at the HP temperature;
    if that TP state does not have the target enthalpy, solve the scalar TP
    enthalpy equation for T and return that state.

    It is intentionally conservative.  If the TP state at the HP temperature is
    already on the assigned enthalpy, nothing is changed.
    """
    tp_options = _tp_options_from_hp_options(hp_options)
    quiet_condensed = _copy_condensed_options_quiet(condensed_options)

    cache: dict[float, CondensedSolveResult] = {}

    def solve_tp_at(T: float) -> tuple[float, CondensedSolveResult] | None:
        T = float(T)
        key = round(T, 10)
        result = cache.get(key)
        if result is None:
            result = solve_with_condensed_phases_tp(
                elements=elements,
                element_totals=element_totals,
                temperature=T,
                pressure=pressure,
                candidates=candidates,
                tp_options=tp_options,
                condensed_options=quiet_condensed,
            )
            cache[key] = result
        if not result.success:
            return None
        return _state_enthalpy(result.state) - float(target_enthalpy), result

    T0 = float(hp_state.temperature)
    initial = solve_tp_at(T0)
    if initial is None:
        return None

    f0, tp0 = initial

    # Absolute tolerance is intentionally looser than the Newton residual.  This
    # check is only deciding whether to replace the HP result with a TP root.
    polish_trigger = max(100.0, 5.0e-5 * max(1.0, abs(float(target_enthalpy))))
    if abs(f0) <= polish_trigger:
        return None

    Tmin, Tmax = _tp_hp_temperature_search_limits(hp_options)

    bracket: tuple[float, float, float, float] | None = None

    def try_direction(direction: int) -> tuple[float, float, float, float] | None:
        T_prev = T0
        f_prev = f0
        for _ in range(80):
            if direction < 0:
                T_next = max(Tmin, T_prev * 0.88)
            else:
                T_next = min(Tmax, T_prev * 1.14)
            if abs(T_next - T_prev) <= 1.0e-9:
                return None
            out = solve_tp_at(T_next)
            if out is None:
                T_prev = T_next
                continue
            f_next, _ = out
            if f_prev == 0.0 or f_prev * f_next <= 0.0:
                return (T_prev, T_next, f_prev, f_next)
            T_prev = T_next
            f_prev = f_next
        return None

    # If TP enthalpy at the HP temperature is too high, the root is normally at
    # lower T; if too low, it is normally at higher T.  Try that direction first.
    bracket = try_direction(-1 if f0 > 0.0 else 1)
    if bracket is None:
        bracket = try_direction(1 if f0 > 0.0 else -1)

    if bracket is None:
        # Last-resort global scan over the CEA accepted temperature interval.
        grid = np.unique(
            np.concatenate(
                (
                    np.linspace(Tmin, min(1200.0, Tmax), 80),
                    np.geomspace(max(Tmin, 1.0), Tmax, 80),
                )
            )
        )
        last_T = None
        last_f = None
        for T in grid:
            out = solve_tp_at(float(T))
            if out is None:
                continue
            f, _ = out
            if last_f is not None and last_f * f <= 0.0:
                bracket = (float(last_T), float(T), float(last_f), float(f))
                break
            last_T = float(T)
            last_f = float(f)

    if bracket is None:
        return None

    Ta, Tb, fa, fb = bracket
    best_T = Ta
    best_f = fa
    best_result = cache.get(round(Ta, 10))

    if abs(fb) < abs(best_f):
        best_T = Tb
        best_f = fb
        best_result = cache.get(round(Tb, 10))

    # Bisection is robust across phase changes and avoids scipy dependency here.
    for _ in range(80):
        Tm = 0.5 * (Ta + Tb)
        out = solve_tp_at(Tm)
        if out is None:
            break
        fm, rm = out
        if abs(fm) < abs(best_f):
            best_T = Tm
            best_f = fm
            best_result = rm
        if abs(fm) <= hp_options.enthalpy_tolerance:
            break
        if fa * fm <= 0.0:
            Tb = Tm
            fb = fm
        else:
            Ta = Tm
            fa = fm
        if abs(Tb - Ta) <= 1.0e-7 * max(1.0, abs(Tm)):
            break

    if best_result is None:
        return None

    # Only replace the HP Newton result if this is a real improvement.
    hp_h_error = abs(float(hp_result.enthalpy_error))
    tp_h_error = abs(float(best_f))
    tp_current_error = abs(float(f0))

    if tp_h_error > min(tp_current_error, max(hp_h_error, polish_trigger)):
        return None

    polished_state = best_result.state
    element_error = polished_state.species.A @ polished_state.n - polished_state.element_totals
    max_element_error = float(np.max(np.abs(element_error))) if element_error.size else 0.0

    message = "HP equilibrium converged by CEA-style TP temperature polish."
    tg1, tg4 = _cea_common_gas_temperature_limits()
    if not (tg1 <= polished_state.temperature <= tg4):
        message = (
            "HP equilibrium temperature is outside the normal CEA thermo range; "
            "returned CEA-style TP-polished warning equilibrium."
        )

    polished_last = HPSolverResult(
        state=polished_state,
        success=True,
        message=message,
        iterations=hp_result.iterations + best_result.inner_iterations,
        max_element_error=max_element_error,
        enthalpy_error=float(best_f),
        max_correction=0.0,
        temperature_correction=0.0,
        residual_norm=getattr(best_result.last_solver_result, "residual_norm", 0.0),
        element_potentials=getattr(best_result.last_solver_result, "element_potentials", None),
    )

    return CondensedSolveResult(
        state=polished_state,
        success=True,
        message=message,
        outer_iterations=best_result.outer_iterations,
        inner_iterations=polished_last.iterations,
        inserted_species=list(best_result.inserted_species),
        removed_species=list(best_result.removed_species),
        last_solver_result=polished_last,
    )

def solve_with_condensed_phases_hp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    pressure: float,
    target_enthalpy: float,
    guess_temperature: float = 3800.0,
    candidates: list[str] | None = None,
    hp_options: HPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    if hp_options is None:
        hp_options = HPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    active_species = _gas_only_species_set(
        elements=elements,
        temperature=guess_temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    dormant_condensed = _all_condensed_species_set(
        elements=elements,
        temperature=None,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    state = initial_hp_state(
        species=active_species,
        element_totals=element_totals,
        pressure=pressure,
        guess_temperature=guess_temperature,
        trace=hp_options.trace,
    )

    inserted: list[str] = []
    removed: list[str] = []
    inner_iterations = 0
    last_result: HPSolverResult | None = None

    for outer in range(1, condensed_options.max_outer_iterations + 1):
        last_result = solve_hp(
            state,
            target_enthalpy=target_enthalpy,
            options=hp_options,
        )
        inner_iterations += last_result.iterations

        # IMPORTANT:
        # Always continue condensed-phase reconciliation/insertion testing,
        # even if solve_hp reports success. In CEA-style out-of-range HP,
        # solve_hp can return success at Tmin before all stable condensed
        # phases have been inserted.
        state = last_result.state

        state, phase_changed = reconcile_condensed_phases(
            state,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )

        if phase_changed:
            continue

        remove_now = condensed_species_to_remove(
            state,
            tolerance=condensed_options.removal_tolerance,
        )

        if remove_now:
            active_species = remove_species_from_set(state.species, remove_now)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_now)

            if condensed_options.verbose:
                print(f"Removed condensed species: {remove_now}")

            continue

        remove_invalid = invalid_trace_species_to_remove(state)

        if remove_invalid:
            active_species = remove_species_from_set(state.species, remove_invalid)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_invalid)

            if condensed_options.verbose:
                print(f"Removed invalid trace species: {remove_invalid}")

            continue

        if condensed_options.enabled:
            dormant_names = [
                name
                for name in dormant_condensed.names
                if name not in state.species.name_to_index
            ]

            if dormant_names:
                from .species import subset_species_set

                keep = [
                    i
                    for i, name in enumerate(dormant_condensed.names)
                    if name in dormant_names
                ]

                dormant_subset = subset_species_set(dormant_condensed, keep)

                tests = condensed_gibbs_test_values(
                    active_state=state,
                    dormant_condensed_species=dormant_subset,
                    element_potentials=getattr(last_result, "element_potentials", None),
                )

                chosen = choose_condensed_species_to_insert(
                    tests,
                    tolerance=condensed_options.insertion_tolerance,
                )

                if chosen is not None:
                    active_species = add_species_to_set(
                        state.species,
                        [chosen.species_name],
                    )

                    state = _transfer_state_to_species_set(
                        state,
                        active_species,
                        initial_new_moles=condensed_options.initial_condensed_moles,
                    )

                    inserted.append(chosen.species_name)

                    if condensed_options.verbose:
                        print(
                            f"Inserted condensed species {chosen.species_name} "
                            f"Gtest={chosen.gibbs_test_value:.6e}"
                        )

                    continue

        if last_result.success:
            polished = _tp_temperature_polish_for_hp(
                elements=elements,
                element_totals=element_totals,
                pressure=pressure,
                target_enthalpy=target_enthalpy,
                candidates=candidates,
                hp_state=state,
                hp_result=last_result,
                hp_options=hp_options,
                condensed_options=condensed_options,
            )

            if polished is not None:
                return CondensedSolveResult(
                    state=polished.state,
                    success=True,
                    message=polished.message,
                    outer_iterations=outer + polished.outer_iterations,
                    inner_iterations=inner_iterations + polished.inner_iterations,
                    inserted_species=list(dict.fromkeys(inserted + polished.inserted_species)),
                    removed_species=list(dict.fromkeys(removed + polished.removed_species)),
                    last_solver_result=polished.last_solver_result,
                )

            tg1, tg4 = _cea_common_gas_temperature_limits()
            message = "HP equilibrium converged with stable condensed-phase set."
            if not (tg1 <= state.temperature <= tg4):
                message = (
                    "HP equilibrium temperature is outside the normal CEA thermo "
                    "range; converged within CEA accepted warning range."
                )

            return CondensedSolveResult(
                state=state,
                success=True,
                message=message,
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        if _is_hp_lower_temperature_unresolved(last_result, hp_options):
            warning_temperature = _cea_extended_low_temperature_for_hp_warning()
            tg1, tg4 = _cea_common_gas_temperature_limits()

            if (0.8 * tg1) <= warning_temperature <= (1.1 * tg4):
                tp_warning = solve_with_condensed_phases_tp(
                    elements=elements,
                    element_totals=element_totals,
                    temperature=warning_temperature,
                    pressure=pressure,
                    candidates=candidates,
                    tp_options=_tp_options_from_hp_options(hp_options),
                    condensed_options=condensed_options,
                )

                if tp_warning.success:
                    return CondensedSolveResult(
                        state=tp_warning.state,
                        success=True,
                        message=(
                            "HP equilibrium temperature is outside the normal CEA "
                            "thermo range; returned CEA-style extended-range "
                            "warning equilibrium at TP with assigned HP enthalpy."
                        ),
                        outer_iterations=outer + tp_warning.outer_iterations,
                        inner_iterations=inner_iterations + tp_warning.inner_iterations,
                        inserted_species=list(
                            dict.fromkeys(inserted + tp_warning.inserted_species)
                        ),
                        removed_species=list(
                            dict.fromkeys(removed + tp_warning.removed_species)
                        ),
                        last_solver_result=last_result,
                    )

        return CondensedSolveResult(
            state=state,
            success=False,
            message=last_result.message,
            outer_iterations=outer,
            inner_iterations=inner_iterations,
            inserted_species=inserted,
            removed_species=removed,
            last_solver_result=last_result,
        )

    return CondensedSolveResult(
        state=state,
        success=False,
        message="Condensed-phase HP outer loop exceeded max iterations.",
        outer_iterations=condensed_options.max_outer_iterations,
        inner_iterations=inner_iterations,
        inserted_species=inserted,
        removed_species=removed,
        last_solver_result=last_result,
    )


def _tp_sp_temperature_search_limits(options: SPSolverOptions) -> tuple[float, float]:
    tg1, tg4 = _cea_common_gas_temperature_limits()
    lo = max(float(options.min_temperature), 0.8 * float(tg1))
    hi = min(float(options.max_temperature), 1.1 * float(tg4))
    if lo >= hi:
        lo = float(options.min_temperature)
        hi = float(options.max_temperature)
    return lo, hi


def solve_with_condensed_phases_sp(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    pressure: float,
    target_entropy: float,
    guess_temperature: float = 3800.0,
    candidates: list[str] | None = None,
    sp_options: SPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    """Solve native SP equilibrium with CEA-style condensed phases.

    This is the native SP path: composition and temperature are corrected
    simultaneously by the fixed-species SP matrix in :mod:`sp_solver`, while
    this wrapper performs the same condensed-phase insertion/removal loop used
    by TP and HP.

    If the native matrix fails for a difficult phase-boundary case, the slower
    TP entropy-root implementation remains available internally as
    ``solve_with_condensed_phases_sp_root`` and is used as a conservative
    fallback so existing ThermoProp capabilities are preserved.
    """
    if sp_options is None:
        sp_options = SPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    if pressure <= 0.0:
        raise ValueError("SP equilibrium requires positive pressure [Pa].")

    if not np.isfinite(target_entropy):
        raise ValueError("SP equilibrium requires a finite target entropy [J/kg-K].")

    active_species = _gas_only_species_set(
        elements=elements,
        temperature=guess_temperature,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    dormant_condensed = _all_condensed_species_set(
        elements=elements,
        temperature=None,
        candidates=candidates,
        include_ions=condensed_options.include_ions,
        include_electron=condensed_options.include_electron,
    )

    state = initial_sp_state(
        species=active_species,
        element_totals=element_totals,
        pressure=pressure,
        guess_temperature=guess_temperature,
        trace=sp_options.trace,
    )

    inserted: list[str] = []
    removed: list[str] = []
    inner_iterations = 0
    last_result: SPSolverResult | None = None

    for outer in range(1, condensed_options.max_outer_iterations + 1):
        last_result = solve_sp(
            state,
            target_entropy=target_entropy,
            options=sp_options,
        )
        inner_iterations += last_result.iterations
        state = last_result.state

        state, phase_changed = reconcile_condensed_phases(
            state,
            initial_new_moles=condensed_options.initial_condensed_moles,
        )

        if phase_changed:
            continue

        remove_now = condensed_species_to_remove(
            state,
            tolerance=condensed_options.removal_tolerance,
        )

        if remove_now:
            active_species = remove_species_from_set(state.species, remove_now)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_now)

            if condensed_options.verbose:
                print(f"Removed condensed species: {remove_now}")

            continue

        remove_invalid = invalid_trace_species_to_remove(state)

        if remove_invalid:
            active_species = remove_species_from_set(state.species, remove_invalid)
            state = _transfer_state_to_species_set(state, active_species)
            removed.extend(remove_invalid)

            if condensed_options.verbose:
                print(f"Removed invalid trace species: {remove_invalid}")

            continue

        if condensed_options.enabled:
            dormant_names = [
                name
                for name in dormant_condensed.names
                if name not in state.species.name_to_index
            ]

            if dormant_names:
                from .species import subset_species_set

                keep = [
                    i
                    for i, name in enumerate(dormant_condensed.names)
                    if name in dormant_names
                ]

                dormant_subset = subset_species_set(dormant_condensed, keep)

                tests = condensed_gibbs_test_values(
                    active_state=state,
                    dormant_condensed_species=dormant_subset,
                    element_potentials=getattr(last_result, "element_potentials", None),
                )

                chosen = choose_condensed_species_to_insert(
                    tests,
                    tolerance=condensed_options.insertion_tolerance,
                )

                if chosen is not None:
                    active_species = add_species_to_set(
                        state.species,
                        [chosen.species_name],
                    )

                    state = _transfer_state_to_species_set(
                        state,
                        active_species,
                        initial_new_moles=condensed_options.initial_condensed_moles,
                    )

                    inserted.append(chosen.species_name)

                    if condensed_options.verbose:
                        print(
                            f"Inserted condensed species {chosen.species_name} "
                            f"Gtest={chosen.gibbs_test_value:.6e}"
                        )

                    continue

        if last_result.success:
            return CondensedSolveResult(
                state=state,
                success=True,
                message="SP equilibrium converged with stable condensed-phase set.",
                outer_iterations=outer,
                inner_iterations=inner_iterations,
                inserted_species=inserted,
                removed_species=removed,
                last_solver_result=last_result,
            )

        # Preserve existing capability for difficult SP states by falling back to
        # the old TP entropy-root path.  Normal nozzle-map states should use the
        # native path above and never reach this branch.
        fallback = solve_with_condensed_phases_sp_root(
            elements=elements,
            element_totals=element_totals,
            pressure=pressure,
            target_entropy=target_entropy,
            guess_temperature=state.temperature,
            candidates=candidates,
            sp_options=sp_options,
            condensed_options=condensed_options,
        )

        if fallback.success:
            return CondensedSolveResult(
                state=fallback.state,
                success=True,
                message="SP native matrix failed; converged by TP entropy-root fallback.",
                outer_iterations=outer + fallback.outer_iterations,
                inner_iterations=inner_iterations + fallback.inner_iterations,
                inserted_species=list(dict.fromkeys(inserted + fallback.inserted_species)),
                removed_species=list(dict.fromkeys(removed + fallback.removed_species)),
                last_solver_result=fallback.last_solver_result,
            )

        return CondensedSolveResult(
            state=state,
            success=False,
            message=last_result.message,
            outer_iterations=outer,
            inner_iterations=inner_iterations,
            inserted_species=inserted,
            removed_species=removed,
            last_solver_result=last_result,
        )

    return CondensedSolveResult(
        state=state,
        success=False,
        message="Condensed-phase SP outer loop exceeded max iterations.",
        outer_iterations=condensed_options.max_outer_iterations,
        inner_iterations=inner_iterations,
        inserted_species=inserted,
        removed_species=removed,
        last_solver_result=last_result,
    )


def solve_with_condensed_phases_sp_root(
    *,
    elements: list[str],
    element_totals: np.ndarray,
    pressure: float,
    target_entropy: float,
    guess_temperature: float = 3800.0,
    candidates: list[str] | None = None,
    sp_options: SPSolverOptions | None = None,
    condensed_options: CondensedOptions | None = None,
) -> CondensedSolveResult:
    """Solve SP equilibrium by a CEA-style TP temperature root.

    For an assigned pressure and entropy, a stable equilibrium state can be
    represented as the TP Gibbs minimum at the temperature whose mixture entropy
    matches ``target_entropy``.  This implementation deliberately reuses the TP
    solver and condensed-phase insertion/removal logic at each trial
    temperature.  That makes SP immediately consistent with ThermoProp's current
    TP/HP species screening and CEA-style condensed phase handling.
    """
    if sp_options is None:
        sp_options = SPSolverOptions()

    if condensed_options is None:
        condensed_options = CondensedOptions()

    if pressure <= 0.0:
        raise ValueError("SP equilibrium requires positive pressure [Pa].")

    if not np.isfinite(target_entropy):
        raise ValueError("SP equilibrium requires a finite target entropy [J/kg-K].")

    tp_options = _tp_options_from_sp_options(sp_options)
    quiet_condensed = _copy_condensed_options_quiet(condensed_options)
    Tmin, Tmax = _tp_sp_temperature_search_limits(sp_options)

    cache: dict[float, CondensedSolveResult] = {}
    evaluation_count = 0
    inner_iterations_total = 0

    def solve_tp_at(T: float) -> tuple[float, CondensedSolveResult] | None:
        nonlocal evaluation_count, inner_iterations_total

        T = float(np.clip(float(T), Tmin, Tmax))
        key = round(T, 10)
        result = cache.get(key)

        if result is None:
            result = solve_with_condensed_phases_tp(
                elements=elements,
                element_totals=element_totals,
                temperature=T,
                pressure=pressure,
                candidates=candidates,
                tp_options=tp_options,
                condensed_options=quiet_condensed,
            )
            cache[key] = result
            evaluation_count += 1
            inner_iterations_total += int(result.inner_iterations)

        if not result.success:
            return None

        return float(_state_entropy(result.state) - target_entropy), result

    T0 = float(np.clip(float(guess_temperature), Tmin, Tmax))
    initial = solve_tp_at(T0)

    if initial is None:
        # If the user's guess is poor, start from a broad grid before giving up.
        initial = None
        for T in np.unique(
            np.concatenate(
                (
                    np.linspace(Tmin, min(1200.0, Tmax), 40),
                    np.geomspace(max(Tmin, 1.0), Tmax, 50),
                )
            )
        ):
            initial = solve_tp_at(float(T))
            if initial is not None:
                T0 = float(T)
                break

    if initial is None:
        dummy_state = next(iter(cache.values())).state if cache else None
        if dummy_state is None:
            raise RuntimeError("SP equilibrium could not evaluate any TP trial state.")
        last = next(iter(cache.values())).last_solver_result
        return CondensedSolveResult(
            state=dummy_state,
            success=False,
            message="SP equilibrium could not evaluate any successful TP trial state.",
            outer_iterations=0,
            inner_iterations=inner_iterations_total,
            inserted_species=[],
            removed_species=[],
            last_solver_result=SPSolverResult(
                state=dummy_state,
                success=False,
                message="SP equilibrium could not evaluate any successful TP trial state.",
                iterations=evaluation_count,
                max_element_error=getattr(last, "max_element_error", np.inf),
                entropy_error=np.inf,
                max_correction=np.inf,
                temperature_correction=np.inf,
                residual_norm=getattr(last, "residual_norm", np.inf),
                element_potentials=getattr(last, "element_potentials", None),
            ),
        )

    f0, tp0 = initial
    best_T = T0
    best_f = f0
    best_result = tp0

    if abs(best_f) <= sp_options.entropy_tolerance:
        state = best_result.state
        element_error = state.species.A @ state.n - state.element_totals
        max_element_error = float(np.max(np.abs(element_error))) if element_error.size else 0.0
        last = best_result.last_solver_result
        return CondensedSolveResult(
            state=state,
            success=True,
            message="SP equilibrium converged by TP entropy root.",
            outer_iterations=best_result.outer_iterations,
            inner_iterations=inner_iterations_total,
            inserted_species=list(best_result.inserted_species),
            removed_species=list(best_result.removed_species),
            last_solver_result=SPSolverResult(
                state=state,
                success=True,
                message="SP equilibrium converged by TP entropy root.",
                iterations=evaluation_count,
                max_element_error=max_element_error,
                entropy_error=float(best_f),
                max_correction=0.0,
                temperature_correction=0.0,
                residual_norm=getattr(last, "residual_norm", 0.0),
                element_potentials=getattr(last, "element_potentials", None),
            ),
        )

    bracket: tuple[float, float, float, float] | None = None

    def try_direction(direction: int) -> tuple[float, float, float, float] | None:
        T_prev = T0
        f_prev = f0

        for _ in range(sp_options.max_bracket_iterations):
            if direction < 0:
                T_next = max(Tmin, T_prev * 0.88)
            else:
                T_next = min(Tmax, T_prev * 1.14)

            if abs(T_next - T_prev) <= 1.0e-12 * max(1.0, abs(T_prev)):
                return None

            out = solve_tp_at(T_next)
            if out is None:
                T_prev = T_next
                continue

            f_next, result_next = out
            if abs(f_next) < abs(best_f):
                nonlocal_best[0] = T_next
                nonlocal_best[1] = f_next
                nonlocal_best[2] = result_next

            if f_prev == 0.0 or f_prev * f_next <= 0.0:
                return (T_prev, T_next, f_prev, f_next)

            T_prev = T_next
            f_prev = f_next

        return None

    # Mutable holder used because Python does not allow nonlocal assignment to
    # variables after using them directly in this nested function in older code
    # style checks.
    nonlocal_best = [best_T, best_f, best_result]

    # Entropy usually increases with T at fixed P, so if f0 is high the root is
    # usually colder; if f0 is low the root is usually hotter.  Try that first.
    bracket = try_direction(-1 if f0 > 0.0 else 1)
    best_T, best_f, best_result = nonlocal_best

    if bracket is None:
        nonlocal_best = [best_T, best_f, best_result]
        bracket = try_direction(1 if f0 > 0.0 else -1)
        best_T, best_f, best_result = nonlocal_best

    if bracket is None:
        grid = np.unique(
            np.concatenate(
                (
                    np.linspace(Tmin, min(1200.0, Tmax), 100),
                    np.geomspace(max(Tmin, 1.0), Tmax, 120),
                    np.array([T0], dtype=float),
                )
            )
        )

        last_T = None
        last_f = None

        for T in grid:
            out = solve_tp_at(float(T))
            if out is None:
                continue

            f, result = out
            if abs(f) < abs(best_f):
                best_T = float(T)
                best_f = float(f)
                best_result = result

            if last_f is not None and last_f * f <= 0.0:
                bracket = (float(last_T), float(T), float(last_f), float(f))
                break

            last_T = float(T)
            last_f = float(f)

    if bracket is not None:
        Ta, Tb, fa, fb = bracket

        # Bisection is robust across condensed-phase switches and avoids adding
        # a dependency at this layer.
        for _ in range(sp_options.max_iterations):
            Tm = 0.5 * (Ta + Tb)
            out = solve_tp_at(Tm)

            if out is None:
                break

            fm, rm = out

            if abs(fm) < abs(best_f):
                best_T = Tm
                best_f = fm
                best_result = rm

            if abs(fm) <= sp_options.entropy_tolerance:
                break

            if fa * fm <= 0.0:
                Tb = Tm
                fb = fm
            else:
                Ta = Tm
                fa = fm

            if abs(Tb - Ta) <= sp_options.temperature_correction_tolerance * max(1.0, abs(Tm)):
                break

    state = best_result.state
    element_error = state.species.A @ state.n - state.element_totals
    max_element_error = float(np.max(np.abs(element_error))) if element_error.size else 0.0
    last = best_result.last_solver_result
    success = abs(float(best_f)) <= max(
        sp_options.entropy_tolerance,
        1.0e-10 * max(1.0, abs(float(target_entropy))),
    )

    if success:
        message = "SP equilibrium converged by TP entropy root."
    elif bracket is None:
        message = "SP equilibrium could not bracket the target entropy within the temperature limits."
    else:
        message = "SP equilibrium did not meet entropy tolerance within max_iterations."

    return CondensedSolveResult(
        state=state,
        success=success,
        message=message,
        outer_iterations=best_result.outer_iterations,
        inner_iterations=inner_iterations_total,
        inserted_species=list(best_result.inserted_species),
        removed_species=list(best_result.removed_species),
        last_solver_result=SPSolverResult(
            state=state,
            success=success,
            message=message,
            iterations=evaluation_count,
            max_element_error=max_element_error,
            entropy_error=float(best_f),
            max_correction=0.0,
            temperature_correction=0.0 if bracket is None else abs(float(best_T - T0)) / max(1.0, abs(float(T0))),
            residual_norm=getattr(last, "residual_norm", 0.0),
            element_potentials=getattr(last, "element_potentials", None),
        ),
    )

def valid_condensed_phase_for_temperature(name: str, T: float) -> str | None:
    return _cea_preferred_condensed_phase(name, T)


def reconcile_condensed_phases(
    state: EquilibriumState,
    *,
    initial_new_moles: float = 1e-12,
) -> tuple[EquilibriumState, bool]:
    T = float(state.temperature)
    species = state.species
    changed = False

    for name in list(species.names):
        idx = species.name_to_index[name]

        if not species.condensed_mask[idx]:
            continue

        replacement = valid_condensed_phase_for_temperature(name, T)

        if replacement is None or replacement == name:
            continue

        old_moles = state.n[idx]

        new_species = remove_species_from_set(state.species, [name])
        new_species = add_species_to_set(new_species, [replacement])

        new_state = _transfer_state_to_species_set(
            state,
            new_species,
            initial_new_moles=initial_new_moles,
        )

        new_idx = new_state.species.name_to_index[replacement]
        new_state.n[new_idx] = max(float(old_moles), initial_new_moles)

        state = new_state
        species = state.species
        changed = True

    return state, changed