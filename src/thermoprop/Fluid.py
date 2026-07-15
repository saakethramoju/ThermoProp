from typing import List, Union, Dict, Tuple

import math
import numpy as np
from scipy.optimize import root, root_scalar
from functools import lru_cache

import CoolProp.CoolProp as CP

from .SpeciesDatabase import SpeciesDatabase
from .ReferenceState import normalize_reference_target
from ._api import PropertyIntrospectionMixin
from ._formatting import format_optional, rounded_dict, format_rows
from ._validation import validate_fraction_vector
from ._state_api import UNSET, is_provided, provided_items
from ._composition import composition_dict
from .Exceptions import ThermoPropFlashError, SpeciesLookupError, ThermoPropConfigurationError

class Fluid(PropertyIntrospectionMixin):
    """
    CoolProp real-fluid property wrapper with a consistent ThermoProp API.

    Fluid supports pure fluids and mixtures using CoolProp's HEOS backend. It is
    the preferred wrapper for real-fluid phase behavior, saturation states, and
    two-phase properties.

    Supported state pairs
    ---------------------

    Exactly two thermodynamic inputs are required:

        Fluid(..., pressure=P, temperature=T)
        Fluid(..., pressure=P, enthalpy=h)
        Fluid(..., pressure=P, quality=x)
        Fluid(..., temperature=T, quality=x)
        Fluid(..., density=rho, internal_energy=u)
        Fluid(..., pressure=P, density=rho)
        Fluid(..., pressure=P, internal_energy=u)
        Fluid(..., temperature=T, density=rho)
        Fluid(..., density=rho, enthalpy=h)
        Fluid(..., temperature=T, enthalpy=h)

    Mixtures
    --------

    Mixtures are passed as dictionaries:

        Fluid({"N2": 0.75, "O2": 0.25}, basis="mass", pressure=P, temperature=T)

    The basis may be "mass" or "mole".

    Reference states
    ----------------

    CoolProp uses its own thermodynamic reference state. Absolute enthalpy,
    internal energy, entropy, Gibbs energy, and Helmholtz/free energy should not be
    compared directly with other wrappers unless a common reference is selected.

    Use set_reference="IdealGas", "Propellant", or "CombustionGas" to apply
    constant offsets at 298.15 K and 101325 Pa.

    This aligns only the reference values. It does not make CoolProp, PYroMat,
    RocketProps, or CEA models equivalent away from the reference state.

    Public API units are SI.
    """
    _BACKEND_NAME = "CoolProp"

    _REFERENCE_TEMPERATURE = 298.15
    _REFERENCE_PRESSURE = 101325.0
    _REFERENCE_CACHE: dict[tuple, tuple[float, float, float]] = {}


    _UNSUPPORTED_PROPERTIES = set()

    _PHASE_NAMES = {
        getattr(CP, "iphase_unknown", -999): "Unknown",
        getattr(CP, "iphase_liquid", -999): "Liquid",
        getattr(CP, "iphase_supercritical", -999): "Supercritical",
        getattr(CP, "iphase_supercritical_gas", -999): "SupercriticalGas",
        getattr(CP, "iphase_supercritical_liquid", -999): "SupercriticalLiquid",
        getattr(CP, "iphase_gas", -999): "Gas",
        getattr(CP, "iphase_twophase", -999): "TwoPhase",
        getattr(CP, "iphase_critical_point", -999): "CriticalPoint",
    }

    _FLASH_PAIRS = {
        frozenset(("pressure", "temperature")): (
            "PT_INPUTS",
            ("pressure", "temperature"),
        ),
        frozenset(("pressure", "enthalpy")): (
            "HmassP_INPUTS",
            ("enthalpy", "pressure"),
        ),
        frozenset(("pressure", "quality")): (
            "PQ_INPUTS",
            ("pressure", "quality"),
        ),
        frozenset(("temperature", "quality")): (
            "QT_INPUTS",
            ("quality", "temperature"),
        ),
        frozenset(("density", "internal_energy")): (
            "DmassUmass_INPUTS",
            ("density", "internal_energy"),
        ),
        frozenset(("pressure", "density")): (
            "DmassP_INPUTS",
            ("density", "pressure"),
        ),
        frozenset(("pressure", "internal_energy")): (
            "PUmass_INPUTS",
            ("pressure", "internal_energy"),
        ),
        frozenset(("temperature", "density")): (
            "DmassT_INPUTS",
            ("density", "temperature"),
        ),
        frozenset(("density", "enthalpy")): (
            "DmassHmass_INPUTS",
            ("density", "enthalpy"),
        ),
        frozenset(("temperature", "enthalpy")): (
            "HmassT_INPUTS",
            ("enthalpy", "temperature"),
        ),
        frozenset(("pressure", "entropy")): (
            "PSmass_INPUTS",
            ("pressure", "entropy"),
        ),
        frozenset(("temperature", "entropy")): (
            "SmassT_INPUTS",
            ("entropy", "temperature"),
        ),
        frozenset(("enthalpy", "entropy")): (
            "HmassSmass_INPUTS",
            ("enthalpy", "entropy"),
        ),
        frozenset(("density", "entropy")): (
            "DmassSmass_INPUTS",
            ("density", "entropy"),
        ),
        frozenset(("internal_energy", "entropy")): (
            "SmassUmass_INPUTS",
            ("entropy", "internal_energy"),
        ),
        frozenset(("quality", "entropy")): (
            "QSmass_INPUTS",
            ("quality", "entropy"),
        ),
    }

    def __init__(
        self,
        fluid: Union[str, Dict[str, float]],
        basis: str = "mass",
        pressure: float = None,
        enthalpy: float = None,
        temperature: float = None,
        quality: float = None,
        density: float = None,
        internal_energy: float = None,
        entropy: float = None,
        set_reference: str | None = None,
    ):
        """Initialize a CoolProp-backed real-fluid state.

        Parameters use SI units.  ``fluid`` may be a canonical ThermoProp species name, a supported alias, or a composition dictionary.  Composition dictionaries use ``basis="mass"`` or ``basis="mole"`` and must contain finite nonnegative fractions that sum to one after normalization.  Exactly one supported thermodynamic flash pair must be supplied, such as pressure-temperature, pressure-enthalpy, pressure-quality, temperature-quality, pressure-density, pressure-internal-energy, temperature-density, density-enthalpy, density-internal-energy, or temperature-enthalpy.

        ``set_reference`` optionally aligns absolute enthalpy, internal energy, and entropy to another ThermoProp wrapper at 298.15 K and 101325 Pa.  Name resolution, composition validation, and flash validation happen during construction so invalid states fail early with ThermoProp exceptions.
        """
        self._reference_target = self._normalize_reference_target(set_reference)
        self._reference_offsets: tuple[float, float, float] | None = None
        basis = str(basis).lower().strip()
        if basis not in ("mass", "mole"):
            raise ThermoPropConfigurationError("basis must be 'mass' or 'mole'.")
        self._basis = basis

        valid_fluids = Fluid.get_available_fluids()

        self._fluids: List[str] = []
        self._display_names: List[str] = []

        if isinstance(fluid, str):
            backend, display = Fluid._normalize_name(fluid)
            if backend not in valid_fluids:
                raise SpeciesLookupError(
                    f"Invalid fluid '{fluid}'. "
                    f"Use Fluid.show_available_fluids() to check valid names."
                )
            self._fluids = [backend]
            self._display_names = [display]
            self._mole_fractions = np.array([1.0])
            self._mass_fractions = np.array([1.0])
            self._mixture = False

        elif isinstance(fluid, dict):
            if len(fluid) == 1:
                f, frac = next(iter(fluid.items()))
                frac = float(frac)

                if not np.isfinite(frac):
                    raise ValueError("Single-component fraction must be finite")

                if frac < 0.0:
                    raise ValueError("Single-component fraction must be nonnegative")

                if not np.isclose(frac, 1.0, atol=1e-12):
                    raise ValueError(f"Single-component dict must have fraction = 1.0, got {frac}")

                backend, display = Fluid._normalize_name(f)
                if backend not in valid_fluids:
                    raise SpeciesLookupError(
                        f"Invalid fluid '{f}'. "
                        f"Use Fluid.show_available_fluids() to check valid names."
                    )
                self._fluids = [backend]
                self._display_names = [display]
                self._mole_fractions = np.array([1.0])
                self._mass_fractions = np.array([1.0])
                self._mixture = False

            else:
                tmp: Dict[str, Tuple[float, List[str]]] = {}
                for user_name, frac in fluid.items():
                    backend, display = Fluid._normalize_name(user_name)
                    if backend not in valid_fluids:
                        raise SpeciesLookupError(
                            f"Invalid fluid '{user_name}' (backend '{backend}' not found). "
                            f"Use Fluid.show_available_fluids() to check valid names."
                        )
                    total, names = tmp.get(backend, (0.0, []))
                    tmp[backend] = (total + float(frac), names + [display])

                self._fluids = list(tmp.keys())
                fractions = np.array([v[0] for v in tmp.values()], dtype=float)
                self._display_names = [", ".join(sorted(set(names))) for _, names in tmp.values()]

                if basis == "mole":
                    fractions = self._validate_fractions(fractions, "Mole fractions")
                    self._mole_fractions = fractions
                    self._mass_fractions = Fluid.mole_to_mass(self._fluids, fractions)
                elif basis == "mass":
                    fractions = self._validate_fractions(fractions, "Mass fractions")
                    self._mass_fractions = fractions
                    self._mole_fractions = Fluid.mass_to_mole(self._fluids, fractions)
                else:
                    raise ThermoPropConfigurationError("basis must be 'mole' or 'mass'.")

                self._mixture = len(self._fluids) > 1

        else:
            raise TypeError("fluid must be a string (pure) or dict (mixture)")

        self._P = None
        self._h = None
        self._last_state_values: dict | None = None
        self._fluid_string = "&".join(self._fluids)
        self._clear_reference_cache()
        self._backend = self._build_state()
        self._pyfluid = self._backend

        flash_values = {
            "pressure": pressure,
            "enthalpy": enthalpy,
            "temperature": temperature,
            "quality": quality,
            "density": density,
            "internal_energy": internal_energy,
            "entropy": entropy,
        }

        provided = {
            key: value
            for key, value in flash_values.items()
            if value is not None
        }

        if len(provided) != 2:
            raise ThermoPropFlashError(
                "Please provide exactly two thermodynamic properties. "
                "Supported names are pressure, temperature, enthalpy, "
                "quality, density, internal_energy, and entropy."
            )

        self._set_state_from_named_pair(provided)


    def update(
        self,
        fluid=UNSET,
        *,
        basis=UNSET,
        pressure=UNSET,
        enthalpy=UNSET,
        temperature=UNSET,
        quality=UNSET,
        density=UNSET,
        internal_energy=UNSET,
        entropy=UNSET,
        mole_fractions=UNSET,
        mass_fractions=UNSET,
        set_reference=UNSET,
    ):
        """Update composition and/or thermodynamic state in place.

        ``update`` is the batched counterpart to the simple property setters.
        It is useful in iterative solvers because multiple inputs can be changed
        once before CoolProp is flashed again.  Calling with a single state value
        preserves the existing setter behavior; calling with a supported pair
        flashes directly from that pair.

        Structural changes such as a new fluid list or reference target rebuild
        the wrapper and preserve the current state unless a new state pair is
        supplied.
        """

        state_updates = provided_items(
            {
                "pressure": pressure,
                "enthalpy": enthalpy,
                "temperature": temperature,
                "quality": quality,
                "density": density,
                "internal_energy": internal_energy,
                "entropy": entropy,
            }
        )

        structural = any(is_provided(v) for v in (fluid, basis, set_reference))

        if structural:
            new_basis = self._basis if not is_provided(basis) else str(basis).lower().strip()
            if new_basis not in ("mass", "mole"):
                raise ThermoPropConfigurationError("basis must be 'mass' or 'mole'.")

            if is_provided(fluid):
                new_fluid = fluid
            elif len(self._display_names) == 1:
                new_fluid = self._display_names[0]
            else:
                fractions = self._mole_fractions if new_basis == "mole" else self._mass_fractions
                new_fluid = {name: float(value) for name, value in zip(self._display_names, fractions)}

            new_reference = self._reference_target if not is_provided(set_reference) else set_reference

            if len(state_updates) >= 2:
                state_values = state_updates
            else:
                state_values = dict(self._last_state_values or {})
                state_values.update(state_updates)

            rebuilt = self.__class__(
                new_fluid,
                basis=new_basis,
                set_reference=new_reference,
                **state_values,
            )
            self.__dict__.update(rebuilt.__dict__)
            return self

        if is_provided(mole_fractions):
            self.mole_fractions = mole_fractions

        if is_provided(mass_fractions):
            self.mass_fractions = mass_fractions

        if len(state_updates) == 1:
            name, value = next(iter(state_updates.items()))
            setattr(self, name, value)
        elif len(state_updates) > 1:
            self._set_state_from_named_pair(state_updates)

        return self


    # ---------------- Reference-state matching ---------------- #

    @classmethod
    def _normalize_reference_target(cls, value):
        return normalize_reference_target(value, cls.__name__)

    @property
    def reference(self) -> str:
        """Return the active thermodynamic reference-state alignment target for this ``Fluid`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return self._reference_target or self.__class__.__name__

    @property
    def set_reference(self) -> str:
        """Return the reference-state alignment target setter for this ``Fluid`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return self.reference

    def _composition_cache_key(self) -> tuple:
        return tuple(
            sorted(
                (name, round(float(w), 15))
                for name, w in zip(self._display_names, self._mass_fractions)
            )
        )

    def _composition_argument(self) -> str | dict[str, float]:
        if len(self._display_names) == 1:
            return self._display_names[0]

        return {
            name: float(w)
            for name, w in zip(self._display_names, self._mass_fractions)
        }

    def _reference_cache_key(self) -> tuple:
        return (
            "Fluid",
            self._reference_target,
            self._composition_cache_key(),
            self._REFERENCE_TEMPERATURE,
            self._REFERENCE_PRESSURE,
        )


    def cache_key(self) -> tuple:
        """Execute the documented ``cache_key`` operation for ``Fluid``.

        Arguments are validated and normalized using the same rules as the high-level
        wrappers.  Return values follow ThermoProp's SI-unit and composition
        conventions, and failures are reported through ThermoProp exception types with
        contextual messages rather than silent fallbacks.
        """

        state = self._last_state_values or {}
        return (
            "Fluid",
            self._basis,
            self._reference_target,
            self._composition_cache_key(),
            tuple(
                sorted(
                    (key, None if value is None else round(float(value), 12))
                    for key, value in state.items()
                )
            ),
        )

    def _raw_reference_properties(self) -> tuple[float, float, float]:
        obj = self.__class__(
            self._composition_argument(),
            basis="mass",
            pressure=self._REFERENCE_PRESSURE,
            temperature=self._REFERENCE_TEMPERATURE,
            set_reference=None,
        )
        return float(obj._h), float(obj._backend.umass()), float(obj._backend.smass())

    def _target_reference_properties(self) -> tuple[float, float, float]:
        target = self._reference_target

        if target is None:
            return self._raw_reference_properties()

        fluid = self._composition_argument()
        T = self._REFERENCE_TEMPERATURE
        P = self._REFERENCE_PRESSURE

        if target == "IdealGas":
            from .IdealGas import IdealGas
            obj = IdealGas(fluid, basis="mass", pressure=P, temperature=T, set_reference=None)
        elif target == "CombustionGas":
            from .CombustionGas import CombustionGas
            obj = CombustionGas(fluid, basis="mass", pressure=P, temperature=T, set_reference=None)
        elif target == "Propellant":
            from .Propellant import Propellant
            if not isinstance(fluid, str):
                raise ValueError("set_reference='Propellant' is only supported for pure Fluid species.")
            obj = Propellant(fluid, pressure=P, temperature=T, set_reference=None)
        else:
            raise ValueError(f"Unsupported reference target: {target!r}")

        return float(obj.enthalpy), float(obj.internal_energy), float(obj.entropy)

    def _get_reference_offsets(self) -> tuple[float, float, float]:
        if self._reference_target is None:
            return 0.0, 0.0, 0.0

        if self._reference_offsets is not None:
            return self._reference_offsets

        key = self._reference_cache_key()
        cached = self._REFERENCE_CACHE.get(key)
        if cached is not None:
            self._reference_offsets = cached
            return cached

        raw_h, raw_u, raw_s = self._raw_reference_properties()
        ref_h, ref_u, ref_s = self._target_reference_properties()
        offsets = (ref_h - raw_h, ref_u - raw_u, ref_s - raw_s)
        self._REFERENCE_CACHE[key] = offsets
        self._reference_offsets = offsets
        return offsets

    def _clear_reference_cache(self) -> None:
        self._reference_offsets = None

    def _to_raw_basis(self, name: str, value: float) -> float:
        if self._reference_target is None:
            return float(value)
        dh, du, ds = self._get_reference_offsets()
        if name == "enthalpy":
            return float(value) - dh
        if name == "internal_energy":
            return float(value) - du
        if name == "entropy":
            return float(value) - ds
        return float(value)

    def _from_raw_basis(self, name: str, value: float | None) -> float | None:
        if value is None:
            return None
        if self._reference_target is None:
            return float(value)
        dh, du, ds = self._get_reference_offsets()
        T = self.temperature
        if name == "enthalpy":
            return float(value) + dh
        if name == "internal_energy":
            return float(value) + du
        if name == "entropy":
            return float(value) + ds
        if name == "gibbs_energy":
            return float(value) + dh - T * ds
        if name in {"free_energy", "helmholtz_energy"}:
            return float(value) + du - T * ds
        return float(value)

    # ---------------- Core ---------------- #
    @property
    def name(self) -> str:
        """Return the canonical ThermoProp display name for this ``Fluid`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return ", ".join(self._display_names)

    @property
    def composition(self) -> dict[str, float]:
        """Return the chainable fluid composition dictionary.

        The returned fractions use this object's active composition basis.
        For the default ``basis="mass"``, this is identical to
        ``mass_fractions``. For ``basis="mole"``, this is identical to
        ``mole_fractions``.
        """

        fractions = self._mole_fractions if self._basis == "mole" else self._mass_fractions
        return composition_dict(self._display_names, fractions)

    @property
    def fluid(self) -> dict[str, float]:
        """Return the fluid identifier or composition supplied to the wrapper for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.composition

    @property
    def basis(self) -> str:
        """Return the composition basis for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self._basis

    @property
    def composition_basis(self) -> str:
        """Return the composition basis alias for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self._basis

    @property
    def backend(self) -> str:
        """Return the backend used by this wrapper for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self._BACKEND_NAME

    @staticmethod
    def _validate_fractions(fractions, label: str, *, atol: float = 1e-6) -> np.ndarray:
        """Validate a mass- or mole-fraction vector and return it as an array."""
        return validate_fraction_vector(fractions, label, atol=atol)

    def _build_state(self):
        """Create and configure a CoolProp AbstractState."""
        state = CP.AbstractState("HEOS", self._fluid_string)
        if self._mixture:
            state.set_mass_fractions([float(x) for x in self._mass_fractions])
        return state

    def _sync_from_backend(self):
        """Synchronize cached pressure and enthalpy from the backend state."""
        self._P = float(self._backend.p())
        self._h = float(self._backend.hmass())
        self._pyfluid = self._backend

    def _set_state_from_named_pair(self, values: dict) -> None:
        self._last_state_values = dict(values)
        keys = frozenset(values.keys())
        raw_values = dict(values)

        if "enthalpy" in raw_values:
            raw_values["enthalpy"] = self._to_raw_basis("enthalpy", raw_values["enthalpy"])

        if "internal_energy" in raw_values:
            raw_values["internal_energy"] = self._to_raw_basis("internal_energy", raw_values["internal_energy"])

        if "entropy" in raw_values:
            raw_values["entropy"] = self._to_raw_basis("entropy", raw_values["entropy"])

        if keys not in Fluid._FLASH_PAIRS:
            raise ThermoPropFlashError(
                f"Unsupported flash pair: {sorted(keys)}. "
                f"Supported pairs are: {self.available_flash_pairs()}."
            )

        if keys == frozenset(("pressure", "enthalpy")) and self._mixture:
            pressure = float(raw_values["pressure"])
            enthalpy = float(raw_values["enthalpy"])

            T, Q = Fluid.get_temperature_and_quality(
                self._backend,
                pressure,
                enthalpy,
            )

            if 0.0 < Q < 1.0:
                self._update_state(CP.PQ_INPUTS, pressure, Q)
            else:
                self._update_state(CP.PT_INPUTS, pressure, T)

            self._sync_from_backend()
            return

        if keys == frozenset(("internal_energy", "entropy")):
            self._set_state_from_internal_energy_entropy(
                float(raw_values["internal_energy"]),
                float(raw_values["entropy"]),
            )
            return

        input_pair_name, order = Fluid._FLASH_PAIRS[keys]

        if not hasattr(CP, input_pair_name):
            raise ThermoPropFlashError(
                f"CoolProp does not expose {input_pair_name} in this installation."
            )

        input_pair = getattr(CP, input_pair_name)
        value1 = raw_values[order[0]]
        value2 = raw_values[order[1]]

        self._update_state(input_pair, value1, value2)
        self._sync_from_backend()



    def _set_state_from_internal_energy_entropy(self, internal_energy: float, entropy: float) -> None:
        pressure_guesses = []
        temperature_guesses = []

        if self._P is not None:
            pressure_guesses.append(self._P)
        pressure_guesses.extend([self._REFERENCE_PRESSURE, 1.0e5, 1.0e6, 1.0e4])

        try:
            temperature_guesses.append(self.temperature)
        except Exception:
            pass
        temperature_guesses.extend([self._REFERENCE_TEMPERATURE, 300.0, 500.0, 100.0])

        pressure_guesses = [p for i, p in enumerate(pressure_guesses) if p is not None and p > 0.0 and p not in pressure_guesses[:i]]
        temperature_guesses = [T for i, T in enumerate(temperature_guesses) if T is not None and T > 0.0 and T not in temperature_guesses[:i]]

        internal_energy_scale = max(abs(float(internal_energy)), 1.0e5)
        entropy_scale = max(abs(float(entropy)), 1.0e3)

        best_error = math.inf
        best_state = None
        last_error = None

        def residual(log_state):
            clipped_state = np.clip(log_state, -700.0, 700.0)
            pressure = float(np.exp(clipped_state[0]))
            temperature = float(np.exp(clipped_state[1]))

            try:
                self._update_state(CP.PT_INPUTS, pressure, temperature)
                u = float(self._backend.umass())
                s = float(self._backend.smass())
            except Exception:
                return np.array([1.0e6, 1.0e6])

            return np.array([
                (u - internal_energy) / internal_energy_scale,
                (s - entropy) / entropy_scale,
            ])

        for pressure_guess in pressure_guesses:
            for temperature_guess in temperature_guesses:
                try:
                    solution = root(
                        residual,
                        np.log([pressure_guess, temperature_guess]),
                        method="hybr",
                    )
                    pressure = float(np.exp(solution.x[0]))
                    temperature = float(np.exp(solution.x[1]))
                    self._update_state(CP.PT_INPUTS, pressure, temperature)
                    error = max(
                        abs(float(self._backend.umass()) - internal_energy) / internal_energy_scale,
                        abs(float(self._backend.smass()) - entropy) / entropy_scale,
                    )

                    if error < best_error:
                        best_error = error
                        best_state = (pressure, temperature)

                    if solution.success and error < 1.0e-7:
                        self._sync_from_backend()
                        return
                except Exception as exc:
                    last_error = exc
                    continue

        if best_state is not None and best_error < 1.0e-5:
            self._update_state(CP.PT_INPUTS, best_state[0], best_state[1])
            self._sync_from_backend()
            return

        raise ThermoPropFlashError(
            "Could not solve Fluid state from internal_energy and entropy."
        ) from last_error

    def set_pyfluid(self):
        """Execute the documented ``set_pyfluid`` operation for ``Fluid``.

        Arguments are validated and normalized using the same rules as the high-level
        wrappers.  Return values follow ThermoProp's SI-unit and composition
        conventions, and failures are reported through ThermoProp exception types with
        contextual messages rather than silent fallbacks.
        """
        if self._mixture:
            T, Q = Fluid.get_temperature_and_quality(self._backend, self._P, self._h)
            if 0.0 < Q < 1.0:
                self._update_state(CP.PQ_INPUTS, self._P, Q)
            else:
                self._update_state(CP.PT_INPUTS, self._P, T)
        else:
            self._update_state(CP.HmassP_INPUTS, self._h, self._P)

    # ---------------- Internal state helpers ---------------- #
    def _enthalpy_from_pressure_temperature(self, pressure: float, temperature: float) -> float:
        self._set_state_from_named_pair(
            {
                "pressure": pressure,
                "temperature": temperature,
            }
        )
        return float(self._backend.hmass())

    def _enthalpy_from_pressure_quality(self, pressure: float, quality: float) -> float:
        self._set_state_from_named_pair(
            {
                "pressure": pressure,
                "quality": quality,
            }
        )
        return float(self._backend.hmass())

    def _state_from_temperature_quality(self, temperature: float, quality: float) -> Tuple[float, float]:
        self._set_state_from_named_pair(
            {
                "temperature": temperature,
                "quality": quality,
            }
        )
        return float(self._backend.p()), float(self._backend.hmass())

    def _state_from_density_internal_energy(self, density: float, internal_energy: float) -> None:
        self._set_state_from_named_pair(
            {
                "density": density,
                "internal_energy": internal_energy,
            }
        )

    def _keyed_output(self, key, default=None):
        try:
            return float(self._backend.keyed_output(key))
        except Exception:
            return default

    def _trivial_output(self, key, default=None):
        try:
            return float(self._backend.trivial_keyed_output(key))
        except Exception:
            return default



    def _update_state(self, input_pair, value1: float, value2: float):
        """Update CoolProp state and keep the old internal alias current.

        Do not force a phase on failed mixture flashes. A previous implementation
        retried every failed mixture flash as gas, which could create metastable
        or invalid dense states and make derivative properties such as speed of
        sound return NaN.
        """
        self._backend.unspecify_phase()
        self._backend.update(input_pair, float(value1), float(value2))
        self._backend.unspecify_phase()
        self._pyfluid = self._backend
    # ---------------- Fractions ---------------- #
    @property
    def mole_fractions(self) -> dict:
        """Return the mole-fraction composition dictionary for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return {f: float(x) for f, x in zip(self._fluids, self._mole_fractions)}

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        """Set the mole-fraction composition dictionary for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mole fractions for a pure fluid")

        self._mole_fractions = self._validate_fractions(value, "Mole fractions")
        self._mass_fractions = Fluid.mole_to_mass(self._fluids, self._mole_fractions)

        self._backend = self._build_state()
        self._pyfluid = self._backend

        if self._last_state_values is not None:
            self._set_state_from_named_pair(self._last_state_values)

    @property
    def mass_fractions(self) -> dict:
        """Return the mass-fraction composition dictionary for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return {f: float(x) for f, x in zip(self._fluids, self._mass_fractions)}

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        """Set the mass-fraction composition dictionary for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mass fractions for a pure fluid")

        self._mass_fractions = self._validate_fractions(value, "Mass fractions")
        self._mole_fractions = Fluid.mass_to_mole(self._fluids, self._mass_fractions)

        self._backend = self._build_state()
        self._pyfluid = self._backend

        if self._last_state_values is not None:
            self._set_state_from_named_pair(self._last_state_values)

    # ---------------- State setters ---------------- #
    @property
    def pressure(self) -> float:
        """Return the thermodynamic pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._P

    @pressure.setter
    def pressure(self, value: float):
        """Set the thermodynamic pressure for this ``Fluid`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (Pa) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self.pressure_enthalpy = (float(value), self.enthalpy)

    @property
    def enthalpy(self) -> float:
        """Return the mass-specific enthalpy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._from_raw_basis("enthalpy", self._h)

    @enthalpy.setter
    def enthalpy(self, value: float):
        """Set the mass-specific enthalpy for this ``Fluid`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (J/kg) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self.pressure_enthalpy = (self.pressure, float(value))

    @property
    def pressure_enthalpy(self) -> Tuple[float, float]:
        """Return the pressure enthalpy for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.pressure, self.enthalpy

    @pressure_enthalpy.setter
    def pressure_enthalpy(self, values: Tuple[float, float]):
        """Set the pressure enthalpy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_enthalpy must be set with (pressure, enthalpy)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    @property
    def pressure_temperature(self) -> Tuple[float, float]:
        """Return the pressure temperature for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float, float]):
        """Set the pressure temperature for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "temperature": float(values[1]),
            }
        )

    @property
    def pressure_quality(self) -> Tuple[float, float]:
        """Return the pressure quality for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.pressure, self.quality

    @pressure_quality.setter
    def pressure_quality(self, values: Tuple[float, float]):
        """Set the pressure quality for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_quality must be set with (pressure, quality)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "quality": float(values[1]),
            }
        )

    @property
    def temperature_quality(self) -> Tuple[float, float]:
        """Return the temperature quality for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.temperature, self.quality

    @temperature_quality.setter
    def temperature_quality(self, values: Tuple[float, float]):
        """Set the temperature quality for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_quality must be set with (temperature, quality)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "quality": float(values[1]),
            }
        )

    @property
    def density_internal_energy(self) -> Tuple[float, float]:
        """Return the density internal energy for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.density, self.internal_energy

    @density_internal_energy.setter
    def density_internal_energy(self, values: Tuple[float, float]):
        """Set the density internal energy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_internal_energy must be set with (density, internal_energy)")

        self._set_state_from_named_pair(
            {
                "density": float(values[0]),
                "internal_energy": float(values[1]),
            }
        )

    @property
    def pressure_density(self) -> Tuple[float, float]:
        """Return the pressure density for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.pressure, self.density

    @pressure_density.setter
    def pressure_density(self, values: Tuple[float, float]):
        """Set the pressure density for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_density must be set with (pressure, density)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "density": float(values[1]),
            }
        )

    @property
    def pressure_internal_energy(self) -> Tuple[float, float]:
        """Return the pressure internal energy for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.pressure, self.internal_energy

    @pressure_internal_energy.setter
    def pressure_internal_energy(self, values: Tuple[float, float]):
        """Set the pressure internal energy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_internal_energy must be set with (pressure, internal_energy)")

        self._set_state_from_named_pair(
            {
                "pressure": float(values[0]),
                "internal_energy": float(values[1]),
            }
        )

    @property
    def temperature_density(self) -> Tuple[float, float]:
        """Return the temperature density for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.temperature, self.density

    @temperature_density.setter
    def temperature_density(self, values: Tuple[float, float]):
        """Set the temperature density for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_density must be set with (temperature, density)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "density": float(values[1]),
            }
        )

    @property
    def density_enthalpy(self) -> Tuple[float, float]:
        """Return the density enthalpy for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.density, self.enthalpy

    @density_enthalpy.setter
    def density_enthalpy(self, values: Tuple[float, float]):
        """Set the density enthalpy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("density_enthalpy must be set with (density, enthalpy)")

        self._set_state_from_named_pair(
            {
                "density": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    @property
    def temperature_enthalpy(self) -> Tuple[float, float]:
        """Return the temperature enthalpy for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self.temperature, self.enthalpy

    @temperature_enthalpy.setter
    def temperature_enthalpy(self, values: Tuple[float, float]):
        """Set the temperature enthalpy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (the corresponding getter units) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("temperature_enthalpy must be set with (temperature, enthalpy)")

        self._set_state_from_named_pair(
            {
                "temperature": float(values[0]),
                "enthalpy": float(values[1]),
            }
        )

    # ---------------- Thermo properties ---------------- #
    @property
    def species(self) -> List[str]:
        """Return the canonical species name or names for this ``Fluid`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return self._display_names

    @property
    def temperature(self) -> float:
        """Return the thermodynamic temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return float(self._backend.T())

    @temperature.setter
    def temperature(self, value: float):
        """Set the thermodynamic temperature for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (K) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if self._P is None:
            raise ValueError(
                "Cannot set temperature without pressure. "
                "Set pressure first."
            )

        self.pressure_temperature = (self.pressure, float(value))

    @property
    def phase(self) -> str:
        """Return the human-readable phase label for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        string.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return Fluid._PHASE_NAMES.get(int(self._backend.phase()), "Unknown")
        except Exception:
            return "Unknown"

    @property
    def compressibility(self) -> float:
        """Return the compressibility factor Z for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        dimensionless.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._keyed_output(CP.iZ)
            
    @property
    def thermal_expansion_coefficient(self) -> float:
        """Return the isobaric thermal expansion coefficient for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        1/K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._keyed_output(CP.iisobaric_expansion_coefficient)

    @property
    def isothermal_compressibility(self) -> float:
        """Return the isothermal compressibility for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        1/Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._keyed_output(CP.iisothermal_compressibility)
        
    @property
    def gas_constant(self) -> float:
        """Return the mass-specific gas constant for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return float(self._backend.gas_constant()) / self.molar_mass

    @property
    def universal_gas_constant(self) -> float:
        """Return the universal molar gas constant for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(mol*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return float(self._backend.gas_constant())

    @property
    def helmholtz_energy(self) -> float:
        """Return the mass-specific Helmholtz free energy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._from_raw_basis("helmholtz_energy", float(self._backend.helmholtzmass()))

    @property
    def gibbs_energy(self) -> float:
        """Return the mass-specific Gibbs free energy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._from_raw_basis("gibbs_energy", float(self._backend.gibbsmass()))

    @property
    def free_energy(self) -> float:
        """Return the mass-specific Helmholtz free energy alias for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.helmholtz_energy

    @property
    def fundamental_derivative_of_gas_dynamics(self) -> float:
        """Return the fundamental derivative of gas dynamics for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        try:
            return float(self._backend.fundamental_derivative_of_gas_dynamics())
        except Exception:
            return None
            
    @property
    def fugacity_coefficients(self) -> list[float]:
        """Return the fugacity coefficients for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        try:
            return [
                float(self._backend.fugacity_coefficient(i))
                for i in range(len(self._fluids))
            ]
        except Exception:
            return None

    @property
    def conductivity(self) -> float:
        """Return the thermal conductivity alias for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        W/(m*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.conductivity())
        except Exception:
            return None
            
    @property
    def thermal_conductivity(self) -> float:
        """Return the thermal conductivity for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        W/(m*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.conductivity

    @property
    def critical_pressure(self) -> float:
        """Return the critical pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.p_critical())
        except Exception:
            return self._trivial_output(CP.iP_critical)

    @property
    def critical_temperature(self) -> float:
        """Return the critical temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.T_critical())
        except Exception:
            return self._trivial_output(CP.iT_critical)

    @property
    def density(self) -> float:
        """Return the mass density for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        kg/m^3.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return float(self._backend.rhomass())

    @density.setter
    def density(self, value: float):
        """Set the mass density for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (kg/m^3) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        self.density_internal_energy = (float(value), self.internal_energy)

    @property
    def dynamic_viscosity(self) -> float:
        """Return the dynamic viscosity for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa*s.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.viscosity())
        except Exception:
            return None

    @property
    def entropy(self) -> float:
        """Return the mass-specific entropy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._from_raw_basis("entropy", float(self._backend.smass()))

    @entropy.setter
    def entropy(self, value: float):
        """Set the mass-specific entropy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (J/(kg*K)) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        self.pressure_entropy = (self.pressure, float(value))

    @property
    def freezing_temperature(self) -> float:
        """Return the freezing temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.minimum_temperature

    @property
    def internal_energy(self) -> float:
        """Return the mass-specific internal energy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._from_raw_basis("internal_energy", float(self._backend.umass()))

    @internal_energy.setter
    def internal_energy(self, value: float):
        """Set the mass-specific internal energy for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (J/kg) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        self.density_internal_energy = (self.density, float(value))

    @property
    def kinematic_viscosity(self) -> float:
        """Return the kinematic viscosity for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        m^2/s.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        mu = self.dynamic_viscosity
        rho = self.density
        if mu is None or rho is None or rho == 0:
            return None
        return mu / rho

    @property
    def maximum_pressure(self) -> float:
        """Return the maximum backend-supported pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.pmax())
        except Exception:
            return self._trivial_output(CP.iP_max)

    @property
    def maximum_temperature(self) -> float:
        """Return the maximum backend-supported temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.Tmax())
        except Exception:
            return self._trivial_output(CP.iT_max)

    @property
    def minimum_pressure(self) -> float:
        """Return the minimum backend-supported pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        # Prefer CoolProp's actual backend minimum-pressure metadata.  Triple
        # pressure is a phase landmark, not a general lower validity bound.
        try:
            pmin = getattr(self._backend, "pmin", None)
            if callable(pmin):
                return float(pmin())
        except Exception:
            pass

        index = getattr(CP, "iP_min", None)
        if index is not None:
            try:
                return self._trivial_output(index)
            except Exception:
                pass

        # CoolProp does not expose a universal finite lower pressure for every
        # backend/fluid combination.  Do not substitute triple pressure.
        return 0.0

    @property
    def minimum_temperature(self) -> float:
        """Return the minimum backend-supported temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.Tmin())
        except Exception:
            return self._trivial_output(CP.iT_min)

    @property
    def molar_mass(self) -> float:
        """Return the molar mass for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        kg/mol.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return float(self._backend.molar_mass())

    @property
    def prandtl(self) -> float:
        """Return the Prandtl number for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        dimensionless.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self._keyed_output(CP.iPrandtl)

    @property
    def speed_of_sound(self) -> float:
        """Return the speed of sound for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        m/s.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            value = float(self._backend.speed_sound())
        except Exception:
            return None

        if not np.isfinite(value) or value <= 0.0:
            return None

        return value

    @property
    def specific_heat_cp(self) -> float:
        """Return the constant-pressure specific heat for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.cpmass())
        except Exception:
            return None

    @property
    def specific_heat_cv(self) -> float:
        """Return the constant-volume specific heat for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.cvmass())
        except Exception:
            return None

    @property
    def specific_heat(self) -> float:
        """Return the default specific heat alias, usually constant-pressure specific heat for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        """Return the specific heat ratio cp/cv for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        dimensionless.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            cp = self.specific_heat_cp
            cv = self.specific_heat_cv

            if cp is None or cv is None or cv == 0.0:
                return None

            return cp / cv

        except Exception:
            return None
        

    @property
    def gamma(self) -> float:
        """Return the specific heat ratio alias for this ``Fluid`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are dimensionless.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self.specific_heat_ratio

    @property
    def specific_volume(self) -> float:
        """Return the specific volume for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        m^3/kg.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        rho = self.density
        if rho is None or rho == 0:
            return None
        return 1.0 / rho

    @property
    def surface_tension(self) -> float:
        """Return the surface tension for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        N/m.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.surface_tension())
        except Exception:
            return None

    @property
    def triple_pressure(self) -> float:
        """Return the triple-point pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.p_triple())
        except Exception:
            return self._trivial_output(CP.iP_triple)

    @property
    def triple_temperature(self) -> float:
        """Return the triple-point temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        try:
            return float(self._backend.Ttriple())
        except Exception:
            return self._trivial_output(CP.iT_triple)

    @property
    def is_mixture(self) -> bool:
        """Return the whether the object represents a mixture for this ``Fluid`` object.

        This property exposes normalized public metadata or solver bookkeeping without
        requiring direct access to private attributes.  Returned dictionaries and lists
        are suitable for reporting, validation, and example code.  When a backend lookup
        is required, ThermoProp applies the same alias and canonical-name rules used by
        the constructors.
        """
        return self._mixture

    @property
    def quality(self) -> float:
        """
        Vapor quality from 0 to 1.

        Quality is only thermodynamically defined in the two-phase region. For
        backward compatibility, single-phase gas-like states return 1.0 and
        liquid-like states return 0.0. Supercritical states return NaN.
        """

        ph = self.phase

        if ph == "TwoPhase":
            try:
                value = float(self._backend.Q())
                return value if np.isfinite(value) else float("nan")
            except Exception:
                return float("nan")

        if ph in ("Gas", "SupercriticalGas"):
            return 1.0

        if ph in ("Liquid", "SupercriticalLiquid"):
            return 0.0

        return float("nan")

    @quality.setter
    def quality(self, value: float):
        """Set the vapor quality when the backend defines a two-phase state for this ``Fluid`` instance.

        Assignments use ThermoProp's public SI-unit convention (dimensionless mass fraction) unless the
        corresponding getter is metadata.  For thermodynamic state setters, the wrapper
        immediately re-flashes or marks itself stale so every later property access uses
        the new state consistently.  Invalid values raise ThermoProp exceptions instead
        of being clipped silently.
        """
        if self._P is None:
            raise ValueError(
                "Cannot set quality without pressure. "
                "Set pressure first."
            )

        Q = float(value)

        if not (0.0 <= Q <= 1.0):
            raise ValueError("Quality must be between 0 and 1.")

        self.pressure_quality = (self.pressure, Q)


    @property
    def saturation_pressure(self) -> float:
        """Return the saturation pressure at the current temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        Tc = self.critical_temperature

        if Tc is not None and self.temperature > Tc:
            return None

        try:
            tmp = self._build_state()
            tmp.update(CP.QT_INPUTS, 1.0, self.temperature)
            return float(tmp.p())

        except Exception:
            return None

    @property
    def saturation_temperature(self) -> float:
        """Return the saturation temperature at the current pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        pc = self.critical_pressure
        if pc is not None and self.pressure > pc:
            return None
        try:
            tmp = self._build_state()
            tmp.update(CP.PQ_INPUTS, self._P, 1.0)
            return float(tmp.T())
        except Exception:
            return None
        

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        """
        Return a CoolProp first partial derivative.

        Format
        ------
            d(of)/d(with_respect_to)|constant

        Example
        -------
            fluid.partial_derivative("Hmass", "T", "P")
        """
        try:
            return float(
                CP.PropsSI(
                    f"d({of})/d({with_respect_to})|{constant}",
                    "P",
                    self.pressure,
                    "T",
                    self.temperature,
                    self._fluid_string,
                )
            )
        except Exception:
            return None


    @property
    def dhdT_const_p(self) -> float:
        """Return the partial derivative of enthalpy with respect to temperature at constant pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.partial_derivative("Hmass", "T", "P")


    @property
    def dhdp_const_T(self) -> float:
        """Return the partial derivative of enthalpy with respect to pressure at constant temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        J/(kg*Pa).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.partial_derivative("Hmass", "P", "T")


    @property
    def drhodT_const_p(self) -> float:
        """Return the partial derivative of density with respect to temperature at constant pressure for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        kg/(m^3*K).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.partial_derivative("Dmass", "T", "P")


    @property
    def drhodp_const_T(self) -> float:
        """Return the partial derivative of density with respect to pressure at constant temperature for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        kg/(m^3*Pa).  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.partial_derivative("Dmass", "P", "T")


    @property
    def dTdp_const_h(self) -> float:
        """Return the Joule-Thomson style dT/dp derivative at constant enthalpy for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K/Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.partial_derivative("T", "P", "Hmass")


    @property
    def joule_thomson_coefficient(self) -> float:
        """Return the Joule-Thomson coefficient for this ``Fluid`` state.

        The value is evaluated from the current state and active backend.  Units are
        K/Pa.  Unsupported species, phases, materials, or state points raise a
        ThermoProp exception with context about the backend and requested property.
        """
        return self.dTdp_const_h


    # ---------------- String output ---------------- #
    def _safe(self, value, fmt=".3e"):
        return format_optional(value, fmt)

    def __str__(self):
        rows = [
            ("Fluid(s)", ", ".join(self._display_names)),
            ("Mole fractions", rounded_dict(self.mole_fractions, 3)),
            ("Mass fractions", rounded_dict(self.mass_fractions, 3)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Saturation temperature [K]", self._safe(self.saturation_temperature, ".2f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("Speed of sound [m/s]", self._safe(self.speed_of_sound, ".6f"))
        ]
        return format_rows(rows)

    def __repr__(self) -> str:
        species_str = ", ".join(self._display_names)
        return (
            f"{self.__class__.__name__}(species=[{species_str}], "
            f"pressure={self._P:.3e} Pa, "
            f"enthalpy={self._h:.3e} J/kg, "
            f"temperature={self.temperature:.2f} K)"
        )

    # ---------------- Utilities ---------------- #
    @classmethod
    def _database_name(cls, user_name: str) -> str:
        """Return the canonical ThermoProp species name from SpeciesDatabase."""
        for method_name in ("_name", "name", "resolve"):
            method = getattr(SpeciesDatabase, method_name, None)
            if method is not None:
                return method(user_name)

        species = getattr(SpeciesDatabase, "species", None)
        if species is not None and str(user_name) in species():
            return str(user_name)

        raise ValueError(f"Unknown ThermoProp species name or alias: {user_name!r}")

    @classmethod
    def _database_coolprop_name(cls, user_name: str) -> str:
        """Return the CoolProp backend name from SpeciesDatabase."""
        for method_name in ("_coolprop_name", "coolprop_name"):
            method = getattr(SpeciesDatabase, method_name, None)
            if method is not None:
                return method(user_name)

        raise ValueError(f"{user_name!r} is not supported by Fluid/CoolProp.")

    @classmethod
    def _database_supported_fluid_species(cls) -> list[str]:
        """Return ThermoProp species names that support the Fluid wrapper."""
        supported_species = getattr(SpeciesDatabase, "supported_species", None)
        if supported_species is not None:
            return list(supported_species("Fluid"))

        supported_names = getattr(SpeciesDatabase, "supported_names", None)
        if supported_names is not None:
            return list(supported_names("Fluid"))

        names = getattr(SpeciesDatabase, "coolprop_supported_names", None)
        if names is not None:
            return list(names)

        raise AttributeError(
            "SpeciesDatabase must expose supported_species('Fluid') or an "
            "equivalent internal Fluid-support list."
        )

    @classmethod
    def _database_supported_coolprop_names(cls) -> list[str]:
        """Return CoolProp backend names supported by the Fluid wrapper."""
        backend_names: set[str] = set()

        for name in cls._database_supported_fluid_species():
            try:
                backend_names.add(cls._database_coolprop_name(name))
            except Exception:
                continue

        return sorted(backend_names)

    @classmethod
    def _normalize_name(cls, user_name: str) -> Tuple[str, str]:
        """Return (CoolProp backend name, ThermoProp display name)."""
        backend = cls._database_coolprop_name(user_name)
        display = cls._database_name(user_name)
        return backend, display


    @staticmethod
    @lru_cache(maxsize=None)
    def _molar_mass_of(fluid: str) -> float:
        """Return pure-fluid molar mass in kg/mol."""
        return float(CP.PropsSI("M", fluid))

    @staticmethod
    def mole_to_mass(fluids: List[str], mole_fractions: List[float]):
        """Convert composition fractions between mole and mass bases.

        Input fractions must be finite and nonnegative.  ThermoProp uses the supplied
        species order, evaluates molecular weights from the active database, normalizes
        the converted fractions to sum to one, and returns the converted list in the same
        order as the input names.
        """
        mole_fractions = Fluid._validate_fractions(mole_fractions, "Mole fractions")
        molar_masses = np.array([Fluid._molar_mass_of(f) for f in fluids])
        m_bar = np.dot(mole_fractions, molar_masses)
        return mole_fractions * molar_masses / m_bar

    @staticmethod
    def mass_to_mole(fluids: List[str], mass_fractions: List[float]):
        """Convert composition fractions between mole and mass bases.

        Input fractions must be finite and nonnegative.  ThermoProp uses the supplied
        species order, evaluates molecular weights from the active database, normalizes
        the converted fractions to sum to one, and returns the converted list in the same
        order as the input names.
        """
        mass_fractions = Fluid._validate_fractions(mass_fractions, "Mass fractions")
        molar_masses = np.array([Fluid._molar_mass_of(f) for f in fluids])
        inv = mass_fractions / molar_masses
        return inv / inv.sum()

    @staticmethod
    def get_temperature_and_quality(fluid, pressure: float, target_enthalpy: float) -> Tuple[float, float]:
        """
        Given a CoolProp AbstractState, pressure, and enthalpy, return (T, Q).

        For mixtures this avoids relying on direct P-H flashes through the dome.
        It mirrors the old pyfluids workaround: compare target enthalpy to the
        saturated liquid/vapor enthalpies, then solve T at fixed P outside dome.
        """
        try:
            fluid.update(CP.PQ_INPUTS, pressure, 0.0)
            h_liquid = float(fluid.hmass())
            T_sat = float(fluid.T())

            fluid.update(CP.PQ_INPUTS, pressure, 1.0)
            h_vapor = float(fluid.hmass())
        except Exception:
            h_liquid = None
            h_vapor = None
            T_sat = None

        h = float(target_enthalpy)

        if h_liquid is not None and h_vapor is not None and h_liquid <= h <= h_vapor:
            denom = h_vapor - h_liquid
            Q = 0.0 if abs(denom) < 1e-15 else (h - h_liquid) / denom
            return T_sat, float(Q)

        def residual(T):
            try:
                fluid.update(CP.PT_INPUTS, pressure, T)
                return float(fluid.hmass()) - h
            except Exception:
                return np.nan

        try:
            Tmin = float(fluid.Tmin())
        except Exception:
            Tmin = 1.0
        try:
            Tmax = float(fluid.Tmax())
        except Exception:
            Tmax = 5000.0

        Ts = np.linspace(Tmin * 1.000001, Tmax * 0.999999, 300)
        vals = []
        for T in Ts:
            r = residual(T)
            vals.append(r if np.isfinite(r) else np.nan)

        bracket = None
        for T1, T2, r1, r2 in zip(Ts[:-1], Ts[1:], vals[:-1], vals[1:]):
            if not (np.isfinite(r1) and np.isfinite(r2)):
                continue
            if r1 == 0:
                bracket = (T1, T1)
                break
            if r1 * r2 <= 0:
                bracket = (T1, T2)
                break

        if bracket is None:
            raise ValueError(
                f"Could not find a valid temperature bracket for pressure={pressure:.6g} Pa, "
                f"enthalpy={h:.6g} J/kg over T=[{Tmin:.6g}, {Tmax:.6g}] K."
            )

        if bracket[0] == bracket[1]:
            T = bracket[0]
        else:
            sol = root_scalar(residual, method="brentq", bracket=bracket)
            T = float(sol.root)

        if h_liquid is not None and h < h_liquid:
            Q = 0.0
        elif h_vapor is not None and h > h_vapor:
            Q = 1.0
        else:
            Q = float("nan")

        return T, Q

    @staticmethod
    def get_available_fluids() -> List[str]:
        """Return the fluids supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
        return Fluid._database_supported_coolprop_names()

    @staticmethod
    def show_available_fluids() -> List[str]:
        """Print and return the available available fluids.

        The printed table is intended for interactive discovery.  The return value
        contains the same information in normal Python data structures so scripts,
        examples, tests, and documentation generators can reuse it without parsing
        stdout.
        """
        fluids = Fluid.get_available_fluids()

        for fluid in fluids:
            print(fluid)

        return fluids

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        """Return the flash pairs supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
        return sorted(
            "-".join(sorted(pair))
            for pair in cls._FLASH_PAIRS
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        """Return the flash pairs supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
        return cls.available_flash_pairs()

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        """Return the flash inputs supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
        return cls.available_flash_pairs()

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        """Return the flash inputs supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
        return cls.available_flash_inputs()
