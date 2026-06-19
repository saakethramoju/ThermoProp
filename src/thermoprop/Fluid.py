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
        """
        Initialize a Fluid state.
        """
        self._reference_target = self._normalize_reference_target(set_reference)
        self._reference_offsets: tuple[float, float, float] | None = None
        self._basis = basis

        valid_fluids = Fluid.get_available_fluids()

        self._fluids: List[str] = []
        self._display_names: List[str] = []

        if isinstance(fluid, str):
            backend, display = Fluid._normalize_name(fluid)
            if backend not in valid_fluids:
                raise ValueError(
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
                    raise ValueError(
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
                        raise ValueError(
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
                    raise ValueError("basis must be 'mole' or 'mass'")

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
            raise LookupError(
                "Please provide exactly two thermodynamic properties. "
                "Supported names are pressure, temperature, enthalpy, "
                "quality, density, and internal_energy."
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
            new_fluid = self._composition_argument() if not is_provided(fluid) else fluid
            new_basis = self._basis if not is_provided(basis) else basis
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
        return self._reference_target or self.__class__.__name__

    @property
    def set_reference(self) -> str:
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
        """Stable state fingerprint for FullFlow ``Lookup`` caching."""

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
        """Alias for ``composition`` used for FullFlow lookup chaining."""
        return self.composition

    @property
    def basis(self) -> str:
        """Composition basis used by ``composition``."""
        return self._basis

    @property
    def composition_basis(self) -> str:
        """Readable alias for ``basis``."""
        return self._basis

    @property
    def backend(self) -> str:
        """Name of the thermodynamic property backend."""
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
            raise ValueError(
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
            raise ValueError(
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

        raise ValueError(
            "Could not solve Fluid state from internal_energy and entropy."
        ) from last_error

    def set_pyfluid(self):
        """Rebuild backend CoolProp state using current pressure and enthalpy."""
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
        """Return mole fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mole_fractions)}

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        """Update mole fractions. Fractions must sum to 1."""
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
        """Return mass fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mass_fractions)}

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        """Update mass fractions. Fractions must sum to 1."""
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
        """Absolute pressure in Pa."""
        return self._P

    @pressure.setter
    def pressure(self, value: float):
        self.pressure_enthalpy = (float(value), self.enthalpy)

    @property
    def enthalpy(self) -> float:
        """Mass-specific enthalpy in J/kg."""
        return self._from_raw_basis("enthalpy", self._h)

    @enthalpy.setter
    def enthalpy(self, value: float):
        self.pressure_enthalpy = (self.pressure, float(value))

    @property
    def pressure_enthalpy(self) -> Tuple[float, float]:
        """Return (pressure [Pa], enthalpy [J/kg])."""
        return self.pressure, self.enthalpy

    @pressure_enthalpy.setter
    def pressure_enthalpy(self, values: Tuple[float, float]):
        """Update state from pressure and enthalpy."""
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
        """Return (pressure [Pa], temperature [K])."""
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: Tuple[float, float]):
        """Update state from pressure and temperature."""
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
        """Return (pressure [Pa], quality [-])."""
        return self.pressure, self.quality

    @pressure_quality.setter
    def pressure_quality(self, values: Tuple[float, float]):
        """Update state from pressure and quality."""
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
        """Return (temperature [K], quality [-])."""
        return self.temperature, self.quality

    @temperature_quality.setter
    def temperature_quality(self, values: Tuple[float, float]):
        """Update state from temperature and quality."""
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
        """Return (density [kg/m^3], internal_energy [J/kg])."""
        return self.density, self.internal_energy

    @density_internal_energy.setter
    def density_internal_energy(self, values: Tuple[float, float]):
        """Update state from density and internal energy."""
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
        """Return (pressure [Pa], density [kg/m^3])."""
        return self.pressure, self.density

    @pressure_density.setter
    def pressure_density(self, values: Tuple[float, float]):
        """Update state from pressure and density."""
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
        """Return (pressure [Pa], internal_energy [J/kg])."""
        return self.pressure, self.internal_energy

    @pressure_internal_energy.setter
    def pressure_internal_energy(self, values: Tuple[float, float]):
        """Update state from pressure and internal energy."""
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
        """Return (temperature [K], density [kg/m^3])."""
        return self.temperature, self.density

    @temperature_density.setter
    def temperature_density(self, values: Tuple[float, float]):
        """Update state from temperature and density."""
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
        """Return (density [kg/m^3], enthalpy [J/kg])."""
        return self.density, self.enthalpy

    @density_enthalpy.setter
    def density_enthalpy(self, values: Tuple[float, float]):
        """Update state from density and enthalpy."""
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
        """Return (temperature [K], enthalpy [J/kg])."""
        return self.temperature, self.enthalpy

    @temperature_enthalpy.setter
    def temperature_enthalpy(self, values: Tuple[float, float]):
        """Update state from temperature and enthalpy."""
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
        return self._display_names

    @property
    def temperature(self) -> float:
        """Absolute temperature in K."""
        return float(self._backend.T())

    @temperature.setter
    def temperature(self, value: float):
        """
        Update temperature while holding pressure constant.

        Requires pressure to already be defined.
        """
        if self._P is None:
            raise ValueError(
                "Cannot set temperature without pressure. "
                "Set pressure first."
            )

        self.pressure_temperature = (self.pressure, float(value))

    @property
    def phase(self) -> str:
        """Thermodynamic phase name reported by CoolProp."""
        try:
            return Fluid._PHASE_NAMES.get(int(self._backend.phase()), "Unknown")
        except Exception:
            return "Unknown"

    @property
    def compressibility(self) -> float:
        """Compressibility factor Z."""
        return self._keyed_output(CP.iZ)
            
    @property
    def thermal_expansion_coefficient(self) -> float:
        """Volumetric thermal expansion coefficient beta [1/K]."""
        return self._keyed_output(CP.iisobaric_expansion_coefficient)

    @property
    def isothermal_compressibility(self) -> float:
        """Isothermal compressibility [1/Pa]."""
        return self._keyed_output(CP.iisothermal_compressibility)
        
    @property
    def gas_constant(self) -> float:
        """Mass-specific gas constant [J/kg-K]."""
        return float(self._backend.gas_constant()) / self.molar_mass

    @property
    def universal_gas_constant(self) -> float:
        """Universal gas constant [J/mol-K]."""
        return float(self._backend.gas_constant())

    @property
    def helmholtz_energy(self) -> float:
        """Mass-specific Helmholtz free energy [J/kg]."""
        return self._from_raw_basis("helmholtz_energy", float(self._backend.helmholtzmass()))

    @property
    def gibbs_energy(self) -> float:
        """Mass-specific Gibbs free energy [J/kg]."""
        return self._from_raw_basis("gibbs_energy", float(self._backend.gibbsmass()))

    @property
    def free_energy(self) -> float:
        """Mass-specific Helmholtz free energy [J/kg]."""
        return self.helmholtz_energy

    @property
    def fundamental_derivative_of_gas_dynamics(self) -> float:
        """Fundamental derivative of gas dynamics [-]."""
        try:
            return float(self._backend.fundamental_derivative_of_gas_dynamics())
        except Exception:
            return None
            
    @property
    def fugacity_coefficients(self) -> list[float]:
        """Mixture fugacity coefficients [-]."""
        try:
            return [
                float(self._backend.fugacity_coefficient(i))
                for i in range(len(self._fluids))
            ]
        except Exception:
            return None

    @property
    def conductivity(self) -> float:
        """Thermal conductivity in W/m-K."""
        try:
            return float(self._backend.conductivity())
        except Exception:
            return None
            
    @property
    def thermal_conductivity(self) -> float:
        """Backward-compatible alias for conductivity."""
        return self.conductivity

    @property
    def critical_pressure(self) -> float:
        """Critical pressure in Pa."""
        try:
            return float(self._backend.p_critical())
        except Exception:
            return self._trivial_output(CP.iP_critical)

    @property
    def critical_temperature(self) -> float:
        """Critical temperature in K."""
        try:
            return float(self._backend.T_critical())
        except Exception:
            return self._trivial_output(CP.iT_critical)

    @property
    def density(self) -> float:
        """Mass density in kg/m^3."""
        return float(self._backend.rhomass())

    @density.setter
    def density(self, value: float):
        """
        Update density while holding internal energy constant.
        """
        self.density_internal_energy = (float(value), self.internal_energy)

    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity in Pa-s."""
        try:
            return float(self._backend.viscosity())
        except Exception:
            return None

    @property
    def entropy(self) -> float:
        """Mass-specific entropy in J/kg-K."""
        return self._from_raw_basis("entropy", float(self._backend.smass()))

    @entropy.setter
    def entropy(self, value: float):
        """Update entropy while holding pressure constant."""
        self.pressure_entropy = (self.pressure, float(value))

    @property
    def freezing_temperature(self) -> float:
        """Freezing/melting temperature in K when available; otherwise Tmin."""
        return self.minimum_temperature

    @property
    def internal_energy(self) -> float:
        """Mass-specific internal energy in J/kg."""
        return self._from_raw_basis("internal_energy", float(self._backend.umass()))

    @internal_energy.setter
    def internal_energy(self, value: float):
        """
        Update internal energy while holding density constant.
        """
        self.density_internal_energy = (self.density, float(value))

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity in m^2/s."""
        mu = self.dynamic_viscosity
        rho = self.density
        if mu is None or rho is None or rho == 0:
            return None
        return mu / rho

    @property
    def maximum_pressure(self) -> float:
        """Maximum valid pressure in Pa."""
        try:
            return float(self._backend.pmax())
        except Exception:
            return self._trivial_output(CP.iP_max)

    @property
    def maximum_temperature(self) -> float:
        """Maximum valid temperature in K."""
        try:
            return float(self._backend.Tmax())
        except Exception:
            return self._trivial_output(CP.iT_max)

    @property
    def minimum_pressure(self) -> float:
        """Minimum valid pressure in Pa."""
        try:
            return float(self._backend.p_triple())
        except Exception:
            return self._trivial_output(CP.iP_triple)

    @property
    def minimum_temperature(self) -> float:
        """Minimum valid temperature in K."""
        try:
            return float(self._backend.Tmin())
        except Exception:
            return self._trivial_output(CP.iT_min)

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        return float(self._backend.molar_mass())

    @property
    def prandtl(self) -> float:
        """Prandtl number."""
        return self._keyed_output(CP.iPrandtl)

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound in m/s, or None if CoolProp cannot evaluate it."""
        try:
            value = float(self._backend.speed_sound())
        except Exception:
            return None

        if not np.isfinite(value) or value <= 0.0:
            return None

        return value

    @property
    def specific_heat_cp(self) -> float:
        """Cp in J/kg-K."""
        try:
            return float(self._backend.cpmass())
        except Exception:
            return None

    @property
    def specific_heat_cv(self) -> float:
        """Cv in J/kg-K."""
        try:
            return float(self._backend.cvmass())
        except Exception:
            return None

    @property
    def specific_heat(self) -> float:
        """Backward-compatible alias for Cp."""
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        """
        Specific heat ratio gamma = Cp/Cv.
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
        return self.specific_heat_ratio

    @property
    def specific_volume(self) -> float:
        """Specific volume in m^3/kg."""
        rho = self.density
        if rho is None or rho == 0:
            return None
        return 1.0 / rho

    @property
    def surface_tension(self) -> float:
        """Surface tension in N/m when available."""
        try:
            return float(self._backend.surface_tension())
        except Exception:
            return None

    @property
    def triple_pressure(self) -> float:
        """Triple point pressure in Pa."""
        try:
            return float(self._backend.p_triple())
        except Exception:
            return self._trivial_output(CP.iP_triple)

    @property
    def triple_temperature(self) -> float:
        """Triple point temperature in K."""
        try:
            return float(self._backend.Ttriple())
        except Exception:
            return self._trivial_output(CP.iT_triple)

    @property
    def is_mixture(self) -> bool:
        """Return True if this fluid is a mixture, False if pure."""
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
        """
        Update vapor quality while holding pressure constant.

        Requires pressure to already be defined.
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
        """
        Saturation pressure in Pa for current temperature,
        only if temperature <= critical temperature.
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
        """Saturation temperature in K for current pressure, only if pressure <= Pc."""
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
        """(∂h/∂T)_p [J/kg-K]. Same as Cp."""
        return self.partial_derivative("Hmass", "T", "P")


    @property
    def dhdp_const_T(self) -> float:
        """(∂h/∂p)_T [J/kg-Pa]."""
        return self.partial_derivative("Hmass", "P", "T")


    @property
    def drhodT_const_p(self) -> float:
        """(∂rho/∂T)_p [kg/m³-K]."""
        return self.partial_derivative("Dmass", "T", "P")


    @property
    def drhodp_const_T(self) -> float:
        """(∂rho/∂p)_T [kg/m³-Pa]."""
        return self.partial_derivative("Dmass", "P", "T")


    @property
    def dTdp_const_h(self) -> float:
        """(∂T/∂p)_h [K/Pa]. Joule-Thomson coefficient."""
        return self.partial_derivative("T", "P", "Hmass")


    @property
    def joule_thomson_coefficient(self) -> float:
        """Joule-Thomson coefficient [K/Pa], computed as (∂T/∂p)_h."""
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
        """Convert mole fractions to mass fractions."""
        mole_fractions = Fluid._validate_fractions(mole_fractions, "Mole fractions")
        molar_masses = np.array([Fluid._molar_mass_of(f) for f in fluids])
        m_bar = np.dot(mole_fractions, molar_masses)
        return mole_fractions * molar_masses / m_bar

    @staticmethod
    def mass_to_mole(fluids: List[str], mass_fractions: List[float]):
        """Convert mass fractions to mole fractions."""
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
        """Return available CoolProp backend fluid names."""
        return Fluid._database_supported_coolprop_names()

    @staticmethod
    def show_available_fluids() -> List[str]:
        """Print and return available CoolProp fluid names."""
        fluids = Fluid.get_available_fluids()

        for fluid in fluids:
            print(fluid)

        return fluids

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        """Return supported two-property CoolProp flash combinations."""
        return sorted(
            "-".join(sorted(pair))
            for pair in cls._FLASH_PAIRS
        )

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        """Return supported two-property CoolProp flash combinations."""
        return cls.available_flash_pairs()

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        """Return supported CoolProp state input combinations."""
        return cls.available_flash_pairs()

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        """Return supported CoolProp state input combinations."""
        return cls.available_flash_inputs()
