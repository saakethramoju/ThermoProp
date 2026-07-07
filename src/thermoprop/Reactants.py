from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .CEADatabase import CEA
from .Propellant import Propellant
from .CombustionGas import CombustionGas
from ._state_api import UNSET, is_provided, provided_items
from .Exceptions import PropertyUnavailableError, ThermoPropConfigurationError


ThermoReactant = Propellant | CombustionGas
ReactantEntry = ThermoReactant | tuple[ThermoReactant, float]
ReactantGroup = ReactantEntry | Iterable[ReactantEntry] | None


@dataclass(frozen=True)
class Reactant:
    """Represent the public ThermoProp ``Reactant`` API object.

    This class or dataclass is intentionally importable and documented for users who
    need to build property models, inspect solver results, or interact with packaged
    databases directly.  Values follow ThermoProp's SI-unit convention unless a
    specific field or method documents that it is metadata.  Instances should be
    created through the public constructor or returned by a public ThermoProp method
    rather than assembled from private implementation details.
    """
    propellant: ThermoReactant
    mass: float
    role: str
    species_name: str | None = None

    @property
    def name(self) -> str:
        """Return the canonical ThermoProp display name for this ``Reactant`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        if self.species_name is not None:
            return self.species_name
        return self.propellant.name

    @property
    def cea_name(self) -> str:
        """Return the public ``cea_name`` value for this ``Reactant`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        if self.species_name is not None:
            return self.species_name
        return self.propellant.cea_name

    @property
    def molar_mass(self) -> float:
        """Return the molar mass for this ``Reactant`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are kg/mol.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        if self.species_name is not None:
            mw = CEA.molar_mass(self.species_name)
        else:
            mw = self.propellant.cea_formula_molar_mass

        if mw is None or mw <= 0.0:
            raise ValueError(
                f"{self.name!r} does not have a valid CEA formula molar mass."
            )

        return float(mw)

    @property
    def moles(self) -> float:
        """Return the public ``moles`` value for this ``Reactant`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.mass / self.molar_mass

    @property
    def kmoles(self) -> float:
        """Return the public ``kmoles`` value for this ``Reactant`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.moles / 1000.0

    @property
    def temperature(self) -> float | None:
        """Return the thermodynamic temperature for this ``Reactant`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are K.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self.propellant.temperature

    @property
    def pressure(self) -> float | None:
        """Return the thermodynamic pressure for this ``Reactant`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are Pa.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self.propellant.pressure

    @property
    def enthalpy(self) -> float | None:
        """Return the mass-specific enthalpy for this ``Reactant`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are J/kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        if self.species_name is None:
            return self.propellant.enthalpy

        gas = CombustionGas(
            self.species_name,
            pressure=self.propellant.pressure,
            temperature=self.propellant.temperature,
            set_reference=self.propellant.set_reference,
        )
        return gas.enthalpy

    @property
    def internal_energy(self) -> float | None:
        """Return the mass-specific internal energy for this ``Reactant`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are J/kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        if self.species_name is None:
            return self.propellant.internal_energy

        gas = CombustionGas(
            self.species_name,
            pressure=self.propellant.pressure,
            temperature=self.propellant.temperature,
            set_reference=self.propellant.set_reference,
        )
        return gas.internal_energy

    @property
    def elemental_composition(self) -> dict[str, float] | None:
        """Return the elemental composition dictionary for this ``Reactant`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        if self.species_name is not None:
            return CEA.elemental_composition(self.species_name)
        return self.propellant.elemental_composition


class Reactants:
    """
    CEA-style reactant mixture definition for equilibrium calculations.

    Reactants groups fuel, oxidizer, inert, and igniter streams and computes
    the elemental inventory and reactant enthalpy needed by Equilibrium.

    Fuels and oxidizers preserve the original mass basis:

        fuel mass = 1 kg
        oxidizer mass = O/F kg

    Optional inert and igniter streams are specified as fractions of total
    propellant mass:

        propellant mass = fuel mass + oxidizer mass
        inert mass = inert_fraction * propellant mass
        igniter mass = igniter_fraction * propellant mass

    Inputs may be Propellant objects, CombustionGas objects, or weighted
    collections of either. Raw strings are intentionally not accepted because
    temperature, pressure, composition, and reference basis should be explicit.

    Examples
    --------

    Standard fuel/oxidizer:

        Reactants(
            fuels=Propellant("RP-1", temperature=298.15),
            oxidizers=Propellant("LOX", temperature=90.17),
            mixture_ratio=2.5,
        )

    Weighted fuel blend:

        Reactants(
            fuels=[
                (Propellant("RP-1", temperature=298.15), 0.8),
                (Propellant("Ethanol", temperature=298.15), 0.2),
            ],
            oxidizers=Propellant("LOX", temperature=90.17),
            mixture_ratio=2.5,
        )

    Inert gas stream:

        Reactants(
            fuels=rp1,
            oxidizers=lox,
            mixture_ratio=2.0,
            inerts=CombustionGas({"N2": 0.8, "Ar": 0.2}, basis="mass", pressure=P, temperature=300),
            inert_fraction=0.10,
        )

    Inert and igniter split streams:

        Reactants(
            fuels=rp1,
            oxidizers=lox,
            mixture_ratio=2.0,
            inerts=[
                (CombustionGas("N2", pressure=P, temperature=300), 0.8),
                (CombustionGas("Ar", pressure=P, temperature=300), 0.2),
            ],
            inert_fraction=0.10,
            igniters=CombustionGas({"H2": 0.7, "O2": 0.3}, basis="mass", pressure=P, temperature=300),
            igniter_fraction=0.02,
        )

    Notes
    -----
    CombustionGas mixtures are expanded internally into their CEA species using
    mass fractions. The mixture temperature, pressure, and reference basis are
    preserved for enthalpy and internal-energy evaluation.
    """

    def __init__(
        self,
        fuels: ReactantGroup,
        oxidizers: ReactantGroup,
        mixture_ratio: float,
        inerts: ReactantGroup = None,
        inert_fraction: float = 0.0,
        igniters: ReactantGroup = None,
        igniter_fraction: float = 0.0,
    ):
        """Initialize a CEA-style reactant mixture for equilibrium calculations.

        ``fuels`` and ``oxidizers`` contain ``Propellant`` objects or weighted ``(Propellant, weight)`` tuples.  ``mixture_ratio`` is oxidizer-to-fuel mass ratio.  Optional ``inerts`` and ``igniters`` can be added by mass fraction relative to the normalized reactant basis.  ThermoProp converts each propellant into mass, mole, elemental-composition, enthalpy, internal-energy, and entropy contributions.

        The normalized basis is one kilogram of fuel plus ``mixture_ratio`` kilograms of oxidizer before optional inert and igniter additions.  The resulting object is deliberately separate from ``Equilibrium`` so model code can update reactant states and mixture ratio before choosing an equilibrium mode.
        """
        self._fuel_inputs = None
        self._oxidizer_inputs = None
        self._inert_inputs = None
        self._igniter_inputs = None

        self._mixture_ratio = None
        self._inert_fraction = None
        self._igniter_fraction = None

        self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")
        self._oxidizer_inputs = self._parse_group_inputs(oxidizers, role="oxidizer")
        self._inert_inputs = self._parse_group_inputs(inerts, role="inert", allow_empty=True)
        self._igniter_inputs = self._parse_group_inputs(igniters, role="igniter", allow_empty=True)

        self._inert_fraction = self._validate_fraction(inert_fraction, "inert_fraction")
        self._igniter_fraction = self._validate_fraction(igniter_fraction, "igniter_fraction")
        self.mixture_ratio = mixture_ratio


    def update(
        self,
        *,
        fuels=UNSET,
        oxidizers=UNSET,
        mixture_ratio=UNSET,
        inerts=UNSET,
        inert_fraction=UNSET,
        igniters=UNSET,
        igniter_fraction=UNSET,
        fuel_weights=UNSET,
        oxidizer_weights=UNSET,
        inert_weights=UNSET,
        igniter_weights=UNSET,
        fuel_mass_fractions=UNSET,
        oxidizer_mass_fractions=UNSET,
        inert_mass_fractions=UNSET,
        igniter_mass_fractions=UNSET,
    ):
        """Update reactant groups, weights, and mixture ratio in one rebuild.

        ``Reactants`` stores precomputed entry masses, moles, and enthalpies.
        Calling ``update`` with no arguments is therefore useful after one of the
        contained ``Propellant`` or ``CombustionGas`` objects has changed state:
        the same reactant objects are reused and the aggregate feed is rebuilt.
        """

        if is_provided(fuels):
            self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")

        if is_provided(oxidizers):
            self._oxidizer_inputs = self._parse_group_inputs(
                oxidizers,
                role="oxidizer",
            )

        if is_provided(inerts):
            self._inert_inputs = self._parse_group_inputs(
                inerts,
                role="inert",
                allow_empty=True,
            )

        if is_provided(igniters):
            self._igniter_inputs = self._parse_group_inputs(
                igniters,
                role="igniter",
                allow_empty=True,
            )

        if is_provided(mixture_ratio):
            value = float(mixture_ratio)
            if value < 0.0:
                raise ValueError("mixture_ratio must be nonnegative.")
            self._mixture_ratio = value

        if is_provided(inert_fraction):
            self._inert_fraction = self._validate_fraction(
                inert_fraction,
                "inert_fraction",
            )

        if is_provided(igniter_fraction):
            self._igniter_fraction = self._validate_fraction(
                igniter_fraction,
                "igniter_fraction",
            )

        if is_provided(fuel_weights):
            if len(fuel_weights) != len(self._fuel_inputs):
                raise ValueError(
                    f"Expected {len(self._fuel_inputs)} fuel weights, "
                    f"got {len(fuel_weights)}."
                )
            self._fuel_inputs = [
                (reactant, float(weight))
                for (reactant, _), weight in zip(self._fuel_inputs, fuel_weights)
            ]

        if is_provided(oxidizer_weights):
            if len(oxidizer_weights) != len(self._oxidizer_inputs):
                raise ValueError(
                    f"Expected {len(self._oxidizer_inputs)} oxidizer weights, "
                    f"got {len(oxidizer_weights)}."
                )
            self._oxidizer_inputs = [
                (reactant, float(weight))
                for (reactant, _), weight in zip(self._oxidizer_inputs, oxidizer_weights)
            ]

        if is_provided(inert_weights):
            if len(inert_weights) != len(self._inert_inputs):
                raise ValueError(
                    f"Expected {len(self._inert_inputs)} inert weights, "
                    f"got {len(inert_weights)}."
                )
            self._inert_inputs = [
                (reactant, float(weight))
                for (reactant, _), weight in zip(self._inert_inputs, inert_weights)
            ]

        if is_provided(igniter_weights):
            if len(igniter_weights) != len(self._igniter_inputs):
                raise ValueError(
                    f"Expected {len(self._igniter_inputs)} igniter weights, "
                    f"got {len(igniter_weights)}."
                )
            self._igniter_inputs = [
                (reactant, float(weight))
                for (reactant, _), weight in zip(self._igniter_inputs, igniter_weights)
            ]

        if is_provided(fuel_mass_fractions):
            self._fuel_inputs = self._updated_group_weights(
                self._fuel_inputs,
                fuel_mass_fractions,
                role="fuel",
            )

        if is_provided(oxidizer_mass_fractions):
            self._oxidizer_inputs = self._updated_group_weights(
                self._oxidizer_inputs,
                oxidizer_mass_fractions,
                role="oxidizer",
            )

        if is_provided(inert_mass_fractions):
            self._inert_inputs = self._updated_group_weights(
                self._inert_inputs,
                inert_mass_fractions,
                role="inert",
            )

        if is_provided(igniter_mass_fractions):
            self._igniter_inputs = self._updated_group_weights(
                self._igniter_inputs,
                igniter_mass_fractions,
                role="igniter",
            )

        self._rebuild_entries()
        return self

    @staticmethod
    def _validate_fraction(value: float, name: str) -> float:
        value = float(value)

        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative.")

        return value

    @staticmethod
    def _is_reactant_object(value) -> bool:
        return isinstance(value, (Propellant, CombustionGas))

    @staticmethod
    def _group_items(reactants: ReactantGroup, role: str) -> list[ReactantEntry]:
        if reactants is None:
            return []

        if isinstance(reactants, str):
            raise TypeError(
                "Reactants does not accept raw string reactants. "
                "Pass Propellant or CombustionGas objects so state is explicit."
            )

        if Reactants._is_reactant_object(reactants):
            return [reactants]

        if isinstance(reactants, tuple):
            if len(reactants) == 2 and Reactants._is_reactant_object(reactants[0]):
                return [reactants]

        if not isinstance(reactants, IterableABC):
            raise TypeError(
                f"{role} must be a Propellant, CombustionGas, weighted "
                f"({role}, weight) tuple, or iterable of those. "
                f"Got {type(reactants).__name__}."
            )

        return list(reactants)

    @staticmethod
    def _parse_group_inputs(
        reactants: ReactantGroup,
        role: str,
        allow_empty: bool = False,
    ) -> list[tuple[ThermoReactant, float]]:
        items = Reactants._group_items(reactants, role=role)

        if not items:
            if allow_empty:
                return []
            raise ValueError(f"At least one {role} is required.")

        parsed: list[tuple[ThermoReactant, float]] = []

        for item in items:
            if isinstance(item, tuple):
                reactant, weight = item
                weight = float(weight)
            else:
                reactant = item
                weight = 1.0

            if not Reactants._is_reactant_object(reactant):
                raise TypeError(
                    f"{role} entries must be Propellant or CombustionGas objects, "
                    f"or ({role}, weight) tuples. Got {type(reactant).__name__}."
                )

            if weight < 0.0:
                raise ValueError(f"{role} weights must be nonnegative.")

            parsed.append((reactant, weight))

        weight_sum = sum(weight for _, weight in parsed)

        if weight_sum <= 0.0:
            raise ValueError(f"{role} weights must sum to a positive value.")

        return parsed

    @staticmethod
    def _expand_reactant(
        reactant: ThermoReactant,
        mass: float,
        role: str,
    ) -> list[Reactant]:
        if isinstance(reactant, Propellant):
            return [
                Reactant(
                    propellant=reactant,
                    mass=mass,
                    role=role,
                )
            ]

        if isinstance(reactant, CombustionGas):
            return [
                Reactant(
                    propellant=reactant,
                    mass=mass * fraction,
                    role=role,
                    species_name=species,
                )
                for species, fraction in reactant.mass_fractions.items()
                if fraction > 0.0
            ]

        raise TypeError(f"Unsupported reactant type: {type(reactant).__name__}")

    @staticmethod
    def _build_group(
        parsed: list[tuple[ThermoReactant, float]],
        total_mass: float,
        role: str,
    ) -> list[Reactant]:
        if not parsed or total_mass <= 0.0:
            return []

        weight_sum = sum(weight for _, weight in parsed)

        if weight_sum <= 0.0:
            raise ValueError(f"{role} weights must sum to a positive value.")

        entries: list[Reactant] = []

        for reactant, weight in parsed:
            reactant_mass = total_mass * weight / weight_sum
            entries.extend(Reactants._expand_reactant(reactant, reactant_mass, role))

        return entries

    def _rebuild_entries(self) -> None:
        if self._mixture_ratio is None:
            return

        if not self._fuel_inputs:
            raise ValueError("At least one fuel is required.")

        if self._mixture_ratio > 0.0 and not self._oxidizer_inputs:
            raise ValueError("At least one oxidizer is required when mixture_ratio > 0.")

        fuel_mass = 1.0
        oxidizer_mass = self._mixture_ratio
        propellant_mass = fuel_mass + oxidizer_mass

        inert_mass = self._inert_fraction * propellant_mass
        igniter_mass = self._igniter_fraction * propellant_mass

        self.fuels = self._build_group(
            self._fuel_inputs,
            total_mass=fuel_mass,
            role="fuel",
        )

        self.oxidizers = self._build_group(
            self._oxidizer_inputs,
            total_mass=oxidizer_mass,
            role="oxidizer",
        )

        self.inerts = self._build_group(
            self._inert_inputs,
            total_mass=inert_mass,
            role="inert",
        )

        self.igniters = self._build_group(
            self._igniter_inputs,
            total_mass=igniter_mass,
            role="igniter",
        )

        self.entries = [
            *self.fuels,
            *self.oxidizers,
            *self.inerts,
            *self.igniters,
        ]

        self.total_mass = sum(entry.mass for entry in self.entries)

        if self.total_mass <= 0.0:
            raise ValueError("Total reactant mass must be positive.")

    @staticmethod
    def _sum_fractions(entries: list[Reactant], total: float) -> dict[str, float]:
        out: dict[str, float] = {}

        if total <= 0.0:
            return out

        for entry in entries:
            out[entry.cea_name] = out.get(entry.cea_name, 0.0) + entry.mass / total

        return dict(sorted(out.items()))

    @staticmethod
    def _sum_mole_fractions(entries: list[Reactant], total_moles: float) -> dict[str, float]:
        out: dict[str, float] = {}

        if total_moles <= 0.0:
            return out

        for entry in entries:
            out[entry.cea_name] = out.get(entry.cea_name, 0.0) + entry.moles / total_moles

        return dict(sorted(out.items()))

    @staticmethod
    def _input_names(reactant: ThermoReactant) -> set[str]:
        names = {reactant.name}

        if isinstance(reactant, Propellant):
            for value in (
                reactant.cea_name,
                reactant.propellant,
                reactant.input_name,
                reactant.registry_name,
                reactant.rocketprops_name,
                reactant.cea_species,
                reactant.cea_reactant,
            ):
                if value is not None:
                    names.add(value)

        elif isinstance(reactant, CombustionGas):
            names.update(reactant.species)
            names.update(reactant.mass_fractions)
            names.update(reactant.mole_fractions)

        return {str(name) for name in names if name is not None}

    @staticmethod
    def _updated_group_weights(
        group: list[tuple[ThermoReactant, float]],
        values: dict[str, float],
        role: str,
    ) -> list[tuple[ThermoReactant, float]]:
        if not isinstance(values, dict):
            raise TypeError(f"{role}_mass_fractions must be a dict.")

        by_name = {}

        for reactant, _ in group:
            for name in Reactants._input_names(reactant):
                by_name[name] = reactant

        updated: list[tuple[ThermoReactant, float]] = []

        for key, fraction in values.items():
            if key not in by_name:
                raise ValueError(f"{key!r} is not present in the {role} group.")

            fraction = float(fraction)

            if fraction < 0.0:
                raise ValueError(f"{role} mass fractions must be nonnegative.")

            updated.append((by_name[key], fraction))

        provided = {reactant for reactant, _ in updated}

        for reactant, weight in group:
            if reactant not in provided:
                updated.append((reactant, 0.0))

        total = sum(weight for _, weight in updated)

        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{role} mass fractions must sum to 1.0. Got {total}."
            )

        return updated


    @staticmethod
    def _object_cache_key(value) -> tuple:
        cache_key = getattr(value, "cache_key", None)

        if callable(cache_key):
            try:
                return ("cache_key", cache_key())
            except Exception:
                pass

        return ("object", type(value).__name__, id(value))

    def cache_key(self) -> tuple:
        """Execute the documented ``cache_key`` operation for ``Reactants``.

        Arguments are validated and normalized using the same rules as the high-level
        wrappers.  Return values follow ThermoProp's SI-unit and composition
        conventions, and failures are reported through ThermoProp exception types with
        contextual messages rather than silent fallbacks.
        """

        def group_key(group):
            return tuple(
                (self._object_cache_key(reactant), round(float(weight), 15))
                for reactant, weight in group
            )

        return (
            "Reactants",
            round(float(self._mixture_ratio), 15),
            round(float(self._inert_fraction), 15),
            round(float(self._igniter_fraction), 15),
            group_key(self._fuel_inputs),
            group_key(self._oxidizer_inputs),
            group_key(self._inert_inputs),
            group_key(self._igniter_inputs),
        )

    @property
    def mixture_ratio(self) -> float:
        """Return the oxidizer-to-fuel mass ratio for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are dimensionless.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self._mixture_ratio

    @mixture_ratio.setter
    def mixture_ratio(self, value: float) -> None:
        """Set the oxidizer-to-fuel mass ratio for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (dimensionless) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        value = float(value)

        if value < 0.0:
            raise ValueError("mixture_ratio must be nonnegative.")

        self._mixture_ratio = value
        self._rebuild_entries()

    @property
    def inert_fraction(self) -> float:
        """Return the public ``inert_fraction`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._inert_fraction

    @inert_fraction.setter
    def inert_fraction(self, value: float) -> None:
        """Set the inert fraction for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._inert_fraction = self._validate_fraction(value, "inert_fraction")
        self._rebuild_entries()

    @property
    def igniter_fraction(self) -> float:
        """Return the public ``igniter_fraction`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._igniter_fraction

    @igniter_fraction.setter
    def igniter_fraction(self, value: float) -> None:
        """Set the igniter fraction for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._igniter_fraction = self._validate_fraction(value, "igniter_fraction")
        self._rebuild_entries()

    @property
    def fuel_weights(self) -> list[float]:
        """Return the public ``fuel_weights`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return [weight for _, weight in self._fuel_inputs]

    @fuel_weights.setter
    def fuel_weights(self, weights: list[float]) -> None:
        """Set the fuel weights for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        if len(weights) != len(self._fuel_inputs):
            raise ValueError(
                f"Expected {len(self._fuel_inputs)} fuel weights, "
                f"got {len(weights)}."
            )

        self._fuel_inputs = [
            (reactant, float(weight))
            for (reactant, _), weight
            in zip(self._fuel_inputs, weights)
        ]

        self._rebuild_entries()

    @property
    def oxidizer_weights(self) -> list[float]:
        """Return the public ``oxidizer_weights`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return [weight for _, weight in self._oxidizer_inputs]

    @oxidizer_weights.setter
    def oxidizer_weights(self, weights: list[float]) -> None:
        """Set the oxidizer weights for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        if len(weights) != len(self._oxidizer_inputs):
            raise ValueError(
                f"Expected {len(self._oxidizer_inputs)} oxidizer weights, "
                f"got {len(weights)}."
            )

        self._oxidizer_inputs = [
            (reactant, float(weight))
            for (reactant, _), weight
            in zip(self._oxidizer_inputs, weights)
        ]

        self._rebuild_entries()

    @property
    def inert_weights(self) -> list[float]:
        """Return the public ``inert_weights`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return [weight for _, weight in self._inert_inputs]

    @inert_weights.setter
    def inert_weights(self, weights: list[float]) -> None:
        """Set the inert weights for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        if len(weights) != len(self._inert_inputs):
            raise ValueError(
                f"Expected {len(self._inert_inputs)} inert weights, "
                f"got {len(weights)}."
            )

        self._inert_inputs = [
            (reactant, float(weight))
            for (reactant, _), weight
            in zip(self._inert_inputs, weights)
        ]

        self._rebuild_entries()

    @property
    def igniter_weights(self) -> list[float]:
        """Return the public ``igniter_weights`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return [weight for _, weight in self._igniter_inputs]

    @igniter_weights.setter
    def igniter_weights(self, weights: list[float]) -> None:
        """Set the igniter weights for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        if len(weights) != len(self._igniter_inputs):
            raise ValueError(
                f"Expected {len(self._igniter_inputs)} igniter weights, "
                f"got {len(weights)}."
            )

        self._igniter_inputs = [
            (reactant, float(weight))
            for (reactant, _), weight
            in zip(self._igniter_inputs, weights)
        ]

        self._rebuild_entries()

    def set_fuel_weights(self, weights: list[float]) -> None:
        """Execute the public ``set_fuel_weights`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self.fuel_weights = weights

    def set_oxidizer_weights(self, weights: list[float]) -> None:
        """Execute the public ``set_oxidizer_weights`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self.oxidizer_weights = weights

    def set_inert_weights(self, weights: list[float]) -> None:
        """Execute the public ``set_inert_weights`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self.inert_weights = weights

    def set_igniter_weights(self, weights: list[float]) -> None:
        """Execute the public ``set_igniter_weights`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self.igniter_weights = weights

    @property
    def fuel_inputs(self) -> list[tuple[ThermoReactant, float]]:
        """Return the public ``fuel_inputs`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return list(self._fuel_inputs)

    @property
    def oxidizer_inputs(self) -> list[tuple[ThermoReactant, float]]:
        """Return the public ``oxidizer_inputs`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return list(self._oxidizer_inputs)

    @property
    def inert_inputs(self) -> list[tuple[ThermoReactant, float]]:
        """Return the public ``inert_inputs`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return list(self._inert_inputs)

    @property
    def igniter_inputs(self) -> list[tuple[ThermoReactant, float]]:
        """Return the public ``igniter_inputs`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return list(self._igniter_inputs)

    def set_fuels(self, fuels: ReactantGroup) -> None:
        """Execute the public ``set_fuels`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")
        self._rebuild_entries()

    def set_oxidizers(self, oxidizers: ReactantGroup) -> None:
        """Execute the public ``set_oxidizers`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self._oxidizer_inputs = self._parse_group_inputs(
            oxidizers,
            role="oxidizer",
        )
        self._rebuild_entries()

    def set_inerts(self, inerts: ReactantGroup) -> None:
        """Execute the public ``set_inerts`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self._inert_inputs = self._parse_group_inputs(
            inerts,
            role="inert",
            allow_empty=True,
        )
        self._rebuild_entries()

    def set_igniters(self, igniters: ReactantGroup) -> None:
        """Execute the public ``set_igniters`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        self._igniter_inputs = self._parse_group_inputs(
            igniters,
            role="igniter",
            allow_empty=True,
        )
        self._rebuild_entries()

    @property
    def fuel_mass(self) -> float:
        """Return the fuel mass used to define the reactant basis for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return sum(entry.mass for entry in self.fuels)

    @property
    def oxidizer_mass(self) -> float:
        """Return the oxidizer mass used to define the reactant basis for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return sum(entry.mass for entry in self.oxidizers)

    @property
    def inert_mass(self) -> float:
        """Return the public ``inert_mass`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return sum(entry.mass for entry in self.inerts)

    @property
    def igniter_mass(self) -> float:
        """Return the public ``igniter_mass`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return sum(entry.mass for entry in self.igniters)

    @property
    def propellant_mass(self) -> float:
        """Return the public ``propellant_mass`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.fuel_mass + self.oxidizer_mass

    @property
    def oxidizer_to_fuel_ratio(self) -> float:
        """Return the oxidizer-to-fuel mass ratio for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are dimensionless.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self.oxidizer_mass / self.fuel_mass

    @property
    def total_moles(self) -> float:
        """Return the public ``total_moles`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return sum(entry.moles for entry in self.entries)

    @property
    def total_kmoles(self) -> float:
        """Return the public ``total_kmoles`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.total_moles / 1000.0

    @property
    def molecular_weight(self) -> float:
        """Return the molecular weight for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are kg/kmol, numerically equal to g/mol.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        return self.total_mass / self.total_moles

    @property
    def molecular_weight_kg_per_kmol(self) -> float:
        """Return the public ``molecular_weight_kg_per_kmol`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self.total_mass / self.total_kmoles

    @property
    def mass_fractions(self) -> dict[str, float]:
        """Return the mass-fraction composition dictionary for this ``Reactants`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return self._sum_fractions(self.entries, self.total_mass)

    @property
    def mole_fractions(self) -> dict[str, float]:
        """Return the mole-fraction composition dictionary for this ``Reactants`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return self._sum_mole_fractions(self.entries, self.total_moles)

    @property
    def fuel_mass_fractions(self) -> dict[str, float]:
        """Return the public ``fuel_mass_fractions`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._sum_fractions(self.fuels, self.fuel_mass)

    @fuel_mass_fractions.setter
    def fuel_mass_fractions(self, values: dict[str, float]) -> None:
        """Set the fuel mass fractions for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._fuel_inputs = self._updated_group_weights(
            self._fuel_inputs,
            values,
            role="fuel",
        )
        self._rebuild_entries()

    @property
    def oxidizer_mass_fractions(self) -> dict[str, float]:
        """Return the public ``oxidizer_mass_fractions`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._sum_fractions(self.oxidizers, self.oxidizer_mass)

    @oxidizer_mass_fractions.setter
    def oxidizer_mass_fractions(self, values: dict[str, float]) -> None:
        """Set the oxidizer mass fractions for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._oxidizer_inputs = self._updated_group_weights(
            self._oxidizer_inputs,
            values,
            role="oxidizer",
        )
        self._rebuild_entries()

    @property
    def inert_mass_fractions(self) -> dict[str, float]:
        """Return the public ``inert_mass_fractions`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._sum_fractions(self.inerts, self.inert_mass)

    @inert_mass_fractions.setter
    def inert_mass_fractions(self, values: dict[str, float]) -> None:
        """Set the inert mass fractions for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._inert_inputs = self._updated_group_weights(
            self._inert_inputs,
            values,
            role="inert",
        )
        self._rebuild_entries()

    @property
    def igniter_mass_fractions(self) -> dict[str, float]:
        """Return the public ``igniter_mass_fractions`` value for this ``Reactants`` object.

        The value is computed from the current wrapper state and follows ThermoProp's SI
        unit convention unless this property is explicitly metadata.  Unsupported values
        raise a ThermoProp exception with context about the selected backend and state."""
        return self._sum_fractions(self.igniters, self.igniter_mass)

    @igniter_mass_fractions.setter
    def igniter_mass_fractions(self, values: dict[str, float]) -> None:
        """Set the igniter mass fractions for this ``Reactants`` instance.

        Assigning this value uses the same SI-unit convention as the corresponding
        getter (see getter documentation) unless the getter documents a metadata value instead.  Setters
        that define a thermodynamic state immediately re-evaluate the wrapper or mark the
        state stale according to that wrapper's update policy, so subsequent property
        access reflects the new input state."""
        self._igniter_inputs = self._updated_group_weights(
            self._igniter_inputs,
            values,
            role="igniter",
        )
        self._rebuild_entries()

    @property
    def element_moles(self) -> dict[str, float]:
        """Return the element mole totals for this ``Reactants`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        element_moles: dict[str, float] = {}

        for entry in self.entries:
            comp = entry.elemental_composition

            if not comp:
                raise ValueError(
                    f"{entry.name!r} does not have CEA elemental composition data."
                )

            for element, count in comp.items():
                element_moles[element] = (
                    element_moles.get(element, 0.0)
                    + entry.moles * float(count)
                )

        return dict(sorted(element_moles.items()))

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        """Return the element mole totals per kilogram for this ``Reactants`` object.

        This metadata is normalized by ThermoProp so user code can inspect the active
        backend, canonical names, composition basis, or solver bookkeeping without
        reaching into private attributes.  Returned mappings and sequences are copies or
        read-only views where practical, so callers can use them for reporting and model
        setup without mutating the wrapper accidentally."""
        return {
            element: value / self.total_mass
            for element, value in self.element_moles.items()
        }

    @property
    def reactant_enthalpy(self) -> float:
        """Return the mass-specific reactant enthalpy of the complete mixture for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are J/kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        total = 0.0

        for entry in self.entries:
            h = entry.enthalpy

            if h is None:
                raise PropertyUnavailableError(f"{entry.name!r} does not have enthalpy data.")

            total += entry.mass * h

        return total / self.total_mass

    @property
    def reactant_internal_energy(self) -> float:
        """Return the mass-specific reactant internal energy of the complete mixture for this ``Reactants`` state.

        The value is evaluated from the active backend and the current state variables.
        Units are J/kg.  If the selected species, material, phase, or thermodynamic
        state does not support this property, ThermoProp raises a descriptive
        ``PropertyUnavailableError`` or backend-specific ThermoProp exception instead of
        silently returning an invalid value."""
        total = 0.0

        for entry in self.entries:
            u = entry.internal_energy

            if u is None:
                raise PropertyUnavailableError(f"{entry.name!r} does not have internal energy data.")

            total += entry.mass * u

        return total / self.total_mass

    def element_vector(
        self,
        elements: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Execute the public ``element_vector`` operation for ``Reactants``.

        This method is part of the importable ThermoProp API rather than an internal
        helper.  Arguments are validated and normalized before use, return values follow
        ThermoProp's SI-unit and composition conventions, and lookup or state failures
        are reported through ThermoProp exception types with contextual messages."""
        if elements is None:
            elements = sorted(self.element_moles_per_kg)

        b = np.array(
            [
                self.element_moles_per_kg.get(element, 0.0)
                for element in elements
            ],
            dtype=float,
        )

        return elements, b

    def _safe_property(self, name: str):
        try:
            return getattr(self, name)
        except Exception:
            return None

    @staticmethod
    def _safe_entry_property(entry: Reactant, name: str):
        try:
            return getattr(entry, name)
        except Exception:
            return None

    def as_dict(self) -> dict:
        """Return a structured representation of the requested ThermoProp data.

        The result is intended for inspection, reporting, testing, and advanced users who
        need direct access to database-backed values.  Names are resolved through the
        same canonicalization and alias rules used by the high-level wrappers."""
        return {
            "mixture_ratio": self.oxidizer_to_fuel_ratio,
            "inert_fraction": self.inert_fraction,
            "igniter_fraction": self.igniter_fraction,
            "total_mass": self.total_mass,
            "propellant_mass": self.propellant_mass,
            "fuel_mass": self.fuel_mass,
            "oxidizer_mass": self.oxidizer_mass,
            "inert_mass": self.inert_mass,
            "igniter_mass": self.igniter_mass,
            "molecular_weight": self.molecular_weight,
            "molecular_weight_kg_per_kmol": self.molecular_weight_kg_per_kmol,
            "total_moles": self.total_moles,
            "total_kmoles": self.total_kmoles,
            "mass_fractions": self.mass_fractions,
            "mole_fractions": self.mole_fractions,
            "fuel_mass_fractions": self.fuel_mass_fractions,
            "oxidizer_mass_fractions": self.oxidizer_mass_fractions,
            "inert_mass_fractions": self.inert_mass_fractions,
            "igniter_mass_fractions": self.igniter_mass_fractions,
            "reactant_enthalpy": self.reactant_enthalpy,
            "reactant_internal_energy": self._safe_property("reactant_internal_energy"),
            "element_moles": self.element_moles,
            "element_moles_per_kg": self.element_moles_per_kg,
            "reactants": [
                {
                    "name": entry.name,
                    "cea_name": entry.cea_name,
                    "role": entry.role,
                    "mass": entry.mass,
                    "moles": entry.moles,
                    "kmoles": entry.kmoles,
                    "mass_fraction": entry.mass / self.total_mass,
                    "mole_fraction": entry.moles / self.total_moles,
                    "temperature": entry.temperature,
                    "pressure": entry.pressure,
                    "enthalpy": entry.enthalpy,
                    "internal_energy": self._safe_entry_property(
                        entry,
                        "internal_energy",
                    ),
                    "elemental_composition": entry.elemental_composition,
                }
                for entry in self.entries
            ],
        }

    def __str__(self) -> str:
        rows = [
            ("Mixture ratio O/F", f"{self.oxidizer_to_fuel_ratio:.6g}"),
            ("Inert fraction", f"{self.inert_fraction:.6g}"),
            ("Igniter fraction", f"{self.igniter_fraction:.6g}"),
        ]

        if self.fuels:
            fuel_text = ", ".join(
                f"{name} ({100 * frac:.3f}%)"
                for name, frac in self.fuel_mass_fractions.items()
            )
            rows.append(("Fuels", fuel_text))

        if self.oxidizers:
            oxidizer_text = ", ".join(
                f"{name} ({100 * frac:.3f}%)"
                for name, frac in self.oxidizer_mass_fractions.items()
            )
            rows.append(("Oxidizers", oxidizer_text))

        if self.inerts:
            inert_text = ", ".join(
                f"{name} ({100 * frac:.3f}%)"
                for name, frac in self.inert_mass_fractions.items()
            )
            rows.append(("Inerts", inert_text))

        if self.igniters:
            igniter_text = ", ".join(
                f"{name} ({100 * frac:.3f}%)"
                for name, frac in self.igniter_mass_fractions.items()
            )
            rows.append(("Igniters", igniter_text))

        width = max(len(key) for key, _ in rows)

        return "\n".join(
            f"{key:<{width}} : {value}"
            for key, value in rows
        )
