from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .CEADatabase import CEA
from .Propellant import Propellant
from .CombustionGas import CombustionGas


ThermoReactant = Propellant | CombustionGas
ReactantEntry = ThermoReactant | tuple[ThermoReactant, float]
ReactantGroup = ReactantEntry | Iterable[ReactantEntry] | None


@dataclass(frozen=True)
class Reactant:
    propellant: ThermoReactant
    mass: float
    role: str
    species_name: str | None = None

    @property
    def name(self) -> str:
        if self.species_name is not None:
            return self.species_name
        return self.propellant.name

    @property
    def cea_name(self) -> str:
        if self.species_name is not None:
            return self.species_name
        return self.propellant.cea_name

    @property
    def molar_mass(self) -> float:
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
        return self.mass / self.molar_mass

    @property
    def kmoles(self) -> float:
        return self.moles / 1000.0

    @property
    def temperature(self) -> float | None:
        return self.propellant.temperature

    @property
    def pressure(self) -> float | None:
        return self.propellant.pressure

    @property
    def enthalpy(self) -> float | None:
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
    def _group_items(reactants: ReactantGroup) -> list[ReactantEntry]:
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

        return list(reactants)

    @staticmethod
    def _parse_group_inputs(
        reactants: ReactantGroup,
        role: str,
        allow_empty: bool = False,
    ) -> list[tuple[ThermoReactant, float]]:
        items = Reactants._group_items(reactants)

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

    @property
    def mixture_ratio(self) -> float:
        return self._mixture_ratio

    @mixture_ratio.setter
    def mixture_ratio(self, value: float) -> None:
        value = float(value)

        if value < 0.0:
            raise ValueError("mixture_ratio must be nonnegative.")

        self._mixture_ratio = value
        self._rebuild_entries()

    @property
    def inert_fraction(self) -> float:
        return self._inert_fraction

    @inert_fraction.setter
    def inert_fraction(self, value: float) -> None:
        self._inert_fraction = self._validate_fraction(value, "inert_fraction")
        self._rebuild_entries()

    @property
    def igniter_fraction(self) -> float:
        return self._igniter_fraction

    @igniter_fraction.setter
    def igniter_fraction(self, value: float) -> None:
        self._igniter_fraction = self._validate_fraction(value, "igniter_fraction")
        self._rebuild_entries()

    @property
    def fuel_weights(self) -> list[float]:
        return [weight for _, weight in self._fuel_inputs]

    @fuel_weights.setter
    def fuel_weights(self, weights: list[float]) -> None:
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
        return [weight for _, weight in self._oxidizer_inputs]

    @oxidizer_weights.setter
    def oxidizer_weights(self, weights: list[float]) -> None:
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
        return [weight for _, weight in self._inert_inputs]

    @inert_weights.setter
    def inert_weights(self, weights: list[float]) -> None:
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
        return [weight for _, weight in self._igniter_inputs]

    @igniter_weights.setter
    def igniter_weights(self, weights: list[float]) -> None:
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
        self.fuel_weights = weights

    def set_oxidizer_weights(self, weights: list[float]) -> None:
        self.oxidizer_weights = weights

    def set_inert_weights(self, weights: list[float]) -> None:
        self.inert_weights = weights

    def set_igniter_weights(self, weights: list[float]) -> None:
        self.igniter_weights = weights

    @property
    def fuel_inputs(self) -> list[tuple[ThermoReactant, float]]:
        return list(self._fuel_inputs)

    @property
    def oxidizer_inputs(self) -> list[tuple[ThermoReactant, float]]:
        return list(self._oxidizer_inputs)

    @property
    def inert_inputs(self) -> list[tuple[ThermoReactant, float]]:
        return list(self._inert_inputs)

    @property
    def igniter_inputs(self) -> list[tuple[ThermoReactant, float]]:
        return list(self._igniter_inputs)

    def set_fuels(self, fuels: ReactantGroup) -> None:
        self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")
        self._rebuild_entries()

    def set_oxidizers(self, oxidizers: ReactantGroup) -> None:
        self._oxidizer_inputs = self._parse_group_inputs(
            oxidizers,
            role="oxidizer",
        )
        self._rebuild_entries()

    def set_inerts(self, inerts: ReactantGroup) -> None:
        self._inert_inputs = self._parse_group_inputs(
            inerts,
            role="inert",
            allow_empty=True,
        )
        self._rebuild_entries()

    def set_igniters(self, igniters: ReactantGroup) -> None:
        self._igniter_inputs = self._parse_group_inputs(
            igniters,
            role="igniter",
            allow_empty=True,
        )
        self._rebuild_entries()

    @property
    def fuel_mass(self) -> float:
        return sum(entry.mass for entry in self.fuels)

    @property
    def oxidizer_mass(self) -> float:
        return sum(entry.mass for entry in self.oxidizers)

    @property
    def inert_mass(self) -> float:
        return sum(entry.mass for entry in self.inerts)

    @property
    def igniter_mass(self) -> float:
        return sum(entry.mass for entry in self.igniters)

    @property
    def propellant_mass(self) -> float:
        return self.fuel_mass + self.oxidizer_mass

    @property
    def oxidizer_to_fuel_ratio(self) -> float:
        return self.oxidizer_mass / self.fuel_mass

    @property
    def total_moles(self) -> float:
        return sum(entry.moles for entry in self.entries)

    @property
    def total_kmoles(self) -> float:
        return self.total_moles / 1000.0

    @property
    def molecular_weight(self) -> float:
        return self.total_mass / self.total_moles

    @property
    def molecular_weight_kg_per_kmol(self) -> float:
        return self.total_mass / self.total_kmoles

    @property
    def mass_fractions(self) -> dict[str, float]:
        return self._sum_fractions(self.entries, self.total_mass)

    @property
    def mole_fractions(self) -> dict[str, float]:
        return self._sum_mole_fractions(self.entries, self.total_moles)

    @property
    def fuel_mass_fractions(self) -> dict[str, float]:
        return self._sum_fractions(self.fuels, self.fuel_mass)

    @fuel_mass_fractions.setter
    def fuel_mass_fractions(self, values: dict[str, float]) -> None:
        self._fuel_inputs = self._updated_group_weights(
            self._fuel_inputs,
            values,
            role="fuel",
        )
        self._rebuild_entries()

    @property
    def oxidizer_mass_fractions(self) -> dict[str, float]:
        return self._sum_fractions(self.oxidizers, self.oxidizer_mass)

    @oxidizer_mass_fractions.setter
    def oxidizer_mass_fractions(self, values: dict[str, float]) -> None:
        self._oxidizer_inputs = self._updated_group_weights(
            self._oxidizer_inputs,
            values,
            role="oxidizer",
        )
        self._rebuild_entries()

    @property
    def inert_mass_fractions(self) -> dict[str, float]:
        return self._sum_fractions(self.inerts, self.inert_mass)

    @inert_mass_fractions.setter
    def inert_mass_fractions(self, values: dict[str, float]) -> None:
        self._inert_inputs = self._updated_group_weights(
            self._inert_inputs,
            values,
            role="inert",
        )
        self._rebuild_entries()

    @property
    def igniter_mass_fractions(self) -> dict[str, float]:
        return self._sum_fractions(self.igniters, self.igniter_mass)

    @igniter_mass_fractions.setter
    def igniter_mass_fractions(self, values: dict[str, float]) -> None:
        self._igniter_inputs = self._updated_group_weights(
            self._igniter_inputs,
            values,
            role="igniter",
        )
        self._rebuild_entries()

    @property
    def element_moles(self) -> dict[str, float]:
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
        return {
            element: value / self.total_mass
            for element, value in self.element_moles.items()
        }

    @property
    def reactant_enthalpy(self) -> float:
        total = 0.0

        for entry in self.entries:
            h = entry.enthalpy

            if h is None:
                raise ValueError(f"{entry.name!r} does not have enthalpy data.")

            total += entry.mass * h

        return total / self.total_mass

    @property
    def reactant_internal_energy(self) -> float:
        total = 0.0

        for entry in self.entries:
            u = entry.internal_energy

            if u is None:
                raise ValueError(f"{entry.name!r} does not have internal energy data.")

            total += entry.mass * u

        return total / self.total_mass

    def element_vector(
        self,
        elements: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
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