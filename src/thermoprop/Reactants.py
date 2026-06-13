from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .Propellant import Propellant


PropellantEntry = Propellant | tuple[Propellant, float]
PropellantGroup = PropellantEntry | Iterable[PropellantEntry]


@dataclass(frozen=True)
class Reactant:
    propellant: Propellant
    mass: float
    role: str

    @property
    def name(self) -> str:
        return self.propellant.name

    @property
    def cea_name(self) -> str:
        return self.propellant.cea_name

    @property
    def molar_mass(self) -> float:
        mw = self.propellant.cea_formula_molar_mass

        if mw is None or mw <= 0.0:
            raise ValueError(
                f"{self.propellant.name!r} does not have a valid "
                "CEA formula molar mass."
            )

        return float(mw)

    @property
    def moles(self) -> float:
        return self.mass / self.molar_mass

    @property
    def kmoles(self) -> float:
        return self.moles / 1000.0


class Reactants:
    """
    CEA-style reactant mixture definition for equilibrium calculations.

    Reactants groups Propellant objects into fuels and oxidizers, applies a mass
    mixture ratio, and computes the elemental inventory and reactant enthalpy
    needed by Equilibrium.

    Inputs
    ------

    Fuels and oxidizers must be Propellant objects or weighted collections of
    Propellant objects. Raw strings are intentionally not accepted because
    temperature, pressure, and optional quality-corrected enthalpy should be
    explicit.

    Single fuel and oxidizer:

        Reactants(
            fuels=Propellant("RP-1", temperature=298.15),
            oxidizers=Propellant("LOX", temperature=90.17),
            mixture_ratio=2.5,
        )

    Multiple weighted entries:

        Reactants(
            fuels=[
                (Propellant("RP-1", temperature=298.15), 80.0),
                (Propellant("Ethanol", temperature=298.15), 20.0),
            ],
            oxidizers=[
                Propellant("LOX", temperature=90.17),
            ],
            mixture_ratio=2.5,
        )

    Mass basis
    ----------

    The internal basis is:

        fuel mass = 1 kg
        oxidizer mass = O/F kg
        total mass = 1 + O/F kg

    Weights inside each fuel or oxidizer group are relative mass weights, similar
    to CEA wt% entries.

    Outputs
    -------

    Reactants provides mass fractions, mole fractions, elemental mole totals,
    molecular weight, total moles, total kmoles, reactant enthalpy, and optional
    internal energy when available.

    Notes
    -----

    Reactants is intended for equilibrium setup, not flow-property modeling. Use
    Fluid, IdealGas, Propellant, or CombustionGas for property evaluation before
    building the reactant set.
    """
    def __init__(
        self,
        fuels: PropellantGroup,
        oxidizers: PropellantGroup,
        mixture_ratio: float,
    ):
        self._fuel_inputs = None
        self._oxidizer_inputs = None
        self._mixture_ratio = None

        self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")
        self._oxidizer_inputs = self._parse_group_inputs(oxidizers, role="oxidizer")
        self.mixture_ratio = mixture_ratio

    @staticmethod
    def _group_items(propellants: PropellantGroup) -> list[PropellantEntry]:
        if isinstance(propellants, str):
            raise TypeError(
                "Reactants does not accept raw string propellants. "
                "Pass Propellant objects so temperature and pressure are explicit."
            )

        if isinstance(propellants, Propellant):
            return [propellants]

        if isinstance(propellants, tuple):
            if len(propellants) == 2 and isinstance(propellants[0], Propellant):
                return [propellants]

        return list(propellants)

    @staticmethod
    def _parse_group_inputs(
        propellants: PropellantGroup,
        role: str,
    ) -> list[tuple[Propellant, float]]:
        items = Reactants._group_items(propellants)

        if not items:
            return []

        parsed: list[tuple[Propellant, float]] = []

        for item in items:
            if isinstance(item, tuple):
                propellant, weight = item
                weight = float(weight)
            else:
                propellant = item
                weight = 1.0

            if not isinstance(propellant, Propellant):
                raise TypeError(
                    f"{role} entries must be Propellant objects or "
                    f"(Propellant, weight) tuples. Got {type(propellant).__name__}."
                )

            if weight < 0.0:
                raise ValueError(f"{role} weights must be nonnegative.")

            parsed.append((propellant, weight))

        weight_sum = sum(weight for _, weight in parsed)

        if weight_sum <= 0.0:
            raise ValueError(f"{role} weights must sum to a positive value.")

        return parsed

    @staticmethod
    def _normalize_group(
        propellants: PropellantGroup,
        total_mass: float,
        role: str,
    ) -> list[Reactant]:
        parsed = Reactants._parse_group_inputs(propellants, role)

        weight_sum = sum(weight for _, weight in parsed)

        return [
            Reactant(
                propellant=propellant,
                mass=total_mass * weight / weight_sum,
                role=role,
            )
            for propellant, weight in parsed
        ]

    def _rebuild_entries(self) -> None:
        if self._mixture_ratio is None:
            return

        if not self._fuel_inputs:
            raise ValueError("At least one fuel is required.")

        if self._mixture_ratio > 0.0 and not self._oxidizer_inputs:
            raise ValueError("At least one oxidizer is required when mixture_ratio > 0.")

        self.fuels = self._build_group(
            self._fuel_inputs,
            total_mass=1.0,
            role="fuel",
        )

        self.oxidizers = self._build_group(
            self._oxidizer_inputs,
            total_mass=self._mixture_ratio,
            role="oxidizer",
        )

        self.entries = [*self.fuels, *self.oxidizers]
        self.total_mass = sum(entry.mass for entry in self.entries)

        if self.total_mass <= 0.0:
            raise ValueError("Total reactant mass must be positive.")

    @staticmethod
    def _build_group(
        parsed: list[tuple[Propellant, float]],
        total_mass: float,
        role: str,
    ) -> list[Reactant]:
        if not parsed:
            return []

        weight_sum = sum(weight for _, weight in parsed)

        if weight_sum <= 0.0:
            raise ValueError(f"{role} weights must sum to a positive value.")

        return [
            Reactant(
                propellant=propellant,
                mass=total_mass * weight / weight_sum,
                role=role,
            )
            for propellant, weight in parsed
        ]

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
            (propellant, float(weight))
            for (propellant, _), weight
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
            (propellant, float(weight))
            for (propellant, _), weight
            in zip(self._oxidizer_inputs, weights)
        ]

        self._rebuild_entries()

    def set_fuel_weights(self, weights: list[float]) -> None:
        self.fuel_weights = weights

    def set_oxidizer_weights(self, weights: list[float]) -> None:
        self.oxidizer_weights = weights

    @property
    def fuel_inputs(self) -> list[tuple[Propellant, float]]:
        return list(self._fuel_inputs)

    @property
    def oxidizer_inputs(self) -> list[tuple[Propellant, float]]:
        return list(self._oxidizer_inputs)

    def set_fuels(self, fuels: PropellantGroup) -> None:
        self._fuel_inputs = self._parse_group_inputs(fuels, role="fuel")
        self._rebuild_entries()

    def set_oxidizers(self, oxidizers: PropellantGroup) -> None:
        self._oxidizer_inputs = self._parse_group_inputs(
            oxidizers,
            role="oxidizer",
        )
        self._rebuild_entries()

    @property
    def fuel_mass(self) -> float:
        return sum(entry.mass for entry in self.fuels)

    @property
    def oxidizer_mass(self) -> float:
        return sum(entry.mass for entry in self.oxidizers)

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
        total = self.total_mass

        return {
            entry.cea_name: entry.mass / total
            for entry in self.entries
        }

    @property
    def mole_fractions(self) -> dict[str, float]:
        total = self.total_moles

        return {
            entry.cea_name: entry.moles / total
            for entry in self.entries
        }

    @property
    def fuel_mass_fractions(self) -> dict[str, float]:
        total = self.fuel_mass

        return {
            entry.cea_name: entry.mass / total
            for entry in self.fuels
        }

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
        total = self.oxidizer_mass

        if total <= 0.0:
            return {}

        return {
            entry.cea_name: entry.mass / total
            for entry in self.oxidizers
        }

    @oxidizer_mass_fractions.setter
    def oxidizer_mass_fractions(self, values: dict[str, float]) -> None:
        self._oxidizer_inputs = self._updated_group_weights(
            self._oxidizer_inputs,
            values,
            role="oxidizer",
        )
        self._rebuild_entries()

    @staticmethod
    def _updated_group_weights(
        group: list[tuple[Propellant, float]],
        values: dict[str, float],
        role: str,
    ) -> list[tuple[Propellant, float]]:
        if not isinstance(values, dict):
            raise TypeError(f"{role}_mass_fractions must be a dict.")

        by_name = {}
        for propellant, _ in group:
            by_name[propellant.cea_name] = propellant
            by_name[propellant.name] = propellant
            by_name[propellant.propellant] = propellant
            by_name[propellant.input_name] = propellant

        updated: list[tuple[Propellant, float]] = []

        for key, fraction in values.items():
            if key not in by_name:
                raise ValueError(f"{key!r} is not present in the {role} group.")

            fraction = float(fraction)

            if fraction < 0.0:
                raise ValueError(f"{role} mass fractions must be nonnegative.")

            updated.append((by_name[key], fraction))

        provided = {propellant for propellant, _ in updated}

        for propellant, weight in group:
            if propellant not in provided:
                updated.append((propellant, 0.0))

        total = sum(weight for _, weight in updated)

        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{role} mass fractions must sum to 1.0. Got {total}."
            )

        return updated

    @property
    def element_moles(self) -> dict[str, float]:
        element_moles: dict[str, float] = {}

        for entry in self.entries:
            comp = entry.propellant.elemental_composition

            if not comp:
                raise ValueError(
                    f"{entry.propellant.name!r} does not have CEA "
                    "elemental composition data."
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
            h = entry.propellant.enthalpy

            if h is None:
                raise ValueError(
                    f"{entry.propellant.name!r} does not have enthalpy data."
                )

            total += entry.mass * h

        return total / self.total_mass

    @property
    def reactant_internal_energy(self) -> float:
        total = 0.0

        for entry in self.entries:
            u = entry.propellant.internal_energy

            if u is None:
                raise ValueError(
                    f"{entry.propellant.name!r} does not have internal energy data."
                )

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
    def _safe_propellant_property(propellant, name: str):
        try:
            return getattr(propellant, name)
        except Exception:
            return None
        
    def as_dict(self) -> dict:
        return {
            "mixture_ratio": self.oxidizer_to_fuel_ratio,
            "total_mass": self.total_mass,
            "fuel_mass": self.fuel_mass,
            "oxidizer_mass": self.oxidizer_mass,
            "molecular_weight": self.molecular_weight,
            "molecular_weight_kg_per_kmol": self.molecular_weight_kg_per_kmol,
            "total_moles": self.total_moles,
            "total_kmoles": self.total_kmoles,
            "mass_fractions": self.mass_fractions,
            "mole_fractions": self.mole_fractions,
            "fuel_mass_fractions": self.fuel_mass_fractions,
            "oxidizer_mass_fractions": self.oxidizer_mass_fractions,
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
                    "temperature": entry.propellant.temperature,
                    "pressure": entry.propellant.pressure,
                    "enthalpy": entry.propellant.enthalpy,
                    "internal_energy": self._safe_propellant_property(
                        entry.propellant,
                        "internal_energy",
                    ),
                    "elemental_composition": entry.propellant.elemental_composition,
                }
                for entry in self.entries
            ],
        }

    def __str__(self) -> str:
        rows = [
            ("Mixture ratio O/F", f"{self.oxidizer_to_fuel_ratio:.6g}"),
        ]

        if self.fuels:
            fuel_text = ", ".join(
                f"{r.cea_name} ({100 * self.fuel_mass_fractions[r.cea_name]:.3f}%)"
                for r in self.fuels
            )
            rows.append(("Fuels", fuel_text))

        if self.oxidizers:
            oxidizer_text = ", ".join(
                f"{r.cea_name} ({100 * self.oxidizer_mass_fractions[r.cea_name]:.3f}%)"
                for r in self.oxidizers
            )
            rows.append(("Oxidizers", oxidizer_text))

        width = max(len(key) for key, _ in rows)

        return "\n".join(
            f"{key:<{width}} : {value}"
            for key, value in rows
        )