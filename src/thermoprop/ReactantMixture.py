from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .Propellant import Propellant


@dataclass(frozen=True)
class ReactantEntry:
    propellant: Propellant
    mass: float
    role: str

    @property
    def moles(self) -> float:
        """
        Amount of reactant formula units [mol].
        """
        mw = self.propellant.cea_formula_molar_mass

        if mw is None or mw <= 0.0:
            raise ValueError(
                f"{self.propellant.name!r} does not have a valid "
                "CEA formula molar mass."
            )

        return self.mass / mw

    @property
    def kmoles(self) -> float:
        """
        Amount of reactant formula units [kmol].
        """
        return self.moles / 1000.0


class ReactantMixture:
    """
    CEA-style reactant mixture builder.

    This class converts any number of Propellant objects into the CEA
    reactant quantities needed by an equilibrium solver:

        element moles per kg mixture
        reactant enthalpy per kg mixture
        reactant internal energy per kg mixture
        reactant mean molecular weight
    """

    def __init__(
        self,
        fuels: Iterable[Propellant | tuple[Propellant, float]],
        oxidizers: Iterable[Propellant | tuple[Propellant, float]],
        mixture_ratio: float,
    ):
        self.mixture_ratio = float(mixture_ratio)

        if self.mixture_ratio < 0.0:
            raise ValueError("mixture_ratio must be nonnegative.")

        self.fuels = self._normalize_entries(
            fuels,
            total_mass=1.0,
            role="fuel",
        )

        self.oxidizers = self._normalize_entries(
            oxidizers,
            total_mass=self.mixture_ratio,
            role="oxidizer",
        )

        if not self.fuels:
            raise ValueError("At least one fuel is required.")

        if not self.oxidizers and self.mixture_ratio > 0.0:
            raise ValueError("At least one oxidizer is required.")

        self.entries = [*self.fuels, *self.oxidizers]
        self.total_mass = sum(entry.mass for entry in self.entries)

        if self.total_mass <= 0.0:
            raise ValueError("Total reactant mass must be positive.")

    @staticmethod
    def _normalize_entries(
        propellants: Iterable[Propellant | tuple[Propellant, float]],
        total_mass: float,
        role: str,
    ) -> list[ReactantEntry]:
        items = list(propellants)

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

            if weight < 0.0:
                raise ValueError(f"{role} weights must be nonnegative.")

            parsed.append((propellant, weight))

        weight_sum = sum(weight for _, weight in parsed)

        if weight_sum <= 0.0:
            raise ValueError(f"{role} weights must sum to a positive value.")

        return [
            ReactantEntry(
                propellant=propellant,
                mass=total_mass * weight / weight_sum,
                role=role,
            )
            for propellant, weight in parsed
        ]

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

    @property
    def total_moles(self) -> float:
        return sum(entry.moles for entry in self.entries)
        
    @property
    def total_kmoles(self) -> float:
        return self.total_moles / 1000.0
        
    @property
    def molecular_weight(self) -> float:
        """
        Mixture molar mass in kg/mol based on CEA formula moles.
        """
        return self.total_mass / self.total_moles

    @property
    def molecular_weight_kg_per_kmol(self) -> float:
        """
        Mixture molecular weight in kg/kmol.
        Numerically equal to g/mol.
        """
        return self.total_mass / self.total_kmoles

    @property
    def fuel_mass(self) -> float:
        return sum(entry.mass for entry in self.fuels)

    @property
    def oxidizer_mass(self) -> float:
        return sum(entry.mass for entry in self.oxidizers)

    @property
    def oxidizer_to_fuel_ratio(self) -> float:
        return self.oxidizer_mass / self.fuel_mass

    def element_vector(self, elements: list[str] | None = None) -> tuple[list[str], np.ndarray]:
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

    def as_dict(self) -> dict:
        return {
            "mixture_ratio": self.oxidizer_to_fuel_ratio,
            "total_mass": self.total_mass,
            "fuel_mass": self.fuel_mass,
            "oxidizer_mass": self.oxidizer_mass,
            "molecular_weight": self.molecular_weight,
            "reactant_enthalpy": self.reactant_enthalpy,
            "reactant_internal_energy": self.reactant_internal_energy,
            "element_moles": self.element_moles,
            "element_moles_per_kg": self.element_moles_per_kg,
            "reactants": [
                {
                    "name": entry.propellant.name,
                    "cea_name": entry.propellant.cea_name,
                    "role": entry.role,
                    "mass": entry.mass,
                    "moles": entry.moles,
                    "temperature": entry.propellant.temperature,
                    "pressure": entry.propellant.pressure,
                    "enthalpy": entry.propellant.enthalpy,
                    "internal_energy": entry.propellant.internal_energy,
                    "elemental_composition": entry.propellant.elemental_composition,
                }
                for entry in self.entries
            ],
        }

    def __str__(self) -> str:
        rows = [
            ("Mixture ratio O/F", f"{self.oxidizer_to_fuel_ratio:.6g}"),
            ("Total mass basis [kg]", f"{self.total_mass:.6g}"),
            ("Fuel mass [kg]", f"{self.fuel_mass:.6g}"),
            ("Oxidizer mass [kg]", f"{self.oxidizer_mass:.6g}"),
            ("CEA mixture MW [kg/mol]", f"{self.molecular_weight:.8g}"),
            ("Reactant enthalpy [J/kg]", f"{self.reactant_enthalpy:.8e}"),
            ("Reactant internal energy [J/kg]", f"{self.reactant_internal_energy:.8e}"),
            ("Element moles per kg", self.element_moles_per_kg),
        ]

        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)