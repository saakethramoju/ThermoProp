from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .Propellant import Propellant


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
    CEA-style reactant set.

    Inputs are Propellant objects grouped as fuels and oxidizers. Optional
    weights inside each group are mass weights, matching CEA wt% behavior.

    The total basis is:

        fuel mass = 1 kg
        oxidizer mass = O/F kg
        total mass = 1 + O/F kg

    Properties are returned per kg total reactants where appropriate.
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

        self.fuels = self._normalize_group(
            fuels,
            total_mass=1.0,
            role="fuel",
        )

        self.oxidizers = self._normalize_group(
            oxidizers,
            total_mass=self.mixture_ratio,
            role="oxidizer",
        )

        if not self.fuels:
            raise ValueError("At least one fuel is required.")

        if self.mixture_ratio > 0.0 and not self.oxidizers:
            raise ValueError("At least one oxidizer is required when mixture_ratio > 0.")

        self.entries = [*self.fuels, *self.oxidizers]
        self.total_mass = sum(entry.mass for entry in self.entries)

        if self.total_mass <= 0.0:
            raise ValueError("Total reactant mass must be positive.")

    @staticmethod
    def _normalize_group(
        propellants: Iterable[Propellant | tuple[Propellant, float]],
        total_mass: float,
        role: str,
    ) -> list[Reactant]:
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
            Reactant(
                propellant=propellant,
                mass=total_mass * weight / weight_sum,
                role=role,
            )
            for propellant, weight in parsed
        ]

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
        """
        Mean reactant formula molar mass [kg/mol].
        """
        return self.total_mass / self.total_moles

    @property
    def molecular_weight_kg_per_kmol(self) -> float:
        """
        Mean reactant formula molecular weight [kg/kmol].
        Numerically equal to g/mol.
        """
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

    @property
    def oxidizer_mass_fractions(self) -> dict[str, float]:
        total = self.oxidizer_mass

        if total <= 0.0:
            return {}

        return {
            entry.cea_name: entry.mass / total
            for entry in self.oxidizers
        }

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
            "reactant_internal_energy": self.reactant_internal_energy,
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
            ("Reactant MW [kg/mol]", f"{self.molecular_weight:.8g}"),
            ("Reactant MW [kg/kmol]", f"{self.molecular_weight_kg_per_kmol:.8g}"),
            ("Reactant enthalpy [J/kg]", f"{self.reactant_enthalpy:.8e}"),
            ("Reactant internal energy [J/kg]", f"{self.reactant_internal_energy:.8e}"),
            ("Mass fractions", self.mass_fractions),
            ("Mole fractions", self.mole_fractions),
            ("Element moles per kg", self.element_moles_per_kg),
        ]

        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)

