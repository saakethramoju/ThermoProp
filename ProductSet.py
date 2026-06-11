from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CEADatabase import CEA
from ReactantMixture import ReactantMixture

DEFAULT_DEBUG_PRODUCTS = [
    "H2",
    "H",
    "O2",
    "O",
    "OH",
    "H2O",
    "CO",
    "CO2",
    "CH4",
    "CH3",
    "CH2",
    "CH",
    "C",

    # nitrogen products
    "N2",
    "N",
    "NO",
    "NO2",
    "N2O",
    "NH",
    "NH2",
    "NH3",
    "CN",
    "HCN",
]


@dataclass(frozen=True)
class ProductSpecies:
    name: str
    molar_mass: float
    elemental_composition: dict[str, float]
    is_gas: bool
    is_condensed: bool
    has_thermo: bool


class ProductSet:
    """
    CEA product species selector.

    Starts with gas species only. Condensed species can be added later once the
    gas-only TP/HP Newton solver is stable.
    """

    def __init__(
        self,
        reactants: ReactantMixture,
        temperature: float,
        candidates: list[str] | None = None,
        include_all_valid_gases: bool = False,
    ):
        self.reactants = reactants
        self.temperature = float(temperature)
        self.allowed_elements = set(reactants.element_moles_per_kg)

        if candidates is None and not include_all_valid_gases:
            candidates = DEFAULT_DEBUG_PRODUCTS

        if include_all_valid_gases:
            candidates = list(CEA.gas_species)

        self.species = self._build_species(candidates)

        if not self.species:
            raise RuntimeError("No valid product species were found.")

    def _build_species(self, candidates: list[str] | None) -> list[ProductSpecies]:
        products: list[ProductSpecies] = []
        seen: set[str] = set()

        if candidates is None:
            candidates = list(CEA.gas_species)

        for candidate in candidates:
            if not CEA.has_species(candidate):
                continue

            name = CEA.resolve_name(candidate)

            if name in seen:
                continue

            if CEA.is_reactant(name):
                continue

            if not CEA.is_gas(name):
                continue

            if not CEA.has_thermo(name):
                continue

            comp = CEA.elemental_composition(name)

            if not comp:
                continue

            if not set(comp).issubset(self.allowed_elements):
                continue

            try:
                CEA.nasa9_coefficients(name, self.temperature)
            except Exception:
                continue

            products.append(
                ProductSpecies(
                    name=name,
                    molar_mass=CEA.molar_mass(name),
                    elemental_composition=dict(comp),
                    is_gas=CEA.is_gas(name),
                    is_condensed=CEA.is_condensed(name),
                    has_thermo=CEA.has_thermo(name),
                )
            )

            seen.add(name)

        return products

    @property
    def names(self) -> list[str]:
        return [sp.name for sp in self.species]

    @property
    def count(self) -> int:
        return len(self.species)

    @property
    def elements(self) -> list[str]:
        return sorted(self.allowed_elements)

    @property
    def molar_masses(self) -> np.ndarray:
        return np.array([sp.molar_mass for sp in self.species], dtype=float)

    @property
    def element_matrix(self) -> np.ndarray:
        elements = self.elements

        return np.array(
            [
                [
                    sp.elemental_composition.get(element, 0.0)
                    for sp in self.species
                ]
                for element in elements
            ],
            dtype=float,
        )

    @property
    def element_vector(self) -> np.ndarray:
        _, b = self.reactants.element_vector(self.elements)
        return b

    @property
    def standard_gibbs_over_RT(self) -> np.ndarray:
        values = []

        for sp in self.species:
            cp, h, s = CEA.thermo_molar(sp.name, self.temperature)
            values.append(
                h / (CEA._RU_KMOL * self.temperature)
                - s / CEA._RU_KMOL
            )

        return np.array(values, dtype=float)

    @property
    def standard_enthalpies_molar(self) -> np.ndarray:
        return np.array(
            [
                CEA.thermo_molar(sp.name, self.temperature)[1]
                for sp in self.species
            ],
            dtype=float,
        )

    @property
    def standard_entropies_molar(self) -> np.ndarray:
        return np.array(
            [
                CEA.thermo_molar(sp.name, self.temperature)[2]
                for sp in self.species
            ],
            dtype=float,
        )

    @property
    def standard_cps_molar(self) -> np.ndarray:
        return np.array(
            [
                CEA.thermo_molar(sp.name, self.temperature)[0]
                for sp in self.species
            ],
            dtype=float,
        )

    def has_species(self, name: str) -> bool:
        return name in self.names

    def composition_of(self, name: str) -> dict[str, float]:
        for sp in self.species:
            if sp.name == name:
                return dict(sp.elemental_composition)

        raise KeyError(name)

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "allowed_elements": sorted(self.allowed_elements),
            "species": [
                {
                    "name": sp.name,
                    "molar_mass": sp.molar_mass,
                    "elemental_composition": sp.elemental_composition,
                    "is_gas": sp.is_gas,
                    "is_condensed": sp.is_condensed,
                    "has_thermo": sp.has_thermo,
                }
                for sp in self.species
            ],
        }

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.species)

    def __contains__(self, name: str) -> bool:
        return self.has_species(name)

    def __str__(self) -> str:
        lines = [
            f"ProductSet",
            f"Temperature [K] : {self.temperature:.6g}",
            f"Elements        : {self.elements}",
            f"Species count   : {self.count}",
            "",
            "Species:",
        ]

        for sp in self.species:
            lines.append(f"  {sp.name:<24s} {sp.elemental_composition}")

        return "\n".join(lines)