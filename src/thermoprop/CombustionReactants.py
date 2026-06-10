from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Propellant import Propellant


class CombustionReactants:
    """
    Reactant-side bookkeeping for a fuel/oxidizer propellant pair.
    """

    def __init__(
        self,
        fuel: Propellant,
        oxidizer: Propellant,
        mixture_ratio: float,
    ):
        if mixture_ratio <= 0.0:
            raise ValueError(
                f"mixture_ratio must be greater than zero. Got {mixture_ratio}."
            )

        self.fuel = fuel
        self.oxidizer = oxidizer
        self.mixture_ratio = float(mixture_ratio)

        self._fuel_mass_fraction = 1.0 / (1.0 + self.mixture_ratio)
        self._oxidizer_mass_fraction = self.mixture_ratio / (1.0 + self.mixture_ratio)

        self._validate_propellant("fuel", self.fuel)
        self._validate_propellant("oxidizer", self.oxidizer)

    @staticmethod
    def _validate_propellant(role: str, propellant: Propellant) -> None:
        required = (
            "propellant",
            "cea_reactant",
            "elemental_composition",
            "cea_molar_mass",
            "temperature",
            "reference_temperature",
            "specific_heat_cp",
            "heat_of_formation",
        )

        for name in required:
            if not hasattr(propellant, name):
                raise TypeError(
                    f"{role} must be a Propellant-like object with property "
                    f"{name!r}."
                )

        if propellant.cea_reactant is None:
            raise ValueError(f"{role} propellant does not have a CEA reactant mapping.")

        if propellant.elemental_composition is None:
            raise ValueError(f"{role} propellant does not have elemental composition data.")

        if propellant.cea_molar_mass is None or propellant.cea_molar_mass <= 0.0:
            raise ValueError(f"{role} propellant does not have a valid CEA molar mass.")

        if propellant.reference_temperature is None:
            raise ValueError(f"{role} propellant does not have a CEA reference temperature.")

        if propellant.specific_heat_cp is None:
            raise ValueError(f"{role} propellant does not have specific_heat_cp data.")

        if propellant.heat_of_formation is None:
            raise ValueError(f"{role} propellant does not have heat_of_formation data.")

    @property
    def fuel_mass_fraction(self) -> float:
        return self._fuel_mass_fraction

    @property
    def oxidizer_mass_fraction(self) -> float:
        return self._oxidizer_mass_fraction

    @property
    def mass_fractions(self) -> dict[str, float]:
        return {
            "fuel": self.fuel_mass_fraction,
            "oxidizer": self.oxidizer_mass_fraction,
        }

    @property
    def reactants(self) -> dict[str, Propellant]:
        return {
            "fuel": self.fuel,
            "oxidizer": self.oxidizer,
        }

    @property
    def cea_reactants(self) -> dict[str, str | None]:
        return {
            "fuel": self.fuel.cea_reactant,
            "oxidizer": self.oxidizer.cea_reactant,
        }

    @property
    def reactant_temperatures(self) -> dict[str, float]:
        return {
            "fuel": self.fuel.temperature,
            "oxidizer": self.oxidizer.temperature,
        }

    @staticmethod
    def _reactant_sensible_enthalpy(propellant: Propellant) -> float:
        return propellant.specific_heat_cp * (
            propellant.temperature - propellant.reference_temperature
        )

    @staticmethod
    def _reactant_enthalpy(propellant: Propellant) -> float:
        return (
            propellant.heat_of_formation
            + CombustionReactants._reactant_sensible_enthalpy(propellant)
        )

    @property
    def fuel_heat_of_formation(self) -> float:
        return self.fuel.heat_of_formation

    @property
    def oxidizer_heat_of_formation(self) -> float:
        return self.oxidizer.heat_of_formation

    @property
    def fuel_sensible_enthalpy(self) -> float:
        return self._reactant_sensible_enthalpy(self.fuel)

    @property
    def oxidizer_sensible_enthalpy(self) -> float:
        return self._reactant_sensible_enthalpy(self.oxidizer)

    @property
    def fuel_enthalpy(self) -> float:
        return self._reactant_enthalpy(self.fuel)

    @property
    def oxidizer_enthalpy(self) -> float:
        return self._reactant_enthalpy(self.oxidizer)

    @property
    def enthalpies(self) -> dict[str, float]:
        return {
            "fuel": self.fuel_enthalpy,
            "oxidizer": self.oxidizer_enthalpy,
        }

    @property
    def elemental_composition(self) -> dict[str, float]:
        """
        Element totals for one kg of total incoming reactants.

        Returned values are kmol atoms per kg of total reactants.
        """
        totals: dict[str, float] = {}

        for propellant, mass_fraction in (
            (self.fuel, self.fuel_mass_fraction),
            (self.oxidizer, self.oxidizer_mass_fraction),
        ):
            kmol_reactant_per_kg = (
                mass_fraction / propellant.cea_molar_mass / 1000.0
            )

            for element, count in propellant.elemental_composition.items():
                totals[element] = totals.get(element, 0.0) + (
                    kmol_reactant_per_kg * count
                )

        return dict(sorted(totals.items()))

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        return self.elemental_composition

    @property
    def elements(self) -> set[str]:
        return set(self.elemental_composition)

    @property
    def oxidizer_to_fuel_ratio(self) -> float:
        return self.mixture_ratio

    @property
    def fuel_to_oxidizer_ratio(self) -> float:
        return 1.0 / self.mixture_ratio

    def _safe(self, value, fmt=".3e"):
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def __str__(self) -> str:
        rows = [
            ("Fuel", self.fuel.propellant),
            ("Oxidizer", self.oxidizer.propellant),
            ("Fuel CEA reactant", self.fuel.cea_reactant),
            ("Oxidizer CEA reactant", self.oxidizer.cea_reactant),
            ("Mixture ratio O/F", self._safe(self.mixture_ratio, ".6f")),
            ("Fuel mass fraction", self._safe(self.fuel_mass_fraction, ".6f")),
            ("Oxidizer mass fraction", self._safe(self.oxidizer_mass_fraction, ".6f")),
            ("Fuel temperature [K]", self._safe(self.fuel.temperature, ".2f")),
            ("Oxidizer temperature [K]", self._safe(self.oxidizer.temperature, ".2f")),
            ("Fuel Hf [J/kg]", self._safe(self.fuel_heat_of_formation, ".3e")),
            ("Oxidizer Hf [J/kg]", self._safe(self.oxidizer_heat_of_formation, ".3e")),
            ("Fuel sensible h [J/kg]", self._safe(self.fuel_sensible_enthalpy, ".3e")),
            ("Oxidizer sensible h [J/kg]", self._safe(self.oxidizer_sensible_enthalpy, ".3e")),
            ("Fuel h [J/kg]", self._safe(self.fuel_enthalpy, ".3e")),
            ("Oxidizer h [J/kg]", self._safe(self.oxidizer_enthalpy, ".3e")),
        ]

        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fuel={self.fuel.propellant!r}, "
            f"oxidizer={self.oxidizer.propellant!r}, "
            f"mixture_ratio={self.mixture_ratio:.6g})"
        )