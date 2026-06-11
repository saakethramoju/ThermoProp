from __future__ import annotations

import numpy as np

from ProductSet import ProductSet


P_REF = 100000.0


class EquilibriumState:
    """
    Gas-only equilibrium state container.

    Mole units are mol species per kg reactant mixture.
    """

    def __init__(
        self,
        products: ProductSet,
        temperature: float,
        pressure: float,
        moles: np.ndarray | None = None,
    ):
        self.products = products
        self.temperature = float(temperature)
        self.pressure = float(pressure)

        if self.pressure <= 0.0:
            raise ValueError("pressure must be positive.")

        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        if moles is None:
            moles = self.initial_moles()

        self.moles = np.asarray(moles, dtype=float)

        if self.moles.shape != (self.products.count,):
            raise ValueError(
                f"moles must have shape ({self.products.count},), "
                f"got {self.moles.shape}."
            )

        if np.any(self.moles < 0.0):
            raise ValueError("moles must be nonnegative.")

    def initial_moles(self) -> np.ndarray:
        """
        Simple CEA-like first guess.

        Distributes total atom moles across product species. This is only an
        initial numerical guess; the TP/HP solver will update it.
        """
        total_atom_moles = float(np.sum(self.products.element_vector))
        total_species_guess = max(total_atom_moles / 2.0, 1e-30)

        return np.full(
            self.products.count,
            total_species_guess / self.products.count,
            dtype=float,
        )

    @property
    def names(self) -> list[str]:
        return self.products.names

    @property
    def elements(self) -> list[str]:
        return self.products.elements

    @property
    def A(self) -> np.ndarray:
        return self.products.element_matrix

    @property
    def b(self) -> np.ndarray:
        return self.products.element_vector

    @property
    def log_moles(self) -> np.ndarray:
        return np.log(np.maximum(self.moles, 1e-300))

    @log_moles.setter
    def log_moles(self, values: np.ndarray):
        values = np.asarray(values, dtype=float)

        if values.shape != self.moles.shape:
            raise ValueError(
                f"log_moles must have shape {self.moles.shape}, "
                f"got {values.shape}."
            )

        self.moles = np.exp(np.clip(values, -700.0, 700.0))

    @property
    def total_moles(self) -> float:
        return float(np.sum(self.moles))

    @property
    def mole_fractions(self) -> np.ndarray:
        ntot = self.total_moles

        if ntot <= 0.0:
            raise RuntimeError("Cannot compute mole fractions with zero total moles.")

        return self.moles / ntot

    @property
    def mole_fraction_dict(self) -> dict[str, float]:
        return {
            name: float(x)
            for name, x in zip(self.names, self.mole_fractions)
        }

    @property
    def nonzero_mole_fraction_dict(self) -> dict[str, float]:
        return {
            name: float(x)
            for name, x in zip(self.names, self.mole_fractions)
            if x > 1e-12
        }

    @property
    def element_moles(self) -> np.ndarray:
        return self.A @ self.moles

    @property
    def element_error(self) -> np.ndarray:
        return self.element_moles - self.b

    @property
    def max_element_error(self) -> float:
        return float(np.max(np.abs(self.element_error)))

    @property
    def element_relative_error(self) -> np.ndarray:
        scale = np.maximum(np.abs(self.b), 1e-30)
        return self.element_error / scale

    @property
    def max_element_relative_error(self) -> float:
        return float(np.max(np.abs(self.element_relative_error)))

    @property
    def standard_gibbs_over_RT(self) -> np.ndarray:
        return self.products.standard_gibbs_over_RT

    @property
    def chemical_potentials_over_RT(self) -> np.ndarray:
        """
        Ideal-gas chemical potential divided by RT.

            mu_j / RT = g0_j / RT + ln(x_j) + ln(P / P_ref)
        """
        return (
            self.standard_gibbs_over_RT
            + np.log(np.maximum(self.mole_fractions, 1e-300))
            + np.log(self.pressure / P_REF)
        )

    @property
    def gibbs_over_RT(self) -> float:
        """
        Dimensionless mixture Gibbs objective divided by RT.

        Since moles are mol/kg, this is scaled as mol/kg, but it is still the
        correct dimensionless objective form for minimization.
        """
        return float(np.sum(self.moles * self.chemical_potentials_over_RT))

    @property
    def enthalpy(self) -> float:
        """
        Mixture enthalpy [J/kg reactant mixture].
        """
        h_molar_kmol = self.products.standard_enthalpies_molar
        h_molar_mol = h_molar_kmol / 1000.0

        return float(np.sum(self.moles * h_molar_mol))

    @property
    def cp(self) -> float:
        """
        Frozen mixture Cp [J/kg-K reactant mixture].
        """
        cp_molar_kmol = self.products.standard_cps_molar
        cp_molar_mol = cp_molar_kmol / 1000.0

        return float(np.sum(self.moles * cp_molar_mol))

    @property
    def molecular_weight(self) -> float:
        """
        Product mixture molar mass [kg/mol].
        """
        mass = float(np.sum(self.moles * self.products.molar_masses))
        ntot = self.total_moles

        if ntot <= 0.0:
            raise RuntimeError("Cannot compute molecular weight with zero total moles.")

        return mass / ntot

    @property
    def gas_constant(self) -> float:
        """
        Product mixture gas constant [J/kg-K].
        """
        return 8.31446261815324 / self.molecular_weight

    @property
    def density(self) -> float:
        """
        Ideal-gas product density [kg/m^3].
        """
        return self.pressure / (self.gas_constant * self.temperature)

    def update_moles(self, moles: np.ndarray):
        moles = np.asarray(moles, dtype=float)

        if moles.shape != self.moles.shape:
            raise ValueError(
                f"moles must have shape {self.moles.shape}, got {moles.shape}."
            )

        if np.any(moles < 0.0):
            raise ValueError("moles must be nonnegative.")

        self.moles = moles

    def copy(self) -> "EquilibriumState":
        return EquilibriumState(
            products=self.products,
            temperature=self.temperature,
            pressure=self.pressure,
            moles=self.moles.copy(),
        )

    def as_dict(self, trace: float = 1e-12) -> dict:
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "total_moles": self.total_moles,
            "molecular_weight": self.molecular_weight,
            "gas_constant": self.gas_constant,
            "density": self.density,
            "enthalpy": self.enthalpy,
            "cp": self.cp,
            "gibbs_over_RT": self.gibbs_over_RT,
            "element_error": self.element_error,
            "max_element_error": self.max_element_error,
            "max_element_relative_error": self.max_element_relative_error,
            "mole_fractions": {
                name: float(x)
                for name, x in zip(self.names, self.mole_fractions)
                if x > trace
            },
        }

    def __str__(self) -> str:
        lines = [
            "EquilibriumState",
            f"Temperature [K]       : {self.temperature:.6g}",
            f"Pressure [Pa]         : {self.pressure:.6e}",
            f"Total moles [mol/kg]  : {self.total_moles:.8e}",
            f"MW [kg/mol]           : {self.molecular_weight:.8e}",
            f"Gas constant [J/kg-K] : {self.gas_constant:.8e}",
            f"Density [kg/m^3]      : {self.density:.8e}",
            f"Enthalpy [J/kg]       : {self.enthalpy:.8e}",
            f"Cp frozen [J/kg-K]    : {self.cp:.8e}",
            f"G/RT                  : {self.gibbs_over_RT:.8e}",
            f"Max element error     : {self.max_element_error:.8e}",
            "",
            "Mole fractions:",
        ]

        for name, x in sorted(
            zip(self.names, self.mole_fractions),
            key=lambda item: item[1],
            reverse=True,
        ):
            if x > 1e-12:
                lines.append(f"  {name:<24s} {x:.8e}")

        return "\n".join(lines)