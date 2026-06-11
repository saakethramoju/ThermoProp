from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from CEADatabase import CEA
from .ReactantMixture import ReactantMixture
from .CombustionGas import CombustionGas


P_REF = 100000.0
RU = 8.31446261815324
RU_KMOL = 8314.46261815324


DEFAULT_DEBUG_PRODUCTS = [
    "H2", "H",
    "O2", "O", "OH", "H2O",
    "CO", "CO2",
    "CH4", "CH3", "CH2", "CH", "C",
    "N2", "N", "NO", "NO2", "N2O",
    "NH", "NH2", "NH3", "CN", "HCN",
]


@dataclass(frozen=True)
class EquilibriumResult:
    success: bool
    message: str
    mode: str
    iterations: int
    max_element_error: float
    max_mole_correction: float
    max_total_mole_correction: float
    enthalpy_error: float | None = None
    temperature_correction: float | None = None


class _CombustionGasReactants:
    def __init__(self, gas: CombustionGas):
        self.gas = gas
        self.total_mass = 1.0

        mole_fractions = gas.mole_fractions

        if not mole_fractions:
            raise ValueError("CombustionGas composition cannot be empty.")

        total_x = sum(float(x) for x in mole_fractions.values())

        if total_x <= 0.0:
            raise ValueError("CombustionGas mole fractions must sum to a positive value.")

        self._mole_fractions = {
            name: float(x) / total_x
            for name, x in mole_fractions.items()
            if float(x) > 0.0
        }

        self._molar_masses = {
            name: CEA.molar_mass(name)
            for name in self._mole_fractions
        }

        self._molecular_weight = sum(
            self._mole_fractions[name] * self._molar_masses[name]
            for name in self._mole_fractions
        )

        if self._molecular_weight <= 0.0:
            raise ValueError("CombustionGas molecular weight is invalid.")

        self._total_moles = 1.0 / self._molecular_weight

    @property
    def element_moles(self) -> dict[str, float]:
        element_moles: dict[str, float] = {}

        for name, x in self._mole_fractions.items():
            n = x * self._total_moles
            comp = CEA.elemental_composition(name)

            for element, count in comp.items():
                element_moles[element] = (
                    element_moles.get(element, 0.0)
                    + n * float(count)
                )

        return dict(sorted(element_moles.items()))

    @property
    def element_moles_per_kg(self) -> dict[str, float]:
        return self.element_moles

    @property
    def reactant_enthalpy(self) -> float:
        return float(self.gas.enthalpy)

    @property
    def reactant_internal_energy(self) -> float:
        return float(self.gas.internal_energy)

    @property
    def total_moles(self) -> float:
        return self._total_moles

    @property
    def total_kmoles(self) -> float:
        return self._total_moles / 1000.0

    @property
    def molecular_weight(self) -> float:
        return self._molecular_weight

    @property
    def molecular_weight_kg_per_kmol(self) -> float:
        return self._molecular_weight * 1000.0

    def element_vector(self, elements: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        if elements is None:
            elements = sorted(self.element_moles_per_kg)

        b = np.array(
            [self.element_moles_per_kg.get(element, 0.0) for element in elements],
            dtype=float,
        )

        return elements, b


class Equilibrium:
    _BACKEND_NAME = "ThermoProp CEA Equilibrium"

    _UNSUPPORTED_PROPERTIES = {
        "quality",
        "surface_tension",
        "vapor_pressure",
        "saturation_pressure",
        "saturation_temperature",
        "heat_of_vaporization",
        "critical_pressure",
        "critical_temperature",
        "critical_density",
        "freezing_temperature",
        "boiling_temperature",
    }

    _FLASH_INPUTS = {
        frozenset(("pressure", "temperature")),
        frozenset(("pressure", "enthalpy")),
    }

    def __init__(
        self,
        reactants: ReactantMixture | CombustionGas,
        *,
        mode: str = "hp",
        temperature: float | None = None,
        pressure: float | None = None,
        guess_temperature: float = 3500.0,
        candidates: list[str] | None = None,
        include_all_valid_gases: bool = True,
        verbose: bool = False,
        element_tol: float = 1e-8,
        enthalpy_tol: float = 1e-3,
        correction_tol: float = 1e-8,
        max_iterations: int = 200,
        trace_moles: float = 1e-300,
        min_temperature: float = 200.0,
        max_temperature: float = 20000.0,
        combustion_gas_trace: float = 1e-8,
        combustion_gas_max_species: int | None = 25,
    ):
        self._mode = str(mode).lower()
        self._pressure = None if pressure is None else float(pressure)
        self._temperature_input = None if temperature is None else float(temperature)
        self._guess_temperature = float(guess_temperature)

        self._input = reactants
        self._reactants = self._resolve_reactants(reactants)

        self._candidates = candidates
        self._include_all_valid_gases = bool(include_all_valid_gases)
        self._verbose = bool(verbose)

        self._element_tol = float(element_tol)
        self._enthalpy_tol = float(enthalpy_tol)
        self._correction_tol = float(correction_tol)
        self._max_iterations = int(max_iterations)
        self._trace_moles = float(trace_moles)
        self._min_temperature = float(min_temperature)
        self._max_temperature = float(max_temperature)

        self._combustion_gas_trace = float(combustion_gas_trace)
        self._combustion_gas_max_species = combustion_gas_max_species

        self._species: list[str] = []
        self._elements: list[str] = []
        self._molar_masses: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._moles: np.ndarray | None = None

        self._result: EquilibriumResult | None = None
        self._gas_cache: CombustionGas | None = None

        self._solve()

    @staticmethod
    def _resolve_reactants(reactants):
        if isinstance(reactants, ReactantMixture):
            return reactants

        if isinstance(reactants, CombustionGas):
            return _CombustionGasReactants(reactants)

        if reactants.__class__.__name__ == "ReactantMixture":
            return reactants

        if reactants.__class__.__name__ == "CombustionGas":
            return _CombustionGasReactants(reactants)

        raise TypeError("reactants must be a ReactantMixture or CombustionGas.")

    def _validate_inputs(self) -> None:
        if self._pressure is None:
            raise ValueError("Equilibrium requires pressure.")

        if self._pressure <= 0.0:
            raise ValueError("pressure must be positive.")

        if self._mode == "tp":
            if self._temperature_input is None:
                raise ValueError("TP equilibrium requires temperature and pressure.")
            if self._temperature_input <= 0.0:
                raise ValueError("temperature must be positive.")

        elif self._mode == "hp":
            if self._guess_temperature <= 0.0:
                raise ValueError("guess_temperature must be positive.")

        else:
            raise ValueError("mode must be 'tp' or 'hp'.")

    def _build_product_set(self, temperature: float) -> None:
        allowed_elements = set(self._reactants.element_moles_per_kg)

        if self._include_all_valid_gases:
            candidates = list(CEA.gas_species)
        elif self._candidates is not None:
            candidates = list(self._candidates)
        else:
            candidates = list(DEFAULT_DEBUG_PRODUCTS)

        species = []
        seen = set()

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

            if not set(comp).issubset(allowed_elements):
                continue

            try:
                CEA.nasa9_coefficients(name, temperature)
            except Exception:
                continue

            species.append(name)
            seen.add(name)

        if not species:
            raise RuntimeError("No valid gas product species were found.")

        self._species = species
        self._elements = sorted(allowed_elements)
        self._molar_masses = np.array(
            [CEA.molar_mass(name) for name in self._species],
            dtype=float,
        )

        self._A = np.array(
            [
                [
                    CEA.elemental_composition(name).get(element, 0.0)
                    for name in self._species
                ]
                for element in self._elements
            ],
            dtype=float,
        )

        _, self._b = self._reactants.element_vector(self._elements)

    def _thermo_arrays(self, temperature: float):
        cp = []
        h = []
        s = []
        g0_RT = []

        for name in self._species:
            cp_i, h_i, s_i = CEA.thermo_molar(name, temperature)
            cp.append(cp_i)
            h.append(h_i)
            s.append(s_i)
            g0_RT.append(h_i / (RU_KMOL * temperature) - s_i / RU_KMOL)

        return (
            np.array(cp, dtype=float),
            np.array(h, dtype=float),
            np.array(s, dtype=float),
            np.array(g0_RT, dtype=float),
        )

    def _initial_moles(self) -> np.ndarray:
        total_atom_moles = float(np.sum(self._b))
        total_species_guess = max(total_atom_moles / 2.0, 1e-30)

        return np.full(
            len(self._species),
            total_species_guess / len(self._species),
            dtype=float,
        )

    def _solve(self) -> None:
        self._validate_inputs()
        self._gas_cache = None

        product_temperature = (
            self._temperature_input
            if self._mode == "tp"
            else self._guess_temperature
        )

        self._temperature = float(product_temperature)
        self._build_product_set(self._temperature)
        self._moles = self._initial_moles()

        if self._mode == "tp":
            self._result = self._solve_tp()
        else:
            self._result = self._solve_hp()

        if not self._result.success:
            raise RuntimeError(f"Equilibrium solve failed: {self._result.message}")

    def _solve_tp(self) -> EquilibriumResult:
        A = self._A
        b = self._b
        pressure = self._pressure
        n = np.maximum(self._moles.astype(float), self._trace_moles)

        _, _, _, g0_RT = self._thermo_arrays(self._temperature)
        lnP = np.log(pressure / P_REF)

        ne, ns = A.shape
        max_mole_correction = np.inf
        max_total_mole_correction = np.inf

        for iteration in range(1, self._max_iterations + 1):
            ntot = float(np.sum(n))

            if ntot <= 0.0:
                raise RuntimeError("Total moles became nonpositive during TP solve.")

            x = n / ntot
            mu_RT = g0_RT + np.log(np.maximum(x, self._trace_moles)) + lnP

            element_current = A @ n

            K = A @ (n[:, None] * A.T)
            c = element_current

            matrix = np.zeros((ne + 1, ne + 1), dtype=float)
            rhs = np.zeros(ne + 1, dtype=float)

            matrix[:ne, :ne] = K
            matrix[:ne, ne] = c
            matrix[ne, :ne] = c
            matrix[ne, ne] = 0.0

            rhs[:ne] = b - element_current + A @ (n * mu_RT)
            rhs[ne] = float(np.sum(n * mu_RT))

            try:
                correction = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

            element_potentials = correction[:ne]
            dln_total_moles = float(correction[ne])

            dln_moles = -mu_RT + A.T @ element_potentials + dln_total_moles

            max_mole_correction = float(np.max(np.abs(dln_moles)))
            max_total_mole_correction = abs(dln_total_moles)

            alpha = 1.0

            if max_mole_correction > 2.0:
                alpha = min(alpha, 2.0 / max_mole_correction)

            if max_total_mole_correction > 0.4:
                alpha = min(alpha, 0.4 / max_total_mole_correction)

            n = n * np.exp(np.clip(alpha * dln_moles, -700.0, 700.0))
            n = np.maximum(n, self._trace_moles)

            self._moles = n
            max_element_error = self.max_element_error

            if self._verbose:
                print(
                    f"{iteration:4d} "
                    f"alpha={alpha:.3e} "
                    f"max|element error|={max_element_error:.3e} "
                    f"max|dln n_j|={max_mole_correction:.3e} "
                    f"|dln n|={max_total_mole_correction:.3e}"
                )

            if (
                max_element_error < self._element_tol
                and max_mole_correction < self._correction_tol
                and max_total_mole_correction < self._correction_tol
            ):
                return EquilibriumResult(
                    success=True,
                    message="converged",
                    mode="tp",
                    iterations=iteration,
                    max_element_error=max_element_error,
                    max_mole_correction=max_mole_correction,
                    max_total_mole_correction=max_total_mole_correction,
                )

        return EquilibriumResult(
            success=False,
            message="maximum iterations exceeded",
            mode="tp",
            iterations=self._max_iterations,
            max_element_error=self.max_element_error,
            max_mole_correction=max_mole_correction,
            max_total_mole_correction=max_total_mole_correction,
        )

    def _solve_hp(self) -> EquilibriumResult:
        A = self._A
        b = self._b
        pressure = self._pressure
        target_enthalpy = self._reactants.reactant_enthalpy

        n = np.maximum(self._moles.astype(float), self._trace_moles)
        T = float(self._temperature)

        ne, ns = A.shape

        max_mole_correction = np.inf
        max_total_mole_correction = np.inf
        temperature_correction = np.inf
        enthalpy_error = np.inf

        for iteration in range(1, self._max_iterations + 1):
            T = float(np.clip(T, self._min_temperature, self._max_temperature))
            self._temperature = T

            cp_kmol, h_kmol, _, g0_RT = self._thermo_arrays(T)

            h_mol = h_kmol / 1000.0
            cp_mol = cp_kmol / 1000.0
            h_RT = h_kmol / (RU_KMOL * T)

            lnP = np.log(pressure / P_REF)
            ntot = float(np.sum(n))

            if ntot <= 0.0:
                raise RuntimeError("Total moles became nonpositive during HP solve.")

            x = n / ntot
            mu_RT = g0_RT + np.log(np.maximum(x, self._trace_moles)) + lnP

            element_current = A @ n
            mixture_enthalpy = float(np.sum(n * h_mol))
            enthalpy_error = mixture_enthalpy - target_enthalpy

            K = A @ (n[:, None] * A.T)
            c = element_current
            q = A @ (n * h_RT)

            h_element = A @ (n * h_mol)
            h_total = mixture_enthalpy
            h_temperature = float(
                np.sum(n * h_mol * h_RT)
                + np.sum(n * cp_mol * T)
            )

            matrix = np.zeros((ne + 2, ne + 2), dtype=float)
            rhs = np.zeros(ne + 2, dtype=float)

            matrix[:ne, :ne] = K
            matrix[:ne, ne] = c
            matrix[:ne, ne + 1] = q

            matrix[ne, :ne] = c
            matrix[ne, ne] = 0.0
            matrix[ne, ne + 1] = float(np.sum(n * h_RT))

            matrix[ne + 1, :ne] = h_element
            matrix[ne + 1, ne] = h_total
            matrix[ne + 1, ne + 1] = h_temperature

            rhs[:ne] = b - element_current + A @ (n * mu_RT)
            rhs[ne] = float(np.sum(n * mu_RT))
            rhs[ne + 1] = (
                target_enthalpy
                - mixture_enthalpy
                + float(np.sum(n * h_mol * mu_RT))
            )

            try:
                correction = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

            element_potentials = correction[:ne]
            dln_total_moles = float(correction[ne])
            dlnT = float(correction[ne + 1])

            dln_moles = (
                -mu_RT
                + A.T @ element_potentials
                + dln_total_moles
                + h_RT * dlnT
            )

            max_mole_correction = float(np.max(np.abs(dln_moles)))
            max_total_mole_correction = abs(dln_total_moles)
            temperature_correction = abs(dlnT)

            alpha = 1.0

            if max_mole_correction > 2.0:
                alpha = min(alpha, 2.0 / max_mole_correction)

            if max_total_mole_correction > 0.4:
                alpha = min(alpha, 0.4 / max_total_mole_correction)

            if temperature_correction > 0.2:
                alpha = min(alpha, 0.2 / temperature_correction)

            n = n * np.exp(np.clip(alpha * dln_moles, -700.0, 700.0))
            n = np.maximum(n, self._trace_moles)

            T = T * np.exp(np.clip(alpha * dlnT, -5.0, 5.0))
            T = float(np.clip(T, self._min_temperature, self._max_temperature))

            self._temperature = T
            self._moles = n

            max_element_error = self.max_element_error
            enthalpy_error = self.enthalpy - target_enthalpy

            if self._verbose:
                print(
                    f"{iteration:4d} "
                    f"alpha={alpha:.3e} "
                    f"T={T:.3f} "
                    f"max|element error|={max_element_error:.3e} "
                    f"h_error={enthalpy_error:.3e} "
                    f"max|dln n_j|={max_mole_correction:.3e} "
                    f"|dln n|={max_total_mole_correction:.3e} "
                    f"|dlnT|={temperature_correction:.3e}"
                )

            if (
                max_element_error < self._element_tol
                and abs(enthalpy_error) < self._enthalpy_tol
                and max_mole_correction < self._correction_tol
                and max_total_mole_correction < self._correction_tol
                and temperature_correction < self._correction_tol
            ):
                return EquilibriumResult(
                    success=True,
                    message="converged",
                    mode="hp",
                    iterations=iteration,
                    max_element_error=max_element_error,
                    max_mole_correction=max_mole_correction,
                    max_total_mole_correction=max_total_mole_correction,
                    enthalpy_error=enthalpy_error,
                    temperature_correction=temperature_correction,
                )

        return EquilibriumResult(
            success=False,
            message="maximum iterations exceeded",
            mode="hp",
            iterations=self._max_iterations,
            max_element_error=self.max_element_error,
            max_mole_correction=max_mole_correction,
            max_total_mole_correction=max_total_mole_correction,
            enthalpy_error=enthalpy_error,
            temperature_correction=temperature_correction,
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def backend(self) -> str:
        return self._BACKEND_NAME

    @property
    def name(self) -> str:
        return "Equilibrium combustion gas"

    @property
    def reactants(self):
        return self._reactants

    @property
    def input(self):
        return self._input

    @property
    def result(self) -> EquilibriumResult:
        return self._result

    @property
    def success(self) -> bool:
        return self._result.success

    @property
    def message(self) -> str:
        return self._result.message

    @property
    def iterations(self) -> int:
        return self._result.iterations

    @property
    def pressure(self) -> float:
        return self._pressure

    @pressure.setter
    def pressure(self, value: float):
        self._pressure = float(value)
        self._solve()

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if self._mode == "tp":
            self._temperature_input = float(value)
        else:
            self._guess_temperature = float(value)

        self._solve()

    @property
    def pressure_temperature(self) -> tuple[float, float]:
        return self.pressure, self.temperature

    @pressure_temperature.setter
    def pressure_temperature(self, values: tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("pressure_temperature must be set with (pressure, temperature).")

        self._mode = "tp"
        self._pressure = float(values[0])
        self._temperature_input = float(values[1])
        self._solve()

    @property
    def TP(self) -> tuple[float, float]:
        return self.temperature, self.pressure

    @TP.setter
    def TP(self, values: tuple[float, float]):
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("TP must be set with (temperature, pressure).")

        self._mode = "tp"
        self._temperature_input = float(values[0])
        self._pressure = float(values[1])
        self._solve()

    @property
    def HP(self) -> tuple[float, float]:
        return self.enthalpy, self.pressure

    @HP.setter
    def HP(self, values: tuple[float, float]):
        raise ValueError(
            "Equilibrium HP is fixed by reactant enthalpy and pressure. "
            "Change the ReactantMixture/CombustionGas input or pressure instead."
        )

    @property
    def species(self) -> list[str]:
        return list(self._species)

    @property
    def elements(self) -> list[str]:
        return list(self._elements)

    @property
    def element_matrix(self) -> np.ndarray:
        return self._A.copy()

    @property
    def element_vector(self) -> np.ndarray:
        return self._b.copy()

    @property
    def moles(self) -> dict[str, float]:
        return {
            name: float(n)
            for name, n in zip(self._species, self._moles)
        }

    @property
    def total_moles(self) -> float:
        return float(np.sum(self._moles))

    @property
    def mole_fractions(self) -> dict[str, float]:
        ntot = self.total_moles

        if ntot <= 0.0:
            raise RuntimeError("Cannot compute mole fractions with zero total moles.")

        return {
            name: float(n / ntot)
            for name, n in zip(self._species, self._moles)
        }

    @property
    def mass_fractions(self) -> dict[str, float]:
        x = np.array(list(self.mole_fractions.values()), dtype=float)
        mass = x * self._molar_masses
        mass = mass / np.sum(mass)

        return {
            name: float(w)
            for name, w in zip(self._species, mass)
        }

    @property
    def element_moles(self) -> np.ndarray:
        return self._A @ self._moles

    @property
    def element_error(self) -> np.ndarray:
        return self.element_moles - self._b

    @property
    def max_element_error(self) -> float:
        return float(np.max(np.abs(self.element_error)))

    @property
    def max_element_relative_error(self) -> float:
        scale = np.maximum(np.abs(self._b), 1e-30)
        return float(np.max(np.abs(self.element_error / scale)))

    @property
    def phase(self) -> str:
        return "Equilibrium Gas"

    @property
    def is_mixture(self) -> bool:
        return True

    @property
    def compressibility(self) -> float:
        return 1.0

    @property
    def molecular_weight(self) -> float:
        return self.molar_mass * 1000.0

    @property
    def molar_mass(self) -> float:
        mass = float(np.sum(self._moles * self._molar_masses))
        ntot = self.total_moles

        if ntot <= 0.0:
            raise RuntimeError("Cannot compute molar mass with zero total moles.")

        return mass / ntot

    @property
    def gas_constant(self) -> float:
        return RU / self.molar_mass

    @property
    def universal_gas_constant(self) -> float:
        return RU

    @property
    def density(self) -> float:
        return self.pressure / (self.gas_constant * self.temperature)

    @density.setter
    def density(self, value: float):
        self._pressure = float(value) * self.gas_constant * self.temperature
        self._solve()

    @property
    def specific_volume(self) -> float:
        return 1.0 / self.density

    @property
    def enthalpy(self) -> float:
        _, h_kmol, _, _ = self._thermo_arrays(self.temperature)
        h_mol = h_kmol / 1000.0
        return float(np.sum(self._moles * h_mol))

    @property
    def internal_energy(self) -> float:
        return self.enthalpy - self.gas_constant * self.temperature

    @property
    def cp_frozen(self) -> float:
        cp_kmol, _, _, _ = self._thermo_arrays(self.temperature)
        cp_mol = cp_kmol / 1000.0
        return float(np.sum(self._moles * cp_mol))

    @property
    def specific_heat_cp(self) -> float:
        return self.combustion_gas.specific_heat_cp

    @property
    def specific_heat_cv(self) -> float:
        return self.combustion_gas.specific_heat_cv

    @property
    def specific_heat(self) -> float:
        return self.specific_heat_cp

    @property
    def specific_heat_ratio(self) -> float:
        return self.combustion_gas.specific_heat_ratio

    @property
    def entropy(self) -> float:
        return self.combustion_gas.entropy

    @property
    def gibbs_energy(self) -> float:
        return self.combustion_gas.gibbs_energy

    @property
    def helmholtz_energy(self) -> float:
        return self.combustion_gas.helmholtz_energy

    @property
    def free_energy(self) -> float:
        return self.helmholtz_energy

    @property
    def chemical_potentials_over_RT(self) -> dict[str, float]:
        _, _, _, g0_RT = self._thermo_arrays(self.temperature)
        x = np.array(list(self.mole_fractions.values()), dtype=float)

        mu_RT = (
            g0_RT
            + np.log(np.maximum(x, self._trace_moles))
            + np.log(self.pressure / P_REF)
        )

        return {
            name: float(mu)
            for name, mu in zip(self._species, mu_RT)
        }

    @property
    def gibbs_over_RT(self) -> float:
        mu = np.array(list(self.chemical_potentials_over_RT.values()), dtype=float)
        return float(np.sum(self._moles * mu))

    def combustion_gas_composition(
        self,
        trace: float | None = None,
        max_species: int | None = None,
    ) -> dict[str, float]:
        if trace is None:
            trace = self._combustion_gas_trace

        if max_species is None:
            max_species = self._combustion_gas_max_species

        items = [
            (name, float(x))
            for name, x in self.mole_fractions.items()
            if float(x) > float(trace)
        ]

        items.sort(key=lambda item: item[1], reverse=True)

        if max_species is not None:
            items = items[:int(max_species)]

        total = sum(x for _, x in items)

        if total <= 0.0:
            name, x = max(
                self.mole_fractions.items(),
                key=lambda item: item[1],
            )
            return {name: 1.0}

        return {
            name: x / total
            for name, x in items
        }

    @property
    def combustion_gas(self) -> CombustionGas:
        if self._gas_cache is None:
            self._gas_cache = CombustionGas(
                self.combustion_gas_composition(),
                basis="mole",
                pressure=self.pressure,
                temperature=self.temperature,
            )

        return self._gas_cache

    @property
    def gas(self) -> CombustionGas:
        return self.combustion_gas

    @property
    def dynamic_viscosity(self) -> float | None:
        try:
            return self.combustion_gas.dynamic_viscosity
        except Exception:
            return None

    @property
    def kinematic_viscosity(self) -> float | None:
        mu = self.dynamic_viscosity

        if mu is None:
            return None

        return mu / self.density

    @property
    def conductivity(self) -> float | None:
        try:
            return self.combustion_gas.conductivity
        except Exception:
            return None

    @property
    def thermal_conductivity(self) -> float | None:
        return self.conductivity

    @property
    def prandtl(self) -> float | None:
        try:
            return self.combustion_gas.prandtl
        except Exception:
            return None

    @property
    def speed_of_sound(self) -> float:
        return self.combustion_gas.speed_of_sound

    @property
    def thermal_expansion_coefficient(self) -> float:
        return 1.0 / self.temperature

    @property
    def isothermal_compressibility(self) -> float:
        return 1.0 / self.pressure

    def partial_derivative(self, of: str, with_respect_to: str, constant: str) -> float:
        return self.combustion_gas.partial_derivative(of, with_respect_to, constant)

    @property
    def dhdT_const_p(self) -> float:
        return self.partial_derivative("Hmass", "T", "P")

    @property
    def dhdp_const_T(self) -> float:
        return self.partial_derivative("Hmass", "P", "T")

    @property
    def drhodT_const_p(self) -> float:
        return self.partial_derivative("Dmass", "T", "P")

    @property
    def drhodp_const_T(self) -> float:
        return self.partial_derivative("Dmass", "P", "T")

    @property
    def dTdp_const_h(self) -> float:
        return self.partial_derivative("T", "P", "Hmass")

    @property
    def joule_thomson_coefficient(self) -> float:
        return self.dTdp_const_h

    @property
    def quality(self):
        raise NotImplementedError("Equilibrium gas does not support vapor quality.")

    @quality.setter
    def quality(self, value: float):
        raise ValueError("Equilibrium gas does not support vapor quality.")

    @property
    def surface_tension(self):
        raise NotImplementedError("Equilibrium gas does not support surface tension.")

    @property
    def vapor_pressure(self):
        raise NotImplementedError("Equilibrium gas does not support vapor pressure.")

    @property
    def saturation_pressure(self):
        raise NotImplementedError("Equilibrium gas does not support saturation pressure.")

    @property
    def saturation_temperature(self):
        raise NotImplementedError("Equilibrium gas does not support saturation temperature.")

    @property
    def heat_of_vaporization(self):
        raise NotImplementedError("Equilibrium gas does not support heat of vaporization.")

    @property
    def critical_pressure(self):
        raise NotImplementedError("Equilibrium gas does not support critical pressure.")

    @property
    def critical_temperature(self):
        raise NotImplementedError("Equilibrium gas does not support critical temperature.")

    @property
    def critical_density(self):
        raise NotImplementedError("Equilibrium gas does not support critical density.")

    @property
    def freezing_temperature(self):
        raise NotImplementedError("Equilibrium gas does not support freezing temperature.")

    @property
    def boiling_temperature(self):
        raise NotImplementedError("Equilibrium gas does not support boiling temperature.")

    @property
    def minimum_pressure(self) -> float:
        return 1e-30

    @property
    def maximum_pressure(self) -> float:
        return np.inf

    @property
    def minimum_temperature(self) -> float:
        return self.combustion_gas.minimum_temperature

    @property
    def maximum_temperature(self) -> float:
        return self.combustion_gas.maximum_temperature

    @property
    def max_mole_correction(self) -> float:
        return self._result.max_mole_correction

    @property
    def max_total_mole_correction(self) -> float:
        return self._result.max_total_mole_correction

    @property
    def enthalpy_error(self) -> float | None:
        return self._result.enthalpy_error

    @property
    def temperature_correction(self) -> float | None:
        return self._result.temperature_correction

    def as_dict(self, trace: float = 1e-8) -> dict:
        return {
            "mode": self.mode,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "density": self.density,
            "enthalpy": self.enthalpy,
            "internal_energy": self.internal_energy,
            "entropy": self.entropy,
            "specific_heat_cp": self.specific_heat_cp,
            "specific_heat_cv": self.specific_heat_cv,
            "specific_heat_ratio": self.specific_heat_ratio,
            "gas_constant": self.gas_constant,
            "molar_mass": self.molar_mass,
            "molecular_weight": self.molecular_weight,
            "mole_fractions": {
                name: value
                for name, value in self.mole_fractions.items()
                if value > trace
            },
            "mass_fractions": {
                name: value
                for name, value in self.mass_fractions.items()
                if value > trace
            },
            "combustion_gas_mole_fractions": self.combustion_gas_composition(),
            "element_error": self.element_error,
            "max_element_error": self.max_element_error,
            "max_mole_correction": self.max_mole_correction,
            "max_total_mole_correction": self.max_total_mole_correction,
            "enthalpy_error": self.enthalpy_error,
            "temperature_correction": self.temperature_correction,
        }

    def _safe(self, value, fmt=".3e"):
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except Exception:
            return str(value)

    def __str__(self) -> str:
        def format_dict(d: dict, decimals=5):
            return {
                k: round(v, decimals)
                for k, v in d.items()
                if v > 1e-8
            }

        rows = [
            ("Equilibrium mode", self.mode.upper()),
            ("Backend", self.backend),
            ("Success", self.success),
            ("Message", self.message),
            ("Iterations", self.iterations),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Mole fractions", format_dict(self.mole_fractions, 5)),
            ("CombustionGas mole fractions", format_dict(self.combustion_gas_composition(), 5)),
            ("Mass fractions", format_dict(self.mass_fractions, 5)),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Cp [J/kg-K]", self._safe(self.specific_heat_cp, ".3f")),
            ("Cv [J/kg-K]", self._safe(self.specific_heat_cv, ".3f")),
            ("Specific heat ratio", self._safe(self.specific_heat_ratio, ".5f")),
            ("Gas constant [J/kg-K]", self._safe(self.gas_constant, ".3f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Prandtl number", self._safe(self.prandtl, ".5f")),
            ("Speed of sound [m/s]", self._safe(self.speed_of_sound, ".3f")),
            ("Max element error", self._safe(self.max_element_error, ".3e")),
            ("Max mole correction", self._safe(self.max_mole_correction, ".3e")),
        ]

        if self.enthalpy_error is not None:
            rows.append(("Enthalpy error [J/kg]", self._safe(self.enthalpy_error, ".3e")))

        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode={self.mode!r}, "
            f"pressure={self.pressure:.3e} Pa, "
            f"temperature={self.temperature:.2f} K, "
            f"success={self.success})"
        )

    @classmethod
    def available_flash_inputs(cls) -> list[str]:
        return sorted("-".join(sorted(inputs)) for inputs in cls._FLASH_INPUTS)

    @classmethod
    def supported_flash_inputs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @classmethod
    def available_flash_pairs(cls) -> list[str]:
        return cls.available_flash_inputs()

    @classmethod
    def supported_flash_pairs(cls) -> list[str]:
        return cls.available_flash_pairs()

    @classmethod
    def supported_properties(cls) -> list[str]:
        unsupported = getattr(cls, "_UNSUPPORTED_PROPERTIES", set())

        return sorted(
            name
            for name, value in vars(cls).items()
            if isinstance(value, property)
            and not name.startswith("_")
            and name not in unsupported
        )

    @classmethod
    def show_supported_properties(cls) -> list[str]:
        properties = cls.supported_properties()

        for prop in properties:
            print(prop)

        return properties