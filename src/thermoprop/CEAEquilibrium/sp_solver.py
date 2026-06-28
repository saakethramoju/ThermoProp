"""
sp_solver.py

Constant-entropy / constant-pressure equilibrium solver controls.

The first ThermoProp SP implementation is intentionally CEA-compatible at the
state level rather than a separate reduced Gibbs matrix: the condensed-phase
wrapper solves a scalar temperature root and calls the existing TP equilibrium
solver at each temperature.  This module contains the SP option/result objects
so the public facade can report SP convergence in the same style as TP/HP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import EquilibriumState


@dataclass(slots=True)
class SPSolverOptions:
    """Numerical controls for fixed-entropy, fixed-pressure solves.

    Entropy is in J/kg-K on the public ThermoProp mass basis.  The solve is
    performed by varying the static temperature until the TP-equilibrium entropy
    at the assigned pressure matches the requested entropy.
    """

    max_iterations: int = 80
    max_bracket_iterations: int = 80

    trace: float = 1e-300
    species_trace: float = 1e-12

    element_tolerance: float = 1e-8
    entropy_tolerance: float = 1e-3
    correction_tolerance: float = 5e-6
    temperature_correction_tolerance: float = 1e-7

    min_temperature: float = 100.0
    max_temperature: float = 20000.0

    size: float = 18.420681
    verbose: bool = False


@dataclass(slots=True)
class SPSolverResult:
    """Result bundle returned by the condensed-phase SP wrapper."""

    state: EquilibriumState
    success: bool
    message: str
    iterations: int
    max_element_error: float
    entropy_error: float
    max_correction: float
    temperature_correction: float
    residual_norm: float
    element_potentials: np.ndarray | None = None
