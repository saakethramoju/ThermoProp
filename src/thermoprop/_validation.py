"""Shared validation helpers for ThermoProp wrapper inputs."""

from __future__ import annotations

import numpy as np


def validate_fraction_vector(fractions, label: str, *, atol: float = 1e-6) -> np.ndarray:
    """Return a normalized NumPy vector after validating mixture fractions.

    Parameters
    ----------
    fractions:
        Sequence of mass or mole fractions.
    label:
        Human-readable name used in error messages.
    atol:
        Absolute tolerance for the sum-to-one check.
    """
    vector = np.asarray(fractions, dtype=float)

    if vector.size == 0:
        raise ValueError(f"{label} cannot be empty")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")

    if np.any(vector < 0.0):
        raise ValueError(f"{label} must be nonnegative")

    if not np.isclose(vector.sum(), 1.0, atol=atol):
        raise ValueError(f"{label} must sum to 1.0")

    return vector
