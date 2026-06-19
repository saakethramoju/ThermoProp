from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def composition_dict(names, fractions) -> dict[str, float]:
    """Return a plain ``{name: fraction}`` composition dictionary.

    The result is intentionally a normal dictionary so it can be passed directly
    into another ThermoProp wrapper or through FullFlow ``Lookup`` chaining.
    """

    return {
        str(name): float(fraction)
        for name, fraction in zip(names, fractions)
    }


def normalize_single_component(value: Any, wrapper_name: str) -> tuple[str, dict[str, float]]:
    """Normalize a pure single-component input.

    ``Propellant`` and ``Material`` are currently pure-component wrappers, but
    they still expose ``composition`` for API consistency and FullFlow chaining.
    This helper lets those wrappers accept either a plain string or a one-item
    composition dictionary such as ``{"lox": 1.0}``.
    """

    if isinstance(value, str):
        name = value
        return name, {name: 1.0}

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{wrapper_name} composition cannot be empty.")

        if len(value) != 1:
            raise ValueError(
                f"{wrapper_name} currently supports only one component. "
                f"Received {len(value)} components."
            )

        name, fraction = next(iter(value.items()))
        fraction = float(fraction)

        if not np.isfinite(fraction):
            raise ValueError(f"{wrapper_name} single-component fraction must be finite.")

        if fraction < 0.0:
            raise ValueError(f"{wrapper_name} single-component fraction must be nonnegative.")

        if fraction == 0.0:
            raise ValueError(f"{wrapper_name} single-component fraction cannot be zero.")

        name = str(name)
        return name, {name: 1.0}

    raise TypeError(
        f"{wrapper_name} input must be a string or a one-item composition dictionary."
    )
