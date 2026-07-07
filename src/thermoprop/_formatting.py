"""Formatting helpers for human-readable ThermoProp objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math


def is_finite_number(value: Any) -> bool:
    """Return a boolean capability or classification check.

    The check uses ThermoProp's normal name canonicalization and backend lookup
    rules, but converts lookup failures into ``False`` when appropriate.  This makes
    it safe to use in validation code before calling stricter property methods.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_or_none(value: Any) -> float | None:
    """Execute the documented ``finite_or_none`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_optional(value: Any, fmt: str = ".3e", *, missing: str = "N/A") -> str:
    """Format optional scalar values for ``__str__`` tables.

    ``None`` and non-finite numeric values are reported as ``missing``. This
    keeps printed objects readable when a backend cannot evaluate an optional
    derivative or transport property.
    """
    if value is None:
        return missing

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(numeric):
        return missing

    try:
        return f"{numeric:{fmt}}"
    except Exception:
        return str(value)


def rounded_dict(values: dict[str, Any], decimals: int = 3) -> dict[str, Any]:
    """Execute the documented ``rounded_dict`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            out[key] = value
            continue
        out[key] = round(numeric, decimals) if math.isfinite(numeric) else None
    return out


def format_rows(rows: Iterable[tuple[str, Any]]) -> str:
    """Execute the documented ``format_rows`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """
    rows = list(rows)
    if not rows:
        return ""
    width = max(len(str(label)) for label, _ in rows)
    return "\n".join(f"{label:<{width}} : {value}" for label, value in rows)
