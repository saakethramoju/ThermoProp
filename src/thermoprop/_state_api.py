"""Shared helpers for mutable ThermoProp wrapper state.

The public wrappers use :meth:`update` for batched state changes.  The sentinel
below lets an ``update`` method distinguish "argument omitted" from "argument
provided as None"; this matters for properties such as optional pressure.
"""

from __future__ import annotations

from typing import Any, Mapping


class _UnsetType:
    """Sentinel type used for omitted update arguments."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "UNSET"


UNSET = _UnsetType()


def is_provided(value: Any) -> bool:
    """Return a boolean capability or classification check.

    The check uses ThermoProp's normal name canonicalization and backend lookup
    rules, but converts lookup failures into ``False`` when appropriate.  This makes
    it safe to use in validation code before calling stricter property methods.
    """

    return value is not UNSET


def provided_items(values: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the documented ``provided_items`` operation for ``ThermoProp``.

    Arguments are validated and normalized using the same rules as the high-level
    wrappers.  Return values follow ThermoProp's SI-unit and composition
    conventions, and failures are reported through ThermoProp exception types with
    contextual messages rather than silent fallbacks.
    """

    return {key: value for key, value in values.items() if is_provided(value)}
