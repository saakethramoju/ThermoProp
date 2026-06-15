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
    """Return ``True`` when an update argument was explicitly provided."""

    return value is not UNSET


def provided_items(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return only entries whose value is not :data:`UNSET`."""

    return {key: value for key, value in values.items() if is_provided(value)}
