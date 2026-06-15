"""Small shared helpers for ThermoProp's public wrapper API."""

from __future__ import annotations

from ._formatting import format_optional


class PropertyIntrospectionMixin:
    """Mixin for wrapper classes that expose property-discovery helpers.

    The mixin only inspects properties defined directly on the concrete class.
    That preserves each wrapper's public API surface while avoiding repeated
    copies of ``supported_properties()``, ``show_supported_properties()``, and
    ``supports_property()`` across wrappers.
    """

    @classmethod
    def supported_properties(cls) -> list[str]:
        """Return public properties intentionally supported by this wrapper."""
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
        """Print and return public properties intentionally supported here."""
        properties = cls.supported_properties()

        for prop in properties:
            print(prop)

        return properties

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        """Return True when ``property_name`` is a supported wrapper property."""
        return property_name in cls.supported_properties()
