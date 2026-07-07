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
        """Return the properties supported by this ThermoProp interface.

        Use this helper before constructing models or exposing choices in a user
        interface.  Results are normalized and sorted where practical, and they reflect
        the installed ThermoProp package data plus any runtime aliases added in the
        current Python process.
        """
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
        """Print and return the available supported properties.

        The printed table is intended for interactive discovery.  The return value
        contains the same information in normal Python data structures so scripts,
        examples, tests, and documentation generators can reuse it without parsing
        stdout.
        """
        properties = cls.supported_properties()

        for prop in properties:
            print(prop)

        return properties

    @classmethod
    def supports_property(cls, property_name: str) -> bool:
        """Return a boolean capability or classification check.

        The check uses ThermoProp's normal name canonicalization and backend lookup
        rules, but converts lookup failures into ``False`` when appropriate.  This makes
        it safe to use in validation code before calling stricter property methods.
        """
        return property_name in cls.supported_properties()
