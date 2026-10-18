"""Conversion between the frozen specification dataclasses and plain data.

Plain data means what JSON and YAML hold: dicts, lists, numbers, strings and ``None``. The
conversion is generic, driven by each dataclass's type hints, so a new field of a
specification class needs no serialisation code. It is exact for the types the specification
uses: nested frozen dataclasses, ``tuple[X, ...]``, ``X | None``, ``Literal`` values, ``float``,
``int`` and ``str``. A dataclass may name, in a ``plain_omits`` attribute, fields its plain form
leaves out: values derived from its other fields (a root from the recipe that defines it), which
it recomputes when rebuilt, so that only what defines it enters its plain form and its digest.

Examples
--------
>>> from pihm.problems.spec import Polynomial
>>> to_plain(Polynomial("x", (1.0, 0.0, -1.0)))
{'axis': 'x', 'coefficients': [1.0, 0.0, -1.0]}
>>> from_plain(Polynomial, {"axis": "x", "coefficients": [1, 0, -1]})
Polynomial(axis='x', coefficients=(1.0, 0.0, -1.0))
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, Literal, get_args, get_origin

__all__ = ["from_plain", "to_plain"]


def to_plain(value: Any) -> Any:
    """Convert a specification dataclass (recursively) to plain data.

    Fields equal to their default are left out, so that adding a new optional field later does
    not change the plain form, or the digest, of existing specifications; so are the fields a
    dataclass names in ``plain_omits`` (module docstring).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        plain: dict[str, Any] = {}
        omitted = getattr(value, "plain_omits", ())
        for item in dataclasses.fields(value):
            current = getattr(value, item.name)
            if item.name in omitted:
                continue
            if item.default is not dataclasses.MISSING and current == item.default:
                continue
            plain[item.name] = to_plain(current)
        return plain
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    return value


def from_plain(annotation: Any, data: Any) -> Any:
    """Convert plain data back to ``annotation``, validating types on the way.

    Raises
    ------
    ValueError
        For an unknown key, a missing sequence or mapping, or a value of the wrong type.
    """
    origin = get_origin(annotation)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not isinstance(data, dict):
            raise ValueError(f"expected a mapping for {annotation.__name__}, got {data!r}")
        hints = typing.get_type_hints(annotation)
        names = {item.name for item in dataclasses.fields(annotation)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)} for {annotation.__name__}")
        return annotation(**{key: from_plain(hints[key], item) for key, item in data.items()})
    if origin is tuple:
        item_type = get_args(annotation)[0]
        if not isinstance(data, (list, tuple)):
            raise ValueError(f"expected a sequence, got {data!r}")
        return tuple(from_plain(item_type, item) for item in data)
    if origin in (types.UnionType, typing.Union):
        options = [option for option in get_args(annotation) if option is not type(None)]
        return None if data is None else from_plain(options[0], data)
    if origin is Literal:
        if data not in get_args(annotation):
            raise ValueError(f"{data!r} is not one of {get_args(annotation)}")
        return data
    if annotation is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ValueError(f"expected a number, got {data!r}")
        return float(data)
    if annotation in (int, str):
        if not isinstance(data, annotation) or isinstance(data, bool):
            raise ValueError(f"expected {annotation.__name__}, got {data!r}")
        return data
    raise TypeError(f"cannot convert to {annotation!r}")
