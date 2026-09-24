"""Base types every simulation object shares."""

from __future__ import annotations

import types
from annotationlib import Format
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeVar, Union, dataclass_transform, get_args, get_origin, get_type_hints

from jax.tree_util import register_dataclass

_DATA_TYPE_NAMES = frozenset({"Input", "State", "Variable", "Parameter", "Process"})
_VALUE_DATA_TYPE_NAMES = frozenset({"Variable", "Parameter"})


def _is_orion_data_type(annotation: object) -> bool:
    if not isinstance(annotation, type):
        return False
    return any(base.__name__ in _DATA_TYPE_NAMES and str(getattr(base, "__module__", "")).startswith("orion.") for base in annotation.__mro__)


def _is_data_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return _is_data_annotation(args[0])
        return all(_is_data_annotation(arg) for arg in args)
    if origin in {Union, types.UnionType}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return len(args) == 1 and _is_data_annotation(args[0])
    return _is_orion_data_type(annotation)


def _is_value_class(cls: type) -> bool:
    return any(base.__name__ in _VALUE_DATA_TYPE_NAMES and str(getattr(base, "__module__", "")).startswith("orion.") for base in cls.__mro__)


def _field_annotations(cls: type) -> dict[str, object]:
    try:
        if Format is not None:
            return get_type_hints(cls, include_extras=True, format=Format.VALUE)
        return get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 — fall back to raw dataclass annotations
        return {field.name: field.type for field in fields(cls)}


def discover_entity_fields(cls: type) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split dataclass fields into JAX pytree data vs meta from their types.

    Data: ``Input``, ``State``, ``Variable``, ``Parameter``, ``Process``, and
    tuples of those. ``Variable`` / ``Parameter`` ``value`` is data. Everything
    else is meta.
    """
    hints = _field_annotations(cls)
    data: list[str] = []
    meta: list[str] = []
    payload = _is_value_class(cls)
    for field in fields(cls):
        annotation = hints.get(field.name, field.type)
        if (payload and field.name == "value") or _is_data_annotation(annotation):
            data.append(field.name)
        else:
            meta.append(field.name)
    return tuple(data), tuple(meta)


@dataclass_transform(frozen_default=True)
@dataclass(frozen=True)
class Entity:
    """A simulation object."""

    name: str
    """Entity name."""

    __entity_data_fields__: ClassVar[tuple[str, ...]]
    __entity_meta_fields__: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__dataclass_params__" in cls.__dict__:
            return
        dataclass(frozen=True)(cls)


T = TypeVar("T", bound=Entity)


@dataclass_transform(frozen_default=True)
def entity(cls: type[T] | None = None, **_unused: object) -> type[T] | Callable[[type[T]], type[T]]:
    """Register an immutable Entity subclass as a JAX pytree.

    Data vs meta roles are discovered from dataclass field types.
    """

    def decorator(target: type[T]) -> type[T]:
        data_fields, meta_fields = discover_entity_fields(target)
        target.__entity_data_fields__ = data_fields
        target.__entity_meta_fields__ = meta_fields
        register_dataclass(
            target,
            data_fields=data_fields,
            meta_fields=meta_fields,
            drop_fields=[],
        )
        return target

    if cls is not None:
        return decorator(cls)
    return decorator
