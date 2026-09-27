"""Constant quantities — values never change during simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orion.core.axis import Axis
    from orion.core.units import Unit

from orion.core.constraint import Constraint
from orion.core.entity import entity
from orion.core.quantity import Quantity
from orion.core.resource import Resource


@entity()
class Constant[T](Quantity[T]):
    """A constant quantity during simulation."""


def const[T](
    name: str,
    unit: Unit,
    value: T,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
    resource: Resource | None = None,
) -> Constant[T]:
    """Create a constant quantity."""
    quantity = Constant[T](name, constraint, unit, value, description, axes, resource)
    return quantity
