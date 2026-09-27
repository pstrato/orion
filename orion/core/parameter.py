"""Variable and learnable quantities — values change in sim / training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orion.core.axis import Axis
from orion.core.constraint import Constraint
from orion.core.entity import entity

if TYPE_CHECKING:
    from orion.core.units import Unit

import jax.numpy as jnp

from orion.core.quantity import Quantity, value_as_array
from orion.core.resource import Resource


@entity()
class Parameter(Quantity[jnp.ndarray]):
    """A parameter during simulation."""

    def set(self, value: jnp.ndarray) -> Parameter:
        """Return a new parameter with the given value."""
        return Parameter(
            self.name,
            self.constraint,
            self.unit,
            value,
            self.description,
            self.axes,
            self.resource,
        )


def param(
    name: str,
    unit: Unit,
    value: jnp.ndarray | float | int | list[float] | list[int] | None = None,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
    resource: Resource | None = None,
) -> Parameter:
    """Create a parameter."""
    return Parameter(name, constraint, unit, value_as_array(value), description, axes, resource)
