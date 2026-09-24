"""Variable and learnable quantities — values change in sim / training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orion.core.constraint import Constraint
from orion.core.entity import entity

if TYPE_CHECKING:
    from orion.core.axis import Axis
    from orion.core.units import Unit

import jax.numpy as jnp

from orion.core.quantity import Quantity, value_as_array


@entity()
class Variable(Quantity[jnp.ndarray]):
    """A variable during simulation."""

    def set(self, value: jnp.ndarray) -> Variable:
        """Return a new variable with the given value."""
        return Variable(
            self.name,
            self.constraint,
            self.unit,
            value,
            self.description,
            self.axes,
        )


def var(
    name: str,
    unit: Unit,
    value: jnp.ndarray | float | int | list[float] | list[int] | None = None,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
) -> Variable:
    """Create a variable quantity."""
    return Variable(name, constraint, unit, value_as_array(value), description, axes)
