"""Variable and learnable quantities — values change in sim / training."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, overload

from orion.core.constraint import Constraint
from orion.core.entity import entity

if TYPE_CHECKING:
    from orion.core.axis import Axis
    from orion.core.units import Unit

import jax.numpy as jnp
import jax_datetime as jdt

from orion.core.quantity import Quantity, value_as_array
from orion.core.resource import Resource

type JaxNumeric = jnp.ndarray
type JaxDate = jdt.Datetime


@entity()
class Variable[T: (JaxNumeric, JaxDate) = JaxNumeric](Quantity[T]):
    """A variable during simulation. Values are JAX numbers, or JAX datetimes."""

    def set(self, value: T) -> Variable[T]:
        """Return a new variable with the given value."""
        return Variable(
            self.name,
            self.constraint,
            self.unit,
            value,
            self.description,
            self.axes,
            self.resource,
        )


@overload
def var(
    name: str,
    unit: Unit,
    value: date | datetime | JaxDate,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
    resource: Resource | None = None,
) -> Variable[JaxDate]: ...


@overload
def var(
    name: str,
    unit: Unit,
    value: jnp.ndarray | float | int | list[float] | list[int] | None = None,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
    resource: Resource | None = None,
) -> Variable[JaxNumeric]: ...


def var(
    name: str,
    unit: Unit,
    value: jnp.ndarray | float | int | list[float] | list[int] | date | datetime | JaxDate | None = None,
    description: str = "",
    constraint: Constraint | None = None,
    axes: tuple[Axis, ...] = (),
    resource: Resource | None = None,
) -> Variable[JaxNumeric] | Variable[JaxDate]:
    """Create a variable quantity. A Python date becomes a JAX datetime; anything else is numeric."""
    if isinstance(value, (date, datetime, jdt.Datetime)):
        return Variable(name, constraint, unit, _as_jax_date(value), description, axes, resource)
    return Variable(name, constraint, unit, value_as_array(value), description, axes, resource)


def _as_jax_date(value: date | datetime | jdt.Datetime) -> jdt.Datetime:
    if isinstance(value, jdt.Datetime):
        return value
    if isinstance(value, datetime):
        return jdt.to_datetime(value)
    return jdt.to_datetime(value.isoformat())
