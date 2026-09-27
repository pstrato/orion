"""Shared quantity schema: Constant, Variable, and Learnable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from orion.core.units import Unit

import jax.numpy as jnp

from orion.core.axis import Axis
from orion.core.constraint import ConstrainedEntity, Constraint, Problem
from orion.core.entity import EntityPath, entity
from orion.core.resource import Resource


@entity()
class Quantity[T](ConstrainedEntity):
    """A single named measurable (e.g. crop biomass, soil water in a layer)."""

    unit: Unit
    """Quantity unit (canonical string)."""

    value: T
    """Quantity value."""

    description: str
    """Human-readable description."""

    axes: tuple[Axis, ...]
    """Ordered axes matching ``value`` rank (empty = scalar)."""

    resource: Resource | None = None
    """Resource this quantity represents, when the unit alone does not say so."""


@entity()
class ShapeConstraint[T](Constraint[Quantity[T]]):
    """A constraint on the shape of a value."""

    shape: tuple[int, ...]
    """Expected value shape."""

    def validate(self, path: EntityPath, entity: Quantity[T]) -> Iterable[Problem]:
        """Whether the value has the expected shape."""
        arr = entity.value
        if not hasattr(arr, "shape"):
            yield Problem(self.name, "entity has no shape for shape constraint", path)
        else:
            shape = getattr(arr, "shape", type(arr))
            if shape != self.shape:
                yield Problem(self.name, f"entity shape {shape} != {self.shape}", path)


@entity()
class ScalarConstraint[T](Constraint[Quantity[T]]):
    """Constraint value to a scalar."""

    def validate(self, path: EntityPath, entity: Quantity[T]):
        arr = entity.value
        shape = getattr(arr, "shape", None)
        if shape is not None:
            if len(shape) > 1:
                yield Problem(self.name, f"has incorrect shape for a scalar/batch scalar, got {shape}.", path)
            return


is_scalar = ScalarConstraint("is scalar")
"""Constraint value to a scalar."""


class PositiveConstraint[T](Constraint[Quantity[T]]):
    """Constraint value to be positive."""

    def validate(self, path: EntityPath, entity: Quantity[T]):
        if jnp.any(jnp.asarray(entity.value) <= 0):
            yield Problem(self.name, "has non-positive values.", path)


is_positive = PositiveConstraint("is positive")
"""Constraint value to be positive."""


@entity()
class NonNegativeConstraint[T](Constraint[Quantity[T]]):
    """Constraint value to be non-negative."""

    def validate(self, path: EntityPath, entity: Quantity[T]):
        if jnp.any(jnp.asarray(entity.value) < 0):
            yield Problem(
                self.name,
                "has negative values.",
                path,
            )


is_non_negative = NonNegativeConstraint("is non-negative")
"""Constraint value to be non-negative."""


@entity()
class BetweenConstraint[T](Constraint[Quantity[T]]):
    """Constraint value between two bounds."""

    lower: float | None
    """Lower bound."""

    upper: float | None
    """Upper bound."""

    strict: bool = False
    """Whether the bounds are strict (exclusive)."""

    def validate(self, path: EntityPath, entity: Quantity[T]):
        """Whether the value is between the bounds."""
        value = jnp.asarray(entity.value)
        if self.strict:
            if (self.lower is not None and jnp.any(value <= self.lower)) or (self.upper is not None and jnp.any(value >= self.upper)):
                yield Problem(self.name, f"has values outside of bounds ({self.lower}, {self.upper}).", path)
        else:
            if (self.lower is not None and jnp.any(value < self.lower)) or (self.upper is not None and jnp.any(value > self.upper)):
                yield Problem(
                    self.name,
                    f"has values outside of bounds [{self.lower}, {self.upper}].",
                    path,
                )


between_0_1_inc = BetweenConstraint("between 0 and 1", lower=0.0, upper=1.0, strict=False)
between_0_1_exc = BetweenConstraint("between 0 and 1", lower=0.0, upper=1.0, strict=True)


@entity()
class NegativeConstraint[T](Constraint[Quantity[T]]):
    """Constraint value to be negative."""

    def validate(self, path: EntityPath, entity: Quantity[T]):
        """Whether the value is negative."""
        if jnp.any(jnp.asarray(entity.value) >= 0):
            yield Problem(self.name, "has non-negative values.", path)


is_negative = NegativeConstraint("is negative")
"""Constraint value to be negative."""


@entity()
class NonPositiveConstraint[T](Constraint[Quantity[T]]):
    """Constraint value to be non-positive."""

    def validate(self, path: EntityPath, entity: Quantity):
        """Whether the value is non-positive."""
        if jnp.any(jnp.asarray(entity.value) > 0):
            yield Problem(
                self.name,
                "has positive values.",
                path,
            )


is_non_positive = NonPositiveConstraint("is non-positive")
"""Constraint value to be non-positive."""


@entity()
class FiniteConstraint[T](Constraint[Quantity[T]]):
    """Constrain value to be finite (no NaN / ±inf)."""

    def validate(self, path: EntityPath, entity: Quantity):
        if not jnp.all(jnp.isfinite(jnp.asarray(entity.value))):
            yield Problem(
                self.name,
                "has non-finite values.",
                path,
            )


is_finite = FiniteConstraint("is finite")
"""Constraint value to be finite."""


@entity()
class SumConstraint[T](Constraint[Quantity[T]]):
    """Constraint sum of values."""

    total: float
    """Expected value sum."""

    tolerance: float = 1e-6
    """Tolerance for the sum constraint."""

    def validate(self, path: EntityPath, entity: Quantity):
        """Whether the value has the expected sum."""
        if abs((sum := jnp.sum(entity.value)) - self.total) > self.tolerance:
            yield Problem(
                self.name,
                f"has incorrect sum, expected {self.total}, got {sum}.",
                path,
            )


def value_as_array(
    value: jnp.ndarray | float | int | list[float] | list[int] | None,
) -> jnp.ndarray:
    if isinstance(value, (float, int)):
        return jnp.array(value)
    if isinstance(value, list):
        return jnp.array(value)
    if value is None:
        return jnp.zeros((), dtype=float)
    return value
