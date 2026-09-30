"""Shared quantity schema: Constant, Variable, and Learnable."""

from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from orion.core.constant import Constant
    from orion.core.units import Unit
    from orion.core.variable import Variable

import jax.numpy as jnp

from orion.core.axis import Axis
from orion.core.constraint import ConstrainedEntity, Constraint, Problem
from orion.core.dimension import Dimension
from orion.core.entity import EntityPath, entity


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

    dimension: Dimension | None = None
    """Dimension this quantity represents, when the unit alone does not say so."""

    def __add__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.add, self, other, _sum_unit, number_unit="match")

    def __radd__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.add, self, other, _sum_unit, number_unit="match")

    def __sub__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.subtract, self, other, _sum_unit, number_unit="match")

    def __rsub__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _reflected(jnp.subtract, self, other, _sum_unit, number_unit="match")

    def __mul__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.multiply, self, other, _product_unit, number_unit="unitless")

    def __rmul__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.multiply, self, other, _product_unit, number_unit="unitless")

    def __truediv__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _binary(jnp.true_divide, self, other, _quotient_unit, number_unit="unitless")

    def __rtruediv__(self, other: object) -> Constant | Variable | NotImplementedType:
        return _reflected(jnp.true_divide, self, other, _quotient_unit, number_unit="unitless")

    def __neg__(self) -> Constant | Variable:
        return _unary(jnp.negative, self)

    def __pos__(self) -> Constant | Variable:
        return _unary(jnp.positive, self)

    def __abs__(self) -> Constant | Variable:
        return _unary(jnp.abs, self)

    def __floor__(self) -> Constant | Variable:
        return _unary(jnp.floor, self)

    def __ceil__(self) -> Constant | Variable:
        return _unary(jnp.ceil, self)

    def __trunc__(self) -> Constant | Variable:
        return _unary(jnp.trunc, self)

    def __round__(self, ndigits: int | None = None) -> Constant | Variable:
        if ndigits is None:
            return _unary(jnp.rint, self)
        return _unary(lambda value: jnp.round(value, ndigits), self)


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


_DIMENSIONLESS = frozenset({"unitless"})
_NON_ARITHMETIC = frozenset({"isodate", "coordinate"})
_PRODUCTS: dict[tuple[Unit, Unit], Unit] = {
    ("m", "m"): "m^2",
    ("m", "kg/m^3"): "kg/m^2",
    ("kg/m^3", "m"): "kg/m^2",
    ("m", "m^2/m^3"): "m^2/m^2",
    ("m^2/m^3", "m"): "m^2/m^2",
    ("kg/kg", "kg/m^2"): "kg/m^2",
    ("kg/m^2", "kg/kg"): "kg/m^2",
    ("kg/kg", "kg/m^3"): "kg/m^3",
    ("kg/m^3", "kg/kg"): "kg/m^3",
}


def floor(quantity: Quantity) -> Constant | Variable:
    """Largest whole number not greater than ``quantity``, in the same unit."""
    return quantity.__floor__()


def ceil(quantity: Quantity) -> Constant | Variable:
    """Smallest whole number not less than ``quantity``, in the same unit."""
    return quantity.__ceil__()


def trunc(quantity: Quantity) -> Constant | Variable:
    """``quantity`` rounded toward zero, in the same unit."""
    return quantity.__trunc__()


def round(quantity: Quantity, ndigits: int | None = None) -> Constant | Variable:
    """``quantity`` rounded to ``ndigits`` decimal places, in the same unit."""
    return quantity.__round__(ndigits)


def _plain_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _binary(op, left: Quantity, right: object, unit_of, *, number_unit: str) -> Constant | Variable | NotImplementedType:
    if isinstance(right, Quantity):
        unit = unit_of(_numeric_unit(left.unit), _numeric_unit(right.unit))
        value = op(jnp.asarray(left.value), jnp.asarray(right.value))
        return _result(left, right, unit, value)
    scalar = _plain_number(right)
    if scalar is None:
        return NotImplemented
    other_unit: Unit = left.unit if number_unit == "match" else "unitless"
    unit = unit_of(_numeric_unit(left.unit), other_unit)
    value = op(jnp.asarray(left.value), jnp.asarray(scalar))
    return _result(left, None, unit, value)


def _reflected(op, quantity: Quantity, other: object, unit_of, *, number_unit: str) -> Constant | Variable | NotImplementedType:
    scalar = _plain_number(other)
    if scalar is None:
        return NotImplemented
    other_unit: Unit = quantity.unit if number_unit == "match" else "unitless"
    unit = unit_of(other_unit, _numeric_unit(quantity.unit))
    value = op(jnp.asarray(scalar), jnp.asarray(quantity.value))
    return _result(quantity, None, unit, value)


def _unary(op, quantity: Quantity) -> Constant | Variable:
    unit = _numeric_unit(quantity.unit)
    return _result(quantity, None, unit, op(jnp.asarray(quantity.value)))


def _result(left: Quantity, right: Quantity | None, unit: Unit, value: jnp.ndarray) -> Constant | Variable:
    from orion.core.constant import Constant
    from orion.core.variable import Variable

    kind = Constant if isinstance(left, Constant) and (right is None or isinstance(right, Constant)) else Variable
    axes = left.axes if right is None else _axes(left, right)
    dimension = left.dimension if right is None else _dimension(left, right)
    return kind(left.name, None, unit, value, left.description, axes, dimension)


def _numeric_unit(unit: Unit) -> Unit:
    if unit in _NON_ARITHMETIC:
        raise ValueError(f"Cannot do arithmetic on {unit}.")
    return unit


def _sum_unit(left: Unit, right: Unit) -> Unit:
    if left != right:
        raise ValueError(f"Cannot combine {left} and {right}.")
    return left


def _product_unit(left: Unit, right: Unit) -> Unit:
    if left in _DIMENSIONLESS:
        return right
    if right in _DIMENSIONLESS:
        return left
    try:
        return _PRODUCTS[(left, right)]
    except KeyError:
        raise ValueError(f"Cannot multiply {left} and {right}.") from None


def _quotient_unit(left: Unit, right: Unit) -> Unit:
    if right in _DIMENSIONLESS:
        return left
    if left == right:
        return "unitless"
    for (factor, other), product in _PRODUCTS.items():
        if product == left and other == right:
            return factor
        if product == left and factor == right:
            return other
    raise ValueError(f"Cannot divide {left} by {right}.")


def _axes(left: Quantity, right: Quantity) -> tuple[Axis, ...]:
    if not left.axes:
        return right.axes
    if not right.axes:
        return left.axes
    if tuple(item.name for item in left.axes) != tuple(item.name for item in right.axes):
        raise ValueError("Cannot combine quantities with different axes.")
    return left.axes


def _dimension(left: Quantity, right: Quantity) -> Dimension | None:
    if left.dimension is None or left.dimension == right.dimension:
        return right.dimension if left.dimension is None else left.dimension
    if right.dimension is None:
        return left.dimension
    return None
