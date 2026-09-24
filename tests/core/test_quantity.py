"""Behaviour: a quantity keeps its definition, and value constraints accept or reject that value."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from orion.core.axis import axis
from orion.core.constant import const
from orion.core.constraint import Constraint
from orion.core.quantity import (
    BetweenConstraint,
    ShapeConstraint,
    SumConstraint,
    between_0_1_exc,
    between_0_1_inc,
    is_finite,
    is_negative,
    is_non_negative,
    is_non_positive,
    is_positive,
    is_scalar,
)
from orion.core.variable import Variable, var


def test_a_quantity_keeps_its_name_unit_description_axes_constraint_and_value():
    layers = axis("layer", description="Soil layer")
    quantity = var(
        "density",
        "kg/m^3",
        [0.1, 0.2],
        description="Root density",
        constraint=cast(Constraint[Variable], is_non_negative),
        axes=(layers,),
    )
    assert quantity.name == "density"
    assert quantity.unit == "kg/m^3"
    assert quantity.description == "Root density"
    assert quantity.axes == (layers,)
    assert quantity.constraint is is_non_negative
    assert jnp.allclose(quantity.value, jnp.array([0.1, 0.2]))


def test_a_shape_constraint_accepts_the_expected_shape():
    quantity = var("density", "kg/m^3", [0.1, 0.2], description="density")
    assert list(ShapeConstraint("shape", (2,)).validate((), quantity)) == []


def test_a_shape_constraint_rejects_a_different_shape():
    quantity = var("density", "kg/m^3", [0.1], description="density")
    assert list(ShapeConstraint("shape", (2,)).validate((), quantity))


def test_a_shape_constraint_rejects_a_value_with_no_shape():
    quantity = const("count", "1", 3, description="count")
    assert list(ShapeConstraint("shape", ()).validate((), quantity))


def test_a_scalar_constraint_accepts_a_scalar_and_a_batch_vector():
    scalar = var("x", "m", 1.0, description="x")
    batch = var("x", "m", [1.0, 2.0], description="x")
    assert list(is_scalar.validate((), scalar)) == []
    assert list(is_scalar.validate((), batch)) == []


def test_a_scalar_constraint_rejects_a_matrix():
    matrix = var("x", "m", jnp.ones((2, 2)), description="x")
    assert list(is_scalar.validate((), matrix))


def test_a_positive_constraint_accepts_a_positive_value():
    assert list(is_positive.validate((), var("x", "m", 1.0, description="x"))) == []


def test_a_positive_constraint_rejects_zero_and_negative_values():
    for value in (0.0, -1.0, [1.0, 0.0]):
        assert list(is_positive.validate((), var("x", "m", value, description="x")))


def test_a_non_negative_constraint_accepts_zero():
    assert list(is_non_negative.validate((), var("x", "m", 0.0, description="x"))) == []


def test_a_non_negative_constraint_rejects_a_negative_value():
    assert list(is_non_negative.validate((), var("x", "m", [1.0, -0.1], description="x")))


def test_a_negative_constraint_accepts_a_negative_value():
    assert list(is_negative.validate((), var("x", "m", -1.0, description="x"))) == []


def test_a_negative_constraint_rejects_zero_and_positive_values():
    for value in (0.0, 1.0):
        assert list(is_negative.validate((), var("x", "m", value, description="x")))


def test_a_non_positive_constraint_accepts_zero():
    assert list(is_non_positive.validate((), var("x", "m", 0.0, description="x"))) == []


def test_a_non_positive_constraint_rejects_a_positive_value():
    assert list(is_non_positive.validate((), var("x", "m", 0.1, description="x")))


def test_an_inclusive_bound_accepts_its_endpoints():
    bounds = BetweenConstraint("bounds", 0.0, 1.0)
    assert list(bounds.validate((), var("x", "1", 0.0, description="x"))) == []
    assert list(bounds.validate((), var("x", "1", 1.0, description="x"))) == []


def test_an_inclusive_bound_rejects_values_outside_the_endpoints():
    bounds = BetweenConstraint("bounds", 0.0, 1.0)
    assert list(bounds.validate((), var("x", "1", -0.1, description="x")))
    assert list(bounds.validate((), var("x", "1", 1.1, description="x")))


def test_an_exclusive_bound_rejects_its_endpoints():
    bounds = BetweenConstraint("bounds", 0.0, 1.0, strict=True)
    assert list(bounds.validate((), var("x", "1", 0.0, description="x")))
    assert list(bounds.validate((), var("x", "1", 1.0, description="x")))
    assert list(bounds.validate((), var("x", "1", 0.5, description="x"))) == []


def test_a_lower_bound_alone_rejects_only_smaller_values():
    bounds = BetweenConstraint("bounds", 0.0, None)
    assert list(bounds.validate((), var("x", "m", -1.0, description="x")))
    assert list(bounds.validate((), var("x", "m", 5.0, description="x"))) == []


def test_closed_and_open_unit_intervals_differ_at_the_endpoints():
    endpoint = var("fraction", "1", 0.0, description="fraction")
    assert list(between_0_1_inc.validate((), endpoint)) == []
    assert list(between_0_1_exc.validate((), endpoint))


def test_a_finite_constraint_accepts_a_finite_value():
    assert list(is_finite.validate((), var("x", "m", 1.0, description="x"))) == []


def test_a_finite_constraint_rejects_nan_and_infinity():
    for value in (float("nan"), float("inf"), float("-inf")):
        assert list(is_finite.validate((), var("x", "m", value, description="x")))


def test_a_sum_constraint_accepts_a_total_within_tolerance():
    parts = var("parts", "1", [0.25, 0.70], description="parts")
    assert list(SumConstraint("sums to one", 1.0, tolerance=0.1).validate((), parts)) == []


def test_a_sum_constraint_rejects_a_total_outside_tolerance():
    parts = var("parts", "1", [0.25, 0.25], description="parts")
    assert list(SumConstraint("sums to one", 1.0).validate((), parts))
