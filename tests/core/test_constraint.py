"""Behaviour: constraints report problems on a quantity and on the path to it."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from orion.core.constraint import Constraint, problems
from orion.core.entity import Entity, EntityRelation, entity
from orion.core.quantity import is_finite, is_positive, is_scalar
from orion.core.variable import Variable, var


def test_a_constraint_without_checks_reports_no_problems():
    quantity = var("x", "m", -1.0, description="x", constraint=Constraint("unchecked"))
    assert list(problems(quantity)) == []


def test_a_quantity_without_a_constraint_reports_no_problems():
    assert list(problems(var("x", "m", -1.0, description="x"))) == []


def test_combined_constraints_report_every_failing_check():
    value = var("x", "m", jnp.asarray([[-1.0, float("nan")]]), description="x")
    reported = list((is_scalar + is_positive + is_finite).validate((), value))
    assert [problem.name for problem in reported] == ["is scalar", "is positive", "is finite"]


def test_problems_name_the_failing_quantity_by_its_relation_path():
    @entity()
    class Plot(Entity):
        layers: tuple[Variable, ...]

    plot = Plot(
        "plot",
        (
            var("water", "m", 1.0, description="water", constraint=cast(Constraint[Variable], is_positive)),
            var("water", "m", 0.0, description="water", constraint=cast(Constraint[Variable], is_positive)),
        ),
    )
    reported = list(problems(plot))
    assert len(reported) == 1
    assert reported[0].path == (EntityRelation("layers", index=1),)
    assert "layers[1]" in str(reported[0])
    assert "non-positive" in reported[0].description
