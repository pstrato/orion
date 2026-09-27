"""Behaviour: a variable value can be replaced, and that value is the JAX leaf."""

from __future__ import annotations

from datetime import date
from typing import cast

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import pytest

from orion.core.axis import axis
from orion.core.constraint import Constraint, problems
from orion.core.quantity import is_non_negative
from orion.core.variable import Variable, var


def test_replacing_a_variable_value_keeps_its_definition_and_constraint():
    layers = axis("layer")
    original = var("mass", "kg/m^2", 1.0, description="Biomass", constraint=cast(Constraint[Variable], is_non_negative), axes=(layers,))
    updated = original.set(jnp.array(-1.0))

    assert updated is not original
    assert float(original.value) == pytest.approx(1.0)
    assert list(problems(original)) == []
    assert float(updated.value) == pytest.approx(-1.0)
    assert list(problems(updated))
    assert updated.name == original.name
    assert updated.unit == original.unit
    assert updated.description == original.description
    assert updated.axes == original.axes
    assert updated.constraint is original.constraint


def test_an_omitted_variable_value_is_zero():
    assert float(var("x", "m").value) == pytest.approx(0.0)
    assert var("x", "m").value.shape == ()


def test_a_numeric_variable_holds_a_jax_array():
    value = var("mass", "kg", 1.0, description="mass").value
    assert isinstance(value, jnp.ndarray)
    assert float(value) == pytest.approx(1.0)


def test_a_date_variable_holds_a_jax_datetime():
    value = var("start", "isodate", date(2024, 1, 1), description="start").value
    assert isinstance(value, jdt.Datetime)
    assert value.to_pydatetime().date() == date(2024, 1, 1)


def test_a_variable_value_is_replaced_when_its_pytree_leaf_changes():
    layers = axis("layer", description="Soil layer")
    quantity = var("density", "kg/m^3", [1.0, 2.0], description="Root density", axes=(layers,))
    leaves, treedef = jax.tree_util.tree_flatten(quantity)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], jnp.array([1.0, 2.0]))

    rebuilt = jax.tree_util.tree_unflatten(treedef, [jnp.array([9.0, 8.0])])
    assert jnp.allclose(rebuilt.value, jnp.array([9.0, 8.0]))
    assert rebuilt.unit == "kg/m^3"
    assert rebuilt.description == "Root density"
    assert rebuilt.axes == (layers,)
