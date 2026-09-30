"""Behaviour: a parameter value can be replaced, and that value is the JAX leaf."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from orion.core.axis import axis
from orion.core.constraint import Constraint, problems
from orion.core.parameter import Parameter, param
from orion.core.quantity import is_non_negative


def test_replacing_a_parameter_value_keeps_its_definition_and_constraint():
    layers = axis("layer")
    original = param("k", "unitless", 0.5, description="Extinction", constraint=cast(Constraint[Parameter], is_non_negative), axes=(layers,))
    updated = original.set(jnp.array(-0.1))

    assert updated is not original
    assert float(original.value) == pytest.approx(0.5)
    assert list(problems(original)) == []
    assert float(updated.value) == pytest.approx(-0.1)
    assert list(problems(updated))
    assert updated.name == original.name
    assert updated.unit == original.unit
    assert updated.description == original.description
    assert updated.axes == original.axes
    assert updated.constraint is original.constraint


def test_an_omitted_parameter_value_is_zero():
    assert float(param("k", "unitless").value) == pytest.approx(0.0)
    assert param("k", "unitless").value.shape == ()


def test_a_parameter_value_is_replaced_when_its_pytree_leaf_changes():
    quantity = param("k", "unitless", [0.4, 0.6], description="Extinction", axes=(axis("organ"),))
    leaves, treedef = jax.tree_util.tree_flatten(quantity)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], jnp.array([0.4, 0.6]))

    rebuilt = jax.tree_util.tree_unflatten(treedef, [jnp.array([0.2, 0.3])])
    assert jnp.allclose(rebuilt.value, jnp.array([0.2, 0.3]))
    assert rebuilt.description == "Extinction"
    assert rebuilt.unit == "unitless"
    assert rebuilt.axes[0].name == "organ"
