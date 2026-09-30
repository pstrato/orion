"""Behaviour: a constant keeps the value it was given, including through a JAX tree."""

from __future__ import annotations

from datetime import date

import jax

from orion.core.constant import const


def test_a_constant_keeps_the_given_value():
    assert const("count", "unitless", 3, description="count").value == 3
    assert const("sowing", "isodate", date(2024, 1, 1), description="sowing date").value == date(2024, 1, 1)


def test_a_constant_value_is_not_a_pytree_leaf():
    quantity = const("start", "m", 1, description="start")
    leaves, treedef = jax.tree_util.tree_flatten(quantity)
    assert leaves == []
    assert jax.tree_util.tree_unflatten(treedef, []).value == 1
