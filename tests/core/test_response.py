"""Behaviour: a ZO response climbs from 0 to 1, and a ZOOZ response climbs, holds, then falls."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import pytest

from orion.core.constant import const
from orion.core.parameter import Parameter, param
from orion.core.response import ZO, ZOOZ, Function, Response, Sigmoid
from orion.core.variable import Variable, var


class _Square(Function):
    """Monotone from 0 to 1, and not a straight line."""

    def __call__(self, x: Variable | Parameter) -> Variable:
        return cast(Variable, x * x)


def _zo(zero: float, one: float) -> ZO:
    return ZO("zo", func=const("func", "unitless", _Square("square")), z=param("z", "°C", zero), o=param("o", "°C", one))


def _at(response: Response, value: float) -> Variable:
    return response(var("temperature", "°C", value))


def test_zo_is_zero_before_zero_follows_the_function_until_one_and_is_one_after():
    response = _zo(0.0, 10.0)

    below = _at(response, -5.0)
    at_zero = _at(response, 0.0)
    between = _at(response, 5.0)
    at_one = _at(response, 10.0)
    above = _at(response, 15.0)

    assert isinstance(between, Variable)
    assert below.unit == "unitless"
    assert float(below.value) == pytest.approx(0.0)
    assert float(at_zero.value) == pytest.approx(0.0)
    assert float(between.value) == pytest.approx(0.25)
    assert float(at_one.value) == pytest.approx(1.0)
    assert float(above.value) == pytest.approx(1.0)


def test_zo_applies_the_function_at_each_value():
    response = _zo(0.0, 10.0)
    series = response(var("temperature", "°C", jnp.array([-5.0, 5.0, 15.0])))

    assert [float(item) for item in series.value] == pytest.approx([0.0, 0.25, 1.0])


def test_zooz_rises_with_the_function_holds_one_and_falls_with_the_function():
    response = ZOOZ(
        "zooz",
        func=const("func", "unitless", _Square("square")),
        lz=param("lz", "°C", 0.0),
        lo=param("lo", "°C", 10.0),
        ho=param("ho", "°C", 20.0),
        hz=param("hz", "°C", 30.0),
    )

    assert float(_at(response, -5.0).value) == pytest.approx(0.0)
    assert float(_at(response, 5.0).value) == pytest.approx(0.25)
    assert float(_at(response, 10.0).value) == pytest.approx(1.0)
    assert float(_at(response, 15.0).value) == pytest.approx(1.0)
    assert float(_at(response, 20.0).value) == pytest.approx(1.0)
    assert float(_at(response, 25.0).value) == pytest.approx(0.25)
    assert float(_at(response, 30.0).value) == pytest.approx(0.0)
    assert float(_at(response, 35.0).value) == pytest.approx(0.0)
    assert _at(response, 15.0).unit == "unitless"


def test_sigmoid_of_zero_is_one_half_and_increases_toward_one():
    sigmoid = Sigmoid("sigmoid")
    at_zero = sigmoid(var("x", "unitless", 0.0))
    at_one = sigmoid(var("x", "unitless", 1.0))

    assert at_zero.name == "sig(x)"
    assert at_zero.unit == "unitless"
    assert float(at_zero.value) == pytest.approx(0.5)
    assert float(at_one.value) == pytest.approx(float(1.0 / (1.0 + jnp.exp(-1.0))))
    assert float(at_zero.value) < float(at_one.value)
