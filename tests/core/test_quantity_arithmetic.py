"""Behaviour: arithmetic on quantities follows the operand kinds and their units."""

from __future__ import annotations

import operator

import jax.numpy as jnp
import pytest

from orion.core.axis import axis
from orion.core.constant import Constant, const
from orion.core.parameter import Parameter, param
from orion.core.variable import Variable, var


def test_adding_two_constants_returns_a_constant():
    total = const("left", "m", 2.0) + const("right", "m", 3.0)

    assert isinstance(total, Constant)
    assert not isinstance(total, Variable)
    assert total.unit == "m"
    assert float(total.value) == pytest.approx(5.0)


def test_adding_a_constant_and_a_variable_returns_a_variable():
    added = const("base", "m", 2.0) + var("depth", "m", 3.0)
    reversed_order = var("depth", "m", 3.0) + const("base", "m", 2.0)

    assert isinstance(added, Variable)
    assert isinstance(reversed_order, Variable)
    assert float(added.value) == pytest.approx(5.0)
    assert float(reversed_order.value) == pytest.approx(5.0)


def test_adding_a_constant_and_a_parameter_returns_a_variable():
    added = const("base", "m", 2.0) + param("offset", "m", 3.0)
    reversed_order = param("offset", "m", 3.0) + const("base", "m", 2.0)

    assert isinstance(added, Variable)
    assert not isinstance(added, Parameter)
    assert isinstance(reversed_order, Variable)
    assert float(added.value) == pytest.approx(5.0)


def test_adding_two_parameters_returns_a_variable():
    total = param("left", "m", 2.0) + param("right", "m", 3.0)

    assert isinstance(total, Variable)
    assert not isinstance(total, Parameter)
    assert float(total.value) == pytest.approx(5.0)


def test_subtracting_a_constant_from_a_variable_returns_the_difference():
    result = var("end", "m", 5.0) - const("start", "m", 2.0)

    assert isinstance(result, Variable)
    assert result.unit == "m"
    assert float(result.value) == pytest.approx(3.0)


def test_unary_plus_keeps_a_constant_and_turns_a_parameter_into_a_variable():
    kept = +const("height", "m", 3.0)
    promoted = +param("k", "unitless", 0.4)

    assert isinstance(kept, Constant)
    assert float(kept.value) == pytest.approx(3.0)
    assert isinstance(promoted, Variable)
    assert not isinstance(promoted, Parameter)
    assert float(promoted.value) == pytest.approx(0.4)


def test_floor_of_a_parameter_is_a_variable_in_the_same_unit():
    from orion.core.quantity import floor

    lowered = floor(param("height", "m", 1.8))

    assert isinstance(lowered, Variable)
    assert lowered.unit == "m"
    assert float(lowered.value) == pytest.approx(1.0)


def test_ceil_trunc_and_round_follow_the_operand_kind():
    from orion.core.quantity import ceil, trunc
    from orion.core.quantity import round as round_quantity

    ceiled = ceil(const("height", "m", 1.2))
    truncated = trunc(var("offset", "m", -1.8))
    rounded = round_quantity(const("height", "m", 1.5))

    assert isinstance(ceiled, Constant)
    assert float(ceiled.value) == pytest.approx(2.0)
    assert isinstance(truncated, Variable)
    assert float(truncated.value) == pytest.approx(-1.0)
    assert isinstance(rounded, Constant)
    assert float(rounded.value) == pytest.approx(2.0)


def test_adding_a_float_uses_the_quantity_unit():
    total = const("height", "m", 2.0) + 0.5
    promoted = param("height", "m", 2.0) + 0.5

    assert isinstance(total, Constant)
    assert total.unit == "m"
    assert float(total.value) == pytest.approx(2.5)
    assert isinstance(promoted, Variable)
    assert float(promoted.value) == pytest.approx(2.5)


def test_a_float_on_the_left_of_a_subtraction_uses_the_quantity_unit():
    result = 5.0 - var("height", "m", 1.5)

    assert isinstance(result, Variable)
    assert result.unit == "m"
    assert float(result.value) == pytest.approx(3.5)


def test_multiplying_and_dividing_by_a_float_scales_without_changing_the_unit():
    scaled = 2 * var("mass", "kg/m^2", 4.0, dimension="water")
    divided = const("height", "m", 6.0) / 2.0

    assert isinstance(scaled, Variable)
    assert scaled.unit == "kg/m^2"
    assert scaled.dimension == "water"
    assert float(scaled.value) == pytest.approx(8.0)
    assert isinstance(divided, Constant)
    assert divided.unit == "m"
    assert float(divided.value) == pytest.approx(3.0)


def test_negating_a_constant_returns_a_constant_and_negating_a_parameter_returns_a_variable():
    assert isinstance(-const("height", "m", 3.0), Constant)
    assert float((-const("height", "m", 3.0)).value) == pytest.approx(-3.0)
    negated = -param("k", "unitless", 0.4)
    assert isinstance(negated, Variable)
    assert float(negated.value) == pytest.approx(-0.4)


def test_absolute_value_of_a_variable_is_a_variable():
    result = abs(var("offset", "m", -2.0))

    assert isinstance(result, Variable)
    assert result.unit == "m"
    assert float(result.value) == pytest.approx(2.0)


def test_multiplying_by_a_dimensionless_quantity_keeps_the_other_unit():
    scaled = const("factor", "unitless", 2.0) * var("mass", "kg/m^2", 4.0, dimension="water")

    assert isinstance(scaled, Variable)
    assert scaled.unit == "kg/m^2"
    assert scaled.dimension == "water"
    assert float(scaled.value) == pytest.approx(8.0)


def test_multiplying_lengths_returns_an_area():
    area = const("width", "m", 3.0) * const("height", "m", 4.0)

    assert isinstance(area, Constant)
    assert area.unit == "m^2"
    assert float(area.value) == pytest.approx(12.0)


def test_multiplying_bulk_density_by_thickness_returns_mass_per_area():
    mass = const("bdod", "kg/m^3", 1300.0) * var("thickness", "m", 0.3)

    assert isinstance(mass, Variable)
    assert mass.unit == "kg/m^2"
    assert float(mass.value) == pytest.approx(390.0)


def test_dividing_a_length_by_a_length_returns_a_dimensionless_quantity():
    ratio = var("numerator", "m", 6.0) / const("denominator", "m", 2.0)

    assert isinstance(ratio, Variable)
    assert ratio.unit == "unitless"
    assert float(ratio.value) == pytest.approx(3.0)


def test_adding_different_units_is_rejected():
    with pytest.raises(ValueError, match="Cannot combine"):
        operator.add(const("length", "m", 1.0), const("mass", "kg/m^2", 1.0))


def test_adding_a_scalar_constant_to_a_layered_variable_keeps_the_layer_axis():
    layer = axis("layer")
    total = const("base", "m", 1.0) + var("depth", "m", [1.0, 2.0], axes=(layer,))

    assert isinstance(total, Variable)
    assert total.axes == (layer,)
    assert jnp.allclose(total.value, jnp.array([2.0, 3.0]))


def test_adding_two_water_quantities_keeps_the_water_dimension():
    total = const("rain", "kg/m^2", 1.0, dimension="water") + var("irrigation", "kg/m^2", 2.0, dimension="water")

    assert total.dimension == "water"
    assert float(total.value) == pytest.approx(3.0)
