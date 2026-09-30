"""Behaviour: soil helpers consume layer quantities and return quantities."""

from __future__ import annotations

import math

import pytest

from orion.core.variable import var
from orion.processes.soil.physics import field_capacity, temperature_factor
from orion.processes.soil.soil import make_soil_layer


def test_temperature_factor_reads_a_temperature_quantity():
    factor = temperature_factor(var("temperature", "°C", 30.0, "Soil temperature"))

    assert factor.unit == "unitless"
    assert float(factor.value) == pytest.approx(math.exp(1.0))


def test_field_capacity_is_water_from_the_layer_quantities():
    layer = make_soil_layer("layer", 0.0, 0.3, clay=0.2, bdod=1300.0)
    capacity = field_capacity(layer)

    assert capacity.unit == "kg/m^2"
    assert capacity.dimension == "water"
    assert float(capacity.value) == pytest.approx(54.0)
