"""Behaviour: weather quantities name the dimension they represent."""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp

from orion.core.constant import const
from orion.core.variable import var
from orion.processes.clock import ClockInput
from orion.processes.weather import Weather, WeatherInput


def test_precipitation_is_water_and_keeps_that_dimension_when_its_value_changes():
    clock = ClockInput(
        "clock",
        var("start", "isodate", date(2024, 1, 1), "start"),
        var("end", "isodate", date(2024, 1, 2), "end"),
        const("delta", "hours", 3, "step"),
    )
    weather = WeatherInput("weather", var("Ts", "°C", 0.0), var("Ps", "kg/m^2", 0.0), var("Rs", "W/m^2", 0.0)).states(clock)
    assert isinstance(weather, Weather)
    assert weather.Ps.dimension == "water"
    assert weather.P.dimension == "water"
    assert weather.Ts.dimension == "heat"
    assert weather.Tmin.dimension == "heat"
    assert weather.Rs.dimension == "light"
    assert weather.R.dimension == "light"

    updated = weather.Ps.set(jnp.array([1.0, 2.0, 3.0]))
    assert updated.dimension == "water"
    assert float(updated.value.sum()) == 6.0


def test_temperature_sum_reads_a_base_temperature_quantity():
    weather = Weather(
        name="weather",
        constraint=None,
        Ts=var("Ts", "°C", jnp.array([12.0, 15.0]), "Air temperature"),
        Ps=var("Ps", "kg/m^2", jnp.zeros(2), "Precipitation"),
        Rs=var("Rs", "W/m^2", jnp.zeros(2), "Radiation"),
    )

    total = weather.Tsum(var("base", "°C", 10.0, "Base temperature"))

    assert total.unit == "°C"
    assert float(total.value) == 7.0
