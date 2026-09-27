"""Behaviour: weather quantities name the resource they represent."""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp

from orion.core.constant import const
from orion.core.variable import var
from orion.processes.clock import ClockInput
from orion.processes.weather import Weather, WeatherInput


def test_precipitation_is_water_and_keeps_that_resource_when_its_value_changes():
    clock = ClockInput(
        "clock",
        var("start", "isodate", date(2024, 1, 1), "start"),
        var("end", "isodate", date(2024, 1, 2), "end"),
        const("delta", "hours", 3, "step"),
    )
    weather = WeatherInput("weather", var("Ts", "°C", 0.0), var("Ps", "kg/m^2", 0.0), var("Rs", "W/m^2", 0.0)).states(clock)
    assert isinstance(weather, Weather)
    assert weather.Ps.resource == "water"
    assert weather.P.resource == "water"
    assert weather.Ts.resource == "heat"
    assert weather.Tmin.resource == "heat"
    assert weather.Rs.resource == "light"
    assert weather.R.resource == "light"

    updated = weather.Ps.set(jnp.array([1.0, 2.0, 3.0]))
    assert updated.resource == "water"
    assert float(updated.value.sum()) == 6.0
