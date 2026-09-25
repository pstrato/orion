"""Behaviour: Open-Meteo supplies hourly weather and the process copies the clock's hour window."""

from __future__ import annotations

from datetime import date
from typing import cast

import jax.numpy as jnp
import numpy as np
import openmeteo_requests
import pytest
from shapely import Point

from orion.clients.openmeteo import (
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_HOURLY,
    OpenMeteoWeatherInput,
    OpenMeteoWeatherProcess,
    clear_openmeteo_memory_cache,
    fetch_openmeteo_hourly,
    hourly_series_from_response,
)
from orion.core.constant import const
from orion.core.input import LocationInput
from orion.core.setting import Settings
from orion.processes.clock import Clock, ClockInput
from orion.processes.weather import Weather


class _Variable:
    def __init__(self, values: list[float]):
        self._values = np.asarray(values, dtype=np.float64)

    def ValuesAsNumpy(self):
        return self._values


class _Hourly:
    def __init__(self, series: list[list[float]] | None):
        self._series = series

    def Variables(self, index: int):
        if self._series is None:
            return None
        return _Variable(self._series[index])


class _Response:
    def __init__(self, series: list[list[float]] | None):
        self._series = series

    def Hourly(self):
        if self._series is None:
            return None
        return _Hourly(self._series)


class _Client:
    def __init__(self, series: list[list[float]] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._series = series or [
            [float(i) for i in range(48)],
            [float(100 + i) for i in range(48)],
            [float(200 + i) for i in range(48)],
        ]

    def weather_api(self, url: str, params: dict):
        self.calls.append((url, params))
        return [_Response(self._series)]


def _settings(tmp_path) -> Settings:
    return Settings("test", tmp_path, False, False, False)


def _location() -> LocationInput:
    return LocationInput("field", const("geometry", "coordinate", Point(0.1, 51.5), description="field"))


def _clock() -> ClockInput:
    return ClockInput(
        "clock",
        const("start", "isodate", date(2024, 1, 1), description="start"),
        const("end", "isodate", date(2024, 1, 2), description="end"),
        const("delta", "hours", 3, description="step length"),
    )


def _clock_at(step: int, delta: int = 3) -> Clock:
    horizon = _clock()
    initial = horizon.states()
    assert isinstance(initial, Clock)
    return Clock(
        name=initial.name,
        start=initial.start,
        delta=const("delta", "hours", delta, description="step length"),
        step=initial.step.set(jnp.array(step)),
        constraint=initial.constraint,
    )


def test_hourly_response_becomes_temperature_precipitation_and_radiation():
    temperature, precipitation, radiation = hourly_series_from_response(_Response([[1.0, 2.0], [0.0, 0.5], [10.0, 20.0]]))
    assert temperature.tolist() == pytest.approx([1.0, 2.0])
    assert precipitation.tolist() == pytest.approx([0.0, 0.5])
    assert radiation.tolist() == pytest.approx([10.0, 20.0])


def test_hourly_response_without_data_is_rejected():
    with pytest.raises(AssertionError):
        hourly_series_from_response(_Response(None))


def test_openmeteo_fetch_requests_the_archive_for_the_location_through_the_day_after_the_end(tmp_path):
    clear_openmeteo_memory_cache()
    client = _Client()
    temperature, precipitation, radiation = fetch_openmeteo_hourly(_location(), _clock(), _settings(tmp_path), cast(openmeteo_requests.Client, client))
    assert len(client.calls) == 1
    url, params = client.calls[0]
    assert url == OPENMETEO_ARCHIVE_URL
    assert params["latitude"] == pytest.approx(51.5)
    assert params["longitude"] == pytest.approx(0.1)
    assert params["start_date"] == "2024-01-01"
    assert params["end_date"] == "2024-01-03"
    assert params["hourly"] == list(OPENMETEO_HOURLY)
    assert len(temperature) == 48
    assert len(precipitation) == 48
    assert len(radiation) == 48


def test_repeated_openmeteo_fetch_for_the_same_place_and_horizon_is_not_sent_again(tmp_path):
    clear_openmeteo_memory_cache()
    client = _Client()
    settings = _settings(tmp_path)
    fetch_openmeteo_hourly(_location(), _clock(), settings, cast(openmeteo_requests.Client, client))
    fetch_openmeteo_hourly(_location(), _clock(), settings, cast(openmeteo_requests.Client, client))
    assert len(client.calls) == 1


def test_openmeteo_weather_starts_with_one_value_per_hour_of_the_step():
    weather = OpenMeteoWeatherInput("weather").states(_clock())
    assert isinstance(weather, Weather)
    assert weather.Ts.value.shape == (3,)
    assert weather.Ps.value.shape == (3,)
    assert weather.Rs.value.shape == (3,)
    assert weather.Ts.unit == "°C"
    assert weather.Ps.unit == "kg/m^2"
    assert weather.Rs.unit == "W/m^2"
    assert weather.Ts.axes[0].name == "within_step"


def test_openmeteo_process_keeps_the_fetched_hours_for_one_location(tmp_path, monkeypatch):
    clear_openmeteo_memory_cache()
    client = _Client()
    monkeypatch.setattr(openmeteo_requests, "Client", lambda *args, **kwargs: client)
    process = OpenMeteoWeatherInput("weather").processes(_settings(tmp_path), _location(), _clock())
    assert isinstance(process, OpenMeteoWeatherProcess)
    assert process.Ts.shape == (1, 48)
    assert process.Ps.shape == (1, 48)
    assert process.Rs.shape == (1, 48)
    assert jnp.allclose(process.Ts[0, :2], jnp.array([0.0, 1.0]))
    assert jnp.allclose(process.Ps[0, :2], jnp.array([100.0, 101.0]))
    assert process.delta == 3


def test_weather_step_copies_the_clock_hour_window_and_starts_precipitation_one_hour_later():
    temperature = jnp.arange(10, dtype=float).reshape(1, 10)
    precipitation = jnp.arange(100, 110, dtype=float).reshape(1, 10)
    radiation = jnp.arange(200, 210, dtype=float).reshape(1, 10)
    process = OpenMeteoWeatherProcess("OpenMeteoWeather", temperature, precipitation, radiation, 3)
    weather = OpenMeteoWeatherInput("weather").states(_clock())
    assert isinstance(weather, Weather)

    first = process.step(_clock_at(0), weather)
    assert jnp.allclose(first.Ts.value, jnp.array([[0.0, 1.0, 2.0]]))
    assert jnp.allclose(first.Ps.value, jnp.array([[101.0, 102.0, 103.0]]))
    assert jnp.allclose(first.Rs.value, jnp.array([[200.0, 201.0, 202.0]]))
    assert first.name == weather.name

    second = process.step(_clock_at(1), first)
    assert jnp.allclose(second.Ts.value, jnp.array([[3.0, 4.0, 5.0]]))
    assert jnp.allclose(second.Ps.value, jnp.array([[104.0, 105.0, 106.0]]))
    assert jnp.allclose(second.Rs.value, jnp.array([[203.0, 204.0, 205.0]]))
