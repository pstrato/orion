from __future__ import annotations

from datetime import timedelta
from typing import Any

import jax.numpy as jnp
import openmeteo_requests
from jax.lax import dynamic_slice

from orion.clients.http_cache import cached_session
from orion.core.axis import WITHIN_STEP
from orion.core.entity import entity
from orion.core.input import Input, Inputs, LocationInput
from orion.core.process import Process
from orion.core.quantity import is_finite, is_non_negative
from orion.core.setting import Settings
from orion.core.variable import var
from orion.processes.clock import Clock, ClockInput
from orion.processes.weather import Weather

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HOURLY = (
    "temperature_2m",
    "precipitation",
    "shortwave_radiation",
)

_hourly_memory: dict[tuple[object, ...], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}


def clear_openmeteo_memory_cache() -> None:
    """Drop in-process hourly archives (tests / explicit invalidation)."""
    _hourly_memory.clear()


def fetch_openmeteo_hourly(
    location: LocationInput,
    clock: ClockInput,
    settings: Settings,
    client: openmeteo_requests.Client,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fetch hourly temperature, precipitation, and shortwave radiation."""
    memory_key = (round(location.geometry.value.y, 5), round(location.geometry.value.x, 5), clock.start.value.isoformat(), clock.end.value.isoformat())
    data = _hourly_memory.get(memory_key)
    if data is not None:
        return data

    session = cached_session(settings.cache_path)

    if client is not None:
        openmeteo = client
    elif session is not None:
        openmeteo = openmeteo_requests.Client(session=session)  # type: ignore[arg-type]
    else:
        openmeteo = openmeteo_requests.Client()

    params = {
        "latitude": location.geometry.value.y,
        "longitude": location.geometry.value.x,
        "start_date": clock.start.value.isoformat(),
        "end_date": (clock.end.value + timedelta(days=1)).isoformat(),
        "hourly": list(OPENMETEO_HOURLY),
    }
    responses = openmeteo.weather_api(OPENMETEO_ARCHIVE_URL, params=params)
    series = hourly_series_from_response(responses[0])
    _hourly_memory[memory_key] = series
    return series


def hourly_series_from_response(response: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Parse an Open-Meteo archive response into Ts, Ps, Rs arrays."""
    hourly = response.Hourly()
    assert hourly is not None, "Hourly data is required for the weather process."
    temperature = hourly.Variables(0)
    precipitation = hourly.Variables(1)
    shortwave = hourly.Variables(2)
    assert temperature is not None, "Hourly temperature_2m data is required."
    assert precipitation is not None, "Hourly precipitation data is required."
    assert shortwave is not None, "Hourly shortwave radiation data is required."
    return (
        jnp.asarray(temperature.ValuesAsNumpy()),
        jnp.asarray(precipitation.ValuesAsNumpy()),
        jnp.asarray(shortwave.ValuesAsNumpy()),
    )


@entity()
class OpenMeteoWeatherInput(Input):
    """Weather from Open-Meteo archive for each location sample point."""

    def states(self, clock: ClockInput):
        zeros = jnp.zeros((clock.delta.value,))
        return Weather(
            name="weather",
            constraint=None,
            Ts=var("Ts", "°C", zeros, description="Air temperature within the step", axes=(WITHIN_STEP,), constraint=is_finite),
            Ps=var(
                "Ps",
                "kg/m^2",
                zeros,
                description="Precipitation within the step (kg/m²; 1 mm ≡ 1 kg/m²)",
                axes=(WITHIN_STEP,),
                constraint=is_non_negative,
            ),
            Rs=var(
                "Rs",
                "W/m^2",
                zeros,
                description="Shortwave radiation within the step",
                axes=(WITHIN_STEP,),
                constraint=is_non_negative,
            ),
        )

    def processes(self, inputs: Inputs, location: LocationInput, clock: ClockInput):
        series = fetch_openmeteo_hourly(location, clock, inputs.settings, openmeteo_requests.Client())
        ts = jnp.stack([s[0] for s in series], axis=0)
        ps = jnp.stack([s[1] for s in series], axis=0)
        rs = jnp.stack([s[2] for s in series], axis=0)
        return OpenMeteoWeatherProcess(
            name="OpenMeteoWeather",
            Ts=ts,
            Ps=ps,
            Rs=rs,
            delta=clock.delta.value,
        )


@entity()
class OpenMeteoWeatherProcess(Process):
    """Weather process that uses OpenMeteo data (batched on leading axis)."""

    Ts: jnp.ndarray
    Ps: jnp.ndarray
    Rs: jnp.ndarray
    delta: int

    def step(self, clock: Clock, weather: Weather) -> Weather:
        """Advance weather by slicing archive series for the current step."""
        start = jnp.asarray(clock.has).reshape(())
        batch = self.Ts.shape[0]
        width = self.delta
        return Weather(
            name=weather.name,
            Ts=weather.Ts.set(dynamic_slice(self.Ts, (0, start), (batch, width))),
            Ps=weather.Ps.set(dynamic_slice(self.Ps, (0, start + 1), (batch, width))),
            Rs=weather.Rs.set(dynamic_slice(self.Rs, (0, start), (batch, width))),
            constraint=weather.constraint,
        )
