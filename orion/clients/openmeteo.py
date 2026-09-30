from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import openmeteo_requests

from orion.clients.http_cache import cached_session
from orion.core.axis import WITHIN_STEP
from orion.core.input import LocationInput
from orion.core.quantity import is_finite, is_non_negative
from orion.core.setting import Settings
from orion.core.variable import var
from orion.processes.clock import ClockInput
from orion.processes.weather import WeatherInput

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HOURLY = (
    "temperature_2m",
    "precipitation",
    "shortwave_radiation",
)

_hourly_memory: dict[tuple[object, ...], WeatherInput] = {}


def clear_openmeteo_memory_cache() -> None:
    """Drop in-process hourly archives (tests / explicit invalidation)."""
    _hourly_memory.clear()


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


def fetch_openmeteo_input(
    location: LocationInput,
    clock: ClockInput,
    settings: Settings,
    client: openmeteo_requests.Client | None,
):
    """Fetch hourly temperature, precipitation, and shortwave radiation."""
    memory_key = (round(location.geometry.value.y, 5), round(location.geometry.value.x, 5), _iso_date(clock.start.value), _iso_date(clock.end.value))
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
        "start_date": _iso_date(clock.start.value),
        "end_date": _iso_date(clock.end.value + jdt.to_timedelta(1, "D")),
        "hourly": list(OPENMETEO_HOURLY),
    }
    responses = openmeteo.weather_api(OPENMETEO_ARCHIVE_URL, params=params)
    ts, ps, rs = hourly_series_from_response(responses[0])
    input = WeatherInput(
        "OpenMeteo Weather",
        Ts=var("Ts", "°C", ts, description="Air temperature within the step", axes=(WITHIN_STEP,), constraint=is_finite, dimension="heat"),
        Ps=var(
            "Ps",
            "kg/m^2",
            ps,
            description="Precipitation within the step (kg/m²; 1 mm ≡ 1 kg/m²)",
            axes=(WITHIN_STEP,),
            constraint=is_non_negative + is_finite,
            dimension="water",
        ),
        Rs=var(
            "Rs",
            "W/m^2",
            rs,
            description="Shortwave radiation within the step",
            axes=(WITHIN_STEP,),
            constraint=is_non_negative + is_finite,
            dimension="light",
        ),
    )
    _hourly_memory[memory_key] = input
    return input


def _iso_date(value: jdt.Datetime) -> str:
    return value.to_pydatetime().date().isoformat()
