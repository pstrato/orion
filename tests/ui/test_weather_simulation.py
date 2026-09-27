"""Behaviour: a UI simulation shows the Open-Meteo hour window, not the zero placeholder."""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import numpy as np
import openmeteo_requests

from orion.clients.openmeteo import clear_openmeteo_memory_cache
from orion.core.model import model
from orion.processes.weather import Weather
from orion.ui.app import AppState, load_simulation_weather, simulation_inputs, simulation_processes, ui_settings
from orion.ui.configuration import default_configuration
from orion.ui.reflect import run_steps


class _Variable:
    def __init__(self, values: list[float]):
        self._values = np.asarray(values, dtype=np.float64)

    def ValuesAsNumpy(self):
        return self._values


class _Hourly:
    def __init__(self, series: list[list[float]]):
        self._series = series

    def Variables(self, index: int):
        return _Variable(self._series[index])


class _Response:
    def __init__(self, series: list[list[float]]):
        self._series = series

    def Hourly(self):
        return _Hourly(self._series)


class _Client:
    def weather_api(self, url: str, params: dict):
        del url, params
        hours = 48
        return [
            _Response(
                [
                    [float(i) for i in range(hours)],
                    [float(100 + i) for i in range(hours)],
                    [float(200 + i) for i in range(hours)],
                ]
            )
        ]


def test_ui_simulation_weather_outputs_are_the_archive_hour_window(tmp_path, monkeypatch):
    clear_openmeteo_memory_cache()
    monkeypatch.setattr(openmeteo_requests, "Client", lambda *args, **kwargs: _Client())
    state = AppState(
        settings=ui_settings(cache_path=tmp_path),
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        step_hours=3,
    )
    loaded = load_simulation_weather(state.settings, simulation_inputs(state, default_configuration()))
    built = model(default_configuration().name, state.settings, loaded)
    final, _history = run_steps(built, state.settings, loaded, simulation_processes(default_configuration()), 1)
    weather = next(item for item in final.states if isinstance(item, Weather))
    assert jnp.allclose(weather.Ts.value, jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(weather.Ps.value, jnp.array([101.0, 102.0, 103.0]))
    assert jnp.allclose(weather.Rs.value, jnp.array([200.0, 201.0, 202.0]))
