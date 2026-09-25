"""Offline weather, soil, and crop inputs for tests without network."""

from __future__ import annotations

from orion.clients.openmeteo import weather_state_for_step
from orion.core.input import CropInput, Inputs, SoilInput, WeatherInput
from orion.core.jax import entity
from orion.core.process import Process
from orion.core.state import State
from orion.processes.clock import Clock
from orion.processes.crop import make_crop
from orion.processes.soil import soil_from_layer_bottoms
from orion.processes.weather import Weather, WeatherProcess


@entity()
class StaticWeatherProcess(WeatherProcess):
    """Leaves weather unchanged each step (offline stub)."""

    def step(self, clock: Clock, weather: Weather) -> Weather:
        return weather


@entity()
class StubWeatherInput(WeatherInput):
    """Weather input that does not call external APIs."""

    def states(self, inputs: Inputs) -> tuple[State, ...]:
        return (weather_state_for_step(inputs.step, inputs.batch_size),)

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return (StaticWeatherProcess(self.name),)


@entity()
class StubSoilInput(SoilInput):
    """Soil input that builds a simple layered profile offline."""

    def states(self, inputs: Inputs) -> tuple[State, ...]:
        return (soil_from_layer_bottoms((0.05, 0.15, 0.30), batch_size=inputs.batch_size),)

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return ()


@entity()
class StubCropInput(CropInput):
    """Crop input that builds a default crop state offline."""

    def states(self, inputs: Inputs) -> tuple[State, ...]:
        return (make_crop(inputs),)

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return ()
