"""UI test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest
from shapely import Point

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Input, Inputs, LocationInput
from orion.core.model import Model, model
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State
from orion.core.variable import var
from orion.processes.clock import ClockInput, ClockProcess
from orion.processes.soil.soil import Soil, make_soil_layer
from orion.ui.reflect import run_steps


@dataclass(frozen=True)
class ClockRun:
    """A clock model plus the settings, inputs, and processes each step needs."""

    model: Model
    settings: Settings
    inputs: Inputs
    processes: tuple[Process, ...]

    @property
    def states(self):
        return self.model.states

    def run(self, steps: int):
        return run_steps(self.model, self.settings, self.inputs, self.processes, steps)


@entity()
class FixtureSoilInput(Input):
    """One soil layer so the UI can show nested constants and variables."""

    def states(self, *args: Input | State) -> Soil:
        del args
        layer = make_soil_layer("layer-0", 0.0, 0.05)
        return Soil(name="soil", constraint=None, layers=(layer,))


@pytest.fixture
def clock_model(tmp_path: Path):
    """Clock, location, and one soil layer. No network."""
    clock = ClockInput(
        name="clock",
        start=var("start", "isodate", date(2024, 1, 1), "Simulation start date"),
        end=var("end", "isodate", date(2024, 1, 11), "Simulation end date"),
        delta=const("delta", "hours", 3, "Simulation delta step in hours"),
    )
    location = LocationInput(
        name="parcel",
        geometry=const("geometry", "coordinate", Point(0.1, 51.5), "Parcel geometry"),
    )
    settings = Settings(name="test", cache_path=tmp_path, validate_inputs=False, validate_initial_states=False, validate_simulation_states=False)
    inputs = Inputs(name="field", inputs=(clock, location, FixtureSoilInput("stub soil")))
    processes = (ClockProcess("clock"),)
    return ClockRun(model("clock only", settings, inputs), settings, inputs, processes)
