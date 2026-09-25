"""UI test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from shapely import Point

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Input, Inputs, LocationInput
from orion.core.model import model
from orion.core.setting import Settings
from orion.core.state import State
from orion.processes.clock import ClockInput
from orion.processes.soil.soil import Soil, make_soil_layer


@entity()
class FixtureSoilInput(Input):
    """One soil layer so the UI can show nested constants and variables."""

    def states(self, *args: Input | State) -> Soil:
        del args
        layer = make_soil_layer("layer-0", 0.0, 0.05)
        return Soil(name="soil", constraint=None, layers=(layer,))

    def processes(self, *args: Input | State) -> None:
        del args
        return None


@pytest.fixture
def clock_model(tmp_path: Path):
    """Clock, location, and one soil layer. No network."""
    clock = ClockInput(
        name="clock",
        start=const("start", "isodate", date(2024, 1, 1), "Simulation start date"),
        end=const("end", "isodate", date(2024, 1, 11), "Simulation end date"),
        delta=const("delta", "hours", 3, "Simulation delta step in hours"),
    )
    location = LocationInput(
        name="parcel",
        geometry=const("geometry", "coordinate", Point(0.1, 51.5), "Parcel geometry"),
    )
    settings = Settings(name="test", cache_path=tmp_path, validate_inputs=False, validate_initial_states=False, validate_simulation_states=False)
    return model("clock only", Inputs(name="field", inputs=(clock, location, FixtureSoilInput("stub soil"))), settings)
