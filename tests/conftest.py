"""Shared fixtures for Orion tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from shapely import Point

from orion.core.input import CropInput, Inputs, LightInterceptionInput, LocationInput, SoilInput, WeatherInput
from orion.core.setting import Settings
from orion.processes.crop import beer_lambert_input, wheat_input
from tests.stubs import StubSoilInput, StubWeatherInput


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(name="test", cache_path=tmp_path)


@pytest.fixture
def location() -> LocationInput:
    return LocationInput("parcel", Point(0.1, 51.5))


@pytest.fixture
def weather_input() -> WeatherInput:
    return StubWeatherInput("stub weather")


@pytest.fixture
def soil_input() -> SoilInput:
    return StubSoilInput("stub soil")


@pytest.fixture
def crop_input() -> CropInput:
    return wheat_input()


@pytest.fixture
def light_interception_input() -> LightInterceptionInput:
    return beer_lambert_input()


@pytest.fixture
def inputs(settings: Settings, location: LocationInput, weather_input: WeatherInput, soil_input: SoilInput, crop_input, light_interception_input) -> Inputs:
    return Inputs(
        name="field run",
        settings=settings,
        location=location,
        start=date(2024, 1, 1),
        end=date(2024, 1, 11),
        step=3,
        weather=weather_input,
        soil=soil_input,
        crop=crop_input,
        light_interception=light_interception_input,
        processes=(),
    )
