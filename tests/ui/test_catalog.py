"""Behaviour: configuration process catalog is discovered from code, grouped by domain."""

from __future__ import annotations

from orion.clients.soilgrids import SoilGridInput
from orion.core.input import Input
from orion.processes.weather import WeatherInput
from orion.ui.catalog import (
    catalog_select_options,
    discover_process_inputs,
    make_soil_input,
    make_weather_input,
    soil_provider_options,
    unimplemented_process_labels,
    weather_provider_options,
)


def test_catalog_skips_horizon_and_provider_inputs():
    labels = {spec.label for spec in discover_process_inputs()}
    assert "Clock" not in labels
    assert "Open Meteo Weather" not in labels
    assert "Soil Grid" not in labels
    assert "Weather" not in labels
    assert "Soil" not in labels


def test_day_length_providers_are_astronomical_and_apparent():
    from orion.processes.day_length import ApparentDayLengthInput, AstronomicalDayLengthInput
    from orion.ui.catalog import implementation_options, make_day_length_input

    assert implementation_options("day_length")["Astronomical"] == "Astronomical"
    assert implementation_options("day_length")["Apparent"] == "Apparent"
    assert isinstance(make_day_length_input("Astronomical"), AstronomicalDayLengthInput)
    assert isinstance(make_day_length_input("Apparent"), ApparentDayLengthInput)


def test_light_interception_provider_is_beer_lambert():
    from orion.processes.crop.light_interception import BeerLambertLightInterceptionInput
    from orion.ui.catalog import implementation_options, make_light_interception_input

    assert implementation_options("light_interception")["Beer-Lambert"] == "Beer-Lambert"
    assert isinstance(make_light_interception_input("Beer-Lambert"), BeerLambertLightInterceptionInput)


def test_provider_catalog_lists_importable_weather_and_soil():
    from orion.processes.clock import ClockInput
    from orion.ui.catalog import implementation_options, make_clock_input

    assert implementation_options("clock")["Clock"] == "Clock"
    assert isinstance(make_clock_input("Clock"), ClockInput)
    assert weather_provider_options()["Open-Meteo"] == "Open-Meteo"
    assert soil_provider_options()["SoilGrids"] == "SoilGrids"
    assert isinstance(make_weather_input("Open-Meteo"), WeatherInput)
    assert isinstance(make_soil_input("SoilGrids"), SoilGridInput)


def test_select_options_follow_discovered_optional_inputs():
    options = catalog_select_options()
    assert set(options) == {spec.label for spec in discover_process_inputs()}


def test_unimplemented_labels_ignore_inputs_outside_the_catalog():
    class LocalInput(Input):
        """Not a packaged process input."""

    assert unimplemented_process_labels((LocalInput("local"),)) == ()
    assert unimplemented_process_labels(()) == ()
