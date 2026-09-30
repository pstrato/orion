"""Behaviour: astronomical day length follows latitude and the clock date."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from shapely import Point

from orion.core.constant import const
from orion.core.input import LocationInput
from orion.core.variable import var
from orion.processes.clock import Clock
from orion.processes.day_length import (
    ApparentDayLengthInput,
    ApparentDayLengthProcess,
    AstronomicalDayLengthInput,
    AstronomicalDayLengthProcess,
    DayLength,
    DayLengthProcess,
    apparent_day_length,
    astronomical_day_length,
)


def _fao_day_length(latitude_deg: float, day_of_year: int) -> float:
    declination = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    latitude = math.radians(latitude_deg)
    argument = min(1.0, max(-1.0, -math.tan(latitude) * math.tan(declination)))
    return 24.0 / math.pi * math.acos(argument)


def _clock(start: date, step: int = 0, delta: int = 24) -> Clock:
    return Clock(
        name="clock",
        constraint=None,
        start=var("start", "isodate", start, "Simulation start date"),
        delta=const("delta", "hours", delta, "Simulation delta step in hours"),
        step=var("step", "step", step, "Current simulation step index"),
    )


def _location(latitude: float, altitude: float = 0.0) -> LocationInput:
    return LocationInput(
        name="location",
        geometry=const("geometry", "coordinate", Point(0.0, latitude), "Location geometry"),
        altitude=const("altitude", "m", altitude, "Altitude above sea level"),
    )


def _apparent_hours(start: date, latitude: float, altitude: float = 0.0) -> float:
    result = ApparentDayLengthProcess("day_length").step(_clock(start), _location(latitude, altitude))
    assert isinstance(result, tuple)
    return float(result[0].hours.value)


def _apparent_formula(latitude_deg: float, day_of_year: int, altitude_m: float) -> float:
    declination = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    latitude = math.radians(latitude_deg)
    height = max(altitude_m, 0.0)
    dip = math.acos(6_371_000.0 / (6_371_000.0 + height))
    solar_elevation = math.radians(-0.833) - dip
    numer = math.sin(solar_elevation) - math.sin(latitude) * math.sin(declination)
    denom = math.cos(latitude) * math.cos(declination)
    if numer >= abs(denom):
        return 0.0
    if numer <= -abs(denom):
        return 24.0
    return 24.0 / math.pi * math.acos(min(1.0, max(-1.0, numer / denom)))


def _hours(start: date, latitude: float, step: int = 0, delta: int = 24) -> float:
    result = AstronomicalDayLengthProcess("day_length").step(_clock(start, step, delta), _location(latitude))
    assert isinstance(result, tuple)
    day_length = result[0]
    assert isinstance(day_length, DayLength)
    return float(day_length.hours.value)


def test_astronomical_day_length_reads_latitude_and_day_of_year_quantities():
    when = date(2024, 6, 21)
    hours = astronomical_day_length(var("latitude", "unitless", 52.0, "Latitude in degrees north"), _clock(when).doy)

    assert hours.unit == "hours"
    assert float(hours.value) == pytest.approx(_fao_day_length(52.0, when.timetuple().tm_yday))


def test_apparent_day_length_reads_the_altitude_quantity():
    when = date(2024, 6, 21)
    hours = apparent_day_length(var("latitude", "unitless", 52.0, "Latitude in degrees north"), _clock(when).doy, const("altitude", "m", 2000.0, "Altitude above sea level"))

    assert hours.unit == "hours"
    assert float(hours.value) == pytest.approx(_apparent_formula(52.0, when.timetuple().tm_yday, 2000.0))


def test_equator_day_length_is_twelve_hours():
    assert _hours(date(2024, 6, 21), 0.0) == pytest.approx(12.0)
    assert _hours(date(2024, 12, 21), 0.0) == pytest.approx(12.0)


def test_a_summer_day_is_longer_than_a_winter_day_at_the_same_latitude():
    summer = _hours(date(2024, 6, 21), 52.0)
    winter = _hours(date(2024, 12, 21), 52.0)
    assert summer > winter
    assert summer == pytest.approx(_fao_day_length(52.0, date(2024, 6, 21).timetuple().tm_yday))
    assert winter == pytest.approx(_fao_day_length(52.0, date(2024, 12, 21).timetuple().tm_yday))


def test_southern_summer_is_longer_in_december():
    december = _hours(date(2024, 12, 21), -35.0)
    june = _hours(date(2024, 6, 21), -35.0)
    assert december > june


def test_polar_day_and_polar_night_span_the_full_day():
    assert _hours(date(2024, 6, 21), 90.0) == pytest.approx(24.0)
    assert _hours(date(2024, 12, 21), 90.0) == pytest.approx(0.0)


def test_day_length_follows_the_clock_across_leap_day_and_new_year():
    cases = (
        (date(2024, 2, 28), 1, 24),
        (date(2024, 12, 31), 1, 24),
        (date(1900, 2, 28), 1, 24),
        (date(2024, 6, 21), 8, 3),
    )
    for start, step, delta in cases:
        when = start + timedelta(hours=step * delta)
        assert _hours(start, 48.0, step, delta) == pytest.approx(_fao_day_length(48.0, when.timetuple().tm_yday))


def test_day_length_input_starts_at_zero_hours_and_builds_the_process():
    created = AstronomicalDayLengthInput("day_length")
    state = created.states()
    assert isinstance(state, DayLength)
    assert float(state.hours.value) == pytest.approx(0.0)
    assert isinstance(created.processes(), AstronomicalDayLengthProcess)


def test_the_abstract_day_length_process_does_not_compute_a_day():
    with pytest.raises(NotImplementedError):
        DayLengthProcess("day_length").step(_clock(date(2024, 6, 21)), _location(0.0))


def test_refraction_lengthens_the_equatorial_day_past_twelve_hours():
    when = date(2024, 6, 21)
    geometric = _hours(when, 0.0)
    apparent = _apparent_hours(when, 0.0)
    assert geometric == pytest.approx(12.0)
    assert apparent > geometric
    assert apparent == pytest.approx(_apparent_formula(0.0, when.timetuple().tm_yday, 0.0))


def test_altitude_lengthens_the_apparent_day_and_ground_below_sea_level_does_not():
    when = date(2024, 6, 21)
    sea = _apparent_hours(when, 52.0, 0.0)
    high = _apparent_hours(when, 52.0, 2000.0)
    below = _apparent_hours(when, 52.0, -400.0)
    assert high > sea
    assert below == pytest.approx(sea)
    assert high == pytest.approx(_apparent_formula(52.0, when.timetuple().tm_yday, 2000.0))


def test_apparent_polar_day_and_night_still_span_the_full_day():
    assert _apparent_hours(date(2024, 6, 21), 90.0) == pytest.approx(24.0)
    assert _apparent_hours(date(2024, 12, 21), 90.0) == pytest.approx(0.0)


def test_apparent_day_length_input_starts_at_zero_and_reads_location_altitude():
    created = ApparentDayLengthInput("day_length")
    state = created.states()
    assert isinstance(state, DayLength)
    assert float(state.hours.value) == pytest.approx(0.0)
    assert isinstance(created.processes(), ApparentDayLengthProcess)
    assert float(_location(52.0).altitude.value) == pytest.approx(0.0)
