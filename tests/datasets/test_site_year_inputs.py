"""Behaviour: site-year horizon and Inputs built for dataset simulation."""

from __future__ import annotations

from datetime import date

import pytest

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Input, Inputs, LocationInput
from orion.core.variable import var
from orion.datasets.convert import to_kg_per_m2
from orion.datasets.events import fertiliser_event, protection_event, sowing_event
from orion.datasets.inputs import inputs_for_site_year
from orion.datasets.io import location_from_row, point_location
from orion.datasets.observations import make_observation
from orion.datasets.site_year import SiteYear
from orion.processes.clock import ClockInput


@entity()
class KeptInput(Input):
    """Template input that site-year assembly must keep."""


def _site() -> SiteYear:
    return SiteYear(
        name="westerfeld:2018:intensive",
        crop="wheat",
        cultivar="RGT Reform",
        season="2018",
        treatment="intensive",
        source="westerfeld",
        location=point_location("westerfeld", 11.702, 51.819),
        events=(
            sowing_event("sowing", date(2018, 10, 12), density=350.0),
            fertiliser_event("fertiliser", date(2019, 4, 10), n=to_kg_per_m2(80.0, "kg/ha")),
            protection_event("protection", date(2019, 5, 20), kind="fungicide", product="Prosaro"),
        ),
        observations=(
            make_observation("yield", date(2019, 7, 25), "yield", "kg/m^2", 0.85),
            make_observation("grain_protein", date(2019, 7, 25), "grain_protein", "kg/kg", 0.125),
        ),
    )


def test_site_year_horizon_runs_from_sowing_through_day_after_last_observation():
    start, end = _site().horizon()
    assert start == date(2018, 10, 12)
    assert end == date(2019, 7, 26)
    assert (end - start).days == 287


def test_site_year_horizon_is_at_least_one_day_when_only_sowing_is_dated():
    site = SiteYear(
        name="sowing-only",
        crop="wheat",
        cultivar="",
        season="2020",
        treatment="a",
        source="demo",
        location=point_location("demo", 0.0, 0.0),
        events=(sowing_event("sowing", date(2020, 3, 1)), protection_event("protection", date(2020, 3, 1), kind="standard")),
        observations=(),
    )
    start, end = site.horizon()
    assert start == date(2020, 3, 1)
    assert end == date(2020, 3, 2)


def test_inputs_for_site_year_use_dataset_location_and_horizon():
    template = Inputs(
        "field input",
        (
            ClockInput(
                "clock",
                var("start", "isodate", date(2024, 1, 1), description="start"),
                var("end", "isodate", date(2024, 1, 11), description="end"),
                const("delta", "hours", 24, description="step"),
            ),
            point_location("field", 0.1, 51.5),
            KeptInput("kept"),
        ),
    )
    site = _site()
    built = inputs_for_site_year(template, site)
    location = next(item for item in built.inputs if isinstance(item, LocationInput))
    clock = next(item for item in built.inputs if isinstance(item, ClockInput))
    kept = next(item for item in built.inputs if isinstance(item, KeptInput))
    assert built.name == site.name
    assert location.geometry.value.x == pytest.approx(11.702)
    assert location.geometry.value.y == pytest.approx(51.819)
    assert clock.start.value.to_pydatetime().date() == date(2018, 10, 12)
    assert clock.end.value.to_pydatetime().date() == date(2019, 7, 26)
    assert clock.delta.value == 24
    assert kept.name == "kept"


def test_a_site_location_keeps_its_altitude():
    location = point_location("westerfeld", 11.702, 51.819, 180.0)
    assert float(location.altitude.value) == pytest.approx(180.0)
    from_row = location_from_row({"latitude": "51.8", "longitude": "11.7", "elevation": "95"}, point_location("fallback", 0.0, 0.0, 10.0))
    assert float(from_row.altitude.value) == pytest.approx(95.0)
    assert float(point_location("sea", 0.0, 0.0).altitude.value) == pytest.approx(100.0)


def test_coordinates_without_an_elevation_keep_the_fallback_altitude():
    from_row = location_from_row({"latitude": "51.8", "longitude": "11.7"}, point_location("fallback", 1.0, 2.0, 10.0))
    assert from_row.geometry.value.x == pytest.approx(11.7)
    assert from_row.geometry.value.y == pytest.approx(51.8)
    assert float(from_row.altitude.value) == pytest.approx(10.0)


def test_inputs_for_site_year_adds_clock_and_location_when_the_template_has_neither():
    built = inputs_for_site_year(Inputs("bare", (KeptInput("kept"),)), _site())
    clock = built.inputs[0]
    assert isinstance(clock, ClockInput)
    assert clock.start.value.to_pydatetime().date() == date(2018, 10, 12)
    assert clock.end.value.to_pydatetime().date() == date(2019, 7, 26)
    assert any(isinstance(item, LocationInput) and item.name == "westerfeld" for item in built.inputs)
    assert any(isinstance(item, KeptInput) for item in built.inputs)
