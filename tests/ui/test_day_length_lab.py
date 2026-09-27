"""Behaviour: the day-length lab is a latitude slider and a day-of-year chart."""

from __future__ import annotations

import inspect

import pytest

from orion.ui.day_length_lab import day_length_figure, day_length_hours
from orion.ui.process_explorer import ParamSpec, day_length_lab_input, get_process, run_process_sweep
from orion.ui.reflect import list_input_fields, slider_limits


def test_latitude_is_a_slider_from_south_pole_to_north_pole():
    lab = day_length_lab_input()
    field = next(item for item in list_input_fields(lab) if item.name == "latitude")

    assert field.lower is not None and field.upper is not None
    assert (field.lower, field.upper) == (-90.0, 90.0)
    assert slider_limits(field.lower, field.upper, strict=field.strict) == (-90.0, 90.0)


def test_day_length_series_is_one_point_per_day_and_matches_the_process():
    series = day_length_hours(52.0)
    info = get_process("orion.processes.day_length.AstronomicalDayLengthProcess")
    summer = run_process_sweep(
        info,
        [
            ParamSpec(path="day_length.latitude", mode="value", value=52.0),
            ParamSpec(path="day_length.doy", mode="value", value=172.0),
        ],
    )

    assert [day for day, _hours in series] == list(range(1, 366))
    assert dict(series)[172] == pytest.approx(summer.outputs[0]["day_length.hours"])
    assert dict(series)[172] > dict(series)[355]


def test_equator_stays_near_twelve_hours_and_the_southern_summer_is_in_december():
    equator = dict(day_length_hours(0.0))
    south = dict(day_length_hours(-52.0))

    assert equator[1] == pytest.approx(12.0, abs=0.2)
    assert equator[172] == pytest.approx(12.0, abs=0.2)
    assert south[355] > south[172]


def test_apparent_day_length_is_longer_than_astronomical_and_grows_with_altitude():
    sea = dict(day_length_hours(52.0, apparent=True))
    high = dict(day_length_hours(52.0, altitude=2000.0, apparent=True))
    geometric = dict(day_length_hours(52.0))

    assert sea[172] > geometric[172]
    assert high[172] > sea[172]


def test_chart_plots_day_of_year_against_day_length():
    figure = day_length_figure(52.0)

    assert figure.layout["xaxis"]["title"]["text"] == "Day of year"
    assert figure.layout["yaxis"]["title"]["text"] == "Day length (h)"
    assert list(figure.data[0].x) == list(range(1, 366))
    assert list(figure.data[0].y)[171] == pytest.approx(dict(day_length_hours(52.0))[172])


def test_day_length_page_uses_a_latitude_slider_and_no_run_button():
    import orion.ui.day_length_lab as lab
    import orion.ui.process_tab as tab

    page = inspect.getsource(lab.render_day_length_lab)
    assert "ui.slider" in page
    assert "day_length_figure" in page
    assert "Run / sweep" not in page
    assert "render_day_length_lab" in inspect.getsource(tab.render_process_tab)
