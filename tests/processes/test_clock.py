"""Behaviour: a clock input carries its dates as variables, and the state exposes the current date."""

from __future__ import annotations

from datetime import date

import jax.numpy as jnp

from orion.core.constant import const
from orion.core.variable import Variable, var
from orion.processes.clock import Clock, ClockInput, ClockProcess


def _input(start: date, end: date, delta: int = 24) -> ClockInput:
    return ClockInput(
        "clock",
        var("start", "isodate", start, "Simulation start date"),
        var("end", "isodate", end, "Simulation end date"),
        const("delta", "hours", delta, "Simulation delta step in hours"),
    )


def _clock(start: date, step: int = 0, delta: int = 24) -> Clock:
    created = _input(start, date(start.year + 1, 1, 1), delta).states()
    assert isinstance(created, Clock)
    return Clock(name=created.name, constraint=created.constraint, start=created.start, delta=created.delta, step=created.step.set(jnp.asarray(step)))


def test_clock_input_stores_start_and_end_as_variables():
    created = _input(date(2024, 1, 1), date(2024, 1, 3))

    assert isinstance(created.start, Variable)
    assert isinstance(created.end, Variable)
    assert created.start.value.to_pydatetime().date() == date(2024, 1, 1)
    assert created.end.value.to_pydatetime().date() == date(2024, 1, 3)


def test_clock_date_and_day_of_year_are_variables_of_the_current_step():
    clock = _clock(date(2024, 6, 21), step=1, delta=24)

    assert isinstance(clock.date, Variable)
    assert isinstance(clock.doy, Variable)
    assert clock.date.value.to_pydatetime().date() == date(2024, 6, 22)
    assert int(clock.doy.value) == date(2024, 6, 22).timetuple().tm_yday


def test_day_of_year_crosses_leap_day_and_the_new_year():
    cases = (
        (date(2024, 2, 28), 1, 24, date(2024, 2, 29)),
        (date(1900, 2, 28), 1, 24, date(1900, 3, 1)),
        (date(2024, 12, 31), 1, 24, date(2025, 1, 1)),
        (date(2024, 6, 21), 8, 3, date(2024, 6, 22)),
    )
    for start, step, delta, expected in cases:
        clock = _clock(start, step, delta)
        assert clock.date.value.to_pydatetime().date() == expected
        assert int(clock.doy.value) == expected.timetuple().tm_yday


def test_hours_after_start_is_a_quantity_of_hours():
    clock = _clock(date(2024, 6, 21), step=2, delta=3)

    assert isinstance(clock.has, Variable)
    assert clock.has.unit == "hours"
    assert float(clock.has.value) == 6.0


def test_the_clock_process_advances_the_date_by_one_step():
    clock = _clock(date(2024, 1, 1))
    advanced = ClockProcess("clock").step(clock)

    assert advanced.date.value.to_pydatetime().date() == date(2024, 1, 2)
    assert int(advanced.doy.value) == 2
