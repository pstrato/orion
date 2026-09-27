"""Behaviour: Simulate finishes every inputs, batching those that share constants."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Input, Inputs
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State
from orion.core.variable import Variable, var
from orion.processes.clock import Clock, ClockInput, ClockProcess
from orion.ui.simulate import finish_inputs

_SETTINGS = Settings("settings", Path("."), False, False, False)


def _clock(end: date) -> ClockInput:
    return ClockInput(
        "clock",
        const("start", "isodate", date(2024, 1, 1), description="start"),
        const("end", "isodate", end, description="end"),
        const("delta", "hours", 24, description="step"),
    )


@entity()
class Total(State):
    amount: Variable


@entity()
class Gain(Input):
    rate: Variable

    def states(self) -> Total:
        return Total("total", None, var("amount", "1", 0.0, description="amount"))


@entity()
class Accumulate(Process):
    def step(self, total: Total, gain: Gain) -> Total:
        return Total(total.name, None, total.amount.set(total.amount.value + gain.rate.value))


def _processes():
    return (ClockProcess("clock"), Accumulate("accumulate"))


def _inputs(rate: float, name: str, *, end: date = date(2024, 1, 3)) -> Inputs:
    return Inputs(name, (_clock(end), Gain("gain", var("rate", "1", rate, description="rate"))))


def _clock_step(model) -> int:
    clock = next(state for state in model.states if isinstance(state, Clock))
    return int(clock.step.value)


def _amount(model) -> float:
    total = next(state for state in model.states if isinstance(state, Total))
    return float(total.amount.value)


def test_inputs_that_share_constants_finish_with_their_own_results():
    processes = _processes()
    finished = _finals(finish_inputs(_SETTINGS, ((_inputs(1.0, "slow"), processes), (_inputs(3.0, "fast"), processes))))

    assert [model.name for model in finished] == ["slow", "fast"]
    assert [_clock_step(model) for model in finished] == [2, 2]
    assert [_amount(model) for model in finished] == pytest.approx([2.0, 6.0])


def test_inputs_with_different_constants_each_finish_on_their_own_horizon():
    processes = _processes()
    long = _inputs(1.0, "long", end=date(2024, 1, 3))
    short = _inputs(1.0, "short", end=date(2024, 1, 2))
    finished = _finals(finish_inputs(_SETTINGS, ((long, processes), (short, processes))))

    assert [_clock_step(model) for model in finished] == [2, 1]
    assert [_amount(model) for model in finished] == pytest.approx([2.0, 1.0])


def test_a_finished_run_keeps_one_clock_value_for_every_step():
    processes = _processes()
    pairs = finish_inputs(_SETTINGS, ((_inputs(1.0, "slow"), processes), (_inputs(3.0, "fast"), processes)))
    assert len(pairs) == 2
    for final, history in pairs:
        clock = next(state for state in history.states if isinstance(state, Clock))
        assert final.axes == ()
        assert [item.name for item in history.axes] == ["step"]
        assert history.axes[0].values == (1, 2)
        assert _clock_step(final) == 2
        assert [int(value) for value in clock.step.value.reshape(-1).tolist()] == [1, 2]


def _finals(pairs: tuple[tuple, ...]):
    return tuple(final for final, _history in pairs)
