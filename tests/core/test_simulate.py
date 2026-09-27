"""Behaviour: a simulation runs to the end of the clock, and a batch runs in parallel."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import jax.numpy as jnp
import pytest

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Input, Inputs
from orion.core.model import axes_of, simulate, simulate_all, validate
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State
from orion.core.variable import Variable, var
from orion.processes.clock import Clock, ClockInput, ClockProcess

_SETTINGS = Settings("settings", Path("."), False, False, False)


def _clock() -> ClockInput:
    return ClockInput(
        "clock",
        var("start", "isodate", date(2024, 1, 1), description="start"),
        var("end", "isodate", date(2024, 1, 3), description="end"),
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


def _inputs(rate: float, name: str) -> Inputs:
    return Inputs(name, (_clock(), Gain("gain", var("rate", "1", rate, description="rate"))))


def _processes() -> tuple[Process, ...]:
    return (ClockProcess("clock"), Accumulate("accumulate"))


def test_simulate_advances_the_clock_to_the_end_of_its_horizon():
    finished, history = simulate(_SETTINGS, _inputs(1.0, "field"), _processes())
    assert history is None
    clock = next(state for state in finished.states if isinstance(state, Clock))
    total = next(state for state in finished.states if isinstance(state, Total))
    assert int(clock.step.value) == 2
    assert float(total.amount.value) == pytest.approx(2.0)


def test_compiled_simulation_reaches_the_horizon_when_state_validation_is_on():
    settings = Settings("settings", Path("."), False, False, True)
    validate(settings, _inputs(1.0, "field"), _processes())
    finished, _history = simulate(settings, _inputs(1.0, "field"), _processes(), jit=True)
    clock = next(state for state in finished.states if isinstance(state, Clock))
    assert int(clock.step.value) == 2


def test_simulate_jit_reaches_the_same_clock_horizon():
    finished, _history = simulate(_SETTINGS, _inputs(1.0, "field"), _processes(), jit=True)
    clock = next(state for state in finished.states if isinstance(state, Clock))
    total = next(state for state in finished.states if isinstance(state, Total))
    assert int(clock.step.value) == 2
    assert float(total.amount.value) == pytest.approx(2.0)


def test_simulate_all_batches_one_result_per_inputs():
    finished, history = simulate_all(_SETTINGS, (_inputs(1.0, "slow"), _inputs(3.0, "fast")), _processes())
    assert history is None
    clock = next(state for state in finished.states if isinstance(state, Clock))
    total = next(state for state in finished.states if isinstance(state, Total))
    assert jnp.allclose(clock.step.value, jnp.array([2.0, 2.0]))
    assert jnp.allclose(total.amount.value, jnp.array([2.0, 6.0]))


def test_simulate_all_jit_batches_one_result_per_inputs():
    finished, _history = simulate_all(_SETTINGS, (_inputs(1.0, "slow"), _inputs(3.0, "fast")), _processes(), jit=True)
    total = next(state for state in finished.states if isinstance(state, Total))
    assert jnp.allclose(total.amount.value, jnp.array([2.0, 6.0]))


def _clock_state(model):
    return next(state for state in model.states if isinstance(state, Clock))


def test_a_single_simulation_final_state_has_no_axis():
    finished, history = simulate(_SETTINGS, _inputs(1.0, "field"), _processes())
    assert history is None
    assert finished.axes == ()
    assert _clock_state(finished).step.value.shape == ()


def test_a_single_simulation_history_is_indexed_by_step():
    final, history = simulate(_SETTINGS, _inputs(1.0, "field"), _processes(), keep_history=True)
    assert final.axes == ()
    assert history is not None
    assert [item.name for item in history.axes] == ["step"]
    assert history.axes[0].values == (1, 2)
    assert _clock_state(history).step.value.shape == (2,)


def test_one_inputs_final_state_is_indexed_by_that_inputs():
    group = (_inputs(1.0, "only"),)
    finished, history = simulate_all(_SETTINGS, group, _processes())
    assert history is None
    assert [item.name for item in finished.axes] == ["inputs"]
    assert finished.axes[0].values == group
    assert _clock_state(finished).step.value.shape == (1,)


def test_several_inputs_final_state_is_indexed_by_those_inputs():
    groups = (_inputs(1.0, "slow"), _inputs(3.0, "fast"))
    finished, _history = simulate_all(_SETTINGS, groups, _processes())
    assert [item.name for item in finished.axes] == ["inputs"]
    assert finished.axes[0].values == groups
    assert _clock_state(finished).step.value.shape == (2,)


def test_history_of_one_inputs_is_indexed_by_inputs_then_step():
    group = (_inputs(1.0, "only"),)
    final, history = simulate_all(_SETTINGS, group, _processes(), keep_history=True)
    assert [item.name for item in final.axes] == ["inputs"]
    assert history is not None
    assert [item.name for item in history.axes] == ["inputs", "step"]
    assert history.axes[0].values == group
    assert history.axes[1].values == (1, 2)
    assert _clock_state(history).step.value.shape == (1, 2)


def test_history_of_several_inputs_is_indexed_by_inputs_then_step():
    groups = (_inputs(1.0, "slow"), _inputs(3.0, "fast"))
    final, history = simulate_all(_SETTINGS, groups, _processes(), keep_history=True)
    assert [item.name for item in final.axes] == ["inputs"]
    assert final.axes[0].values == groups
    assert history is not None
    assert _clock_state(history).step.axes == ()
    assert [item.name for item in axes_of(history, _clock_state(history).step)] == ["inputs", "step"]
    assert history.axes[1].values == (1, 2)
    assert _clock_state(history).step.value.shape == (2, 2)


def test_simulate_all_keeps_each_inputs_own_start_date():
    def group(start: date, name: str) -> Inputs:
        return Inputs(
            name,
            (
                ClockInput(
                    "clock",
                    var("start", "isodate", start, description="start"),
                    var("end", "isodate", start + timedelta(days=2), description="end"),
                    const("delta", "hours", 24, description="step"),
                ),
                Gain("gain", var("rate", "1", 1.0, description="rate")),
            ),
        )

    early = date(2024, 1, 1)
    late = date(2024, 6, 1)
    finished, _history = simulate_all(_SETTINGS, (group(early, "early"), group(late, "late")), _processes())
    got = _clock_state(finished).date.value.delta.days
    expected = jnp.array([(early + timedelta(days=2) - date(1970, 1, 1)).days, (late + timedelta(days=2) - date(1970, 1, 1)).days])
    assert jnp.array_equal(got, expected)


def test_simulate_requires_a_clock():
    with pytest.raises(ValueError, match="no clock"):
        simulate(_SETTINGS, Inputs("bare", ()), ())


def test_simulate_all_requires_the_same_clock_horizon():
    short = ClockInput(
        "clock",
        var("start", "isodate", date(2024, 1, 1), description="start"),
        var("end", "isodate", date(2024, 1, 2), description="end"),
        const("delta", "hours", 24, description="step"),
    )
    with pytest.raises(ValueError, match="same clock horizon"):
        simulate_all(_SETTINGS, (_inputs(1.0, "slow"), Inputs("short", (short, Gain("gain", var("rate", "1", 3.0, description="rate"))))), _processes())
