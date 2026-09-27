"""Behaviour: methods are called with the single instance of each concrete argument type."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from pathlib import Path

import pytest

from orion.core.input import Input, Inputs
from orion.core.invoke import invoke_input_states, invoke_process_step
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State


class Clock(State):
    pass


class Reading(State):
    source: Clock | None = None


class ClockInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class ReadingInput(Input):
    def states(self, clock: Clock) -> Reading:
        return Reading("reading", None, clock)


class NeedsClockFirst(Input):
    def states(self, clock: Clock) -> Reading:
        return Reading("reading", None, clock)


class AlsoClock(Input):
    def states(self) -> Clock:
        return Clock("other", None)


class ReadsClockInput(Input):
    def states(self, clock_input: ClockInput) -> Reading:
        return Reading(clock_input.name, None)


class Marker(Process):
    label: str = ""

    def step(self, clock: Clock, source: ClockInput) -> Clock:
        return Clock(f"{source.name}:{clock.name}", None)


def test_states_from_an_earlier_input_are_passed_by_concrete_type():
    clock_input = ClockInput("clock-input")
    states = invoke_input_states([clock_input, ReadingInput("reading-input")])
    clock, reading = states
    assert isinstance(clock, Clock)
    assert isinstance(reading, Reading)
    assert reading.source is clock


def test_states_from_a_later_input_are_not_available_yet():
    with pytest.raises(ValueError, match="Clock"):
        invoke_input_states([NeedsClockFirst("reading-input"), ClockInput("clock-input")])


def test_an_input_is_passed_to_states_by_its_concrete_type():
    clock_input = ClockInput("clock-input")
    states = invoke_input_states([ReadsClockInput("reading-input"), clock_input])
    assert states[0].name == "clock-input"


def test_a_second_input_of_the_same_concrete_type_is_rejected():
    with pytest.raises(ValueError, match="ClockInput"):
        invoke_input_states([ClockInput("one"), ClockInput("two")])


def test_a_second_state_of_the_same_concrete_type_is_rejected():
    with pytest.raises(ValueError, match="Clock"):
        invoke_input_states([ClockInput("clock-input"), AlsoClock("again")])


class NeedsAnyState(Input):
    def states(self, state: State) -> Reading:
        return Reading(state.name, None)


def test_a_base_type_matches_the_only_concrete_instance():
    states = invoke_input_states([ClockInput("clock-input"), NeedsAnyState("needs")])
    assert states[1].name == "clock"


class OptionalClock(Input):
    def states(self, clock: Clock | None = None) -> Reading:
        assert clock is not None
        return Reading(clock.name, None, clock)


def test_an_invoke_parameter_cannot_have_a_default():
    with pytest.raises(TypeError, match="cannot have a default"):
        invoke_input_states([ClockInput("clock-input"), OptionalClock("optional")])


def test_process_step_receives_the_input_and_the_state_it_requires():
    clock_input = ClockInput("clock-input")
    clock = Clock("clock", None)
    settings = Settings("settings", Path("."), False, False, False)
    updated = invoke_process_step(settings, Inputs("run", (clock_input,)), (clock,), Marker("marker"))
    assert isinstance(updated, Clock)
    assert updated.name == "clock-input:clock"
    assert clock.name == "clock"
    assert updated is not clock
