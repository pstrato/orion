"""Behaviour: a model plans process calls once, then steps by state-tuple index."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from pathlib import Path

import pytest

from orion.core.input import Input, Inputs
from orion.core.invoke import Assignment
from orion.core.model import model
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State

_SETTINGS = Settings("settings", Path("."), False, False, False)


def _model(inputs: Inputs, processes: tuple[Process, ...] = (), settings: Settings = _SETTINGS):
    return model("run", settings, inputs, processes)


class Clock(State):
    pass


class Reading(State):
    pass


class ClockInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class ReadingInput(Input):
    def states(self) -> Reading:
        return Reading("reading", None)


class Tick(Process):
    def step(self, clock: Clock) -> Clock:
        return Clock(clock.name + "1", None)


class Tock(Process):
    def step(self, clock: Clock, reading: Reading) -> tuple[Clock, Reading]:
        return Clock(clock.name + "2", None), Reading(reading.name + "2", None)


class Holds(Process):
    def step(self, clock: Clock) -> None:
        return None


class HoldInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class UsesAnyState(Process):
    def step(self, state: State) -> Clock:
        return Clock(state.name + "!", None)


class OnlyClock(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class UsesState(Process):
    def step(self, state: State) -> None:
        return None


def test_processes_run_in_the_order_they_are_given_and_state_positions_stay_fixed():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (Tick("tick"), Tock("tock")))
    assert [type(state) for state in built.states] == [Clock, Reading]
    assert [type(process) for process in built.processes] == [Tick, Tock]
    assert built.process_arguments[0] == (0,)
    assert built.process_arguments[1] == (0, 1)

    stepped = built.step()
    assert stepped is not built
    assert stepped.state_classes is built.state_classes
    assert [type(state) for state in stepped.states] == [Clock, Reading]
    assert stepped.states[0].name == "clock12"
    assert stepped.states[1].name == "reading2"
    assert built.states[0].name == "clock"
    assert built.states[1] is not stepped.states[1]


def test_a_process_that_returns_nothing_leaves_the_state_tuple_unchanged():
    built = _model(Inputs("run", (HoldInput("hold-input"),)), (Holds("hold"),))
    assert built.process_assignments == (Assignment((), False),)
    clock = built.states[0]
    stepped = built.step()
    assert stepped.states == (clock,)


def test_a_base_state_matches_the_only_concrete_state():
    built = _model(Inputs("run", (OnlyClock("clock-input"),)), (UsesAnyState("uses"),))
    assert built.step().states[0].name == "clock!"


def test_a_base_type_with_two_concrete_states_is_rejected_when_the_model_is_built():
    with pytest.raises(ValueError, match="more than one concrete type"):
        _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (UsesState("ambiguous"),))


class Boxes(Process):
    def step(self, clock: Clock) -> tuple[Clock]:
        return (Clock(clock.name + "b", None),)


class BoxInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class Reverse(Process):
    def step(self, clock: Clock, reading: Reading) -> tuple[Reading, Clock]:
        return Reading(reading.name + "r", None), Clock(clock.name + "c", None)


class ReturnsState(Process):
    def step(self, clock: Clock) -> State:
        return Clock(clock.name + "!", None)


class ReturnsStateInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)


class ReturnsEither(Process):
    def step(self) -> State:
        return Clock("x", None)


class UsesInput(Process):
    def step(self, source: ClockInput) -> None:
        return None


def test_return_assignments_are_planned_from_the_step_signature():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (Tick("tick"), Tock("tock")))
    assert built.process_assignments == (Assignment((0,), False), Assignment((0, 1), True))


def test_a_one_state_tuple_is_unpacked_at_the_planned_index():
    built = _model(Inputs("run", (BoxInput("box-input"),)), (Boxes("box"),))
    assert built.process_assignments == (Assignment((0,), True),)
    assert built.step().states[0].name == "clockb"


def test_a_tuple_return_is_written_in_signature_order():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (Tick("tick"), Tock("tock"), Reverse("reverse")))
    assert built.process_assignments[2] == Assignment((1, 0), True)
    stepped = built.step()
    assert stepped.states[0].name == "clock12c"
    assert stepped.states[1].name == "reading2r"


def test_a_base_return_type_matches_the_only_concrete_state():
    built = _model(Inputs("run", (ReturnsStateInput("returns-input"),)), (ReturnsState("returns"),))
    assert built.process_assignments == (Assignment((0,), False),)
    assert built.step().states[0].name == "clock!"


def test_a_base_return_type_with_two_concrete_states_is_rejected_when_the_model_is_built():
    with pytest.raises(ValueError, match="more than one concrete type"):
        _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (ReturnsEither("either"),))


def test_a_step_argument_must_be_a_state():
    with pytest.raises(TypeError, match="is not a state"):
        _model(Inputs("run", (ClockInput("clock-input"),)), (UsesInput("uses"),))


class LeafInput(Input):
    def states(self) -> Reading:
        return Reading("from-leaf", None)


class PlantInput(Input):
    leaf: LeafInput

    def states(self, leaf: LeafInput) -> Clock:
        return Clock(leaf.name, None)


class NeedsContext(Input):
    def states(self, box: Inputs, settings: Settings) -> Clock:
        return Clock(f"{box.name}:{settings.name}", None)


def test_a_nested_input_is_available_and_is_invoked_with_its_parent():
    built = _model(Inputs("run", (PlantInput("plant", LeafInput("leaf")),)))
    assert [type(state) for state in built.states] == [Clock, Reading]
    assert built.states[0].name == "leaf"
    assert built.states[1].name == "from-leaf"
    assert built.processes == ()


def test_inputs_and_settings_are_available_when_states_are_created():
    settings = Settings("settings", Path("cache"), False, False, False)
    box = Inputs("run", (NeedsContext("needs"),))
    built = _model(box, (), settings)
    assert built.settings is settings
    assert built.states[0].name == "run:settings"


def test_a_step_does_not_parse_annotations_again(monkeypatch):
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))), (Tick("tick"), Tock("tock")))

    def fail(*args, **kwargs):
        raise AssertionError("annotations parsed during simulation")

    monkeypatch.setattr("orion.core.invoke.get_annotations", fail)
    monkeypatch.setattr("orion.core.model.get_annotations", fail, raising=False)
    stepped = built.step()
    assert stepped.states[0].name == "clock12"
