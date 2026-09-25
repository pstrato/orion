"""Behaviour: a model plans process calls once, then steps by state-tuple index."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from pathlib import Path

import pytest

from orion.core.input import Input, Inputs
from orion.core.invoke import Argument, Assignment
from orion.core.model import model
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State

_SETTINGS = Settings("settings", Path("."), False, False, False)


def _model(inputs: Inputs, settings: Settings = _SETTINGS):
    return model("run", inputs, settings)


class Clock(State):
    pass


class Reading(State):
    pass


class ClockInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)

    def processes(self) -> Tick:
        return Tick("tick")


class ReadingInput(Input):
    def states(self) -> Reading:
        return Reading("reading", None)

    def processes(self) -> Tock:
        return Tock("tock")


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

    def processes(self) -> Holds:
        return Holds("hold")


class UsesAnyState(Process):
    def step(self, state: State) -> Clock:
        return Clock(state.name + "!", None)


class OnlyClock(Input):
    def states(self) -> Clock:
        return Clock("clock", None)

    def processes(self) -> UsesAnyState:
        return UsesAnyState("uses")


class UsesState(Process):
    def step(self, state: State) -> None:
        return None


class Ambiguous(Input):
    def processes(self) -> UsesState:
        return UsesState("ambiguous")


def test_processes_run_in_creation_order_and_state_positions_stay_fixed():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))))
    assert [type(state) for state in built.states] == [Clock, Reading]
    assert [type(process) for process in built.processes] == [Tick, Tock]
    assert built.process_arguments[0] == (Argument(0),)
    assert built.process_arguments[1] == (Argument(0), Argument(1))

    stepped = built.step()
    assert stepped is not built
    assert stepped.state_classes is built.state_classes
    assert [type(state) for state in stepped.states] == [Clock, Reading]
    assert stepped.states[0].name == "clock12"
    assert stepped.states[1].name == "reading2"
    assert built.states[0].name == "clock"
    assert built.states[1] is not stepped.states[1]


def test_a_process_that_returns_nothing_leaves_the_state_tuple_unchanged():
    built = _model(Inputs("run", (HoldInput("hold-input"),)))
    assert built.process_assignments == (Assignment((), False),)
    clock = built.states[0]
    stepped = built.step()
    assert stepped.states == (clock,)


def test_a_base_state_matches_the_only_concrete_state():
    built = _model(Inputs("run", (OnlyClock("clock-input"),)))
    assert built.step().states[0].name == "clock!"


def test_a_base_type_with_two_concrete_states_is_rejected_when_the_model_is_built():
    with pytest.raises(ValueError, match="more than one concrete type"):
        _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"), Ambiguous("ambiguous"))))


class Boxes(Process):
    def step(self, clock: Clock) -> tuple[Clock]:
        return (Clock(clock.name + "b", None),)


class BoxInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)

    def processes(self) -> Boxes:
        return Boxes("box")


class Reverse(Process):
    def step(self, clock: Clock, reading: Reading) -> tuple[Reading, Clock]:
        return Reading(reading.name + "r", None), Clock(clock.name + "c", None)


class ReverseInput(Input):
    def processes(self) -> Reverse:
        return Reverse("reverse")


class ReturnsState(Process):
    def step(self, clock: Clock) -> State:
        return Clock(clock.name + "!", None)


class ReturnsStateInput(Input):
    def states(self) -> Clock:
        return Clock("clock", None)

    def processes(self) -> ReturnsState:
        return ReturnsState("returns")


class ReturnsEither(Process):
    def step(self) -> State:
        return Clock("x", None)


class ReturnsEitherInput(Input):
    def processes(self) -> ReturnsEither:
        return ReturnsEither("either")


class UsesInput(Process):
    def step(self, source: ClockInput) -> None:
        return None


class UsesInputInput(Input):
    def processes(self, source: ClockInput) -> UsesInput:
        return UsesInput("uses")


def test_return_assignments_are_planned_from_the_step_signature():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))))
    assert built.process_assignments == (Assignment((0,), False), Assignment((0, 1), True))


def test_a_one_state_tuple_is_unpacked_at_the_planned_index():
    built = _model(Inputs("run", (BoxInput("box-input"),)))
    assert built.process_assignments == (Assignment((0,), True),)
    assert built.step().states[0].name == "clockb"


def test_a_tuple_return_is_written_in_signature_order():
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"), ReverseInput("reverse"))))
    assert built.process_assignments[2] == Assignment((1, 0), True)
    stepped = built.step()
    assert stepped.states[0].name == "clock12c"
    assert stepped.states[1].name == "reading2r"


def test_a_base_return_type_matches_the_only_concrete_state():
    built = _model(Inputs("run", (ReturnsStateInput("returns-input"),)))
    assert built.process_assignments == (Assignment((0,), False),)
    assert built.step().states[0].name == "clock!"


def test_a_base_return_type_with_two_concrete_states_is_rejected_when_the_model_is_built():
    with pytest.raises(ValueError, match="more than one concrete type"):
        _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"), ReturnsEitherInput("either"))))


def test_a_step_argument_must_be_a_state():
    with pytest.raises(TypeError, match="is not a state"):
        _model(Inputs("run", (ClockInput("clock-input"), UsesInputInput("uses-input"))))


class LeafInput(Input):
    def states(self) -> Reading:
        return Reading("from-leaf", None)

    def processes(self) -> LeafMark:
        return LeafMark("leaf-mark")


class PlantInput(Input):
    leaf: LeafInput

    def states(self, leaf: LeafInput) -> Clock:
        return Clock(leaf.name, None)

    def processes(self, leaf: LeafInput) -> PlantMark:
        return PlantMark(leaf.name)


class PlantMark(Process):
    def step(self, clock: Clock) -> None:
        return None


class LeafMark(Process):
    def step(self, clock: Clock) -> None:
        return None


class Noted(Process):
    note: str = ""

    def step(self, clock: Clock) -> None:
        return None


class NeedsContext(Input):
    def states(self, box: Inputs, settings: Settings) -> Clock:
        return Clock(f"{box.name}:{settings.name}", None)

    def processes(self, box: Inputs, settings: Settings) -> Noted:
        return Noted("noted", f"{box.name}:{settings.cache_path.name}")


def test_a_nested_input_is_available_and_is_invoked_with_its_parent():
    built = _model(Inputs("run", (PlantInput("plant", LeafInput("leaf")),)))
    assert [type(state) for state in built.states] == [Clock, Reading]
    assert built.states[0].name == "leaf"
    assert built.states[1].name == "from-leaf"
    assert [process.name for process in built.processes] == ["leaf", "leaf-mark"]


def test_inputs_and_settings_are_available_to_inputs_and_processes():
    settings = Settings("settings", Path("cache"), False, False, False)
    box = Inputs("run", (NeedsContext("needs"),))
    built = _model(box, settings)
    assert built.settings is settings
    assert built.states[0].name == "run:settings"
    assert isinstance(built.processes[0], Noted)
    assert built.processes[0].note == "run:cache"


def test_a_step_does_not_parse_annotations_again(monkeypatch):
    built = _model(Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input"))))

    def fail(*args, **kwargs):
        raise AssertionError("annotations parsed during simulation")

    monkeypatch.setattr("orion.core.invoke.get_annotations", fail)
    monkeypatch.setattr("orion.core.model.get_annotations", fail, raising=False)
    stepped = built.step()
    assert stepped.states[0].name == "clock12"
