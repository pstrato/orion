"""Behaviour: a model holds states, and each step receives settings, inputs, and processes."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from pathlib import Path

import pytest

from orion.core.input import Input, Inputs
from orion.core.model import model, step, validate
from orion.core.process import Process
from orion.core.quantity import is_positive
from orion.core.setting import Settings
from orion.core.state import State
from orion.core.variable import Variable, var

_SETTINGS = Settings("settings", Path("."), False, False, False)


def _model(inputs: Inputs, settings: Settings = _SETTINGS):
    return model("run", settings, inputs)


def _step(inputs: Inputs, processes: tuple[Process, ...], settings: Settings = _SETTINGS):
    return step(_model(inputs, settings), settings, inputs, processes)


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
    inputs = Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input")))
    built = _model(inputs)
    assert [type(state) for state in built.states] == [Clock, Reading]

    stepped = step(built, _SETTINGS, inputs, (Tick("tick"), Tock("tock")))
    assert stepped is not built
    assert [type(state) for state in stepped.states] == [Clock, Reading]
    assert stepped.states[0].name == "clock12"
    assert stepped.states[1].name == "reading2"
    assert built.states[0].name == "clock"
    assert built.states[1] is not stepped.states[1]


def test_a_process_that_returns_nothing_leaves_the_state_tuple_unchanged():
    inputs = Inputs("run", (HoldInput("hold-input"),))
    built = _model(inputs)
    clock = built.states[0]
    stepped = step(built, _SETTINGS, inputs, (Holds("hold"),))
    assert stepped.states == (clock,)


def test_a_base_state_matches_the_only_concrete_state():
    inputs = Inputs("run", (OnlyClock("clock-input"),))
    assert _step(inputs, (UsesAnyState("uses"),)).states[0].name == "clock!"


def test_a_base_type_with_two_concrete_states_is_rejected_when_the_process_steps():
    inputs = Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input")))
    with pytest.raises(ValueError, match="more than one concrete type"):
        _step(inputs, (UsesState("ambiguous"),))


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


def test_a_one_state_tuple_replaces_that_state():
    inputs = Inputs("run", (BoxInput("box-input"),))
    assert _step(inputs, (Boxes("box"),)).states[0].name == "clockb"


def test_a_tuple_return_replaces_each_state_and_keeps_creation_order():
    inputs = Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input")))
    stepped = _step(inputs, (Tick("tick"), Tock("tock"), Reverse("reverse")))
    assert stepped.states[0].name == "clock12c"
    assert stepped.states[1].name == "reading2r"


def test_a_base_return_type_matches_the_only_concrete_state():
    inputs = Inputs("run", (ReturnsStateInput("returns-input"),))
    assert _step(inputs, (ReturnsState("returns"),)).states[0].name == "clock!"


def test_a_base_return_type_with_two_concrete_states_is_rejected_when_the_process_steps():
    inputs = Inputs("run", (ClockInput("clock-input"), ReadingInput("reading-input")))
    with pytest.raises(ValueError, match="more than one concrete type"):
        _step(inputs, (ReturnsEither("either"),))


def test_a_process_can_require_an_input():
    inputs = Inputs("run", (ClockInput("clock-input"),))
    built = _model(inputs)
    stepped = step(built, _SETTINGS, inputs, (UsesInput("uses"),))
    assert stepped.states[0] is built.states[0]


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


def test_inputs_and_settings_are_available_when_states_are_created():
    settings = Settings("settings", Path("cache"), False, False, False)
    box = Inputs("run", (NeedsContext("needs"),))
    built = model("run", settings, box)
    assert built.states[0].name == "run:settings"


class Measured(Input):
    level: Variable


class Level(State):
    amount: Variable


class LevelInput(Input):
    def states(self) -> Level:
        return Level("level", None, var("amount", "m", 1.0, description="amount", constraint=is_positive))


class BrokenLevelInput(Input):
    def states(self) -> Level:
        return Level("level", None, var("amount", "m", -1.0, description="amount", constraint=is_positive))


class Breaks(Process):
    def step(self, level: Level) -> Level:
        return Level(level.name, None, var("amount", "m", -1.0, description="amount", constraint=is_positive))


def test_validate_inputs_rejects_a_quantity_that_breaks_its_constraint():
    level = var("level", "m", -1.0, description="level", constraint=is_positive)
    settings = Settings("settings", Path("."), True, False, False)
    with pytest.raises(ValueError, match="is positive"):
        validate(settings, Inputs("run", (Measured("measured", level),)), ())


def test_validate_initial_states_rejects_a_state_that_breaks_its_constraint():
    settings = Settings("settings", Path("."), False, True, False)
    with pytest.raises(ValueError, match="is positive"):
        validate(settings, Inputs("run", (BrokenLevelInput("level"),)), ())


def test_simulation_state_constraints_are_checked_when_validation_finishes():
    settings = Settings("settings", Path("."), False, False, True)
    inputs = Inputs("run", (LevelInput("level"),))
    with pytest.raises(ValueError, match="is positive"):
        validate(settings, inputs, (Breaks("breaks"),))
    stepped = _step(inputs, (Breaks("breaks"),), settings)
    level = next(state for state in stepped.states if isinstance(state, Level))
    assert float(level.amount.value) == pytest.approx(-1.0)
