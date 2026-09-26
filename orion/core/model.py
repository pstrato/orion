"""A model is the inputs' states and processes, stepped by index."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from orion.core.entity import Entity, entity
from orion.core.input import Input, Inputs
from orion.core.invoke import Assignment, invoke_input_states, process_argument_slots, process_assignment
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.state import State


def model(name: str, settings: Settings, inputs: Inputs, processes: tuple[Process, ...]) -> Model:
    """Build a model to simulate inputs."""
    catalog = _listed_inputs(inputs)
    state_map = invoke_input_states(catalog, inputs, settings)
    state_classes = tuple(state_map)
    state_index = MappingProxyType({cls: index for index, cls in enumerate(state_classes)})
    process_arguments = tuple(process_argument_slots(process, catalog, state_map, inputs, settings) for process in processes)
    process_assignments = tuple(process_assignment(process, state_map) for process in processes)
    return Model(
        name=name,
        inputs=inputs,
        settings=settings,
        states=tuple(state_map.values()),
        processes=processes,
        state_classes=state_classes,
        state_index=state_index,
        process_arguments=process_arguments,
        process_assignments=process_assignments,
    )


@entity()
class Model(Entity):
    """Processes applied in creation order to a fixed tuple of states."""

    inputs: Inputs
    """Inputs that created the states and processes."""

    settings: Settings
    """Model simulation settings."""

    states: tuple[State, ...]
    """Current states, in ``state_classes`` order."""

    processes: tuple[Process, ...]
    """Processes in the order the inputs created them."""

    state_classes: tuple[type[State], ...]
    """Concrete state types. This order does not change."""

    state_index: Mapping[type[State], int]
    """Index of each concrete state type in ``states``."""

    process_arguments: tuple[tuple[int, ...], ...]
    """State indexes passed to each process, aligned with ``processes``."""

    process_assignments: tuple[Assignment, ...]
    """State indexes written by each process, aligned with ``processes``."""

    def step(self) -> Model:
        """Run every process once, reading and writing the state tuple by planned indexes."""
        states = list(self.states)
        for process, arguments, assignment in zip(self.processes, self.process_arguments, self.process_assignments, strict=True):
            call = cast(tuple[Input | State, ...], tuple(states[index] for index in arguments))
            _write(states, assignment, process.step(*call))
        return Model(
            name=self.name,
            inputs=self.inputs,
            settings=self.settings,
            states=tuple(states),
            processes=self.processes,
            state_classes=self.state_classes,
            state_index=self.state_index,
            process_arguments=self.process_arguments,
            process_assignments=self.process_assignments,
        )


def _listed_inputs(inputs: Inputs) -> tuple[Input, ...]:
    """Every input on ``inputs``, including nested ones, parent before child."""
    return tuple(item for _, item in inputs.all_entities(of_type=Input) if isinstance(item, Input))


def _write(states: list[State], assignment: Assignment, result: object) -> None:
    if assignment.as_tuple:
        for index, state in zip(assignment.indexes, cast(tuple[State, ...], result), strict=True):
            states[index] = state
    elif assignment.indexes:
        states[assignment.indexes[0]] = cast(State, result)
