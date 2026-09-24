from __future__ import annotations

from annotationlib import Format
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import get_type_hints

import jax

from orion.core.entity import Entity, entity_meta_fields
from orion.core.input import Input, Inputs
from orion.core.jax import entity
from orion.core.process import Process
from orion.core.protocol import invoke_protocol, validate_protocol
from orion.core.setting import Settings
from orion.core.state import State


def model(name: str, inputs: Inputs) -> Model:
    """Create a model from top-level inputs.

    Clock state/process come from ``inputs.start`` / ``end`` / ``step``.
    Weather, soil, and each entry in ``inputs.processes`` must seed every
    required state via ``Input.states`` (no State defaults / initial()).
    """
    states = dict[type, State]()
    for input in inputs.inputs:
        validate_protocol(input.states, (Inputs, State), (State, tuple[State, ...]))
        input_states = invoke_protocol(input.states, {**states, Inputs: inputs})
        if input_states is None:
            continue
        if not isinstance(input_states, tuple):
            input_states = (input_states,)
        for state in input_states:
            cls = type(state)
            if cls in states:
                raise ValueError(f"Duplicate initial state for type {cls.__name__}.")
            states[cls] = state

    process_by_type = dict[type[Process], Process]()
    order = dict[int, Process]()
    for input in inputs.inputs:
        validate_protocol(input.processes, (Inputs, State), (Process, tuple[Process, ...]))
        input_processes = invoke_protocol(input.processes, {**states, Inputs: inputs})
        if input_processes is None:
            continue
        if not isinstance(input_processes, tuple):
            input_processes = (input_processes,)
        for process in input_processes:
            cls = type(process)
            if cls in process_by_type:
                raise ValueError(f"Duplicate process of type {cls.__name__}.")
            process_by_type[cls] = process
            order[len(order)] = process

    processes = tuple(order[i] for i in sorted(order))

    required_inputs, required_states = _input_state_classes(processes)
    for cls in required_states:
        if cls not in states:
            raise ValueError(f"No input of type {cls.__name__}, required by a process.")

    classes = tuple(sorted(states.keys(), key=lambda t: t.__name__))
    states = tuple(states[cls] for cls in classes)
    state_index: Mapping[type[State], int] = MappingProxyType({cls: i for i, cls in enumerate(classes)})
    index_state: Mapping[int, type[State]] = MappingProxyType({i: cls for cls, i in state_index.items()})
    arg_indices = _process_arg_indices(input_index, state_index, processes)

    return Model(
        name=name,
        inputs=inputs,
        processes=processes,
        state_classes=classes,
        state_index=state_index,
        index_state=index_state,
        states=states,
        process_arg_indices=arg_indices,
    )


def _input_state_classes(processes: tuple[Process, ...]):
    """State types required as inputs by process step signatures."""
    input_classes = set[type[Input]]()
    state_classes = set[type[State]]()
    for process in processes:
        process_step = getattr(process, "step", None)
        if process_step is None:
            raise ValueError(f"Process {process} does not have a step method.")
        step_signature = get_type_hints(process_step, format=Format.VALUE)
        for method_var, method_type in step_signature.items():
            if method_var == "return":
                continue
            if isinstance(method_type, type) and issubclass(method_type, Input):
                input_classes.add(method_type)
            if isinstance(method_type, type) and issubclass(method_type, State):
                state_classes.add(method_type)
    return tuple(sorted(input_classes, key=lambda t: t.__name__)), tuple(sorted(state_classes, key=lambda t: t.__name__))


def _process_arg_indices(input_index: Mapping[type[Input], int], state_index: Mapping[type[State], int], processes: tuple[Process, ...]) -> tuple[tuple[int, ...], ...]:
    """Input and state indices for each process ``step`` argument."""
    all_indices: list[tuple[int, ...]] = []
    for process in processes:
        process_step = getattr(process, "step", None)
        if process_step is None:
            raise ValueError(f"Process {process} does not have a step method.")
        indices: list[int] = []
        for method_var, method_type in get_type_hints(process_step, format=Format.VALUE).items():
            if method_var == "return":
                continue
            if isinstance(method_type, type) and issubclass(method_type, State):
                index = state_index.get(method_type)
                if index is None:
                    raise ValueError(f"No state {method_type} for {process.name}")
            elif isinstance(method_type, type) and issubclass(method_type, Input):
                index = input_index.get(method_type)
                if index is None:
                    raise ValueError(f"No input {method_type} for {process.name}")
            indices.append(index)
        all_indices.append(tuple(indices))
    return tuple(all_indices)


@entity(
    data=("states", "processes"),
    meta=(
        *entity_meta_fields,
        "inputs",
        "state_classes",
        "state_index",
        "index_state",
        "process_arg_indices",
    ),
)
class Model(Entity):
    """Collection of processes that keeps the current simulation state.

    ``processes`` are pytree data so forcing arrays (e.g. weather series) are
    traced inputs: same shapes reuse a jitted compile across datasets.
    Simulation history stacks **states** only — not process payloads.
    """

    inputs: Inputs
    state_classes: tuple[type[State], ...]
    state_index: Mapping[type[State], int]
    index_state: Mapping[int, type[State]]
    states: tuple[State, ...]
    processes: tuple[Process, ...]
    process_arg_indices: tuple[tuple[int, ...], ...]

    @property
    def entity_kind(self) -> str:
        return "model"

    @property
    def settings(self) -> Settings:
        return self.inputs.settings

    def step(self) -> Model:
        states = self.states
        for process, indices in zip(self.processes, self.process_arg_indices, strict=True):
            arguments = tuple(states[i] for i in indices)
            processed = process.step(*arguments)
            states = self.update_states(processed, states)

        return Model(
            name=self.name,
            inputs=self.inputs,
            state_classes=self.state_classes,
            state_index=self.state_index,
            index_state=self.index_state,
            processes=self.processes,
            states=tuple(states),
            process_arg_indices=self.process_arg_indices,
        )

    def simulate(self, step_count: int) -> tuple[Model, Model]:
        """Advance ``step_count`` steps using the shared jitted scan."""

        final, history = jitted_simulate(with_jit_stable_meta(self), int(step_count))
        return restore_site_meta(final, template), restore_site_meta(history, template)

    def update_states(self, processed: State | tuple[State, ...], states: tuple[State, ...]) -> tuple[State, ...]:
        if isinstance(processed, State):
            processed = (processed,)
        updated_states = list(states)
        for state in processed:
            index = self.state_index[type(state)]
            updated_states[index] = state
        return tuple(updated_states)


@partial(jax.jit, static_argnums=(1,))
def jitted_simulate(model: Model, step_count: int) -> tuple[Model, Model]:
    """XLA-compiled scan over states only (process payloads are not stacked into history)."""

    def step_fn(states: tuple[State, ...], _: None) -> tuple[tuple[State, ...], tuple[State, ...]]:
        next_model = Model(
            name=model.name,
            inputs=model.inputs,
            state_classes=model.state_classes,
            state_index=model.state_index,
            index_state=model.index_state,
            processes=model.processes,
            states=states,
            process_arg_indices=model.process_arg_indices,
        ).step()
        return next_model.states, next_model.states

    final_states, history_states = jax.lax.scan(step_fn, model.states, None, length=step_count)
    final = Model(
        name=model.name,
        inputs=model.inputs,
        state_classes=model.state_classes,
        state_index=model.state_index,
        index_state=model.index_state,
        processes=model.processes,
        states=final_states,
        process_arg_indices=model.process_arg_indices,
    )
    history = Model(
        name=model.name,
        inputs=model.inputs,
        state_classes=model.state_classes,
        state_index=model.state_index,
        index_state=model.index_state,
        processes=model.processes,
        states=history_states,
        process_arg_indices=model.process_arg_indices,
    )
    return final, history
