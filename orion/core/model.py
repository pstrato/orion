"""A model is a tuple of states. Settings, inputs, and processes are passed to each step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp

from orion.core.axis import Axis, inputs_axis, step_axis
from orion.core.constant import Constant
from orion.core.constraint import problems
from orion.core.device import compute_device
from orion.core.entity import Entity, entity
from orion.core.input import Input, Inputs
from orion.core.invoke import invoke_input_states, invoke_process_step
from orion.core.process import Process
from orion.core.quantity import Quantity
from orion.core.setting import Settings
from orion.core.state import State
from orion.processes.clock import ClockInput


def validate(settings: Settings, inputs: Inputs | tuple[Inputs, ...], processes: Process | tuple[Process, ...]) -> None:
    """Reject broken inputs, broken initial states, and process calls that cannot be resolved."""
    groups = inputs if isinstance(inputs, tuple) else (inputs,)
    steps = processes if isinstance(processes, tuple) else (processes,)
    for group in groups:
        if settings.validate_inputs:
            _reject_problems((group,))
        states = invoke_input_states(_listed_inputs(group), group, settings)
        if settings.validate_initial_states:
            _reject_problems(states)
        for process in steps:
            states = _update(states, invoke_process_step(settings, group, states, process))
        if settings.validate_simulation_states:
            _reject_problems(states)


def model(name: str, settings: Settings, inputs: Inputs) -> Model:
    """Build a model to simulate inputs."""
    catalog = _listed_inputs(inputs)
    states = invoke_input_states(catalog, inputs, settings)
    return Model(
        name=name,
        states=states,
        axes=(),
    )


@entity()
class Model(Entity):
    """The states of one simulation, in the order the inputs created them."""

    states: tuple[State, ...]
    """Current states. This order does not change."""

    axes: tuple[Axis, ...]
    """Leading ndarray axes added by the simulation, before each quantity's own axes."""


def axes_of(model: Model, quantity: Quantity) -> tuple[Axis, ...]:
    """Axes of one quantity array: simulation axes, then the quantity's own axes."""
    return model.axes + quantity.axes


def step(model: Model, settings: Settings, inputs: Inputs, processes: tuple[Process, ...]) -> Model:
    """Run every process once. Settings, inputs, and processes come from the caller."""
    states = model.states
    for process in processes:
        states = _update(states, invoke_process_step(settings, inputs, states, process))
    return Model(name=model.name, states=states, axes=model.axes)


def simulate(settings: Settings, inputs: Inputs, processes: tuple[Process, ...], *, jit: bool = False, keep_history: bool = False) -> tuple[Model, Model | None]:
    """Step until the clock horizon, using ``lax.scan``. ``jit=True`` compiles that run. ``keep_history`` keeps sucessive simulation steps."""
    final, history = _compiled_run(settings, inputs, processes, jit=jit, keep_history=keep_history)
    return final, history


def simulate_all(settings: Settings, inputs: tuple[Inputs, ...], processes: tuple[Process, ...], *, jit: bool = False, keep_history: bool = False) -> tuple[Model, Model | None]:
    """Run every inputs to its clock horizon in one batched ``vmap``. ``jit=True`` compiles that batch. ``keep_history`` keeps sucessive simulation steps."""
    finals, histories = _compiled_batch(settings, inputs, processes, jit=jit, keep_history=keep_history)
    return finals, histories


def _compiled_run(settings: Settings, inputs: Inputs, processes: tuple[Process, ...], *, jit: bool, keep_history: bool) -> tuple[Model, Model | None]:
    count = _clock_steps(inputs)
    device = compute_device(settings.use_gpu)

    def run(settings: Settings, inputs: Inputs, processes: tuple[Process, ...]) -> tuple[Model, Model | None]:
        return _run(settings, inputs, processes, count, keep_history=keep_history)

    compiled = jax.jit(run) if jit else run
    with jax.default_device(device):
        final, history = compiled(settings, inputs, processes)

    if keep_history:
        assert history is not None
        history = replace(
            history,
            axes=(step_axis(count),),
        )
    return final, history


def _compiled_batch(settings: Settings, inputs: tuple[Inputs, ...], processes: tuple[Process, ...], *, jit: bool, keep_history: bool) -> tuple[Model, Model | None]:
    if not inputs:
        raise ValueError("simulate_all requires inputs.")
    _require_same_constants(inputs)
    count = _clock_steps(inputs[0])
    batched = _stack_inputs(inputs)
    device = compute_device(settings.use_gpu)

    def run(settings: Settings, batched_inputs: Inputs, processes: tuple[Process, ...]) -> tuple[Model, Model | None]:
        def one(single: Inputs) -> tuple[Model, Model | None]:
            return _run(settings, single, processes, count, keep_history=keep_history)

        return cast(tuple[Model, Model | None], jax.vmap(one)(batched_inputs))

    compiled = jax.jit(run) if jit else run
    with jax.default_device(device):
        finals, histories = compiled(settings, batched, processes)

    if keep_history:
        assert histories is not None
        histories = replace(histories, axes=(inputs_axis(inputs), step_axis(count)))
    finals = replace(finals, axes=(inputs_axis(inputs),))
    return finals, histories


def _run(settings: Settings, inputs: Inputs, processes: tuple[Process, ...], count: int, *, keep_history: bool) -> tuple[Model, Model | None]:
    current = model(inputs.name, settings, inputs)

    def body(current: Model, _index: jax.Array) -> tuple[Model, Model | None]:
        del _index
        nxt = step(current, settings, inputs, processes)
        return nxt, nxt if keep_history else None

    final, history = jax.lax.scan(body, current, jnp.arange(count))
    return cast(Model, final), cast(Model | None, history)


def _listed_inputs(inputs: Inputs) -> tuple[Input, ...]:
    """Every input on ``inputs``, including nested ones, parent before child."""
    return tuple(item for _, item in inputs.all_entities(of_type=Input) if isinstance(item, Input))


def _update(states: tuple[State, ...], update: None | State | tuple[State, ...]) -> tuple[State, ...]:
    """Replace held states with the ones a process returned, matched by concrete type."""
    if update is None:
        return states
    items = update if isinstance(update, tuple) else (update,)
    if not items:
        return states
    positions = {type(state): index for index, state in enumerate(states)}
    updated = list(states)
    for item in items:
        if not isinstance(item, State):
            raise TypeError(f"A process must return a state, not {type(item).__name__}.")
        try:
            index = positions[type(item)]
        except KeyError:
            raise ValueError(f"{type(item).__name__} is not held by the model.") from None
        updated[index] = item
    return tuple(updated)


def _reject_problems(entities: Sequence[Entity]) -> None:
    found = [problem for entity in entities for problem in problems(entity)]
    if found:
        raise ValueError("\n".join(str(problem) for problem in found))


def _clock_steps(inputs: Inputs) -> int:
    """Steps implied by the clock input: days × 24 / delta hours."""
    for _, item in inputs.all_entities(of_type=ClockInput):
        if isinstance(item, ClockInput):
            days = (item.end.value - item.start.value).days
            delta = int(item.delta.value)
            if delta <= 0:
                raise ValueError("Step hours must be positive.")
            count = int(days) * 24 // delta
            if count < 1:
                raise ValueError("Horizon is shorter than one step.")
            return count
    raise ValueError("Inputs have no clock.")


def _require_same_constants(groups: tuple[Inputs, ...]) -> None:
    """Meta is the constants. Every inputs in a batch must carry the same ones."""
    reference = _constants(groups[0])
    for group in groups[1:]:
        if _constants(group) != reference:
            raise ValueError("simulate_all requires every inputs to have the same constants.")


def _constants(inputs: Inputs) -> tuple[Constant, ...]:
    return tuple(item for _, item in inputs.all_entities(of_type=Constant) if isinstance(item, Constant))


def _stack_inputs(groups: tuple[Inputs, ...]) -> Inputs:
    """Stack array leaves so ``vmap`` can split one inputs tree per batch row.

    Meta fields stay on the first inputs. Only data leaves are batched.
    """
    rows = [jax.tree.leaves(group) for group in groups]
    if not rows[0]:
        raise ValueError("simulate_all requires inputs whose quantities can be batched.")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("simulate_all requires inputs with the same structure.")
    stacked = [jnp.stack([jnp.asarray(leaf) for leaf in column]) for column in zip(*rows, strict=True)]
    return cast(Inputs, jax.tree.unflatten(jax.tree.structure(groups[0]), stacked))
