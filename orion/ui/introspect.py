"""Reflect on Orion models to drive UI generation.

Discovery walks ``direct_entities`` / ``all_entities``. States are read-only.
Constants on inputs stay fixed during a step and can still be edited before it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp

from orion.core.entity import Entity
from orion.core.input import Inputs, LocationInput
from orion.core.model import Model, axes_of
from orion.core.process import Process
from orion.core.quantity import Quantity
from orion.core.state import State
from orion.core.variable import Variable
from orion.processes.clock import ClockInput
from orion.ui.reflect import format_path, on_within_step, owned_quantities, quantity_kind, run_steps

__all__ = [
    "EntityView",
    "ModelView",
    "history_series",
    "inspect_model",
    "inspect_process",
    "inspect_state",
    "jax_device_label",
    "list_plottable_variables",
    "run_steps",
]


@dataclass(frozen=True)
class EntityView:
    """One reflected entity and its direct children."""

    name: str
    type_name: str
    kind: str
    doc: str
    unit: str
    fields: tuple[EntityView, ...] = ()


@dataclass(frozen=True)
class ModelView:
    """Full model snapshot for the UI."""

    name: str
    doc: str
    processes: tuple[EntityView, ...]
    states: tuple[EntityView, ...]
    step_hours: int
    days: int
    location_name: str
    latitude: float
    longitude: float
    device: str


def jax_device_label() -> str:
    """Human-readable default JAX device."""
    try:
        import jax

        device = jax.devices()[0]
        return f"{device.platform}:{device.id}"
    except Exception as exc:  # noqa: BLE001 — UI must stay up without GPU
        return f"unavailable ({exc})"


def inspect_process(process: Process) -> EntityView:
    """Describe a process from its entity children."""
    return _view(process, kind="process")


def inspect_state(state: State) -> EntityView:
    """Describe a state from its entity children."""
    return _view(state, kind="state")


def inspect_model(model: Model, inputs: Inputs, processes: tuple[Process, ...] = ()) -> ModelView:
    """Reflect the model states, plus the inputs and processes supplied for the run."""
    step_hours, days = _clock_horizon(inputs)
    location_name, latitude, longitude = _location(inputs)
    doc = inspect.getdoc(type(model)) or ""
    return ModelView(
        name=model.name,
        doc=doc.split("\n", 1)[0],
        processes=tuple(inspect_process(process) for process in processes),
        states=tuple(inspect_state(state) for state in model.states),
        step_hours=step_hours,
        days=days,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        device=jax_device_label(),
    )


def _plot_priority(state: State) -> int:
    module = type(state).__module__
    if ".crop." in module or module.endswith(".crop"):
        return 0
    if ".management." in module:
        return 1
    if ".soil." in module:
        return 2
    return 3


def history_series(history: Model, state_name: str, variable_name: str) -> tuple[list[float], str, str | None]:
    """Extract a 1-D time series from stepped history for plotting."""
    variable = _variable_named_in(history, state_name, variable_name)
    arr = jnp.asarray(variable.value).reshape(-1)
    return [float(item) for item in arr.tolist()], variable.unit, variable.resource


def history_days(history: Model, state_name: str, variable_name: str, step_hours: int) -> list[float]:
    """Day coordinate for each flattened sample.

    A step-only series puts one point at the start of each step. A ``within_step``
    axis holds one sample per hour of ``clock.delta``, so those samples share the
    same day span as the other plots instead of counting as one day each.
    """
    variable = _variable_named_in(history, state_name, variable_name)
    arr = jnp.asarray(variable.value)
    axes = axes_of(history, variable)
    within = next((index for index, item in enumerate(axes) if item.name == "within_step"), None)
    if within is None or arr.ndim != len(axes) or int(arr.shape[within]) < 1:
        return [index * step_hours / 24 for index in range(int(arr.size))]
    within_size = int(arr.shape[within])
    step = next((index for index, item in enumerate(axes) if item.name == "step"), None)
    positions = jnp.unravel_index(jnp.arange(arr.size), arr.shape)
    step_index = positions[step] if step is not None else jnp.zeros(arr.size, dtype=jnp.int32)
    hour_index = positions[within]
    axis = axes[within]
    if len(axis.values) == within_size:
        hours = jnp.asarray(tuple(axis.values), dtype=jnp.float32)[hour_index]
    else:
        hours = hour_index.astype(jnp.float32)
    days = (step_index.astype(jnp.float32) * step_hours + hours * (step_hours / within_size)) / 24
    return [float(item) for item in days.reshape(-1).tolist()]


def _variable_named_in(history: Model, state_name: str, variable_name: str) -> Variable:
    for state in history.states:
        if state.name != state_name:
            continue
        variable = _variable_named(state, variable_name)
        if variable is not None:
            return variable
    raise KeyError(f"{state_name}.{variable_name}")


def list_plottable_variables(history: Model) -> Iterable[tuple[str, str, str]]:
    """Yield (state_name, variable_name, unit), crop domain first.

    Only variables are plotted. Constants on inputs do not form a series; edit those inputs instead.
    """
    states = sorted(history.states, key=lambda state: (_plot_priority(state), state.name))
    for state in states:
        for path, variable in owned_quantities(state, Variable):
            if not isinstance(variable.value, jnp.ndarray):
                continue
            if on_within_step(variable):
                continue
            yield state.name, format_path(path), variable.unit


def _view(entity: Entity, kind: str | None = None) -> EntityView:
    from orion.core.axis import Axis
    from orion.core.constraint import Constraint

    children = tuple(_view(child) for _relation, child in entity.direct_entities() if not isinstance(child, (Constraint, Axis)))
    unit = entity.unit if isinstance(entity, Quantity) else ""
    description = entity.description if isinstance(entity, Quantity) else ""
    doc = description or (inspect.getdoc(type(entity)) or "").split("\n", 1)[0]
    return EntityView(
        name=entity.name,
        type_name=type(entity).__name__,
        kind=kind or quantity_kind(entity),
        doc=doc,
        unit=str(unit or ""),
        fields=children,
    )


def _variable_named(state: State, variable_name: str) -> Variable | None:
    by_label = {format_path(path): variable for path, variable in owned_quantities(state, Variable)}
    found = by_label.get(variable_name)
    if isinstance(found, Variable):
        return found
    named = [variable for variable in by_label.values() if isinstance(variable, Variable) and variable.name == variable_name]
    if len(named) == 1:
        return named[0]
    return None


def _clock_horizon(inputs: Inputs) -> tuple[int, int]:
    for _, entity in inputs.all_entities():
        if isinstance(entity, ClockInput):
            days = int((entity.end.value - entity.start.value).days)
            return int(entity.delta.value), days
    return 0, 0


def _location(inputs: Inputs) -> tuple[str, float, float]:
    for _, entity in inputs.all_entities():
        if isinstance(entity, LocationInput):
            point = entity.centroid.value
            return entity.name, float(point.y), float(point.x)
    return "", 0.0, 0.0
