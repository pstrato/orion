"""Discover processes, introspect step I/O, and sweep numeric inputs."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from annotationlib import Format
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Literal, get_args, get_origin, get_type_hints

import jax.numpy as jnp

from orion.core.constant import Constant
from orion.core.entity import Entity
from orion.core.input import Input
from orion.core.parameter import Parameter
from orion.core.process import Process
from orion.core.quantity import Quantity
from orion.core.state import State
from orion.core.variable import Variable
from orion.ui.reflect import format_path, numeric_scalar, owned_quantities, quantity_kind, set_quantity_value

ParamMode = Literal["value", "range"]


@dataclass(frozen=True)
class ProcessInfo:
    """One discoverable process class."""

    key: str
    cls: type
    label: str
    module: str
    doc: str
    bases: tuple[str, ...]
    implemented: bool
    input_states: tuple[type[State], ...]
    output_states: tuple[type[State], ...]


@dataclass(frozen=True)
class ScalarField:
    """Numeric quantity. Editable when it belongs to an input, including constants."""

    path: str
    state_name: str
    unit: str | None
    kind: str
    value: float
    description: str = ""


@dataclass(frozen=True)
class ParamSpec:
    """UI edit for one scalar: fixed value or linspace range."""

    path: str
    mode: ParamMode = "value"
    value: float = 0.0
    start: float = 0.0
    stop: float = 1.0
    steps: int = 5


@dataclass(frozen=True)
class SweepResult:
    """Outputs collected from one or more process evaluations."""

    param_paths: tuple[str, ...]
    param_values: tuple[tuple[float, ...], ...]
    outputs: tuple[dict[str, float], ...]
    output_units: dict[str, str]


def _iter_modules(package_name: str) -> Iterable[str]:
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        yield package_name
        return
    for module in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        yield module.name


def _is_process_class(cls: type) -> bool:
    if not isinstance(cls, type) or not issubclass(cls, Process) or cls is Process:
        return False
    step = getattr(cls, "step", None)
    return step is not None and callable(step)


def _step_state_types(cls: type) -> tuple[tuple[type[State], ...], tuple[type[State], ...]]:
    step = getattr(cls, "step", None)
    if step is None or not callable(step):
        return (), ()
    hints = get_type_hints(step, format=Format.VALUE)
    inputs: list[type[State]] = []
    for name, annotation in hints.items():
        if name in {"self", "return"}:
            continue
        if isinstance(annotation, type) and issubclass(annotation, State):
            inputs.append(annotation)
    outputs: list[type[State]] = []
    ret = hints.get("return")
    if ret is not None:
        origin = get_origin(ret)
        if origin is tuple:
            for arg in get_args(ret):
                if isinstance(arg, type) and issubclass(arg, State):
                    outputs.append(arg)
        elif isinstance(ret, type) and issubclass(ret, State):
            outputs.append(ret)
    return tuple(inputs), tuple(outputs)


def _is_implemented(cls: type) -> bool:
    step = getattr(cls, "step", None)
    if step is None or not callable(step):
        return False
    try:
        source = inspect.getsource(step)
    except OSError, TypeError:
        return True
    return "raise NotImplementedError" not in source


def discover_process_classes() -> tuple[ProcessInfo, ...]:
    """Find process classes under orion.processes and orion.clients."""
    found: dict[str, ProcessInfo] = {}
    for package_name in ("orion.processes", "orion.clients"):
        for module_name in _iter_modules(package_name):
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001 — skip broken optional imports
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module_name:
                    continue
                if not _is_process_class(obj):
                    continue
                key = f"{obj.__module__}.{obj.__name__}"
                if key in found:
                    continue
                inputs, outputs = _step_state_types(obj)
                bases = tuple(base.__name__ for base in obj.__mro__[1:] if base.__name__.endswith("Process"))
                found[key] = ProcessInfo(
                    key=key,
                    cls=obj,
                    label=obj.__name__,
                    module=obj.__module__,
                    doc=(inspect.getdoc(obj) or "").strip().split("\n", 1)[0],
                    bases=bases,
                    implemented=_is_implemented(obj),
                    input_states=inputs,
                    output_states=outputs,
                )
    return tuple(sorted(found.values(), key=lambda p: p.key))


def process_hierarchy_tree(processes: Sequence[ProcessInfo] | None = None) -> list[dict[str, Any]]:
    """Build a NiceGUI tree of processes grouped by inheritance and package."""
    processes = list(processes if processes is not None else discover_process_classes())
    by_label = {p.label: p for p in processes}
    children_of: dict[str | None, list[ProcessInfo]] = {}
    for info in processes:
        parent = info.bases[0] if info.bases else None
        if parent is not None and parent not in by_label:
            parent = None
        children_of.setdefault(parent, []).append(info)

    def _process_node(info: ProcessInfo) -> dict[str, Any]:
        kids = children_of.get(info.label, [])
        node: dict[str, Any] = {
            "id": info.key,
            "label": info.label + ("" if info.implemented else " (abstract)"),
            "process_key": info.key,
        }
        if kids:
            node["children"] = [_process_node(k) for k in sorted(kids, key=lambda i: i.label)]
        return node

    def package_group(infos: list[ProcessInfo]) -> list[dict[str, Any]]:
        groups: dict[str, list[ProcessInfo]] = {}
        for info in infos:
            parts = info.module.split(".")
            if len(parts) > 2 and parts[1] == "processes":
                group = parts[2]
            elif len(parts) > 1:
                group = parts[1]
            else:
                group = "other"
            groups.setdefault(group, []).append(info)
        nodes: list[dict[str, Any]] = []
        for group, items in sorted(groups.items()):
            nodes.append(
                {
                    "id": f"group:{group}",
                    "label": group,
                    "children": [_process_node(p) for p in sorted(items, key=lambda i: i.label)],
                }
            )
        return nodes

    roots = children_of.get(None, [])
    top_level_parents = [p for p in roots if p.label in children_of]
    top_level_leaves = [p for p in roots if p.label not in children_of]
    nodes: list[dict[str, Any]] = [_process_node(p) for p in sorted(top_level_parents, key=lambda i: i.label)]
    nodes.extend(package_group(top_level_leaves))
    return nodes


def _input_class_for_process(process_cls: type) -> type[Input] | None:
    module = importlib.import_module(process_cls.__module__)
    name = process_cls.__name__.removesuffix("Process") + "Input"
    obj = getattr(module, name, None)
    if isinstance(obj, type) and issubclass(obj, Input) and obj is not Input:
        return obj
    return None


def default_input_for(info: ProcessInfo) -> Input:
    """Input whose quantities the lab can edit. States are built from it, not edited."""
    from datetime import date

    from orion.core.constant import const
    from orion.processes.clock import ClockInput

    if not isinstance(info.cls, type):
        raise TypeError(f"No input for {info.label}. States are read-only in the UI.")
    cls = _input_class_for_process(info.cls)
    if cls is None:
        raise TypeError(f"No input for {info.label}. States are read-only in the UI.")
    if cls is ClockInput:
        return ClockInput(
            name="clock",
            start=const("start", "isodate", date(2024, 1, 1), "Simulation start date"),
            end=const("end", "isodate", date(2024, 1, 11), "Simulation end date"),
            delta=const("delta", "hours", 3, "Simulation delta step in hours"),
        )
    return cls(info.label)


def instantiate_process(info: ProcessInfo) -> Entity:
    """Construct a process instance for the lab."""
    from orion.processes.clock import ClockProcess

    cls = info.cls
    if cls is ClockProcess:
        return ClockProcess("clock")
    return cls(info.label)  # type: ignore[call-arg]


def list_scalar_fields(owner_name: str, entity: Entity) -> tuple[ScalarField, ...]:
    """Flatten numeric scalar quantities, including constants."""
    out: list[ScalarField] = []
    for path, quantity in owned_quantities(entity):
        number = numeric_scalar(quantity.value)
        if number is None:
            continue
        label = format_path(path)
        out.append(
            ScalarField(
                path=f"{owner_name}.{label}",
                state_name=owner_name,
                unit=quantity.unit,
                kind=quantity_kind(quantity),
                value=number,
                description=quantity.description,
            )
        )
    return tuple(out)


def _set_on_entity(entity: Entity, parts: list[str], value: float) -> Entity:
    if isinstance(entity, State):
        raise TypeError("States are read-only in the UI. Edit the Input that creates them, including constants.")
    if not parts:
        return entity
    head, *tail = parts
    if not is_dataclass(entity):
        raise TypeError(f"Cannot set path on non-dataclass {type(entity)}")
    try:
        current = getattr(entity, head)
    except AttributeError:
        raise TypeError(f"States are read-only in the UI. {type(entity).__name__} has no input quantity {head!r}.") from None
    if not tail:
        if isinstance(current, (Variable, Parameter, Constant, Quantity)):
            return replace(entity, **{head: set_quantity_value(current, value)})
        raise TypeError(f"Cannot assign numeric value to {head} on {type(entity).__name__}")
    if isinstance(current, Entity):
        return replace(entity, **{head: _set_on_entity(current, tail, value)})
    if isinstance(current, tuple) and current and all(isinstance(v, Entity) for v in current):
        index = int(tail[0])
        updated = list(current)
        updated[index] = _set_on_entity(updated[index], tail[1:], value)
        return replace(entity, **{head: tuple(updated)})
    raise TypeError(f"Cannot traverse {head} on {type(entity).__name__}")


def _apply_input_overrides(inp: Input, overrides: dict[str, float]) -> Input:
    """Return a copy of an input with quantity overrides applied. Does not write states."""
    updated: Entity = inp
    for path, value in overrides.items():
        owner, _, rest = path.partition(".")
        if owner != inp.name or not rest:
            raise KeyError(path)
        updated = _set_on_entity(updated, rest.split("."), value)
    if not isinstance(updated, Input):
        raise TypeError(f"Editing {inp.name} did not produce an input.")
    return updated


def _states_of(inp: Input) -> tuple[State, ...]:
    value = inp.states()
    if value is None:
        return ()
    if isinstance(value, State):
        return (value,)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, State))
    raise TypeError(f"{type(inp).__name__}.states() must return a state or a tuple of states.")


def _param_grid(params: Sequence[ParamSpec]) -> tuple[tuple[str, ...], list[tuple[float, ...]]]:
    paths = tuple(p.path for p in params)
    axes: list[list[float]] = []
    for param in params:
        if param.mode == "range":
            steps = max(int(param.steps), 2)
            axes.append([float(x) for x in jnp.linspace(param.start, param.stop, steps).tolist()])
        else:
            axes.append([float(param.value)])
    grid: list[tuple[float, ...]] = [()]
    for axis in axes:
        grid = [prefix + (value,) for prefix in grid for value in axis]
    if len(grid) > 200:
        raise ValueError(f"Parameter sweep has {len(grid)} points; limit is 200.")
    return paths, grid


def _flatten_output_scalars(states: Sequence[State]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    units: dict[str, str] = {}
    for state in states:
        for field in list_scalar_fields(state.name, state):
            if field.kind not in {"variable", "parameter"}:
                continue
            values[field.path] = field.value
            units[field.path] = field.unit or ""
    return values, units


def run_process_sweep(info: ProcessInfo, params: Sequence[ParamSpec]) -> SweepResult:
    """Instantiate process, apply param grid to defaults, collect output scalars."""
    if not info.implemented:
        raise ValueError(f"{info.label} is abstract and cannot be run.")
    process = instantiate_process(info)
    if not isinstance(process, Process):
        raise TypeError(f"{info.label} is not a Process")

    sample = default_input_for(info)
    step = type(process).step
    hints = get_type_hints(step, format=Format.VALUE)
    arg_names = [name for name, annotation in hints.items() if name not in {"self", "return"} and isinstance(annotation, type) and issubclass(annotation, State)]

    paths, grid = _param_grid(params)
    outputs: list[dict[str, float]] = []
    units: dict[str, str] = {}
    for point in grid:
        overrides = {path: value for path, value in zip(paths, point, strict=True)}
        edited = _apply_input_overrides(sample, overrides)
        produced = {type(state): state for state in _states_of(edited)}
        args = [_state_for_argument(produced, hints[name]) for name in arg_names]
        result = process.step(*args)
        if isinstance(result, State):
            result_states: tuple[State, ...] = (result,)
        elif isinstance(result, tuple):
            result_states = tuple(state for state in result if isinstance(state, State))
        else:
            result_states = ()
        flat, flat_units = _flatten_output_scalars(result_states)
        outputs.append(flat)
        units.update(flat_units)
    return SweepResult(param_paths=paths, param_values=tuple(grid), outputs=tuple(outputs), output_units=units)


def get_process(key: str, processes: Sequence[ProcessInfo] | None = None) -> ProcessInfo:
    processes = processes if processes is not None else discover_process_classes()
    for info in processes:
        if info.key == key:
            return info
    raise KeyError(key)


def _state_for_argument(produced: dict[type[State], State], annotation: type[State]) -> State:
    if annotation in produced:
        return produced[annotation]
    matches = [state for cls, state in produced.items() if issubclass(cls, annotation)]
    if len(matches) == 1:
        return matches[0]
    raise TypeError(f"No state of type {annotation.__name__} was built from the input.")


def editable_fields_for(info: ProcessInfo) -> tuple[ScalarField, ...]:
    """Numeric quantities on the process input, including constants. States are not listed."""
    try:
        sample = default_input_for(info)
    except TypeError:
        return ()
    return list_scalar_fields(sample.name, sample)
