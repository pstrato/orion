"""Call input and process methods with the one instance of each concrete argument type.

``Input.states`` may take any concrete input, and any state already produced by an
earlier input. ``Input.processes`` may take any concrete input and any state those
inputs produced. ``Process.step`` may take any concrete input and any state the
model is holding. A second instance of the same concrete input or state type is rejected.

A parameter may name a base type. That matches the single concrete instance of a
subclass. If more than one concrete type can be used, parsing fails.

Invoked parameters cannot have defaults. A planned ``step`` call is indexes into
the model state tuple. The return signature plans which of those indexes are written.
"""

from __future__ import annotations

from annotationlib import Format, get_annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
from inspect import signature
from typing import Any, NamedTuple, get_args, get_origin

from orion.core.input import Input
from orion.core.process import Process
from orion.core.state import State


class Assignment(NamedTuple):
    """Indexes written by one ``step`` result, in return-signature order.

    ``as_tuple`` is true when the signature returns a tuple of states. Empty
    ``indexes`` and ``as_tuple`` false means the process returns ``None``.
    """

    indexes: tuple[int, ...]
    as_tuple: bool


def invoke(method: Callable[..., Any], available: Mapping[type, Any]) -> Any:
    """Call ``method``, filling each parameter with the instance of its annotated type."""
    positional, keyword = _bind(method, available)
    return method(*positional, **keyword)


def process_argument_slots(process: Process, inputs: Sequence[Input], states: Mapping[type[State], State], *extras: object) -> tuple[int, ...]:
    """Plan ``process.step`` arguments as indexes into the state tuple."""
    available = _index([*extras, *inputs, *states.values()])
    state_index = {cls: index for index, cls in enumerate(states)}
    slots: list[int] = []
    for value in _bound_values(process.step, available):
        concrete = type(value)
        try:
            index = state_index[concrete]
        except KeyError:
            raise TypeError(f"{_method_name(process.step)} argument {concrete.__name__} is not a state.") from None
        slots.append(index)
    return tuple(slots)


def process_assignment(process: Process, states: Mapping[type[State], State]) -> Assignment:
    """Plan where ``process.step`` writes, from its return annotation."""
    annotation = _return_annotation(process.step)
    if annotation is None or annotation is type(None):
        return Assignment((), False)
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not args or any(arg is Ellipsis for arg in args):
            raise TypeError(f"{_method_name(process.step)} return type must name each state, got {annotation!r}.")
        return Assignment(tuple(_state_index(arg, states, process.step) for arg in args), True)
    if origin is not None or not isinstance(annotation, type):
        raise TypeError(f"{_method_name(process.step)} return type must be None, a state, or a tuple of states, got {annotation!r}.")
    return Assignment((_state_index(annotation, states, process.step),), False)


def invoke_input_states(inputs: Sequence[Input], *extras: object) -> dict[type[State], State]:
    """Invoke each input's ``states`` in order.

    Every concrete input is available immediately. A state becomes available only
    after the input that created it has been invoked.
    """
    available = _index([*extras, *inputs])
    states: dict[type[State], State] = {}
    for item in inputs:
        for state in _expect(invoke(item.states, available), State, item.states):
            _put(available, state)
            _put(states, state)
    return states


def invoke_input_processes(inputs: Sequence[Input], states: Mapping[type[State], State], *extras: object) -> tuple[Process, ...]:
    """Invoke each input's ``processes``. States already created by the inputs are available."""
    available = _index([*extras, *inputs, *states.values()])
    processes: list[Process] = []
    for item in inputs:
        processes.extend(_expect(invoke(item.processes, available), Process, item.processes))
    return tuple(processes)


def invoke_process_step(process: Process, inputs: Iterable[Input], states: Mapping[type[State], State], *extras: object) -> dict[type[State], State]:
    """Invoke ``process.step`` and replace each returned state of the same concrete type."""
    available = _index([*extras, *inputs, *states.values()])
    updated = dict(states)
    for state in _expect(invoke(process.step, available), State, process.step):
        updated[type(state)] = state
    return updated


def _index(objects: Iterable[object]) -> dict[type, Any]:
    indexed: dict[type, Any] = {}
    for obj in objects:
        _put(indexed, obj)
    return indexed


def _put(indexed: dict[type, Any], obj: object) -> None:
    cls = type(obj)
    if cls in indexed:
        raise ValueError(f"Duplicate {cls.__name__}.")
    indexed[cls] = obj


def _expect(value: object, kind: type[Any], method: Callable[..., Any]) -> tuple[Any, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, tuple) else (value,)
    for item in items:
        if not isinstance(item, kind):
            raise TypeError(f"{_method_name(method)} must return {kind.__name__} or a tuple of {kind.__name__}, got {type(item).__name__}.")
    return items


def _method_name(method: Callable[..., Any]) -> str:
    return getattr(method, "__qualname__", getattr(method, "__name__", "method"))


def _bind(method: Callable[..., Any], available: Mapping[type, Any]) -> tuple[list[Any], dict[str, Any]]:
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for name, param, value in _resolved_parameters(method, available):
        if param.kind is param.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keyword[name] = value
    return positional, keyword


def _bound_values(method: Callable[..., Any], available: Mapping[type, Any]) -> list[Any]:
    """Values in signature order, for a positional ``step`` call."""
    values: list[Any] = []
    for name, param, value in _resolved_parameters(method, available):
        if param.kind is param.KEYWORD_ONLY:
            raise TypeError(f"{_method_name(method)} parameter {name} must be positional so the call can be planned by index.")
        values.append(value)
    return values


def _resolved_parameters(method: Callable[..., Any], available: Mapping[type, Any]):
    hints = _parameter_annotations(method)
    for name, param in signature(method).parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            raise TypeError(f"{_method_name(method)} parameter {name} cannot have a default.")
        annotation = hints.get(name)
        if not isinstance(annotation, type):
            raise TypeError(f"{_method_name(method)} parameter {name} must be annotated with a type, got {annotation!r}.")
        yield name, param, _match(annotation, available, method)


def _state_index(annotation: Any, states: Mapping[type[State], State], method: Callable[..., Any]) -> int:
    if not isinstance(annotation, type):
        raise TypeError(f"{_method_name(method)} return type must be None, a state, or a tuple of states, got {annotation!r}.")
    concrete = type(_match(annotation, states, method, action="returns"))
    for index, cls in enumerate(states):
        if cls is concrete:
            return index
    raise ValueError(f"{_method_name(method)} returns {concrete.__name__}, but that state is not held by the model.")


def _match(annotation: type, available: Mapping[type, Any], method: Callable[..., Any], *, action: str = "requires") -> Any:
    if annotation in available:
        return available[annotation]
    matches = [obj for cls, obj in available.items() if cls is not annotation and issubclass(cls, annotation)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(type(obj).__name__ for obj in matches)
        raise ValueError(f"{_method_name(method)} {action} {annotation.__name__}, but more than one concrete type can be used: {names}.")
    raise ValueError(f"{_method_name(method)} {action} {annotation.__name__}, but no {annotation.__name__} is available.")


def _parameter_annotations(method: Callable[..., Any]) -> dict[str, Any]:
    """Resolve parameter annotations, ignoring the return annotation.

    The return annotation is not needed to bind arguments. On ``Input`` it names
    ``Process`` / ``State`` that are imported only for type checkers.
    """
    func = getattr(method, "__func__", method)
    globalns = getattr(func, "__globals__", {})
    skipped = {name for name, param in signature(method).parameters.items() if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)}
    annotations = get_annotations(func, format=Format.FORWARDREF)
    resolved: dict[str, Any] = {}
    for name, value in annotations.items():
        if name == "return" or name in skipped:
            continue
        resolved[name] = _resolve_annotation(value, globalns)
    return resolved


def _return_annotation(method: Callable[..., Any]) -> Any:
    func = getattr(method, "__func__", method)
    globalns = getattr(func, "__globals__", {})
    annotations = get_annotations(func, format=Format.FORWARDREF)
    if "return" not in annotations:
        raise TypeError(f"{_method_name(method)} return type must be None, a state, or a tuple of states.")
    return _resolve_annotation(annotations["return"], globalns)


def _resolve_annotation(value: Any, globalns: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return eval(value, globalns, globalns)  # noqa: S307 — annotation expression, same as get_type_hints
    evaluate = getattr(value, "evaluate", None)
    if evaluate is not None:
        return evaluate(globals=globalns, locals=globalns, format=Format.VALUE)
    return value
