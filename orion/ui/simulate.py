"""Finish planned inputs with core's batched, compiled simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import jax

from orion.core.constant import Constant
from orion.core.input import Inputs
from orion.core.model import Model, clock_steps, simulate_all, validate
from orion.core.process import Process
from orion.core.setting import Settings

Job = tuple[Inputs, tuple[Process, ...]]


def finish_inputs(settings: Settings, jobs: Sequence[Job]) -> tuple[tuple[Model, Model], ...]:
    """Run every inputs to its clock horizon.

    Each result is the final model and a history whose variables hold one value per step.
    Inputs that share constants and processes run together in one compiled batch.
    """
    if not jobs:
        return ()
    results: list[tuple[Model, Model] | None] = [None] * len(jobs)
    for bucket in _batches(jobs):
        batch = tuple(jobs[index][0] for index in bucket)
        processes = jobs[bucket[0]][1]
        validate(settings, batch, processes)
        finals, histories = simulate_all(settings, batch, processes, jit=True, keep_history=True)
        if histories is None:
            raise RuntimeError("simulate_all did not keep a history.")
        for offset, index in enumerate(bucket):
            name = jobs[index][0].name
            final = replace(_slice_batch(finals, offset, len(bucket)), name=name)
            history = replace(_slice_batch(histories, offset, len(bucket)), name=name)
            results[index] = (final, history)
    return tuple(cast(tuple[Model, Model], pair) for pair in results)


def _batches(jobs: Sequence[Job]) -> tuple[tuple[int, ...], ...]:
    buckets: list[list[int]] = []
    for index, job in enumerate(jobs):
        for bucket in buckets:
            if _same_batch(jobs[bucket[0]], job):
                bucket.append(index)
                break
        else:
            buckets.append([index])
    return tuple(tuple(bucket) for bucket in buckets)


def _same_batch(left: Job, right: Job) -> bool:
    left_inputs, left_processes = left
    right_inputs, right_processes = right
    if tuple((type(process), process.name) for process in left_processes) != tuple((type(process), process.name) for process in right_processes):
        return False
    return _constants(left_inputs) == _constants(right_inputs) and clock_steps(left_inputs) == clock_steps(right_inputs)


def _constants(inputs: Inputs) -> tuple[Constant, ...]:
    return tuple(item for _, item in inputs.all_entities(of_type=Constant) if isinstance(item, Constant))


def _slice_batch(model: Model, index: int, count: int) -> Model:
    """Take one inputs row. A single inputs has no inputs axis to remove."""
    del count
    drop_inputs = bool(model.axes) and model.axes[0].name == "inputs"

    def take(leaf: object) -> object:
        if drop_inputs and isinstance(leaf, jax.Array) and leaf.ndim >= 1:
            return leaf[index]
        return leaf

    sliced = cast(Model, jax.tree.map(take, model))
    if not drop_inputs:
        return sliced
    return replace(sliced, axes=model.axes[1:])
