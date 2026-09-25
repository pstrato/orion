from __future__ import annotations

from datetime import date, timedelta

import jax.numpy as jnp

from orion.core.constant import Constant, const
from orion.core.entity import entity
from orion.core.input import Input
from orion.core.process import Process
from orion.core.state import State
from orion.core.variable import Variable


@entity()
class ClockInput(Input):
    """Clock input: provides initial ``Clock`` state from start / end / step."""

    start: Constant[date]
    """Simulation start date."""
    end: Constant[date]
    """Simulation end date."""
    delta: Constant[int]
    """Simulation step duration in hours."""

    def states(self):
        """Initial clock state from start / end / step."""
        from orion.core.quantity import is_non_negative, is_scalar
        from orion.core.variable import var
        from orion.processes.clock import Clock

        return Clock(
            name="clock",
            start=const("start", "isodate", self.start.value, "Simulation start date"),
            delta=const("delta", "hours", self.delta.value, "Simulation delta step in hours"),
            step=var(
                "step",
                "step",
                0,
                description="Current simulation step index",
                constraint=is_scalar + is_non_negative,
            ),
            constraint=None,
        )

    def processes(self):
        """Clock process to advance the clock state each step."""
        return ClockProcess("clock")


@entity()
class Clock(State):
    """State of the clock process."""

    start: Constant[date]
    """Simulation start date."""

    delta: Constant[int]
    """Delta step."""

    step: Variable
    """Simulation step."""

    @property
    def date(self) -> date:
        return self.start.value + timedelta(hours=int(self.step.value * self.delta.value))

    @property
    def doy(self) -> int:
        return self.date.timetuple().tm_yday

    @property
    def das(self) -> int:
        return (self.date - self.start.value).days

    @property
    def has(self) -> jnp.ndarray:
        return self.step.value * self.delta.value


@entity()
class ClockProcess(Process):
    """Clock process: advances ``Clock.step`` each model step."""

    def step(self, clock: Clock) -> Clock:
        return Clock(
            name=clock.name,
            start=clock.start,
            delta=clock.delta,
            step=clock.step.set(clock.step.value + 1),
            constraint=clock.constraint,
        )
