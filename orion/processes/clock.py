from __future__ import annotations

import jax.numpy as jnp
import jax_datetime as jdt

from orion.core.constant import Constant
from orion.core.entity import entity
from orion.core.input import Input
from orion.core.process import Process
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.state import State
from orion.core.variable import JaxDate, Variable, var


@entity()
class ClockInput(Input):
    """Clock input: provides initial ``Clock`` state from start / end / step."""

    start: Variable[JaxDate]
    """Simulation start date."""
    end: Variable[JaxDate]
    """Simulation end date."""
    delta: Constant[int]
    """Simulation step duration in hours."""

    def states(self):
        """Initial clock state from start / end / step."""
        return Clock(
            name="clock",
            start=self.start,
            delta=self.delta,
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

    start: Variable[JaxDate]
    """Simulation start date."""

    delta: Constant[int]
    """Delta step."""

    step: Variable
    """Simulation step."""

    @property
    def date(self) -> Variable[JaxDate]:
        """Current date, including hours since the start date."""
        return var("date", "isodate", _at(self), "Current simulation date")

    @property
    def doy(self) -> Variable:
        """Day of year of the current date. 1 January is 1."""
        return var("doy", "days", _day_of_year(self.date), "Day of year", is_scalar + is_non_negative)

    @property
    def das(self) -> Variable:
        """Days after the simulation start."""
        return var("das", "days", jnp.asarray((_at(self) - self.start.value).days), "Days after start", is_scalar + is_non_negative)

    @property
    def has(self) -> Variable:
        """Hours after the simulation start."""
        return var("has", "hours", self.step.value * self.delta.value, "Hours after start", is_scalar + is_non_negative)


def _at(clock: Clock) -> jdt.Datetime:
    """Clock date as a JAX datetime."""
    hours = jnp.asarray(clock.step.value * clock.delta.value).astype(jnp.int32)
    return clock.start.value + jdt.to_timedelta(hours, "h")


def _trunc_div(numerator: jnp.ndarray, denominator: int) -> jnp.ndarray:
    return jnp.trunc(numerator / denominator).astype(jnp.int32)


def _day_of_year(current: Variable[JaxDate]) -> jnp.ndarray:
    """Calendar day of year from days since the Unix epoch."""
    shifted = jnp.asarray(current.value.delta.days).astype(jnp.int32) + jnp.int32(719468)
    era_days = jnp.where(shifted >= 0, shifted, shifted - jnp.int32(146096))
    era = _trunc_div(era_days, 146097)
    day_of_era = shifted - era * jnp.int32(146097)
    year_of_era = _trunc_div(day_of_era - day_of_era // 1460 + day_of_era // 36524 - day_of_era // 146096, 365)
    year = year_of_era + era * jnp.int32(400)
    march_doy = day_of_era - (jnp.int32(365) * year_of_era + year_of_era // 4 - year_of_era // 100)
    month_index = _trunc_div(jnp.int32(5) * march_doy + jnp.int32(2), 153)
    day = march_doy - _trunc_div(jnp.int32(153) * month_index + jnp.int32(2), 5) + jnp.int32(1)
    month = jnp.where(month_index < 10, month_index + jnp.int32(3), month_index - jnp.int32(9))
    year = jnp.where(month <= 2, year + jnp.int32(1), year)
    before = jnp.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=jnp.int32)[month - jnp.int32(1)]
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    before = jnp.where(leap & (month > 2), before + jnp.int32(1), before)
    return before + day


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
