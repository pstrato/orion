from __future__ import annotations

import jax.lax
import jax.numpy as jnp

from orion.core.axis import WITHIN_STEP
from orion.core.entity import entity
from orion.core.input import Input
from orion.core.process import Process
from orion.core.quantity import is_finite, is_non_negative, is_scalar
from orion.core.state import State
from orion.core.variable import Variable, var
from orion.processes.clock import Clock, ClockInput


@entity()
class WeatherInput(Input):
    """Weather inputs."""

    Ts: Variable
    """All air temperature in °C."""
    Ps: Variable
    """All precipitation in kg/m² (1 mm water ≡ 1 kg/m²)."""
    Rs: Variable
    """All shortwave radiation in W/m²."""

    def states(self, clock: ClockInput):
        zeros = jnp.zeros((clock.delta.value,))
        return Weather(
            name="weather",
            constraint=None,
            Ts=var("Ts", "°C", zeros, description="Air temperature within the step", axes=(WITHIN_STEP,), constraint=is_finite, resource="heat"),
            Ps=var(
                "Ps",
                "kg/m^2",
                zeros,
                description="Precipitation within the step (kg/m²; 1 mm ≡ 1 kg/m²)",
                axes=(WITHIN_STEP,),
                constraint=is_non_negative + is_finite,
                resource="water",
            ),
            Rs=var(
                "Rs",
                "W/m^2",
                zeros,
                description="Shortwave radiation within the step",
                axes=(WITHIN_STEP,),
                constraint=is_non_negative + is_finite,
                resource="light",
            ),
        )


@entity()
class Weather(State):
    """Weather state (optionally batched on the leading axis)."""

    Ts: Variable
    """Air temperature in °C."""
    Ps: Variable
    """Precipitation in kg/m² (1 mm water ≡ 1 kg/m²)."""
    Rs: Variable
    """Shortwave radiation in W/m²."""

    @property
    def Tmin(self):
        return Variable("Minimum temperature", is_scalar, "°C", self.Ts.value.min(axis=-1), "Minimum air temperature.", (), self.Ts.resource)

    @property
    def Tmax(self):
        return Variable("Maximum temperature", is_scalar, "°C", self.Ts.value.max(axis=-1), "Maximum air temperature.", (), self.Ts.resource)

    def Tsum(self, base: float | jnp.ndarray = 0):
        zero = jnp.zeros(())
        sum = jax.lax.scan(
            lambda carry, t: (carry + jnp.maximum(t - base, zero), zero),
            zero,
            self.Ts.value,
        )[0]

        return Variable("Temperature sum", is_scalar, "°C", sum, "Sum of air temperature.", (), self.Ts.resource)

    @property
    def P(self):
        return Variable("Precipitation", is_scalar, "kg/m^2", self.Ps.value.sum(axis=-1), "Total precipitation.", (), self.Ps.resource)

    @property
    def R(self):
        return Variable("Radiation", is_scalar, "W/m^2", self.Rs.value.sum(axis=-1), "Total radiation.", (), self.Rs.resource)


@entity()
class WeatherProcess(Process):
    """Weather process."""

    def step(self, clock: Clock, input: WeatherInput, weather: Weather) -> Weather:
        """Advance weather by slicing archive series for the current step."""
        start = clock.has
        delta = int(clock.delta.value)
        return Weather(
            name=weather.name,
            Ts=weather.Ts.set(jax.lax.dynamic_slice(input.Ts.value, (start,), (delta,))),
            Ps=weather.Ps.set(jax.lax.dynamic_slice(input.Ps.value, (start + 1,), (delta,))),
            Rs=weather.Rs.set(jax.lax.dynamic_slice(input.Rs.value, (start,), (delta,))),
            constraint=weather.constraint,
        )
