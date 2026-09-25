from __future__ import annotations

import jax.lax
import jax.numpy as jnp

from orion.core.entity import entity
from orion.core.state import State
from orion.core.variable import Variable


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
    def Tmin(self) -> jnp.ndarray:
        return self.Ts.value.min(axis=-1)

    @property
    def Tmax(self) -> jnp.ndarray:
        return self.Ts.value.max(axis=-1)

    def Tsum(self, base: float | jnp.ndarray = 0) -> jnp.ndarray:
        zero = jnp.zeros(())
        flat = self.Ts.value.reshape(-1)
        return jax.lax.scan(
            lambda carry, t: (carry + jnp.maximum(t - base, zero), zero),
            zero,
            flat,
        )[0]

    @property
    def P(self) -> jnp.ndarray:
        return self.Ps.value.sum(axis=-1)

    @property
    def R(self) -> jnp.ndarray:
        return self.Rs.value.sum(axis=-1)
