"""Shared helpers for default soil process implementations."""

from __future__ import annotations

import jax.numpy as jnp

from orion.core.quantity import Quantity
from orion.core.variable import Variable, var
from orion.processes.soil.soil import SoilLayer


def temperature_factor(temperature: Quantity) -> Variable:
    """Q10-style factor centred on 20 °C."""
    factor = jnp.exp(0.1 * (jnp.asarray(temperature.value) - 20.0))
    return var("temperature_factor", "unitless", factor, "Q10-style factor centred on 20 °C")


def field_capacity(layer: SoilLayer) -> Variable:
    """Rough field capacity from bulk density, clay, and layer thickness."""
    thickness_m = jnp.asarray(layer.thickness.value)
    bdod = jnp.asarray(layer.bdod.value)
    clay = jnp.asarray(layer.clay.value)
    theta = 0.1 + 0.4 * clay
    return var("field_capacity", "kg/m^2", theta * thickness_m * 1000.0 * (bdod / 1300.0), "Field capacity", dimension="water")
