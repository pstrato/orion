"""Shared helpers for default soil process implementations."""

from __future__ import annotations

import jax.numpy as jnp

from orion.processes.soil.soil import SoilLayer


def temperature_factor(temperature_c: jnp.ndarray) -> jnp.ndarray:
    """Q10-style factor centred on 20 °C."""
    return jnp.exp(0.1 * (temperature_c - 20.0))


def field_capacity(layer: SoilLayer) -> jnp.ndarray:
    """Rough field capacity (kg/m²) from bulk density and clay."""
    thickness_m = jnp.asarray(layer.bottom.value) - jnp.asarray(layer.top.value)
    bdod = jnp.asarray(layer.bdod.value)
    clay = jnp.asarray(layer.clay.value)
    theta = 0.1 + 0.4 * clay
    return theta * thickness_m * 1000.0 * (bdod / 1300.0)
