"""Canopy light interception.

Organs compete by height. Every organ top and bottom is a layer boundary.
Inside a layer the area density of a rectangle or a triangle is linear, so
the extinction density is linear too. Light travels from the top downward,
and each organ absorbs in proportion to ``k`` times its density at that height.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from orion.core.entity import entity
from orion.core.input import Input
from orion.core.process import Process
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.state import State
from orion.core.variable import Variable, var
from orion.processes.crop.canopy_organ import CanopyOrgan
from orion.processes.crop.crop import Crop
from orion.processes.crop.shape import Shape
from orion.processes.weather import Weather


@entity()
class LightInterception(State):
    """Outcome of the light interception process for one step."""

    soil: Variable
    """Light reaching the soil (W/m²)."""
    plant: Variable
    """Light intercepted by the canopy (W/m²)."""
    canopy: tuple[Variable, ...]
    """Light intercepted by each canopy organ (W/m²), in ``crop.canopy`` order."""


@entity()
class LightInterceptionProcess(Process):
    """Canopy light interception."""

    def step(self, crop: Crop, weather: Weather) -> tuple[LightInterception]:
        raise NotImplementedError()


# Gauss–Legendre nodes on [-1, 1]. Enough for a linear density times the
# exponential light profile inside one layer.
_GL_XI, _GL_W = (jnp.asarray(values) for values in np.polynomial.legendre.leggauss(8))


def beer_lambert_interception(
    ks: jnp.ndarray,
    area_indexes: jnp.ndarray,
    bottoms: jnp.ndarray,
    tops: jnp.ndarray,
    incoming: jnp.ndarray,
    shapes: tuple[Shape, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Intercept one canopy.

    ``ks``, ``area_indexes``, ``bottoms`` and ``tops`` have shape ``(organ,)``.
    ``incoming`` is a scalar. ``shapes[i]`` is that organ's profile (rectangle,
    upward triangle, or downward triangle). Returns soil, plant, and per-organ
    interception in organ order.
    """
    bounds = jnp.sort(jnp.concatenate([bottoms, tops], axis=0), axis=0)
    layer_lo = jnp.flip(bounds[:-1], axis=0)
    layer_hi = jnp.flip(bounds[1:], axis=0)
    height = layer_hi - layer_lo
    dtype = incoming.dtype
    xi = _GL_XI.astype(dtype)
    weights = _GL_W.astype(dtype)
    fraction = jnp.concatenate([(xi + 1.0) * 0.5, jnp.asarray([0.25, 0.75], dtype=dtype)])
    sample_z = layer_lo[:, None] + fraction[None, :] * height[:, None]
    density = jnp.stack([shapes[index].density(area_indexes[index], bottoms[index], tops[index], sample_z) for index in range(len(shapes))])
    extinction = jnp.sum(ks[:, None, None] * density, axis=0)
    mu_quarter, mu_three_quarter = extinction[:, -2], extinction[:, -1]
    mu_lo = (3.0 * mu_quarter - mu_three_quarter) / 2.0
    mu_hi = (3.0 * mu_three_quarter - mu_quarter) / 2.0
    tau_layer = height * (mu_lo + mu_hi) * 0.5
    entered = incoming * jnp.exp(-(jnp.cumsum(tau_layer) - tau_layer))
    t = fraction[:-2]
    depth = mu_lo[:, None] * (0.5 - t + 0.5 * jnp.square(t)) + mu_hi[:, None] * (0.5 * (1.0 - jnp.square(t)))
    light = entered[:, None] * jnp.exp(-height[:, None] * depth)
    dz = weights[None, :] * height[:, None] * 0.5
    raw = jnp.sum(ks[:, None, None] * density[:, :, :-2] * light[None, :, :] * dz[None, :, :], axis=2)
    absorbed = entered * (-jnp.expm1(-tau_layer))
    raw_total = jnp.sum(raw, axis=0)
    share = raw * jnp.where(raw_total > 0, absorbed / jnp.where(raw_total > 0, raw_total, 1.0), 0.0)
    per_organ = jnp.sum(share, axis=1)
    soil = incoming * jnp.exp(-jnp.sum(tau_layer))
    plant = jnp.sum(per_organ)
    return soil, plant, per_organ


def _column(organs: tuple[CanopyOrgan, ...], component) -> jnp.ndarray:
    return jnp.stack([jnp.asarray(component(organ)) for organ in organs])


def _intercept(organs: tuple[CanopyOrgan, ...], incoming: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return beer_lambert_interception(
        _column(organs, lambda organ: organ.k.value),
        _column(organs, lambda organ: organ.area_index.value),
        _column(organs, lambda organ: organ.bottom.value),
        _column(organs, lambda organ: organ.top.value),
        incoming,
        tuple(organ.shape.value for organ in organs),
    )


def _light(name: str, value: jnp.ndarray, description: str) -> Variable:
    return var(name, "W/m^2", value, description, is_scalar + is_non_negative)


def _outcome(organs: tuple[CanopyOrgan, ...], soil: jnp.ndarray, plant: jnp.ndarray, per_organ: jnp.ndarray) -> LightInterception:
    return LightInterception(
        name="light_interception",
        constraint=None,
        soil=_light("soil", soil, "Light reaching the soil"),
        plant=_light("plant", plant, "Light intercepted by the canopy"),
        canopy=tuple(_light(organ.name, per_organ[index], f"Light intercepted by {organ.name}") for index, organ in enumerate(organs)),
    )


@entity()
class BeerLambertLightInterceptionProcess(LightInterceptionProcess):
    """Beer–Lambert interception with organs competing by their vertical span."""

    def step(self, crop: Crop, weather: Weather) -> tuple[LightInterception]:
        organs = crop.canopy
        soil, plant, per_organ = _intercept(organs, weather.R.value)
        return (_outcome(organs, soil, plant, per_organ),)


@entity()
class BeerLambertLightInterceptionInput(Input):
    """Beer–Lambert canopy light interception."""

    def states(self, crop: Crop) -> LightInterception:
        """Initial fluxes: one canopy slot per organ, all zero."""
        organs = crop.canopy
        zero = jnp.zeros(())
        return _outcome(organs, zero, zero, jnp.zeros((len(organs),)))

    def processes(self) -> BeerLambertLightInterceptionProcess:
        return BeerLambertLightInterceptionProcess(self.name)
