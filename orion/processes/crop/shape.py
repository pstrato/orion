"""Vertical distribution of an organ quantity above the soil."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from orion.core.entity import Entity, entity
from orion.core.quantity import Quantity


def _as_array(quantity: Quantity) -> jnp.ndarray:
    return jnp.asarray(quantity.value)


class Shape(Protocol):
    """Organ shape: how a quantity is spread between the organ bottom and top."""

    def density(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, height: Quantity) -> jnp.ndarray:
        """Area per metre at ``height``. Zero outside the organ."""
        ...

    def intersect(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, layer_bottom: Quantity, layer_top: Quantity) -> jnp.ndarray:
        """Part of ``quantity`` that falls inside the layer."""
        ...


def _bounds(quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    amount = _as_array(quantity)
    bottom = _as_array(organ_bottom)
    top = _as_array(organ_top)
    span = top - bottom
    active = (span > 0) & (amount > 0)
    return amount, bottom, top, jnp.where(span > 0, span, 1.0), active


def _inside(bottom: jnp.ndarray, top: jnp.ndarray, height: Quantity, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    z = _as_array(height)
    return z, active & (z > bottom) & (z < top)


def _overlap(bottom: jnp.ndarray, top: jnp.ndarray, span: jnp.ndarray, active: jnp.ndarray, layer_bottom: Quantity, layer_top: Quantity) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    lo = jnp.maximum(bottom, _as_array(layer_bottom))
    hi = jnp.minimum(top, _as_array(layer_top))
    u0 = (lo - bottom) / span
    u1 = (hi - bottom) / span
    covered = active & (hi > lo)
    return u0, u1, covered


@entity()
class Rectangle(Entity, Shape):
    """Constant area density between bottom and top."""

    def density(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, height: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, amount / span, jnp.zeros_like(z, dtype=amount.dtype))

    def intersect(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, layer_bottom: Quantity, layer_top: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        return jnp.where(covered, amount * (u1 - u0), jnp.zeros_like(u1))


@entity()
class DownwardTriangle(Entity, Shape):
    """Area density rises linearly from 0 at the bottom to a maximum at the top."""

    def density(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, height: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, 2.0 * amount * (z - bottom) / jnp.square(span), jnp.zeros_like(z, dtype=amount.dtype))

    def intersect(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, layer_bottom: Quantity, layer_top: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        return jnp.where(covered, amount * (jnp.square(u1) - jnp.square(u0)), jnp.zeros_like(u1))


@entity()
class UpwardTriangle(Entity, Shape):
    """Area density falls linearly from a maximum at the bottom to 0 at the top."""

    def density(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, height: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, 2.0 * amount * (top - z) / jnp.square(span), jnp.zeros_like(z, dtype=amount.dtype))

    def intersect(self, quantity: Quantity, organ_bottom: Quantity, organ_top: Quantity, layer_bottom: Quantity, layer_top: Quantity) -> jnp.ndarray:
        amount, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        fraction = jnp.square(1.0 - u0) - jnp.square(1.0 - u1)
        return jnp.where(covered, amount * fraction, jnp.zeros_like(u1))
