"""Vertical distribution of an organ quantity above the soil."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from orion.core.entity import Entity, entity


class Shape(Protocol):
    """Organ shape: how a quantity is spread between the organ bottom and top."""

    def density(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, height: jnp.ndarray) -> jnp.ndarray:
        """Area per metre at ``height``. Zero outside the organ."""
        ...

    def intersect(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, layer_bottom: jnp.ndarray, layer_top: jnp.ndarray) -> jnp.ndarray:
        """Part of ``quantity`` that falls inside the layer."""
        ...


def _bounds(quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    quantity = jnp.asarray(quantity)
    bottom = jnp.asarray(organ_bottom)
    top = jnp.asarray(organ_top)
    span = top - bottom
    active = (span > 0) & (quantity > 0)
    return quantity, bottom, top, jnp.where(span > 0, span, 1.0), active


def _inside(bottom: jnp.ndarray, top: jnp.ndarray, height: jnp.ndarray, active: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    z = jnp.asarray(height)
    return z, active & (z > bottom) & (z < top)


def _overlap(bottom: jnp.ndarray, top: jnp.ndarray, span: jnp.ndarray, active: jnp.ndarray, layer_bottom: jnp.ndarray, layer_top: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    lo = jnp.maximum(bottom, jnp.asarray(layer_bottom))
    hi = jnp.minimum(top, jnp.asarray(layer_top))
    u0 = (lo - bottom) / span
    u1 = (hi - bottom) / span
    covered = active & (hi > lo)
    return u0, u1, covered


@entity()
class Rectangle(Entity, Shape):
    """Constant area density between bottom and top."""

    def density(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, height: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, quantity / span, jnp.zeros_like(z, dtype=quantity.dtype))

    def intersect(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, layer_bottom: jnp.ndarray, layer_top: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        return jnp.where(covered, quantity * (u1 - u0), jnp.zeros_like(u1))


@entity()
class DownwardTriangle(Entity, Shape):
    """Area density rises linearly from 0 at the bottom to a maximum at the top."""

    def density(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, height: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, 2.0 * quantity * (z - bottom) / jnp.square(span), jnp.zeros_like(z, dtype=quantity.dtype))

    def intersect(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, layer_bottom: jnp.ndarray, layer_top: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        return jnp.where(covered, quantity * (jnp.square(u1) - jnp.square(u0)), jnp.zeros_like(u1))


@entity()
class UpwardTriangle(Entity, Shape):
    """Area density falls linearly from a maximum at the bottom to 0 at the top."""

    def density(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, height: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        z, inside = _inside(bottom, top, height, active)
        return jnp.where(inside, 2.0 * quantity * (top - z) / jnp.square(span), jnp.zeros_like(z, dtype=quantity.dtype))

    def intersect(self, quantity: jnp.ndarray, organ_bottom: jnp.ndarray, organ_top: jnp.ndarray, layer_bottom: jnp.ndarray, layer_top: jnp.ndarray) -> jnp.ndarray:
        quantity, bottom, top, span, active = _bounds(quantity, organ_bottom, organ_top)
        u0, u1, covered = _overlap(bottom, top, span, active, layer_bottom, layer_top)
        fraction = jnp.square(1.0 - u0) - jnp.square(1.0 - u1)
        return jnp.where(covered, quantity * fraction, jnp.zeros_like(u1))
