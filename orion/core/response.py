"""Responses map an input through a monotone function onto the unit interval."""

from __future__ import annotations

from typing import cast

import jax.nn
import jax.numpy as jnp

from orion.core.constant import Constant
from orion.core.entity import Entity, entity
from orion.core.parameter import Parameter
from orion.core.quantity import between_0_1_inc, is_finite
from orion.core.variable import Variable, var


class Response(Entity):
    """A model response."""

    def __call__(self, value: Variable | Parameter) -> Variable:
        """Response at given value."""
        raise NotImplementedError()


class Function(Entity):
    """A function used in a response. A function is defined between 0 and 1 and monotonic increasing from 0 to 1."""

    def __call__(self, x: Variable | Parameter) -> Variable:
        raise NotImplementedError()


class Identity(Function):
    """Identity response."""

    def __call__(self, x: Variable | Parameter) -> Variable:
        return cast(Variable, x + 0)


class Sigmoid(Function):
    """Sigmoid response."""

    def __call__(self, x: Variable | Parameter) -> Variable:
        return var(f"sig({x.name})", "unitless", jax.nn.sigmoid(x.value), x.description, is_finite + between_0_1_inc, x.axes, x.dimension)


def _limited(value: Variable | Parameter, fraction: Variable) -> Variable:
    """Fraction clamped to the unit interval, where the response function is defined."""
    return var(value.name, fraction.unit, jnp.clip(jnp.asarray(fraction.value), 0.0, 1.0), value.description, axes=fraction.axes)


def _zo(func: Function, zero: Parameter, one: Parameter, value: Variable | Parameter) -> Variable:
    fraction = cast(Variable, (value - zero) / (one - zero))
    return func(_limited(value, fraction))


@entity()
class ZO(Response):
    """Zero to one response."""

    func: Constant[Function]
    """Function applied from 0 to 1."""

    z: Parameter
    """Value below which the response is 0."""

    o: Parameter
    """Value above which the response is 1."""

    def __call__(self, value: Variable | Parameter) -> Variable:
        return _zo(self.func.value, self.z, self.o, value)


@entity()
class ZOOZ(Response):
    """Zero to one, then one to zero."""

    func: Constant[Function]
    """Function applied from 0 to 1, and mirrored from 1 to 0."""

    lz: Parameter
    """Value below which the response is 0."""

    lo: Parameter
    """Value at which the response reaches 1."""

    ho: Parameter
    """Value at which the response leaves 1."""

    hz: Parameter
    """Value above which the response is 0."""

    def __call__(self, value: Variable | Parameter) -> Variable:
        rise = _zo(self.func.value, self.lz, self.lo, value)
        reflected = cast(Variable, self.ho + self.hz - value)
        fall = _zo(self.func.value, self.ho, self.hz, reflected)
        return cast(Variable, rise * fall)
