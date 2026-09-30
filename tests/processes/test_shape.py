"""Behaviour: an organ shape spreads a quantity between its bottom and top."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from orion.core.variable import var
from orion.processes.crop.shape import Rectangle


def test_rectangle_density_reads_the_organ_quantities():
    density = Rectangle("rectangle").density(
        var("area_index", "m^2/m^2", 2.0, "Area index"),
        var("bottom", "m", 0.0, "Bottom"),
        var("top", "m", 1.0, "Top"),
        var("height", "m", jnp.array([0.5, 1.5]), "Height"),
    )

    assert density == pytest.approx(jnp.array([2.0, 0.0]))
