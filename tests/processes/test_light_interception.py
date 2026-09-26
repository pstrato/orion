"""Behaviour: Beer–Lambert light interception shares irradiance by organ height."""

from __future__ import annotations

import math
from dataclasses import replace

import jax.numpy as jnp
import pytest

from orion.core.axis import WITHIN_STEP
from orion.core.constant import const
from orion.core.entity import entity
from orion.core.parameter import param
from orion.core.variable import var
from orion.processes.crop.canopy_organ import CanopyOrgan, CanopyOrganInput, canopy_organ
from orion.processes.crop.crop import Crop
from orion.processes.crop.light_interception import BeerLambertLightInterceptionInput, BeerLambertLightInterceptionProcess, LightInterception
from orion.processes.crop.roots import Roots
from orion.processes.crop.shape import DownwardTriangle, Rectangle, Shape, UpwardTriangle
from orion.processes.weather import Weather

_INCOMING = 100.0


@entity()
class _Crop(Crop):
    above: tuple[CanopyOrgan, ...]

    @property
    def canopy(self) -> tuple[CanopyOrgan, ...]:
        return self.above

    @property
    def organs(self):
        return (self.roots, *self.above)


def _organ(name: str, *, bottom: float, top: float, area_index: float, k: float, shape: Shape | None = None) -> CanopyOrgan:
    return CanopyOrgan(
        name=name,
        constraint=None,
        top=var("top", "m", top, "Organ top"),
        bottom=var("bottom", "m", bottom, "Organ bottom"),
        area_index=var("area_index", "m^2/m^2", area_index, "Area index"),
        k=param("k", "1", k, "Extinction coefficient"),
        shape=const("shape", "1", Rectangle(name) if shape is None else shape, "Vertical area distribution"),
    )


def _weather(radiation: float | list[float]) -> Weather:
    values = jnp.asarray(radiation, dtype=jnp.float32).reshape(-1)
    return Weather(
        name="weather",
        constraint=None,
        Ts=var("Ts", "°C", 15.0, "Air temperature"),
        Ps=var("Ps", "kg/m^2", 0.0, "Precipitation"),
        Rs=var("Rs", "W/m^2", values, "Shortwave radiation", axes=(WITHIN_STEP,)),
    )


def _intercept(*organs: CanopyOrgan, radiation: float | list[float] = _INCOMING) -> LightInterception:
    crop = _Crop("crop", None, Roots("roots", None), organs)
    result = BeerLambertLightInterceptionProcess("beer_lambert").step(crop, _weather(radiation))
    assert len(result) == 1
    return result[0]


def _floats(value: jnp.ndarray) -> list[float]:
    return [float(item) for item in jnp.atleast_1d(value)]


def test_one_organ_intercepts_incoming_light_by_beer_lambert():
    k, area = 0.5, 2.0
    result = _intercept(_organ("leaves", bottom=0.0, top=1.0, area_index=area, k=k))
    soil = _INCOMING * math.exp(-k * area)
    assert _floats(result.soil.value) == pytest.approx([soil])
    assert _floats(result.plant.value) == pytest.approx([_INCOMING - soil])
    assert [flux.name for flux in result.canopy] == ["leaves"]
    assert _floats(result.canopy[0].value) == pytest.approx([_INCOMING - soil])


def test_organs_with_the_same_vertical_span_share_light_by_extinction():
    leaves_extinction, ears_extinction = 0.2 * 2.0, 0.8 * 1.0
    tau = leaves_extinction + ears_extinction
    absorbed = _INCOMING * (1.0 - math.exp(-tau))
    result = _intercept(
        _organ("leaves", bottom=0.0, top=1.0, area_index=2.0, k=0.2),
        _organ("ears", bottom=0.0, top=1.0, area_index=1.0, k=0.8),
    )
    assert _floats(result.canopy[0].value) == pytest.approx([absorbed * leaves_extinction / tau])
    assert _floats(result.canopy[1].value) == pytest.approx([absorbed * ears_extinction / tau])
    assert _floats(result.soil.value) == pytest.approx([_INCOMING * math.exp(-tau)])
    assert _floats(result.plant.value) == pytest.approx([absorbed])


def test_a_higher_organ_shades_the_organ_below_it():
    tau = 0.5 * 2.0
    upper = _INCOMING * (1.0 - math.exp(-tau))
    transmitted = _INCOMING * math.exp(-tau)
    lower = transmitted * (1.0 - math.exp(-tau))
    result = _intercept(
        _organ("leaves", bottom=0.0, top=1.0, area_index=2.0, k=0.5),
        _organ("ears", bottom=1.0, top=2.0, area_index=2.0, k=0.5),
    )
    assert [flux.name for flux in result.canopy] == ["leaves", "ears"]
    assert _floats(result.canopy[1].value) == pytest.approx([upper])
    assert _floats(result.canopy[0].value) == pytest.approx([lower])
    assert upper > lower
    assert _floats(result.soil.value) == pytest.approx([transmitted * math.exp(-tau)])
    assert _floats(result.plant.value) == pytest.approx([upper + lower])


def test_organs_compete_only_on_the_heights_they_share():
    # Leaves fill 0–2 m. Ears fill only the upper metre, so they compete there
    # and leaves alone occupy 0–1 m.
    top_tau = 0.5 * 1.0 + 0.5 * 1.0
    top_absorbed = _INCOMING * (1.0 - math.exp(-top_tau))
    entered_below = _INCOMING * math.exp(-top_tau)
    leaves_below = entered_below * (1.0 - math.exp(-0.5))
    result = _intercept(
        _organ("leaves", bottom=0.0, top=2.0, area_index=2.0, k=0.5),
        _organ("ears", bottom=1.0, top=2.0, area_index=1.0, k=0.5),
    )
    assert _floats(result.canopy[1].value) == pytest.approx([0.5 / top_tau * top_absorbed])
    assert _floats(result.canopy[0].value) == pytest.approx([0.5 / top_tau * top_absorbed + leaves_below])
    assert _floats(result.soil.value) == pytest.approx([entered_below * math.exp(-0.5)])


def test_organ_with_no_thickness_intercepts_no_light():
    result = _intercept(_organ("leaves", bottom=1.0, top=1.0, area_index=2.0, k=0.5))
    assert _floats(result.soil.value) == pytest.approx([_INCOMING])
    assert _floats(result.plant.value) == pytest.approx([0.0])
    assert _floats(result.canopy[0].value) == pytest.approx([0.0])


def test_organ_with_no_area_does_not_shade_the_canopy():
    tau = 0.5 * 2.0
    result = _intercept(
        _organ("ears", bottom=1.0, top=2.0, area_index=0.0, k=0.5),
        _organ("leaves", bottom=0.0, top=1.0, area_index=2.0, k=0.5),
    )
    assert _floats(result.canopy[0].value) == pytest.approx([0.0])
    assert _floats(result.canopy[1].value) == pytest.approx([_INCOMING * (1.0 - math.exp(-tau))])
    assert _floats(result.soil.value) == pytest.approx([_INCOMING * math.exp(-tau)])


def test_a_downward_triangle_intercepts_more_than_an_upward_triangle_on_the_same_span():
    area, k = 1.0, 0.5
    tau = 2.0 * k * area
    wide_at_top = _INCOMING * (tau - 1.0 + math.exp(-tau)) / tau
    wide_at_bottom = _INCOMING * (1.0 - math.exp(-tau)) - wide_at_top
    result = _intercept(
        _organ("leaves", bottom=0.0, top=1.0, area_index=area, k=k, shape=UpwardTriangle("upward")),
        _organ("stems", bottom=0.0, top=1.0, area_index=area, k=k, shape=DownwardTriangle("downward")),
    )
    assert wide_at_top > wide_at_bottom
    assert _floats(result.canopy[0].value) == pytest.approx([wide_at_bottom])
    assert _floats(result.canopy[1].value) == pytest.approx([wide_at_top])
    assert _floats(result.soil.value) == pytest.approx([_INCOMING * math.exp(-tau)])
    assert _floats(result.plant.value) == pytest.approx([wide_at_top + wide_at_bottom])


def test_a_triangle_extinguishes_its_whole_area_index():
    soil = _INCOMING * math.exp(-0.4 * 1.5)
    for shape in (UpwardTriangle("upward"), DownwardTriangle("downward")):
        result = _intercept(_organ("leaves", bottom=0.2, top=1.4, area_index=1.5, k=0.4, shape=shape))
        assert _floats(result.soil.value) == pytest.approx([soil])
        assert _floats(result.canopy[0].value) == pytest.approx([_INCOMING - soil])


def test_step_radiation_is_the_weather_radiation_total():
    result = _intercept(_organ("leaves", bottom=0.0, top=1.0, area_index=0.0, k=0.5), radiation=[0.0, 100.0, 200.0])
    assert _floats(result.soil.value) == pytest.approx([300.0])


def test_organ_built_from_its_input_uses_that_extinction_coefficient():
    leaves = CanopyOrganInput("leaves", param("k", "1", 0.5, "Extinction coefficient"), const("shape", "1", Rectangle("rectangle"), "Uniform area profile"))
    organ = replace(
        canopy_organ("leaves", leaves),
        top=var("top", "m", 1.0, "Organ top"),
        bottom=var("bottom", "m", 0.0, "Organ bottom"),
        area_index=var("area_index", "m^2/m^2", 2.0, "Area index"),
    )
    result = _intercept(organ)
    soil = _INCOMING * math.exp(-0.5 * 2.0)
    assert _floats(result.canopy[0].value) == pytest.approx([_INCOMING - soil])


def test_input_exposes_one_flux_slot_per_canopy_organ():
    crop = _Crop(
        "crop",
        None,
        Roots("roots", None),
        (
            _organ("leaves", bottom=0.0, top=1.0, area_index=1.0, k=0.5),
            _organ("ears", bottom=1.0, top=1.4, area_index=0.2, k=0.6),
        ),
    )
    state = BeerLambertLightInterceptionInput("beer-lambert").states(crop)
    assert isinstance(state, LightInterception)
    assert [flux.name for flux in state.canopy] == ["leaves", "ears"]
    assert _floats(state.soil.value) == pytest.approx([0.0])
    assert _floats(state.plant.value) == pytest.approx([0.0])
    assert _floats(state.canopy[0].value) == pytest.approx([0.0])
    process = BeerLambertLightInterceptionInput("beer-lambert").processes()
    assert isinstance(process, BeerLambertLightInterceptionProcess)
