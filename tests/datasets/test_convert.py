"""Behaviour: published units convert into Orion SI quantity units."""

from __future__ import annotations

import pytest

from orion.datasets.convert import to_kg_per_kg, to_kg_per_m2, to_kg_per_m3


def test_mass_per_area_converts_to_kg_per_m2():
    assert to_kg_per_m2(0.4, "kg/m^2") == pytest.approx(0.4)
    assert to_kg_per_m2(400.0, "g/m^2") == pytest.approx(0.4)
    assert to_kg_per_m2(7.5, "t/ha") == pytest.approx(0.75)
    assert to_kg_per_m2(85.0, "dt/ha") == pytest.approx(0.85)
    assert to_kg_per_m2(80.0, "kg/ha") == pytest.approx(0.008)


def test_water_depth_mm_is_kg_per_m2():
    assert to_kg_per_m2(20.0, "mm") == pytest.approx(20.0)
    assert to_kg_per_m2(5.0, "mm/m^2") == pytest.approx(5.0)


def test_bulk_density_converts_to_kg_per_m3():
    assert to_kg_per_m3(1300.0, "kg/m^3") == pytest.approx(1300.0)
    assert to_kg_per_m3(1.3, "g/cm^3") == pytest.approx(1300.0)
    assert to_kg_per_m3(1.3, "kg/dm^3") == pytest.approx(1300.0)


def test_unknown_units_raise():
    with pytest.raises(ValueError, match="kg/m"):
        to_kg_per_m2(1.0, "lb/acre")
    with pytest.raises(ValueError, match="kg/m"):
        to_kg_per_m3(1.0, "lb/ft^3")


def test_concentration_converts_to_kg_per_kg():
    assert to_kg_per_kg(0.12, "kg/kg") == pytest.approx(0.12)
    assert to_kg_per_kg(12.0, "%") == pytest.approx(0.12)
    assert to_kg_per_kg(120.0, "g/kg") == pytest.approx(0.12)
    assert to_kg_per_kg(120_000.0, "mg/kg") == pytest.approx(0.12)
