"""Behaviour: calibration archives expose published wheat site-years for crop-growth development."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from orion.datasets.convert import to_kg_per_m2
from orion.datasets.demo import available_site_years


def _source(cache: Path, source: str):
    years = available_site_years(cache)
    return tuple(site for site in years if site.source == source)


def test_eest_n1_records_arminda_sowing_zero_nitrogen_dates_and_harvest(tmp_path: Path):
    site = _source(tmp_path, "eest")[0]
    assert site.cultivar == "Arminda"
    assert site.treatment == "N1"
    assert site.sowings()[0].on.value == date(1982, 10, 19)
    assert site.location.geometry.value.y == pytest.approx(52.6167)
    assert site.location.geometry.value.x == pytest.approx(5.75)
    assert [event.on.value for event in site.fertilisers()] == [date(1983, 2, 14), date(1983, 5, 11), date(1983, 6, 21)]
    assert all(event.n.value == 0.0 and event.n.unit == "kg/m^2" and event.product == "CAN" for event in site.fertilisers())
    assert site.irrigations() == ()
    stage = site.latest("phenology_stage")
    assert stage is not None
    assert stage.on.value == date(1983, 8, 3)
    assert stage.reading.unit == "step"
    assert stage.reading.value == pytest.approx(89.0)


def test_pagv_records_sowing_and_harvest_without_a_nitrogen_rate(tmp_path: Path):
    site = _source(tmp_path, "pagv")[0]
    assert site.cultivar == "Arminda"
    assert site.sowings()[0].on.value == date(1982, 10, 25)
    assert site.location.geometry.value.y == pytest.approx(52.5)
    assert site.location.geometry.value.x == pytest.approx(5.5)
    assert site.fertilisers() == ()
    stage = site.latest("phenology_stage")
    assert stage is not None
    assert stage.on.value == date(1983, 8, 2)


def test_wheat_site_years_keep_the_altitude_of_each_experiment(tmp_path: Path):
    years = available_site_years(tmp_path)
    found = {(site.source, site.location.name): float(site.location.altitude.value) for site in years}
    expected = {
        ("westerfeld", "westerfeld"): 94.0,
        ("broadbalk", "broadbalk"): 130.0,
        ("muncheberg", "muncheberg"): 62.0,
        ("braunschweig", "braunschweig"): 79.0,
        ("agmip_kassie", "Maricopa"): 361.0,
        ("agmip_kassie", "Lincoln"): 7.0,
        ("wageningen", "De Bouwing"): 7.0,
        ("cunderdin", "Cunderdin"): 220.0,
        ("obregon", "Obregon"): 38.0,
        ("luancheng", "Luancheng"): 50.1,
        ("ludhiana", "Ludhiana"): 247.0,
        ("balcarce", "Balcarce"): 130.0,
        ("egypt_nile", "Sakha"): 6.0,
        ("iwyp_valdivia", "Valdivia"): 12.0,
        ("german_met", "Kiel"): 8.0,
        ("eest", "De Eest"): -4.0,
        ("pagv", "PAGV"): -5.0,
        ("hot_serial_cereal", "Maricopa"): 361.0,
    }
    assert set(found) == set(expected)
    for key, altitude in expected.items():
        assert found[key] == pytest.approx(altitude)


def test_hot_serial_cereal_control_sowings_use_published_density_and_planting_nitrogen(tmp_path: Path):
    years = _source(tmp_path, "hot_serial_cereal")
    assert {site.sowings()[0].on.value for site in years} == {date(2007, 3, 13), date(2008, 2, 13), date(2008, 3, 13), date(2009, 1, 12)}
    for site in years:
        assert site.cultivar == "Yecora Rojo"
        sowing = site.sowings()[0]
        density = sowing.density
        assert density is not None
        assert density.unit == "seeds/m^2"
        assert density.value == pytest.approx(288.0)
        assert site.location.geometry.value.y == pytest.approx(33.0667)
        assert site.location.geometry.value.x == pytest.approx(-111.9667)
        assert len(site.fertilisers()) == 1
        assert site.fertilisers()[0].on.value == sowing.on.value
        assert site.fertilisers()[0].n.unit == "kg/m^2"
        assert site.fertilisers()[0].n.value == pytest.approx(to_kg_per_m2(50.0, "kg/ha"))
        assert site.fertilisers()[0].product == "ammonium phosphate"
        assert site.irrigations() == ()
