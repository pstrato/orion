"""Behaviour of wheat site-year dataset records."""

from __future__ import annotations

from datetime import date

import pytest

from orion.core.constant import Constant
from orion.core.input import LocationInput
from orion.datasets.convert import grain_protein_from_nitrogen, to_kg_per_m2
from orion.datasets.events import fertiliser_event, irrigation_event, protection_event, sowing_event
from orion.datasets.io import point_location
from orion.datasets.observations import make_observation
from orion.datasets.site_year import SiteYear


def _location() -> LocationInput:
    return point_location("westerfeld", 11.702, 51.819)


def test_site_year_exposes_sowing_fertiliser_protection_yield_protein_and_dry_mass():
    site = SiteYear(
        name="westerfeld:2018:int-1",
        crop="wheat",
        cultivar="RGT Reform",
        season="2018",
        treatment="intensive",
        source="westerfeld",
        location=_location(),
        events=(
            sowing_event("sowing-2018-10-12", date(2018, 10, 12), density=350.0),
            fertiliser_event("fertiliser-2019-04-10", date(2019, 4, 10), n=to_kg_per_m2(80.0, "kg/ha"), product="KAS"),
            protection_event("protection-2019-05-20", date(2019, 5, 20), kind="fungicide", product="Prosaro", rate=0.8, rate_unit="l/ha"),
        ),
        observations=(
            make_observation("yield", date(2019, 7, 25), "yield", "kg/m^2", to_kg_per_m2(85.0, "dt/ha")),
            make_observation("grain_protein", date(2019, 7, 25), "grain_protein", "kg/kg", 0.125),
            make_observation("biomass", date(2019, 7, 25), "biomass", "kg/m^2", to_kg_per_m2(160.0, "dt/ha")),
        ),
    )

    assert site.crop == "wheat"
    sowing = site.sowings()[0]
    assert isinstance(sowing.on, Constant)
    assert sowing.on.unit == "isodate"
    assert sowing.on.value == date(2018, 10, 12)
    assert sowing.density is not None
    assert sowing.density.unit == "seeds/m^2"
    assert sowing.density.value == 350.0
    nitrogen = site.fertilisers()[0].n
    assert nitrogen.unit == "kg/m^2"
    assert nitrogen.value == pytest.approx(0.008)
    assert site.protections()[0].kind == "fungicide"
    assert site.protections()[0].product == "Prosaro"
    assert site.protections()[0].rate == 0.8
    yield_obs = site.latest("yield")
    protein = site.latest("grain_protein")
    biomass = site.latest("biomass")
    assert yield_obs is not None
    assert protein is not None
    assert biomass is not None
    assert yield_obs.reading.name == "yield"
    assert yield_obs.reading.unit == "kg/m^2"
    assert yield_obs.reading.value == 0.85
    assert protein.reading.value == pytest.approx(0.125)
    assert biomass.reading.value == 1.6
    assert site.grain_protein_fraction() == pytest.approx(0.125)
    child_names = {entity.name for _, entity in site.all_entities()}
    assert {"on", "density", "n", "yield", "grain_protein", "biomass"} <= child_names


def test_grain_nitrogen_observation_converts_to_grain_protein():
    site = SiteYear(
        name="broadbalk:2015:n6",
        crop="wheat",
        cultivar="Crusoe",
        season="2015",
        treatment="N6",
        source="broadbalk",
        location=point_location("broadbalk", -0.37, 51.81),
        events=(
            sowing_event("sowing", date(2014, 10, 15)),
            fertiliser_event("fertiliser", date(2015, 4, 1), n=to_kg_per_m2(192.0, "kg/ha")),
            protection_event("protection", date(2015, 4, 1), kind="standard"),
        ),
        observations=(make_observation("grain_n", date(2015, 8, 20), "grain_n", "kg/kg", 0.021),),
    )

    assert site.grain_protein_fraction() == pytest.approx(grain_protein_from_nitrogen(0.021))
    assert site.grain_protein_fraction() == pytest.approx(0.021 * 5.7)


def test_mass_units_convert_to_kilograms_per_square_metre():
    assert to_kg_per_m2(7.5, "t/ha") == 0.75
    assert to_kg_per_m2(85.0, "dt/ha") == 0.85
    assert to_kg_per_m2(80.0, "kg/ha") == 0.008
    assert to_kg_per_m2(0.4, "kg/m^2") == 0.4


def test_site_year_without_protection_rows_still_records_a_protection_event():
    site = SiteYear(
        name="agmip:maricopa:wet",
        crop="wheat",
        cultivar="Yecora Rojo",
        season="1993",
        treatment="WET-HIGHN",
        source="agmip_kassie",
        location=point_location("maricopa", -111.98, 33.06),
        events=(
            sowing_event("sowing", date(1992, 12, 15)),
            fertiliser_event("fertiliser", date(1992, 12, 15), n=to_kg_per_m2(261.0, "kg/ha")),
            irrigation_event("irrigation", date(1993, 3, 1), amount=40.0),
            protection_event("protection", date(1992, 12, 15), kind="standard"),
        ),
        observations=(
            make_observation("yield", date(1993, 5, 20), "yield", "kg/m^2", 0.75),
            make_observation("biomass", date(1993, 5, 20), "biomass", "kg/m^2", 1.5),
            make_observation("grain_n", date(1993, 5, 20), "grain_n", "kg/kg", 0.020),
        ),
    )

    assert len(site.protections()) == 1
    assert site.protections()[0].kind == "standard"
    assert site.irrigations()[0].amount.unit == "kg/m^2"
    assert site.irrigations()[0].amount.value == 40.0
    assert site.grain_protein_fraction() == pytest.approx(0.114)
