"""Behaviour: management packs carry in-season growth series and dated nutrient and water events."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from orion.datasets import parse_management_pack, to_kg_per_m2


def _write(path: Path, header: str, *rows: str) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def test_experimental_table_maps_organ_biomass_lai_and_phenology(tmp_path: Path):
    _write(
        tmp_path / "management.csv",
        "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude",
        "De Eest,N3,1982-10-19,140,0,Arminda,52.6167,5.75",
    )
    _write(
        tmp_path / "experimental_data.csv",
        "site,treatment,date,variable,value,unit",
        "De Eest,N3,1983-05-11,lai,3.5,m^2/m^2",
        "De Eest,N3,1983-05-11,laid,4.0,1",
        "De Eest,N3,1983-05-11,lwad,800,kg/ha",
        "De Eest,N3,1983-05-11,swad,1200,kg/ha",
        "De Eest,N3,1983-05-11,ewad,200,kg/ha",
        "De Eest,N3,1983-05-11,rwad,400,kg/ha",
        "De Eest,N3,1983-05-11,cwad,2200,kg/ha",
        "De Eest,N3,1983-05-11,bbch,31,BBCH",
        "De Eest,N3,1983-05-11,canopy_n,2.4,%",
        "De Eest,N3,1983-05-11,ndvi,0.8,1",
    )

    site = parse_management_pack(tmp_path, source="eest")[0]
    assert [item.reading.value for item in site.observations if item.reading.name == "lai"] == pytest.approx([3.5, 4.0])
    leaf = site.latest("leaf_biomass")
    stem = site.latest("stem_biomass")
    ear = site.latest("ear_biomass")
    root = site.latest("root_biomass")
    biomass = site.latest("biomass")
    stage = site.latest("phenology_stage")
    canopy_n = site.latest("canopy_n")
    assert leaf is not None and leaf.reading.unit == "kg/m^2" and leaf.reading.value == pytest.approx(to_kg_per_m2(800.0, "kg/ha"))
    assert stem is not None and stem.reading.value == pytest.approx(to_kg_per_m2(1200.0, "kg/ha"))
    assert ear is not None and ear.reading.value == pytest.approx(to_kg_per_m2(200.0, "kg/ha"))
    assert root is not None and root.reading.value == pytest.approx(to_kg_per_m2(400.0, "kg/ha"))
    assert biomass is not None and biomass.reading.value == pytest.approx(to_kg_per_m2(2200.0, "kg/ha"))
    assert stage is not None and stage.reading.value == pytest.approx(31.0) and stage.reading.unit == "step"
    assert canopy_n is not None and canopy_n.reading.unit == "kg/kg" and canopy_n.reading.value == pytest.approx(0.024)
    assert all(item.reading.name != "ndvi" for item in site.observations)


def test_dated_fertiliser_and_irrigation_replace_seasonal_totals(tmp_path: Path):
    _write(
        tmp_path / "management.csv",
        "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude,harvest_date",
        "De Eest,N1,1982-10-19,160,40,Arminda,52.6167,5.75,1983-08-03",
    )
    _write(
        tmp_path / "fertiliser.csv",
        "site,treatment,date,n_kg_ha,p_kg_ha,k_kg_ha,product",
        "De Eest,N1,1983-02-14,0,0,0,CAN",
        "De Eest,N1,1983-05-11,0,0,0,CAN",
        "De Eest,N1,1983-06-21,0,0,0,CAN",
    )
    _write(
        tmp_path / "irrigation.csv",
        "site,treatment,date,irrigation_mm",
        "De Eest,N1,1983-06-01,12",
    )
    _write(tmp_path / "experimental_data.csv", "site,treatment,date,variable,value,unit")

    site = parse_management_pack(tmp_path, source="eest")[0]
    assert [event.on.value for event in site.fertilisers()] == [date(1983, 2, 14), date(1983, 5, 11), date(1983, 6, 21)]
    assert all(event.n.value == 0.0 for event in site.fertilisers())
    assert site.fertilisers()[0].product == "CAN"
    assert len(site.irrigations()) == 1
    assert site.irrigations()[0].on.value == date(1983, 6, 1)
    assert site.irrigations()[0].amount.value == pytest.approx(12.0)
    stage = site.latest("phenology_stage")
    assert stage is not None
    assert stage.reading.value == pytest.approx(89.0)
    assert stage.on.value == date(1983, 8, 3)
    assert site.horizon() == (date(1982, 10, 19), date(1983, 8, 4))


def test_blank_nitrogen_does_not_invent_a_zero_application(tmp_path: Path):
    _write(
        tmp_path / "management.csv",
        "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude",
        "PAGV,1983,1982-10-25,,,Arminda,52.5,5.5",
    )
    _write(tmp_path / "experimental_data.csv", "site,treatment,date,variable,value,unit")

    site = parse_management_pack(tmp_path, source="pagv")[0]
    assert site.fertilisers() == ()
    assert site.irrigations() == ()
