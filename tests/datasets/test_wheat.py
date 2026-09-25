"""Behaviour: wheat archive parsers produce site-years with management and observations."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from orion.datasets import (
    DATASETS_CACHE,
    SiteYear,
    cache_wheat_archive,
    grain_protein_from_nitrogen,
    list_wheat_site_years,
    parse_agmip_kassie,
    parse_braunschweig,
    parse_broadbalk,
    parse_management_pack,
    parse_muncheberg,
    parse_westerfeld,
    to_kg_per_m2,
)


def _write_csv(path: Path, header: str, *rows: str) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _observed(site: SiteYear, quantity: str) -> float:
    observed = site.latest(quantity)
    assert observed is not None
    return observed.reading.value


def test_westerfeld_tables_produce_wheat_site_year_with_protection(tmp_path: Path):
    _write_csv(
        tmp_path / "PLOT.csv",
        "Plot_ID,Latitude,Longitude,Treatment,Crop",
        "1,51.819,11.702,intensive,winter wheat",
        "2,51.819,11.702,intensive,grain maize",
    )
    _write_csv(
        tmp_path / "SOWING.csv",
        "Plot_ID,Experimental_Year,Crop,Cultivar,Date,Seeding_Rate",
        "1,2018,winter wheat,RGT Reform,2018-10-12,350",
        "2,2018,grain maize,P8000,2018-04-20,8",
    )
    _write_csv(
        tmp_path / "FERTILIZATION.csv",
        "Plot_ID,Experimental_Year,Date,N_kg_ha,P_kg_ha,K_kg_ha,Product",
        "1,2018,2019-04-10,80,0,0,KAS",
    )
    _write_csv(
        tmp_path / "PLANT_PROTECTION.csv",
        "Plot_ID,Experimental_Year,Date,Type,Product,Rate_l_ha,Rate_kg_ha",
        "1,2018,2019-05-20,fungicide,Prosaro,0.8,",
    )
    _write_csv(
        tmp_path / "YIELD.csv",
        "Plot_ID,Experimental_Year,Yield_dt_ha,Protein_percent,Thousand_kernel_mass_g,Aboveground_biomass_dt_ha",
        "1,2018,85.0,12.5,42.0,160.0",
    )

    years = parse_westerfeld(tmp_path)
    assert len(years) == 1
    site = years[0]
    assert site.crop == "wheat"
    assert site.cultivar == "RGT Reform"
    assert site.treatment == "intensive"
    assert site.source == "westerfeld"
    sowing = site.sowings()[0]
    assert sowing.on.value == date(2018, 10, 12)
    density = sowing.density
    assert density is not None
    assert density.value == 350.0
    assert density.unit == "seeds/m^2"
    assert site.fertilisers()[0].n.value == to_kg_per_m2(80.0, "kg/ha")
    assert site.protections()[0].kind == "fungicide"
    assert site.protections()[0].product == "Prosaro"
    assert _observed(site, "yield") == to_kg_per_m2(85.0, "dt/ha")
    assert _observed(site, "grain_protein") == pytest.approx(0.125)
    assert _observed(site, "biomass") == to_kg_per_m2(160.0, "dt/ha")


def test_muncheberg_tables_produce_site_year_with_biomass_and_crop_n(tmp_path: Path):
    _write_csv(
        tmp_path / "plots.csv",
        "plot,latitude,longitude,intensity,cultivar",
        "1,52.52,14.12,intensive,Borenos",
    )
    _write_csv(
        tmp_path / "management.csv",
        "plot,date,action,product,n_kg_ha,p_kg_ha,k_kg_ha,intensity",
        "1,1994-10-05,sowing,Borenos,0,0,0,intensive",
        "1,1995-03-20,fertiliser,KAS,120,0,0,intensive",
        "1,1995-05-12,protection,Opus,0,0,0,intensive",
    )
    _write_csv(
        tmp_path / "crop.csv",
        "plot,date,bbch,aboveground_biomass_dt_ha,root_biomass_dt_ha,n_percent,yield_dt_ha,grain_n_percent,grain_protein_percent",
        "1,1995-06-15,65,90.0,12.0,1.8,,2.0,",
        "1,1995-07-28,89,150.0,14.0,1.2,72.0,2.0,11.4",
    )

    years = parse_muncheberg(tmp_path)
    assert len(years) == 1
    site = years[0]
    assert site.source == "muncheberg"
    assert site.crop == "wheat"
    assert site.sowings()[0].on.value == date(1994, 10, 5)
    assert site.fertilisers()[0].n.value == to_kg_per_m2(120.0, "kg/ha")
    assert site.protections()[0].kind == "fungicide"
    assert site.protections()[0].product == "Opus"
    assert _observed(site, "yield") == to_kg_per_m2(72.0, "dt/ha")
    assert _observed(site, "biomass") == to_kg_per_m2(150.0, "dt/ha")
    assert _observed(site, "grain_protein") == pytest.approx(0.114)
    assert any(o.reading.name == "phenology_stage" for o in site.observations)


def test_broadbalk_tables_record_fungicide_exclusion(tmp_path: Path):
    _write_csv(
        tmp_path / "yields.csv",
        "year,section,plot,sowing_date,harvest_date,cultivar,n_kg_ha,grain_t_ha,straw_t_ha,grain_n_percent,herbicides,fungicides",
        "2015,6,9,2014-10-15,2015-08-20,Crusoe,192,7.1,6.4,2.1,yes,no",
        "2015,1,9,2014-10-15,2015-08-20,Crusoe,192,8.0,7.0,2.0,yes,yes",
    )

    years = parse_broadbalk(tmp_path)
    assert len(years) == 2
    excluded = next(site for site in years if site.treatment.endswith("section-6-plot-9"))
    protected = next(site for site in years if site.treatment.endswith("section-1-plot-9"))
    assert excluded.source == "broadbalk"
    assert any(event.kind == "none" and event.product == "fungicide" for event in excluded.protections())
    assert any(event.kind == "standard" for event in protected.protections())
    assert excluded.sowings()[0].on.value == date(2014, 10, 15)
    assert excluded.fertilisers()[0].n.value == to_kg_per_m2(192.0, "kg/ha")
    assert _observed(excluded, "yield") == to_kg_per_m2(7.1, "t/ha")
    assert _observed(excluded, "biomass") == to_kg_per_m2(7.1 + 6.4, "t/ha")
    assert excluded.grain_protein_fraction() == pytest.approx(grain_protein_from_nitrogen(0.021))


def test_braunschweig_management_records_pesticide_and_grain_protein(tmp_path: Path):
    _write_csv(
        tmp_path / "management.csv",
        "date,measure,product,rate,n_kg_ha,irrigation_mm,treatment,cultivar",
        "2013-10-29,sowing,Batis,380,0,0,N190,Batis",
        "2014-04-08,fertilisation,KAS,190,190,0,N190,Batis",
        "2014-05-20,pesticide,Capalo,1.4,0,0,N190,Batis",
        "2014-06-01,irrigation,,0,0,20,N190,Batis",
    )
    _write_csv(
        tmp_path / "grain_quality.csv",
        "treatment,date,protein_percent",
        "N190,2014-08-04,13.2",
    )
    _write_csv(
        tmp_path / "biomass.csv",
        "treatment,date,yield_t_ha,biomass_t_ha",
        "N190,2014-08-04,8.1,16.2",
    )

    years = parse_braunschweig(tmp_path)
    assert len(years) == 1
    site = years[0]
    assert site.source == "braunschweig"
    assert site.cultivar == "Batis"
    assert site.sowings()[0].on.value == date(2013, 10, 29)
    assert site.fertilisers()[0].n.value == to_kg_per_m2(190.0, "kg/ha")
    assert site.protections()[0].kind == "fungicide"
    assert site.protections()[0].product == "Capalo"
    assert site.irrigations()[0].amount.value == 20.0
    assert _observed(site, "yield") == to_kg_per_m2(8.1, "t/ha")
    assert _observed(site, "biomass") == to_kg_per_m2(16.2, "t/ha")
    assert site.grain_protein_fraction() == pytest.approx(0.132)


def test_agmip_pack_uses_standard_protection_when_sprays_are_not_listed(tmp_path: Path):
    _write_csv(
        tmp_path / "management.csv",
        "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude",
        "Maricopa,WET-HIGHN,1992-12-15,261,40,Yecora Rojo,33.06,-111.98",
        "Lincoln,IRRIGATED,1991-06-01,150,80,Rongotea,-43.65,172.48",
    )
    _write_csv(
        tmp_path / "experimental_data.csv",
        "site,treatment,date,variable,value,unit",
        "Maricopa,WET-HIGHN,1993-05-20,yield,7.5,t/ha",
        "Maricopa,WET-HIGHN,1993-05-20,biomass,15.0,t/ha",
        "Maricopa,WET-HIGHN,1993-05-20,grain_n,2.1,%",
        "Lincoln,IRRIGATED,1992-01-20,yield,9.0,t/ha",
        "Lincoln,IRRIGATED,1992-01-20,biomass,18.0,t/ha",
        "Lincoln,IRRIGATED,1992-01-20,grain_n,1.9,%",
    )

    years = parse_agmip_kassie(tmp_path)
    assert len(years) == 2
    maricopa = next(site for site in years if site.treatment == "WET-HIGHN")
    assert maricopa.source == "agmip_kassie"
    assert maricopa.protections()[0].kind == "standard"
    assert maricopa.sowings()[0].on.value == date(1992, 12, 15)
    assert maricopa.fertilisers()[0].n.value == to_kg_per_m2(261.0, "kg/ha")
    assert maricopa.irrigations()[0].amount.value == 40.0
    assert _observed(maricopa, "yield") == to_kg_per_m2(7.5, "t/ha")
    assert _observed(maricopa, "biomass") == to_kg_per_m2(15.0, "t/ha")
    assert maricopa.grain_protein_fraction() == pytest.approx(grain_protein_from_nitrogen(0.021))


def test_management_pack_uses_named_source_and_protection_column(tmp_path: Path):
    _write_csv(
        tmp_path / "management.csv",
        "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude,protection",
        "Kiel,HN-WF,2018-10-05,220,0,Julius,54.31,10.13,fungicide",
        "Kiel,HN-NF,2018-10-05,220,0,Julius,54.31,10.13,none",
    )
    _write_csv(
        tmp_path / "experimental_data.csv",
        "site,treatment,date,variable,value,unit",
        "Kiel,HN-WF,2019-08-05,yield,9.8,t/ha",
        "Kiel,HN-WF,2019-08-05,biomass,18.0,t/ha",
        "Kiel,HN-WF,2019-08-05,grain_protein,12.5,%",
        "Kiel,HN-NF,2019-08-05,yield,8.2,t/ha",
        "Kiel,HN-NF,2019-08-05,biomass,16.0,t/ha",
        "Kiel,HN-NF,2019-08-05,grain_protein,12.8,%",
    )

    years = parse_management_pack(tmp_path, source="german_met")
    assert {site.source for site in years} == {"german_met"}
    protected = next(site for site in years if site.treatment == "HN-WF")
    untreated = next(site for site in years if site.treatment == "HN-NF")
    assert protected.protections()[0].kind == "fungicide"
    assert untreated.protections()[0].kind == "none"
    assert protected.sowings()[0].on.value == date(2018, 10, 5)
    assert protected.fertilisers()[0].n.value == to_kg_per_m2(220.0, "kg/ha")
    assert _observed(protected, "yield") == to_kg_per_m2(9.8, "t/ha")
    assert _observed(protected, "grain_protein") == pytest.approx(0.125)


def test_list_wheat_site_years_reads_cached_archives(tmp_path: Path):
    cache = tmp_path / "cache"
    westerfeld = {
        "PLOT.csv": "Plot_ID,Latitude,Longitude,Treatment,Crop\n1,51.819,11.702,extensive,winter wheat\n",
        "SOWING.csv": "Plot_ID,Experimental_Year,Crop,Cultivar,Date,Seeding_Rate\n1,2019,winter wheat,RGT Reform,2019-10-08,320\n",
        "FERTILIZATION.csv": "Plot_ID,Experimental_Year,Date,N_kg_ha,P_kg_ha,K_kg_ha,Product\n1,2019,2020-04-02,40,0,0,KAS\n",
        "PLANT_PROTECTION.csv": "Plot_ID,Experimental_Year,Date,Type,Product,Rate_l_ha,Rate_kg_ha\n1,2019,2020-04-20,herbicide,Atlantis,0.45,\n",
        "YIELD.csv": "Plot_ID,Experimental_Year,Yield_dt_ha,Protein_percent,Thousand_kernel_mass_g,Aboveground_biomass_dt_ha\n1,2019,62.0,11.1,38.0,120.0\n",
    }
    agmip = {
        "management.csv": "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude\nCunderdin,RAIN,1999-05-20,80,0,Wyalkatchem,-31.65,117.24\n",
        "experimental_data.csv": "site,treatment,date,variable,value,unit\nCunderdin,RAIN,1999-11-15,yield,3.2,t/ha\nCunderdin,RAIN,1999-11-15,biomass,7.0,t/ha\nCunderdin,RAIN,1999-11-15,grain_n,1.8,%\n",
    }
    cache_wheat_archive(cache, "westerfeld", westerfeld)
    cache_wheat_archive(cache, "agmip_kassie", agmip)

    years = list_wheat_site_years(cache)
    names = {site.source for site in years}
    assert names == {"westerfeld", "agmip_kassie"}
    assert (cache / DATASETS_CACHE / "westerfeld" / "SOWING.csv").is_file()
    assert (cache / DATASETS_CACHE / "agmip_kassie" / "management.csv").is_file()
    westerfeld_site = next(site for site in years if site.source == "westerfeld")
    assert westerfeld_site.protections()[0].kind == "herbicide"
    agmip_site = next(site for site in years if site.source == "agmip_kassie")
    assert agmip_site.protections()[0].kind == "standard"


def test_list_wheat_site_years_includes_worldwide_management_packs(tmp_path: Path):
    cache = tmp_path / "cache"
    wageningen = {
        "management.csv": "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude\nDe Bouwing,N3,1982-10-21,160,0,Arminda,51.95,5.75\n",
        "experimental_data.csv": "site,treatment,date,variable,value,unit\nDe Bouwing,N3,1983-08-01,yield,8.3,t/ha\nDe Bouwing,N3,1983-08-01,biomass,16.5,t/ha\nDe Bouwing,N3,1983-08-01,grain_n,2.1,%\n",
    }
    ludhiana = {
        "management.csv": "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude\nLudhiana,IRRIGATED,2006-11-10,120,200,PBW 343,30.90,75.85\n",
        "experimental_data.csv": "site,treatment,date,variable,value,unit\nLudhiana,IRRIGATED,2007-04-10,yield,4.8,t/ha\nLudhiana,IRRIGATED,2007-04-10,biomass,11.0,t/ha\nLudhiana,IRRIGATED,2007-04-10,grain_n,1.8,%\n",
    }
    cache_wheat_archive(cache, "wageningen", wageningen)
    cache_wheat_archive(cache, "ludhiana", ludhiana)

    years = list_wheat_site_years(cache)
    names = {site.source for site in years}
    assert names == {"wageningen", "ludhiana"}
    netherlands = next(site for site in years if site.source == "wageningen")
    assert netherlands.cultivar == "Arminda"
    india = next(site for site in years if site.source == "ludhiana")
    assert india.irrigations()[0].amount.value == to_kg_per_m2(200.0, "mm")


def test_list_wheat_site_years_is_empty_when_cache_has_no_archives(tmp_path: Path):
    assert list_wheat_site_years(tmp_path / "empty") == ()
