"""Demo wheat archives so the Inputs tab has datasets without a prior download."""

from __future__ import annotations

from pathlib import Path

from orion.datasets.site_year import SiteYear
from orion.datasets.wheat.catalogue import DATASETS_CACHE, cache_wheat_archive, list_wheat_site_years


def _csv(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _pack(*rows: str, observations: tuple[str, ...], protection: bool = False) -> dict[str, str]:
    header = "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude"
    if protection:
        header += ",protection"
    return {
        "management.csv": _csv(header, *rows),
        "experimental_data.csv": _csv("site,treatment,date,variable,value,unit", *observations),
    }


DEMO_ARCHIVES: dict[str, dict[str, str]] = {
    "westerfeld": {
        "PLOT.csv": _csv("Plot_ID,Latitude,Longitude,Treatment,Crop", "1,51.819,11.702,intensive,winter wheat"),
        "SOWING.csv": _csv("Plot_ID,Experimental_Year,Crop,Cultivar,Date,Seeding_Rate", "1,2018,winter wheat,RGT Reform,2018-10-12,350"),
        "FERTILIZATION.csv": _csv("Plot_ID,Experimental_Year,Date,N_kg_ha,P_kg_ha,K_kg_ha,Product", "1,2018,2019-04-10,80,0,0,KAS"),
        "PLANT_PROTECTION.csv": _csv("Plot_ID,Experimental_Year,Date,Type,Product,Rate_l_ha,Rate_kg_ha", "1,2018,2019-05-20,fungicide,Prosaro,0.8,"),
        "YIELD.csv": _csv(
            "Plot_ID,Experimental_Year,Yield_dt_ha,Protein_percent,Thousand_kernel_mass_g,Aboveground_biomass_dt_ha",
            "1,2018,85.0,12.5,42.0,160.0",
        ),
    },
    "agmip_kassie": _pack(
        "Maricopa,WET-HIGHN,1992-12-15,261,40,Yecora Rojo,33.06,-111.98",
        "Lincoln,IRRIGATED,1991-06-01,150,80,Rongotea,-43.65,172.48",
        observations=(
            "Maricopa,WET-HIGHN,1993-05-20,yield,7.5,t/ha",
            "Maricopa,WET-HIGHN,1993-05-20,biomass,15.0,t/ha",
            "Maricopa,WET-HIGHN,1993-05-20,grain_n,2.1,%",
            "Lincoln,IRRIGATED,1992-01-20,yield,9.0,t/ha",
            "Lincoln,IRRIGATED,1992-01-20,biomass,18.0,t/ha",
            "Lincoln,IRRIGATED,1992-01-20,grain_n,1.9,%",
        ),
    ),
    "broadbalk": {
        "yields.csv": _csv(
            "year,section,plot,sowing_date,harvest_date,cultivar,n_kg_ha,grain_t_ha,straw_t_ha,grain_n_percent,herbicides,fungicides",
            "2015,6,9,2014-10-15,2015-08-20,Crusoe,192,7.1,6.4,2.1,yes,no",
            "2015,1,9,2014-10-15,2015-08-20,Crusoe,192,8.0,7.0,2.0,yes,yes",
        ),
    },
    "muncheberg": {
        "plots.csv": _csv("plot,latitude,longitude,intensity,cultivar", "1,52.52,14.12,intensive,Borenos"),
        "management.csv": _csv(
            "plot,date,action,product,n_kg_ha,p_kg_ha,k_kg_ha,intensity",
            "1,1994-10-05,sowing,Borenos,0,0,0,intensive",
            "1,1995-03-20,fertiliser,KAS,120,0,0,intensive",
            "1,1995-05-12,protection,Opus,0,0,0,intensive",
        ),
        "crop.csv": _csv(
            "plot,date,bbch,aboveground_biomass_dt_ha,root_biomass_dt_ha,n_percent,yield_dt_ha,grain_n_percent,grain_protein_percent",
            "1,1995-06-15,65,90.0,12.0,1.8,,2.0,",
            "1,1995-07-28,89,150.0,14.0,1.2,72.0,2.0,11.4",
        ),
    },
    "braunschweig": {
        "management.csv": _csv(
            "date,measure,product,rate,n_kg_ha,irrigation_mm,treatment,cultivar",
            "2013-10-29,sowing,Batis,380,0,0,N190,Batis",
            "2014-04-08,fertilisation,KAS,190,190,0,N190,Batis",
            "2014-05-20,pesticide,Capalo,1.4,0,0,N190,Batis",
            "2014-06-01,irrigation,,0,0,20,N190,Batis",
        ),
        "grain_quality.csv": _csv("treatment,date,protein_percent", "N190,2014-08-04,13.2"),
        "biomass.csv": _csv("treatment,date,yield_t_ha,biomass_t_ha", "N190,2014-08-04,8.1,16.2"),
    },
    "wageningen": _pack(
        "De Bouwing,N1,1982-10-21,0,0,Arminda,51.95,5.75",
        "De Bouwing,N3,1982-10-21,160,0,Arminda,51.95,5.75",
        observations=(
            "De Bouwing,N1,1983-08-01,yield,5.4,t/ha",
            "De Bouwing,N1,1983-08-01,biomass,11.0,t/ha",
            "De Bouwing,N1,1983-08-01,grain_n,1.6,%",
            "De Bouwing,N3,1983-08-01,yield,8.3,t/ha",
            "De Bouwing,N3,1983-08-01,biomass,16.5,t/ha",
            "De Bouwing,N3,1983-08-01,grain_n,2.1,%",
        ),
    ),
    "cunderdin": _pack(
        "Cunderdin,RAIN,1997-06-06,50,0,Wilgoyne,-31.65,117.24",
        observations=(
            "Cunderdin,RAIN,1997-11-15,yield,2.4,t/ha",
            "Cunderdin,RAIN,1997-11-15,biomass,6.8,t/ha",
            "Cunderdin,RAIN,1997-11-15,grain_n,1.9,%",
        ),
    ),
    "obregon": _pack(
        "Obregon,IRRIGATED,1994-11-23,225,420,Yecora Rojo,27.33,-109.93",
        observations=(
            "Obregon,IRRIGATED,1995-04-15,yield,6.5,t/ha",
            "Obregon,IRRIGATED,1995-04-15,biomass,14.0,t/ha",
            "Obregon,IRRIGATED,1995-04-15,grain_n,2.2,%",
        ),
    ),
    "luancheng": _pack(
        "Luancheng,IRRIGATED,1998-10-05,200,150,Jimai 22,37.89,114.67",
        observations=(
            "Luancheng,IRRIGATED,1999-06-10,yield,6.2,t/ha",
            "Luancheng,IRRIGATED,1999-06-10,biomass,13.5,t/ha",
            "Luancheng,IRRIGATED,1999-06-10,grain_n,2.0,%",
        ),
    ),
    "ludhiana": _pack(
        "Ludhiana,IRRIGATED,2006-11-10,120,200,PBW 343,30.90,75.85",
        observations=(
            "Ludhiana,IRRIGATED,2007-04-10,yield,4.8,t/ha",
            "Ludhiana,IRRIGATED,2007-04-10,biomass,11.0,t/ha",
            "Ludhiana,IRRIGATED,2007-04-10,grain_n,1.8,%",
        ),
    ),
    "balcarce": _pack(
        "Balcarce,RAIN,1992-07-20,110,0,Klein Escorpion,-37.75,-58.30",
        observations=(
            "Balcarce,RAIN,1992-12-20,yield,5.5,t/ha",
            "Balcarce,RAIN,1992-12-20,biomass,12.0,t/ha",
            "Balcarce,RAIN,1992-12-20,grain_n,2.0,%",
        ),
    ),
    "egypt_nile": _pack(
        "Sakha,IRRIGATED,2009-11-20,180,400,Sakha 93,31.09,30.95",
        observations=(
            "Sakha,IRRIGATED,2010-04-25,yield,7.2,t/ha",
            "Sakha,IRRIGATED,2010-04-25,biomass,15.0,t/ha",
            "Sakha,IRRIGATED,2010-04-25,grain_n,2.0,%",
        ),
    ),
    "iwyp_valdivia": _pack(
        "Valdivia,HIGH-YIELD,2008-09-01,250,80,Bacanora,-39.78,-73.23",
        observations=(
            "Valdivia,HIGH-YIELD,2009-02-15,yield,12.0,t/ha",
            "Valdivia,HIGH-YIELD,2009-02-15,biomass,22.0,t/ha",
            "Valdivia,HIGH-YIELD,2009-02-15,grain_n,1.9,%",
        ),
    ),
    "german_met": _pack(
        "Kiel,HN-WF,2018-10-05,220,0,Julius,54.31,10.13,fungicide",
        "Kiel,HN-NF,2018-10-05,220,0,Julius,54.31,10.13,none",
        observations=(
            "Kiel,HN-WF,2019-08-05,yield,9.8,t/ha",
            "Kiel,HN-WF,2019-08-05,biomass,18.0,t/ha",
            "Kiel,HN-WF,2019-08-05,grain_protein,12.5,%",
            "Kiel,HN-NF,2019-08-05,yield,8.2,t/ha",
            "Kiel,HN-NF,2019-08-05,biomass,16.0,t/ha",
            "Kiel,HN-NF,2019-08-05,grain_protein,12.8,%",
        ),
        protection=True,
    ),
    # Groot & Verberne (1991): De Eest N1, winter wheat cv. Arminda. Application dates are the
    # published zero-N schedule; harvest is recorded as BBCH 89.
    "eest": {
        "management.csv": _csv(
            "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude,harvest_date",
            "De Eest,N1,1982-10-19,,,Arminda,52.6167,5.75,1983-08-03",
        ),
        "fertiliser.csv": _csv(
            "site,treatment,date,n_kg_ha,p_kg_ha,k_kg_ha,product",
            "De Eest,N1,1983-02-14,0,0,0,CAN",
            "De Eest,N1,1983-05-11,0,0,0,CAN",
            "De Eest,N1,1983-06-21,0,0,0,CAN",
        ),
        "experimental_data.csv": _csv("site,treatment,date,variable,value,unit"),
    },
    # Groot & Verberne (1991): PAGV (Lelystad). Sowing and harvest only; N rates are not in this extract.
    "pagv": {
        "management.csv": _csv(
            "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude,harvest_date",
            "PAGV,1983,1982-10-25,,,Arminda,52.5,5.5,1983-08-02",
        ),
        "experimental_data.csv": _csv("site,treatment,date,variable,value,unit"),
    },
    # Hot Serial Cereal (Kimball et al.; Maricopa). Named control sowings, 288 seeds/m²,
    # ~50 kg N/ha ammonium phosphate at planting. Seasonal irrigation is not in this extract.
    "hot_serial_cereal": {
        "management.csv": _csv(
            "site,treatment,sowing_date,n_kg_ha,irrigation_mm,cultivar,latitude,longitude,seeding_rate",
            "Maricopa,2007-03-13,2007-03-13,,,Yecora Rojo,33.0667,-111.9667,288",
            "Maricopa,2008-02-13,2008-02-13,,,Yecora Rojo,33.0667,-111.9667,288",
            "Maricopa,2008-03-13,2008-03-13,,,Yecora Rojo,33.0667,-111.9667,288",
            "Maricopa,2009-01-12,2009-01-12,,,Yecora Rojo,33.0667,-111.9667,288",
        ),
        "fertiliser.csv": _csv(
            "site,treatment,date,n_kg_ha,p_kg_ha,k_kg_ha,product",
            "Maricopa,2007-03-13,2007-03-13,50,0,0,ammonium phosphate",
            "Maricopa,2008-02-13,2008-02-13,50,0,0,ammonium phosphate",
            "Maricopa,2008-03-13,2008-03-13,50,0,0,ammonium phosphate",
            "Maricopa,2009-01-12,2009-01-12,50,0,0,ammonium phosphate",
        ),
        "experimental_data.csv": _csv("site,treatment,date,variable,value,unit"),
    },
}


def ensure_demo_datasets(cache_path: Path) -> None:
    """Write demo archives that are not already present under the cache."""
    root = Path(cache_path) / DATASETS_CACHE
    for source, files in DEMO_ARCHIVES.items():
        if (root / source).is_dir():
            continue
        cache_wheat_archive(cache_path, source, files)


def available_site_years(cache_path: Path) -> tuple[SiteYear, ...]:
    """Site-years listed on the Inputs tab (demo archives plus any cached extras)."""
    ensure_demo_datasets(cache_path)
    return list_wheat_site_years(cache_path)
