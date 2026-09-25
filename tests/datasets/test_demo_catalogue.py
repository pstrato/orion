"""Behaviour: demo archives populate the Inputs-tab dataset catalogue."""

from __future__ import annotations

from pathlib import Path

from orion.datasets.demo import DEMO_ARCHIVES, available_site_years, ensure_demo_datasets


def test_available_site_years_lists_demo_archives_from_cache(tmp_path: Path):
    years = available_site_years(tmp_path)
    names = {site.name for site in years}
    assert any(name.startswith("westerfeld:") for name in names)
    assert any(name.startswith("agmip_kassie:") for name in names)
    assert any(name.startswith("wageningen:") for name in names)
    assert any(name.startswith("cunderdin:") for name in names)
    assert any(name.startswith("obregon:") for name in names)
    assert any(name.startswith("luancheng:") for name in names)
    assert any(name.startswith("ludhiana:") for name in names)
    assert any(name.startswith("balcarce:") for name in names)
    assert any(name.startswith("egypt_nile:") for name in names)
    assert any(name.startswith("iwyp_valdivia:") for name in names)
    assert any(name.startswith("german_met:") for name in names)
    assert (tmp_path / "datasets" / "westerfeld" / "SOWING.csv").is_file()
    assert set(DEMO_ARCHIVES) <= {path.name for path in (tmp_path / "datasets").iterdir()}


def test_ensure_demo_datasets_does_not_overwrite_existing_archive(tmp_path: Path):
    ensure_demo_datasets(tmp_path)
    sowing = tmp_path / "datasets" / "westerfeld" / "SOWING.csv"
    original = sowing.read_text(encoding="utf-8")
    sowing.write_text("kept\n", encoding="utf-8")
    ensure_demo_datasets(tmp_path)
    assert sowing.read_text(encoding="utf-8") == "kept\n"
    sowing.write_text(original, encoding="utf-8")
