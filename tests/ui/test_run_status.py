"""Behaviour: Done status includes client and simulate timings."""

from __future__ import annotations

from orion.ui.run_status import RunTiming, format_done_status


def test_done_status_includes_run_count_and_timing_summary():
    timing = RunTiming(seconds={"openmeteo": 1.2, "soilgrids": 0.3, "simulate": 2.0, "display": 0.4})
    assert format_done_status(2, timing) == "Done · 2 run(s) · Open-Meteo 1.20s · SoilGrids 0.30s · simulate 2.00s · display 0.40s · total 3.90s"


def test_done_status_with_empty_timing_still_shows_total():
    assert format_done_status(1, RunTiming()) == "Done · 1 run(s) · total 0.00s"
