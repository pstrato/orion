"""Behaviour: warm result display patches plots without remounting or re-introspecting."""

from __future__ import annotations

from orion.ui.configuration import default_configuration
from orion.ui.introspect import run_steps
from orion.ui.preferences import UiPreferences, merge_plot_order
from orion.ui.results_data import list_plot_keys
from orion.ui.runs import SimulationRun
from orion.ui.simulation_results import (
    build_visible_plots,
    results_layout_key,
    should_persist_plot_order,
)


def test_results_layout_key_stable_when_only_series_values_change(clock_model):
    final_a, history_a = run_steps(clock_model, 2)
    final_b, history_b = run_steps(clock_model, 4)
    run = SimulationRun(name="site", color="#C45C26", configuration=default_configuration())
    results_a = {run.name: (final_a, history_a)}
    results_b = {run.name: (final_b, history_b)}
    order = list_plot_keys(history_a)
    assert results_layout_key(results_a, (run,), order) == results_layout_key(results_b, (run,), order)


def test_results_layout_key_changes_when_run_set_changes(clock_model):
    final, history = run_steps(clock_model, 2)
    first = SimulationRun(name="a", color="#3D7FBF", configuration=default_configuration())
    second = SimulationRun(name="b", color="#C45C26", configuration=default_configuration())
    order = list_plot_keys(history)
    one = {first.name: (final, history)}
    two = {first.name: (final, history), second.name: (final, history)}
    assert results_layout_key(one, (first,), order) != results_layout_key(two, (first, second), order)


def test_build_visible_plots_precomputes_overlay_series(clock_model):
    final, history = run_steps(clock_model, 2)
    run = SimulationRun(name="site", color="#C45C26", configuration=default_configuration())
    results = {run.name: (final, history)}
    tiles = build_visible_plots(results, (run,), list_plot_keys(history), 3, {})
    assert tiles
    step = next(tile for tile in tiles if tile.key == "clock.step")
    assert step.unit == "step"
    assert step.series == ((run.name, run.color, (1.0, 2.0)),)


def test_should_persist_plot_order_only_when_merged_order_changes():
    available = ["clock.step", "weather.Ts"]
    preferred = ["weather.Ts"]
    merged = merge_plot_order(preferred, available)
    assert should_persist_plot_order(preferred, merged) is True
    assert should_persist_plot_order(merged, merged) is False
    prefs = UiPreferences(plot_order=merged)
    assert should_persist_plot_order(prefs.plot_order, merge_plot_order(prefs.plot_order, available)) is False


def test_precomputed_tiles_drive_layout_without_extra_keys(clock_model):
    final, history = run_steps(clock_model, 2)
    run = SimulationRun(name="site", color="#C45C26", configuration=default_configuration())
    results = {run.name: (final, history)}
    tiles = build_visible_plots(results, (run,), list_plot_keys(history), 3, {})
    layout = results_layout_key(results, (run,), [tile.key for tile in tiles])
    assert layout == results_layout_key(results, (run,), list_plot_keys(history))
