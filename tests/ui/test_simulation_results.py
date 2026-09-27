"""Behaviour: scalar table rows and UI preference persistence."""

from __future__ import annotations

from pathlib import Path

from orion.ui.configuration import default_configuration
from orion.ui.preferences import UiPreferences, load_preferences, merge_plot_order, move_plot_key, save_preferences
from orion.ui.results_data import collect_scalar_data, list_plot_keys, plot_key
from orion.ui.runs import SimulationRun
from orion.ui.simulation_results import overlay_series_for_key


def test_collect_scalar_data_exposes_name_unit_value_description(clock_model):
    final, _ = clock_model.run(3)
    rows = collect_scalar_data(final)
    by_name = {r.name: r for r in rows}
    assert "clock.step" in by_name
    assert by_name["clock.step"].unit == "step"
    assert by_name["clock.step"].value == "3"
    assert "simulation" in by_name["clock.step"].description.lower() or by_name["clock.step"].description
    assert "clock.delta" in by_name
    assert by_name["clock.delta"].unit == "hours"


def test_collect_scalar_data_includes_nested_soil_scalars(clock_model):
    rows = {r.name for r in collect_scalar_data(clock_model.model)}
    assert any(name.startswith("soil.layers.0.") for name in rows)
    assert "soil.layers.0.clay" in rows or any(".clay" in n for n in rows)


def test_preferences_round_trip_in_config_path(tmp_path: Path):
    path = tmp_path / "ui_preferences.json"
    prefs = UiPreferences(plot_order=["clock.step", "weather.Ts"], unit_alternatives={"kg/kg": "g/kg", "kg/m^2": "t/ha"})
    saved = save_preferences(prefs, path)
    assert saved == path
    loaded = load_preferences(path)
    assert loaded.plot_order == ["clock.step", "weather.Ts"]
    assert loaded.unit_alternatives == {"kg/kg": "g/kg", "kg/m^2": "t/ha"}


def test_preferences_drop_unknown_unit_alternatives(tmp_path: Path):
    path = tmp_path / "ui_preferences.json"
    path.write_text('{"plot_order": [], "unit_alternatives": {"kg/kg": "g/kg", "m": "furlong", "bogus": "t/ha"}}', encoding="utf-8")
    loaded = load_preferences(path)
    assert loaded.unit_alternatives == {"kg/kg": "g/kg"}


def test_preferences_omit_default_unit_alternatives_on_save(tmp_path: Path):
    path = tmp_path / "ui_preferences.json"
    save_preferences(UiPreferences(unit_alternatives={"kg/kg": "%", "kg/m^2": "kg/ha", "water": "mm"}), path)
    loaded = load_preferences(path)
    assert loaded.unit_alternatives == {}


def test_merge_plot_order_keeps_saved_then_appends_new():
    assert merge_plot_order(["b", "a"], ["a", "b", "c"]) == ["b", "a", "c"]


def test_dragging_a_plot_tile_moves_it_to_the_drop_index():
    assert move_plot_key(["clock.step", "weather.Ts", "crop.biomass"], 2, 0) == ["crop.biomass", "clock.step", "weather.Ts"]
    assert move_plot_key(["a", "b", "c"], 0, 2) == ["b", "c", "a"]
    assert move_plot_key(["a", "b", "c"], 1, 1) == ["a", "b", "c"]


def test_plot_tile_drag_uses_fallback_so_wrapped_tiles_can_reorder():
    from orion.ui.simulation_results import PLOT_TILE_SORTABLE_OPTIONS

    assert PLOT_TILE_SORTABLE_OPTIONS["forceFallback"] is True
    assert PLOT_TILE_SORTABLE_OPTIONS["direction"] == "horizontal"


def test_plot_tile_drag_rebinds_sortable_to_the_live_tile_row():
    from orion.ui.simulation_results import sortable_rebind_javascript

    script = sortable_rebind_javascript("c260")
    assert "c260" in script
    assert "forceFallback" in script
    assert ".plot-drag-handle" in script
    assert "sortend" in script


def test_dragging_a_plot_tile_ignores_out_of_range_indices():
    assert move_plot_key(["a", "b"], -1, 0) == ["a", "b"]
    assert move_plot_key(["a", "b"], 0, 9) == ["a", "b"]


def test_successive_plot_tile_drags_compose():
    order = move_plot_key(["a", "b", "c"], 2, 0)
    assert order == ["c", "a", "b"]
    assert move_plot_key(order, 0, 2) == ["a", "b", "c"]


def test_list_plot_keys_from_history(clock_model):
    _, history = clock_model.run(2)
    keys = list_plot_keys(history)
    assert plot_key("clock", "step") in keys


def test_plot_overlay_labels_series_with_simulation_run_names(clock_model):
    final, history = clock_model.run(2)
    run = SimulationRun(name="westerfeld:2018:intensive", color="#C45C26", configuration=default_configuration())
    results = {run.name: (final, history)}
    series, unit = overlay_series_for_key(results, (run,), "clock.step")
    assert unit == "step"
    assert series == [(run.name, run.color, [1.0, 2.0])]


def test_overlay_legend_lists_each_dataset_colour_once_in_run_order():
    from orion.ui.simulation_results import overlay_legend_entries

    first = SimulationRun(name="agmip_kassie:1991:IRRIGATED", color="#3D7FBF", configuration=default_configuration())
    second = SimulationRun(name="westerfeld:2018:intensive", color="#C45C26", configuration=default_configuration())
    skipped = SimulationRun(name="unused", color="#6B46C1", configuration=default_configuration())
    results = {first.name: object(), second.name: object()}
    assert overlay_legend_entries((first, skipped, second), results) == [
        (first.name, first.color),
        (second.name, second.color),
    ]


def test_overlay_series_reuse_the_same_colour_for_a_dataset_on_every_plot(clock_model):
    final, history = clock_model.run(2)
    first = SimulationRun(name="site-a", color="#3D7FBF", configuration=default_configuration())
    second = SimulationRun(name="site-b", color="#C45C26", configuration=default_configuration())
    results = {first.name: (final, history), second.name: (final, history)}
    keys = [key for key in list_plot_keys(history) if key.startswith("clock.")][:2]
    assert len(keys) >= 1
    palettes = [{name: color for name, color, _ in overlay_series_for_key(results, (first, second), key)[0]} for key in keys]
    expected = {first.name: first.color, second.name: second.color}
    assert all(palette == expected for palette in palettes)


def test_overlay_figure_hides_legend_so_tiles_share_one_at_the_top():
    from orion.ui.simulation_results import overlay_figure

    fig = overlay_figure(
        [("site-a", "#3D7FBF", [1.0, 2.0]), ("site-b", "#C45C26", [1.0, 3.0])],
        "step",
        3,
    )
    payload = fig.to_plotly_json()
    assert payload["layout"]["showlegend"] is False
    assert [trace["line"]["color"] for trace in payload["data"]] == ["#3D7FBF", "#C45C26"]
    assert all(trace["showlegend"] is False for trace in payload["data"])


def test_plot_reorder_hint_is_a_tooltip_on_the_tile_handle():
    import inspect

    from orion.ui.simulation_results import PLOT_REORDER_HINT, render_plot_grid

    source = inspect.getsource(render_plot_grid)
    assert "reorder" in PLOT_REORDER_HINT.lower()
    assert "ui.label(PLOT_REORDER_HINT" not in source
    assert "plot-drag-handle" in source
    assert "tooltip(PLOT_REORDER_HINT)" in source


def test_overlay_figure_stays_compact_without_a_duplicate_title():
    from orion.ui.simulation_results import overlay_figure

    fig = overlay_figure(
        [("site-a", "#3D7FBF", [1.0, 2.0])],
        "step",
        3,
    )
    layout = fig.to_plotly_json()["layout"]
    title = layout.get("title")
    text = title.get("text") if isinstance(title, dict) else title
    assert not text
    assert layout["height"] <= 180


def test_plot_overlay_converts_values_when_unit_alternative_is_set(clock_model):
    final, history = clock_model.run(2)
    run = SimulationRun(name="default", color="#C45C26", configuration=default_configuration())
    results = {run.name: (final, history)}
    # clock.step has no alternative; preference for kg/kg must not disturb it
    series, unit = overlay_series_for_key(results, (run,), "clock.step", {"kg/kg": "%"})
    assert unit == "step"
    assert series[0][2] == [1.0, 2.0]


def test_plot_keys_list_soil_variables_before_clock(clock_model):
    keys = list_plot_keys(clock_model.model)
    soil_i = next(i for i, key in enumerate(keys) if key.startswith("soil."))
    clock_i = next(i for i, key in enumerate(keys) if key.startswith("clock."))
    assert soil_i < clock_i
    assert plot_key("soil", "layers.0.water") in keys
