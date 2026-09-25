"""NiceGUI entrypoint for Orion.

Tabs: Settings, Inputs, Configurations, Processes, Simulation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter
from typing import Any

import jax
from nicegui import run as nicegui_run
from nicegui import ui
from shapely import Point

from orion.core.constant import const
from orion.core.input import Input, Inputs, LocationInput
from orion.core.model import Model, model
from orion.core.setting import Settings
from orion.processes.clock import ClockInput
from orion.ui.catalog import configured_process_inputs, make_clock_input, unimplemented_process_labels
from orion.ui.configuration import Configuration, default_configuration
from orion.ui.configuration_tab import render_configuration_tab
from orion.ui.fetch_status import FetchProgress, discard_fetch_progress, fetch_progress, latest_fetch_progress
from orion.ui.preferences import UiPreferences, load_preferences, merge_plot_order, save_preferences
from orion.ui.entity_editor import render_entity_values
from orion.ui.process_tab import render_process_tab
from orion.ui.reflect import apply_quantity_edits, set_entity_value
from orion.ui.results_data import list_plot_keys
from orion.ui.run_status import RunTiming, collect_timing, format_done_status, timed
from orion.ui.runs import (
    DATASET_SUMMARY_COLUMNS,
    INPUT_CUSTOM,
    INPUT_DATASETS,
    SimulationRun,
    SiteYear,
    active_site_years,
    dataset_hierarchy,
    dataset_map_markers,
    dataset_map_view,
    dataset_summary_rows,
    planned_runs,
    selected_site_years,
)
from orion.ui.settings_units import unit_alternative_options
from orion.ui.simulation_results import ResultsPanels, VisiblePlot, build_visible_plots
from orion.ui.theme import FAVICON_PATH, ICON_PATH, apply_theme, setting_row
from orion.ui.units import display_unit_for, resolve_unit_alternatives
from orion.ui.warm_model_key import ConfigurationCompileCache, WarmModelCache, warm_model_key

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache"
DEFAULT_YEAR = 2024


def ui_settings(**overrides: object) -> Settings:
    """A ``Settings`` entity filled from its fields, so new fields appear without a form rewrite."""
    from dataclasses import MISSING
    from dataclasses import fields as dataclass_fields

    values: dict[str, object] = {"name": "orion ui"}
    for item in dataclass_fields(Settings):
        if item.name == "name" or item.name in overrides:
            continue
        if item.default is not MISSING or item.default_factory is not MISSING:  # type: ignore[misc]
            continue
        values[item.name] = _blank_setting(item.type)
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _blank_setting(annotation: object) -> object:
    if annotation in {Path, "Path"}:
        return CACHE_PATH
    if annotation in {bool, "bool"}:
        return False
    if annotation in {str, "str"}:
        return ""
    if annotation in {int, "int"}:
        return 0
    if annotation in {float, "float"}:
        return 0.0
    if annotation in {date, "date"}:
        return date(DEFAULT_YEAR, 1, 1)
    raise TypeError(f"No UI default for settings field of type {annotation!r}.")


@dataclass
class AppState:
    """Mutable UI session state."""

    settings: Settings = field(default_factory=ui_settings)
    latitude: float = 51.5
    longitude: float = 0.1
    location_name: str = "field"
    start: date = field(default_factory=lambda: date(DEFAULT_YEAR, 1, 1))
    end: date = field(default_factory=lambda: date(DEFAULT_YEAR + 1, 1, 1))
    step_hours: int = 3
    input_mode: str = INPUT_DATASETS
    selected_dataset_names: list[str] = field(default_factory=list)
    configurations: list[Configuration] = field(default_factory=lambda: [default_configuration()])
    selected_configuration_index: int = 0
    selected_input_key: str = "clock"
    runs: list[SimulationRun] = field(default_factory=list)
    results: dict[str, tuple[Model, Model]] = field(default_factory=dict)
    prefs: UiPreferences = field(default_factory=load_preferences)
    compile_cache: ConfigurationCompileCache = field(default_factory=ConfigurationCompileCache)
    warm_models: WarmModelCache = field(default_factory=WarmModelCache)

    @property
    def cache_path(self) -> Path:
        return Path(self.settings.cache_path)

    @cache_path.setter
    def cache_path(self, value: Path) -> None:
        self.settings = replace(self.settings, cache_path=Path(value))


def build_inputs(state: AppState, configuration: Configuration) -> Inputs:
    """Assemble top-level inputs from session state and the chosen configuration.

    Clock and location are entities on ``Inputs.inputs``. Settings are an
    argument of ``model``, not a field of ``Inputs``.
    """
    clock = make_clock_input(configuration.clock)
    if isinstance(clock, ClockInput):
        clock = replace(
            clock,
            start=const("start", "isodate", state.start, "Simulation start date"),
            end=const("end", "isodate", state.end, "Simulation end date"),
            delta=const("delta", "hours", int(state.step_hours), "Simulation delta step in hours"),
        )
    clock = apply_quantity_edits(clock, configuration.parameter_edits, "clock")
    location = LocationInput(
        name=state.location_name,
        geometry=const("geometry", "coordinate", Point(float(state.longitude), float(state.latitude)), "Location geometry"),
    )
    pieces: list[Input] = [clock, location, *configured_process_inputs(configuration)]
    return Inputs(name="field input", inputs=tuple(pieces))


def catalogue_site_years(state: AppState) -> tuple[Any, ...]:
    """Wheat site-years listed on the Inputs tab."""
    try:
        from orion.datasets.demo import available_site_years
    except Exception:  # noqa: BLE001 — dataset package is mid-refactor
        return ()
    return tuple(available_site_years(state.cache_path))


def selected_datasets(state: AppState) -> tuple[SiteYear, ...]:
    """Site-years Simulate uses when Inputs is in datasets mode."""
    return active_site_years(state.input_mode, catalogue_site_years(state), state.selected_dataset_names)


def planned_simulation_runs(state: AppState) -> tuple[SimulationRun, ...]:
    """Runs Simulate will execute: datasets × enabled configurations."""
    return planned_runs(tuple(state.configurations), selected_datasets(state))


def simulation_start_error(state: AppState) -> str | None:
    """Why Simulate should not start, or None when runs can proceed."""
    if state.input_mode == INPUT_DATASETS and not selected_datasets(state):
        return "Select at least one dataset"
    runs = planned_simulation_runs(state)
    if not runs:
        return "Enable at least one configuration"
    for run in runs:
        abstract = unimplemented_process_labels(run.configuration.processes)
        if abstract:
            return f"{run.configuration.name} has abstract processes: {', '.join(abstract)}"
    return None


def build_model(state: AppState, configuration: Configuration, site_year: Any | None = None) -> Model:
    template = build_inputs(state, configuration)
    inputs = template
    if site_year is not None:
        from orion.datasets.inputs import inputs_for_site_year

        inputs = inputs_for_site_year(template, site_year)
    return model(configuration.name, inputs, state.settings)


def _step_count(built: Model) -> int:
    """Steps implied by the clock input: days × 24 / delta hours."""
    for _, entity in built.inputs.all_entities():
        if isinstance(entity, ClockInput):
            days = (entity.end.value - entity.start.value).days
            delta = int(entity.delta.value)
            if delta <= 0:
                raise ValueError("Step hours must be positive.")
            count = int(days) * 24 // delta
            if count < 1:
                raise ValueError("Horizon is shorter than one step.")
            return count
    raise ValueError("Model has no clock input.")


def _block_ready(value: Model) -> Model:
    try:
        return jax.block_until_ready(value)
    except Exception:  # noqa: BLE001 — a plain entity tree is already concrete
        return value


def _open_quick_edit(state: AppState, refresh_all: Callable[[], None]) -> None:
    with ui.dialog() as dialog, ui.card().classes("w-[32rem] max-w-full p-3 gap-2"):
        ui.label("Quick edit").classes("text-subtitle1 font-medium")
        cache = ui.input("Cache path", value=str(state.cache_path)).classes("w-full")
        lat = ui.number("Latitude", value=state.latitude, format="%.4f", step=0.01)
        lon = ui.number("Longitude", value=state.longitude, format="%.4f", step=0.01)
        start = ui.input("Start date", value=state.start.isoformat()).props("type=date")
        end = ui.input("End date", value=state.end.isoformat()).props("type=date")
        step = ui.number("Step (hours)", value=state.step_hours, min=1, max=24, step=1)

        def apply() -> None:
            state.cache_path = Path(str(cache.value))
            state.latitude = float(lat.value or 0.0)
            state.longitude = float(lon.value or 0.0)
            state.start = date.fromisoformat(str(start.value))
            state.end = date.fromisoformat(str(end.value))
            state.step_hours = int(step.value or 1)
            refresh_all()
            dialog.close()
            ui.notify("Settings and inputs updated")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            ui.button("Apply", on_click=apply).props("dense")
    dialog.open()


def _render_settings_tab(state: AppState, host, on_prefs_changed: Callable[[UiPreferences], None] | None = None) -> Callable[[], None]:
    def render() -> None:
        host.clear()
        with host:

            def on_setting(path: str, value: object) -> None:
                state.settings = set_entity_value(state.settings, path, value)

            render_entity_values(state.settings, on_setting)

            ui.label("Display units").classes("text-subtitle2 font-medium").tooltip("Internal quantities stay SI; choose agronomic alternatives for tables and plots.")
            selectors: dict[str, Any] = {}
            for row in unit_alternative_options():
                canonical = str(row["canonical"])
                choices = row["choices"]
                assert isinstance(choices, dict)
                current = display_unit_for(canonical, state.prefs.unit_alternatives)
                with setting_row(str(row["label"])):
                    selectors[canonical] = ui.select(choices, value=current).props("dense options-dense")

            def save_settings() -> None:
                chosen = {canonical: str(selector.value) for canonical, selector in selectors.items()}
                state.prefs.unit_alternatives = resolve_unit_alternatives(chosen)
                save_preferences(state.prefs)
                if on_prefs_changed is not None:
                    on_prefs_changed(state.prefs)
                ui.notify(f"Settings saved (cache: {state.cache_path})")

            ui.button("Save settings", on_click=save_settings).props("dense")

    return render


def _render_inputs_tab(state: AppState, host) -> Callable[[], None]:
    def render() -> None:
        host.clear()
        with host:
            mode = ui.toggle({INPUT_DATASETS: "Datasets", INPUT_CUSTOM: "User details"}, value=state.input_mode).props("dense")
            mode.tooltip("Start from wheat datasets, or enter a location and horizon yourself.")

            def on_mode(_) -> None:
                next_mode = str(mode.value or INPUT_DATASETS)
                if next_mode == state.input_mode:
                    return
                state.input_mode = next_mode
                render()

            mode.on_value_change(on_mode)

            if state.input_mode == INPUT_DATASETS:
                _render_dataset_inputs(state)
            else:
                _render_custom_inputs(state)

    return render


def _render_dataset_inputs(state: AppState) -> None:
    catalogue = catalogue_site_years(state)
    valid = {site.name for site in catalogue}
    state.selected_dataset_names = [name for name in state.selected_dataset_names if name in valid]
    selected = set(state.selected_dataset_names)
    boxes: dict[str, Any] = {}
    parents: list[tuple[Any, list[str]]] = []
    syncing = {"on": False}

    def show_selected() -> None:
        detail_host.clear()
        sites = selected_site_years(catalogue, state.selected_dataset_names)
        with detail_host:
            if not sites:
                return
            center_lat, center_lon, zoom = dataset_map_view(sites)
            leaflet = ui.leaflet(center=(center_lat, center_lon), zoom=zoom).classes("w-full h-48")
            for marker_lat, marker_lon in dataset_map_markers(sites):
                leaflet.marker(latlng=(marker_lat, marker_lon))
            ui.table(columns=DATASET_SUMMARY_COLUMNS, rows=dataset_summary_rows(sites)).classes("w-full").props("dense")

    def sync_leaves() -> None:
        if syncing["on"]:
            return
        chosen = {name for name, box in boxes.items() if bool(box.value)}
        state.selected_dataset_names = [site.name for site in catalogue if site.name in chosen]
        syncing["on"] = True
        try:
            for parent, names in parents:
                parent.value = bool(names) and all(name in chosen for name in names)
        finally:
            syncing["on"] = False
        show_selected()

    ui.label("Wheat datasets").classes("text-subtitle2").tooltip("Select datasets in the tree to simulate together.")
    with ui.row().classes("w-full items-start gap-3 flex-wrap"):
        tree_host = ui.column().classes("dataset-tree min-w-52 w-72 gap-0")
        detail_host = ui.column().classes("flex-1 min-w-80 gap-1")
    with tree_host:
        _render_hierarchy_nodes(dataset_hierarchy(catalogue), boxes, parents, selected, 0)
    _bind_hierarchy_handlers(boxes, parents, sync_leaves, syncing)
    show_selected()
    with setting_row("Step (hours)"):
        step_input = ui.number(value=state.step_hours, min=1, max=24, step=1).props("dense")

    def save_inputs() -> None:
        sync_leaves()
        state.step_hours = int(step_input.value or 1)
        ui.notify("Inputs saved")

    ui.button("Save inputs", on_click=save_inputs).props("dense")


def _render_hierarchy_nodes(
    nodes: list[dict[str, Any]],
    boxes: dict[str, Any],
    parents: list[tuple[Any, list[str]]],
    selected: set[str],
    depth: int,
) -> None:
    indent = depth * 12
    for node in nodes:
        children = node["children"]
        names = list(node["site_names"])
        if not children:
            box = ui.checkbox(str(node["label"]), value=str(node["id"]) in selected).props("dense").style(f"margin-left:{indent}px")
            boxes[str(node["id"])] = box
            continue
        parent = ui.checkbox(str(node["label"]), value=bool(names) and all(name in selected for name in names)).props("dense").style(f"margin-left:{indent}px")
        parent.classes("font-medium")
        parents.append((parent, names))
        _render_hierarchy_nodes(children, boxes, parents, selected, depth + 1)


def _bind_hierarchy_handlers(
    boxes: dict[str, Any],
    parents: list[tuple[Any, list[str]]],
    on_leaf_change: Callable[[], None],
    syncing: dict[str, bool],
) -> None:
    for box in boxes.values():
        box.on_value_change(lambda _: on_leaf_change())
    for parent, names in parents:

        def make_parent(ns: list[str] = names, group=parent):
            def on_parent(_) -> None:
                if syncing["on"]:
                    return
                syncing["on"] = True
                try:
                    for name in ns:
                        if name in boxes:
                            boxes[name].value = bool(group.value)
                finally:
                    syncing["on"] = False
                on_leaf_change()

            return on_parent

        parent.on_value_change(make_parent())


def _render_custom_inputs(state: AppState) -> None:
    ui.label("Location and horizon").classes("text-subtitle2").tooltip("Simulate enabled configurations at this location.")

    with setting_row("Location name"):
        name_input = ui.input(value=state.location_name).props("dense")
    with setting_row("Start date"):
        start_input = ui.input(value=state.start.isoformat()).props("dense type=date")
    with setting_row("End date"):
        end_input = ui.input(value=state.end.isoformat()).props("dense type=date")
    with setting_row("Latitude"):
        lat_input = ui.number(value=state.latitude, format="%.4f", step=0.01).props("dense")
    with setting_row("Longitude"):
        lon_input = ui.number(value=state.longitude, format="%.4f", step=0.01).props("dense")

    leaflet = ui.leaflet(center=(state.latitude, state.longitude), zoom=6).classes("w-full h-40 md:h-48")
    marker = leaflet.marker(latlng=(state.latitude, state.longitude))

    def sync_marker(lat_v: float, lon_v: float) -> None:
        state.latitude = lat_v
        state.longitude = lon_v
        lat_input.value = lat_v
        lon_input.value = lon_v
        marker.move(lat_v, lon_v)

    def on_map_click(e) -> None:
        args = getattr(e, "args", e)
        if not isinstance(args, dict):
            return
        latlng = args.get("latlng") or args
        if not isinstance(latlng, dict):
            return
        sync_marker(float(latlng.get("lat", state.latitude)), float(latlng.get("lng", state.longitude)))

    leaflet.on("map-click", on_map_click)
    lat_input.on_value_change(lambda _: sync_marker(float(lat_input.value or 0.0), float(lon_input.value or 0.0)))
    lon_input.on_value_change(lambda _: sync_marker(float(lat_input.value or 0.0), float(lon_input.value or 0.0)))

    with setting_row("Step (hours)"):
        step_input = ui.number(value=state.step_hours, min=1, max=24, step=1).props("dense")

    def save_inputs() -> None:
        state.location_name = str(name_input.value or "field")
        state.start = date.fromisoformat(str(start_input.value))
        state.end = date.fromisoformat(str(end_input.value))
        state.step_hours = int(step_input.value or 1)
        state.latitude = float(lat_input.value or 0.0)
        state.longitude = float(lon_input.value or 0.0)
        ui.notify("Inputs saved")

    ui.button("Save inputs", on_click=save_inputs).props("dense")


def _render_configs_tab(state: AppState, host) -> Callable[[], None]:
    def render() -> None:
        def on_change(configurations: list[Configuration], index: int, selected_input_key: str) -> None:
            state.configurations = configurations
            state.selected_configuration_index = index
            state.selected_input_key = selected_input_key
            render()

        render_configuration_tab(host, state.configurations, state.selected_configuration_index, on_change, state.selected_input_key)

    return render


def _simulate_planned_runs(
    state: AppState,
    runs: tuple[SimulationRun, ...],
    progress_queue: SimpleQueue[FetchProgress],
) -> tuple[dict[str, tuple[Model, Model]], RunTiming, tuple[VisiblePlot, ...]]:
    out: dict[str, tuple[Model, Model]] = {}
    with collect_timing() as timing:
        before = state.compile_cache.fingerprint
        state.compile_cache.sync(tuple(state.configurations))
        if before is not None and before != state.compile_cache.fingerprint:
            state.warm_models.clear()
        with fetch_progress(progress_queue.put):
            for run in runs:
                key = warm_model_key(state, run)
                built = state.warm_models.resolve(key, lambda r=run: build_model(state, r.configuration, r.site_year))
                with timed("simulate"):
                    steps = _step_count(built)
                    final, history = state.compile_cache.simulate(built, steps)
                    final = _block_ready(final)
                    history = _block_ready(history)
                out[run.name] = (final, history)
    first_history = next(iter(out.values()))[1]
    order = merge_plot_order(state.prefs.plot_order, list_plot_keys(first_history))
    tiles = build_visible_plots(out, runs, order, state.step_hours, state.prefs.unit_alternatives)
    return out, timing, tiles


def _render_simulation_tab(state: AppState, sim_host, results_host, status_label, fetch_bar) -> tuple[Callable[[], None], Callable[[], None]]:  # noqa: C901
    panels = ResultsPanels(results_host)

    def persist_prefs(prefs: UiPreferences) -> None:
        state.prefs = prefs
        save_preferences(prefs)

    def refresh_results(tiles: Sequence[VisiblePlot] | None = None) -> None:
        panels.show(
            state.results,
            state.runs,
            state.step_hours,
            state.prefs,
            persist_prefs,
            tiles=tiles,
        )

    def render() -> None:
        sim_host.clear()
        with sim_host:
            with ui.row().classes("w-full gap-2 flex-wrap items-center"):
                for index, config in enumerate(state.configurations):
                    with ui.row().classes("items-center gap-2").style(f"border: 1px solid {config.color}; border-radius: 6px; padding: 4px 8px;"):
                        sw = ui.switch(config.name, value=config.enabled)

                        def make_sw(i: int, switch=sw):
                            def on_toggle(_) -> None:
                                c = state.configurations[i]
                                state.configurations[i] = c.with_enabled(bool(switch.value))

                            return on_toggle

                        sw.on_value_change(make_sw(index))
                        ui.element("div").style(f"width:14px;height:14px;border-radius:50%;background:{config.color}")

            run_button = ui.button("Simulate", icon="play_arrow").props("dense").tooltip("Uses Inputs plus each enabled configuration.")

            async def run_simulation() -> None:
                blocked = simulation_start_error(state)
                if blocked:
                    ui.notify(blocked, type="warning")
                    return
                runs = planned_simulation_runs(state)
                run_button.disable()
                status_label.set_text("Running…")
                fetch_bar.value = 0
                fetch_bar.set_visibility(False)
                state.runs = list(runs)
                progress_queue: SimpleQueue[FetchProgress] = SimpleQueue()
                progress_live = True
                progress_stopped = False

                def drain_fetch_progress() -> None:
                    if not progress_live:
                        return
                    latest = latest_fetch_progress(progress_queue)
                    if latest is None:
                        return
                    status_label.set_text(latest.message)
                    fetch_bar.value = latest.fraction
                    fetch_bar.set_visibility(True)

                def stop_progress_ui() -> None:
                    nonlocal progress_live, progress_stopped
                    progress_live = False
                    if progress_stopped:
                        return
                    progress_stopped = True
                    progress_timer.deactivate()
                    progress_timer.delete()
                    discard_fetch_progress(progress_queue)
                    fetch_bar.value = 0
                    fetch_bar.set_visibility(False)

                progress_timer = ui.timer(0.15, drain_fetch_progress)
                try:
                    payload = await nicegui_run.io_bound(lambda: _simulate_planned_runs(state, runs, progress_queue))
                    if payload is None:
                        raise RuntimeError("simulation returned no result")
                    result, timing, tiles = payload
                    state.results = result
                    stop_progress_ui()
                    status_label.set_text(format_done_status(len(result), timing))
                    display_started = perf_counter()
                    refresh_results(tiles)
                    timing.add("display", perf_counter() - display_started)
                    status_label.set_text(format_done_status(len(result), timing))
                except Exception as exc:  # noqa: BLE001 — show ops errors in UI
                    status_label.set_text(f"Error: {exc}")
                    ui.notify(str(exc), type="negative")
                finally:
                    stop_progress_ui()
                    run_button.enable()

            run_button.on_click(run_simulation)

    return render, refresh_results


@ui.page("/")
def create_page() -> None:
    """Build the multi-tab ops UI."""
    apply_theme()
    state = AppState()
    refreshers: dict[str, Callable[[], None]] = {}

    def refresh_all() -> None:
        for fn in refreshers.values():
            fn()

    def on_unit_prefs_changed(prefs: UiPreferences) -> None:
        state.prefs = prefs
        if state.results:
            refresh_results_holder[0]()

    refresh_results_holder: list[Callable[[], None]] = []

    with ui.header().classes("items-center px-2 gap-1"):
        ui.image(ICON_PATH).classes("orion-mark")
        ui.label("Orion").classes("orion-title text-h6 font-bold tracking-tight")
        ui.label("crop growth · JAX").classes("orion-tagline text-xs opacity-80")
        with ui.tabs().classes("text-white") as tabs:
            settings_tab = ui.tab("Settings")
            inputs_tab = ui.tab("Inputs")
            configs_tab = ui.tab("Configurations")
            process_tab = ui.tab("Processes")
            sim_tab = ui.tab("Simulation")
        ui.space()
        ui.button("Quick edit", icon="edit", on_click=lambda: _open_quick_edit(state, refresh_all)).props("flat dense color=white")

    with ui.column().classes("w-full max-w-[1600px] mx-auto px-1 sm:px-2"):
        with ui.tab_panels(tabs, value=settings_tab).classes("w-full"):
            with ui.tab_panel(settings_tab):
                settings_host = ui.column().classes("w-full gap-1")
                refreshers["settings"] = _render_settings_tab(state, settings_host, on_unit_prefs_changed)
                refreshers["settings"]()

            with ui.tab_panel(inputs_tab):
                inputs_host = ui.column().classes("w-full gap-1")
                refreshers["inputs"] = _render_inputs_tab(state, inputs_host)
                refreshers["inputs"]()

            with ui.tab_panel(configs_tab):
                configs_host = ui.column().classes("w-full gap-1")
                refreshers["configs"] = _render_configs_tab(state, configs_host)
                refreshers["configs"]()

            with ui.tab_panel(process_tab):
                process_host = ui.column().classes("w-full gap-1")
                render_process_tab(process_host)

            with ui.tab_panel(sim_tab):
                sim_host = ui.column().classes("w-full gap-1")
                results_host = ui.column().classes("w-full")
                status_label = ui.label("").classes("text-sm text-gray-500")
                fetch_bar = ui.linear_progress(value=0, show_value=False).props("animation-speed=0 instant-feedback").classes("w-full max-w-xl orion-fetch-bar")
                fetch_bar.set_visibility(False)
                refreshers["sim"], refresh_results = _render_simulation_tab(state, sim_host, results_host, status_label, fetch_bar)
                refresh_results_holder.append(refresh_results)
                refreshers["sim"]()


def run(host: str = "127.0.0.1", port: int = 8080, reload: bool = False) -> None:
    """Start the Orion UI server."""
    ui.run(
        title="Orion",
        favicon=FAVICON_PATH,
        host=host,
        port=port,
        reload=reload,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
