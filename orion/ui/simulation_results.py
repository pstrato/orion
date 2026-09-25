"""Simulation results: Tables and Plots sub-tabs with in-place plot updates."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
from nicegui import ui
from nicegui.elements.row import Row

from orion.core.model import Model
from orion.ui.introspect import history_series
from orion.ui.preferences import UiPreferences, merge_plot_order, move_plot_key
from orion.ui.results_data import ScalarDatum, apply_unit_alternatives, collect_scalar_data, list_plot_keys
from orion.ui.runs import SimulationRun
from orion.ui.units import convert_for_display, display_class_for, display_unit_for

PLOT_REORDER_HINT = "Drag the title bar to reorder. Order is saved automatically."
PLOT_TILE_SORTABLE_OPTIONS: dict[str, Any] = {
    "forceFallback": True,
    "fallbackOnBody": True,
    "swapThreshold": 0.65,
    "direction": "horizontal",
}


@dataclass(frozen=True)
class VisiblePlot:
    """One plot tile's data, ready for Plotly mount or ``update_figure``."""

    key: str
    series: tuple[tuple[str, str, tuple[float, ...]], ...]
    unit: str


def results_layout_key(
    results: Mapping[str, object],
    runs: Sequence[SimulationRun],
    plot_keys: Sequence[str],
) -> tuple[object, ...]:
    """Identity of the results chrome (runs + which plot tiles), not series values."""
    run_ids = tuple((run.name, run.color) for run in runs if run.name in results)
    return (run_ids, tuple(sorted(plot_keys)))


def build_visible_plots(
    results: dict[str, tuple[Model, Model]],
    runs: Sequence[SimulationRun],
    plot_order: Sequence[str],
    step_hours: int,
    unit_alternatives: Mapping[str, str] | None = None,
) -> tuple[VisiblePlot, ...]:
    """Precompute overlay series for each plot key that has data.

    ``step_hours`` is accepted for call-site symmetry with figure building; series
    themselves are step-index based until turned into a figure.
    """
    del step_hours  # series are in steps; figures apply step_hours when drawn
    tiles: list[VisiblePlot] = []
    alternatives = unit_alternatives or {}
    for key in plot_order:
        series, unit = overlay_series_for_key(results, runs, key, alternatives)
        if series:
            tiles.append(
                VisiblePlot(
                    key=key,
                    series=tuple((name, color, tuple(ys)) for name, color, ys in series),
                    unit=unit,
                )
            )
    return tuple(tiles)


def sortable_rebind_javascript(element_id: str) -> str:
    """Re-attach SortableJS after NiceGUI re-renders the tile row.

    ``make_sortable`` registers ``on_end`` after the element is first sent, which
    makes NiceGUI replace the DOM node. SortableJS stays bound to the old node, so
    drags stop working until we create a new instance on the live element.
    """
    options = {
        "animation": 150,
        "ghostClass": "opacity-50",
        "handle": ".plot-drag-handle",
        **PLOT_TILE_SORTABLE_OPTIONS,
    }
    return f"""
    (async () => {{
      const el = document.getElementById({element_id!r});
      if (!el) return;
      const {{ Sortable }} = await import("nicegui-sortable");
      const existingKey = Object.keys(el).find((k) => k.startsWith("Sortable"));
      if (existingKey && el[existingKey] && el.isConnected && el.children.length) {{
        return;
      }}
      if (existingKey && el[existingKey] && typeof el[existingKey].destroy === "function") {{
        el[existingKey].destroy();
      }}
      Sortable.create(el, {{
        ...{json.dumps(options)},
        onEnd: (evt) => {{
          const fromId = parseInt(evt.from.id.substring(1));
          const toId = parseInt(evt.to.id.substring(1));
          if (isNaN(fromId) || isNaN(toId)) return;
          const fromSlot = window.mounted_app?.elements?.[fromId]?.slots?.default;
          const toSlot = window.mounted_app?.elements?.[toId]?.slots?.default;
          if (fromSlot && fromSlot.ids) {{
            const itemId = fromSlot.ids.splice(evt.oldIndex, 1)[0];
            if (fromId === toId) fromSlot.ids.splice(evt.newIndex, 0, itemId);
            else if (toSlot && toSlot.ids) toSlot.ids.splice(evt.newIndex, 0, itemId);
          }}
          el.dispatchEvent(new CustomEvent("sortend", {{
            detail: {{
              item_id: parseInt(evt.item.id.substring(1)),
              from_id: fromId,
              to_id: toId,
              old_index: evt.oldIndex,
              new_index: evt.newIndex,
            }},
            bubbles: false,
          }}));
        }},
      }});
    }})()
    """


def overlay_legend_entries(runs: Sequence[SimulationRun], results: Mapping[str, object]) -> list[tuple[str, str]]:
    """One (name, colour) per run that produced results, in run order."""
    return [(run.name, run.color) for run in runs if run.name in results]


def overlay_figure(
    series: Sequence[tuple[str, str, Sequence[float]]],
    unit: str,
    step_hours: int,
) -> go.Figure:
    """One tile: traces use run colours; legend and titles live outside the figure."""
    fig = go.Figure()
    for name, color, ys in series:
        xs = [i * step_hours / 24 for i in range(len(ys))]
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines", name=name, showlegend=False, line=dict(width=1.5, color=color)))
    fig.update_layout(
        margin=dict(l=28, r=6, t=8, b=24),
        height=180,
        xaxis_title="days",
        yaxis_title=unit,
        template="plotly_white",
        showlegend=False,
        font=dict(size=10),
        uirevision=True,
        transition={"duration": 0},
    )
    return fig


def should_persist_plot_order(previous: Sequence[str], merged: Sequence[str]) -> bool:
    """True when merged plot order differs from what is already saved."""
    return list(previous) != list(merged)


def _datum_cell(datum: ScalarDatum) -> None:
    tip = datum.description.replace('"', "'") if datum.description else ""
    unit = f" [{datum.unit}]" if datum.unit else ""
    with ui.row().classes("items-baseline gap-2 no-wrap").style("min-width: 0"):
        label = ui.label(f"{datum.name}{unit}").classes("text-xs text-gray-600 truncate").style("max-width: 55%")
        if tip:
            label.tooltip(tip)
        ui.label(datum.value).classes("font-mono text-sm font-medium")


def render_scalar_tables(
    host,
    results: dict[str, tuple[Model, Model]],
    runs: Sequence[SimulationRun],
    unit_alternatives: Mapping[str, str] | None = None,
) -> None:
    """Multi-column const/scalar tables per simulation run."""
    host.clear()
    alternatives = unit_alternatives or {}
    if not results:
        with host:
            ui.label("No results yet.").classes("text-gray-500")
        return
    with host:
        for run in runs:
            if run.name not in results:
                continue
            final, _ = results[run.name]
            rows = apply_unit_alternatives(collect_scalar_data(final), alternatives)
            with ui.card().classes("w-full p-2").style(f"border-left: 4px solid {run.color}"):
                ui.label(run.name).classes("text-subtitle2 mb-1")
                if not rows:
                    ui.label("No scalar or constant values.").classes("text-gray-500 text-sm")
                    continue
                with ui.element("div").classes("w-full").style("display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 4px 12px;"):
                    for datum in rows:
                        with ui.element("div").classes("py-1"):
                            _datum_cell(datum)


def overlay_series_for_key(
    results: dict[str, tuple[Model, Model]],
    runs: Sequence[SimulationRun],
    key: str,
    unit_alternatives: Mapping[str, str] | None = None,
) -> tuple[list[tuple[str, str, list[float]]], str]:
    """One (name, colour, values) series per run that recorded ``key``."""
    state_name, _, var_name = key.partition(".")
    series: list[tuple[str, str, list[float]]] = []
    unit = ""
    alternatives = unit_alternatives or {}
    for run in runs:
        if run.name not in results:
            continue
        _, history = results[run.name]
        try:
            ys, series_unit = history_series(history, state_name, var_name)
        except KeyError:
            continue
        display = display_unit_for(display_class_for(key, series_unit), alternatives)
        converted: list[float] = []
        out_unit = series_unit
        for y in ys:
            value, out_unit = convert_for_display(float(y), series_unit, display)
            converted.append(value)
        series.append((run.name, run.color, converted))
        unit = out_unit
    return series, unit


def render_plot_grid(
    host,
    results: dict[str, tuple[Model, Model]],
    runs: Sequence[SimulationRun],
    step_hours: int,
    prefs: UiPreferences,
    on_prefs_changed: Callable[[UiPreferences], None],
) -> Callable[[], None] | None:
    """Reorderable grid of plots; order is persisted via preferences."""
    host.clear()
    if not results:
        with host:
            ui.label("No results yet.").classes("text-gray-500")
        return None
    first_history = next(iter(results.values()))[1]
    available = list_plot_keys(first_history)
    if not available:
        with host:
            ui.label("No plottable variables.").classes("text-gray-500")
        return None
    order = merge_plot_order(prefs.plot_order, available)
    prefs.plot_order = order
    on_prefs_changed(prefs)

    tiles = build_visible_plots(results, runs, order, step_hours, prefs.unit_alternatives)
    if not tiles:
        with host:
            ui.label("No plottable variables.").classes("text-gray-500")
        return None

    visible = list(tiles)

    def on_sort_end(e) -> None:
        keys = [tile.key for tile in visible]
        old_index = int(getattr(e, "old_index", -1))
        new_index = int(getattr(e, "new_index", -1))
        reordered = move_plot_key(keys, old_index, new_index)
        by_key = {tile.key: tile for tile in visible}
        visible[:] = [by_key[key] for key in reordered]
        hidden = [key for key in order if key not in reordered]
        prefs.plot_order = reordered + hidden
        on_prefs_changed(prefs)

    grid_holder: dict[str, Row] = {}

    def enable_sortable() -> None:
        grid = grid_holder.get("grid")
        if grid is None:
            return
        if getattr(grid, "_sortable", None) is None:
            grid.make_sortable(handle=".plot-drag-handle", on_end=on_sort_end, options=PLOT_TILE_SORTABLE_OPTIONS)

        def rebind() -> None:
            ui.run_javascript(sortable_rebind_javascript(grid.html_id))

        ui.timer(0.2, rebind, once=True)

    with host:
        legend = overlay_legend_entries(runs, results)
        if len(legend) > 1:
            with ui.row().classes("w-full flex-wrap items-center gap-x-3 gap-y-0"):
                for name, color in legend:
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.element("div").style(f"width:10px;height:10px;border-radius:2px;background:{color};flex-shrink:0")
                        ui.label(name).classes("text-xs")
        with ui.row().classes("w-full flex-wrap gap-1 items-stretch") as grid:
            for tile in visible:
                with ui.card().classes("p-1").style("flex: 1 1 220px; max-width: 100%;"):
                    with ui.row().classes("plot-drag-handle w-full items-center gap-1 cursor-grab active:cursor-grabbing").props(f'role=button aria-label="Reorder {tile.key}"') as handle:
                        handle.tooltip(PLOT_REORDER_HINT)
                        ui.icon("drag_indicator").classes("text-gray-500 text-sm")
                        ui.label(tile.key).classes("text-xs font-medium")
                    ui.plotly(overlay_figure(tile.series, tile.unit, step_hours)).classes("w-full")
        grid_holder["grid"] = grid
        ui.timer(0.2, enable_sortable, once=True)
    return enable_sortable


class ResultsPanels:
    """Keeps Tables/Plots mounted and patches Plotly figures on warm re-runs."""

    def __init__(self, host) -> None:
        self._host = host
        self._layout: tuple[object, ...] | None = None
        self._plot_widgets: dict[str, Any] = {}
        self._tables_host: Any | None = None
        self._visible: list[VisiblePlot] = []
        self._order: list[str] = []
        self._sortable_hooks: list[Callable[[], None]] = []
        self._mounted = False
        self._tables_payload: tuple[dict[str, tuple[Model, Model]], Sequence[SimulationRun], Mapping[str, str]] | None = None

    def show(
        self,
        results: dict[str, tuple[Model, Model]],
        runs: Sequence[SimulationRun],
        step_hours: int,
        prefs: UiPreferences,
        on_prefs_changed: Callable[[UiPreferences], None],
        tiles: Sequence[VisiblePlot] | None = None,
    ) -> None:
        if not results:
            self._host.clear()
            self._mounted = False
            self._layout = None
            self._plot_widgets.clear()
            self._tables_payload = None
            with self._host:
                ui.label("No results yet.").classes("text-gray-500")
            return

        self._tables_payload = (results, runs, prefs.unit_alternatives)

        if tiles is not None:
            order = [tile.key for tile in tiles]
            if should_persist_plot_order(prefs.plot_order, merge_plot_order(prefs.plot_order, order)):
                prefs.plot_order = merge_plot_order(prefs.plot_order, order)
                on_prefs_changed(prefs)
            layout = results_layout_key(results, runs, order)
            if self._mounted and layout == self._layout and self._plot_widgets:
                self._patch(tiles, step_hours)
                return
            self._remount(results, runs, step_hours, prefs, on_prefs_changed, tiles, list(order), layout)
            return

        first_history = next(iter(results.values()))[1]
        available = list_plot_keys(first_history)
        order = merge_plot_order(prefs.plot_order, available)
        if should_persist_plot_order(prefs.plot_order, order):
            prefs.plot_order = order
            on_prefs_changed(prefs)
        else:
            prefs.plot_order = order
        built_tiles = build_visible_plots(results, runs, order, step_hours, prefs.unit_alternatives)
        layout = results_layout_key(results, runs, [tile.key for tile in built_tiles])
        if self._mounted and layout == self._layout and self._plot_widgets:
            self._patch(built_tiles, step_hours)
            return
        self._remount(results, runs, step_hours, prefs, on_prefs_changed, built_tiles, order, layout)

    def _patch(self, tiles: Sequence[VisiblePlot], step_hours: int) -> None:
        """Update plot traces only — no DOM remount, no table rebuild, no prefs I/O."""
        self._visible = list(tiles)
        for tile in tiles:
            widget = self._plot_widgets.get(tile.key)
            if widget is not None:
                widget.update_figure(overlay_figure(tile.series, tile.unit, step_hours))

    def _remount(
        self,
        results: dict[str, tuple[Model, Model]],
        runs: Sequence[SimulationRun],
        step_hours: int,
        prefs: UiPreferences,
        on_prefs_changed: Callable[[UiPreferences], None],
        tiles: Sequence[VisiblePlot],
        order: list[str],
        layout: tuple[object, ...],
    ) -> None:
        self._host.clear()
        self._plot_widgets.clear()
        self._sortable_hooks.clear()
        self._visible = list(tiles)
        self._order = list(order)
        self._layout = layout

        if not tiles:
            self._mounted = False
            with self._host:
                ui.label("No plottable variables.").classes("text-gray-500")
            return

        def on_result_tab_change(e) -> None:
            tab = str(getattr(e, "value", ""))
            if tab == "Tables" and self._tables_payload is not None and self._tables_host is not None:
                payload_results, payload_runs, alts = self._tables_payload
                render_scalar_tables(self._tables_host, payload_results, payload_runs, alts)
            if tab != "Plots":
                return
            for enable in self._sortable_hooks:
                enable()

        def on_sort_end(e) -> None:
            keys = [tile.key for tile in self._visible]
            old_index = int(getattr(e, "old_index", -1))
            new_index = int(getattr(e, "new_index", -1))
            reordered = move_plot_key(keys, old_index, new_index)
            by_key = {tile.key: tile for tile in self._visible}
            self._visible = [by_key[key] for key in reordered]
            hidden = [key for key in self._order if key not in reordered]
            prefs.plot_order = reordered + hidden
            on_prefs_changed(prefs)

        grid_holder: dict[str, Row] = {}

        def enable_sortable() -> None:
            grid = grid_holder.get("grid")
            if grid is None:
                return
            if getattr(grid, "_sortable", None) is None:
                grid.make_sortable(handle=".plot-drag-handle", on_end=on_sort_end, options=PLOT_TILE_SORTABLE_OPTIONS)

            def rebind() -> None:
                ui.run_javascript(sortable_rebind_javascript(grid.html_id))

            ui.timer(0.2, rebind, once=True)

        with self._host:
            with ui.tabs().classes("w-full") as tabs:
                tables_tab = ui.tab("Tables")
                plots_tab = ui.tab("Plots")
            with ui.tab_panels(tabs, value=plots_tab, on_change=on_result_tab_change).classes("w-full"):
                with ui.tab_panel(tables_tab):
                    self._tables_host = ui.column().classes("w-full gap-3")
                    render_scalar_tables(self._tables_host, results, runs, prefs.unit_alternatives)
                with ui.tab_panel(plots_tab):
                    plots_host = ui.column().classes("w-full gap-2")
                    with plots_host:
                        legend = overlay_legend_entries(runs, results)
                        if len(legend) > 1:
                            with ui.row().classes("w-full flex-wrap items-center gap-x-3 gap-y-0"):
                                for name, color in legend:
                                    with ui.row().classes("items-center gap-1 no-wrap"):
                                        ui.element("div").style(f"width:10px;height:10px;border-radius:2px;background:{color};flex-shrink:0")
                                        ui.label(name).classes("text-xs")
                        with ui.row().classes("w-full flex-wrap gap-1 items-stretch") as grid:
                            for tile in tiles:
                                with ui.card().classes("p-1").style("flex: 1 1 220px; max-width: 100%;"):
                                    with ui.row().classes("plot-drag-handle w-full items-center gap-1 cursor-grab active:cursor-grabbing").props(f'role=button aria-label="Reorder {tile.key}"') as handle:
                                        handle.tooltip(PLOT_REORDER_HINT)
                                        ui.icon("drag_indicator").classes("text-gray-500 text-sm")
                                        ui.label(tile.key).classes("text-xs font-medium")
                                    plot = ui.plotly(overlay_figure(tile.series, tile.unit, step_hours)).classes("w-full")
                                    self._plot_widgets[tile.key] = plot
                        grid_holder["grid"] = grid
                        ui.timer(0.2, enable_sortable, once=True)
                    self._sortable_hooks.append(enable_sortable)

        self._mounted = True


def render_results_panels(
    results_host,
    results: dict[str, tuple[Model, Model]],
    runs: Sequence[SimulationRun],
    step_hours: int,
    prefs: UiPreferences,
    on_prefs_changed: Callable[[UiPreferences], None],
) -> None:
    """Tables / Plots sub-tabs for simulation results (full remount)."""
    ResultsPanels(results_host).show(results, runs, step_hours, prefs, on_prefs_changed)
