"""Process lab tab: hierarchy tree, editable inputs, output explorer."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from nicegui import ui

from orion.ui.entity_editor import render_entity_editor
from orion.ui.process_explorer import (
    ParamSpec,
    ProcessInfo,
    SweepResult,
    default_input_for,
    discover_process_classes,
    editable_fields_for,
    get_process,
    process_hierarchy_tree,
    run_process_sweep,
)
from orion.ui.reflect import InputField


def _explore_figure(result: SweepResult, output_key: str, x_param: str | None) -> go.Figure:
    ys = [row.get(output_key, float("nan")) for row in result.outputs]
    unit = result.output_units.get(output_key, "")
    if x_param and x_param in result.param_paths:
        idx = result.param_paths.index(x_param)
        xs = [point[idx] for point in result.param_values]
        x_title = x_param
    else:
        xs = list(range(len(ys)))
        x_title = "run"
    fig = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="lines+markers", name=output_key)])
    fig.update_layout(
        margin=dict(l=28, r=8, t=8, b=24),
        height=180,
        xaxis_title=x_title,
        yaxis_title=unit or output_key,
        template="plotly_white",
        font=dict(size=10),
    )
    return fig


def _render_explore(explore, result: SweepResult) -> None:
    explore.clear()
    keys = sorted({key for row in result.outputs for key in row})
    if not keys:
        with explore:
            ui.label("No scalar output variables.").classes("text-gray-500")
        return
    range_params = [p for p in result.param_paths if len({point[result.param_paths.index(p)] for point in result.param_values}) > 1]
    with explore:
        ui.label("Output explorer").classes("text-subtitle2").tooltip("Run the process to plot output variables.")
        with ui.row().classes("gap-2 flex-wrap items-end"):
            out_select = ui.select(keys, value=keys[0], label="Output variable").classes("min-w-72")
            x_options = ["(index)", *range_params] if range_params else ["(index)"]
            x_select = ui.select(x_options, value=range_params[0] if range_params else "(index)", label="X axis").classes("min-w-72")
        plot = ui.plotly(_explore_figure(result, keys[0], range_params[0] if range_params else None)).classes("w-full")

        def update(_) -> None:
            x_param = None if x_select.value == "(index)" else str(x_select.value)
            plot.update_figure(_explore_figure(result, str(out_select.value), x_param))

        out_select.on_value_change(update)
        x_select.on_value_change(update)

        with ui.expansion("Result table", value=False).classes("w-full"):
            rows = []
            for point, row in zip(result.param_values, result.outputs, strict=True):
                item = {path: value for path, value in zip(result.param_paths, point, strict=True)}
                item.update(row)
                rows.append(item)
            if rows:
                ui.table(columns=[{"name": k, "label": k, "field": k} for k in rows[0]], rows=rows).classes("w-full").props("dense")


def _bind_sweep_row(path: str, edit: dict[str, Any]) -> None:
    """Range controls for a process sweep. The quantity value itself uses the shared entity editor."""
    with ui.row().classes("w-full items-end gap-2 flex-wrap"):
        ui.label(path).classes("font-mono text-xs min-w-56")
        mode = ui.toggle(["value", "range"], value=edit["mode"])
        start_box = ui.number("Start", value=edit["start"], format="%.4g").classes("w-28")
        stop_box = ui.number("Stop", value=edit["stop"], format="%.4g").classes("w-28")
        steps_box = ui.number("Steps", value=edit["steps"], min=2, max=50, step=1).classes("w-24")

        def sync_visibility() -> None:
            is_range = mode.value == "range"
            start_box.set_visibility(is_range)
            stop_box.set_visibility(is_range)
            steps_box.set_visibility(is_range)

        def on_change(_) -> None:
            edit["mode"] = str(mode.value)
            edit["start"] = float(start_box.value or 0.0)
            edit["stop"] = float(stop_box.value or 0.0)
            edit["steps"] = int(steps_box.value or 2)
            sync_visibility()

        mode.on_value_change(on_change)
        start_box.on_value_change(on_change)
        stop_box.on_value_change(on_change)
        steps_box.on_value_change(on_change)
        sync_visibility()


def _render_detail(detail, explore, info: ProcessInfo, selected: dict[str, Any]) -> None:
    detail.clear()
    fields = editable_fields_for(info) if info.input_states else ()
    selected["info"] = info
    selected["fields"] = fields
    selected["edits"] = {f.path: {"mode": "value", "value": f.value, "start": f.value, "stop": f.value + 1.0, "steps": 5} for f in fields}
    selected["result"] = None
    with detail:
        ui.label(info.label).classes("text-subtitle1 font-medium").tooltip(info.doc or info.module)
        ui.label(info.module).classes("text-xs text-gray-400 font-mono")
        with ui.row().classes("gap-2 flex-wrap"):
            ui.badge("implemented" if info.implemented else "abstract").props("outline")
            for cls in info.input_states:
                ui.badge(f"in: {cls.__name__}").props("outline color=primary")
            for cls in info.output_states:
                ui.badge(f"out: {cls.__name__}").props("outline color=secondary")

        if not info.implemented:
            ui.label("This process is abstract and cannot be run.").classes("text-amber-700")
            explore.clear()
            return

        if not fields:
            ui.label("No numeric scalar inputs.").classes("text-gray-500")
        else:
            ui.label("Inputs").classes("text-subtitle2 mt-2")
            try:
                sample = default_input_for(info)
            except TypeError:
                sample = None
            if sample is None:
                ui.label("No numeric scalar inputs.").classes("text-gray-500")
            else:
                owner = sample.name

                def on_quantity(name: str, value: float) -> None:
                    edit = selected["edits"].get(f"{owner}.{name}")
                    if edit is not None:
                        edit["value"] = value

                def after_field(field: InputField) -> None:
                    path = f"{owner}.{field.name}"
                    edit = selected["edits"].get(path)
                    if edit is not None:
                        _bind_sweep_row(path, edit)

                with ui.expansion(f"Input · {owner}", value=True).classes("w-full"):
                    render_entity_editor(sample, on_quantity, live=True, after_field=after_field)

        def on_run() -> None:
            params = [
                ParamSpec(
                    path=path,
                    mode=edit["mode"],  # type: ignore[arg-type]
                    value=float(edit["value"]),
                    start=float(edit["start"]),
                    stop=float(edit["stop"]),
                    steps=int(edit["steps"]),
                )
                for path, edit in selected["edits"].items()
            ]
            try:
                result = run_process_sweep(info, params)
            except Exception as exc:  # noqa: BLE001
                ui.notify(str(exc), type="negative")
                return
            selected["result"] = result
            _render_explore(explore, result)
            ui.notify(f"{len(result.outputs)} evaluation(s)")

        ui.button("Run / sweep", icon="science", on_click=on_run).props("dense")

    explore.clear()
    with explore:
        ui.label("Output explorer").classes("text-subtitle2").tooltip("Run the process to plot output variables.")


def render_process_tab(host) -> None:
    """Render the process exploration workspace into host."""
    host.clear()
    processes = discover_process_classes()
    tree_nodes = process_hierarchy_tree(processes)
    selected: dict[str, Any] = {"info": None, "fields": (), "edits": {}, "result": None}

    with host:
        with ui.row().classes("w-full gap-2 items-start flex-wrap"):
            with ui.column().classes("w-64 min-w-56 gap-1"):
                ui.label("Class hierarchy").classes("text-subtitle2").tooltip("Select a process, then edit inputs (value or range) and explore outputs.")
                tree = ui.tree(tree_nodes, node_key="id", label_key="label", children_key="children").classes("w-full")
                tree.props("dense default-expand-all")

            detail = ui.column().classes("flex-1 min-w-64 gap-2")
            explore = ui.column().classes("w-full gap-2")

        def on_select(e) -> None:
            key = e.value
            if not key or str(key).startswith("group:"):
                return
            try:
                info = get_process(str(key), processes)
            except KeyError:
                return
            _render_detail(detail, explore, info, selected)

        tree.on_select(on_select)
