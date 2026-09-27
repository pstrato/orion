"""Process lab tab: pick a process type and experiment with an implementation."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from nicegui import ui

from orion.ui.canopy_lab import render_light_interception_lab
from orion.ui.entity_editor import render_entity_editor
from orion.ui.process_explorer import (
    ParamSpec,
    ProcessInfo,
    SweepResult,
    abstract_process_types,
    default_input_for,
    discover_process_classes,
    editable_fields_for,
    implementations_of,
    is_light_interception,
    playground_label,
    run_process_sweep,
)
from orion.ui.reflect import InputField
from orion.ui.theme import setting_row


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
    """Playground: pick an abstract process type, then an implementation to run."""
    processes = tuple(discover_process_classes())
    types = abstract_process_types(processes)
    chosen = {"type": types[0].key if types else "", "implementation": ""}

    def paint() -> None:
        host.clear()
        kind = next((item for item in types if item.key == chosen["type"]), None)
        impls = implementations_of(kind, processes) if kind is not None else ()
        if impls and chosen["implementation"] not in {item.key for item in impls}:
            chosen["implementation"] = impls[0].key
        impl = next((item for item in impls if item.key == chosen["implementation"]), None)
        with host:
            with ui.row().classes("w-full gap-3 items-start no-wrap"):
                _render_type_list(types, chosen["type"], paint, chosen)
                with ui.column().classes("flex-1 min-w-0 gap-2"):
                    if kind is None:
                        ui.label("No process types are available.").classes("text-xs text-gray-500")
                        return
                    if impl is None or not is_light_interception(impl):
                        ui.label(playground_label(kind)).classes("text-subtitle1 font-medium").tooltip(kind.doc or kind.module)
                    _render_implementation_select(impls, chosen, paint, wide=impl is not None and is_light_interception(impl))
                    if impl is None:
                        ui.label("This process type has no runnable implementation.").classes("text-amber-700")
                        return
                    if is_light_interception(impl):
                        render_light_interception_lab(ui.column().classes("w-full"), impl)
                        return
                    detail = ui.column().classes("w-full gap-2")
                    explore = ui.column().classes("w-full gap-2")
                    _render_detail(detail, explore, impl, {"info": None, "fields": (), "edits": {}, "result": None})

    paint()


def _render_type_list(types: tuple[ProcessInfo, ...], selected_key: str, paint, chosen: dict[str, str]) -> None:
    with ui.column().classes("w-56 min-w-44 shrink-0 gap-1"):
        ui.label("Processes").classes("text-subtitle2").tooltip("Abstract process types. Select one, then choose an implementation to experiment with.")
        for info in types:
            selected_cls = " orion-config-list-item--selected" if info.key == selected_key else ""

            def select(key: str = info.key) -> None:
                if key != chosen["type"]:
                    chosen["type"] = key
                    chosen["implementation"] = ""
                    paint()

            with ui.row().classes(f"orion-config-list-item items-center gap-2 no-wrap cursor-pointer w-full{selected_cls}").on("click", select):
                ui.label(playground_label(info)).classes("text-xs truncate")


def _render_implementation_select(impls: tuple[ProcessInfo, ...], chosen: dict[str, str], paint, *, wide: bool = False) -> None:
    if not impls:
        return
    options = {info.key: playground_label(info) for info in impls}
    with setting_row("Implementation", wide=wide):
        box = ui.select(options, value=chosen["implementation"]).props("dense options-dense")

    def on_implementation(_) -> None:
        value = str(box.value or "")
        if value and value != chosen["implementation"]:
            chosen["implementation"] = value
            paint()

    box.on_value_change(on_implementation)
