"""Canopy playground for light interception: drag organs, read light as it changes."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from nicegui import ui

from orion.core.parameter import Parameter
from orion.ui.process_explorer import CANOPY_SHAPE_LABELS, LightInterceptionLabInput, ProcessInfo, SweepResult, canopy_shape, light_interception_lab_input, run_process_sweep
from orion.ui.reflect import list_input_fields, numeric_scalar, set_quantity_value, slider_limits

VIEW_W = 300
VIEW_H = 360
PLOT_LEFT = 42.0
PLOT_RIGHT = 284.0
PLOT_TOP = 16.0
PLOT_BOTTOM = 312.0
MIN_SPAN = 0.05

_ORGANS = (
    ("leaves", "#3FA34D", "#1B4332"),
    ("ears", "#E6B325", "#8A6A00"),
)
_READINGS = (
    ("light_interception.soil", "Soil", "#8D6E63"),
    ("light_interception.plant", "Plant", "#2F9E44"),
    ("light_interception.canopy.0", "Leaves", "#3FA34D"),
    ("light_interception.canopy.1", "Ears", "#E6B325"),
)


@dataclass(frozen=True)
class CanopyFrame:
    """Fixed plot so a drag does not rescale under the pointer."""

    height_max: float = 3.0
    density_max: float = 4.0
    plot_left: float = PLOT_LEFT
    plot_right: float = PLOT_RIGHT
    plot_top: float = PLOT_TOP
    plot_bottom: float = PLOT_BOTTOM


def canopy_frame() -> CanopyFrame:
    return CanopyFrame()


def height_to_y(frame: CanopyFrame, height: float) -> float:
    span = frame.plot_bottom - frame.plot_top
    t = min(max(height / frame.height_max, 0.0), 1.0)
    return frame.plot_bottom - t * span


def y_to_height(frame: CanopyFrame, y: float) -> float:
    span = frame.plot_bottom - frame.plot_top
    t = (frame.plot_bottom - y) / span
    return min(max(t, 0.0), 1.0) * frame.height_max


def density_to_x(frame: CanopyFrame, density: float) -> float:
    span = frame.plot_right - frame.plot_left
    t = min(max(density / frame.density_max, 0.0), 1.0)
    return frame.plot_left + t * span


def x_to_density(frame: CanopyFrame, x: float) -> float:
    span = frame.plot_right - frame.plot_left
    t = (x - frame.plot_left) / span
    return min(max(t, 0.0), 1.0) * frame.density_max


def organ_density(heights: tuple[float, ...], *, area_index: float, bottom: float, top: float, shape: str = "rectangle") -> list[float]:
    """Area per metre at each height, using the same shape the lab simulates."""
    import jax.numpy as jnp

    from orion.core.variable import var

    values = canopy_shape(shape, "organ").density(
        var("area_index", "m^2/m^2", area_index, "Area index"),
        var("bottom", "m", bottom, "Bottom"),
        var("top", "m", top, "Top"),
        var("height", "m", jnp.asarray(heights, dtype=jnp.float32), "Height"),
    )
    return [float(value) for value in jnp.reshape(values, (-1)).tolist()]


def drag_organ(
    *,
    bottom: float,
    top: float,
    area_index: float,
    edge: str,
    height: float,
    density: float | None = None,
    height_max: float | None = None,
) -> tuple[float, float, float]:
    """Move one edge. Area stays put unless the density knob is dragged."""
    if edge == "top":
        top = max(float(height), bottom + MIN_SPAN)
        if height_max is not None and top > height_max:
            top = height_max
            bottom = max(0.0, min(bottom, top - MIN_SPAN))
    elif edge == "bottom":
        bottom = min(float(height), top - MIN_SPAN)
        if bottom < 0.0:
            bottom = 0.0
            top = max(top, MIN_SPAN)
    elif edge == "move":
        bottom = bottom + float(height)
        top = top + float(height)
        if bottom < 0.0:
            top -= bottom
            bottom = 0.0
        if height_max is not None and top > height_max:
            shift = top - height_max
            top = height_max
            bottom -= shift
            if bottom < 0.0:
                bottom = 0.0
    elif edge == "density":
        span = max(top - bottom, MIN_SPAN)
        area_index = max(0.0, float(density or 0.0)) * span
    return bottom, top, area_index


def edit_canopy(
    lab: LightInterceptionLabInput,
    organ: str,
    *,
    edge: str,
    height: float,
    density: float | None = None,
    height_max: float | None = None,
) -> LightInterceptionLabInput:
    current = getattr(lab, organ)
    bottom, top, area = drag_organ(
        bottom=_scalar(current.bottom),
        top=_scalar(current.top),
        area_index=_scalar(current.area_index),
        edge=edge,
        height=height,
        density=density,
        height_max=height_max,
    )
    return _replace_geometry(lab, organ, bottom, top, area)


def evaluate_light_interception(info: ProcessInfo, lab: LightInterceptionLabInput) -> SweepResult:
    """Run the selected implementation once, at the canopy's current values."""
    return run_process_sweep(info, (), base=lab)


def set_canopy_shape(lab: LightInterceptionLabInput, organ: str, shape: str) -> LightInterceptionLabInput:
    """Replace one organ's vertical area distribution."""
    if shape not in CANOPY_SHAPE_LABELS:
        raise ValueError(f"Unknown canopy shape {shape!r}.")
    current = getattr(lab, organ)
    updated = replace(current, shape=shape)
    if organ == "leaves":
        return replace(lab, leaves=updated)
    if organ == "ears":
        return replace(lab, ears=updated)
    raise KeyError(organ)


def canopy_svg(lab: LightInterceptionLabInput, prefix: str = "canopy") -> str:
    frame = canopy_frame()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}" width="100%" height="100%" preserveAspectRatio="none">',
        _axis_markup(frame),
    ]
    for name, fill, stroke in _ORGANS:
        parts.append(f'<g data-organ="{name}">')
        parts.extend(_markup(node) for node in _organ_nodes(lab, prefix, name, fill, stroke, frame))
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def canopy_update_script(lab: LightInterceptionLabInput, prefix: str, host_id: str) -> str:
    frame = canopy_frame()
    lines: list[str] = []
    for name, fill, stroke in _ORGANS:
        for node in _organ_nodes(lab, prefix, name, fill, stroke, frame):
            for key, value in node.attrs:
                lines.append(f"document.getElementById({json.dumps(node.id)})?.setAttribute({json.dumps(key)},{json.dumps(value)})")
    lines.append(fit_canopy_labels_script(host_id))
    return ";".join(lines)


def fit_canopy_labels_script(host_id: str) -> str:
    """Keep plot labels at 12px, the same size as the rest of the UI, when the figure is stretched."""
    return (
        "(() => {"
        f"const host = document.getElementById({json.dumps(host_id)});"
        "if (!host) return;"
        "const fit = () => {"
        "const rect = host.getBoundingClientRect();"
        "if (!rect.width || !rect.height) return;"
        f"const ax = {VIEW_W} / rect.width;"
        f"const ay = {VIEW_H} / rect.height;"
        "host.querySelectorAll('text[data-x]').forEach((node) => {"
        "const x = node.getAttribute('data-x');"
        "const y = node.getAttribute('data-y');"
        "node.setAttribute('transform', 'translate(' + x + ',' + y + ') scale(' + ax + ',' + ay + ')');"
        "node.setAttribute('font-size', '12');"
        "});"
        "};"
        "fit();"
        "if (!host._orionLabelFit) {"
        "host._orionLabelFit = new ResizeObserver(fit);"
        "host._orionLabelFit.observe(host);"
        "}"
        "})()"
    )


def render_light_interception_lab(host, info: ProcessInfo) -> None:
    """Canopy on the left, light and sliders on the right. Every edit recalculates."""
    host.clear()
    state: dict[str, object] = {"lab": light_interception_lab_input(), "refresh": lambda: None, "sync": lambda: None}
    drag = _Drag()
    fills: dict[str, object] = {}
    readouts: dict[str, object] = {}

    with host:
        with ui.row().classes("w-full items-start gap-3 no-wrap"):
            view = ui.html("", sanitize=False).classes("orion-canopy")
            with ui.column().classes("orion-light-panel"):
                with ui.column().classes("w-full gap-1"):
                    ui.label("Light").classes("text-subtitle2")
                    for key, title, color in _READINGS:
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.element("span").style(f"width:8px;height:8px;border-radius:2px;background:{color};flex:0 0 auto")
                            ui.label(title).classes("text-xs w-14 shrink-0")
                            with ui.element("div").classes("orion-light-track"):
                                fills[key] = ui.element("div").classes("orion-light-fill").style(f"background:{color}")
                            readouts[key] = ui.label("—").classes("orion-light-readout")
                with ui.column().classes("w-full gap-1"):
                    _bind_parameter_sliders(state)
                    ui.label("Drag a band to move it, its edges to resize, and the knob to change density.").classes("text-xs text-gray-500")
                    status = ui.label("").classes("text-xs text-negative")

    prefix = f"canopy-{view.id}"
    view.set_content(canopy_svg(_lab(state), prefix))
    ui.run_javascript(fit_canopy_labels_script(view.html_id))

    def sync() -> None:
        ui.run_javascript(canopy_update_script(_lab(state), prefix, view.html_id))

    def refresh() -> None:
        try:
            result = evaluate_light_interception(info, _lab(state))
        except Exception as exc:  # noqa: BLE001
            status.set_text(str(exc))
            return
        status.set_text("")
        row = result.outputs[0] if result.outputs else {}
        scale = _scalar(_lab(state).radiation) or 1.0
        for key, _title, color in _READINGS:
            value = row.get(key)
            fill = fills[key]
            readout = readouts[key]
            if value is None:
                readout.set_text("—")  # type: ignore[attr-defined]
                fill.style("width: 0%")  # type: ignore[attr-defined]
                continue
            share = max(0.0, min(100.0, 100.0 * float(value) / scale))
            fill.style(f"width: {share:.1f}%; background: {color}")  # type: ignore[attr-defined]
            unit = result.output_units.get(key, "")
            readout.set_text(f"{float(value):.1f} {unit}".strip())  # type: ignore[attr-defined]

    state["refresh"] = refresh
    state["sync"] = sync

    def on_down(event) -> None:
        payload = _pointer_payload(event)
        organ = str(payload.get("organ") or "")
        edge = str(payload.get("edge") or "")
        if organ not in {"leaves", "ears"} or edge not in {"move", "top", "bottom", "density"}:
            return
        current = getattr(_lab(state), organ)
        drag.organ = organ
        drag.edge = edge
        drag.bottom = _scalar(current.bottom)
        drag.top = _scalar(current.top)
        drag.area = _scalar(current.area_index)
        drag.anchor = _number(payload.get("y"))

    def on_move(event) -> None:
        if not drag.organ:
            return
        payload = _pointer_payload(event)
        frame = canopy_frame()
        pointer_y = _number(payload.get("y"))
        if drag.edge == "move":
            height = y_to_height(frame, pointer_y) - y_to_height(frame, drag.anchor)
            density = None
        elif drag.edge == "density":
            height = 0.0
            density = x_to_density(frame, _number(payload.get("x")))
        else:
            height = y_to_height(frame, pointer_y)
            density = None
        bottom, top, area = drag_organ(
            bottom=drag.bottom,
            top=drag.top,
            area_index=drag.area,
            edge=drag.edge,
            height=height,
            density=density,
            height_max=frame.height_max,
        )
        state["lab"] = _replace_geometry(_lab(state), drag.organ, bottom, top, area)
        sync()
        refresh()

    def on_up(_event) -> None:
        if not drag.organ:
            return
        drag.organ = ""
        sync()
        refresh()

    pointer_js = _pointer_javascript()
    view.on("pointerdown", on_down, js_handler=pointer_js)
    view.on("pointermove", on_move, throttle=0.05, js_handler=pointer_js)
    view.on("pointerup", on_up, js_handler=pointer_js)
    view.on("pointercancel", on_up, js_handler=pointer_js)
    refresh()


@dataclass
class _Drag:
    organ: str = ""
    edge: str = ""
    bottom: float = 0.0
    top: float = 0.0
    area: float = 0.0
    anchor: float = 0.0


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


@dataclass(frozen=True)
class _Node:
    id: str
    tag: str
    attrs: tuple[tuple[str, str], ...]
    text: str = ""


def _lab(state: dict[str, object]) -> LightInterceptionLabInput:
    lab = state["lab"]
    if not isinstance(lab, LightInterceptionLabInput):
        raise TypeError("Canopy lab state is missing.")
    return lab


def _bind_parameter_sliders(state: dict[str, object]) -> None:
    fields = {field.name: field for field in list_input_fields(_lab(state))}
    extinction = fields["leaves.k"]
    if extinction.lower is None or extinction.upper is None:
        raise TypeError("Extinction coefficient has no bounds.")
    lower, upper = slider_limits(extinction.lower, extinction.upper, strict=extinction.strict)
    for organ_name, label in (("leaves", "Leaves k"), ("ears", "Ears k")):
        _slider(
            label,
            value=_scalar(getattr(_lab(state), organ_name).k),
            lower=lower,
            upper=upper,
            step=0.01,
            digits=2,
            on_value=lambda value, name=organ_name: _commit_k(state, name, value),
        )
        _shape_select(state, organ_name, "Leaves" if organ_name == "leaves" else "Ears")
    _slider(
        "Radiation",
        value=_scalar(_lab(state).radiation),
        lower=0.0,
        upper=400.0,
        step=1.0,
        digits=0,
        on_value=lambda value: _commit_radiation(state, value),
    )


def _commit_k(state: dict[str, object], organ_name: str, value: float) -> None:
    lab = _lab(state)
    if organ_name == "leaves":
        state["lab"] = replace(lab, leaves=replace(lab.leaves, k=_set_parameter(lab.leaves.k, value)))
    elif organ_name == "ears":
        state["lab"] = replace(lab, ears=replace(lab.ears, k=_set_parameter(lab.ears.k, value)))
    else:
        raise KeyError(organ_name)
    refresh = state["refresh"]
    if callable(refresh):
        refresh()


def _commit_radiation(state: dict[str, object], value: float) -> None:
    lab = _lab(state)
    state["lab"] = replace(lab, radiation=_set_parameter(lab.radiation, value))
    refresh = state["refresh"]
    if callable(refresh):
        refresh()


def _shape_select(state: dict[str, object], organ_name: str, label: str) -> None:
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        ui.label(label).classes("text-xs text-gray-600 w-20 shrink-0")
        box = ui.select(CANOPY_SHAPE_LABELS, value=getattr(_lab(state), organ_name).shape).props("dense options-dense").classes("flex-1")

    def commit(_event=None, name: str = organ_name, control=box) -> None:
        value = str(control.value or "")
        if not value or value == getattr(_lab(state), name).shape:
            return
        state["lab"] = set_canopy_shape(_lab(state), name, value)
        sync = state["sync"]
        refresh = state["refresh"]
        if callable(sync):
            sync()
        if callable(refresh):
            refresh()

    box.on_value_change(commit)


def _slider(label: str, *, value: float, lower: float, upper: float, step: float, digits: int, on_value) -> None:
    shown = min(max(value, lower), upper)
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        ui.label(label).classes("text-xs text-gray-600 w-20 shrink-0")
        box = ui.slider(min=lower, max=upper, step=step, value=shown).classes("flex-1")
        readout = ui.label(f"{shown:.{digits}f}").classes("orion-light-readout")

    def commit(event) -> None:
        raw = getattr(event, "value", None)
        if raw is None:
            return
        number = float(raw)
        readout.set_text(f"{number:.{digits}f}")
        on_value(number)

    box.on_value_change(commit)


def _pointer_javascript() -> str:
    return f"""(e) => {{
  const host = e.currentTarget;
  const edgeEl = e.target && e.target.closest ? e.target.closest("[data-edge]") : null;
  const organEl = e.target && e.target.closest ? e.target.closest("[data-organ]") : null;
  if (e.type === "pointerdown") {{
    if (!edgeEl) return;
    host._orionDrag = true;
    if (host.setPointerCapture) host.setPointerCapture(e.pointerId);
    e.preventDefault();
  }}
  if (e.type === "pointermove" && !host._orionDrag) return;
  if (e.type === "pointerup" || e.type === "pointercancel") host._orionDrag = false;
  const rect = host.getBoundingClientRect();
  const x = rect.width ? (e.clientX - rect.left) / rect.width * {VIEW_W} : 0;
  const y = rect.height ? (e.clientY - rect.top) / rect.height * {VIEW_H} : 0;
  emit({{
    x: x,
    y: y,
    organ: organEl ? organEl.getAttribute("data-organ") : "",
    edge: edgeEl ? edgeEl.getAttribute("data-edge") : "",
  }});
}}"""


def _pointer_payload(event) -> dict[str, object]:
    raw = getattr(event, "args", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        return raw[0]
    return {}


def _replace_geometry(lab: LightInterceptionLabInput, organ_name: str, bottom: float, top: float, area: float) -> LightInterceptionLabInput:
    if organ_name == "leaves":
        organ = lab.leaves
        return replace(lab, leaves=_geometry(organ, bottom, top, area))
    if organ_name == "ears":
        organ = lab.ears
        return replace(lab, ears=_geometry(organ, bottom, top, area))
    raise KeyError(organ_name)


def _geometry(organ, bottom: float, top: float, area: float):
    return replace(
        organ,
        bottom=_set_parameter(organ.bottom, bottom),
        top=_set_parameter(organ.top, top),
        area_index=_set_parameter(organ.area_index, area),
    )


def _set_parameter(quantity: Parameter, value: float) -> Parameter:
    updated = set_quantity_value(quantity, value)
    if not isinstance(updated, Parameter):
        raise TypeError(quantity.name)
    return updated


def _scalar(quantity) -> float:
    value = numeric_scalar(quantity.value)
    if value is None:
        raise TypeError(f"{quantity.name} is not a scalar.")
    return value


def _organ_nodes(lab: LightInterceptionLabInput, prefix: str, name: str, fill: str, stroke: str, frame: CanopyFrame) -> tuple[_Node, ...]:
    organ = getattr(lab, name)
    bottom = _scalar(organ.bottom)
    top = _scalar(organ.top)
    area = _scalar(organ.area_index)
    span = max(top - bottom, MIN_SPAN)
    density = area / span
    y_bottom = height_to_y(frame, bottom)
    y_top = height_to_y(frame, top)
    x_density = density_to_x(frame, density)
    y_lo, y_hi = min(y_bottom, y_top), max(y_bottom, y_top)
    samples = tuple(bottom + (top - bottom) * (index + 0.5) / 16 for index in range(16))
    profile = organ_density(samples, area_index=area, bottom=bottom, top=top, shape=organ.shape) if top > bottom else [0.0]
    if not profile:
        profile = [0.0]
    heights = (bottom, *samples, top)
    densities = (profile[0], *profile, profile[-1])
    right = [(density_to_x(frame, item), height_to_y(frame, height)) for height, item in zip(heights, densities, strict=True)]
    points = [(frame.plot_left, y_bottom), (frame.plot_left, y_top), *reversed(right)]
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    mid_y = (y_lo + y_hi) / 2
    bottom_x = density_to_x(frame, profile[0])
    top_x = density_to_x(frame, profile[-1])
    label_x = frame.plot_left + 6.0
    return (
        _Node(
            f"{prefix}-{name}-hit",
            "rect",
            (
                ("data-edge", "move"),
                ("x", f"{frame.plot_left:.1f}"),
                ("y", f"{y_lo:.1f}"),
                ("width", f"{max(density_to_x(frame, max(profile)) - frame.plot_left, 14.0):.1f}"),
                ("height", f"{max(y_hi - y_lo, 8.0):.1f}"),
                ("fill", "transparent"),
                ("style", "cursor: grab"),
            ),
        ),
        _Node(
            f"{prefix}-{name}-poly",
            "polygon",
            (
                ("data-edge", "move"),
                ("points", point_text),
                ("fill", fill),
                ("fill-opacity", "0.78"),
                ("stroke", stroke),
                ("stroke-width", "1"),
                ("style", "cursor: grab"),
            ),
        ),
        _Node(
            f"{prefix}-{name}-top",
            "line",
            (
                ("data-edge", "top"),
                ("x1", f"{frame.plot_left:.1f}"),
                ("y1", f"{y_top:.1f}"),
                ("x2", f"{max(top_x, frame.plot_left + 14.0):.1f}"),
                ("y2", f"{y_top:.1f}"),
                ("stroke", stroke),
                ("stroke-width", "5"),
                ("stroke-linecap", "round"),
                ("style", "cursor: ns-resize"),
            ),
        ),
        _Node(
            f"{prefix}-{name}-bottom",
            "line",
            (
                ("data-edge", "bottom"),
                ("x1", f"{frame.plot_left:.1f}"),
                ("y1", f"{y_bottom:.1f}"),
                ("x2", f"{max(bottom_x, frame.plot_left + 14.0):.1f}"),
                ("y2", f"{y_bottom:.1f}"),
                ("stroke", stroke),
                ("stroke-width", "5"),
                ("stroke-linecap", "round"),
                ("style", "cursor: ns-resize"),
            ),
        ),
        _Node(
            f"{prefix}-{name}-knob",
            "rect",
            (
                ("data-edge", "density"),
                ("x", f"{x_density - 5.0:.1f}"),
                ("y", f"{mid_y - 5.0:.1f}"),
                ("width", "10"),
                ("height", "10"),
                ("rx", "2"),
                ("fill", "white"),
                ("stroke", stroke),
                ("stroke-width", "2"),
                ("style", "cursor: ew-resize"),
            ),
        ),
        _label_node(f"{prefix}-{name}-label", label_x, mid_y, name, fill=stroke),
    )


def _label_node(element_id: str, x: float, y: float, text: str, *, anchor: str = "start", fill: str = "#64748B") -> _Node:
    return _Node(
        element_id,
        "text",
        (
            ("x", "0"),
            ("y", "0"),
            ("data-x", f"{x:.1f}"),
            ("data-y", f"{y:.1f}"),
            ("text-anchor", anchor),
            ("font-size", "12"),
            ("fill", fill),
            ("font-family", "sans-serif"),
            ("pointer-events", "none"),
            ("transform", f"translate({x:.1f},{y:.1f})"),
        ),
        text,
    )


def _markup(node: _Node) -> str:
    attrs = " ".join(f'{key}="{value}"' for key, value in node.attrs)
    if node.tag == "text":
        return f'<{node.tag} id="{node.id}" {attrs}>{node.text}</{node.tag}>'
    return f'<{node.tag} id="{node.id}" {attrs}></{node.tag}>'


def _axis_markup(frame: CanopyFrame) -> str:
    parts = [
        f'<rect x="{frame.plot_left:.1f}" y="{frame.plot_top:.1f}" width="{frame.plot_right - frame.plot_left:.1f}" height="{frame.plot_bottom - frame.plot_top:.1f}" fill="#F8FBFE" stroke="#D7E3EE"></rect>',
    ]
    for step in range(7):
        height = step * 0.5
        y = height_to_y(frame, height)
        label = f"{height:.0f}" if height == int(height) else f"{height:.1f}"
        parts.append(f'<line x1="{frame.plot_left:.1f}" y1="{y:.1f}" x2="{frame.plot_right:.1f}" y2="{y:.1f}" stroke="#E2E8F0"></line>')
        parts.append(_markup(_label_node(f"axis-h-{step}", frame.plot_left - 4.0, y, label, anchor="end")))
    for density in (0.0, 2.0, 4.0):
        x = density_to_x(frame, density)
        parts.append(_markup(_label_node(f"axis-d-{density:.0f}", x, frame.plot_bottom + 14.0, f"{density:.0f}", anchor="middle")))
    parts.append(_markup(_label_node("axis-density", (frame.plot_left + frame.plot_right) / 2, frame.plot_bottom + 28.0, "density (m²/m³)", anchor="middle")))
    parts.append(f'<line x1="{frame.plot_left:.1f}" y1="{height_to_y(frame, 0.0):.1f}" x2="{frame.plot_right:.1f}" y2="{height_to_y(frame, 0.0):.1f}" stroke="#8D6E63" stroke-width="2"></line>')
    return "".join(parts)
