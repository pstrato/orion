"""Day-length playground: a latitude slider and the hours of daylight through the year."""

from __future__ import annotations

import plotly.graph_objects as go
from nicegui import ui

from orion.ui.process_explorer import ProcessInfo, day_length_lab_input
from orion.ui.reflect import list_input_fields, numeric_scalar, slider_limits
from orion.ui.theme import PRIMARY


def day_length_hours(latitude: float, *, altitude: float = 0.0, apparent: bool = False) -> tuple[tuple[int, float], ...]:
    """Day length for each day of a 365-day year, at one latitude."""
    import jax.numpy as jnp

    from orion.processes.day_length import apparent_day_length, astronomical_day_length

    days = jnp.arange(1, 366)
    lat = jnp.asarray(latitude, dtype=jnp.float32)
    if apparent:
        hours = apparent_day_length(lat, days.astype(jnp.float32), jnp.asarray(altitude, dtype=jnp.float32))
    else:
        hours = astronomical_day_length(lat, days.astype(jnp.float32))
    return tuple((int(day), float(hour)) for day, hour in zip(days.tolist(), jnp.reshape(hours, (-1)).tolist(), strict=True))


def day_length_figure(latitude: float, *, altitude: float = 0.0, apparent: bool = False) -> go.Figure:
    """Line chart of day length against day of year."""
    series = day_length_hours(latitude, altitude=altitude, apparent=apparent)
    figure = go.Figure(data=[go.Scatter(x=[day for day, _hours in series], y=[hours for _day, hours in series], mode="lines", line=dict(color=PRIMARY, width=2))])
    figure.update_layout(
        margin=dict(l=48, r=16, t=12, b=40),
        autosize=True,
        xaxis=dict(title="Day of year", range=[1, 365], dtick=30),
        yaxis=dict(title="Day length (h)", range=[0, 24], dtick=4),
        template="plotly_white",
        font=dict(size=12),
        showlegend=False,
    )
    return figure


def _apparent(info: ProcessInfo) -> bool:
    from orion.processes.day_length import ApparentDayLengthProcess

    return isinstance(info.cls, type) and issubclass(info.cls, ApparentDayLengthProcess)


def render_day_length_lab(host, info: ProcessInfo) -> None:
    """Latitude slider above the yearly day-length curve. Moving the slider redraws the curve."""
    host.clear()
    lab = day_length_lab_input()
    apparent = _apparent(info)
    latitude = next(field for field in list_input_fields(lab) if field.name == "latitude")
    if latitude.lower is None or latitude.upper is None:
        raise TypeError("Latitude has no bounds.")
    lower, upper = slider_limits(latitude.lower, latitude.upper, strict=latitude.strict)
    current = numeric_scalar(lab.latitude.value)
    if current is None:
        raise TypeError("Latitude is not a scalar.")
    chosen = {"latitude": current, "altitude": 0.0}
    alt_box = None
    alt_readout = None

    with host:
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            name = ui.label("Latitude").classes("text-xs text-gray-600 w-20 shrink-0")
            name.tooltip(latitude.description or "Latitude in degrees north")
            box = ui.slider(min=lower, max=upper, step=1.0, value=current).classes("flex-1")
            readout = ui.label(_latitude_text(current)).classes("orion-light-readout")
        if apparent:
            altitude = next(field for field in list_input_fields(lab) if field.name == "altitude")
            if altitude.lower is None or altitude.upper is None:
                raise TypeError("Altitude has no bounds.")
            alt_lower, alt_upper = slider_limits(altitude.lower, altitude.upper, strict=altitude.strict)
            alt_current = numeric_scalar(lab.altitude.value)
            if alt_current is None:
                raise TypeError("Altitude is not a scalar.")
            chosen["altitude"] = alt_current
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                alt_name = ui.label("Altitude").classes("text-xs text-gray-600 w-20 shrink-0")
                alt_name.tooltip(altitude.description or "Altitude above sea level")
                alt_box = ui.slider(min=alt_lower, max=alt_upper, step=50.0, value=alt_current).classes("flex-1")
                alt_readout = ui.label(_altitude_text(alt_current)).classes("orion-light-readout")
        plot = ui.plotly(day_length_figure(current, altitude=chosen["altitude"], apparent=apparent)).classes("w-full orion-daylength-plot")

    def redraw() -> None:
        plot.update_figure(day_length_figure(chosen["latitude"], altitude=chosen["altitude"], apparent=apparent))

    def on_latitude(event) -> None:
        raw = getattr(event, "value", None)
        if raw is None:
            return
        chosen["latitude"] = float(raw)
        readout.set_text(_latitude_text(chosen["latitude"]))
        redraw()

    box.on_value_change(on_latitude)
    if alt_box is not None and alt_readout is not None:

        def on_altitude(event) -> None:
            raw = getattr(event, "value", None)
            if raw is None:
                return
            chosen["altitude"] = float(raw)
            assert alt_readout is not None
            alt_readout.set_text(_altitude_text(chosen["altitude"]))
            redraw()

        alt_box.on_value_change(on_altitude)


def _latitude_text(value: float) -> str:
    hemisphere = "N" if value > 0 else "S" if value < 0 else ""
    return f"{abs(value):.0f}°{hemisphere}"


def _altitude_text(value: float) -> str:
    return f"{value:.0f} m"
