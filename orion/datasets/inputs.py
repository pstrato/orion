"""Build simulation Inputs from a wheat site-year."""

from __future__ import annotations

from orion.core.constant import const
from orion.core.input import Input, Inputs, LocationInput
from orion.core.variable import var
from orion.datasets.site_year import SiteYear
from orion.processes.clock import ClockInput


def inputs_for_site_year(template: Inputs, site: SiteYear) -> Inputs:
    """Copy template inputs; location and clock horizon come from ``site``."""
    start, end = site.horizon()
    clock = ClockInput(
        "clock",
        var("start", "isodate", start, description="Simulation start date"),
        var("end", "isodate", end, description="Simulation end date"),
        const("delta", "hours", 24, description="Simulation delta step in hours"),
    )
    inputs: list[Input] = []
    saw_location = False
    saw_clock = False
    for item in template.inputs:
        if isinstance(item, LocationInput):
            inputs.append(site.location)
            saw_location = True
        elif isinstance(item, ClockInput):
            inputs.append(ClockInput(item.name, clock.start, clock.end, item.delta))
            saw_clock = True
        else:
            inputs.append(item)
    if not saw_location:
        inputs.append(site.location)
    if not saw_clock:
        inputs.insert(0, clock)
    return Inputs(site.name, tuple(inputs))
