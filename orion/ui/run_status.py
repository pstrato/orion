"""Format Simulation-tab status lines after a run."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

_LABELS = {"openmeteo": "Open-Meteo", "soilgrids": "SoilGrids"}
_ORDER = ("openmeteo", "soilgrids", "simulate", "display")

_active: ContextVar[RunTiming | None] = ContextVar("orion_ui_run_timing", default=None)


@dataclass
class RunTiming:
    """Wall time accumulated while a simulation run is in progress."""

    seconds: dict[str, float] = field(default_factory=dict)

    def add(self, source: str, seconds: float) -> None:
        self.seconds[source] = self.seconds.get(source, 0.0) + seconds


def format_run_timing(timing: RunTiming) -> str:
    """Clients, then simulate and display, then the total."""
    parts: list[str] = []
    seen: set[str] = set()
    for key in _ORDER:
        if key not in timing.seconds:
            continue
        seen.add(key)
        parts.append(f"{_LABELS.get(key, key)} {timing.seconds[key]:.2f}s")
    for key, seconds in timing.seconds.items():
        if key in seen:
            continue
        parts.append(f"{_LABELS.get(key, key)} {seconds:.2f}s")
    total = sum(timing.seconds.values())
    parts.append(f"total {total:.2f}s")
    return " · ".join(parts)


def format_done_status(run_count: int, timing: RunTiming) -> str:
    """Done line with run count and client/simulate timings."""
    return f"Done · {run_count} run(s) · {format_run_timing(timing)}"


@contextmanager
def collect_timing():
    timing = RunTiming()
    token = _active.set(timing)
    try:
        yield timing
    finally:
        _active.reset(token)


@contextmanager
def timed(source: str):
    timing = _active.get()
    started = perf_counter()
    try:
        yield
    finally:
        if timing is not None:
            timing.add(source, perf_counter() - started)
