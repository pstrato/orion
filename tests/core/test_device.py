"""Behaviour: a simulation runs on the device selected by settings.use_gpu."""

# Subclasses replace ``*args`` with the concrete parameters they require.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from datetime import date
from pathlib import Path

import jax
import pytest

from orion.core.constant import const
from orion.core.entity import entity
from orion.core.input import Inputs
from orion.core.model import simulate
from orion.core.process import Process
from orion.core.setting import Settings
from orion.core.variable import var
from orion.processes.clock import Clock, ClockInput

_SEEN: list[object] = []


@entity()
class RememberDevice(Process):
    def step(self, clock: Clock) -> None:
        del clock
        _SEEN.append(jax.default_device.value)


def _settings(*, use_gpu: bool) -> Settings:
    return Settings("settings", Path("."), False, False, False, use_gpu)


def _inputs() -> Inputs:
    clock = ClockInput(
        "clock",
        var("start", "isodate", date(2024, 1, 1), description="start"),
        var("end", "isodate", date(2024, 1, 2), description="end"),
        const("delta", "hours", 24, description="step"),
    )
    return Inputs("field", (clock,))


def test_a_simulation_runs_on_the_cpu_when_the_gpu_flag_is_off():
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", None)
    _SEEN.clear()
    try:
        simulate(_settings(use_gpu=False), _inputs(), (RememberDevice("remember"),), jit=True)
    finally:
        jax.config.update("jax_default_device", cpu)
    assert _SEEN[0] is cpu


def test_a_simulation_runs_on_the_gpu_when_the_flag_is_on(monkeypatch: pytest.MonkeyPatch):
    cpu = jax.devices("cpu")[0]
    real_devices = jax.devices
    requested: list[str | None] = []

    def devices(platform: str | None = None):
        requested.append(platform)
        if platform == "gpu":
            return [cpu]
        return real_devices(platform)

    monkeypatch.setattr(jax, "devices", devices)
    jax.config.update("jax_default_device", None)
    _SEEN.clear()
    try:
        simulate(_settings(use_gpu=True), _inputs(), (RememberDevice("remember"),), jit=True)
    finally:
        jax.config.update("jax_default_device", cpu)
    assert "gpu" in requested
    assert _SEEN[0] is cpu


def test_a_simulation_is_rejected_when_the_gpu_flag_is_on_and_no_gpu_is_present(monkeypatch: pytest.MonkeyPatch):
    cpu = jax.devices("cpu")[0]
    real_devices = jax.devices

    def devices(platform: str | None = None):
        if platform == "gpu":
            raise RuntimeError("Unknown backend: 'gpu'")
        return real_devices(platform)

    monkeypatch.setattr(jax, "devices", devices)
    jax.config.update("jax_default_device", cpu)
    with pytest.raises(ValueError, match="GPU"):
        simulate(_settings(use_gpu=True), _inputs(), (RememberDevice("remember"),))
    assert jax.default_device.value == cpu
