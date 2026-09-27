"""Behaviour: the GPU setting selects the JAX device used for simulation."""

from __future__ import annotations

import jax
import pytest

from orion.ui.device import apply_compute_device


def test_leaving_gpu_off_selects_the_cpu():
    label = apply_compute_device(False)
    device = jax.default_device.value
    assert label.startswith("cpu:")
    assert device.platform == "cpu"


def test_gpu_setting_selects_a_gpu_when_one_is_present(monkeypatch: pytest.MonkeyPatch):
    class GpuDevice:
        platform = "gpu"
        id = 0

    cpu = jax.devices("cpu")[0]

    def devices(platform: str | None = None):
        if platform == "gpu":
            return [GpuDevice()]
        if platform == "cpu":
            return [cpu]
        return [cpu]

    monkeypatch.setattr(jax, "devices", devices)
    try:
        assert apply_compute_device(True) == "gpu:0"
        assert jax.default_device.value.platform == "gpu"
    finally:
        apply_compute_device(False)


def test_gpu_setting_is_rejected_when_no_gpu_is_present(monkeypatch: pytest.MonkeyPatch):
    real_devices = jax.devices

    def devices(platform: str | None = None):
        if platform == "gpu":
            raise RuntimeError("Unknown backend: 'gpu'")
        return real_devices(platform)

    monkeypatch.setattr(jax, "devices", devices)
    before = jax.default_device.value
    with pytest.raises(ValueError, match="GPU"):
        apply_compute_device(True)
    assert jax.default_device.value == before
