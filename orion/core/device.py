"""Select the JAX device a simulation runs on."""

from __future__ import annotations

import jax


def compute_device(use_gpu: bool):
    """The device selected by a settings GPU flag."""
    platform = "gpu" if use_gpu else "cpu"
    try:
        devices = jax.devices(platform)
    except RuntimeError:
        devices = []
    if not devices:
        raise ValueError(f"No {platform.upper()} is available.")
    return devices[0]
