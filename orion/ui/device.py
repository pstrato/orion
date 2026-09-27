"""Choose the JAX device the UI uses for simulation."""

from __future__ import annotations


def apply_compute_device(use_gpu: bool) -> str:
    """Select a GPU or the CPU for later UI work. Returns a label such as ``cpu:0``."""
    import jax

    from orion.core.device import compute_device

    device = compute_device(use_gpu)
    jax.config.update("jax_default_device", device)
    return f"{device.platform}:{device.id}"
