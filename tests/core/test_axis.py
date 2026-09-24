"""Behaviour: named axes label quantity dimensions for discovery."""

from __future__ import annotations

from orion.core.axis import axis


def test_axis_exposes_name_description_and_optional_labels():
    layers = axis("layer", description="Soil layer", labels=("0-5cm", "5-15cm"))
    assert layers.name == "layer"
    assert layers.description == "Soil layer"
    assert layers.labels == ("0-5cm", "5-15cm")
