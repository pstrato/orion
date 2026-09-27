"""What a quantity measures, beyond the unit it is stored in."""

from __future__ import annotations

from typing import Literal

Resource = Literal["water", "nitrogen", "phosphorus", "potassium", "heat", "light"]
"""A quantity may represent one of these resources. Storage stays in its unit."""
