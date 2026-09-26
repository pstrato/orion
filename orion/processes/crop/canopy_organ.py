from orion.core.constant import Constant
from orion.core.entity import entity
from orion.core.parameter import Parameter
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.variable import Variable, var
from orion.processes.crop.organ import Organ, OrganInput
from orion.processes.crop.shape import Shape


@entity()
class CanopyOrgan(Organ):
    """One canopy organ (leaves, stems, or ears)."""

    top: Variable
    """Height of the organ top above the soil (m)."""
    bottom: Variable
    """Height of the organ bottom above the soil (m)."""
    area_index: Variable
    """Organ area per ground area (m²/m²)."""
    k: Parameter
    """Light-extinction coefficient (dimensionless)."""
    shape: Constant[Shape]
    """Vertical distribution of area index between bottom and top."""

    @property
    def thickness(self) -> Variable:
        """Thickness of the organ (m)."""
        return var(
            "thickness",
            "m",
            self.top.value - self.bottom.value,
            "Thickness of the organ.",
            is_scalar + is_non_negative,
            self.top.axes,
        )


class CanopyOrganInput(OrganInput):  # type: ignore
    """Canopy organ parameters."""

    k: Parameter
    """Light-interception extinction coefficient (0 < k < 1)."""
    shape: Constant[Shape]
    """Canopy organ shape: density at height above the soil surface."""


def canopy_organ(name: str, input: CanopyOrganInput) -> CanopyOrgan:
    """Canopy organ state carrying this input's extinction coefficient and shape.

    Height and area index start at zero; growth processes replace them.
    """
    return CanopyOrgan(
        name=name,
        constraint=None,
        top=var("top", "m", 0.0, f"{name} top"),
        bottom=var("bottom", "m", 0.0, f"{name} bottom"),
        area_index=var("area_index", "m^2/m^2", 0.0, f"{name} area index"),
        k=input.k,
        shape=input.shape,
    )
