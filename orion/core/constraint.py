from typing import Iterable

from orion.core.entity import Entity, EntityPath, entity


@entity()
class Problem(Entity):
    """A problem raised by a constraint."""

    description: str
    """Description of the problem."""

    path: EntityPath
    """Path to the entity that violates the constraint."""

    def __str__(self) -> str:
        return f"{'/'.join(map(str, self.path))} {self.name}: {self.description}"

    def __repr__(self) -> str:
        return str(self)


class Constraint[T: Entity](Entity):
    """A constraint on an entity."""

    def validate(self, path: EntityPath, entity: T) -> Iterable[Problem]:
        return ()

    def __add__(self, other: Constraint[T]):
        if isinstance(self, ConstraintGroup):
            if isinstance(other, ConstraintGroup):
                return ConstraintGroup[T](
                    name="constraint",
                    constraints=(*self.constraints, *other.constraints),
                )
            return ConstraintGroup(name="constraint", constraints=(*self.constraints, other))
        if isinstance(other, ConstraintGroup):
            return ConstraintGroup(name="constraint", constraints=(self, *other.constraints))
        return ConstraintGroup("constraint", (self, other))


class ConstraintGroup[T: Entity](Constraint[T]):
    """A group of constraints."""

    constraints: tuple[Constraint, ...]
    """Constraint group."""

    def validate(self, path: EntityPath, entity: T):
        for constraint in self.constraints:
            yield from constraint.validate(path, entity)


class ConstrainedEntity(Entity):
    constraint: Constraint | None
    """Entity constraint."""


def problems(entity: Entity):
    for path, e in entity.all_entities(of_type=ConstrainedEntity):
        if isinstance(e, ConstrainedEntity) and e.constraint:
            yield from e.constraint.validate(path, e)
