"""Behaviour: an entity lists its child entities and the relation path to every descendant."""

from __future__ import annotations

from orion.core.entity import Entity, EntityRelation


def test_direct_entities_pairs_each_immediate_child_entity_with_its_field_relation():
    class Cell(Entity):
        pass

    class Organ(Entity):
        tip: Cell

    class Crop(Entity):
        leaves: Organ
        roots: Organ
        label: str

    leaves = Organ("leaves", Cell("leaf tip"))
    roots = Organ("roots", Cell("root tip"))
    crop = Crop("wheat", leaves, roots, "spring")

    assert list(crop.direct_entities()) == [
        (EntityRelation("leaves"), leaves),
        (EntityRelation("roots"), roots),
    ]


def test_direct_entities_indexes_entities_stored_in_a_tuple_field():
    class Organ(Entity):
        pass

    class Crop(Entity):
        organs: tuple[Organ | None, ...]

    first = Organ("first")
    third = Organ("third")
    crop = Crop("wheat", (first, None, third))

    assert list(crop.direct_entities()) == [
        (EntityRelation("organs", index=0), first),
        (EntityRelation("organs", index=2), third),
    ]


def test_all_entities_lists_self_and_descendants_with_their_relation_path():
    class Cell(Entity):
        pass

    class Organ(Entity):
        tip: Cell

    class Crop(Entity):
        organs: tuple[Organ, ...]
        label: str

    leaf_tip = Cell("leaf tip")
    root_tip = Cell("root tip")
    leaves = Organ("leaves", leaf_tip)
    roots = Organ("roots", root_tip)
    crop = Crop("wheat", (leaves, roots), "spring")

    assert list(crop.all_entities()) == [
        ((), crop),
        ((EntityRelation("organs", index=0),), leaves),
        ((EntityRelation("organs", index=0), EntityRelation("tip")), leaf_tip),
        ((EntityRelation("organs", index=1),), roots),
        ((EntityRelation("organs", index=1), EntityRelation("tip")), root_tip),
    ]


def test_all_entities_limits_the_walk_to_one_requested_type():
    class Cell(Entity):
        pass

    class Organ(Entity):
        tip: Cell

    class Crop(Entity):
        organs: tuple[Organ, ...]

    leaf_tip = Cell("leaf tip")
    root_tip = Cell("root tip")
    leaves = Organ("leaves", leaf_tip)
    roots = Organ("roots", root_tip)
    crop = Crop("wheat", (leaves, roots))

    assert list(crop.all_entities(of_type=Cell)) == [
        ((EntityRelation("organs", index=0), EntityRelation("tip")), leaf_tip),
        ((EntityRelation("organs", index=1), EntityRelation("tip")), root_tip),
    ]


def test_all_entities_limits_the_walk_to_any_of_the_requested_types():
    class Cell(Entity):
        pass

    class Organ(Entity):
        tip: Cell

    class Crop(Entity):
        organs: tuple[Organ, ...]

    leaf_tip = Cell("leaf tip")
    leaves = Organ("leaves", leaf_tip)
    roots = Organ("roots", Cell("root tip"))
    crop = Crop("wheat", (leaves, roots))

    assert list(crop.all_entities(of_type=(Organ, Cell))) == [
        ((EntityRelation("organs", index=0),), leaves),
        ((EntityRelation("organs", index=0), EntityRelation("tip")), leaf_tip),
        ((EntityRelation("organs", index=1),), roots),
        ((EntityRelation("organs", index=1), EntityRelation("tip")), roots.tip),
    ]


def test_direct_entities_includes_an_entity_returned_by_a_property():
    class Measure(Entity):
        pass

    class Organ(Entity):
        height: float

        @property
        def thickness(self) -> Measure:
            return Measure("thickness")

    organ = Organ("leaf", 1.0)

    assert list(organ.direct_entities()) == [(EntityRelation("thickness"), Measure("thickness"))]


def test_direct_entities_indexes_entities_returned_by_a_tuple_property():
    class Organ(Entity):
        pass

    class Crop(Entity):
        @property
        def organs(self) -> tuple[Organ | None, ...]:
            return (Organ("first"), None, Organ("third"))

    crop = Crop("wheat")
    relations = list(crop.direct_entities())

    assert [relation for relation, _entity in relations] == [
        EntityRelation("organs", index=0),
        EntityRelation("organs", index=2),
    ]
    assert [entity.name for _relation, entity in relations] == ["first", "third"]


def test_direct_entities_skips_properties_that_do_not_return_entities():
    class Clock(Entity):
        step: int

        @property
        def label(self) -> str:
            return "noon"

    assert list(Clock("clock", 0).direct_entities()) == []


def test_direct_entities_does_not_read_a_property_annotated_as_a_non_entity():
    class Clock(Entity):
        step: int

        @property
        def date(self) -> str:
            raise RuntimeError("date is not an entity")

    assert list(Clock("clock", 0).direct_entities()) == []


def test_direct_entities_includes_a_property_declared_on_a_base_class():
    class Organ(Entity):
        pass

    class Crop(Entity):
        @property
        def roots(self) -> Organ:
            return Organ("roots")

    class Wheat(Crop):
        leaves: Organ

    leaves = Organ("leaves")
    wheat = Wheat("wheat", leaves)
    relations = list(wheat.direct_entities())

    assert relations[0] == (EntityRelation("leaves"), leaves)
    assert relations[1][0] == EntityRelation("roots")
    assert relations[1][1].name == "roots"


def test_all_entities_follows_entities_reached_through_a_property():
    class Cell(Entity):
        pass

    class Organ(Entity):
        tip: Cell

    class Crop(Entity):
        source: Organ

        @property
        def leaves(self) -> Organ:
            return self.source

    tip = Cell("tip")
    source = Organ("source", tip)
    crop = Crop("wheat", source)

    assert list(crop.all_entities()) == [
        ((), crop),
        ((EntityRelation("source"),), source),
        ((EntityRelation("source"), EntityRelation("tip")), tip),
        ((EntityRelation("leaves"),), source),
        ((EntityRelation("leaves"), EntityRelation("tip")), tip),
    ]
