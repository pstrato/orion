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
