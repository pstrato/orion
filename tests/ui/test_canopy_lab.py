"""Behaviour: the light-interception lab shows a canopy you can drag, and light updates with it."""

from __future__ import annotations

import inspect

import pytest

from orion.ui.canopy_lab import canopy_frame, canopy_svg, density_to_x, drag_organ, edit_canopy, evaluate_light_interception, height_to_y, organ_density, set_canopy_shape, x_to_density, y_to_height
from orion.ui.process_explorer import get_process, light_interception_lab_input
from orion.ui.reflect import list_input_fields, slider_limits


def test_extinction_coefficient_is_bounded_between_zero_and_one():
    lab = light_interception_lab_input()
    field = next(item for item in list_input_fields(lab) if item.name == "leaves.k")

    assert field.lower is not None and field.upper is not None
    assert (field.lower, field.upper) == (0.0, 1.0)
    assert field.strict is True
    assert slider_limits(field.lower, field.upper, strict=True) == (0.01, 0.99)


def test_rectangle_canopy_density_is_area_over_height_inside_the_organ():
    samples = organ_density((0.25, 0.75, 1.5), area_index=2.0, bottom=0.0, top=1.0)

    assert samples == pytest.approx((2.0, 2.0, 0.0))


def test_upward_triangle_is_denser_at_the_bottom_and_downward_at_the_top():
    upward = organ_density((0.2, 0.8), area_index=2.0, bottom=0.0, top=1.0, shape="upward")
    downward = organ_density((0.2, 0.8), area_index=2.0, bottom=0.0, top=1.0, shape="downward")

    assert upward[0] > upward[1]
    assert downward[0] < downward[1]


def test_canopy_labels_use_the_same_font_size_as_the_rest_of_the_ui():
    svg = canopy_svg(light_interception_lab_input())

    assert 'font-size="12"' in svg
    assert 'font-size="10"' not in svg
    assert 'font-size="11"' not in svg


def test_plot_coordinates_round_trip_height_and_density():
    frame = canopy_frame()

    assert y_to_height(frame, height_to_y(frame, 1.25)) == pytest.approx(1.25)
    assert x_to_density(frame, density_to_x(frame, 3.0)) == pytest.approx(3.0)
    assert y_to_height(frame, frame.plot_bottom) == pytest.approx(0.0)


def test_dragging_the_top_keeps_area_and_a_minimum_thickness():
    bottom, top, area = drag_organ(bottom=0.0, top=1.0, area_index=2.0, edge="top", height=0.01)

    assert bottom == pytest.approx(0.0)
    assert top == pytest.approx(0.05)
    assert area == pytest.approx(2.0)


def test_dragging_the_body_slides_the_organ_without_crossing_the_ground_or_the_top():
    raised_bottom, raised_top, raised_area = drag_organ(bottom=0.2, top=1.2, area_index=2.0, edge="move", height=0.5)
    grounded_bottom, grounded_top, grounded_area = drag_organ(bottom=0.2, top=1.2, area_index=2.0, edge="move", height=-1.0)
    capped_bottom, capped_top, capped_area = drag_organ(bottom=2.2, top=2.8, area_index=1.0, edge="move", height=1.0, height_max=3.0)

    assert (raised_bottom, raised_top, raised_area) == pytest.approx((0.7, 1.7, 2.0))
    assert (grounded_bottom, grounded_top, grounded_area) == pytest.approx((0.0, 1.0, 2.0))
    assert (capped_bottom, capped_top, capped_area) == pytest.approx((2.4, 3.0, 1.0))


def test_dragging_the_density_knob_changes_area_and_keeps_the_organ_in_place():
    bottom, top, area = drag_organ(bottom=0.0, top=1.0, area_index=2.0, edge="density", height=0.0, density=4.0)

    assert bottom == pytest.approx(0.0)
    assert top == pytest.approx(1.0)
    assert area == pytest.approx(4.0)


def test_canopy_drawing_places_organs_by_height_and_density():
    frame = canopy_frame()
    svg = canopy_svg(light_interception_lab_input())

    assert 'data-organ="leaves"' in svg
    assert 'data-organ="ears"' in svg
    assert 'data-edge="density"' in svg
    assert f"{height_to_y(frame, 0.0):.1f}" in svg
    assert f"{height_to_y(frame, 2.0):.1f}" in svg
    assert f"{density_to_x(frame, 2.0):.1f}" in svg


def test_an_upward_triangle_is_drawn_wider_at_the_bottom():
    frame = canopy_frame()
    lab = set_canopy_shape(light_interception_lab_input(), "leaves", "upward")
    low, high = organ_density((0.5 / 16, 15.5 / 16), area_index=2.0, bottom=0.0, top=1.0, shape="upward")
    svg = canopy_svg(lab)

    assert low > high
    assert f"{density_to_x(frame, low):.1f}" in svg
    assert f"{density_to_x(frame, high):.1f}" in svg


def test_leaf_shape_changes_how_much_light_the_leaves_take_where_they_overlap_the_ears():
    info = get_process("orion.processes.crop.light_interception.BeerLambertLightInterceptionProcess")
    overlapped = edit_canopy(light_interception_lab_input(), "ears", edge="bottom", height=0.2)
    overlapped = edit_canopy(overlapped, "ears", edge="top", height=1.2)
    downward = evaluate_light_interception(info, set_canopy_shape(overlapped, "leaves", "downward"))
    upward = evaluate_light_interception(info, set_canopy_shape(overlapped, "leaves", "upward"))

    assert downward.outputs[0]["light_interception.canopy.0"] > upward.outputs[0]["light_interception.canopy.0"]


def test_lowering_leaf_density_lets_more_light_reach_the_soil():
    info = get_process("orion.processes.crop.light_interception.BeerLambertLightInterceptionProcess")
    covered = light_interception_lab_input()
    open_leaves = edit_canopy(covered, "leaves", edge="density", height=0.0, density=0.0)

    covered_soil = evaluate_light_interception(info, covered).outputs[0]["light_interception.soil"]
    open_soil = evaluate_light_interception(info, open_leaves).outputs[0]["light_interception.soil"]

    assert open_soil > covered_soil


def test_light_interception_page_recalculates_without_a_run_button():
    import orion.ui.canopy_lab as lab
    import orion.ui.process_tab as tab

    page = inspect.getsource(lab.render_light_interception_lab)
    assert "evaluate_light_interception" in page
    assert "Run / sweep" not in page
    assert "render_light_interception_lab" in inspect.getsource(tab.render_process_tab)
