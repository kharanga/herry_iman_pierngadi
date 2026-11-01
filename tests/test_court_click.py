import numpy as np
import pytest

from vision.court import CORNER_NAMES, check_points_fix_a_homography, clickable_point_names, court_point_catalog, court_point_meters, points_are_enough, points_problem, same_point_pairs


def test_the_click_list_has_each_physical_point_once_with_the_corners_first():
    names = clickable_point_names()
    catalog = court_point_catalog()
    assert len(names) == 32 and names[:4] == list(CORNER_NAMES)
    assert all(name in catalog for name in names)
    assert len({catalog[name] for name in names}) == len(names)
    assert {catalog[name] for name in names} == set(catalog.values())


def test_same_point_pairs_find_two_names_of_one_point_in_either_order():
    catalog = court_point_catalog()
    corner = CORNER_NAMES[0]
    alias = next(name for name in catalog if name != corner and catalog[name] == catalog[corner])
    assert same_point_pairs([corner, alias]) == [(corner, alias)]
    assert same_point_pairs([alias, corner]) == [(alias, corner)]
    assert same_point_pairs(list(CORNER_NAMES)) == []
    assert same_point_pairs(["not a point", corner]) == []


def test_points_problem_names_the_reason_or_none_when_enter_can_save():
    catalog = court_point_catalog()
    corner = CORNER_NAMES[0]
    alias = next(name for name in catalog if name != corner and catalog[name] == catalog[corner])
    pixels = {name: (10 * index, 7 * index) for index, name in enumerate(CORNER_NAMES)}
    assert points_problem(pixels) is None and points_are_enough(pixels)
    doubled = {**pixels, alias: (500, 500)}
    assert points_problem(doubled) == f"same point clicked twice: {corner} / {alias}" and not points_are_enough(doubled)
    three = dict(list(pixels.items())[:3])
    with pytest.raises(ValueError) as expected:
        check_points_fix_a_homography(court_point_meters(list(three)))
    assert points_problem(three) == str(expected.value)
    assert points_problem({}) is not None


def test_points_on_one_line_are_refused_with_the_homography_check_message():
    catalog = court_point_catalog()
    by_y = {}
    for name in clickable_point_names():
        by_y.setdefault(catalog[name][1], []).append(name)
    line = next(names for names in by_y.values() if len(names) >= 4)[:4]
    message = points_problem({name: (index, index) for index, name in enumerate(line)})
    assert message is not None and "same point" not in message


def test_both_names_of_a_corner_still_resolve_for_points_files():
    catalog = court_point_catalog()
    for corner in CORNER_NAMES:
        aliases = [name for name in catalog if catalog[name] == catalog[corner]]
        assert len(aliases) >= 2
        assert all(np.allclose(court_point_meters([name])[0], catalog[corner]) for name in aliases)
