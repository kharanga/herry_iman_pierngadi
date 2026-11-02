import numpy as np

from vision.court import CORNER_NAMES, click_window_frame, clickable_point_names, court_diagram


def test_the_window_shows_the_diagram_top_left_circles_at_placed_points_and_no_names_beside_them():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    names = clickable_point_names()
    placed = {CORNER_NAMES[0]: (900, 500)}
    shown = click_window_frame(frame, names, names[1], placed)
    diagram = court_diagram(names, names[1], placed)
    assert np.array_equal(shown[10 : 10 + diagram.shape[0], 10 : 10 + diagram.shape[1]], diagram)
    assert (shown[10, 10 : 10 + diagram.shape[1]] == 255).all()
    assert shown[500, 900 + 6].any() and not shown[500, 900].any()
    assert not shown[470:480, 905:1100].any()
    assert shown[700 - 15 : 700, 20:400].any()
    assert not frame.any()


def test_the_diagram_is_360_high_bordered_and_its_unplaced_points_are_bright():
    names = clickable_point_names()
    diagram = court_diagram(names, names[0], {})
    assert diagram.shape[0] == 360 and (diagram[0] == 255).all() and (diagram[-1] == 255).all()
    assert (diagram.reshape(-1, 3) == (190, 190, 190)).all(axis=1).any()

