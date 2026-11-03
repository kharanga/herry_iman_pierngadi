import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision import court
from vision.alignment import open_video, read_frame
from vision.camera import camera_reference_meters, fit_camera, reprojection_errors as camera_reprojection_errors
from vision.court import (
    CORNER_NAMES,
    court_corner_meters,
    court_line_segments,
    court_lines_across,
    court_lines_along,
    court_point_catalog,
    court_point_meters,
    court_to_pixels,
    check_points_fix_a_homography,
    homography_from_corners,
    homography_from_court_points,
    line_offsets_to_bright_pixels,
    load_calibration,
    main,
    parse_point_argument,
    point_calibration,
    points_are_enough,
    reference_points,
    save_calibration,
)
from tests.results_pages import docs_on_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "court.md"
BROADCAST_VIDEO = PROJECT_ROOT / "data" / "match.mp4"
BROADCAST_CALIBRATION = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
STAND_IN_VIDEO = PROJECT_ROOT / "data" / "recordings" / "stand_in" / "set1.mp4"
STAND_IN_CALIBRATION = PROJECT_ROOT / "data" / "calibration" / "set1" / "court.json"
STAND_IN_CAMERA = PROJECT_ROOT / "data" / "calibration" / "set1" / "camera.json"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
BROADCAST_CORNERS = [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]]
BWF_ACROSS_M = {"near_baseline": 0.0, "near_long_service_line": 0.76, "near_short_service_line": 4.72, "far_short_service_line": 8.68, "far_long_service_line": 12.64, "far_baseline": 13.4}
BWF_ALONG_M = {"left_doubles_sideline": 0.0, "left_singles_sideline": 0.46, "centre_line": 3.05, "right_singles_sideline": 5.64, "right_doubles_sideline": 6.1}
HIDDEN_CORNER_POINTS = {
    "near_right": (987.2, 657.4),
    "far_right": (864.2, 301.6),
    "far_left": (412.6, 303.0),
    "near_long_service_line x left_doubles_sideline": (301.0, 629.5),
    "near_short_service_line x left_doubles_sideline": (347.5, 495.0),
}
STAND_IN_POINTS = {
    "near_left": (217.0, 762.0),
    "near_long_service_line x left_doubles_sideline": (383.0, 736.0),
    "near_long_service_line x centre_line": (574.0, 889.0),
    "near_baseline x centre_line": (356.0, 960.0),
    "near_short_service_line x centre_line": (1140.0, 718.0),
    "near_short_service_line x right_singles_sideline": (1584.0, 810.0),
    "near_short_service_line x right_doubles_sideline": (1714.0, 837.0),
}
STAND_IN_NET_POSTS = [[1004.0, 496.0], [1748.0, 500.0]]


def point_arguments(points: dict) -> list[str]:
    return [f"{name}={x},{y}" for name, (x, y) in points.items()]


def test_catalog_has_the_30_line_crossings_at_bwf_positions_plus_corner_aliases_and_post_bases():
    catalog = court_point_catalog()
    assert len(catalog) == 36 and court_lines_across() == pytest.approx(BWF_ACROSS_M) and court_lines_along() == pytest.approx(BWF_ALONG_M)
    for across, y_m in BWF_ACROSS_M.items():
        for along, x_m in BWF_ALONG_M.items():
            assert catalog[f"{across} x {along}"] == pytest.approx((x_m, y_m))
    for name, crossing in zip(CORNER_NAMES, ("near_baseline x left_doubles_sideline", "near_baseline x right_doubles_sideline", "far_baseline x right_doubles_sideline", "far_baseline x left_doubles_sideline")):
        assert catalog[name] == pytest.approx(catalog[crossing])
    assert catalog["left_net_post_base"] == pytest.approx((0.0, 6.7)) and catalog["right_net_post_base"] == pytest.approx((6.1, 6.7))
    assert list(catalog)[:4] == list(CORNER_NAMES)


def test_list_points_prints_every_name_with_its_position(tmp_path):
    result = subprocess.run([sys.executable, "-m", "vision.court", str(tmp_path / "any.mp4"), "--list-points"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    assert result.returncode == 0 and len(lines) == 36
    assert "near_short_service_line x centre_line: (3.05, 4.72) m" in lines


def test_point_checks_name_each_problem():
    with pytest.raises(ValueError, match="unknown court point"):
        court_point_meters(["near_left", "middle_of_nowhere"])
    with pytest.raises(ValueError, match="at least 4"):
        check_points_fix_a_homography(court_point_meters(["near_left", "near_right", "far_left"]))
    with pytest.raises(ValueError, match="twice"):
        check_points_fix_a_homography(court_point_meters(["near_left", "near_baseline x left_doubles_sideline", "far_left", "far_right", "near_right"]))
    on_two_lines = ["near_baseline x left_doubles_sideline", "near_baseline x centre_line", "near_baseline x right_doubles_sideline", "near_long_service_line x centre_line"]
    with pytest.raises(ValueError, match="3 on one line"):
        check_points_fix_a_homography(court_point_meters(on_two_lines))
    check_points_fix_a_homography(court_point_meters(on_two_lines + ["near_short_service_line x left_singles_sideline"]))
    check_points_fix_a_homography(court_point_meters(["near_baseline x left_doubles_sideline", "near_baseline x right_doubles_sideline", "near_long_service_line x centre_line", "far_baseline x centre_line"]))
    names = list(HIDDEN_CORNER_POINTS)
    pixels = np.array(list(HIDDEN_CORNER_POINTS.values()))
    with pytest.raises(ValueError, match="pixels for"):
        homography_from_court_points(pixels[:4], names)
    with pytest.raises(ValueError, match="finite"):
        homography_from_court_points(np.vstack([pixels[:4], [np.nan, 1.0]]), names)
    assert points_are_enough(dict(zip(names, pixels))) and not points_are_enough(dict(zip(names[:3], pixels[:3])))


def test_point_argument_parsing_keeps_names_with_spaces():
    assert parse_point_argument("near_long_service_line x left_doubles_sideline=301,629.5") == ("near_long_service_line x left_doubles_sideline", (301.0, 629.5))
    for bad in ("301,629.5", "=1,2", "near_left=1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_point_argument(bad)


def broadcast_homography() -> np.ndarray:
    return homography_from_corners(BROADCAST_CORNERS)


def exact_pixels(names: list[str]) -> np.ndarray:
    return court_to_pixels(broadcast_homography(), court_point_meters(names))


def test_four_points_are_exact_and_more_points_recover_the_same_homography():
    homography = broadcast_homography()
    names = ["near_right", "far_right", "far_left", "near_short_service_line x left_doubles_sideline"]
    four = homography_from_court_points(exact_pixels(names), names)
    assert four / four[2, 2] == pytest.approx(homography / homography[2, 2], rel=1e-6, abs=1e-9)
    names += ["near_long_service_line x centre_line", "far_short_service_line x right_singles_sideline", "far_long_service_line x left_singles_sideline"]
    seven = homography_from_court_points(exact_pixels(names), names)
    assert court_to_pixels(seven, court_corner_meters()) == pytest.approx(np.array(BROADCAST_CORNERS), abs=1e-3)


def test_point_calibration_projects_hidden_corners_and_reports_every_points_error():
    names = ["near_right", "far_right", "far_left", "near_long_service_line x left_doubles_sideline", "near_short_service_line x left_doubles_sideline"]
    pixels = exact_pixels(names) + np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, -1.0], [0.0, 0.0]])
    calibration = point_calibration(Path("v.mp4"), 7, (1280, 720), dict(zip(names, map(tuple, pixels))))
    assert calibration["corner_pixels_source"] == "projected from the reference points"
    assert [point["name"] for point in calibration["reference_points"]] == names
    errors = np.array([point["error_px"] for point in calibration["reference_points"]])
    assert calibration["max_reference_error_px"] == pytest.approx(errors.max()) and calibration["rms_reference_error_px"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert 0 < errors.max() < 1.5 and np.linalg.norm(np.array(calibration["corner_pixels"][0]) - BROADCAST_CORNERS[0]) < 1.5
    assert reference_points(calibration)[0] == names
    clicked = point_calibration(Path("v.mp4"), 7, (1280, 720), dict(zip(CORNER_NAMES, map(tuple, BROADCAST_CORNERS))))
    assert clicked["corner_pixels_source"] == "clicked" and np.array(clicked["corner_pixels"]) == pytest.approx(np.array(BROADCAST_CORNERS))


def test_old_corner_calibrations_load_and_keep_exactly_the_old_keys(tmp_path):
    saved = save_calibration(tmp_path / "court.json", Path("v.mp4"), 3, (1280, 720), np.array(BROADCAST_CORNERS))
    assert not {"reference_points", "corner_pixels_source", "max_reference_error_px", "rms_reference_error_px"} & set(saved)
    names, pixels, meters = reference_points(load_calibration(tmp_path / "court.json"))
    assert names == list(CORNER_NAMES) and pixels == pytest.approx(np.array(BROADCAST_CORNERS)) and meters == pytest.approx(court_corner_meters())


def write_blank_video(video_path: Path, width: int = 1280, height: int = 720) -> Path:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))
    for _ in range(2):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    return video_path


def test_command_with_points_writes_a_point_calibration_and_rejects_bad_input(tmp_path, capsys):
    video = write_blank_video(tmp_path / "club.avi")
    names = list(HIDDEN_CORNER_POINTS)
    arguments = point_arguments(dict(zip(names, map(tuple, exact_pixels(names)))))
    assert main([str(video), "--frame", "1", "--output-dir", str(tmp_path / "out"), "--points", *arguments]) == 0
    saved = load_calibration(tmp_path / "out" / court.video_id(video) / "court.json")
    assert saved["max_reference_error_px"] < 1e-3 and (tmp_path / "out" / court.video_id(video) / "court_overlay.jpg").exists()
    for bad in (arguments[:3], arguments[:4] + ["nowhere=1,2"], arguments + ["near_baseline x right_doubles_sideline=987,657"]):
        assert main([str(video), "--frame", "1", "--output-dir", str(tmp_path / "bad"), "--points", *bad]) == 1
    assert not (tmp_path / "bad").exists() or not list((tmp_path / "bad").rglob("court.json"))
    both = subprocess.run([sys.executable, "-m", "vision.court", str(video), "--points", *arguments, "--corners", "1,2", "3,4", "5,6", "7,8"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert both.returncode == 2


class StepWindow:
    def __init__(self, steps):
        self.steps = list(steps)
        self.mouse_callback = None
        self.destroyed = False
        self.shown = []

    def wait_key(self, delay):
        if not self.steps:
            return 27
        clicks, key = self.steps.pop(0)
        for x, y in clicks:
            self.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return key


@pytest.fixture
def step_window(monkeypatch):
    def install(steps) -> StepWindow:
        window = StepWindow(steps)
        monkeypatch.setattr(cv2, "namedWindow", lambda *arguments: None)
        monkeypatch.setattr(cv2, "imshow", lambda name, frame: window.shown.append(frame.copy()))
        monkeypatch.setattr(cv2, "setMouseCallback", lambda name, callback: setattr(window, "mouse_callback", callback))
        monkeypatch.setattr(cv2, "waitKey", window.wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *arguments: 1.0)
        monkeypatch.setattr(cv2, "destroyWindow", lambda name: setattr(window, "destroyed", True))
        return window

    return install


def picker_frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_picker_saves_four_corners_clicked_in_order(step_window):
    window = step_window([([(100, 600), (900, 600), (800, 200), (300, 200)], -1), ([], 13)])
    assert court.click_court_points(picker_frame()) == {"near_left": (100.0, 600.0), "near_right": (900.0, 600.0), "far_right": (800.0, 200.0), "far_left": (300.0, 200.0)}
    assert window.destroyed and window.shown


def test_picker_skips_a_hidden_corner_and_saves_once_four_points_fix_a_homography(step_window):
    long_service = court.clickable_point_names().index("near_long_service_line x left_doubles_sideline")
    steps = [([], ord("n")), ([(900, 600), (800, 200), (300, 200)], -1), ([], 13)]
    steps += [([], ord("n"))] * (long_service - 4) + [([(150, 560)], 13)]
    window = step_window(steps)
    picked = court.click_court_points(picker_frame())
    assert picked == {"near_right": (900.0, 600.0), "far_right": (800.0, 200.0), "far_left": (300.0, 200.0), "near_long_service_line x left_doubles_sideline": (150.0, 560.0)}
    assert window.steps == []


def test_picker_delete_previous_and_escape(step_window):
    clicks = [(100, 600), (900, 600), (800, 200), (300, 200)]
    window = step_window([(clicks, ord("p")), ([], ord("d")), ([], 13), ([], 27)])
    assert court.click_court_points(picker_frame()) is None and window.steps == [] and window.destroyed
    window = step_window([([(100, 600)], 13), ([], 13), ([], 27)])
    assert court.click_court_points(picker_frame()) is None and window.steps == []


def floor_line_errors(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return np.abs(np.concatenate([line_offsets_to_bright_pixels(frame, homography, start, end) for name, (start, end) in court_line_segments().items() if name != "net"]))


@pytest.mark.skipif(not (BROADCAST_VIDEO.exists() and BROADCAST_CALIBRATION.exists()), reason="needs data/match.mp4 and its court calibration")
def test_broadcast_without_the_near_left_corner_matches_the_four_corner_lines_and_the_results_doc(tmp_path):
    assert main([str(BROADCAST_VIDEO), "--frame", "12300", "--output-dir", str(tmp_path), "--points", *point_arguments(HIDDEN_CORNER_POINTS)]) == 0
    hidden = load_calibration(tmp_path / "match" / "court.json")
    corners = load_calibration(BROADCAST_CALIBRATION)
    doc = RESULTS_PATH.read_text()
    errors = [point["error_px"] for point in hidden["reference_points"]]
    assert f"| Reference point errors | 0 (exact with 4) | {min(errors):.2f} to {max(errors):.2f} px |" in doc
    hidden_corner = np.array(hidden["corner_pixels"][0])
    assert f"({hidden_corner[0]:.1f}, {hidden_corner[1]:.1f}), {np.linalg.norm(hidden_corner - corners['corner_pixels'][0]):.2f} px away" in doc
    capture = open_video(BROADCAST_VIDEO)
    all_errors = {"corners": [], "hidden": []}
    try:
        for frame_index in (12300, 95659, 126000):
            frame = read_frame(capture, frame_index)
            for name, calibration in (("corners", corners), ("hidden", hidden)):
                all_errors[name].append(floor_line_errors(frame, np.array(calibration["pixel_to_court_homography"])))
    finally:
        capture.release()
    corner_errors, hidden_errors = np.concatenate(all_errors["corners"]), np.concatenate(all_errors["hidden"])
    assert abs(np.median(hidden_errors) - np.median(corner_errors)) <= 2.0
    assert f"| All three frames | median {np.median(corner_errors):.1f}, mean {corner_errors.mean():.2f} | median {np.median(hidden_errors):.1f}, mean {hidden_errors.mean():.2f} |" in doc


@pytest.mark.skipif(not (STAND_IN_VIDEO.exists() and STAND_IN_CALIBRATION.exists() and STAND_IN_CAMERA.exists()), reason="needs data/recordings/stand_in/set1.mp4 and its calibration")
def test_stand_in_calibrates_from_seven_points_and_reproduces_the_shipped_calibration_camera_and_doc(tmp_path):
    assert main([str(STAND_IN_VIDEO), "--frame", "600", "--output-dir", str(tmp_path), "--points", *point_arguments(STAND_IN_POINTS)]) == 0
    mine = load_calibration(tmp_path / "set1" / "court.json")
    shipped = json.loads(STAND_IN_CALIBRATION.read_text())
    assert {key: value for key, value in json.loads((tmp_path / "set1" / "court.json").read_text()).items() if key != "video"} == {key: value for key, value in shipped.items() if key != "video"}
    doc = re.sub(r"\s+", " ", RESULTS_PATH.read_text())
    for point in mine["reference_points"]:
        assert f"| `{point['name']}` | {point['pixel'][0]:.0f}, {point['pixel'][1]:.0f} | {point['error_px']:.1f} |" in doc
    assert f"RMS {mine['rms_reference_error_px']:.1f} px" in doc and mine["corner_pixels_source"] == "projected from the reference points"
    near_right = np.round(mine["corner_pixels"][1]).astype(int)
    assert f"near-right one at ({near_right[0]}, {near_right[1]})" in doc
    names, pixels, meters = reference_points(mine)
    leave_one_out = []
    for left_out in range(len(names)):
        keep = [index for index in range(len(names)) if index != left_out]
        homography = homography_from_court_points(pixels[keep], [names[index] for index in keep])
        leave_one_out.append(np.linalg.norm(court_to_pixels(homography, [meters[left_out]])[0] - pixels[left_out]))
    assert f"within {min(leave_one_out):.0f} to {max(leave_one_out):.0f} px" in doc
    world = camera_reference_meters(meters)
    all_pixels = np.vstack([pixels, STAND_IN_NET_POSTS])
    camera = fit_camera(world, all_pixels, (mine["frame_width"], mine["frame_height"]))
    shipped_camera = json.loads(STAND_IN_CAMERA.read_text())
    assert camera.focal_px == pytest.approx(shipped_camera["focal_px"], rel=1e-6) and np.allclose(camera.rotation, shipped_camera["rotation"], atol=1e-6)
    x, y, z = camera.centre
    errors = camera_reprojection_errors(camera, world, all_pixels)
    assert f"x {x:.2f}, y {y:.2f}".replace("-", "−") in doc and f"z {z:.2f} m" in doc
    assert f"focal length {camera.focal_px:.0f} px (a {np.degrees(2 * np.arctan(mine['frame_width'] / 2 / camera.focal_px)):.0f}° horizontal field of view" in doc
    assert f"reference errors {errors.min():.1f} to {errors.max():.1f} px" in doc


def test_court_docs_have_no_task_ids_and_source_has_no_comments():
    for doc in docs_on_disk(("docs/vision/court.md", "docs/results/court.md")):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())
    source = (PROJECT_ROOT / "vision" / "court.py").read_text()
    assert '"""' not in source and "'''" not in source and not re.search(r"(?m)^\s*#", source)
