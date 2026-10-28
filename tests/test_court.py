import ast
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import vision.court as court
from vision.alignment import open_video, read_frame
from vision.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    apply_homography,
    as_corner_array,
    calibrate_video,
    calibration_path,
    court_corner_meters,
    court_line_segments,
    court_to_pixels,
    draw_court_lines,
    homography_from_corners,
    line_offsets_to_bright_pixels,
    load_calibration,
    main,
    parse_corner_argument,
    pixels_to_court,
    reprojection_errors,
    save_calibration,
    shuttleset_corner_pixels,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COURT_SOURCE_PATH = PROJECT_ROOT / "vision" / "court.py"
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
SAVED_CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
REAL_HOMOGRAPHY_CSV = PROJECT_ROOT / "data" / "CoachAI-Projects" / "ShuttleSet" / "set" / "homography.csv"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
SYNTHETIC_CORNERS = [[100.0, 300.0], [400.0, 300.0], [350.0, 100.0], [150.0, 100.0]]
METERS_TO_PIXELS_GROUND_TRUTH = np.array(
    [[90.0, 12.0, 320.0], [-4.0, -38.0, 650.0], [0.001, 0.035, 1.0]]
)

requires_shuttleset_csv = pytest.mark.skipif(not REAL_HOMOGRAPHY_CSV.exists(), reason="run ingest/fetch_shuttleset.sh first")
requires_match_video = pytest.mark.skipif(
    not MATCH_VIDEO_PATH.exists() or not REAL_HOMOGRAPHY_CSV.exists(),
    reason="needs data/match.mp4 and ShuttleSet (run ingest/fetch_shuttleset.sh)",
)


def ground_truth_corner_pixels() -> np.ndarray:
    return apply_homography(METERS_TO_PIXELS_GROUND_TRUTH, court_corner_meters())


def court_grid_meters(step: float = 0.5) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(np.arange(0, COURT_WIDTH_M + 1e-9, step), np.arange(0, COURT_LENGTH_M + 1e-9, step))
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def write_blank_video(video_path: Path, width: int, height: int, frame_count: int) -> Path:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))
    for _ in range(frame_count):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    return video_path


def corner_arguments(corners) -> list[str]:
    return [f"{x},{y}" for x, y in corners]


def line_intersection(first_start, first_end, second_start, second_end) -> np.ndarray:
    first_line = np.cross([*first_start, 1.0], [*first_end, 1.0])
    second_line = np.cross([*second_start, 1.0], [*second_end, 1.0])
    point = np.cross(first_line, second_line)
    return point[:2] / point[2]


def test_corners_map_to_court_corners_with_near_left_origin():
    homography = homography_from_corners(SYNTHETIC_CORNERS)
    np.testing.assert_allclose(
        pixels_to_court(homography, SYNTHETIC_CORNERS), [[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], atol=1e-9
    )


def test_axis_directions_x_across_and_y_away_from_camera():
    homography = homography_from_corners(SYNTHETIC_CORNERS)
    near_left_inside_x, near_left_inside_y = pixels_to_court(homography, [[110.0, 295.0]])[0]
    assert 0 < near_left_inside_x < 1 and 0 < near_left_inside_y < 1
    rightward = pixels_to_court(homography, [[260.0, 250.0]])[0] - pixels_to_court(homography, [[240.0, 250.0]])[0]
    upward = pixels_to_court(homography, [[250.0, 240.0]])[0] - pixels_to_court(homography, [[250.0, 260.0]])[0]
    assert rightward[0] > 0 and abs(rightward[1]) < 1e-9
    assert upward[1] > 0 and abs(upward[0]) < 1e-9


def test_round_trip_pixels_to_meters_to_pixels():
    homography = homography_from_corners(SYNTHETIC_CORNERS)
    random_pixels = np.random.default_rng(7).uniform([150, 120], [350, 290], size=(200, 2))
    np.testing.assert_allclose(court_to_pixels(homography, pixels_to_court(homography, random_pixels)), random_pixels, atol=1e-8)
    grid = court_grid_meters()
    np.testing.assert_allclose(pixels_to_court(homography, court_to_pixels(homography, grid)), grid, atol=1e-9)


def test_recovers_known_perspective_for_service_lines_and_centre():
    homography = homography_from_corners(ground_truth_corner_pixels())
    known_points = np.array(
        [[3.05, 6.7], [0.46, 4.72], [5.64, 8.68], [3.05, 0.76], [3.05, 12.64], [0.46, 13.4], [6.1, 6.7]]
    )
    np.testing.assert_allclose(
        court_to_pixels(homography, known_points), apply_homography(METERS_TO_PIXELS_GROUND_TRUTH, known_points), atol=1e-6
    )
    np.testing.assert_allclose(
        pixels_to_court(homography, apply_homography(METERS_TO_PIXELS_GROUND_TRUTH, known_points)), known_points, atol=1e-9
    )


def test_court_centre_is_image_intersection_of_corner_diagonals():
    corners = ground_truth_corner_pixels()
    homography = homography_from_corners(corners)
    diagonal_crossing = line_intersection(corners[0], corners[2], corners[1], corners[3])
    np.testing.assert_allclose(pixels_to_court(homography, diagonal_crossing)[0], [3.05, 6.7], atol=1e-9)


def test_homography_matches_opencv_perspective_transform():
    corners = ground_truth_corner_pixels()
    opencv_homography = cv2.getPerspectiveTransform(corners.astype(np.float32), court_corner_meters().astype(np.float32))
    np.testing.assert_allclose(
        homography_from_corners(corners) / homography_from_corners(corners)[2, 2], opencv_homography / opencv_homography[2, 2], rtol=1e-4, atol=1e-7
    )


def test_reprojection_errors_measure_pixel_distance():
    homography = homography_from_corners(SYNTHETIC_CORNERS)
    np.testing.assert_allclose(
        reprojection_errors(homography, [[103.0, 304.0], [400.0, 300.0]], [[0, 0], [6.1, 0]]), [5.0, 0.0], atol=1e-9
    )


def test_court_lines_have_official_badminton_dimensions():
    segments = court_line_segments()
    assert len(segments) == 13
    assert (COURT_WIDTH_M, COURT_LENGTH_M, NET_Y_M) == (6.1, 13.4, 6.7)
    assert segments["near_baseline"] == ((0.0, 0.0), (6.1, 0.0))
    assert segments["far_baseline"] == ((0.0, 13.4), (6.1, 13.4))
    assert segments["left_doubles_sideline"] == ((0.0, 0.0), (0.0, 13.4))
    assert segments["right_doubles_sideline"] == ((6.1, 0.0), (6.1, 13.4))
    assert segments["left_singles_sideline"][0][0] == pytest.approx(0.46)
    assert segments["right_singles_sideline"][0][0] - segments["left_singles_sideline"][0][0] == pytest.approx(5.18)
    assert segments["near_short_service_line"][0][1] == pytest.approx(4.72)
    assert segments["far_short_service_line"][0][1] == pytest.approx(8.68)
    assert segments["near_doubles_long_service_line"][0][1] == pytest.approx(0.76)
    assert segments["far_doubles_long_service_line"][0][1] == pytest.approx(12.64)
    assert segments["net"] == ((0.0, 6.7), (6.1, 6.7))
    np.testing.assert_allclose(segments["near_centre_line"], [[3.05, 0.0], [3.05, 4.72]])
    np.testing.assert_allclose(segments["far_centre_line"], [[3.05, 8.68], [3.05, 13.4]])
    for start, end in segments.values():
        for x, y in (start, end):
            assert 0 <= x <= 6.1 and 0 <= y <= 13.4


@pytest.mark.parametrize(
    "corners",
    [
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
        [[100, 300], [400, 300], [350, 100], [150, np.nan]],
        [[100, 300], [400, 300], [np.inf, 100], [150, 100]],
    ],
)
def test_bad_corner_shapes_or_values_raise_value_error(corners):
    with pytest.raises(ValueError):
        homography_from_corners(corners)


@pytest.mark.parametrize(
    "corners",
    [
        [[0, 300], [200, 300], [400, 300], [200, 100]],
        [[100, 300], [100, 300], [350, 100], [150, 100]],
        [[100, 300], [400, 300], [400, 300], [150, 100]],
    ],
)
def test_collinear_or_repeated_corners_raise_degenerate(corners):
    with pytest.raises(ValueError, match="degenerate"):
        homography_from_corners(corners)


def test_crossing_edges_raise_convex_error_naming_order():
    with pytest.raises(ValueError, match="near_left, near_right, far_right, far_left"):
        homography_from_corners([[100, 300], [350, 100], [400, 300], [150, 100]])


@pytest.mark.parametrize(
    "corners",
    [
        [[150, 100], [350, 100], [400, 300], [100, 300]],
        [[400, 300], [100, 300], [150, 100], [350, 100]],
        [[350, 100], [150, 100], [100, 300], [400, 300]],
        [[100, 300], [150, 100], [350, 100], [400, 300]],
    ],
)
def test_rotated_mirrored_or_reversed_order_raises_order_error(corners):
    with pytest.raises(ValueError, match="corner order looks wrong|convex"):
        homography_from_corners(corners)


def test_concave_quadrilateral_raises_convex_error():
    with pytest.raises(ValueError, match="convex"):
        homography_from_corners([[100, 300], [400, 300], [250, 250], [150, 100]])


def test_slightly_rotated_camera_is_accepted():
    angle = np.radians(5)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_corners = (np.array(SYNTHETIC_CORNERS) - 250) @ rotation.T + 250
    np.testing.assert_allclose(
        pixels_to_court(homography_from_corners(rotated_corners), rotated_corners), court_corner_meters(), atol=1e-9
    )


@requires_shuttleset_csv
def test_every_shuttleset_match_corners_pass_validation():
    for match_folder in pd.read_csv(REAL_HOMOGRAPHY_CSV)["video"]:
        corners = shuttleset_corner_pixels(match_folder, REAL_HOMOGRAPHY_CSV)
        assert reprojection_errors(homography_from_corners(corners), corners, court_corner_meters()).max() < 1e-6


def test_as_corner_array_returns_float_copy():
    corners = as_corner_array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert corners.dtype == np.float64 and corners.shape == (4, 2)


def test_draw_court_lines_leaves_input_unchanged_and_draws_on_baseline():
    frame = np.zeros((400, 500, 3), dtype=np.uint8)
    drawn_frame = draw_court_lines(frame, homography_from_corners(SYNTHETIC_CORNERS))
    assert frame.max() == 0
    assert drawn_frame[300, 250, 2] > 200


def test_line_offsets_find_painted_line_below_projected_line():
    painted_frame = np.zeros((400, 500), dtype=np.uint8)
    painted_frame[302, :] = 255
    offsets = line_offsets_to_bright_pixels(painted_frame, homography_from_corners(SYNTHETIC_CORNERS), (0, 0), (6.1, 0), sample_count=10)
    assert offsets.tolist() == [2.0] * 10


def test_line_offsets_on_blank_frame_are_empty():
    blank_frame = np.zeros((400, 500, 3), dtype=np.uint8)
    assert len(line_offsets_to_bright_pixels(blank_frame, homography_from_corners(SYNTHETIC_CORNERS), (0, 0), (6.1, 0))) == 0


def test_save_and_load_calibration_round_trip(tmp_path):
    calibration_file = tmp_path / "nested" / "my_match" / "court.json"
    saved = save_calibration(calibration_file, Path("my_match.mp4"), 300, (500, 400), SYNTHETIC_CORNERS)
    on_disk = json.loads(calibration_file.read_text())
    assert on_disk == saved
    assert set(on_disk) == {
        "video",
        "frame_index",
        "frame_width",
        "frame_height",
        "corner_order",
        "corner_pixels",
        "court_corner_meters",
        "pixel_to_court_homography",
        "max_corner_reprojection_error_px",
    }
    assert on_disk["corner_order"] == ["near_left", "near_right", "far_right", "far_left"]
    loaded = load_calibration(calibration_file)
    np.testing.assert_allclose(loaded["pixel_to_court_homography"], homography_from_corners(SYNTHETIC_CORNERS))
    np.testing.assert_allclose(loaded["corner_pixels"], SYNTHETIC_CORNERS)
    assert (loaded["frame_index"], loaded["frame_width"], loaded["frame_height"]) == (300, 500, 400)


def test_save_calibration_with_bad_corners_writes_nothing(tmp_path):
    calibration_file = tmp_path / "bad" / "court.json"
    with pytest.raises(ValueError):
        save_calibration(calibration_file, Path("bad.mp4"), 0, (500, 400), [[100, 300], [350, 100], [400, 300], [150, 100]])
    assert not calibration_file.exists()


def test_load_missing_calibration_names_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        load_calibration(tmp_path / "missing.json")


def test_calibration_path_is_per_video_stem():
    assert calibration_path(Path("videos/game_3.mp4"), Path("out")) == Path("out/game_3/court.json")


def test_shuttleset_corner_pixels_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        shuttleset_corner_pixels("Some_Match", tmp_path / "missing.csv")
    homography_csv = tmp_path / "homography.csv"
    homography_csv.write_text(
        "id,video,upleft_x,upright_x,downleft_x,downright_x,upleft_y,upright_y,downleft_y,downright_y\n"
        "1,Some_Match,150,350,100,400,100,100,300,300\n"
    )
    np.testing.assert_allclose(shuttleset_corner_pixels("Some_Match", homography_csv), SYNTHETIC_CORNERS)
    with pytest.raises(ValueError, match="Other_Match"):
        shuttleset_corner_pixels("Other_Match", homography_csv)


@pytest.mark.parametrize("corner_text", ["1;2", "1,2,3", "a,b", "", "5"])
def test_parse_corner_argument_rejects_bad_text(corner_text):
    with pytest.raises(Exception) as raised:
        parse_corner_argument(corner_text)
    assert type(raised.value).__name__ == "ArgumentTypeError"


def test_main_headless_corners_writes_calibration_and_overlay(tmp_path, capsys):
    video_path = write_blank_video(tmp_path / "tiny.avi", 500, 400, 3)
    output_dir = tmp_path / "calibration"
    assert main([str(video_path), "--frame", "2", "--corners", *corner_arguments(SYNTHETIC_CORNERS), "--output-dir", str(output_dir)]) == 0
    calibration = load_calibration(output_dir / "tiny" / "court.json")
    assert calibration["frame_index"] == 2
    assert (calibration["frame_width"], calibration["frame_height"]) == (500, 400)
    assert (output_dir / "tiny" / "court_overlay.jpg").is_file()
    assert "saved" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra_arguments, expected_message",
    [
        (["--frame", "50"], "could not read frame 50"),
        (["--frame", "-1"], "could not read frame -1"),
    ],
)
def test_main_unreadable_frame_returns_one(tmp_path, capsys, extra_arguments, expected_message):
    video_path = write_blank_video(tmp_path / "tiny.avi", 500, 400, 3)
    arguments = [str(video_path), *extra_arguments, "--corners", *corner_arguments(SYNTHETIC_CORNERS), "--output-dir", str(tmp_path / "out")]
    assert main(arguments) == 1
    assert expected_message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_missing_and_unopenable_video_return_one(tmp_path, capsys):
    assert main([str(tmp_path / "missing.mp4"), "--corners", *corner_arguments(SYNTHETIC_CORNERS), "--output-dir", str(tmp_path / "out")]) == 1
    assert "video not found" in capsys.readouterr().err
    garbage_path = tmp_path / "garbage.mp4"
    garbage_path.write_bytes(b"not a video")
    assert main([str(garbage_path), "--corners", *corner_arguments(SYNTHETIC_CORNERS), "--output-dir", str(tmp_path / "out")]) == 1
    assert "could not open video" in capsys.readouterr().err


@requires_shuttleset_csv
def test_main_unknown_shuttleset_match_returns_one(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    video_path = write_blank_video(tmp_path / "tiny.avi", 1280, 720, 1)
    assert main([str(video_path), "--shuttleset-match", "No_Such_Match", "--output-dir", str(tmp_path / "out")]) == 1
    assert "match not in" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad_arguments",
    [
        ["video.mp4", "--corners", "1,2", "3,4", "5,6"],
        ["video.mp4", "--corners", "1;2", "3,4", "5,6", "7,8"],
        ["video.mp4", "--corners", "1,2", "3,4", "5,6", "7,8", "--shuttleset-match", "M"],
        ["video.mp4", "--frame", "x"],
        [],
    ],
)
def test_main_invalid_arguments_exit_two(bad_arguments, capsys):
    with pytest.raises(SystemExit) as raised:
        main(bad_arguments)
    assert raised.value.code == 2


def test_main_without_corners_uses_click_window_and_cancel_returns_one(tmp_path, monkeypatch, capsys):
    video_path = write_blank_video(tmp_path / "tiny.avi", 500, 400, 3)
    shown_frames = []
    monkeypatch.setattr(court, "click_corners", lambda frame: shown_frames.append(frame.shape) or None)
    assert main([str(video_path), "--output-dir", str(tmp_path / "out")]) == 1
    assert shown_frames == [(400, 500, 3)]
    assert "cancelled: no calibration saved" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_without_corners_saves_clicked_corners(tmp_path, monkeypatch):
    video_path = write_blank_video(tmp_path / "tiny.avi", 500, 400, 3)
    monkeypatch.setattr(court, "click_corners", lambda frame: np.array(SYNTHETIC_CORNERS))
    assert main([str(video_path), "--output-dir", str(tmp_path / "out")]) == 0
    np.testing.assert_allclose(load_calibration(tmp_path / "out" / "tiny" / "court.json")["corner_pixels"], SYNTHETIC_CORNERS)


@requires_shuttleset_csv
def test_shuttleset_corners_are_scaled_to_video_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    video_path = write_blank_video(tmp_path / "half_size.avi", 640, 360, 1)
    assert main([str(video_path), "--shuttleset-match", AXELSEN_GINTING, "--output-dir", str(tmp_path / "out")]) == 0
    saved_corners = load_calibration(tmp_path / "out" / "half_size" / "court.json")["corner_pixels"]
    np.testing.assert_allclose(saved_corners, shuttleset_corner_pixels(AXELSEN_GINTING, REAL_HOMOGRAPHY_CSV) * 0.5)


def test_court_source_has_no_comments_or_docstrings():
    source_text = COURT_SOURCE_PATH.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.fixture(scope="module")
def axelsen_ginting_calibration(tmp_path_factory):
    if not MATCH_VIDEO_PATH.exists() or not REAL_HOMOGRAPHY_CSV.exists():
        pytest.skip("needs data/match.mp4 and ShuttleSet")
    output_dir = tmp_path_factory.mktemp("court_calibration")
    corners = shuttleset_corner_pixels(AXELSEN_GINTING, REAL_HOMOGRAPHY_CSV)
    calibration_file = calibrate_video(MATCH_VIDEO_PATH, 12300, corners, output_dir)
    return load_calibration(calibration_file), corners


@requires_match_video
def test_rerun_matches_saved_calibration_and_shuttleset_corners(axelsen_ginting_calibration):
    calibration, corners = axelsen_ginting_calibration
    np.testing.assert_allclose(calibration["corner_pixels"], [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])
    assert calibration["max_corner_reprojection_error_px"] < 1e-9
    if SAVED_CALIBRATION_PATH.exists():
        saved = json.loads(SAVED_CALIBRATION_PATH.read_text())
        np.testing.assert_allclose(saved["pixel_to_court_homography"], calibration["pixel_to_court_homography"], atol=1e-12)
        np.testing.assert_allclose(saved["corner_pixels"], corners)


@requires_match_video
def test_shuttleset_matrix_maps_to_300_by_660_diagram_and_agrees_with_ours(axelsen_ginting_calibration):
    calibration, corners = axelsen_ginting_calibration
    homography_rows = pd.read_csv(REAL_HOMOGRAPHY_CSV)
    shuttleset_matrix = np.array(json.loads(homography_rows[homography_rows["video"] == AXELSEN_GINTING].iloc[0]["homography_matrix"]))
    np.testing.assert_allclose(
        apply_homography(shuttleset_matrix, corners), [[27.4, 810], [327.6, 810], [327.6, 150], [27.4, 150]], atol=1e-3
    )
    grid = court_grid_meters()
    diagram_points = np.column_stack([27.4 + grid[:, 0] * 300.2 / 6.1, 810 - grid[:, 1] * 660 / 13.4])
    pixel_differences = np.linalg.norm(
        apply_homography(np.linalg.inv(shuttleset_matrix), diagram_points)
        - court_to_pixels(calibration["pixel_to_court_homography"], grid),
        axis=1,
    )
    assert pixel_differences.max() < 1e-3


@requires_match_video
@pytest.mark.parametrize(
    "frame_index, sample_count, median_offset, mean_offset, percentile_95",
    [(12300, 475, 2.0, 2.4, 6.0), (95659, 473, 2.0, 2.3, 5.0), (126000, 473, 2.5, 2.6, 6.0)],
)
def test_line_offset_numbers_in_results_doc(axelsen_ginting_calibration, frame_index, sample_count, median_offset, mean_offset, percentile_95):
    calibration, _ = axelsen_ginting_calibration
    capture = open_video(MATCH_VIDEO_PATH)
    try:
        frame = read_frame(capture, frame_index)
    finally:
        capture.release()
    floor_lines = {name: segment for name, segment in court_line_segments().items() if name != "net"}
    absolute_offsets = np.abs(
        np.concatenate(
            [
                line_offsets_to_bright_pixels(frame, calibration["pixel_to_court_homography"], start, end)
                for start, end in floor_lines.values()
            ]
        )
    )
    assert len(absolute_offsets) == sample_count
    assert np.median(absolute_offsets) == median_offset
    assert round(absolute_offsets.mean(), 1) == mean_offset
    assert np.percentile(absolute_offsets, 95) == percentile_95
    assert np.median(absolute_offsets) <= 3.0


TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b|\bV\d\b|\bM\d\b"


@pytest.mark.parametrize("doc_path", [PROJECT_ROOT / "docs" / "vision" / "court.md", PROJECT_ROOT / "docs" / "results" / "court.md"])
def test_court_docs_have_no_task_ids(doc_path):
    assert re.findall(TASK_ID_PATTERN, doc_path.read_text()) == []


def test_court_doc_states_shuttleset_frame_assumption_and_file_name_limitation():
    doc_text = (PROJECT_ROOT / "docs" / "vision" / "court.md").read_text()
    assert "1280 x 720" in doc_text and "16:9" in doc_text
    assert "a/match.mp4" in doc_text and "overwrites" in doc_text


@requires_shuttleset_csv
@pytest.mark.parametrize("width, height, scale", [(1920, 1080, 1.5), (640, 360, 0.5), (1280, 720, 1.0)])
def test_shuttleset_match_end_to_end_on_other_resolutions(tmp_path, monkeypatch, width, height, scale):
    monkeypatch.chdir(PROJECT_ROOT)
    video_path = write_blank_video(tmp_path / f"video_{width}.avi", width, height, 2)
    assert main([str(video_path), "--frame", "1", "--shuttleset-match", AXELSEN_GINTING, "--output-dir", str(tmp_path / "out")]) == 0
    calibration = load_calibration(tmp_path / "out" / f"video_{width}" / "court.json")
    expected_corners = shuttleset_corner_pixels(AXELSEN_GINTING, REAL_HOMOGRAPHY_CSV) * scale
    np.testing.assert_allclose(calibration["corner_pixels"], expected_corners)
    assert (calibration["frame_width"], calibration["frame_height"]) == (width, height)
    np.testing.assert_allclose(pixels_to_court(calibration["pixel_to_court_homography"], expected_corners), court_corner_meters(), atol=1e-9)
    assert np.all(calibration["corner_pixels"][:, 0] < width) and np.all(calibration["corner_pixels"][:, 1] < height)
    overlay = cv2.imread(str(tmp_path / "out" / f"video_{width}" / "court_overlay.jpg"))
    assert overlay.shape == (height, width, 3)


def test_explicit_corners_are_never_scaled(tmp_path):
    video_path = write_blank_video(tmp_path / "small.avi", 640, 360, 1)
    corners = [[100.0, 300.0], [500.0, 300.0], [420.0, 80.0], [180.0, 80.0]]
    assert main([str(video_path), "--corners", *corner_arguments(corners), "--output-dir", str(tmp_path / "out")]) == 0
    np.testing.assert_allclose(load_calibration(tmp_path / "out" / "small" / "court.json")["corner_pixels"], corners)


def test_label_frame_scaling_uses_width_only_on_non_16_by_9_video(tmp_path):
    video_path = write_blank_video(tmp_path / "four_by_three.avi", 640, 480, 1)
    label_corners = [[200.0, 600.0], [800.0, 600.0], [700.0, 200.0], [300.0, 200.0]]
    calibration_file = calibrate_video(video_path, 0, label_corners, tmp_path / "out", corners_in_shuttleset_label_frame=True)
    np.testing.assert_allclose(load_calibration(calibration_file)["corner_pixels"], np.array(label_corners) * 0.5)


class FakeWindow:
    def __init__(self, keys, visible_after=None, clicks_at_first_key=()):
        self.keys = list(keys)
        self.visible_after = visible_after
        self.clicks_at_first_key = list(clicks_at_first_key)
        self.key_calls = 0
        self.mouse_callback = None
        self.destroyed = False

    def set_mouse_callback(self, window_name, callback):
        self.mouse_callback = callback

    def wait_key(self, delay):
        self.key_calls += 1
        if self.key_calls == 1:
            for x, y in self.clicks_at_first_key:
                self.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return self.keys.pop(0) if self.keys else -1

    def window_property(self, window_name, property_id):
        if self.visible_after is not None and self.key_calls >= self.visible_after:
            return 0.0
        return 1.0

    def destroy_window(self, window_name):
        self.destroyed = True


@pytest.fixture
def install_fake_window(monkeypatch):
    def install(fake_window: FakeWindow) -> FakeWindow:
        monkeypatch.setattr(cv2, "namedWindow", lambda *arguments: None)
        monkeypatch.setattr(cv2, "imshow", lambda *arguments: None)
        monkeypatch.setattr(cv2, "setMouseCallback", fake_window.set_mouse_callback)
        monkeypatch.setattr(cv2, "waitKey", fake_window.wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", fake_window.window_property)
        monkeypatch.setattr(cv2, "destroyWindow", fake_window.destroy_window)
        return fake_window

    return install


def blank_frame() -> np.ndarray:
    return np.zeros((400, 500, 3), dtype=np.uint8)


def test_click_corners_closed_window_cancels_without_destroy(install_fake_window):
    fake_window = install_fake_window(FakeWindow(keys=[-1, -1, -1], visible_after=2))
    assert court.click_corners(blank_frame()) is None
    assert fake_window.key_calls == 2
    assert fake_window.destroyed is False


def test_click_corners_escape_cancels_and_destroys_window(install_fake_window):
    fake_window = install_fake_window(FakeWindow(keys=[27]))
    assert court.click_corners(blank_frame()) is None
    assert fake_window.destroyed is True


def test_click_corners_returns_four_clicks_on_enter(install_fake_window):
    clicks = [(100, 300), (400, 300), (350, 100), (150, 100)]
    fake_window = install_fake_window(FakeWindow(keys=[-1, 13], clicks_at_first_key=clicks))
    np.testing.assert_allclose(court.click_corners(blank_frame()), clicks)
    assert fake_window.destroyed is True


def test_click_corners_enter_with_fewer_than_four_clicks_keeps_waiting(install_fake_window):
    fake_window = install_fake_window(FakeWindow(keys=[13, 13, 27], clicks_at_first_key=[(100, 300), (400, 300)]))
    assert court.click_corners(blank_frame()) is None
    assert fake_window.key_calls == 3


def test_click_corners_ignores_fifth_click_and_undo_removes_last(install_fake_window):
    clicks = [(100, 300), (400, 300), (350, 100), (150, 100), (5, 5)]
    fake_window = install_fake_window(FakeWindow(keys=[ord("u"), -1, 13, 27], clicks_at_first_key=clicks))
    assert court.click_corners(blank_frame()) is None
    assert fake_window.key_calls == 4


def test_window_was_closed_true_when_property_raises(monkeypatch):
    def raise_opencv_error(*arguments):
        raise cv2.error("no window")

    monkeypatch.setattr(cv2, "getWindowProperty", raise_opencv_error)
    assert court.window_was_closed("anything") is True
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *arguments: 1.0)
    assert court.window_was_closed("anything") is False
