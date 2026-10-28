import ast
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import vision.players
from vision.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    court_line_segments,
    court_to_pixels,
    homography_from_corners,
    save_calibration,
)
from vision.players import (
    BYSTANDER_COLOR_BGR,
    DEFAULT_MODEL_PATH,
    PLAYER_COLOR_BGR,
    POSITION_COLUMNS,
    FrameDetections,
    PlayerTrackingState,
    TrackingRun,
    chosen_detection_index,
    court_line_sample_share,
    court_positions_of_feet,
    draw_players,
    empty_detections,
    foot_points,
    id_continuity,
    is_main_view,
    label_players,
    load_person_tracker,
    main,
    on_court_mask,
    positions_path,
    rally_frame_range,
    save_positions,
    share_in_own_half,
    side_of_court,
    sides_with_new_tracks,
    split_frames_by_view,
    summary_lines,
    track_people,
    track_players,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYERS_SOURCE_PATH = PROJECT_ROOT / "vision" / "players.py"
PLAYERS_DOC_PATH = PROJECT_ROOT / "docs" / "vision" / "players.md"
PLAYERS_RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "players.md"
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
MATCH_CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
MODEL_PATH = PROJECT_ROOT / DEFAULT_MODEL_PATH
MATCH_CORNERS = [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]]
MATCH_HOMOGRAPHY = homography_from_corners(MATCH_CORNERS)
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
SHUTTLESET_SET_DIR_RELATIVE = vision.players.SHUTTLESET_SET_DIR

requires_yolo_and_match_video = pytest.mark.skipif(
    not MODEL_PATH.exists() or not MATCH_VIDEO_PATH.exists() or not MATCH_CALIBRATION_PATH.exists(),
    reason="needs data/models/yolo11s.pt, data/match.mp4 and its court calibration",
)


def boxes_with_feet_at(court_points) -> np.ndarray:
    feet_pixels = court_to_pixels(MATCH_HOMOGRAPHY, np.asarray(court_points, dtype=float).reshape(-1, 2))
    return np.array([[foot_x - 20, foot_y - 120, foot_x + 20, foot_y] for foot_x, foot_y in feet_pixels])


def detections_at(court_points, track_ids, confidences) -> tuple[FrameDetections, np.ndarray]:
    boxes = boxes_with_feet_at(court_points)
    detections = FrameDetections(boxes, np.array(track_ids, dtype=int), np.array(confidences, dtype=float))
    return detections, court_positions_of_feet(MATCH_HOMOGRAPHY, boxes)


def court_drawing() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)
    for start, end in court_line_segments().values():
        start_pixel, end_pixel = np.round(court_to_pixels(MATCH_HOMOGRAPHY, [start, end])).astype(int)
        cv2.line(frame, tuple(map(int, start_pixel)), tuple(map(int, end_pixel)), (255, 255, 255), 3)
    return frame


def write_view_video(video_path: Path, frame_is_main_view: list[bool]) -> Path:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (FRAME_WIDTH, FRAME_HEIGHT))
    main_view_frame = court_drawing()
    cutaway_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)
    for is_main in frame_is_main_view:
        writer.write(main_view_frame if is_main else cutaway_frame)
    writer.release()
    return video_path


@pytest.fixture
def view_video_and_calibration(tmp_path) -> tuple[Path, Path, list[bool]]:
    frame_is_main_view = [False, False, True, True, True, False, True, True]
    video_path = write_view_video(tmp_path / "views.avi", frame_is_main_view)
    calibration_file = tmp_path / "court.json"
    save_calibration(calibration_file, video_path, 2, (FRAME_WIDTH, FRAME_HEIGHT), MATCH_CORNERS)
    return video_path, calibration_file, frame_is_main_view


def test_foot_points_are_bottom_centre_for_list_single_box_and_empty():
    np.testing.assert_allclose(foot_points([[10, 20, 30, 80]]), [[20.0, 80.0]])
    np.testing.assert_allclose(foot_points([10, 20, 31, 80]), [[20.5, 80.0]])
    assert foot_points(np.zeros((0, 4))).shape == (0, 2)


def test_court_positions_of_feet_maps_court_corners_and_handles_no_boxes():
    corner_meters = [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]]
    np.testing.assert_allclose(court_positions_of_feet(MATCH_HOMOGRAPHY, boxes_with_feet_at(corner_meters)), corner_meters, atol=1e-9)
    assert court_positions_of_feet(MATCH_HOMOGRAPHY, []).shape == (0, 2)


@pytest.mark.parametrize(
    "court_point, expected_on_court",
    [
        ((-1.0, 3.0), True),
        ((-1.01, 3.0), False),
        ((COURT_WIDTH_M + 1.0, 3.0), True),
        ((COURT_WIDTH_M + 1.01, 3.0), False),
        ((3.0, -1.0), True),
        ((3.0, -1.01), False),
        ((3.0, COURT_LENGTH_M + 2.0), True),
        ((3.0, COURT_LENGTH_M + 2.01), False),
        ((-1.5, COURT_LENGTH_M + 1.5), False),
        ((3.0, NET_Y_M), True),
    ],
)
def test_on_court_mask_boundaries_are_inclusive_and_far_margin_only_behind_far_baseline(court_point, expected_on_court):
    assert on_court_mask([court_point]).tolist() == [expected_on_court]


def test_on_court_mask_empty_and_custom_margins():
    assert on_court_mask(np.zeros((0, 2))).tolist() == []
    assert on_court_mask([[-0.5, 3.0], [3.0, 14.0]], margin_m=0.0, far_margin_m=0.0).tolist() == [False, False]


@pytest.mark.parametrize("y_m, expected_side", [(0.0, "me"), (NET_Y_M - 1e-9, "me"), (NET_Y_M, "opp"), (13.4, "opp"), (-0.5, "me")])
def test_side_of_court_near_half_is_me_and_net_line_is_far(y_m, expected_side):
    assert side_of_court(y_m) == expected_side


@pytest.mark.parametrize("share, expected", [(0.6, True), (0.5999, False), (1.0, True), (0.0, False)])
def test_is_main_view_threshold_is_inclusive(share, expected):
    assert is_main_view(share) is expected


def test_court_line_sample_share_high_on_drawn_court_and_low_on_blank_or_shifted_court():
    drawn_court = court_drawing()
    assert court_line_sample_share(drawn_court, MATCH_HOMOGRAPHY) > 0.9
    assert court_line_sample_share(cv2.cvtColor(drawn_court, cv2.COLOR_BGR2GRAY), MATCH_HOMOGRAPHY) > 0.9
    assert court_line_sample_share(np.full_like(drawn_court, 90), MATCH_HOMOGRAPHY) == 0.0
    shifted_court = np.roll(drawn_court, 60, axis=0)
    assert court_line_sample_share(shifted_court, MATCH_HOMOGRAPHY) < 0.6


def test_sides_with_new_tracks_never_changes_a_known_side_and_does_not_mutate_input():
    known_sides = {1: "opp"}
    updated = sides_with_new_tracks(known_sides, [1, 2, 3], [[3.0, 2.0], [3.0, 12.0], [3.0, NET_Y_M]])
    assert updated == {1: "opp", 2: "opp", 3: "opp"}
    assert known_sides == {1: "opp"}
    assert sides_with_new_tracks({}, [], np.zeros((0, 2))) == {}


def test_chosen_detection_index_prefers_previous_track_over_higher_confidence():
    track_ids, confidences = np.array([4, 5, 6]), np.array([0.3, 0.95, 0.5])
    assert chosen_detection_index([0, 1, 2], track_ids, confidences, previous_track_id=4) == 0
    assert chosen_detection_index([0, 2], track_ids, confidences, previous_track_id=5) == 2
    assert chosen_detection_index([], track_ids, confidences, previous_track_id=None) is None


def test_label_players_near_is_me_far_is_opp_and_rows_follow_position_columns():
    detections, positions = detections_at([[3.0, 2.0], [3.0, 11.0]], [7, 8], [0.8, 0.9])
    rows, state = label_players(100, detections, positions, PlayerTrackingState())
    assert [(row["player"], row["track_id"]) for row in rows] == [("me", 7), ("opp", 8)]
    assert all(list(row) == POSITION_COLUMNS for row in rows)
    assert rows[0]["x_m"] == pytest.approx(3.0, abs=1e-3) and rows[0]["y_m"] == pytest.approx(2.0, abs=1e-3)
    assert state == PlayerTrackingState({7: "me", 8: "opp"}, {"me": 7, "opp": 8})


def test_label_players_labels_by_side_not_by_detection_order():
    detections, positions = detections_at([[3.0, 11.0], [3.0, 2.0]], [8, 7], [0.9, 0.8])
    rows, _ = label_players(100, detections, positions, PlayerTrackingState())
    assert [(row["player"], row["track_id"]) for row in rows] == [("me", 7), ("opp", 8)]


def test_label_players_drops_confident_bystanders_just_outside_the_margins():
    detections, positions = detections_at(
        [[3.0, 2.0], [3.0, 11.0], [-1.2, 4.0], [7.3, 9.0], [3.0, -1.3], [3.0, 15.7]],
        [1, 2, 3, 4, 5, 6],
        [0.5, 0.5, 0.99, 0.99, 0.99, 0.99],
    )
    rows, state = label_players(100, detections, positions, PlayerTrackingState())
    assert [row["track_id"] for row in rows] == [1, 2]
    assert set(state.side_by_track_id) == {1, 2}


def test_label_players_line_judge_inside_the_margin_is_not_chosen_over_the_player_track():
    player_only, player_positions = detections_at([[3.0, 2.0]], [1], [0.5])
    _, state = label_players(100, player_only, player_positions, PlayerTrackingState())
    detections, positions = detections_at([[3.0, 2.0], [-0.8, 3.0]], [1, 9], [0.5, 0.99])
    rows, _ = label_players(101, detections, positions, state)
    assert [(row["player"], row["track_id"]) for row in rows] == [("me", 1)]


def test_label_players_with_two_new_near_tracks_picks_most_confident():
    detections, positions = detections_at([[2.0, 2.0], [4.0, 3.0]], [3, 4], [0.6, 0.7])
    rows, _ = label_players(100, detections, positions, PlayerTrackingState())
    assert [(row["player"], row["track_id"]) for row in rows] == [("me", 4)]


def test_label_players_track_keeps_its_side_after_crossing_the_net():
    detections, positions = detections_at([[3.0, 2.0], [3.0, 11.0]], [1, 2], [0.9, 0.9])
    _, state = label_players(100, detections, positions, PlayerTrackingState())
    crossed, crossed_positions = detections_at([[3.0, 9.0], [3.0, 4.0]], [1, 2], [0.9, 0.9])
    rows, _ = label_players(101, crossed, crossed_positions, state)
    assert [(row["player"], row["track_id"]) for row in rows] == [("me", 1), ("opp", 2)]
    assert share_in_own_half(pd.DataFrame(rows), "me") == 0.0


def test_label_players_empty_frame_and_one_sided_frame():
    rows, state = label_players(100, empty_detections(), np.zeros((0, 2)), PlayerTrackingState())
    assert rows == [] and state == PlayerTrackingState()
    detections, positions = detections_at([[3.0, 11.0]], [2], [0.9])
    rows, _ = label_players(101, detections, positions, PlayerTrackingState())
    assert [row["player"] for row in rows] == ["opp"]


def test_label_players_does_not_mutate_the_input_state():
    state = PlayerTrackingState({1: "me"}, {"me": 1})
    detections, positions = detections_at([[3.0, 2.0], [3.0, 11.0]], [1, 2], [0.9, 0.9])
    label_players(100, detections, positions, state)
    assert state == PlayerTrackingState({1: "me"}, {"me": 1})


def positions_table(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"frame": frame, "player": player, "x_m": 3.0, "y_m": y_m, "track_id": track_id, "confidence": 0.9} for frame, player, y_m, track_id in rows],
        columns=POSITION_COLUMNS,
    )


def test_id_continuity_counts_missing_frames_and_other_ids_as_failures_and_ignores_frames_outside():
    table = positions_table([(1, "me", 2.0, 5), (2, "me", 2.0, 5), (3, "me", 2.0, 6), (9, "me", 2.0, 6), (9, "me", 2.0, 6)])
    continuity = id_continuity(table, "me", [1, 2, 3, 4])
    assert (continuity["frames"], continuity["frames_detected"], continuity["track_ids"], continuity["modal_track_id"]) == (4, 3, 2, 5)
    assert continuity["share_with_one_id"] == 0.5


def test_id_continuity_with_no_frames_or_no_rows():
    assert id_continuity(positions_table([]), "opp", [])["share_with_one_id"] == 0.0
    no_rows = id_continuity(positions_table([]), "opp", [1, 2])
    assert (no_rows["modal_track_id"], no_rows["frames_with_modal_id"], no_rows["share_with_one_id"]) == (None, 0, 0.0)


def test_share_in_own_half_net_line_belongs_to_far_half():
    table = positions_table([(1, "me", NET_Y_M, 1), (2, "me", 2.0, 1), (1, "opp", NET_Y_M, 2), (2, "opp", 3.0, 2)])
    assert share_in_own_half(table, "me") == 0.5
    assert share_in_own_half(table, "opp") == 0.5
    assert share_in_own_half(positions_table([]), "me") == 0.0


def test_rally_frame_range_ignores_missing_frames_and_raises_for_unknown_rally():
    shots = pd.DataFrame({"rally": [3, 3, 3, 4], "frame_num": [np.nan, 500.0, 440.0, 900.0]})
    assert rally_frame_range(shots, 3) == (440, 500)
    with pytest.raises(ValueError, match="rally 5"):
        rally_frame_range(shots, 5)
    with pytest.raises(ValueError):
        rally_frame_range(pd.DataFrame({"rally": [6], "frame_num": [np.nan]}), 6)


def test_draw_players_colours_players_and_greys_bystanders_without_changing_input():
    detections, positions = detections_at([[3.0, 2.0], [3.0, 11.0], [-2.0, 5.0]], [1, 2, 3], [0.9, 0.9, 0.9])
    rows, _ = label_players(100, detections, positions, PlayerTrackingState())
    blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    drawn = draw_players(blank, detections, rows)
    feet = np.round(foot_points(detections.boxes_xyxy)).astype(int)
    assert drawn[feet[0][1], feet[0][0]].tolist() == list(PLAYER_COLOR_BGR["me"])
    assert drawn[feet[1][1], feet[1][0]].tolist() == list(PLAYER_COLOR_BGR["opp"])
    assert drawn[feet[2][1], feet[2][0]].tolist() == list(BYSTANDER_COLOR_BGR)
    assert int(blank.sum()) == 0


def test_positions_path_and_save_positions_round_trip(tmp_path):
    assert positions_path(Path("videos/game_2.mp4"), 5, 9, tmp_path) == tmp_path / "game_2" / "positions_5_9.csv"
    table = positions_table([(1, "me", 2.0, 5)])
    saved_path = save_positions(table, tmp_path / "nested" / "positions.csv")
    pd.testing.assert_frame_equal(pd.read_csv(saved_path), table)


def test_summary_lines_report_view_counts_and_player_continuity():
    run = TrackingRun(positions_table([(1, "me", 2.0, 5), (1, "opp", 10.0, 6)]), [1, 2], [3], [])
    lines = summary_lines(run, [1, 2])
    assert lines[0] == "frames: 3, main view: 2, skipped (not main view): 1"
    assert lines[1].startswith("me: detected 1/2 frames, 1 track ids, one id (#5) 1/2 = 50.0%")


def test_load_person_tracker_missing_weights_raises_and_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    with pytest.raises(FileNotFoundError, match="model weights not found"):
        load_person_tracker(Path("data/models/yolo11s.pt"))
    assert list(tmp_path.rglob("*")) == []


def test_split_frames_by_view_separates_cutaways(view_video_and_calibration):
    video_path, calibration_file, frame_is_main_view = view_video_and_calibration
    main_view_frames, skipped_frames = split_frames_by_view(video_path, 0, len(frame_is_main_view) - 1, calibration_file)
    assert main_view_frames == [index for index, is_main in enumerate(frame_is_main_view) if is_main]
    assert skipped_frames == [index for index, is_main in enumerate(frame_is_main_view) if not is_main]


def test_split_frames_by_view_stops_at_end_of_video(view_video_and_calibration):
    video_path, calibration_file, _ = view_video_and_calibration
    main_view_frames, skipped_frames = split_frames_by_view(video_path, 6, 50, calibration_file)
    assert (main_view_frames, skipped_frames) == ([6, 7], [])


def test_split_frames_by_view_negative_start_does_not_mislabel_frames(view_video_and_calibration):
    video_path, calibration_file, frame_is_main_view = view_video_and_calibration
    try:
        main_view_frames, skipped_frames = split_frames_by_view(video_path, -2, 3, calibration_file)
    except ValueError:
        return
    assert main_view_frames == [2, 3]
    assert skipped_frames == [0, 1]


def test_reversed_range_and_missing_video_are_reported_before_calibration(tmp_path):
    with pytest.raises(ValueError, match="before start frame"):
        split_frames_by_view(tmp_path / "missing.mp4", 10, 5)
    with pytest.raises(ValueError, match="before start frame"):
        track_players(tmp_path / "missing.mp4", 10, 5)
    with pytest.raises(FileNotFoundError, match="video not found"):
        split_frames_by_view(tmp_path / "missing.mp4", 0, 5, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="video not found"):
        track_players(tmp_path / "missing.mp4", 0, 5, tmp_path / "missing.json")


def test_track_players_missing_calibration_or_weights_raises(view_video_and_calibration, tmp_path):
    video_path, calibration_file, _ = view_video_and_calibration
    with pytest.raises(FileNotFoundError, match="calibration not found"):
        track_players(video_path, 0, 3, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="model weights not found"):
        track_players(video_path, 0, 3, calibration_file, model_path=tmp_path / "missing.pt")


def test_main_views_only_prints_counts_and_returns_zero(view_video_and_calibration, capsys):
    video_path, calibration_file, _ = view_video_and_calibration
    exit_code = main([str(video_path), "--start", "0", "--end", "7", "--calibration", str(calibration_file), "--views-only"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "frames: 8, main view: 5, skipped (not main view): 3"


def test_main_range_past_end_of_video_returns_one_as_documented(view_video_and_calibration, capsys):
    video_path, calibration_file, _ = view_video_and_calibration
    exit_code = main([str(video_path), "--start", "100", "--end", "105", "--calibration", str(calibration_file), "--views-only"])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["missing.mp4", "--start", "0", "--end", "5"], "video not found"),
        (["missing.mp4"], "give --start and --end"),
        (["missing.mp4", "--start", "9", "--end", "5"], "before start frame"),
        (["missing.mp4", "--shuttleset-rally", "No_Such_Match", "1", "1"], "set CSV not found"),
    ],
)
def test_main_errors_print_message_and_return_one(arguments, message, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(arguments) == 1
    assert message in capsys.readouterr().err
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("arguments", [[], ["video.mp4", "--start", "x"], ["video.mp4", "--shuttleset-rally", "M", "1"]])
def test_main_invalid_arguments_exit_two(arguments):
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


def test_players_source_has_no_comments_or_docstrings():
    source_text = PLAYERS_SOURCE_PATH.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


def touches_yolo(function_node: ast.FunctionDef) -> bool:
    calls_track = any(isinstance(inner, ast.Attribute) and inner.attr == "track" for inner in ast.walk(function_node))
    imports_ultralytics = any(isinstance(inner, ast.ImportFrom) and inner.module == "ultralytics" for inner in ast.walk(function_node))
    return calls_track or imports_ultralytics


def test_yolo_is_only_called_inside_the_thin_wrappers():
    module = ast.parse(PLAYERS_SOURCE_PATH.read_text())
    functions_touching_yolo = {
        node.name for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and touches_yolo(node)
    }
    assert functions_touching_yolo == {"load_person_tracker", "track_people"}
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and "ultralytics" in ast.dump(node) for node in module.body)


@pytest.mark.parametrize("doc_path", [PLAYERS_DOC_PATH, PLAYERS_RESULTS_PATH])
def test_players_docs_have_no_task_ids(doc_path):
    assert re.findall(TASK_ID_PATTERN, doc_path.read_text()) == []


def test_no_model_weights_outside_data_models():
    weight_files = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.suffix in {".pt", ".pth", ".onnx"} and ".venv" not in path.parts and path.parent != PROJECT_ROOT / "data" / "models"
    ]
    assert weight_files == []
    tracked_files = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
    assert [name for name in tracked_files if name.endswith((".pt", ".pth", ".onnx"))] == []


@pytest.fixture(scope="module")
def numpy_print_options_restored():
    print_options_before = np.get_printoptions()
    yield
    np.set_printoptions(**print_options_before)


@pytest.fixture(scope="module")
def match_tracker(numpy_print_options_restored):
    if not MODEL_PATH.exists() or not MATCH_VIDEO_PATH.exists() or not MATCH_CALIBRATION_PATH.exists():
        pytest.skip("needs data/models/yolo11s.pt, data/match.mp4 and its court calibration")
    return load_person_tracker(MODEL_PATH)


def match_frame(frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(MATCH_VIDEO_PATH))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = capture.read()
    capture.release()
    return frame


@requires_yolo_and_match_video
def test_yolo_frame_has_one_near_and_one_far_player_and_drops_bystanders(match_tracker):
    detections = track_people(match_tracker, match_frame(14700))
    positions = court_positions_of_feet(MATCH_HOMOGRAPHY, detections.boxes_xyxy)
    rows, _ = label_players(14700, detections, positions, PlayerTrackingState())
    assert len(detections.track_ids) > 2
    assert [row["player"] for row in rows] == ["me", "opp"]
    assert rows[0]["y_m"] < NET_Y_M <= rows[1]["y_m"]
    assert int(on_court_mask(positions).sum()) == 2


@requires_yolo_and_match_video
def test_track_players_short_run_keeps_both_ids(tmp_path, numpy_print_options_restored):
    run = track_players(MATCH_VIDEO_PATH, 14700, 14707, MATCH_CALIBRATION_PATH, MODEL_PATH, (14703,), tmp_path)
    assert run.main_view_frames == list(range(14700, 14708)) and run.skipped_frames == []
    for player in ("me", "opp"):
        assert id_continuity(run.positions, player, run.main_view_frames)["share_with_one_id"] == 1.0
        assert share_in_own_half(run.positions, player) == 1.0
    assert [path.name for path in run.overlay_paths] == ["match_players_frame14703.jpg"]
    assert run.overlay_paths[0].parent == tmp_path


@pytest.mark.parametrize(
    "start_frame, end_frame, message",
    [(-1, 5, "start frame must be >= 0"), (-5, -2, "start frame must be >= 0"), (6, 5, "before start frame")],
)
def test_check_frame_range_rejects_negative_start_and_reversed_range(start_frame, end_frame, message):
    with pytest.raises(ValueError, match=message):
        vision.players.check_frame_range(start_frame, end_frame)
    vision.players.check_frame_range(0, 0)


def test_frames_in_range_numbers_frames_by_their_video_index(view_video_and_calibration):
    video_path, _, frame_is_main_view = view_video_and_calibration
    capture = cv2.VideoCapture(str(video_path))
    try:
        read_frames = list(vision.players.frames_in_range(capture, 5, 50))
    finally:
        capture.release()
    assert [frame_index for frame_index, _ in read_frames] == [5, 6, 7]
    drawn_court_share = [court_line_sample_share(frame, MATCH_HOMOGRAPHY) > 0.6 for _, frame in read_frames]
    assert drawn_court_share == frame_is_main_view[5:8]


def test_frames_in_range_past_end_of_video_raises(view_video_and_calibration):
    video_path, _, _ = view_video_and_calibration
    capture = cv2.VideoCapture(str(video_path))
    try:
        with pytest.raises(ValueError, match="no frames could be read from 8 to 10"):
            list(vision.players.frames_in_range(capture, 8, 10))
    finally:
        capture.release()


def test_main_negative_start_returns_one(view_video_and_calibration, capsys):
    video_path, calibration_file, _ = view_video_and_calibration
    exit_code = main([str(video_path), "--start", "-3", "--end", "4", "--calibration", str(calibration_file), "--views-only"])
    assert exit_code == 1
    assert "start frame must be >= 0" in capsys.readouterr().err


def test_shuttleset_rally_padding_is_clamped_at_frame_zero(tmp_path, monkeypatch):
    set_dir = tmp_path / SHUTTLESET_SET_DIR_RELATIVE / "Early_Match"
    set_dir.mkdir(parents=True)
    pd.DataFrame({"rally": [1, 1, 2], "frame_num": [10, 40, 300]}).to_csv(set_dir / "set1.csv", index=False)
    monkeypatch.chdir(tmp_path)
    parsed = vision.players.parse_arguments(["video.mp4", "--shuttleset-rally", "Early_Match", "1", "1"])
    start_frame, end_frame, measured_frames = vision.players.frames_to_track(parsed)
    assert (start_frame, end_frame) == (0, 70)
    assert measured_frames == list(range(10, 41))


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="needs data/models/yolo11s.pt")
def test_main_tracking_past_end_of_video_returns_one_and_writes_no_csv(view_video_and_calibration, tmp_path, monkeypatch, capsys):
    video_path, calibration_file, _ = view_video_and_calibration
    monkeypatch.chdir(tmp_path)
    print_options_before = np.get_printoptions()
    exit_code = main([str(video_path), "--start", "100", "--end", "105", "--calibration", str(calibration_file), "--model", str(MODEL_PATH)])
    assert exit_code == 1
    assert "no frames could be read" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
    assert np.get_printoptions() == print_options_before


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="needs data/models/yolo11s.pt")
def test_load_person_tracker_leaves_numpy_print_options_unchanged():
    print_options_before = np.get_printoptions()
    load_person_tracker(MODEL_PATH)
    assert np.get_printoptions() == print_options_before
