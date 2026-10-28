import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vision.hits
from vision.court import homography_from_corners
from vision.hits import (
    HIT_SCORE_MIN,
    MIN_HIT_GAP_FRAMES,
    REACH_MAX_BODY_HEIGHTS,
    SPEED_AFTER_MIN_PX_PER_FRAME,
    SPEED_FLOOR_PX_PER_FRAME,
    VELOCITY_WINDOW_FRAMES,
    FrameTracks,
    HitDetectionSettings,
    boxes_by_frame,
    detect_hits,
    distance_to_box,
    evaluation_lines,
    frame_tracks,
    is_at_rest,
    labelled_hits,
    load_frame_tracks,
    main,
    match_hits,
    nearest_player,
    output_paths,
    peaks_with_minimum_gap,
    precision_recall,
    rally_end_frame,
    rally_numbers,
    reach_distances,
    segment_rallies,
    shuttle_rest_frame,
)
from vision.players import BOX_COLUMNS, FrameDetections, boxes_path, player_box_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
MATCH_CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
SAVED_TRACK_FILES = [
    PROJECT_ROOT / "data" / "shuttle" / "match" / "shuttle_14201_14947.csv",
    PROJECT_ROOT / "data" / "players" / "match" / "boxes_14201_14947.csv",
    PROJECT_ROOT / "data" / "players" / "match" / "positions_14201_14947.csv",
    PROJECT_ROOT / "data" / "CoachAI-Projects" / "ShuttleSet" / "set" / "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals" / "set1.csv",
]
MATCH_HOMOGRAPHY = homography_from_corners([[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])
NEAR_BOX = (250.0, 200.0, 350.0, 400.0)
FAR_BOX = (900.0, 50.0, 950.0, 150.0)
AWAY_BOX = (2000.0, 2000.0, 2100.0, 2100.0)
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"

requires_saved_tracks = pytest.mark.skipif(
    not MATCH_VIDEO_PATH.exists() or not MATCH_CALIBRATION_PATH.exists() or not all(path.exists() for path in SAVED_TRACK_FILES),
    reason="needs data/match.mp4, its calibration, ShuttleSet and the saved tracks for frames 14201-14947",
)


def tracks_from(points, valid=None, near_boxes=None, far_boxes=None, first_frame: int = 0) -> FrameTracks:
    point_array = np.asarray(points, dtype=float)
    frame_count = len(point_array)
    return FrameTracks(
        frames=np.arange(first_frame, first_frame + frame_count),
        shuttle_points=point_array,
        shuttle_valid=np.ones(frame_count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        near_boxes=np.tile(NEAR_BOX, (frame_count, 1)) if near_boxes is None else np.asarray(near_boxes, dtype=float),
        far_boxes=np.tile(FAR_BOX, (frame_count, 1)) if far_boxes is None else np.asarray(far_boxes, dtype=float),
    )


def reversal_path(hit_index: int = 20, length: int = 41, speed_after: float = 10.0, y: float = 300.0) -> list[tuple[float, float]]:
    return [(100.0 + 10.0 * t, y) if t <= hit_index else (100.0 + 10.0 * hit_index - speed_after * (t - hit_index), y) for t in range(length)]


def test_settings_defaults_are_the_named_constants():
    settings = HitDetectionSettings()
    assert (settings.velocity_window_frames, settings.speed_floor_px_per_frame, settings.hit_score_min) == (VELOCITY_WINDOW_FRAMES, SPEED_FLOOR_PX_PER_FRAME, HIT_SCORE_MIN)
    assert (settings.speed_after_min_px_per_frame, settings.reach_max_body_heights, settings.min_hit_gap_frames) == (SPEED_AFTER_MIN_PX_PER_FRAME, REACH_MAX_BODY_HEIGHTS, MIN_HIT_GAP_FRAMES)


def test_clean_reversal_inside_the_near_box_is_one_hit_by_the_near_player():
    hits = detect_hits(tracks_from(reversal_path()))
    assert hits["frame"].tolist() == [20] and hits["hitter"].tolist() == ["me"]
    assert hits["reach"].iloc[0] == 0.0 and hits["score"].iloc[0] == pytest.approx(20 / 22, abs=1e-3)


def test_straight_flight_is_no_hit():
    assert detect_hits(tracks_from([(100.0 + 10.0 * t, 300.0) for t in range(41)])).empty


def test_hitter_is_the_player_whose_box_is_closest_and_a_tie_goes_to_the_near_player():
    far_at_shuttle = tracks_from(reversal_path(), near_boxes=np.tile(AWAY_BOX, (41, 1)), far_boxes=np.tile(NEAR_BOX, (41, 1)))
    assert detect_hits(far_at_shuttle)["hitter"].tolist() == ["opp"]
    assert nearest_player(0.0, 0.0) == "me" and nearest_player(0.3, 0.2) == "opp"


def test_slow_motion_after_the_reversal_is_not_a_hit():
    tumbling = detect_hits(tracks_from(reversal_path(speed_after=3.0)))
    assert tumbling.empty
    fast_enough = detect_hits(tracks_from(reversal_path(speed_after=8.0)))
    assert fast_enough["frame"].tolist() == [20]


def test_reversal_far_from_both_players_is_not_a_hit():
    assert detect_hits(tracks_from(reversal_path(), near_boxes=np.tile(AWAY_BOX, (41, 1)), far_boxes=np.tile(AWAY_BOX, (41, 1)))).empty


def test_missing_boxes_cannot_be_the_hitter():
    nan_boxes = np.full((41, 4), np.nan)
    assert detect_hits(tracks_from(reversal_path(), near_boxes=nan_boxes, far_boxes=nan_boxes)).empty
    far_only = detect_hits(tracks_from(reversal_path(), near_boxes=nan_boxes, far_boxes=np.tile(NEAR_BOX, (41, 1))))
    assert far_only["hitter"].tolist() == ["opp"]


def lift_leaving_the_top(length: int = 70) -> tuple[list[tuple[float, float]], list[bool]]:
    points, valid = [], []
    for t in range(length):
        if t < 30:
            points.append((400.0, 300.0 - 10.0 * t))
            valid.append(True)
        elif t <= 40:
            points.append((0.0, 0.0))
            valid.append(False)
        else:
            points.append((400.0, 10.0 * (t - 40)))
            valid.append(True)
    return points, valid


def test_lift_leaving_the_top_of_the_frame_gives_no_hit_and_does_not_end_the_rally():
    points, valid = lift_leaving_the_top()
    tracks = tracks_from(points, valid, near_boxes=np.tile(AWAY_BOX, (70, 1)), far_boxes=np.tile(AWAY_BOX, (70, 1)))
    assert detect_hits(tracks).empty
    assert shuttle_rest_frame(tracks, 0, 69) is None
    assert rally_numbers([5, 60], tracks) == [1, 1]


def test_lob_apex_is_not_a_hit():
    apex = [(5.0 * t, 300.0 - 0.1 * (t - 40) ** 2) for t in range(81)]
    boxes = np.tile((0.0, 0.0, 1280.0, 720.0), (81, 1))
    assert detect_hits(tracks_from(apex, near_boxes=boxes, far_boxes=boxes)).empty


def test_smoothly_filled_gap_is_no_hit_but_a_sharp_kink_in_the_fill_is():
    straight = [(100.0 + 10.0 * t, 300.0) for t in range(41)]
    boxes = np.tile((0.0, 0.0, 1280.0, 720.0), (41, 1))
    assert detect_hits(tracks_from(straight, near_boxes=boxes, far_boxes=boxes)).empty
    kinked = [(100.0 + 10.0 * t, 300.0) if t <= 20 else (300.0, 300.0 + 10.0 * (t - 20)) for t in range(41)]
    assert detect_hits(tracks_from(kinked, near_boxes=boxes, far_boxes=boxes))["frame"].tolist() == [20]


def test_jumping_player_box_still_contains_the_racket():
    rising_boxes = np.array([(250.0, 200.0 - 5.0 * t, 350.0, 400.0 - 5.0 * t) for t in range(41)])
    high_hit = tracks_from(reversal_path(y=150.0), near_boxes=rising_boxes, far_boxes=np.tile(AWAY_BOX, (41, 1)))
    hits = detect_hits(high_hit)
    assert hits["hitter"].tolist() == ["me"] and hits["reach"].iloc[0] == 0.0


def test_two_candidates_within_the_minimum_gap_keep_the_stronger():
    scores = np.zeros(40)
    scores[[10, 12, 30]] = [0.5, 0.9, 0.6]
    assert peaks_with_minimum_gap([10, 12, 30], scores, 8) == [12, 30]
    assert peaks_with_minimum_gap([], scores, 8) == []


def test_distance_to_box_and_reach_in_body_heights():
    assert distance_to_box((300, 300), NEAR_BOX) == 0.0
    assert distance_to_box((350, 500), NEAR_BOX) == 100.0
    assert distance_to_box((200, 150), NEAR_BOX) == pytest.approx(np.hypot(50, 50))
    reaches = reach_distances([(300, 300), (300, 500), (300, 300), (300, 300)], [NEAR_BOX, NEAR_BOX, (0, 0, 10, 0), (np.nan, np.nan, np.nan, np.nan)])
    assert reaches.tolist() == [0.0, 0.5, np.inf, np.inf]


def test_is_at_rest_needs_every_point_valid_and_within_the_radius():
    still = [(400.0 + (t % 2), 300.0) for t in range(20)]
    assert is_at_rest(still, [True] * 20)
    assert not is_at_rest(still, [True] * 19 + [False])
    assert not is_at_rest([(400.0 + t, 300.0) for t in range(20)], [True] * 20)
    assert not is_at_rest([], [])


def resting_after_flight(length: int = 70) -> list[tuple[float, float]]:
    return [(100.0 + 10.0 * min(t, 30), 300.0) if t < 50 else (400.0 + 10.0 * (t - 49), 300.0) for t in range(length)]


def test_rest_frame_is_the_first_frame_of_a_still_window_and_the_rally_end():
    tracks = tracks_from(resting_after_flight())
    assert shuttle_rest_frame(tracks, 0, 69) == 30
    assert shuttle_rest_frame(tracks, 29, 69) == 30
    assert shuttle_rest_frame(tracks, 30, 69) is None
    assert rally_end_frame(10, 100, tracks) == 30
    assert rally_end_frame(60, 100, tracks) == 69


def test_rallies_split_on_a_rest_or_a_long_gap_and_end_at_the_rest_frame():
    tracks = tracks_from(resting_after_flight())
    assert rally_numbers([10, 20, 60], tracks) == [1, 1, 2]
    assert rally_numbers([10, 20], tracks_from([(100.0 + 10.0 * t, 300.0) for t in range(300)]), rally_gap_frames=120) == [1, 1]
    assert rally_numbers([10, 200], tracks_from([(100.0 + 10.0 * t, 300.0) for t in range(300)]), rally_gap_frames=120) == [1, 2]
    hits = pd.DataFrame({"frame": [10, 20, 60], "hitter": ["me", "opp", "me"], "score": [0.9, 0.8, 0.7], "reach": [0.0, 0.1, 0.0]})
    numbered, rallies = segment_rallies(hits, tracks)
    assert numbered["rally"].tolist() == [1, 1, 2] and list(numbered.columns) == ["rally", "frame", "hitter", "score", "reach"]
    assert rallies.to_dict("records") == [
        {"rally": 1, "start_frame": 10, "end_frame": 30, "hits": 2},
        {"rally": 2, "start_frame": 60, "end_frame": 69, "hits": 1},
    ]


def test_segment_rallies_with_no_hits_gives_empty_tables():
    numbered, rallies = segment_rallies(pd.DataFrame(columns=["frame", "hitter", "score", "reach"]), tracks_from(resting_after_flight()))
    assert numbered.empty and rallies.empty and list(rallies.columns) == ["rally", "start_frame", "end_frame", "hits"]


def test_matching_is_one_to_one_closest_first_and_never_double_counts():
    assert match_hits([10, 12], [11], 5) == [(10, 11)]
    assert match_hits([10], [9, 11], 5) == [(10, 9)]
    assert match_hits([10, 20], [12, 13], 5) == [(10, 12)]
    assert match_hits([10, 20], [12, 13], 7) == [(10, 12), (20, 13)]
    assert match_hits([100], [106], 5) == [] and match_hits([100], [105], 5) == [(100, 105)]
    assert match_hits([], [1, 2], 5) == []
    assert precision_recall(1, 2, 2) == (0.5, 0.5) and precision_recall(0, 0, 0) == (0.0, 0.0)


def test_evaluation_lines_count_hitters_only_on_labelled_hitters():
    hits = pd.DataFrame({"frame": [9, 20, 40], "hitter": ["me", "opp", "me"]})
    labels = pd.DataFrame({"frame": [10, 21, 30], "hitter": ["me", "me", None]})
    lines = evaluation_lines(hits, labels)
    assert lines[0].endswith("matched within ±5 frames: 2") and lines[1] == "precision 66.7%, recall 66.7%"
    assert lines[2] == "hitter correct: 1/2 = 50.0%" and "median -1.0" in lines[3]


def test_labelled_hits_map_player_position_to_the_near_or_far_player():
    shots = pd.DataFrame(
        {
            "rally": [1, 1, 1, 2, 2],
            "frame_num": [100.0, 120.0, 140.0, 500.0, np.nan],
            "player_location_x": [640.0, 640.0, np.nan, 640.0, 640.0],
            "player_location_y": [600.0, 320.0, np.nan, 600.0, 600.0],
            "type": ["smash", "clear", "short service", "lob", "lob"],
        }
    )
    labels = labelled_hits(shots, MATCH_HOMOGRAPHY, 90, 200)
    assert labels["frame"].tolist() == [100, 120, 140]
    assert labels["hitter"].tolist()[:2] == ["me", "opp"] and pd.isna(labels["hitter"].iloc[2])


def test_matched_label_without_a_hitter_position_is_not_judged():
    shots = pd.DataFrame({"rally": [1, 1], "frame_num": [100.0, 120.0], "player_location_x": [640.0, np.nan], "player_location_y": [600.0, np.nan], "type": ["smash", "short service"]})
    labels = labelled_hits(shots, MATCH_HOMOGRAPHY, 0, 200)
    hits = pd.DataFrame({"frame": [99, 119], "hitter": ["me", "me"]})
    assert evaluation_lines(hits, labels)[2] == "hitter correct: 1/1 = 100.0%"


def test_frame_tracks_masks_cutaway_frames_and_missing_boxes():
    shuttle = pd.DataFrame({"frame": [2, 0, 1], "x": [12, 10, 11], "y": [22, 20, 21], "visibility": [1, 1, 1], "inpainted": [0, 0, 0]})
    boxes = pd.DataFrame({"frame": [0, 2, 0], "player": ["me", "me", "opp"], "left": [1.0, 2.0, 3.0], "top": [1.0, 2.0, 3.0], "right": [5.0, 6.0, 7.0], "bottom": [9.0, 9.0, 9.0]})
    tracks = frame_tracks(shuttle, boxes, [0, 2])
    assert tracks.frames.tolist() == [0, 1, 2] and tracks.shuttle_points[1].tolist() == [11.0, 21.0]
    assert tracks.shuttle_valid.tolist() == [True, False, True]
    assert np.isnan(tracks.near_boxes[1]).all() and tracks.near_boxes[2].tolist() == [2.0, 2.0, 6.0, 9.0]
    assert np.isnan(tracks.far_boxes[1:]).all() and tracks.far_boxes[0].tolist() == [3.0, 3.0, 7.0, 9.0]
    assert np.isnan(boxes_by_frame(boxes, np.array([5]), "me")).all()


def test_player_box_rows_follow_the_labelled_players_and_round_to_a_tenth():
    detections = FrameDetections(np.array([[1.04, 2.0, 3.0, 4.0], [5.0, 6.06, 7.0, 8.0]]), np.array([7, 9]), np.array([0.9, 0.8]))
    rows = player_box_rows(detections, [{"frame": 5, "player": "opp", "track_id": 9}, {"frame": 5, "player": "me", "track_id": 7}])
    assert rows == [
        {"frame": 5, "player": "opp", "left": 5.0, "top": 6.1, "right": 7.0, "bottom": 8.0},
        {"frame": 5, "player": "me", "left": 1.0, "top": 2.0, "right": 3.0, "bottom": 4.0},
    ]
    assert list(rows[0]) == BOX_COLUMNS
    assert player_box_rows(detections, []) == []


def test_output_paths_follow_the_video_stem():
    assert boxes_path(Path("videos/game.mp4"), 5, 9).as_posix() == "data/players/game/boxes_5_9.csv"
    hits_file, rallies_file = output_paths(Path("videos/game.mp4"), 5, 9)
    assert (hits_file.as_posix(), rallies_file.as_posix()) == ("data/hits/game/hits_5_9.csv", "data/hits/game/rallies_5_9.csv")


def test_main_missing_track_file_exits_one_with_the_command_to_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["clip.mp4", "--start", "1", "--end", "2"]) == 1
    error = capsys.readouterr().err
    assert "track file not found" in error and "python -m vision.shuttle clip.mp4 --start 1 --end 2" in error
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("arguments", [[], ["clip.mp4"], ["clip.mp4", "--start", "x", "--end", "2"], ["clip.mp4", "--start", "1", "--end", "2", "--evaluate", "M"]])
def test_main_invalid_arguments_exit_two(arguments):
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


@requires_saved_tracks
def test_saved_tracks_of_rally_four_reproduce_the_documented_numbers():
    from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots
    from vision.court import calibration_path, load_calibration

    tracks = load_frame_tracks(MATCH_VIDEO_PATH, 14201, 14947)
    hits, rallies = segment_rallies(detect_hits(tracks), tracks)
    homography = load_calibration(calibration_path(MATCH_VIDEO_PATH))["pixel_to_court_homography"]
    labels = labelled_hits(load_set_shots(SHUTTLESET_SET_DIR, "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals", 1), homography, 14201, 14947)
    matched = match_hits(hits["frame"].tolist(), labels["frame"].tolist())
    assert (len(labels), len(hits), len(matched)) == (26, 25, 24)
    assert evaluation_lines(hits, labels)[2] == "hitter correct: 24/24 = 100.0%"
    assert np.median([detected - labelled for detected, labelled in matched]) == -1
    assert len(rallies) == 1 and rallies["hits"].iloc[0] == 25
    saved_hits = pd.read_csv(PROJECT_ROOT / "data" / "hits" / "match" / "hits_14201_14947.csv")
    assert saved_hits["frame"].tolist() == hits["frame"].tolist() and saved_hits["hitter"].tolist() == hits["hitter"].tolist()
    errors = vision.hits.error_breakdown(hits, labels, tracks)
    assert list(errors.columns) == ["kind", "frame", "cause", "inpainted"]
    assert errors["kind"].value_counts().to_dict() == {"miss": 2, "false positive": 1}
    assert errors.loc[errors["kind"] == "false positive", "frame"].tolist() == [14593]
    assert errors.loc[errors["frame"] == 14593, "inpainted"].iloc[0]
    assert errors.loc[errors["frame"] == 14231, "cause"].iloc[0].startswith("serve")


def test_hits_source_has_no_comments_or_docstrings():
    source_text = (PROJECT_ROOT / "vision" / "hits.py").read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.mark.parametrize("doc_name", ["vision/hits.md", "results/hits.md", "vision/players.md"])
def test_hit_docs_have_no_task_ids(doc_name):
    assert re.findall(TASK_ID_PATTERN, (PROJECT_ROOT / "docs" / doc_name).read_text()) == []


def test_own_footage_criterion_is_marked_pending_not_claimed():
    results_text = (PROJECT_ROOT / "docs" / "results" / "hits.md").read_text()
    own_footage = results_text.split("## Own footage")[1]
    assert "pending" in own_footage and re.search(r"no\s+recording", own_footage)
