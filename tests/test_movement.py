import ast
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest

import vision.movement
from vision.hits import HIT_COLUMNS, RALLY_COLUMNS, output_paths
from vision.movement import (
    FEATURE_NAMES,
    HIT_MODEL_PARAMETERS,
    MODEL_PATH,
    OTHER_FEATURE_NAMES,
    SELF_FEATURE_NAMES,
    SPEED_CAP_M_PER_S,
    HitCandidate,
    LabelledRange,
    MovementSettings,
    PlayerTracks,
    SERVE_TO_RETURN_SECONDS,
    add_totals,
    detected_hits_only,
    inferred_serve,
    inferred_serves_only,
    serve_evaluation_line,
    serve_gap_frames,
    with_inferred_serves,
    alternating_hit_chain,
    both_still_from,
    box_centre,
    box_row_at,
    crop_around,
    cross_validation_lines,
    detect_hits,
    evaluation_summary,
    filled,
    fit_hit_model,
    gap_penalty,
    hit_candidates,
    hit_feature_matrix,
    hit_frame_labels,
    hit_scores,
    load_hit_model,
    load_player_tracks,
    local_maxima,
    main,
    miss_lines,
    motion_features,
    moving_average,
    other_player,
    player_tracks,
    rally_end_frame,
    rally_tables,
    save_hit_model,
    settings_for_video,
    shifted,
    spot_check_path,
    spot_check_sample,
    split_into_rallies,
    stacked_training_data,
    summary_line,
    trim_rally_ends,
    write_spot_check,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOVEMENT_SOURCE_PATH = PROJECT_ROOT / "vision" / "movement.py"
MOVEMENT_DOC_PATH = PROJECT_ROOT / "docs" / "vision" / "movement.md"
MOVEMENT_RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "hits_movement.md"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
STAND_IN_VIDEO_PATH = PROJECT_ROOT / "data" / "recordings" / "stand_in" / "set1.mp4"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
LABELLED_RANGES = [(14201, 14947, 1), (16551, 17343, 1), (81360, 82116, 2), (107560, 108103, 3), (114673, 116641, 3)]


def tracks_exist(video_path: Path, start_frame: int, end_frame: int) -> bool:
    return all((PROJECT_ROOT / path).is_file() for path in (vision.movement.player_positions_path(video_path, start_frame, end_frame), vision.movement.boxes_path(video_path, start_frame, end_frame)))


requires_match_tracks_and_model = pytest.mark.skipif(
    not MATCH_VIDEO_PATH.exists() or not (PROJECT_ROOT / MODEL_PATH).exists() or not all(tracks_exist(Path("data/match.mp4"), start, end) for start, end, _ in LABELLED_RANGES)
    or not (PROJECT_ROOT / "data" / "calibration" / "match" / "court.json").exists() or not (PROJECT_ROOT / vision.movement.SHUTTLESET_SET_DIR / AXELSEN_GINTING).exists(),
    reason="needs data/match.mp4, its calibration, the player tracks of the five labelled ranges, ShuttleSet and data/models/movement_hits.joblib",
)


def straight_tracks(frame_count: int = 60, start_frame: int = 100) -> PlayerTracks:
    frames = np.arange(start_frame, start_frame + frame_count)
    near = np.column_stack([np.full(frame_count, 3.0), np.full(frame_count, 2.0)])
    far = np.column_stack([np.full(frame_count, 3.0), np.full(frame_count, 11.0)])
    near_boxes = np.tile([600.0, 400.0, 700.0, 650.0], (frame_count, 1))
    far_boxes = np.tile([620.0, 250.0, 660.0, 350.0], (frame_count, 1))
    return PlayerTracks(frames, near, far, near_boxes, far_boxes)


def rally_tracks(hit_frames_me, hit_frames_opp, frame_count: int = 400) -> PlayerTracks:
    frames = np.arange(frame_count)
    near_y = np.full(frame_count, 2.0)
    far_y = np.full(frame_count, 11.0)
    for hit in hit_frames_me:
        near_y[max(hit - 8, 0) : hit] = np.linspace(2.0, 4.0, min(8, hit))[-min(8, hit):]
        near_y[hit : hit + 8] = np.linspace(4.0, 2.0, 8)[: frame_count - hit]
    for hit in hit_frames_opp:
        far_y[max(hit - 8, 0) : hit] = np.linspace(11.0, 9.0, min(8, hit))[-min(8, hit):]
        far_y[hit : hit + 8] = np.linspace(9.0, 11.0, 8)[: frame_count - hit]
    near = np.column_stack([np.full(frame_count, 3.0), near_y])
    far = np.column_stack([np.full(frame_count, 3.0), far_y])
    return PlayerTracks(frames, near, far, np.tile([600.0, 400.0, 700.0, 650.0], (frame_count, 1)), np.tile([620.0, 250.0, 660.0, 350.0], (frame_count, 1)))


class ScoreByFrameModel:
    def __init__(self, score_by_row, far_score_by_row=None):
        self.score_by_row = np.asarray(score_by_row, dtype=float)
        self.far_score_by_row = self.score_by_row if far_score_by_row is None else np.asarray(far_score_by_row, dtype=float)
        self.calls = 0

    def predict_proba(self, feature_matrix):
        self.calls += 1
        own_y = feature_matrix[:, SELF_FEATURE_NAMES.index("y_m")]
        is_far_player = len(own_y) > 0 and np.nanmean(own_y) > 6.7
        scores = (self.far_score_by_row if is_far_player else self.score_by_row)[: len(feature_matrix)]
        return np.column_stack([1 - scores, scores])


def test_feature_names_are_the_fourteen_own_and_six_opponent_features():
    assert len(SELF_FEATURE_NAMES) == 14 and len(OTHER_FEATURE_NAMES) == 6 and len(FEATURE_NAMES) == 20
    assert FEATURE_NAMES[:14] == tuple(f"self_{name}" for name in SELF_FEATURE_NAMES) and FEATURE_NAMES[14:] == tuple(f"other_{name}" for name in OTHER_FEATURE_NAMES)
    assert set(OTHER_FEATURE_NAMES) <= set(SELF_FEATURE_NAMES)
    assert HIT_MODEL_PARAMETERS["random_state"] == 0 and HIT_MODEL_PARAMETERS["class_weight"] == "balanced"


def test_settings_scale_with_fps_and_width_and_stay_at_the_defaults_for_the_reference_video():
    assert settings_for_video(30.0, 1280) == MovementSettings()
    scaled = settings_for_video(60.0, 1920)
    for name in ("position_smoothing_frames", "speed_smoothing_frames", "direction_smoothing_frames", "context_frames", "score_smoothing_frames", "interpolation_limit_frames", "label_tolerance_frames", "min_hit_gap_frames", "max_hit_gap_frames", "still_frames"):
        assert getattr(scaled, name) == 2 * getattr(MovementSettings(), name), name
    assert scaled.usual_gap_frames == (30, 110) and scaled.fps == 60.0 and scaled.pixel_scale == pytest.approx(1280 / 1920)
    assert scaled.unusual_gap_penalty_per_10_frames == pytest.approx(0.15)
    assert (scaled.hit_score_min, scaled.rally_end_score_min, scaled.still_speed_m_per_s) == (0.5, 0.7, 0.8)
    assert scaled.infer_serves is True and scaled.serve_to_return_seconds == SERVE_TO_RETURN_SECONDS == 0.76
    assert serve_gap_frames(MovementSettings()) == 23 and serve_gap_frames(scaled) == 46 and serve_gap_frames(settings_for_video(25.0, 1280)) == 19
    slow = settings_for_video(15.0, 640)
    assert slow.speed_smoothing_frames >= 1 and slow.pixel_scale == 2.0 and slow.unusual_gap_penalty_per_10_frames == pytest.approx(0.6)


def test_player_tracks_reindex_every_frame_and_ignore_duplicate_rows_and_unknown_players():
    positions = pd.DataFrame({"frame": [10, 10, 11, 10, 12], "player": ["me", "me", "me", "opp", "judge"], "x_m": [3.0, 9.0, 3.1, 2.0, 0.0], "y_m": [2.0, 9.0, 2.2, 10.0, 0.0]})
    boxes = pd.DataFrame({"frame": [10, 11, 10], "player": ["me", "me", "opp"], "left": [600, 610, 620], "top": [300, 300, 200], "right": [700, 710, 660], "bottom": [500, 505, 300]})
    tracks = player_tracks(positions, boxes, 9, 12)
    np.testing.assert_array_equal(tracks.frames, [9, 10, 11, 12])
    np.testing.assert_array_equal(tracks.near_points[1:3], [[3.0, 2.0], [3.1, 2.2]])
    assert np.isnan(tracks.near_points[0]).all() and np.isnan(tracks.near_points[3]).all()
    np.testing.assert_array_equal(tracks.far_points[1], [2.0, 10.0]) and np.isnan(tracks.far_points[2]).all()
    assert tracks.near_boxes.shape == (4, 4) and tracks.far_boxes.shape == (4, 4)
    assert tracks.points_of("me") is tracks.near_points and tracks.boxes_of("opp") is tracks.far_boxes
    empty = player_tracks(positions.iloc[:0], boxes.iloc[:0], 0, 2)
    assert empty.near_points.shape == (3, 2) and np.isnan(empty.near_points).all()


def test_filled_interpolates_short_gaps_and_the_doc_describes_long_gaps_as_held():
    np.testing.assert_allclose(filled([1.0, np.nan, 3.0], 5), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(filled([np.nan, np.nan, 2.0, np.nan], 5), [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(filled([np.nan, np.nan], 3), [0.0, 0.0])
    long_gap = filled([1.0, np.nan, np.nan, np.nan, np.nan, 5.0], interpolation_limit=1)
    assert long_gap[0] == 1.0 and long_gap[1] == pytest.approx(1.8) and long_gap[-1] == 5.0
    assert np.all(np.isfinite(long_gap))


def test_moving_average_and_shifted_keep_length_and_hold_the_ends():
    np.testing.assert_allclose(moving_average([0, 0, 3, 0, 0], 3), [0, 1, 1, 1, 0])
    np.testing.assert_allclose(moving_average([2, 4, 6, 8], 2), [3, 5, 7, 8])
    np.testing.assert_array_equal(moving_average([1, 2, 3], 1), [1, 2, 3])
    assert moving_average([], 3).shape == (0,)
    np.testing.assert_array_equal(shifted(np.array([1, 2, 3, 4]), 2), [3, 4, 4, 4])
    np.testing.assert_array_equal(shifted(np.array([1, 2, 3, 4]), -3), [1, 1, 1, 1])
    np.testing.assert_array_equal(shifted(np.array([1, 2, 3]), 10), [3, 3, 3])
    np.testing.assert_array_equal(shifted(np.array([[1, 1], [2, 2], [3, 3]]), 1), [[2, 2], [3, 3], [3, 3]])
    assert shifted(np.array([]), 2).shape == (0,)


def test_motion_features_measure_speed_turns_box_changes_and_detection():
    frame_count = 60
    y = np.concatenate([np.linspace(1.0, 3.0, 30), np.linspace(3.0, 1.0, 30)])
    points = np.column_stack([np.full(frame_count, 3.0), y])
    boxes = np.tile([600.0, 300.0, 700.0, 500.0], (frame_count, 1))
    boxes[40:, 1] -= 60.0
    points[50:53] = np.nan
    features = motion_features(points, boxes, MovementSettings())
    assert sorted(features) == sorted(SELF_FEATURE_NAMES) and all(len(values) == frame_count for values in features.values())
    assert features["speed"][15] == pytest.approx(2.0 / 29 * 30, rel=0.05)
    assert features["direction_change"][30] < 0 < features["direction_change"][15]
    assert features["y_move"][15] > 0 > features["y_move"][45]
    assert features["box_height_rise"][40:45].max() > 0 and features["box_height_rise"][:30].max() == 0
    np.testing.assert_array_equal(features["detected"][50:53], 0.0)
    assert features["detected"].sum() == frame_count - 3
    assert np.isfinite(np.column_stack(list(features.values()))).all()
    jump = np.column_stack([np.full(10, 3.0), np.concatenate([np.zeros(5), np.full(5, 100.0)])])
    assert motion_features(jump, np.tile([0.0, 0.0, 10.0, 10.0], (10, 1)), MovementSettings()).get("speed").max() == SPEED_CAP_M_PER_S


def test_motion_features_scale_box_pixels_and_give_zeros_for_tiny_or_empty_tracks():
    boxes = np.tile([600.0, 300.0, 700.0, 500.0], (20, 1))
    boxes[10:, 1] -= 30.0
    points = np.column_stack([np.full(20, 3.0), np.full(20, 2.0)])
    full = motion_features(points, boxes, MovementSettings(pixel_scale=1.0))["box_height_rise"].max()
    half = motion_features(points, boxes, MovementSettings(pixel_scale=0.5))["box_height_rise"].max()
    assert half == pytest.approx(full / 2) and full > 0
    for short in (np.zeros((0, 2)), np.array([[1.0, 2.0]])):
        tiny = motion_features(short, np.zeros((len(short), 4)), MovementSettings())
        assert all(np.array_equal(values, np.zeros(len(short))) for values in tiny.values())
    all_missing = motion_features(np.full((20, 2), np.nan), np.full((20, 4), np.nan), MovementSettings())
    assert all_missing["speed"].max() == 0 and all_missing["detected"].sum() == 0


def test_hit_feature_matrix_puts_own_then_opponent_features_in_feature_name_order():
    tracks = straight_tracks()
    tracks.far_points[:, 1] = np.concatenate([np.linspace(11.0, 8.0, 30), np.linspace(8.0, 11.0, 30)])
    me_matrix = hit_feature_matrix(tracks, "me", MovementSettings())
    opp_matrix = hit_feature_matrix(tracks, "opp", MovementSettings())
    assert me_matrix.shape == opp_matrix.shape == (60, 20)
    own_speed_of_opp = motion_features(tracks.far_points, tracks.far_boxes, MovementSettings())["speed"]
    np.testing.assert_allclose(opp_matrix[:, 0], own_speed_of_opp)
    np.testing.assert_allclose(me_matrix[:, 14], own_speed_of_opp)
    assert me_matrix[:, 0].max() == 0 and opp_matrix[:, 14].max() == 0
    assert other_player("me") == "opp" and other_player("opp") == "me"
    assert hit_feature_matrix(PlayerTracks(np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 4)), np.zeros((0, 4))), "me", MovementSettings()).shape == (0, 20)


def test_hit_frame_labels_mark_a_window_around_each_hit():
    np.testing.assert_array_equal(hit_frame_labels(np.arange(100, 110), [104, 108], 1), [0, 0, 0, 1, 1, 1, 0, 1, 1, 1])
    np.testing.assert_array_equal(hit_frame_labels(np.arange(5), [], 2), np.zeros(5))
    np.testing.assert_array_equal(hit_frame_labels(np.arange(5), [10], 2), np.zeros(5))
    assert hit_frame_labels(np.arange(0), [1], 2).shape == (0,)


def test_fit_and_score_are_deterministic_and_scores_are_smoothed_probabilities():
    random = np.random.default_rng(0)
    features = random.normal(size=(300, 20))
    labels = (features[:, 0] > 0.5).astype(float)
    first, second = fit_hit_model(features, labels), fit_hit_model(features, labels)
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
    scores = hit_scores(first, features, MovementSettings(score_smoothing_frames=1))
    np.testing.assert_allclose(scores, first.predict_proba(features)[:, 1])
    smoothed = hit_scores(first, features, MovementSettings(score_smoothing_frames=3))
    np.testing.assert_allclose(smoothed, moving_average(first.predict_proba(features)[:, 1], 3))
    assert hit_scores(first, np.zeros((0, 20)), MovementSettings()).shape == (0,)


def test_local_maxima_keeps_the_highest_with_a_minimum_gap_above_the_floor():
    assert local_maxima([0.1, 0.9, 0.8, 0.2, 0.7, 0.1, 0.6], 2, 0.5) == [1, 4, 6]
    assert local_maxima([0.9, 0.95, 0.9], 2, 0.5) == [1]
    assert local_maxima([0.4, 0.4], 1, 0.5) == [] and local_maxima([], 3, 0.1) == []
    assert local_maxima([0.6, 0.6, 0.6], 1, 0.5) == [0, 1, 2]


def test_hit_candidates_come_from_both_players_in_frame_order():
    frames = np.arange(200, 210)
    scores = {"me": np.array([0, 0.9, 0, 0, 0, 0, 0, 0, 0.7, 0.0]), "opp": np.array([0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0.0])}
    candidates = hit_candidates(frames, scores, MovementSettings(min_hit_gap_frames=2))
    assert candidates == [HitCandidate(201, "me", 0.9), HitCandidate(205, "opp", 0.8), HitCandidate(208, "me", 0.7)]
    assert hit_candidates(frames, {"me": np.zeros(10), "opp": np.zeros(10)}, MovementSettings()) == []


def test_gap_penalty_is_zero_in_the_usual_window_and_grows_outside():
    settings = MovementSettings(usual_gap_frames=(15, 55), unusual_gap_penalty_per_10_frames=0.3)
    assert gap_penalty(15, settings) == gap_penalty(30, settings) == gap_penalty(55, settings) == 0.0
    assert gap_penalty(5, settings) == pytest.approx(0.3) and gap_penalty(65, settings) == pytest.approx(0.3) and gap_penalty(75, settings) == pytest.approx(0.6)


def test_alternating_hit_chain_alternates_hitters_respects_the_minimum_gap_and_starts_new_rallies():
    settings = MovementSettings(hit_score_min=0.5, min_hit_gap_frames=10, max_hit_gap_frames=70, usual_gap_frames=(15, 55))
    candidates = [HitCandidate(100, "me", 0.9), HitCandidate(104, "opp", 0.6), HitCandidate(130, "opp", 0.8), HitCandidate(160, "me", 0.7), HitCandidate(400, "opp", 0.9)]
    chain = alternating_hit_chain(candidates, settings)
    assert [(hit.frame, hit.hitter) for hit in chain] == [(100, "me"), (130, "opp"), (160, "me"), (400, "opp")]
    assert alternating_hit_chain(list(reversed(candidates)), settings) == chain
    assert alternating_hit_chain([], settings) == []
    same_side = [HitCandidate(100, "me", 0.9), HitCandidate(130, "me", 0.95)]
    assert [hit.frame for hit in alternating_hit_chain(same_side, settings)] == [130]
    after_long_gap = [HitCandidate(100, "me", 0.9), HitCandidate(200, "me", 0.9)]
    assert [hit.frame for hit in alternating_hit_chain(after_long_gap, settings)] == [100, 200]
    weak = [HitCandidate(100, "me", 0.3), HitCandidate(130, "opp", 0.4)]
    assert len(alternating_hit_chain(weak, settings)) == 1
    for hits in (chain,):
        for earlier, later in zip(hits, hits[1:]):
            assert later.frame - earlier.frame > settings.max_hit_gap_frames or (earlier.hitter != later.hitter and later.frame - earlier.frame >= settings.min_hit_gap_frames)


def test_alternating_hit_chain_prefers_two_usual_gaps_over_one_unusual_gap_only_when_the_middle_hit_is_strong_enough():
    settings = MovementSettings(hit_score_min=0.5, min_hit_gap_frames=10, max_hit_gap_frames=70, usual_gap_frames=(15, 55), unusual_gap_penalty_per_10_frames=0.3)
    ends = [HitCandidate(100, "me", 0.9), HitCandidate(165, "me", 0.9)]
    assert [hit.frame for hit in alternating_hit_chain(ends + [HitCandidate(132, "opp", 0.9)], settings)] == [100, 132, 165]
    assert [hit.frame for hit in alternating_hit_chain(ends + [HitCandidate(132, "opp", 0.45)], settings)] == [100, 132, 165]
    assert [hit.frame for hit in alternating_hit_chain(ends + [HitCandidate(132, "opp", 0.1)], settings)] == [100]
    far_apart = [HitCandidate(100, "me", 0.9), HitCandidate(180, "me", 0.9)]
    assert [hit.frame for hit in alternating_hit_chain(far_apart + [HitCandidate(140, "opp", 0.1)], settings)] == [100, 180]
    assert [hit.frame for hit in alternating_hit_chain(far_apart + [HitCandidate(140, "opp", 0.95)], settings)] == [100, 140, 180]


def test_rallies_are_split_at_long_gaps_and_trimmed_of_weak_ends():
    chain = [HitCandidate(100, "me", 0.6), HitCandidate(130, "opp", 0.9), HitCandidate(160, "me", 0.95), HitCandidate(190, "opp", 0.65), HitCandidate(400, "me", 0.9)]
    rallies = split_into_rallies(chain, 70)
    assert [[hit.frame for hit in rally] for rally in rallies] == [[100, 130, 160, 190], [400]]
    assert [hit.frame for hit in trim_rally_ends(rallies[0], 0.7)] == [130, 160]
    assert trim_rally_ends(rallies[0], 0.99) == [] and trim_rally_ends([], 0.5) == []
    assert split_into_rallies([], 70) == []


def test_rally_end_is_the_first_still_window_else_the_gap_after_the_last_hit():
    frames = np.arange(300)
    settings = MovementSettings(still_frames=15, still_speed_m_per_s=0.8, max_hit_gap_frames=70)
    near = np.where(frames < 120, 3.0, 0.1)
    far = np.where(frames < 110, 2.0, 0.2)
    assert both_still_from(120, near, far, settings) and not both_still_from(115, near, far, settings) and not both_still_from(290, near, far, settings)
    assert rally_end_frame(100, 1000, frames, near, far, settings) == 120
    assert rally_end_frame(100, 1000, frames, np.full(300, 3.0), far, settings) == 170
    assert rally_end_frame(100, 150, frames, np.full(300, 3.0), far, settings) == 149
    assert rally_end_frame(280, 1000, frames, np.full(300, 3.0), np.full(300, 3.0), settings) == 299
    assert rally_end_frame(100, 1000, frames + 1000, np.full(300, 3.0), far, settings) == 170


def test_rally_tables_number_rallies_and_use_the_output_columns():
    tracks = straight_tracks(frame_count=300, start_frame=0)
    rallies = [[HitCandidate(10, "me", 0.91234), HitCandidate(40, "opp", 0.8)], [HitCandidate(200, "opp", 0.7), HitCandidate(230, "me", 0.75), HitCandidate(260, "opp", 0.9)]]
    hits, table = rally_tables(rallies, tracks, MovementSettings(infer_serves=False))
    assert list(hits.columns) == HIT_COLUMNS and list(table.columns) == RALLY_COLUMNS
    assert hits["rally"].tolist() == [1, 1, 2, 2, 2] and hits["frame"].tolist() == [10, 40, 200, 230, 260]
    assert hits["hitter"].tolist() == ["me", "opp", "opp", "me", "opp"] and hits["score"].tolist()[0] == 0.912
    assert hits["reach"].isna().all()
    assert table.to_dict("records") == [{"rally": 1, "start_frame": 10, "end_frame": 41, "hits": 2}, {"rally": 2, "start_frame": 200, "end_frame": 261, "hits": 3}]
    empty_hits, empty_table = rally_tables([], tracks, MovementSettings())
    assert empty_hits.empty and list(empty_hits.columns) == HIT_COLUMNS and empty_table.empty and list(empty_table.columns) == RALLY_COLUMNS


def test_rally_tables_prepend_an_inferred_serve_for_the_other_player_when_it_fits_in_the_range():
    tracks = straight_tracks(frame_count=300, start_frame=0)
    rallies = [[HitCandidate(10, "me", 0.9), HitCandidate(40, "opp", 0.8)], [HitCandidate(200, "opp", 0.7), HitCandidate(230, "me", 0.75), HitCandidate(260, "opp", 0.9)]]
    hits, table = rally_tables(rallies, tracks, MovementSettings())
    assert hits["rally"].tolist() == [1, 1, 2, 2, 2, 2] and hits["frame"].tolist() == [10, 40, 177, 200, 230, 260]
    assert hits["hitter"].tolist() == ["me", "opp", "me", "opp", "me", "opp"]
    assert hits["score"].isna().tolist() == [False, False, True, False, False, False] and hits["reach"].isna().all()
    assert table.to_dict("records") == [{"rally": 1, "start_frame": 10, "end_frame": 41, "hits": 2}, {"rally": 2, "start_frame": 177, "end_frame": 261, "hits": 4}]
    assert detected_hits_only(hits)["frame"].tolist() == [10, 40, 200, 230, 260] and inferred_serves_only(hits)["frame"].tolist() == [177]
    later_start = straight_tracks(frame_count=300, start_frame=190)
    hits, _ = rally_tables([rallies[1]], later_start, MovementSettings())
    assert hits["frame"].tolist() == [200, 230, 260]
    hits, _ = rally_tables([rallies[1]], straight_tracks(frame_count=300, start_frame=177), MovementSettings())
    assert hits["frame"].tolist() == [177, 200, 230, 260]


def test_inferred_serve_helpers():
    rally = [HitCandidate(100, "me", 0.9), HitCandidate(130, "opp", 0.8)]
    serve = inferred_serve(rally, 0, MovementSettings())
    assert (serve.frame, serve.hitter) == (77, "opp") and np.isnan(serve.score)
    assert inferred_serve(rally, 78, MovementSettings()) is None and inferred_serve(rally, 77, MovementSettings()) is not None
    assert inferred_serve(rally, 0, settings_for_video(60.0, 1920)).frame == 54
    assert inferred_serve([HitCandidate(100, "opp", 0.9)], 0, MovementSettings()).hitter == "me"
    served = with_inferred_serves([rally, [HitCandidate(400, "opp", 0.9), HitCandidate(430, "me", 0.7)]], 0, MovementSettings())
    assert [[hit.frame for hit in rally_hits] for rally_hits in served] == [[77, 100, 130], [377, 400, 430]]
    assert with_inferred_serves([rally], 0, MovementSettings(infer_serves=False)) == [rally]
    assert with_inferred_serves([], 0, MovementSettings()) == []
    table = pd.DataFrame({"rally": [1, 1], "frame": [77, 100], "hitter": ["opp", "me"], "score": [np.nan, 0.9], "reach": np.nan})
    assert detected_hits_only(table)["frame"].tolist() == [100] and inferred_serves_only(table)["frame"].tolist() == [77]
    without_score = table.drop(columns=["score"])
    assert len(detected_hits_only(without_score)) == 2 and inferred_serves_only(without_score).empty


def test_evaluation_ignores_serve_rows_and_the_serve_line_scores_them_against_the_first_shots():
    hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [77, 100, 130], "hitter": ["opp", "me", "opp"], "score": [np.nan, 0.9, 0.8], "reach": np.nan})
    labels = pd.DataFrame({"rally": [1, 1, 1], "frame": [80, 102, 131], "hitter": ["opp", "me", "opp"], "type": ["serve", "net", "lift"]})
    assert evaluation_summary(hits, labels, 5) == {"labelled": 3, "detected": 2, "matched": 2, "hitter_correct": 2, "hitter_judged": 2}
    assert miss_lines(hits, labels, 5) == ["miss at 80: serve (first shot of the rally)"]
    assert serve_evaluation_line(hits, labels, 5) == "inferred serves: 1, labelled first shots: 1, within ±5 frames: 1, server right: 1/1, offset mean -3.0"
    assert serve_evaluation_line(hits, labels.assign(hitter=[None, "me", "opp"]), 2) == "inferred serves: 1, labelled first shots: 1, within ±2 frames: 0, server right: 0/0"
    assert serve_evaluation_line(detected_hits_only(hits), labels, 5).startswith("inferred serves: 0, labelled first shots: 1, within ±5 frames: 0")


def test_detect_hits_turns_alternating_score_peaks_into_hits_and_rallies_without_touching_files(monkeypatch):
    tracks = rally_tracks([100, 160, 220], [130, 190])
    near_scores, far_scores = np.zeros(400), np.zeros(400)
    for frame in (100, 160, 220):
        near_scores[frame] = 0.95
    for frame in (130, 190):
        far_scores[frame] = 0.9
    near_scores[300] = 0.85
    far_scores[330] = 0.55
    model = ScoreByFrameModel(near_scores, far_scores)
    monkeypatch.setattr(vision.movement, "open_video", lambda *arguments: (_ for _ in ()).throw(AssertionError("video opened")))
    hits, rallies = detect_hits(tracks, model, MovementSettings(score_smoothing_frames=1, infer_serves=False))
    assert model.calls == 2
    assert hits["frame"].tolist() == [100, 130, 160, 190, 220] and hits["rally"].tolist() == [1] * 5
    assert hits["hitter"].tolist() == ["me", "opp", "me", "opp", "me"]
    assert len(rallies) == 1 and rallies.loc[0, "hits"] == 5 and rallies.loc[0, "start_frame"] == 100 and 220 < rallies.loc[0, "end_frame"] <= 290
    assert hits["reach"].isna().all() and list(hits.columns) == HIT_COLUMNS
    served, served_rallies = detect_hits(tracks, model, MovementSettings(score_smoothing_frames=1))
    assert served["frame"].tolist() == [77, 100, 130, 160, 190, 220] and served["hitter"].tolist() == ["opp", "me", "opp", "me", "opp", "me"]
    assert served["score"].isna().tolist() == [True, False, False, False, False, False] and served["rally"].tolist() == [1] * 6
    assert served_rallies.loc[0, "start_frame"] == 77 and served_rallies.loc[0, "hits"] == 6 and served_rallies.loc[0, "end_frame"] == rallies.loc[0, "end_frame"]


def test_detect_hits_with_no_peaks_or_a_lone_hit_gives_empty_tables():
    tracks = straight_tracks(frame_count=200, start_frame=0)
    hits, rallies = detect_hits(tracks, ScoreByFrameModel(np.zeros(200)), MovementSettings())
    assert hits.empty and rallies.empty and list(hits.columns) == HIT_COLUMNS and list(rallies.columns) == RALLY_COLUMNS
    lone = np.zeros(200)
    lone[50] = 0.9
    hits, rallies = detect_hits(tracks, ScoreByFrameModel(lone), MovementSettings(score_smoothing_frames=1))
    assert hits.empty and rallies.empty
    empty = PlayerTracks(np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 4)), np.zeros((0, 4)))
    hits, rallies = detect_hits(empty, ScoreByFrameModel(np.zeros(0)), MovementSettings())
    assert hits.empty and rallies.empty


def test_model_file_round_trip_and_refusals(tmp_path):
    model_path = save_hit_model("a model", tmp_path / "models" / "hits.joblib", trained_on=["match 1-2 set 1"])
    assert model_path.is_file() and load_hit_model(model_path) == "a model"
    stored = joblib.load(model_path)
    assert stored["feature_names"] == list(FEATURE_NAMES) and stored["trained_on"] == ["match 1-2 set 1"]
    assert joblib.load(save_hit_model("m", tmp_path / "plain.joblib"))["trained_on"] == []
    with pytest.raises(FileNotFoundError, match="python -m vision.movement --train"):
        load_hit_model(tmp_path / "missing.joblib")
    joblib.dump({"model": "m", "feature_names": list(FEATURE_NAMES)[:-1]}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="trained on other features"):
        load_hit_model(tmp_path / "old.joblib")
    joblib.dump({"model": "m"}, tmp_path / "nameless.joblib")
    with pytest.raises(ValueError):
        load_hit_model(tmp_path / "nameless.joblib")


def test_load_player_tracks_names_the_players_command_when_a_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(vision.movement, "player_positions_path", lambda video, start, end: tmp_path / "positions.csv")
    monkeypatch.setattr(vision.movement, "boxes_path", lambda video, start, end: tmp_path / "boxes.csv")
    with pytest.raises(FileNotFoundError, match=r"python -m vision.players club.mp4 --start 0 --end 9"):
        load_player_tracks(Path("club.mp4"), 0, 9)
    pd.DataFrame({"frame": [3], "player": ["me"], "x_m": [3.0], "y_m": [2.0], "track_id": [1], "confidence": [0.9]}).to_csv(tmp_path / "positions.csv", index=False)
    with pytest.raises(FileNotFoundError, match="boxes"):
        load_player_tracks(Path("club.mp4"), 0, 9)
    pd.DataFrame({"frame": [3], "player": ["me"], "left": [1], "top": [2], "right": [3], "bottom": [4]}).to_csv(tmp_path / "boxes.csv", index=False)
    tracks = load_player_tracks(Path("club.mp4"), 0, 9)
    assert len(tracks.frames) == 10 and tracks.near_points[3].tolist() == [3.0, 2.0] and np.isnan(tracks.far_points).all()


def test_stacked_training_data_concatenates_both_players_of_every_range():
    first = (None, None, {"me": np.ones((3, 20)), "opp": np.full((3, 20), 2.0)}, {"me": np.array([1.0, 0, 0]), "opp": np.zeros(3)})
    second = (None, None, {"me": np.full((2, 20), 3.0), "opp": np.full((2, 20), 4.0)}, {"me": np.zeros(2), "opp": np.array([0.0, 1.0])})
    features, targets = stacked_training_data([first, second])
    assert features.shape == (10, 20) and features[:, 0].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    assert targets.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_evaluation_summary_matches_one_to_one_and_judges_hitters_only_with_a_labelled_side():
    detected = pd.DataFrame({"frame": [100, 130, 200], "hitter": ["me", "opp", "me"], "score": [0.9, 0.8, 0.7]})
    labels = pd.DataFrame({"rally": [1, 1, 1], "frame": [102, 131, 160], "hitter": ["me", "me", "opp"], "type": ["serve", "net", "lift"]})
    summary = evaluation_summary(detected, labels, 5)
    assert summary == {"labelled": 3, "detected": 3, "matched": 2, "hitter_correct": 1, "hitter_judged": 2}
    unlabelled_side = labels.assign(hitter=[None, "me", "opp"])
    assert evaluation_summary(detected, unlabelled_side, 5)["hitter_judged"] == 1
    assert evaluation_summary(detected.iloc[:0], labels, 5) == {"labelled": 3, "detected": 0, "matched": 0, "hitter_correct": 0, "hitter_judged": 0}
    totals = add_totals(add_totals({}, summary), summary)
    assert totals["matched"] == 4
    assert summary_line("x", summary, 5) == "x ±5: labelled 3, detected 3, matched 2, precision 66.7%, recall 66.7%, hitter correct 1/2"
    assert "precision 0.0%, recall 0.0%" in summary_line("none", {"labelled": 0, "detected": 0, "matched": 0, "hitter_correct": 0, "hitter_judged": 0}, 5)
    lines = miss_lines(detected, labels, 5)
    assert lines == ["miss at 160: lift", "false positive at 200 (me): outside the labelled rallies"]
    assert miss_lines(pd.DataFrame({"frame": [103, 150], "hitter": ["me", "opp"], "score": [0.9, 0.9]}), labels, 5) == ["miss at 131: net", "miss at 160: lift", "false positive at 150 (opp): within a labelled rally"]
    assert miss_lines(detected.iloc[:0], labels.iloc[:1], 5) == ["miss at 102: serve (first shot of the rally)"]


def test_cross_validation_holds_out_each_range_and_never_loads_the_shipped_model(monkeypatch):
    random = np.random.default_rng(1)
    ranges = [LabelledRange(Path("club.mp4"), 100 * index, 100 * index + 99, 1, "folder") for index in range(3)]
    trained_on = []

    def fake_range_data(labelled_range, settings):
        tracks = straight_tracks(100, labelled_range.start_frame)
        labels = pd.DataFrame({"rally": [1, 1], "frame": [labelled_range.start_frame + 20, labelled_range.start_frame + 50], "hitter": ["me", "opp"], "type": ["clear", "lift"]})
        features = {player: random.normal(size=(100, 20)) + labelled_range.start_frame + 50 for player in ("me", "opp")}
        targets = {player: hit_frame_labels(tracks.frames, labels.loc[labels["hitter"] == player, "frame"], 2) for player in ("me", "opp")}
        return tracks, labels, features, targets

    def fake_fit(features, labels):
        trained_on.append(sorted({int(round(value)) // 100 * 100 for value in features[:, 0]}))
        return ScoreByFrameModel(np.zeros(100))

    monkeypatch.setattr(vision.movement, "labelled_range_data", fake_range_data)
    monkeypatch.setattr(vision.movement, "fit_hit_model", fake_fit)
    monkeypatch.setattr(vision.movement, "load_hit_model", lambda *arguments: (_ for _ in ()).throw(AssertionError("shipped model loaded")))
    lines = cross_validation_lines(ranges, MovementSettings(), tolerances=(5,))
    assert trained_on == [[100, 200], [0, 200], [0, 100]]
    assert len(lines) == 4 and lines[-1].startswith("all ranges, each held out ±5: labelled 6, detected 0")


def test_spot_check_helpers_sample_deterministically_and_crop_inside_the_frame():
    many = pd.DataFrame({"rally": 1, "frame": range(0, 300, 10), "hitter": "me", "score": 0.9, "reach": np.nan})
    sample = spot_check_sample(many, 5)
    assert len(sample) == 5 and sample["frame"].is_monotonic_increasing and sample["frame"].tolist() == spot_check_sample(many, 5)["frame"].tolist()
    assert sample["frame"].tolist() != spot_check_sample(many, 5, seed=2)["frame"].tolist()
    assert len(spot_check_sample(many.iloc[:3], 20)) == 3 and len(spot_check_sample(many, 0)) == 0
    assert spot_check_path(Path("data/recordings/stand_in/set1.mp4"), 0, 3599).as_posix() == "data/checks/hits/stand_in_set1_movement_spot_check_0_3599.jpg"
    boxes = pd.DataFrame({"frame": [10, 10], "player": ["opp", "me"], "left": [620, 100], "top": [200, 400], "right": [660, 200], "bottom": [300, 700]})
    assert box_centre(box_row_at(boxes, "opp", 10)) == (640.0, 250.0) and box_centre(box_row_at(boxes, "me", 10)) == (150.0, 550.0)
    assert box_row_at(boxes, "me", 11) is None and box_centre(None) == (0.0, 0.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    for centre in ((640, 360), (0, 0), (1279, 719), (-50, 5000)):
        assert crop_around(frame, centre, "caption").shape == (360, 480, 3)
    assert tuple(crop_around(frame, (0, 0), "")[0, 0]) == (1, 2, 3)
    assert crop_around(np.zeros((100, 100, 3), dtype=np.uint8), (50, 50), "").shape == (360, 480, 3)


def test_write_spot_check_draws_three_tiles_per_sampled_hit(tmp_path):
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(40):
        writer.write(np.full((480, 640, 3), 120, dtype=np.uint8))
    writer.release()
    hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [2, 20, 38], "hitter": ["me", "opp", "me"], "score": [0.9, 0.8, 0.7], "reach": np.nan})
    boxes = pd.DataFrame({"frame": [2, 20], "player": ["me", "opp"], "left": [100, 300], "top": [100, 50], "right": [200, 360], "bottom": [300, 200]})
    sheet = write_spot_check(video_path, hits, boxes, 3, tmp_path / "checks" / "sheet.jpg", 30.0)
    image = cv2.imread(str(sheet))
    assert image is not None and image.shape == (3 * 360, 3 * 480, 3)
    assert image[:360, :480].max() == 0 and image[:360, 480:960].mean() > 50
    assert image[720:, 960:].max() == 0
    empty = write_spot_check(video_path, hits.iloc[:0], boxes, 3, tmp_path / "checks" / "empty.jpg", 30.0)
    assert cv2.imread(str(empty)).shape == (360, 3 * 480, 3)


def fake_track_files(tmp_path, monkeypatch, frame_count=400):
    positions = pd.DataFrame({"frame": list(range(frame_count)) * 2, "player": ["me"] * frame_count + ["opp"] * frame_count, "x_m": 3.0, "y_m": [2.0] * frame_count + [11.0] * frame_count, "track_id": 1, "confidence": 0.9})
    boxes = pd.DataFrame({"frame": list(range(frame_count)) * 2, "player": ["me"] * frame_count + ["opp"] * frame_count, "left": 600, "top": 300, "right": 700, "bottom": 500})
    positions.to_csv(tmp_path / "positions.csv", index=False)
    boxes.to_csv(tmp_path / "boxes.csv", index=False)
    monkeypatch.setattr(vision.movement, "player_positions_path", lambda video, start, end: tmp_path / "positions.csv")
    monkeypatch.setattr(vision.movement, "boxes_path", lambda video, start, end: tmp_path / "boxes.csv")
    monkeypatch.setattr(vision.movement, "require_video_file", lambda video: None)
    monkeypatch.setattr(vision.movement, "video_fps_and_width", lambda video: (30.0, 1280.0))


def test_command_writes_the_hit_files_from_the_tracks_and_the_model_alone(tmp_path, monkeypatch, capsys):
    fake_track_files(tmp_path, monkeypatch)
    near_scores, far_scores = np.zeros(400), np.zeros(400)
    for frame in (100, 160, 220):
        near_scores[frame - 1 : frame + 2] = 0.95
    for frame in (130, 190):
        far_scores[frame - 1 : frame + 2] = 0.9
    save_hit_model(ScoreByFrameModel(near_scores, far_scores), tmp_path / "model.joblib")
    monkeypatch.setattr(vision.movement, "open_video", lambda *arguments: (_ for _ in ()).throw(AssertionError("video opened")))
    assert main(["club.mp4", "--start", "0", "--end", "399", "--model", str(tmp_path / "model.joblib"), "--hits-dir", str(tmp_path / "hits")]) == 0
    printed = capsys.readouterr().out
    assert "rallies: 1, hits: 5, inferred serves: 1" in printed
    hits_file, rallies_file = output_paths(Path("club.mp4"), 0, 399, tmp_path / "hits")
    assert hits_file == tmp_path / "hits" / "club" / "hits_0_399.csv" and hits_file.is_file() and rallies_file.is_file()
    hits = pd.read_csv(hits_file)
    assert list(hits.columns) == HIT_COLUMNS and hits["frame"].tolist() == [77, 100, 130, 160, 190, 220] and hits["reach"].isna().all()
    assert hits["hitter"].tolist() == ["opp", "me", "opp", "me", "opp", "me"] and hits["score"].isna().tolist() == [True] + [False] * 5
    rallies = pd.read_csv(rallies_file)
    assert list(rallies.columns) == RALLY_COLUMNS and rallies.loc[0, "start_frame"] == 77 and rallies.loc[0, "hits"] == 6
    assert main(["club.mp4", "--start", "0", "--end", "399", "--model", str(tmp_path / "model.joblib"), "--hits-dir", str(tmp_path / "hits"), "--no-serves"]) == 0
    assert "rallies: 1, hits: 5, inferred serves: 0" in capsys.readouterr().out
    assert pd.read_csv(hits_file)["frame"].tolist() == [100, 130, 160, 190, 220] and pd.read_csv(rallies_file).loc[0, "hits"] == 5


def test_command_errors_name_the_missing_input_and_exit_one(tmp_path, monkeypatch, capsys):
    assert main([str(tmp_path / "missing.mp4"), "--start", "0", "--end", "9"]) == 1
    assert "error: video not found" in capsys.readouterr().err
    fake_track_files(tmp_path, monkeypatch)
    assert main(["club.mp4", "--start", "0", "--end", "9", "--model", str(tmp_path / "none.joblib"), "--hits-dir", str(tmp_path / "hits")]) == 1
    assert "movement hit model not found" in capsys.readouterr().err
    assert main(["club.mp4", "--model", str(tmp_path / "none.joblib")]) == 1
    assert "give --start and --end" in capsys.readouterr().err
    assert main(["club.mp4", "--train"]) == 1 and "--match" in capsys.readouterr().err
    assert main(["club.mp4", "--cross-validate", "--range", "0", "9", "1"]) == 1 and "--match" in capsys.readouterr().err
    assert not (tmp_path / "hits").exists()
    with pytest.raises(SystemExit) as exit_info:
        main(["club.mp4", "--start", "0", "--end", "9", "--range", "1", "2"])
    assert exit_info.value.code == 2


def test_movement_module_reads_no_shuttle_or_contact_data_and_has_no_comments():
    source = MOVEMENT_SOURCE_PATH.read_text()
    module = ast.parse(source)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source.splitlines() if line.lstrip().startswith("#")] == []
    imported = {node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)}
    assert not any(name and name.startswith(("vision.shuttle", "vision.contact", "vision.camera")) for name in imported)
    assert not re.search(r"\bshuttle[._(]", source) and "shuttle_" not in source


@pytest.mark.parametrize("doc_path", [MOVEMENT_DOC_PATH, MOVEMENT_RESULTS_PATH])
def test_movement_docs_have_no_task_ids_and_state_the_serve_gap(doc_path):
    text = doc_path.read_text()
    assert re.findall(TASK_ID_PATTERN, text) == []
    if doc_path == MOVEMENT_RESULTS_PATH:
        assert "7 of 7" in text and "Serves are not found" in text


@requires_match_tracks_and_model
def test_shipped_model_reproduces_the_documented_in_sample_numbers_and_rallies_on_the_first_range():
    from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots
    from vision.court import calibration_path, load_calibration
    from vision.hits import labelled_hits

    settings = settings_for_video(30.0, 1280.0)
    tracks = load_player_tracks(Path("data/match.mp4"), 14201, 14947)
    hits, rallies = detect_hits(tracks, load_hit_model(PROJECT_ROOT / MODEL_PATH), settings)
    homography = load_calibration(calibration_path(Path("data/match.mp4")))["pixel_to_court_homography"]
    labels = labelled_hits(load_set_shots(SHUTTLESET_SET_DIR, AXELSEN_GINTING, 1), homography, 14201, 14947)
    summary = evaluation_summary(hits, labels, 5)
    assert (summary["labelled"], summary["detected"], summary["matched"]) == (26, 25, 25)
    assert summary["hitter_correct"] == summary["hitter_judged"] == 25
    assert len(hits) == 26 and len(rallies) == 1 and rallies.loc[0, "hits"] == 26 and rallies.loc[0, "start_frame"] == 14209
    serve = inferred_serves_only(hits)
    assert serve["frame"].tolist() == [14209] and serve["hitter"].tolist() == ["me"] and detected_hits_only(hits)["frame"].min() == 14232
    assert serve_evaluation_line(hits, labels, 5).startswith("inferred serves: 1, labelled first shots: 1, within ±5 frames: 0")
    saved = pd.read_csv(PROJECT_ROOT / "data" / "hits" / "match" / "hits_14201_14947.csv")
    pd.testing.assert_frame_equal(saved, hits, check_dtype=False)


@pytest.mark.skipif(
    not STAND_IN_VIDEO_PATH.exists() or not (PROJECT_ROOT / MODEL_PATH).exists() or not tracks_exist(Path("data/recordings/stand_in/set1.mp4"), 0, 3599),
    reason="needs the stand-in video, its player tracks of frames 0-3599 and data/models/movement_hits.joblib",
)
def test_shipped_model_gives_the_documented_stand_in_detections():
    settings = settings_for_video(60.0, 1920.0)
    hits, rallies = detect_hits(load_player_tracks(Path("data/recordings/stand_in/set1.mp4"), 0, 3599), load_hit_model(PROJECT_ROOT / MODEL_PATH), settings)
    detected = detected_hits_only(hits)
    assert (len(rallies), len(hits), len(detected)) == (5, 44, 39)
    serves = inferred_serves_only(hits)
    assert len(serves) == 5 and (detected.groupby("rally")["frame"].min().to_numpy() - serves["frame"].to_numpy() == 46).all()
    assert (rallies["start_frame"].to_numpy() == serves["frame"].to_numpy()).all() and rallies["hits"].sum() == 44
    documented_frames = [int(frame) for frame in re.findall(r"^\| (\d+) \| (?:me|opp) \|", MOVEMENT_RESULTS_PATH.read_text(), flags=re.MULTILINE)]
    assert spot_check_sample(detected, 20)["frame"].tolist() == documented_frames
    documented_hitters = re.findall(r"^\| \d+ \| (me|opp) \|", MOVEMENT_RESULTS_PATH.read_text(), flags=re.MULTILINE)
    assert spot_check_sample(detected, 20)["hitter"].tolist() == documented_hitters
    pd.testing.assert_frame_equal(pd.read_csv(PROJECT_ROOT / "data" / "hits" / "stand_in_set1" / "hits_0_3599.csv"), hits, check_dtype=False)
