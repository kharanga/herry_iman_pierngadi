import cv2
import numpy as np
import pandas as pd
import pytest

import vision.players
from vision.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from vision.players import (
    DETECTION_COLUMNS,
    POSITION_COLUMNS,
    STILL_LIMIT_SECONDS,
    FrameDetections,
    PlayerTrackingState,
    at_image_side,
    box_heights,
    detection_rows,
    detections_of_frame,
    empty_detections,
    frames_for_seconds,
    inside_lines_mask,
    label_players,
    players_from_detections,
    still_frame_counts,
    track_players,
    updated_anchors,
)
from tests.test_players import MODEL_PATH, detections_at, view_video_and_calibration
from tests.test_players_batches import FakeTracker, two_players_boxes

STILL_LIMIT = 60


def test_inside_lines_allows_the_margin_only():
    points = [[-0.3, 0.0], [-0.31, 5.0], [COURT_WIDTH_M + 0.3, COURT_LENGTH_M + 0.3], [3.0, COURT_LENGTH_M + 0.31], [3.0, 6.0]]
    assert inside_lines_mask(points).tolist() == [True, False, True, False, True]
    assert inside_lines_mask(np.zeros((0, 2))).tolist() == []


def test_anchors_restart_when_a_track_moves_past_the_radius_and_unseen_tracks_are_forgotten():
    anchors = updated_anchors({}, [1, 2], [[1.0, 1.0], [5.0, 5.0]], 100)
    anchors = updated_anchors(anchors, [1, 2], [[1.39, 1.0], [5.41, 5.0]], 130)
    assert anchors == {1: (1.0, 1.0, 100), 2: (5.41, 5.0, 130)}
    assert still_frame_counts(anchors, [1, 2], 160).tolist() == [60, 30]
    assert updated_anchors(anchors, [2], [[5.41, 5.0]], 161) == {2: (5.41, 5.0, 130)}


def test_box_heights_and_the_image_side_edge():
    boxes = [[0, 10, 50, 110], [4, 0, 40, 30], [5, 0, 1275, 30], [600, 0, 1276, 30]]
    assert box_heights(boxes).tolist() == [100, 30, 30, 30]
    assert at_image_side(boxes, 1280).tolist() == [True, True, False, True]
    assert at_image_side(boxes, None).tolist() == [False] * 4


def run_frames(frames, still_limit=STILL_LIMIT, frame_width=None):
    state, rows = PlayerTrackingState(), []
    for frame_index, (points, track_ids, confidences) in frames:
        if points:
            detections, positions = detections_at(points, track_ids, confidences)
        else:
            detections, positions = empty_detections(), np.zeros((0, 2))
        frame_rows, state = label_players(frame_index, detections, positions, state, still_limit_frames=still_limit, frame_width=frame_width)
        rows.extend(frame_rows)
    return pd.DataFrame(rows, columns=POSITION_COLUMNS), state


def test_a_still_stand_in_outside_the_lines_is_held_until_the_limit_then_someone_moving_inside_wins():
    judge = [6.9, 12.5]
    frames = [(frame, ([judge], [9], [0.99])) for frame in range(0, 50)]
    frames += [(frame, ([judge, [3.0, 10.0 + 0.02 * frame]], [9, 4], [0.99, 0.5])) for frame in range(50, 80)]
    rows, _ = run_frames(frames)
    opp = rows[rows["player"] == "opp"].set_index("frame")
    assert (opp.loc[0:49, "track_id"] == 9).all()
    assert (opp.loc[50:59, "track_id"] == 9).all() and opp.loc[50:59, "doubtful"].all()
    assert (opp.loc[60:79, "track_id"] == 4).all() and not opp.loc[60:79, "doubtful"].any()


def test_a_still_stand_in_with_nobody_inside_stays_but_is_doubtful():
    rows, _ = run_frames([(frame, ([[6.9, 12.5]], [9], [0.99])) for frame in range(0, 70)])
    assert (rows["track_id"] == 9).all()
    assert not rows.set_index("frame").loc[0:59, "doubtful"].any() and rows.set_index("frame").loc[60:69, "doubtful"].all()


def test_an_oversized_box_at_the_image_side_is_never_the_player_and_marks_the_frame_doubtful():
    player_points = [[3.0, 2.0]]
    detections, positions = detections_at(player_points, [1], [0.9])
    state = PlayerTrackingState()
    for frame in range(0, 5):
        _, state = label_players(frame, detections, positions, state, still_limit_frames=STILL_LIMIT, frame_width=1280)
    player_box = np.asarray(detections.boxes_xyxy, dtype=float)[0]
    height = player_box[3] - player_box[1]
    passer_box = [0.0, player_box[3] - 2 * height, player_box[2] - player_box[0], player_box[3]]
    both = FrameDetections(np.vstack([player_box, passer_box]), np.array([1, 2]), np.array([0.9, 0.99]))
    both_positions = np.vstack([positions, positions + [[0.2, 0.2]]])
    rows, _ = label_players(5, both, both_positions, state, still_limit_frames=STILL_LIMIT, frame_width=1280)
    assert [(row["player"], row["track_id"], row["doubtful"]) for row in rows] == [("me", 1, True)]
    alone = FrameDetections(np.array([passer_box]), np.array([2]), np.array([0.99]))
    rows, _ = label_players(6, alone, positions + [[0.2, 0.2]], state, still_limit_frames=STILL_LIMIT, frame_width=1280)
    assert rows == []


def test_the_detection_table_round_trips_one_frame():
    detections, positions = detections_at([[3.0, 2.0], [3.0, 11.0]], [7, 8], [0.8, 0.9])
    table = pd.DataFrame(detection_rows(5, detections, positions), columns=DETECTION_COLUMNS)
    replayed, replayed_positions = detections_of_frame(table)
    assert replayed.track_ids.tolist() == [7, 8] and np.allclose(replayed.boxes_xyxy, detections.boxes_xyxy, atol=0.05)
    assert np.allclose(replayed_positions, positions, atol=1e-3)


def live_and_replayed(frames, fps):
    still_limit = frames_for_seconds(STILL_LIMIT_SECONDS, fps)
    live, _ = run_frames(frames, still_limit=still_limit)
    table_rows = []
    for frame_index, (points, track_ids, confidences) in frames:
        if points:
            detections, positions = detections_at(points, track_ids, confidences)
            table_rows += detection_rows(frame_index, detections, positions)
    replayed, _ = players_from_detections(pd.DataFrame(table_rows, columns=DETECTION_COLUMNS), fps, None)
    return live, replayed


def test_replaying_saved_detections_gives_the_live_rows():
    frames = [(frame, ([[3.0, 2.0 + 0.01 * frame], [6.9, 12.5]], [1, 9], [0.9, 0.99])) for frame in range(0, 40)]
    frames += [(frame, ([[3.0, 2.4], [6.9, 12.5], [3.0, 10.0 + 0.03 * frame]], [1, 9, 4], [0.9, 0.99, 0.5])) for frame in range(40, 90)]
    live, replayed = live_and_replayed(frames, 30.0)
    pd.testing.assert_frame_equal(live.reset_index(drop=True), replayed.reset_index(drop=True), check_dtype=False)


def test_replaying_matches_live_across_a_main_view_frame_with_no_detection():
    frames = [(frame, ([[3.0, 2.0], [3.0, 11.0]], [1, 2], [0.9, 0.9])) for frame in range(0, 5)]
    frames += [(5, ([], [], []))]
    frames += [(frame, ([[3.0, NET_Y_M + 0.3], [3.0, NET_Y_M - 0.3]], [1, 2], [0.9, 0.9])) for frame in range(6, 10)]
    live, replayed = live_and_replayed(frames, 30.0)
    pd.testing.assert_frame_equal(live.reset_index(drop=True), replayed.reset_index(drop=True), check_dtype=False)


def test_track_players_still_limit_is_two_seconds_of_video_frames_at_any_stride(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)
    limits = []
    real_label_players = vision.players.label_players

    def recording_label_players(*arguments, **keywords):
        limits.append(keywords["still_limit_frames"])
        return real_label_players(*arguments, **keywords)

    monkeypatch.setattr(vision.players, "label_players", recording_label_players)
    monkeypatch.setattr(vision.players, "load_person_tracker", lambda model_path: FakeTracker(two_players_boxes()))
    for stride in (1, 2):
        limits.clear()
        track_players(video_path, 0, 7, calibration_file, MODEL_PATH, frame_stride=stride)
        assert set(limits) == {frames_for_seconds(STILL_LIMIT_SECONDS, fps)}, stride


def test_remembered_tracks_keep_up_to_the_memory_limit():
    from vision.players import TRACK_MEMORY_FRAMES_AT_30_FPS, TRACK_MEMORY_SECONDS, remembered_tracks
    assert TRACK_MEMORY_FRAMES_AT_30_FPS == frames_for_seconds(TRACK_MEMORY_SECONDS, 30.0) == 15
    assert remembered_tracks({1: 100, 2: 99, 3: 85, 4: 84}, 100, 15) == {1, 2, 3}


def gap_run(gap_frames, return_y):
    before = [(frame, ([[3.0, 2.0], [3.0, 11.0]], [1, 2], [0.9, 0.9])) for frame in range(0, 5)]
    back = 4 + gap_frames
    after = [(frame, ([[3.0, return_y], [3.0, 11.0]], [1, 2], [0.9, 0.9])) for frame in range(back, back + 3)]
    rows, state = run_frames(before + after)
    return rows[rows["frame"] >= back], state


def test_a_short_gap_keeps_a_tracks_side_at_the_net_and_a_longer_one_decides_it_afresh():
    kept, kept_state = gap_run(15, NET_Y_M + 0.3)
    assert set(kept[kept["track_id"] == 1]["player"]) == {"me"} and kept_state.side_by_track_id[1] == "me"
    redecided, redecided_state = gap_run(16, NET_Y_M + 0.3)
    assert redecided_state.side_by_track_id[1] == "opp"
    assert "me" not in set(redecided["player"]) and set(redecided[redecided["player"] == "opp"]["track_id"]) == {2}


def test_a_short_gap_keeps_the_still_clock_and_a_longer_one_restarts_it():
    _, kept = gap_run(15, 2.0)
    assert kept.anchor_by_track_id[1][2] == 0
    _, restarted = gap_run(16, 2.0)
    assert restarted.anchor_by_track_id[1][2] == 4 + 16


def test_an_empty_frame_carries_the_memory():
    state = PlayerTrackingState()
    detections, positions = detections_at([[3.0, 2.0]], [1], [0.9])
    _, state = label_players(0, detections, positions, state)
    _, after_empty = label_players(1, empty_detections(), np.zeros((0, 2)), state)
    assert after_empty.side_by_track_id == state.side_by_track_id and after_empty.anchor_by_track_id == state.anchor_by_track_id
    assert after_empty.last_seen_by_track_id == {1: 0}


def test_live_and_replay_agree_at_stride_two_with_empty_frames_and_gaps():
    fps, stride = 30.0, 2
    frames = []
    for frame in range(0, 200, stride):
        if frame in (40, 42, 120):
            frames.append((frame, ([], [], [])))
        elif 60 <= frame < 100:
            frames.append((frame, ([[6.9, 12.5]], [9], [0.99])))
        else:
            frames.append((frame, ([[3.0, NET_Y_M - 0.4 + 0.004 * frame], [6.9, 12.5], [3.0, 10.0 + 0.01 * frame]], [1, 9, 4], [0.9, 0.99, 0.5])))
    still_limit, memory = frames_for_seconds(STILL_LIMIT_SECONDS, fps), frames_for_seconds(0.5, fps)
    state, live_rows = PlayerTrackingState(), []
    table_rows = []
    for frame_index, (points, track_ids, confidences) in frames:
        if points:
            detections, positions = detections_at(points, track_ids, confidences)
            table_rows += detection_rows(frame_index, detections, positions)
        else:
            detections, positions = empty_detections(), np.zeros((0, 2))
        frame_rows, state = label_players(frame_index, detections, positions, state, still_limit_frames=still_limit, memory_frames=memory)
        live_rows += frame_rows
    replayed, _ = players_from_detections(pd.DataFrame(table_rows, columns=DETECTION_COLUMNS), fps, None)
    live = pd.DataFrame(live_rows, columns=POSITION_COLUMNS)
    pd.testing.assert_frame_equal(live.reset_index(drop=True), replayed.reset_index(drop=True), check_dtype=False)
    assert live["doubtful"].any() and set(live["player"]) == {"me", "opp"}
