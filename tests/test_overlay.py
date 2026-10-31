from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vision import overlay
from vision.court import video_id
from vision.hits import output_paths
from vision.overlay import (
    HIT_FLASH_COLOR_BGR,
    PLAYER_COLOR_BGR,
    SERVE_FLASH_COLOR_BGR,
    FrameState,
    HitFlash,
    blank_court_diagram,
    court_paths_diagram,
    court_to_diagram,
    draw_tracking_overlay,
    frame_state,
    hit_flash_at,
    load_overlay_inputs,
    rally_at,
    rally_of_frame,
    rows_by_player,
    scaled_box,
    write_tracking_video,
)
from vision.players import boxes_path, positions_path, tracking_record, write_tracking_record

VIDEO_FRAMES = 40
VIDEO_SIZE = (320, 240)


def tables(frames=range(10, 31)):
    frames = list(frames)
    positions = pd.DataFrame(
        {"frame": frames * 2, "player": ["me"] * len(frames) + ["opp"] * len(frames), "x_m": 3.0, "y_m": [2.0] * len(frames) + [11.0] * len(frames), "track_id": 1, "confidence": 0.9}
    )
    boxes = pd.DataFrame(
        {"frame": frames * 2, "player": ["me"] * len(frames) + ["opp"] * len(frames), "left": [100] * len(frames) + [150] * len(frames), "top": [120] * len(frames) + [40] * len(frames), "right": [140] * len(frames) + [180] * len(frames), "bottom": [220] * len(frames) + [100] * len(frames)}
    )
    hits = pd.DataFrame({"rally": [1, 1, 1, 2], "frame": [12, 16, 24, 40], "hitter": ["me", "opp", "me", "opp"], "score": [np.nan, 0.9, 0.8, 0.7], "reach": np.nan})
    rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [12, 40], "end_frame": [28, 60], "hits": [3, 1]})
    return positions, boxes, hits, rallies


def test_rows_and_rally_lookups_at_the_edges():
    positions, boxes, hits, rallies = tables()
    assert rows_by_player(boxes, 9) == {"me": None, "opp": None}
    assert int(rows_by_player(boxes, 10)["opp"]["left"]) == 150
    assert int(rally_at(12, rallies)["rally"]) == 1 and int(rally_at(28, rallies)["rally"]) == 1
    assert rally_at(29, rallies) is None and rally_at(11, rallies) is None
    assert rally_of_frame(40, rallies) == 2 and rally_of_frame(35, rallies) is None


def test_a_hit_flashes_for_exactly_flash_frames_and_the_latest_hit_wins():
    _, _, hits, _ = tables()
    assert hit_flash_at(11, hits) is None
    assert hit_flash_at(12, hits) == HitFlash("me", 12, True)
    assert hit_flash_at(15, hits) == HitFlash("me", 12, True)
    assert hit_flash_at(16, hits) == HitFlash("opp", 16, False)
    assert hit_flash_at(25, hits) == HitFlash("me", 24, False)
    assert hit_flash_at(33, hits) == HitFlash("me", 24, False) and hit_flash_at(34, hits) is None
    assert hit_flash_at(14, hits, flash_frames=2) is None


def test_frame_state_counts_the_shots_of_the_current_rally_only():
    positions, boxes, hits, rallies = tables()
    assert frame_state(11, positions, boxes, hits, rallies).rally_label == "between rallies"
    state = frame_state(16, positions, boxes, hits, rallies)
    assert (state.rally_label, state.shot_count) == ("rally 1 of 2", 2)
    assert frame_state(28, positions, boxes, hits, rallies).shot_count == 3
    assert frame_state(40, positions, boxes, hits, rallies).shot_count == 1


def test_scaled_box_rounds_each_edge():
    assert scaled_box({"left": 101, "top": 3, "right": 205, "bottom": 7.4}, 0.5) == (50, 2, 102, 4)


def pixels_of(image, color):
    return int(np.all(image == np.array(color, dtype=np.uint8), axis=2).sum())


def test_the_overlay_leaves_the_input_frame_alone_and_colours_by_flash_kind():
    positions, boxes, hits, rallies = tables()
    frame = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
    serve = draw_tracking_overlay(frame, frame_state(12, positions, boxes, hits, rallies))
    hit = draw_tracking_overlay(frame, frame_state(16, positions, boxes, hits, rallies))
    quiet = draw_tracking_overlay(frame, frame_state(11, positions, boxes, hits, rallies))
    assert int(frame.sum()) == 0 and serve.shape == frame.shape
    assert pixels_of(serve, SERVE_FLASH_COLOR_BGR) > 0 and pixels_of(serve, HIT_FLASH_COLOR_BGR) == 0
    assert pixels_of(hit, HIT_FLASH_COLOR_BGR) > 0 and pixels_of(hit, SERVE_FLASH_COLOR_BGR) == 0
    assert pixels_of(quiet, HIT_FLASH_COLOR_BGR) == 0 and pixels_of(quiet, SERVE_FLASH_COLOR_BGR) == 0
    assert pixels_of(quiet, PLAYER_COLOR_BGR["me"]) > 0 and pixels_of(quiet, PLAYER_COLOR_BGR["opp"]) > 0


def test_a_missing_player_is_named_and_not_boxed():
    frame = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
    nobody = FrameState(5, {"me": None, "opp": None}, {"me": None, "opp": None}, "between rallies", None, None)
    drawn = draw_tracking_overlay(frame, nobody)
    assert pixels_of(drawn, PLAYER_COLOR_BGR["me"]) == 0 and pixels_of(drawn, PLAYER_COLOR_BGR["opp"]) == 0
    assert pixels_of(drawn, (0, 200, 255)) > 0


def test_court_diagram_puts_the_near_baseline_at_the_bottom_and_draws_each_rally():
    corners = court_to_diagram([[0.0, 0.0], [6.1, 0.0], [0.0, 13.4]])
    assert corners.tolist() == [[40.0, 844.0], [406.0, 844.0], [40.0, 40.0]]
    positions, _, _, rallies = tables()
    moving = positions.assign(x_m=positions["frame"] * 0.1)
    blank = blank_court_diagram()
    drawn = court_paths_diagram(moving, rallies)
    assert drawn.shape == blank.shape and (drawn != blank).any()
    assert pixels_of(court_paths_diagram(moving.iloc[:0], rallies), overlay.RALLY_COLORS_BGR[0]) == 0


def write_inputs(tmp_path, video_path, start_frame, end_frame, frame_stride=1):
    positions, boxes, hits, rallies = tables(range(start_frame, end_frame + 1, frame_stride))
    players_dir, hits_dir = tmp_path / "players", tmp_path / "hits"
    positions_file = positions_path(video_path, start_frame, end_frame, players_dir)
    positions_file.parent.mkdir(parents=True)
    positions.to_csv(positions_file, index=False)
    boxes.to_csv(boxes_path(video_path, start_frame, end_frame, players_dir), index=False)
    if frame_stride != 1:
        write_tracking_record(positions_file, tracking_record(30.0, VIDEO_SIZE[0], frame_stride, Path("m.pt"), "cpu", 1))
    hits_file, rallies_file = output_paths(video_path, start_frame, end_frame, hits_dir)
    hits_file.parent.mkdir(parents=True)
    hits.to_csv(hits_file, index=False)
    rallies.to_csv(rallies_file, index=False)
    return players_dir, hits_dir


@pytest.fixture
def small_video(tmp_path):
    video_path = tmp_path / "videos" / "game.avi"
    video_path.parent.mkdir()
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, VIDEO_SIZE)
    for index in range(VIDEO_FRAMES):
        writer.write(np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), index * 5, dtype=np.uint8))
    writer.release()
    return video_path


def test_missing_track_file_is_named(tmp_path, small_video):
    with pytest.raises(FileNotFoundError, match="track file not found"):
        load_overlay_inputs(small_video, 10, 30, tmp_path / "players", tmp_path / "hits")


def video_frame_count(path):
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()


@pytest.mark.parametrize("ffmpeg_on_path", [True, False])
def test_the_tracking_video_is_written_only_to_its_output(tmp_path, small_video, monkeypatch, ffmpeg_on_path):
    players_dir, hits_dir = write_inputs(tmp_path, small_video, 10, 30)
    scratch_root = tmp_path / "system_temp"
    scratch_root.mkdir()
    monkeypatch.setattr(overlay.tempfile, "tempdir", str(scratch_root))
    if not ffmpeg_on_path:
        monkeypatch.setattr(overlay.shutil, "which", lambda name: None)
    before = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    output = write_tracking_video(small_video, 10, 30, tmp_path / "out" / "check.mp4", players_dir, hits_dir, output_width=160, output_stride=2)
    after = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert output == tmp_path / "out" / "check.mp4" and video_frame_count(output) == (11, 160)
    assert [path for path in after if path not in before] == [output]
    assert list(scratch_root.iterdir()) == []


def test_strided_tracks_draw_the_nearest_tracked_frame_at_or_before(tmp_path, small_video, monkeypatch):
    players_dir, hits_dir = write_inputs(tmp_path, small_video, 11, 30, frame_stride=3)
    drawn_states = []
    real_draw = overlay.draw_tracking_overlay
    monkeypatch.setattr(overlay, "draw_tracking_overlay", lambda frame, state, scale=1.0: drawn_states.append(state) or real_draw(frame, state, scale))
    monkeypatch.setattr(overlay.shutil, "which", lambda name: None)
    write_tracking_video(small_video, 11, 30, tmp_path / "out.mp4", players_dir, hits_dir, output_width=160, output_stride=2)
    assert [state.frame_index for state in drawn_states] == [11, 11, 14, 17, 17, 20, 23, 23, 26, 29]
    assert all(state.boxes["me"] is not None for state in drawn_states)


def test_a_failed_re_encode_leaves_nothing_in_the_temp_dir(tmp_path, small_video, monkeypatch):
    players_dir, hits_dir = write_inputs(tmp_path, small_video, 10, 30)
    scratch_root = tmp_path / "system_temp"
    scratch_root.mkdir()
    monkeypatch.setattr(overlay.tempfile, "tempdir", str(scratch_root))

    def failing_ffmpeg(command, check):
        raise overlay.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(overlay.subprocess, "run", failing_ffmpeg)
    with pytest.raises(overlay.subprocess.CalledProcessError):
        write_tracking_video(small_video, 10, 30, tmp_path / "out.mp4", players_dir, hits_dir, output_width=160)
    assert list(scratch_root.rglob("*")) == []
