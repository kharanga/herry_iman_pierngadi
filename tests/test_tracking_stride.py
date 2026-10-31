import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_players import view_video_and_calibration
from vision.features import effective_frame_numbers, is_inferred_serve, own_shot_rows, video_frame_of_row
from vision.movement import MovementSettings, PlayerTracks, detect_hits, player_tracks, settings_for_video
from vision.players import (
    TRACKING_RECORD_SUFFIX,
    check_frame_stride,
    effective_fps,
    frame_stride_of,
    frames_in_range,
    is_default_dir,
    read_tracking_record,
    sampled_frames,
    tracking_record,
    tracking_record_path,
    write_tracking_record,
)
from vision.alignment import open_video


class ScoreBySampleModel:
    def __init__(self, near_scores, far_scores):
        self.near_scores = np.asarray(near_scores, dtype=float)
        self.far_scores = np.asarray(far_scores, dtype=float)

    def predict_proba(self, feature_matrix):
        own_y = feature_matrix[:, 11]
        scores = self.far_scores if np.nanmean(own_y) > 6.7 else self.near_scores
        return np.column_stack([1 - scores[: len(feature_matrix)], scores[: len(feature_matrix)]])


def still_tracks(frames: np.ndarray) -> PlayerTracks:
    count = len(frames)
    near = np.column_stack([np.full(count, 3.0), np.full(count, 2.0)])
    far = np.column_stack([np.full(count, 3.0), np.full(count, 11.0)])
    return PlayerTracks(frames, near, far, np.tile([600.0, 400.0, 700.0, 650.0], (count, 1)), np.tile([620.0, 250.0, 660.0, 350.0], (count, 1)))


def alternating_scores(sample_count: int, near_samples, far_samples) -> ScoreBySampleModel:
    near, far = np.zeros(sample_count), np.zeros(sample_count)
    near[list(near_samples)] = 0.95
    far[list(far_samples)] = 0.9
    return ScoreBySampleModel(near, far)


def test_sampled_frames_and_stride_checks():
    assert sampled_frames(10, 17, 2).tolist() == [10, 12, 14, 16]
    assert sampled_frames(10, 11, 5).tolist() == [10]
    assert sampled_frames(3, 5).tolist() == [3, 4, 5]
    with pytest.raises(ValueError, match="frame stride must be >= 1"):
        check_frame_stride(0)
    with pytest.raises(ValueError):
        effective_fps(60.0, 0)
    assert effective_fps(60.0, 2) == 30.0 and effective_fps(30.0, 1) == 30.0


def test_frames_in_range_with_a_stride_reads_only_the_sampled_frames(view_video_and_calibration):
    video_path, _, frame_is_main_view = view_video_and_calibration
    capture = open_video(video_path)
    try:
        read = [frame_index for frame_index, _ in frames_in_range(capture, 1, 7, 3)]
    finally:
        capture.release()
    assert read == [1, 4, 7]


def test_tracking_record_round_trip_and_missing_record_is_stride_one(tmp_path):
    positions_file = tmp_path / "positions_0_99.csv"
    assert tracking_record_path(positions_file).name == "positions_0_99.csv" + TRACKING_RECORD_SUFFIX
    assert read_tracking_record(positions_file) == {"frame_stride": 1} and frame_stride_of(positions_file) == 1
    record = tracking_record(60.0, 1920.0, 2, Path("data/models/yolo11n.pt"), "mps", 8)
    assert record == {"frame_stride": 2, "video_fps": 60.0, "effective_fps": 30.0, "frame_width": 1920.0, "model": "yolo11n.pt", "device": "mps", "batch_frames": 8}
    written = write_tracking_record(positions_file, record)
    assert json.loads(written.read_text()) == record and frame_stride_of(positions_file) == 2


def test_default_directory_check():
    assert is_default_dir(None, Path("data/players")) and is_default_dir(Path("data/players"), Path("data/players"))
    assert not is_default_dir(Path("elsewhere"), Path("data/players"))


def test_player_tracks_with_a_stride_hold_the_sampled_video_frames():
    positions = pd.DataFrame({"frame": [10, 12, 14], "player": "me", "x_m": [3.0, 3.1, 3.2], "y_m": 2.0, "track_id": 1, "confidence": 0.9})
    boxes = pd.DataFrame({"frame": [10, 12, 14], "player": "me", "left": 1, "top": 2, "right": 3, "bottom": 4})
    tracks = player_tracks(positions, boxes, 10, 15, 2)
    assert tracks.frames.tolist() == [10, 12, 14] and tracks.near_points[:, 0].tolist() == [3.0, 3.1, 3.2]


def test_strided_rows_get_the_effective_rate_and_frame_numbers():
    assert effective_frame_numbers(pd.DataFrame({"frame": [100, 102, 105]}), 2)["frame"].tolist() == [50, 51, 52]
    same = pd.DataFrame({"frame": [1, 2]})
    assert effective_frame_numbers(same, 1) is same
    hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 130], "hitter": ["me", "opp"], "score": 0.9, "reach": np.nan})
    positions = pd.DataFrame({"frame": [100, 130, 100, 130], "player": ["me", "me", "opp", "opp"], "x_m": [3.0, 3.0, 2.5, 2.5], "y_m": [2.0, 2.0, 10.0, 10.0]})
    rows = own_shot_rows(hits, positions, "club_1", 1, 30.0, frame_stride=2)
    assert [(row["frame_num"], row["fps"]) for row in rows] == [(50, 30.0), (65, 30.0)]
    assert rows[0]["landing_x"] is not None


def test_strided_tracks_give_the_same_hits_and_rallies_as_the_same_samples_numbered_one_apart():
    sample_count = 400
    near_samples, far_samples = (100, 180, 260), (140, 220)
    model = alternating_scores(sample_count, near_samples, far_samples)
    settings = settings_for_video(30.0, 1280.0, MovementSettings(score_smoothing_frames=1, infer_serves=False))
    effective_hits, effective_rallies = detect_hits(still_tracks(np.arange(sample_count)), model, settings)
    assert effective_hits["frame"].tolist() == [100, 140, 180, 220, 260] and len(effective_rallies) == 1
    strided_hits, strided_rallies = detect_hits(still_tracks(np.arange(0, 2 * sample_count, 2)), model, settings)
    assert strided_hits["frame"].tolist() == [200, 280, 360, 440, 520]
    assert len(strided_rallies) == 1 and strided_hits["rally"].tolist() == [1] * 5


def test_an_inferred_serve_on_strided_tracks_is_the_serve_gap_in_seconds_before_the_first_hit():
    sample_count = 400
    model = alternating_scores(sample_count, (100, 140), (120,))
    settings = settings_for_video(30.0, 1280.0, MovementSettings(score_smoothing_frames=1))
    effective_hits, _ = detect_hits(still_tracks(np.arange(sample_count)), model, settings)
    strided_hits, _ = detect_hits(still_tracks(np.arange(0, 2 * sample_count, 2)), model, settings)
    assert effective_hits["frame"].tolist()[0] == 100 - 23
    assert strided_hits["frame"].tolist() == [2 * frame for frame in effective_hits["frame"].tolist()]


def moving_tracks(frames: np.ndarray) -> PlayerTracks:
    count = len(frames)
    steps = np.arange(count)
    rest = (steps % 120) > 90
    near_x = 3.0 + np.where(rest, 0.0, 0.6 * np.sin(steps / 7.0))
    far_x = 3.0 + np.where(rest, 0.0, 0.6 * np.cos(steps / 9.0))
    near = np.column_stack([near_x, np.full(count, 2.0)])
    far = np.column_stack([far_x, np.full(count, 11.0)])
    return PlayerTracks(frames, near, far, np.tile([600.0, 400.0, 700.0, 650.0], (count, 1)), np.tile([620.0, 250.0, 660.0, 350.0], (count, 1)))


def video_frame_hits(tracks: PlayerTracks, model, settings: MovementSettings):
    from vision.movement import alternating_hit_chain, hit_candidates, rally_tables, sample_hit_scores, split_into_rallies, trim_rally_ends

    scores = sample_hit_scores(tracks, model, settings)
    chain = alternating_hit_chain(hit_candidates(tracks.frames, scores, settings), settings)
    rallies = [trimmed for rally in split_into_rallies(chain, settings.max_hit_gap_frames) if len(trimmed := trim_rally_ends(rally, settings.rally_end_score_min)) >= 2]
    return rally_tables(rallies, tracks, settings)


@pytest.mark.parametrize("first_frame", [0, 1000, 777])
def test_stride_one_gives_the_same_tables_as_working_in_video_frames_directly(first_frame):
    sample_count = 400
    model = alternating_scores(sample_count, (40, 80, 120, 260, 300), (60, 100, 280, 320))
    settings = settings_for_video(30.0, 1280.0, MovementSettings(score_smoothing_frames=1))
    tracks = moving_tracks(np.arange(first_frame, first_frame + sample_count))
    hits, rallies = detect_hits(tracks, model, settings)
    direct_hits, direct_rallies = video_frame_hits(tracks, model, settings)
    pd.testing.assert_frame_equal(hits.reset_index(drop=True), direct_hits.reset_index(drop=True))
    pd.testing.assert_frame_equal(rallies.reset_index(drop=True), direct_rallies.reset_index(drop=True))
    assert len(rallies) == 2 and hits["frame"].min() >= first_frame


def test_tracks_starting_off_the_stride_grid_keep_their_own_frames():
    sample_count = 400
    model = alternating_scores(sample_count, (100, 180), (140,))
    settings = settings_for_video(30.0, 1280.0, MovementSettings(score_smoothing_frames=1))
    hits, rallies = detect_hits(still_tracks(np.arange(101, 101 + 2 * sample_count, 2)), model, settings)
    assert hits["frame"].tolist() == [101 + 2 * (100 - 23), 301, 381, 461]
    assert all((frame - 101) % 2 == 0 for frame in rallies[["start_frame", "end_frame"]].to_numpy().ravel())


def test_movement_refuses_ranges_tracked_at_different_strides(tmp_path, capsys):
    from vision.movement import main as movement_main
    from vision.players import positions_path

    video_path = tmp_path / "game.mp4"
    video_path.write_bytes(b"")
    players_dir = tmp_path / "players"
    for (start_frame, end_frame), frame_stride in (((0, 99), 1), ((200, 299), 2)):
        positions_file = positions_path(video_path, start_frame, end_frame, players_dir)
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        write_tracking_record(positions_file, tracking_record(30.0, 1280.0, frame_stride, Path("m.pt"), "cpu", 1))
    arguments = [str(video_path), "--train", "--match", "m", "--range", "0", "99", "1", "--range", "200", "299", "1", "--players-dir", str(players_dir), "--model", str(tmp_path / "model.joblib")]
    assert movement_main(arguments) == 1
    assert "different frame strides: [1, 2]" in capsys.readouterr().err
    assert not (tmp_path / "model.joblib").exists()


def test_spot_check_finds_the_inferred_serve_of_strided_rows(tmp_path):
    from vision.features import inferred_serve_frames
    from vision.hits import output_paths

    video_path = tmp_path / "videos" / "game.mp4"
    hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [100, 130, 160], "hitter": ["me", "opp", "me"], "score": [np.nan, 0.9, 0.8], "reach": np.nan})
    positions = pd.DataFrame({"frame": [100, 130, 160] * 2, "player": ["me"] * 3 + ["opp"] * 3, "x_m": 3.0, "y_m": [2.0] * 3 + [10.0] * 3})
    hits_file, _ = output_paths(video_path, 0, 999, tmp_path / "hits")
    hits_file.parent.mkdir(parents=True)
    hits.to_csv(hits_file, index=False)
    rows = list(pd.DataFrame(own_shot_rows(hits, positions, "club_1", 1, 30.0, frame_stride=2)).itertuples())
    serve_frames = inferred_serve_frames(video_path, tmp_path / "hits")
    assert [row.frame_num for row in rows] == [50, 65, 80]
    assert [is_inferred_serve(row, serve_frames, 60.0) for row in rows] == [True, False, False]
    assert [video_frame_of_row(row, 60.0) for row in rows] == [100, 130, 160]


def test_rows_at_the_video_rate_or_without_an_fps_are_stride_one():
    from vision.features import frame_stride_of_row

    every_frame = pd.DataFrame({"frame_num": [101, 131], "fps": [60.0, 60.0]}).itertuples()
    assert [(frame_stride_of_row(row, 60.0), video_frame_of_row(row, 60.0)) for row in every_frame] == [(1, 101), (1, 131)]
    no_fps = next(pd.DataFrame({"frame_num": [7], "fps": [float("nan")]}).itertuples())
    assert frame_stride_of_row(no_fps, 60.0) == 1 and frame_stride_of_row(no_fps, 0.0) == 1
    assert is_inferred_serve(next(pd.DataFrame({"frame_num": [101], "fps": [60.0]}).itertuples()), {101}, 60.0)


def test_off_grid_strided_rows_still_find_their_serve():
    off_grid = pd.DataFrame({"frame_num": [101 // 2, 131 // 2], "fps": [30.0, 30.0]}).itertuples()
    assert [is_inferred_serve(row, {101}, 60.0) for row in off_grid] == [True, False]
