import duckdb
import numpy as np
import pandas as pd
import pytest

import vision.flight as flight
from vision.flight import frames_seen_after, has_jump, hit_coverage, in_flight_hitter, shuttle_track
from vision.flight_types import FLIGHT_FEATURE_NAMES, Source, flight_features, labels_as_hits, labelled_rallies_table, pixels_per_meter, write_predictions


def test_frames_seen_after_counts_frames_strictly_after_the_hit_up_to_the_window_end():
    detected = np.array([100, 101, 105, 115, 116])
    assert frames_seen_after(100, detected, 15) == 3
    assert frames_seen_after(99, detected, 2) == 2
    assert frames_seen_after(200, detected, 15) == 0


def test_a_jump_is_judged_per_frame_so_a_long_step_over_several_frames_is_not_one():
    width = 1000
    one_frame = pd.DataFrame({"frame": [0, 1], "x": [0.0, 160.0], "y": [0.0, 0.0]})
    spread = pd.DataFrame({"frame": [0, 4], "x": [0.0, 160.0], "y": [0.0, 0.0]})
    assert has_jump(one_frame, width) and not has_jump(spread, width)
    assert not has_jump(one_frame.iloc[:1], width)


def test_smooth_hits_are_a_subset_of_usable_ones():
    labels = pd.DataFrame({"frame_num": [0, 100, 200], "player": ["me", "opp", "me"], "type_group": ["serve", "lift", "net"]})
    points = [(frame, 10.0 * frame, 0.0) for frame in range(1, 16)]
    points += [(frame, 100.0, 0.0) for frame in range(101, 106)]
    points += [(frame, 0.0 if frame % 2 else 900.0, 0.0) for frame in range(201, 216)]
    coverage = hit_coverage(labels, pd.DataFrame(points, columns=["frame", "x", "y"]), 30.0, 1000)
    assert coverage[["seen", "usable", "smooth"]].values.tolist() == [[True, True, True], [True, False, False], [True, True, False]]
    assert (coverage["usable"] | ~coverage["smooth"]).all()


def test_frames_before_the_first_hit_belong_to_the_first_hitter():
    hits = pd.DataFrame({"frame_num": [120, 100], "player": ["opp", "me"]})
    assert in_flight_hitter(np.array([90, 100, 110, 120, 130]), hits).tolist() == ["me", "me", "me", "opp", "opp"]


def test_the_track_joins_overlapping_stretches_and_refuses_an_empty_range(tmp_path, monkeypatch):
    folder = tmp_path / "v"
    folder.mkdir()
    for start, end in ((0, 9), (10, 19), (20, 29)):
        pd.DataFrame({"frame": range(start, end + 1), "x": 0.0, "y": 0.0, "visibility": 1, "inpainted": 0}).to_csv(folder / f"shuttle_{start}_{end}.csv", index=False)
    monkeypatch.setattr(flight, "video_id", lambda path: "v")
    track = shuttle_track(tmp_path / "v.mp4", 5, 14, tmp_path)
    assert track["frame"].tolist() == list(range(5, 15))
    with pytest.raises(FileNotFoundError):
        shuttle_track(tmp_path / "v.mp4", 100, 200, tmp_path)


def flight_points(frames, ys):
    return pd.DataFrame({"frame": frames, "x": 0.0, "y": ys})


def test_flight_features_need_three_early_points_an_early_first_one_and_no_jump():
    fps, width = 30.0, 1280
    rising = flight_points(range(101, 121), [500 - 10 * k for k in range(20)])
    features = flight_features(100, 130, rising, 50.0, fps, width)
    assert features["flight_start_rising"] == 1.0 and features["flight_peak_m"] == pytest.approx(190 / 50)
    assert features["flight_seen_share"] == pytest.approx(20 / 29)
    missing = lambda points: all(np.isnan(value) for value in flight_features(100, 130, points, 50.0, fps, width).values())
    assert missing(flight_points([101, 102], [500, 490]))
    assert missing(flight_points([106, 107, 108, 109], [500, 490, 480, 470]))
    assert missing(flight_points([101, 102, 103, 104], [500, 490, 100, 90]).assign(x=[0, 0, 400, 400]))
    assert all(np.isnan(value) for value in flight_features(100, 130, rising, None, fps, width).values())
    assert list(features) == list(FLIGHT_FEATURE_NAMES)


def test_pixels_per_meter_on_a_scaling_homography():
    court_to_pixel_scale = 40.0
    pixel_to_court = np.diag([1 / court_to_pixel_scale, 1 / court_to_pixel_scale, 1.0])
    assert pixels_per_meter(pixel_to_court, 3.0, 6.0) == pytest.approx(court_to_pixel_scale)


def test_labels_become_hits_without_unknown_hitters_and_rallies_count_them():
    labels = pd.DataFrame({"rally": [1, 1, 2, 3], "frame_num": [10, 20, 30, 40], "player": ["me", "?", "opp", "me"], "type_group": ["serve"] * 4})
    hits = labels_as_hits(labels)
    assert hits[["rally", "frame", "hitter"]].values.tolist() == [[1, 10, "me"], [2, 30, "opp"], [3, 40, "me"]]
    rallies = pd.DataFrame({"rally": [1, 2, 4], "start_frame": [5, 25, 60], "end_frame": [22, 35, 70]})
    assert labelled_rallies_table(rallies, hits).values.tolist() == [[1, 5, 22, 1], [2, 25, 35, 1]]


def test_write_predictions_clears_only_this_matchs_own_rows_and_leaves_the_source_alone(tmp_path):
    source_db = tmp_path / "source.duckdb"
    source = Source(tmp_path / "game.mp4", source_db, 100)
    connection = duckdb.connect(str(source_db))
    connection.execute("CREATE TABLE shots (source VARCHAR, match_id VARCHAR, frame_num INTEGER, type_pred VARCHAR, type_detail_pred VARCHAR)")
    connection.execute("INSERT INTO shots VALUES ('own', ?, 10, 'net', 'net'), ('own', ?, 20, 'net', 'net'), ('own', 'other', 30, 'clear', 'clear'), ('shuttleset', ?, 40, 'drop', 'drop')", [source.match_id] * 3)
    row_id = connection.execute("SELECT rowid FROM shots WHERE frame_num = 10").fetchone()[0]
    before = connection.execute("SELECT * FROM shots ORDER BY frame_num").fetchall()
    connection.close()
    output = write_predictions(source, tmp_path / "out" / "typed.duckdb", [row_id], ["serve_long"])
    connection = duckdb.connect(str(output), read_only=True)
    assert connection.execute("SELECT frame_num, type_pred, type_detail_pred FROM shots ORDER BY frame_num").fetchall() == [(10, "serve", "serve_long"), (20, None, None), (30, "clear", "clear"), (40, "drop", "drop")]
    connection.close()
    connection = duckdb.connect(str(source_db), read_only=True)
    assert connection.execute("SELECT * FROM shots ORDER BY frame_num").fetchall() == before
    connection.close()


def test_the_low_memory_background_is_off_by_default():
    from vision.shuttle import parse_arguments
    assert parse_arguments(["v.mp4", "--start", "0", "--end", "10"]).low_memory_background is False
