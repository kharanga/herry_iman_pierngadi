import json

import numpy as np
import pandas as pd
import pytest

import vision.halve_fps as halve_fps
from vision.halve_fps import halve_label_files, halved_hit_frames, halved_rallies, halved_rally_ends, halved_rally_starts, label_problems, main
from vision.label_scores import rallies_file_for


def test_hits_round_half_up_starts_down_and_ends_up():
    assert halved_hit_frames([0, 1, 2, 3, 100, 101]).tolist() == [0, 1, 1, 2, 50, 51]
    assert halved_rally_starts([0, 1, 2, 3, 101]).tolist() == [0, 0, 1, 1, 50]
    assert halved_rally_ends([0, 1, 2, 3, 101]).tolist() == [0, 1, 1, 2, 51]


def write_labels(tmp_path, hits, rallies):
    labels_file = tmp_path / "game.csv"
    pd.DataFrame(hits, columns=["rally", "frame_num", "player", "type_group", "type_detail", "source"]).to_csv(labels_file, index=False)
    pd.DataFrame(rallies, columns=["rally", "start_frame", "end_frame", "bounds_checked"]).to_csv(rallies_file_for(labels_file), index=False)
    return labels_file


def test_a_clean_file_is_halved_with_other_columns_kept(tmp_path):
    labels_file = write_labels(tmp_path, [(1, 101, "me", "serve", "serve_short", "hand"), (1, 140, "opp", "net", "net", "hand")], [(1, 99, 201, True)])
    output = tmp_path / "out" / "game_30.csv"
    assert halve_label_files(labels_file, output) == []
    assert pd.read_csv(output).values.tolist() == [[1, 51, "me", "serve", "serve_short", "hand"], [1, 70, "opp", "net", "net", "hand"]]
    assert pd.read_csv(rallies_file_for(output)).values.tolist() == [[1, 49, 101, True]]


def test_colliding_hits_and_a_hit_outside_its_rally_are_refused_and_nothing_is_written(tmp_path):
    collide = write_labels(tmp_path, [(1, 101, "me", "lift", "lift", "hand"), (1, 102, "opp", "net", "net", "hand")], [(1, 90, 200, True)])
    output = tmp_path / "collide_30.csv"
    problems = halve_label_files(collide, output)
    assert problems == ["hits collide at frame 51"] and not output.exists() and not rallies_file_for(output).exists()
    hits = pd.DataFrame({"rally": [1], "frame_num": [60]})
    assert label_problems(hits, pd.DataFrame({"rally": [1], "start_frame": [61], "end_frame": [80]})) == ["hit at frame 60 falls outside rally 1"]


def test_back_to_back_rallies_do_not_overlap_after_halving():
    rallies = pd.DataFrame({"rally": [1, 2, 3], "start_frame": [11, 34, 511], "end_frame": [33, 510, 1268]})
    halved = halved_rallies(rallies).sort_values("start_frame")
    assert all(first.end_frame < second.start_frame for first, second in zip(halved.itertuples(), halved.iloc[1:].itertuples()))


def test_the_calibration_copy_halves_the_frame_index_and_keeps_the_rest(tmp_path, monkeypatch):
    source_dir, target_dir = tmp_path / "cal" / "orig", tmp_path / "cal" / "orig_30"
    source_dir.mkdir(parents=True)
    calibration = {"frame_index": 8401, "pixel_to_court_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "rms_reference_error_px": 3.6}
    (source_dir / "court.json").write_text(json.dumps(calibration))
    monkeypatch.setattr(halve_fps, "calibration_path", lambda video: tmp_path / "cal" / video.stem / "court.json")
    target = halve_fps.halve_calibration(tmp_path / "orig.mp4", tmp_path / "orig_30.mp4")
    copied = json.loads(target.read_text())
    assert copied["frame_index"] == 4200 and copied["pixel_to_court_homography"] == calibration["pixel_to_court_homography"] and copied["rms_reference_error_px"] == 3.6


def test_the_cli_reports_label_problems_with_exit_one(tmp_path, monkeypatch, capsys):
    collide = write_labels(tmp_path, [(1, 101, "me", "lift", "lift", "hand"), (1, 102, "opp", "net", "net", "hand")], [(1, 90, 200, True)])
    monkeypatch.setattr(halve_fps, "halve_calibration", lambda video, halved: tmp_path / "court.json")
    assert main([str(tmp_path / "v.mp4"), str(tmp_path / "v_30.mp4"), "--labels", str(collide), str(tmp_path / "x_30.csv")]) == 1
    assert "hits collide at frame 51" in capsys.readouterr().err


def test_shuttle_tracking_defaults_to_the_cpu():
    from vision.shuttle import parse_arguments
    assert parse_arguments(["v.mp4", "--start", "0", "--end", "10"]).device == "cpu"
