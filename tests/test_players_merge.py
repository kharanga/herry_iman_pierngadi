import json

import cv2
import numpy as np
import pandas as pd
import pytest

from vision import players
from vision.players import (
    boxes_path,
    even_stretches,
    main,
    merge_stretch_tracks,
    merged_tables,
    positions_path,
    read_stretches,
    read_tracking_record,
    write_stretches,
    write_tracking_record,
)


def test_back_to_back_stretches_cover_every_frame_once():
    for frame_count in (1, 2, 99, 100, 101, 31800, 29711):
        for stretch_frames in (1, 7, 100, 4320, 50000):
            stretches = even_stretches(frame_count, stretch_frames)
            frames = [frame for start, end in stretches for frame in range(start, end + 1)]
            assert frames == list(range(frame_count))
            assert all(end - start + 1 <= stretch_frames for start, end in stretches)


def test_spread_stretches_have_full_length_stay_in_their_slice_and_in_the_video():
    for frame_count in range(1, 200):
        for stretch_count in range(1, 8):
            for stretch_frames in range(1, frame_count // stretch_count + 1):
                stretches = even_stretches(frame_count, stretch_frames, stretch_count)
                assert len(stretches) == stretch_count
                assert all(end - start + 1 == stretch_frames for start, end in stretches)
                assert stretches[0][0] >= 0 and stretches[-1][1] <= frame_count - 1
                assert all(first[1] < second[0] for first, second in zip(stretches, stretches[1:]))


def test_bad_plans_raise():
    for arguments in [(0, 10), (10, 0), (100, 10, 0), (100, 30, 4)]:
        with pytest.raises(ValueError):
            even_stretches(*arguments)


def test_written_stretches_read_back(tmp_path):
    stretches = even_stretches(1000, 300)
    saved = write_stretches(stretches, tmp_path / "deep" / "plan.csv")
    assert read_stretches(saved) == stretches


def test_merged_tables_keep_the_first_row_of_a_frame_and_player_and_sort():
    first = pd.DataFrame({"frame": [10, 11, 11], "player": ["opp", "me", "opp"], "x_m": [1.0, 2.0, 3.0]})
    second = pd.DataFrame({"frame": [11, 12, 0], "player": ["me", "me", "me"], "x_m": [9.0, 4.0, 5.0]})
    merged = merged_tables([first, second])
    assert merged[["frame", "player", "x_m"]].values.tolist() == [[0, "me", 5.0], [10, "opp", 1.0], [11, "me", 2.0], [11, "opp", 3.0], [12, "me", 4.0]]
    assert list(merged.index) == list(range(5))


def fake_stretch(video, start, end, players_dir, stride=1):
    frames = np.arange(start, end + 1, stride)
    positions = pd.DataFrame({"frame": np.repeat(frames, 2), "player": ["me", "opp"] * len(frames), "x_m": 1.0, "y_m": 2.0})
    boxes = positions.assign(x1=0, y1=0, x2=10, y2=10)
    positions_file = positions_path(video, start, end, players_dir)
    positions_file.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(positions_file, index=False)
    boxes.to_csv(boxes_path(video, start, end, players_dir), index=False)
    write_tracking_record(positions_file, {"frame_stride": stride, "video_fps": 24.0, "effective_fps": 24.0 / stride})


def test_merge_writes_whole_video_files_and_a_record_listing_the_stretches(tmp_path):
    video = tmp_path / "v.mp4"
    stretches = [(0, 9), (10, 19), (20, 24)]
    for start, end in stretches:
        fake_stretch(video, start, end, tmp_path)
    merged_positions, merged_boxes = merge_stretch_tracks(video, stretches, 24, tmp_path)
    assert merged_positions == positions_path(video, 0, 24, tmp_path) and merged_boxes == boxes_path(video, 0, 24, tmp_path)
    positions = pd.read_csv(merged_positions)
    assert positions["frame"].tolist() == [frame for frame in range(25) for _ in range(2)]
    assert len(pd.read_csv(merged_boxes)) == 50
    record = read_tracking_record(merged_positions)
    assert record["stretches"] == [[0, 9], [10, 19], [20, 24]] and record["frame_stride"] == 1


def test_merge_with_an_untracked_stretch_writes_nothing(tmp_path):
    video = tmp_path / "v.mp4"
    fake_stretch(video, 0, 9, tmp_path)
    before = sorted(path.name for path in positions_path(video, 0, 9, tmp_path).parent.iterdir())
    with pytest.raises(ValueError, match="10-19"):
        merge_stretch_tracks(video, [(0, 9), (10, 19)], 19, tmp_path)
    assert sorted(path.name for path in positions_path(video, 0, 9, tmp_path).parent.iterdir()) == before


def test_merge_refuses_stretches_tracked_at_different_strides(tmp_path):
    video = tmp_path / "v.mp4"
    fake_stretch(video, 0, 9, tmp_path, stride=1)
    fake_stretch(video, 10, 19, tmp_path, stride=3)
    with pytest.raises(ValueError):
        merge_stretch_tracks(video, [(0, 9), (10, 19)], 19, tmp_path)
    assert not positions_path(video, 0, 19, tmp_path).exists()


def synthetic_video(path, frame_count, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    return path


def test_plan_stretches_cli_writes_the_plan_and_tracks_nothing(tmp_path, capsys, monkeypatch):
    video = synthetic_video(tmp_path / "v.mp4", 50, 10.0)
    monkeypatch.setattr(players, "track_stretches", lambda *arguments, **keywords: pytest.fail("tracked"))
    assert main([str(video), "--plan-stretches", str(tmp_path / "plan.csv"), "--stretch-seconds", "2"]) == 0
    assert read_stretches(tmp_path / "plan.csv") == [(0, 19), (20, 39), (40, 49)]
    assert main([str(video), "--plan-stretches", str(tmp_path / "spread.csv"), "--stretch-seconds", "1", "--stretch-count", "2"]) == 0
    assert read_stretches(tmp_path / "spread.csv") == [(8, 17), (32, 41)]
    capsys.readouterr()
    assert main([str(video), "--plan-stretches", str(tmp_path / "bad.csv"), "--stretch-seconds", "0"]) == 1
    assert "error:" in capsys.readouterr().err and not (tmp_path / "bad.csv").exists()
    assert main([str(tmp_path / "missing.mp4"), "--plan-stretches", str(tmp_path / "none.csv")]) == 1
    assert not (tmp_path / "none.csv").exists()


def test_stretches_cli_with_merge_tracks_then_merges(tmp_path, capsys, monkeypatch):
    video = synthetic_video(tmp_path / "v.mp4", 30, 10.0)
    plan = write_stretches([(0, 14), (15, 29)], tmp_path / "plan.csv")

    def pretend_to_track(video_path, stretches, *arguments):
        players_dir = arguments[-1]
        for start, end in stretches:
            fake_stretch(video_path, start, end, players_dir)
        return len(stretches), 0, 30

    monkeypatch.setattr(players, "track_stretches", pretend_to_track)
    assert main([str(video), "--stretches", str(plan), "--merge", "--players-dir", str(tmp_path / "players")]) == 0
    output = capsys.readouterr().out
    assert output.index("2 stretches: 2 tracked") < output.index("saved")
    assert pd.read_csv(positions_path(video, 0, 29, tmp_path / "players"))["frame"].nunique() == 30


def test_merge_without_stretches_is_an_error_and_tracks_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(players, "track_players", lambda *arguments, **keywords: pytest.fail("tracked"), raising=False)
    assert main([str(tmp_path / "v.mp4"), "--merge", "--start", "0", "--end", "10"]) == 2
    assert "--stretches" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
