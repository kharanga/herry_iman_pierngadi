import pandas as pd

from vision.hit_training import window_text
from vision.hits import MATCH_TOLERANCE_SECONDS, match_hits, match_tolerances, tolerance_frames, tolerance_label


def test_the_two_windows_are_defined_once_in_seconds():
    assert MATCH_TOLERANCE_SECONDS == (0.25, 0.50)


def test_windows_become_frames_at_each_videos_own_rate_rounded_half_up_and_at_least_one():
    assert [tolerance_frames(0.25, fps) for fps in (24, 25, 30, 60)] == [6, 6, 8, 15]
    assert [tolerance_frames(0.5, fps) for fps in (24, 25, 30, 60)] == [12, 13, 15, 30]
    assert tolerance_frames(0.25, 29.97) == 7 and tolerance_frames(0.001, 30) == 1 and tolerance_frames(0.0, 30) == 1
    assert match_tolerances(25.0) == [(0.25, 6), (0.5, 13)]


def test_labels_state_seconds_frames_and_fps():
    assert tolerance_label(0.25, 30) == "±0.25 s (±8 frames at 30 fps)"
    assert tolerance_label(0.5, 23.976) == "±0.5 s (±12 frames at 23.976 fps)"
    rows = pd.DataFrame({"fps": [30.0, 25.0, 30.0], "tolerance_frames": [8, 6, 8]})
    assert window_text(0.25, rows) == "±0.25 s (±6 frames at 25 fps, ±8 frames at 30 fps)"


def test_the_same_time_offset_matches_at_every_rate():
    for fps in (24, 25, 30, 60):
        offset = int(0.24 * fps)
        assert len(match_hits([100 + offset], [100], tolerance_frames(0.25, fps))) == 1
        assert len(match_hits([100 + tolerance_frames(0.25, fps) + 1], [100], tolerance_frames(0.25, fps))) == 0
