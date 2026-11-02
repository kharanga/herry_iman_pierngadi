from dataclasses import replace

import pandas as pd
import pytest

from vision.label import (
    RallySpan,
    detector_tables,
    handle_key,
    initial_state,
    labels_frame,
    main,
    rallies_frame,
    segment_seconds_of,
    segment_spans,
    status_line,
)
from vision.label_ui import apply_action, state_json

LAST_FRAME = 3599
FPS = 24.0


def blank_state(segment_seconds=60):
    no_hits, no_rallies = detector_tables(pd.io.common.Path("never.mp4"), LAST_FRAME, blank=True)
    return initial_state(no_hits, no_rallies, LAST_FRAME, FPS, segment_seconds=segment_seconds, blank=True)


def pressed(state, keys):
    for key in keys:
        state = handle_key(state, key)
    return state


def at(state, frame):
    return apply_action(state, {"type": "jump", "frame": frame})


def spans(state):
    return [(span.rally, span.start_frame, span.end_frame) for span in state.rallies]


def test_segments_cover_every_frame_once_back_to_back():
    for last_frame, fps, seconds in [(3599, 24.0, 60), (3600, 24.0, 60), (0, 30.0, 60), (100, 30.0, 0.01), (89999, 29.97, 60)]:
        segments = segment_spans(last_frame, fps, seconds)
        frames = [frame for segment in segments for frame in range(segment.start_frame, segment.end_frame + 1)]
        assert frames == list(range(last_frame + 1))
        assert [segment.rally for segment in segments] == list(range(1, len(segments) + 1))


def test_segment_seconds_default_only_with_blank_and_must_be_positive():
    assert segment_seconds_of(True, None) == 60.0
    assert segment_seconds_of(False, None) is None
    assert segment_seconds_of(True, 15) == 15
    for bad in (0, -5):
        with pytest.raises(ValueError):
            segment_seconds_of(True, bad)


def test_cli_rejects_a_non_positive_segment_length_before_opening_anything(tmp_path, capsys):
    assert main(["never.mp4", "--match-id", "x", "--blank", "--segment-seconds", "0", "--labels", str(tmp_path / "x.csv")]) == 1
    assert "error:" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_blank_tables_need_no_hits_file_and_without_blank_a_missing_file_is_an_error(tmp_path):
    hits, rallies = detector_tables(tmp_path / "v.mp4", LAST_FRAME, tmp_path, blank=True)
    assert len(hits) == 0 and len(rallies) == 0
    with pytest.raises(FileNotFoundError):
        detector_tables(tmp_path / "v.mp4", LAST_FRAME, tmp_path, blank=False)


def test_stepping_and_jumping_keep_the_segment_with_the_frame():
    state = blank_state()
    assert (state.frame, state.segment_index) == (0, 0)
    state = pressed(state, [">"] * 144)
    assert (state.frame, state.segment_index) == (1440, 1)
    state = pressed(state, [","])
    assert (state.frame, state.segment_index) == (1439, 0)
    state = at(state, 3000)
    assert state.segment_index == 2 and state.view == RallySpan(3, 2880, 3599)
    assert pressed(state, ["]"]).segment_index == 2
    assert pressed(state, ["[", "[", "["]).segment_index == 0
    assert pressed(state, ["<"] * 400).frame == 0


def test_b_and_e_build_rallies_that_never_overlap_and_can_cross_segments():
    state = blank_state()
    state = pressed(at(state, 1400), ["b"])
    assert spans(state) == [(1, 1400, LAST_FRAME)]
    state = pressed(at(state, 1500), ["e"])
    assert spans(state) == [(1, 1400, 1500)]
    state = pressed(at(state, 2000), ["b"])
    state = pressed(at(state, 100), ["b"])
    assert spans(state) == [(1, 100, 1399), (2, 1400, 1500), (3, 2000, LAST_FRAME)]
    state = pressed(at(state, 200), ["e"])
    state = pressed(at(state, 1700), ["e"])
    assert spans(state) == [(1, 100, 200), (2, 1400, 1700), (3, 2000, LAST_FRAME)]
    state = pressed(at(state, 1450), ["b"])
    assert spans(state) == [(1, 100, 200), (2, 1450, 1700), (3, 2000, LAST_FRAME)]
    for first, second in zip(state.rallies, state.rallies[1:]):
        assert first.end_frame < second.start_frame


def test_e_before_any_rally_and_r_or_minus_outside_every_rally_do_nothing():
    state = blank_state()
    assert pressed(at(state, 500), ["e"]).rallies == ()
    state = pressed(at(state, 1000), ["b"])
    state = pressed(at(state, 1100), ["e"])
    outside = at(state, 3000)
    assert pressed(outside, ["-"]).rallies == outside.rallies
    assert pressed(outside, ["r"]).rallies == outside.rallies
    assert pressed(at(state, 900), ["e"]).rallies == state.rallies


def test_hits_take_the_rally_holding_them_and_a_hit_outside_every_rally_is_refused():
    state = blank_state()
    state = pressed(at(state, 1440), ["b", "h", "m", "s", "s"])
    state = pressed(at(state, 1470), ["h", "o", "n"])
    state = pressed(at(state, 1560), ["e"])
    outside = pressed(at(state, 3000), ["h"])
    assert outside.pending is None and outside.labels == state.labels and outside.notice
    assert pressed(outside, ["."]).notice == ""
    assert [(label.rally, label.frame, label.type_group, label.type_detail) for label in state.labels] == [(1, 1440, "serve", "serve_short"), (1, 1470, "net", "net")]
    orphaning = pressed(at(state, 1450), ["r"])
    assert orphaning.rallies == state.rallies and "1440" in orphaning.notice


def test_undo_reverses_b_and_e_one_at_a_time():
    state = blank_state()
    first = pressed(at(state, 1000), ["b"])
    second = pressed(at(first, 1200), ["e"])
    assert pressed(second, ["u"]).rallies == first.rallies
    assert pressed(second, ["u", "u"]).rallies == ()


def test_saved_files_hold_video_frames_and_no_segments(tmp_path):
    state = pressed(at(blank_state(), 2000), ["b", "h", "m", "s", "l"])
    state = pressed(at(state, 2120), ["e"])
    assert labels_frame(state).to_dict("records") == [
        {"rally": 1, "frame_num": 2000, "player": "me", "type_group": "serve", "type_detail": "serve_long", "source": "hand"}
    ]
    assert rallies_frame(state).to_dict("records") == [{"rally": 1, "start_frame": 2000, "end_frame": 2120, "bounds_checked": True}]
    reloaded = initial_state(
        *detector_tables(pd.io.common.Path("never.mp4"), LAST_FRAME, blank=True), LAST_FRAME, FPS, labels_frame(state), rallies_frame(state), segment_seconds=60, blank=True
    )
    assert reloaded.labels == state.labels and reloaded.rallies == state.rallies and reloaded.segment_index == 0


def test_state_json_reports_segments_and_the_rallies_in_view():
    state = pressed(at(blank_state(), 1400), ["b"])
    state = pressed(at(state, 1500), ["e"])
    state = pressed(at(state, 3000), ["b"])
    shown = state_json(at(state, 1450))
    assert shown["blank"] is True and shown["segment_count"] == 3
    assert shown["rally"] == {"rally": 2, "start_frame": 1440, "end_frame": 2879}
    assert shown["user_rallies"] == [[1, 1400, 1500, True]]
    assert shown["user_rally"] == [1, 1400, 1500]
    assert state_json(at(state, 2000))["user_rally"] is None
    assert "segment 2 of 3" in status_line(at(state, 1450))


def test_every_action_leaves_the_frame_inside_the_current_segment():
    state = pressed(at(blank_state(), 100), ["b"])
    state = pressed(at(state, 200), ["e"])
    state = pressed(at(state, 3000), ["b"])
    state = pressed(at(state, 3100), ["e"])
    for keys in (["-"], ["u"], ["x"], ["+"], ["r"], ["b"], ["e"]):
        for frame in (150, 1500, 3050):
            after = pressed(at(state, frame), keys)
            assert after.view.start_frame <= after.frame <= after.view.end_frame, (keys, frame, after.frame, after.view)


def test_without_segments_the_view_is_the_rally_as_before():
    hits = pd.DataFrame({"frame": [110, 150], "hitter": ["me", "opp"], "score": [0.9, 0.8]})
    rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [200]})
    state = initial_state(hits, rallies, 1000, 30.0)
    assert state.segments == () and state.view == state.rally == RallySpan(1, 100, 200)
    assert state_json(state)["user_rallies"] == []


def test_the_rally_action_moves_the_segment_with_the_frame():
    state = pressed(at(blank_state(), 3000), ["b"])
    state = pressed(at(state, 3100), ["e"])
    after = apply_action(at(state, 100), {"type": "rally", "index": 0})
    assert after.frame == 3000 and after.view == RallySpan(3, 2880, 3599)


def test_minus_keeps_the_frame_in_segment_mode_and_jumps_without_segments():
    state = pressed(at(blank_state(), 100), ["b"])
    state = pressed(at(state, 200), ["e"])
    state = pressed(at(state, 3000), ["b"])
    after = pressed(at(state, 150), ["-"])
    assert after.frame == 150 and spans(after) == [(1, 3000, LAST_FRAME)]
    hits = pd.DataFrame({"frame": [110], "hitter": ["me"], "score": [0.9]})
    rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [100, 500], "end_frame": [200, 600]})
    plain = initial_state(hits, rallies, 1000, 30.0)
    assert pressed(plain, ["-"]).frame == 500


def test_the_page_marks_only_the_detector_buttons_to_hide_in_blank_mode():
    import re
    from vision.label_ui import PAGE_HTML
    hidden_keys = re.findall(r'<button data-key="([^"]+)" class="detector">', PAGE_HTML)
    assert sorted(hidden_keys) == ["+", "a", "r"]
    assert 'querySelectorAll("button.detector").forEach(b=>b.hidden=s.blank)' in PAGE_HTML
