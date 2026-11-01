from dataclasses import replace

import numpy as np
import pandas as pd

from vision.label import (
    RallySpan,
    detail_for,
    detector_tables,
    handle_key,
    initial_state,
    labels_frame,
    labels_from_frame,
    rallies_frame,
    rallies_from_frame,
    status_line,
    type_text,
    unfinished_counts,
)
from vision.label_ui import state_json

LAST_FRAME = 3599
FPS = 24.0


def old_files(label_rows, rally_rows):
    labels = pd.DataFrame(label_rows, columns=["rally", "frame_num", "player", "type_group", "source"])
    rallies = pd.DataFrame(rally_rows, columns=["rally", "start_frame", "end_frame"])
    return labels, rallies


def relabel_state(labels, rallies):
    no_hits, no_rallies = detector_tables(pd.io.common.Path("never.mp4"), LAST_FRAME, blank=True)
    return initial_state(no_hits, no_rallies, LAST_FRAME, FPS, labels, rallies, segment_seconds=60, blank=True)


def at(state, frame, *keys):
    state = replace(state, frame=frame)
    for key in keys:
        state = handle_key(state, key)
    return state


def spans(state):
    return [(span.rally, span.start_frame, span.end_frame, span.bounds_checked) for span in state.rallies]


def frames(state):
    return [label.frame for label in state.labels]


def test_old_details_follow_the_eight_labels():
    assert detail_for("net", "return_net") == "net" and detail_for("drive", "push") == "drive"
    assert detail_for("clear", "") == "clear" and detail_for("clear", float("nan")) == "clear"
    assert detail_for("serve", "") == "" and detail_for("serve", "serve_long") == "serve_long"
    assert detail_for("unknown", "") == "unknown" and detail_for("flick", "") == ""
    table = pd.DataFrame({"rally": [1, 1, 1], "frame_num": [10, 20, 30], "player": ["me", "opp", "me"], "type_group": ["serve", "net", "drive"], "type_detail": [np.nan, "return_net", "push"], "source": ["hand"] * 3})
    assert [(label.type_group, label.type_detail) for label in labels_from_frame(table)] == [("serve", ""), ("net", "net"), ("drive", "drive")]


def test_serves_without_a_length_show_as_open_are_counted_and_a_serve_key_fills_them():
    labels, rallies = old_files([(1, 100, "me", "serve", "hand"), (1, 130, "opp", "net", "hand"), (2, 400, "opp", "serve", "hand")], [(1, 100, 300), (2, 400, 600)])
    state = relabel_state(labels, rallies)
    assert [type_text(label) for label in state.labels] == ["serve ?", "net", "serve ?"]
    assert unfinished_counts(state) == {"serves_without_length": 2, "unchecked_rallies": 2}
    filled = at(state, 100, "h", "m", "s", "l")
    assert [(label.frame, label.type_detail) for label in filled.labels] == [(100, "serve_long"), (130, "net"), (400, "")]
    assert unfinished_counts(filled)["serves_without_length"] == 1
    shown = state_json(at(state, 120))
    assert [label["type_text"] for label in shown["labels"]] == ["serve ?", "net", "serve ?"]
    assert shown["progress"]["serves_without_length"] == 2 and shown["progress"]["unchecked_rallies"] == 2
    assert [row[3] for row in shown["user_rallies"]] == [False, False]


def test_a_rally_is_checked_only_after_both_b_and_e_and_the_flag_survives_a_save():
    labels, rallies = old_files([(1, 100, "me", "serve", "hand")], [(1, 100, 300)])
    state = relabel_state(labels, rallies)
    half = at(state, 95, "b")
    assert spans(half) == [(1, 95, 300, False)] and rallies_frame(half)["bounds_checked"].tolist() == [False]
    both = at(half, 280, "e")
    assert spans(both) == [(1, 95, 280, True)] and unfinished_counts(both)["unchecked_rallies"] == 0
    reloaded = rallies_from_frame(rallies_frame(both))
    assert [span.bounds_checked for span in reloaded] == [True]
    assert spans(at(both, 270, "r")) == [(1, 95, 270, False)]
    assert spans(at(both, 270, "u")) == spans(half)


def test_checked_flags_read_text_and_booleans_and_a_missing_column_is_unchecked():
    table = pd.DataFrame({"rally": [1, 2, 3, 4, 5], "start_frame": [0, 10, 20, 30, 40], "end_frame": [5, 15, 25, 35, 45], "bounds_checked": ["True", "true", "1", "False", np.nan]})
    assert [span.bounds_checked for span in rallies_from_frame(table)] == [True, True, True, False, False]
    assert [span.bounds_checked for span in rallies_from_frame(table.drop(columns="bounds_checked"))] == [False] * 5
    assert [span.bounds_checked for span in rallies_from_frame(table.assign(bounds_checked=[True, False, True, False, True]))] == [True, False, True, False, True]


def test_relabelling_back_to_back_old_rallies_moves_bounds_and_keeps_every_hit():
    labels, rallies = old_files(
        [(1, 170, "me", "serve", "hand"), (1, 250, "opp", "clear", "hand"), (1, 480, "me", "drop", "hand"), (2, 560, "opp", "serve", "hand"), (2, 900, "me", "lift", "hand")],
        [(1, 100, 540), (2, 541, 1000)],
    )
    state = relabel_state(labels, rallies)
    state = at(state, 169, "b")
    state = at(state, 533, "e")
    assert spans(state) == [(1, 169, 533, True), (2, 541, 1000, False)]
    state = at(state, 553, "b")
    assert spans(state) == [(1, 169, 533, True), (2, 553, 1000, False)]
    state = at(state, 980, "e")
    assert spans(state) == [(1, 169, 533, True), (2, 553, 980, True)]
    assert frames(state) == [170, 250, 480, 560, 900]
    assert [label.rally for label in state.labels] == [1, 1, 1, 2, 2]


def test_b_and_e_inside_a_merged_old_rally_split_it_without_dropping_hits():
    labels, rallies = old_files([(1, 110, "me", "serve", "hand"), (1, 200, "opp", "clear", "hand"), (1, 400, "opp", "serve", "hand"), (1, 480, "me", "net", "hand")], [(1, 100, 600)])
    state = relabel_state(labels, rallies)
    split_by_e = at(state, 260, "e")
    assert spans(split_by_e) == [(1, 100, 260, False), (2, 261, 600, False)] and frames(split_by_e) == [110, 200, 400, 480]
    split_by_b = at(state, 395, "b")
    assert spans(split_by_b) == [(1, 100, 394, False), (2, 395, 600, False)] and frames(split_by_b) == [110, 200, 400, 480]
    assert [label.rally for label in split_by_b.labels] == [1, 1, 2, 2]


def test_b_shortly_before_an_unchecked_rally_moves_its_start_back_and_far_before_starts_a_new_one():
    labels, rallies = old_files([(1, 200, "me", "serve", "hand")], [(1, 200, 400)])
    state = relabel_state(labels, rallies)
    assert spans(at(state, 200 - 72, "b")) == [(1, 128, 400, False)]
    assert spans(at(state, 200 - 73, "b")) == [(1, 127, 199, False), (2, 200, 400, False)]


def test_random_b_and_e_presses_on_old_rallies_never_drop_a_hit_or_overlap():
    rng = np.random.default_rng(7)
    for _ in range(150):
        cuts = np.sort(rng.choice(np.arange(50, LAST_FRAME - 50), size=rng.integers(2, 9), replace=False))
        rally_rows = [(index + 1, int(start), int(end) - 1) for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]))]
        label_rows = []
        for rally, start, end in rally_rows:
            for frame in sorted(set(rng.integers(start, end + 1, size=rng.integers(0, 5)).tolist())):
                label_rows.append((rally, int(frame), "me", "clear", "hand"))
        state = relabel_state(*old_files(label_rows, rally_rows))
        hit_frames = frames(state)
        for _ in range(12):
            state = at(state, int(rng.integers(cuts[0] - 40, cuts[-1] + 40)), str(rng.choice(["b", "e"])))
            assert frames(state) == hit_frames
            ordered = sorted(state.rallies, key=lambda span: span.start_frame)
            assert all(span.start_frame <= span.end_frame for span in ordered)
            assert all(first.end_frame < second.start_frame for first, second in zip(ordered, ordered[1:]))
            assert all(any(span.start_frame <= frame <= span.end_frame for span in state.rallies) for frame in hit_frames)


def test_the_status_line_counts_what_is_still_open_only_with_segments():
    labels, rallies = old_files([(1, 100, "me", "serve", "hand")], [(1, 100, 300)])
    with_segments = status_line(relabel_state(labels, rallies))
    assert "1 serves need short/long, 1 rallies unchecked" in with_segments
    plain = initial_state(pd.DataFrame(columns=["rally", "frame", "hitter", "score", "reach"]), rallies, LAST_FRAME, FPS, labels, rallies)
    assert "unchecked" not in status_line(plain) and "1 serves need short/long" in status_line(plain)


def test_e_never_hands_a_checked_start_to_the_previous_rally():
    labels, rallies = old_files([(1, 100, "me", "serve", "hand"), (2, 400, "opp", "serve", "hand")], [(1, 90, 299), (2, 300, 500)])
    state = relabel_state(labels, rallies)
    marked = at(state, 300, "b")
    assert [(span.start_frame, span.end_frame, span.start_marked) for span in marked.rallies] == [(90, 299, False), (300, 500, True)]
    ended = at(marked, 320, "e")
    assert spans(ended) == [(1, 90, 299, False), (2, 300, 320, True), (3, 321, 500, False)]
    assert frames(ended) == [100, 400] and [label.rally for label in ended.labels] == [1, 3]
    handed = at(state, 320, "e")
    assert spans(handed) == [(1, 90, 320, False), (2, 321, 500, False)]
    assert handed.rallies[0].end_marked and not handed.rallies[1].start_marked
