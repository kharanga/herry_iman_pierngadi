import pandas as pd

from vision.label import (
    LABEL_COLUMNS,
    RALLY_COLUMNS,
    SOURCE_ACCEPTED,
    SOURCE_HAND,
    handle_key,
    initial_state,
    labels_frame,
    rallies_frame,
    save_labels,
)


def starting_state():
    hits = pd.DataFrame({"rally": [1, 1, 1, 2], "frame": [77, 100, 130, 400], "hitter": ["opp", "me", "opp", "me"], "score": [float("nan"), 0.9, 0.8, 0.7], "reach": float("nan")})
    rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [77, 380], "end_frame": [200, 500], "hits": [3, 1]})
    return initial_state(hits, rallies, 1000, 60.0)


def pressed(state, keys):
    for key in keys:
        state = handle_key(state, key)
    return state


def test_proposals_skip_inferred_serves_and_the_first_rally_is_open():
    state = starting_state()
    assert [proposal.frame for proposal in state.proposals] == [100, 130, 400]
    assert state.frame == 77 and state.rally.rally == 1 and not state.dirty


def test_a_hand_label_needs_a_hitter_then_a_type_and_replaces_an_existing_one():
    state = pressed(starting_state(), ["h", "s"])
    assert state.labels == () and state.pending is not None
    state = pressed(state, ["o", "s"])
    assert [(label.frame, label.player, label.type_group, label.source, label.rally) for label in state.labels] == [(77, "opp", "serve", SOURCE_HAND, 1)]
    state = pressed(state, ["h", "m", "c"])
    assert [(label.frame, label.player, label.type_group) for label in state.labels] == [(77, "me", "clear")]
    state = pressed(state, ["u"])
    assert [(label.player, label.type_group) for label in state.labels] == [("opp", "serve")]


def test_accepting_a_proposal_jumps_to_it_and_keeps_its_hitter():
    state = pressed(starting_state(), [".", ".", "a", "n"])
    assert [(label.frame, label.player, label.type_group, label.source) for label in state.labels] == [(100, "me", "net", SOURCE_ACCEPTED)]


def test_escape_cancels_a_pending_hit_and_unknown_keys_do_nothing():
    state = pressed(starting_state(), ["h", "esc"])
    assert state.pending is None and state.labels == ()
    assert pressed(starting_state(), ["z"]) == starting_state()


def test_rally_navigation_add_and_remove_relabel_and_drop_labels_outside():
    state = pressed(starting_state(), ["]"])
    assert state.frame == 380 and state.rally.rally == 2
    state = pressed(state, ["[", "h", "o", "s", "]", "h", "m", "l"])
    assert [(label.rally, label.frame) for label in state.labels] == [(1, 77), (2, 380)]
    state = pressed(state, ["-"])
    assert [span.rally for span in state.rallies] == [1] and [(label.rally, label.frame) for label in state.labels] == [(1, 77)]
    state = pressed(state, ["+"])
    assert len(state.rallies) == 1
    state = pressed(state, [">"] * 20 + ["+"])
    assert [(span.start_frame, span.end_frame) for span in state.rallies] == [(77, 200), (277, 397)]


def test_saving_writes_both_files_and_reloading_gives_the_same_state(tmp_path):
    state = pressed(starting_state(), ["h", "o", "s", ".", ".", "a", "n"])
    saved = save_labels(state, tmp_path / "labels.csv", tmp_path / "labels_rallies.csv")
    assert not saved.dirty
    labels, rallies = pd.read_csv(tmp_path / "labels.csv"), pd.read_csv(tmp_path / "labels_rallies.csv")
    assert list(labels.columns) == LABEL_COLUMNS and list(rallies.columns) == RALLY_COLUMNS
    pd.testing.assert_frame_equal(labels, labels_frame(state))
    hits = pd.DataFrame({"rally": [1], "frame": [100], "hitter": ["me"], "score": [0.9], "reach": [float("nan")]})
    reloaded = initial_state(hits, pd.DataFrame(columns=["rally", "start_frame", "end_frame", "hits"]), 1000, 60.0, labels, rallies)
    assert reloaded.labels == state.labels and reloaded.rallies == state.rallies
    pd.testing.assert_frame_equal(rallies_frame(reloaded), rallies)
