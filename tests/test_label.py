import pandas as pd

from vision.label import (
    TYPE_KEYS,
    group_and_detail,
    labels_from_frame,
    pending_prompt,
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
    assert state.labels == () and state.pending.choosing_serve_length
    state = pressed(state, ["s"])
    assert [(label.frame, label.player, label.type_group, label.type_detail, label.source, label.rally) for label in state.labels] == [(77, "opp", "serve", "serve_short", SOURCE_HAND, 1)]
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
    state = pressed(state, ["[", "h", "o", "s", "l", "]", "h", "m", "l"])
    assert [(label.rally, label.frame) for label in state.labels] == [(1, 77), (2, 380)]
    state = pressed(state, ["-"])
    assert [span.rally for span in state.rallies] == [1] and [(label.rally, label.frame) for label in state.labels] == [(1, 77)]
    state = pressed(state, ["+"])
    assert len(state.rallies) == 1
    state = pressed(state, [">"] * 20 + ["+"])
    assert [(span.start_frame, span.end_frame) for span in state.rallies] == [(77, 200), (277, 397)]


def test_saving_writes_both_files_and_reloading_gives_the_same_state(tmp_path):
    state = pressed(starting_state(), ["h", "o", "s", "l", ".", ".", "a", "t"])
    saved = save_labels(state, tmp_path / "labels.csv", tmp_path / "labels_rallies.csv")
    assert not saved.dirty and [(label.type_group, label.type_detail) for label in state.labels] == [("serve", "serve_long"), ("net", "return_net")]
    labels, rallies = pd.read_csv(tmp_path / "labels.csv"), pd.read_csv(tmp_path / "labels_rallies.csv")
    assert list(labels.columns) == LABEL_COLUMNS and list(rallies.columns) == RALLY_COLUMNS
    pd.testing.assert_frame_equal(labels, labels_frame(state))
    hits = pd.DataFrame({"rally": [1], "frame": [100], "hitter": ["me"], "score": [0.9], "reach": [float("nan")]})
    reloaded = initial_state(hits, pd.DataFrame(columns=["rally", "start_frame", "end_frame", "hits"]), 1000, 60.0, labels, rallies)
    assert reloaded.labels == state.labels and reloaded.rallies == state.rallies
    pd.testing.assert_frame_equal(rallies_frame(reloaded), rallies)


def test_rally_labels_after_a_long_serve_keep_their_types():
    state = pressed(starting_state(), ["]", "[", "h", "o", "s", "l", "]", "h", "m", "l"])
    assert [(label.frame, label.type_group, label.type_detail) for label in state.labels] == [(77, "serve", "serve_long"), (380, "lift", "lift")]


def test_inside_the_serve_prompt_only_short_long_and_escape_count():
    asked = pressed(starting_state(), ["h", "m", "s"])
    assert asked.pending.choosing_serve_length and pending_prompt(asked.pending) == " by me: SERVE, press s for SHORT or l for LONG (esc cancels)"
    assert pressed(asked, ["c", "n", "o", "?"]).pending == asked.pending and pressed(asked, ["c"]).labels == ()
    long_serve = pressed(asked, ["l"])
    assert [(label.player, label.type_group, label.type_detail) for label in long_serve.labels] == [("me", "serve", "serve_long")]
    cancelled = pressed(asked, ["esc"])
    assert cancelled.pending is None and cancelled.labels == ()


def test_serve_before_a_hitter_does_nothing_and_every_type_key_gives_its_group():
    no_hitter = pressed(starting_state(), ["h", "s"])
    assert not no_hitter.pending.choosing_serve_length and no_hitter.labels == ()
    for key, detail in TYPE_KEYS.items():
        state = pressed(starting_state(), ["h", "o", key])
        assert [(label.type_group, label.type_detail) for label in state.labels] == [group_and_detail(detail)]
    assert group_and_detail("push") == ("drive", "push") and group_and_detail("return_net") == ("net", "return_net")
    assert group_and_detail("serve") == ("serve", "") and group_and_detail("unknown")[0] == "unknown"


def test_an_old_label_file_without_details_loads_and_saves_with_an_empty_detail(tmp_path):
    old = pd.DataFrame({"rally": [1, 1], "frame_num": [130, 77], "player": ["opp", "opp"], "type_group": ["clear", "serve"], "source": ["hand", "hand"]})
    labels = labels_from_frame(old)
    assert [(label.frame, label.type_group, label.type_detail) for label in labels] == [(77, "serve", ""), (130, "clear", "")]
    state = initial_state(pd.DataFrame(columns=["rally", "frame", "hitter", "score", "reach"]), pd.DataFrame(columns=["rally", "start_frame", "end_frame", "hits"]), 1000, 60.0, old, pd.DataFrame({"rally": [1], "start_frame": [77], "end_frame": [200]}))
    save_labels(replace_dirty(state), tmp_path / "labels.csv", tmp_path / "labels_rallies.csv")
    saved = pd.read_csv(tmp_path / "labels.csv")
    assert list(saved.columns) == LABEL_COLUMNS and saved["type_detail"].isna().all()
    assert labels_from_frame(saved) == labels


def replace_dirty(state):
    from dataclasses import replace

    return replace(state, dirty=True)
