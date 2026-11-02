import shutil
from dataclasses import replace

import pandas as pd
import pytest

from vision.label import (
    HitLabel,
    LabelState,
    RallySpan,
    handle_key,
    labels_frame,
    labels_from_frame,
    rejected_path_for,
    review_counts,
    review_target,
    reviewed_source,
    save_labels,
)
from vision.prelabel import main as prelabel_main, model_labels

FPS = 30.0


def model(frame, detail="lift", group="lift", player="opp", rally=1):
    return HitLabel(rally, frame, player, group, "model", detail, score=0.8, model_frame=frame, model_type_detail=detail)


def review_state(labels):
    return LabelState(tuple(labels), (RallySpan(1, 0, 1000, True, True),), (), 0, 100, 1000, FPS, review=True)


def at(state, frame, *keys):
    state = replace(state, frame=frame)
    for key in keys:
        state = handle_key(state, key)
    return state


def test_moved_beats_corrected_and_an_unchanged_label_is_confirmed():
    label = model(100)
    assert reviewed_source(label) == "confirmed"
    assert reviewed_source(replace(label, type_detail="drive", type_group="drive")) == "corrected_type"
    assert reviewed_source(replace(label, frame=103, type_detail="drive", type_group="drive")) == "moved"


def test_the_review_target_is_the_nearest_model_label_within_half_a_second():
    hand = HitLabel(1, 101, "me", "net", "hand", "net")
    state = review_state([model(100), hand, model(140)])
    assert review_target(replace(state, frame=105)).frame == 100
    assert review_target(replace(state, frame=125)).frame == 140
    assert review_target(replace(state, frame=200)) is None
    assert review_target(review_state([hand])) is None


def test_confirm_retype_serve_length_move_and_reject_each_record_their_source():
    state = review_state([model(100), model(200, "clear", "clear"), model(300), model(400), model(500, "serve_short", "serve")])
    state = at(state, 100, "y")
    state = at(state, 200, "d")
    state = at(state, 503, "s", "l")
    state = at(state, 306, "g")
    state = at(state, 400, "x")
    by_frame = {label.frame: label for label in state.labels}
    assert (by_frame[100].source, by_frame[200].source, by_frame[200].type_detail, by_frame[306].source, by_frame[500].type_detail, by_frame[500].source) == ("confirmed", "corrected_type", "drop", "moved", "serve_long", "corrected_type")
    assert by_frame[306].model_frame == 300 and 400 not in by_frame
    assert [(label.frame, label.source) for label in state.rejected] == [(400, "rejected")]
    assert review_counts(state) == {"unreviewed": 0, "confirmed": 1, "corrected_type": 2, "corrected_hitter": 0, "moved": 1, "rejected": 1, "added": 0}


def test_undo_brings_a_rejected_label_back_and_x_on_a_hand_label_deletes_it():
    state = at(review_state([model(100)]), 100, "x")
    assert state.labels == () and len(state.rejected) == 1
    undone = handle_key(state, "u")
    assert [label.frame for label in undone.labels] == [100] and undone.rejected == ()
    hand_state = at(review_state([HitLabel(1, 200, "me", "net", "hand", "net")]), 200, "x")
    assert hand_state.labels == () and hand_state.rejected == ()


def test_h_still_adds_a_hand_hit_in_review_mode():
    state = at(review_state([model(100)]), 600, "h", "m", "c")
    added = [label for label in state.labels if label.frame == 600]
    assert added and added[0].source == "hand" and not added[0].from_model
    assert review_counts(state)["added"] == 1


def test_saving_writes_review_columns_and_a_rejected_file_only_when_needed(tmp_path):
    reviewed = at(review_state([model(100), model(200)]), 200, "x")
    labels_file = tmp_path / "game.csv"
    save_labels(reviewed, labels_file, tmp_path / "game_rallies.csv")
    saved = pd.read_csv(labels_file)
    assert {"score", "model_frame", "model_type_detail"} <= set(saved.columns)
    assert pd.read_csv(rejected_path_for(labels_file))["frame_num"].tolist() == [200]
    assert [label.frame for label in labels_from_frame(saved)] == [100]
    plain = LabelState((HitLabel(1, 50, "me", "net", "hand", "net"),), (RallySpan(1, 0, 100, True, True),), (), 0, 50, 100, FPS)
    plain_file = tmp_path / "plain.csv"
    save_labels(plain, plain_file, tmp_path / "plain_rallies.csv")
    assert list(pd.read_csv(plain_file).columns) == ["rally", "frame_num", "player", "type_group", "type_detail", "source"]
    assert not rejected_path_for(plain_file).exists()


def test_an_old_labels_file_round_trips_byte_for_byte(tmp_path):
    source = "data/labels/stand_in_set1.csv"
    copy = tmp_path / "set1.csv"
    shutil.copyfile(source, copy)
    shutil.copyfile("data/labels/stand_in_set1_rallies.csv", tmp_path / "set1_rallies.csv")
    labels = labels_from_frame(pd.read_csv(copy))
    rallies = pd.read_csv(tmp_path / "set1_rallies.csv")
    from vision.label import rallies_from_frame
    state = LabelState(labels, rallies_from_frame(rallies), (), 0, 0, 30000, 60.0)
    save_labels(replace(state, dirty=True), copy, tmp_path / "set1_rallies.csv")
    assert copy.read_bytes() == open(source, "rb").read()
    assert not rejected_path_for(copy).exists()


def test_prelabel_types_known_hits_marks_the_rest_unknown_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    hits = pd.DataFrame({"rally": [1, 1, 1, 1], "frame": [10, 20, 20, 30], "hitter": ["me", "opp", "opp", "?"], "score": [0.9, 0.8, 0.8, 0.7], "reach": [None] * 4})
    typed = pd.DataFrame({"frame_num": [10], "type_pred": ["serve"], "type_detail_pred": ["serve_short"], "type_conf": [0.66]})
    labels = model_labels(hits, typed)
    assert labels[["frame_num", "type_group", "type_detail", "source", "model_frame", "model_type_detail"]].values.tolist() == [[10, "serve", "serve_short", "model", 10, "serve_short"], [20, "unknown", "unknown", "model", 20, "unknown"]]
    assert labels["score"].iloc[0] == pytest.approx(0.66) and pd.isna(labels["score"].iloc[1])
    existing = tmp_path / "game.csv"
    existing.write_text("keep me")
    assert prelabel_main(["v.mp4", "--hits", str(tmp_path / "hits_0_9.csv"), "--database", str(tmp_path / "db.duckdb"), "--match-id", "m", "--labels", str(existing)]) != 0
    assert existing.read_text() == "keep me"


def test_m_and_o_fix_the_hitter_of_the_model_label_in_reach_and_keep_the_models_hitter():
    state = review_state([model(100, player="opp"), model(400, player="me")])
    fixed = at(state, 103, "m")
    label = next(label for label in fixed.labels if label.frame == 100)
    assert (label.player, label.model_player, label.source) == ("me", "opp", "corrected_hitter")
    twice = at(fixed, 103, "o")
    back = next(label for label in twice.labels if label.frame == 100)
    assert (back.player, back.model_player, back.source) == ("opp", "opp", "confirmed")
    nothing_in_reach = at(state, 250, "m")
    assert nothing_in_reach.labels == state.labels


def test_sources_follow_moved_then_type_then_hitter_and_counts_record_every_fix():
    state = review_state([model(100, player="opp"), model(200, player="opp"), model(300, player="opp")])
    state = at(state, 100, "m")
    state = at(state, 200, "m", "v")
    state = at(state, 300, "m")
    state = at(state, 305, "g")
    by_frame = {label.frame: label for label in state.labels}
    assert (by_frame[100].source, by_frame[200].source, by_frame[305].source) == ("corrected_hitter", "corrected_type", "moved")
    assert all(label.model_player == "opp" for label in state.labels)
    counts = review_counts(state)
    assert (counts["corrected_hitter"], counts["corrected_type"], counts["moved"], counts["confirmed"]) == (3, 1, 1, 0)


def test_h_then_m_still_sets_a_pending_hits_hitter_in_review_mode():
    state = at(review_state([model(100)]), 600, "h", "m")
    assert state.pending is not None and state.pending.player == "me"
    assert [label.player for label in state.labels] == ["opp"]


def test_a_review_file_without_model_player_takes_each_model_labels_own_hitter(tmp_path):
    table = pd.DataFrame({"rally": [1, 1], "frame_num": [100, 200], "player": ["opp", "me"], "type_group": ["lift", "net"], "type_detail": ["lift", "net"], "source": ["confirmed", "hand"], "score": [0.8, None], "model_frame": [100, None], "model_type_detail": ["lift", ""]})
    labels = labels_from_frame(table)
    assert [(label.frame, label.model_player) for label in labels] == [(100, "opp"), (200, "")]
    reviewed = replace(review_state(labels), dirty=True)
    save_labels(reviewed, tmp_path / "r.csv", tmp_path / "r_rallies.csv")
    saved = pd.read_csv(tmp_path / "r.csv")
    assert "model_player" in saved.columns and saved["model_player"].fillna("").tolist() == ["opp", ""]
