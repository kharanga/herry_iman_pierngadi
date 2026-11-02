from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from vision.label import HitLabel, LabelState, RallySpan, handle_key, lost_labels, reviewed_rows_only

FPS = 30.0


def model(frame, rally=1):
    return HitLabel(rally, frame, "opp", "lift", "model", "lift", score=0.8, model_frame=frame, model_type_detail="lift")


def state_with(labels, rallies=((1, 100, 300),), review=True):
    return LabelState(tuple(labels), tuple(RallySpan(r, s, e, True, True) for r, s, e in rallies), (), 0, 150, 1000, FPS, review=review, segments=(RallySpan(1, 0, 1000),))


def at(state, frame, *keys):
    state = replace(state, frame=frame)
    for key in keys:
        state = handle_key(state, key)
    return state


def unchanged(before, after):
    return after.labels == before.labels and after.rallies == before.rallies and after.history == before.history and after.rejected == before.rejected


def test_g_outside_every_rally_is_refused_and_a_move_inside_takes_the_landing_rally():
    state = state_with([model(290)], rallies=((1, 100, 300), (2, 301, 500)))
    outside_state = state_with([model(290)], rallies=((1, 100, 300),))
    outside = at(outside_state, 305, "g")
    assert unchanged(outside_state, outside) and outside.notice
    moved = at(state, 302, "g")
    assert [(label.frame, label.rally, label.source) for label in moved.labels] == [(302, 2, "moved")]


def test_e_inside_a_rally_still_splits_around_its_hits_and_loses_none():
    state = state_with([model(150), model(250)], review=False)
    split = at(state, 200, "e")
    assert [(span.start_frame, span.end_frame) for span in split.rallies] == [(100, 200), (201, 300)] and len(split.labels) == 2
    assert lost_labels(split.labels, split.rallies) == []


def test_the_notice_clears_on_the_next_key():
    state = state_with([model(150)])
    refused = at(state, 600, "h")
    assert refused.notice and refused.pending is None
    assert at(refused, 601).notice == refused.notice
    assert handle_key(refused, ".").notice == ""


def test_reviewed_rows_only_refuses_model_rows_and_passes_reviewed_ones(tmp_path):
    reviewed = pd.DataFrame({"source": ["confirmed", "hand", "moved", "corrected_type"]})
    assert reviewed_rows_only(reviewed, "x.csv") is reviewed
    with pytest.raises(ValueError, match="2 unreviewed model labels"):
        reviewed_rows_only(pd.DataFrame({"source": ["model", "hand", "model"]}), "x.csv")
    assert reviewed_rows_only(pd.DataFrame({"rally": [1]}), "x.csv") is not None


def model_labels_file(tmp_path):
    labels = tmp_path / "game.csv"
    pd.DataFrame({"rally": [1, 1], "frame_num": [10, 20], "player": ["me", "opp"], "type_group": ["serve", "net"], "type_detail": ["serve_short", "net"], "source": ["model", "confirmed"]}).to_csv(labels, index=False)
    pd.DataFrame({"rally": [1], "start_frame": [5], "end_frame": [30], "bounds_checked": [False]}).to_csv(tmp_path / "game_rallies.csv", index=False)
    return labels


def test_every_scoring_and_training_reader_refuses_unreviewed_model_rows(tmp_path):
    from vision.hit_training import club_labels
    from vision.label_scores import read_labels
    labels = model_labels_file(tmp_path)
    for reader in (read_labels, club_labels):
        with pytest.raises(ValueError, match="unreviewed model labels"):
            reader(labels)


def test_the_classifier_trainers_refuse_unreviewed_model_rows(tmp_path, monkeypatch):
    import vision.club_classifier as club_classifier
    import vision.flight_types as flight_types
    labels = model_labels_file(tmp_path)
    rows = pd.DataFrame({"row_id": [1, 2], "frame_num": [10, 20], "has_points": [True, True]})
    monkeypatch.setattr(club_classifier, "shot_rows", lambda source: rows)
    monkeypatch.setattr(flight_types, "shot_rows", lambda source: rows)
    monkeypatch.setattr(flight_types, "with_flight_features", lambda rows, source: rows)
    with pytest.raises(ValueError, match="unreviewed model labels"):
        club_classifier.club_rows(None, labels)
    with pytest.raises(ValueError, match="unreviewed model labels"):
        flight_types.training_rows(None, labels)


def test_movement_evaluate_labels_refuses_unreviewed_model_rows(tmp_path):
    from vision.movement import labels_from_file
    with pytest.raises(ValueError, match="unreviewed model labels"):
        labels_from_file(model_labels_file(tmp_path), 0, 100)


def test_a_hit_already_outside_every_rally_never_blocks_edits_and_is_kept():
    orphan = HitLabel(0, 700, "me", "net", "hand", "net")
    inside = HitLabel(1, 150, "opp", "lift", "hand", "lift")
    state = state_with([inside, orphan], review=False)
    moved_start = at(state, 120, "b")
    assert [(span.start_frame, span.end_frame) for span in moved_start.rallies] == [(120, 300)] and not moved_start.notice
    assert sorted(label.frame for label in moved_start.labels) == [150, 700]
    refused = at(state, 160, "r")
    assert unchanged(state, refused) and "f150" in refused.notice and "f700" not in refused.notice


def test_the_measurement_tools_refuse_unreviewed_model_rows(tmp_path, capsys):
    import vision.flight
    import vision.track_check
    from vision.label import reviewed_rows_only
    from vision.zone_split import main as zone_split_main
    labels = model_labels_file(tmp_path)
    assert zone_split_main(["--labels", str(labels), "--database", str(tmp_path / "db.duckdb"), "--match-id", "m", "--out", str(tmp_path / "out")]) == 1
    assert "unreviewed model labels" in capsys.readouterr().err
    assert vision.flight.reviewed_rows_only is reviewed_rows_only and vision.track_check.reviewed_rows_only is reviewed_rows_only
