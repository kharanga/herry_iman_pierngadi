import argparse

import numpy as np
import pandas as pd
import pytest

import vision.in_play as in_play
from vision.in_play import club_sample_mask, doubtful_frames, hand_marked_spans, main
from vision.rally_bounds import bound_lines, bound_row, frames_right, outside_unchecked_only


def rallies(rows, checked=None):
    table = pd.DataFrame(rows, columns=["rally", "start_frame", "end_frame"])
    return table if checked is None else table.assign(bounds_checked=checked)


def test_hand_marked_spans_split_checked_from_unchecked_and_a_file_without_the_column_counts_as_checked():
    table = rallies([(2, 300, 400), (1, 100, 200), (3, 500, 600)], [True, False, True])
    assert hand_marked_spans(table) == ([(300, 400), (500, 600)], [(100, 200)])
    assert hand_marked_spans(table.drop(columns="bounds_checked")) == ([(100, 200), (300, 400), (500, 600)], [])


def test_hand_marked_spans_read_text_flags_like_the_label_tool():
    table = rallies([(1, 100, 200), (2, 300, 400), (3, 500, 600)], ["False", "True", np.nan])
    assert hand_marked_spans(table) == ([(300, 400)], [(100, 200), (500, 600)])


def test_doubtful_frames_are_the_frames_with_any_doubtful_player_row():
    positions = pd.DataFrame({"frame": [1, 1, 2, 3, 3], "player": ["me", "opp", "me", "me", "opp"], "doubtful": [False, True, False, True, True]})
    assert doubtful_frames(positions).tolist() == [1, 3]
    assert doubtful_frames(positions.drop(columns="doubtful")).tolist() == []


def test_the_club_sample_takes_every_other_frame_minus_unchecked_rallies_and_doubtful_frames():
    frames = np.arange(100, 120)
    mask = club_sample_mask(frames, [(104, 107)], np.array([110, 111]), step=2)
    assert frames[mask].tolist() == [100, 102, 108, 112, 114, 116, 118]


def test_save_with_clubs_weights_the_clubs_to_their_share_and_without_clubs_fits_unweighted(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    play_rows = {name: (rng.normal(size=(n, 3)), (rng.random(n) > 0.5).astype(int)) for name, n in (("a", 60), ("b", 40))}
    club_rows = (rng.normal(size=(30, 3)), (rng.random(30) > 0.5).astype(int))
    seen = []
    monkeypatch.setattr(in_play, "fit_in_play_model", lambda features, targets, weights=None: seen.append((len(targets), None if weights is None else np.asarray(weights))) or "model")
    monkeypatch.setattr(in_play, "save_in_play_model", lambda model, path, sources: path)
    monkeypatch.setattr(in_play, "load_club_rows", lambda video, labels, hit_model: ("club:x", club_rows))
    plain = argparse.Namespace(club=[], club_weight=1 / 3, save=tmp_path / "m.joblib")
    in_play.save_with_clubs(play_rows, ["a", "b"], plain)
    assert seen[-1] == (100, None)
    with_club = argparse.Namespace(club=[("v.mp4", "l.csv", "h.joblib")], club_weight=0.25, save=tmp_path / "m.joblib")
    in_play.save_with_clubs(play_rows, ["a", "b"], with_club)
    count, weights = seen[-1]
    assert count == 130 and weights is not None
    assert weights[100:].sum() / weights.sum() == pytest.approx(0.25)


def test_a_detected_rally_touching_only_unchecked_rallies_is_not_extra():
    checked, unchecked = [(100, 200)], [(400, 500)]
    found = [(110, 190), (420, 480), (700, 800), (180, 450)]
    assert outside_unchecked_only(found, checked, unchecked) == [(110, 190), (700, 800), (180, 450)]


def test_frames_inside_unchecked_rallies_are_not_judged():
    frames = np.arange(0, 20)
    right, judged = frames_right([(2, 5), (12, 15)], frames, [(2, 6)], [(10, 14)])
    assert judged == 15 and right == 13


def test_a_bound_row_scores_checked_rallies_only():
    frames = np.arange(0, 1000)
    row = bound_row("m", [(105, 195), (420, 480), (700, 760)], frames, [(100, 200), (300, 350)], [(400, 500)], 10.0)
    assert (row["checked_rallies"], row["found"], row["matched"], row["missed"], row["extra"], row["merged"], row["split"]) == (2, 3, 1, 1, 1, 0, 0)
    assert row["start_error_mean_s"] == pytest.approx(0.5) and row["end_error_mean_s"] == pytest.approx(-0.5)
    assert row["frames_judged"] == 1000 - 101
    line = bound_lines([row])[2]
    assert line.startswith("| m | ") and "| 3 | 1 of 2 | 0 | 0 | 1 | 1 |" in line and line.endswith("| +0.50 | -0.50 |")


def test_cli_refuses_club_without_save_and_a_weight_outside_the_range(capsys):
    assert main(["list.txt", "--held-out", "--club", "v.mp4", "l.csv", "h.joblib"]) == 2
    assert "--club needs --save" in capsys.readouterr().err
    for weight in ("1", "-0.1", "1.5"):
        assert main(["list.txt", "--save", "m.joblib", "--club-weight", weight]) == 2
        assert "--club-weight" in capsys.readouterr().err
