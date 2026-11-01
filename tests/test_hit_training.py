import math
from pathlib import Path

import pandas as pd
import pytest

import vision.hit_training as hit_training
from vision.hit_training import (
    MatchData,
    exclusion_reason,
    f1_of,
    held_out_rows,
    learning_curve_rows,
    players_of_matches,
    read_match_list,
    totals_line,
    training_matches,
)
from vision.movement import MovementSettings

NAMES = ["a_vs_b", "a_vs_c", "d_vs_e", "f_vs_g"]
PLAYERS = {"a_vs_b": {"A", "B"}, "a_vs_c": {"A", "C"}, "d_vs_e": {"D", "E"}, "f_vs_g": {"F", "G"}}


def fake_match(name):
    labels = pd.DataFrame({"frame": [10, 40], "hitter": ["me", "opp"]})
    return MatchData(name, Path(f"{name}.mp4"), 30.0, MovementSettings(), (), labels)


@pytest.fixture
def recorded_fits(monkeypatch):
    fits = []

    def fit_on(names, rows_by_match, club_names=(), club_weight=0.0):
        fits.append(tuple(sorted(names)))
        return tuple(sorted(names))

    monkeypatch.setattr(hit_training, "fit_on", fit_on)
    monkeypatch.setattr(hit_training, "match_scores", lambda model, match: [{"model": model, "scored": match.name}])
    monkeypatch.setattr(hit_training, "detected_hits", lambda match, scores, settings: pd.DataFrame({"frame": [10], "hitter": ["me"], "score": [0.9]}))
    return fits


def test_read_match_list_skips_comments_and_blank_lines_and_refuses_a_line_without_a_video(tmp_path):
    good = tmp_path / "matches.txt"
    good.write_text("# folder video\nm1 data/v1.mp4\n\nm2 data/v2.mp4\n")
    assert read_match_list(good) == [("m1", Path("data/v1.mp4")), ("m2", Path("data/v2.mp4"))]
    bad = tmp_path / "bad.txt"
    bad.write_text("m1 data/v1.mp4\nm2\n")
    with pytest.raises(ValueError):
        read_match_list(bad)


def test_exclusion_reason_needs_a_clear_coarse_and_a_clear_fine_offset():
    assert exclusion_reason({"clarity": 0.4, "fine_offset_frames": 0, "fine_clarity": 0.3}) is None
    assert exclusion_reason({"clarity": 0.1, "fine_offset_frames": 0, "fine_clarity": 0.2}) is None
    assert "coarse" in exclusion_reason({"clarity": 0.026})
    assert "coarse" in exclusion_reason({"clarity": float("nan"), "fine_offset_frames": 0, "fine_clarity": 0.3})
    assert "coarse" in exclusion_reason({})
    assert "--fine" in exclusion_reason({"clarity": 0.5})
    assert "fine offset unclear" in exclusion_reason({"clarity": 0.5, "fine_offset_frames": 3, "fine_clarity": 0.19})


def test_training_matches_never_hold_the_held_out_match_and_by_player_drops_shared_players():
    for held_out in NAMES:
        assert held_out not in training_matches(held_out, NAMES)
        assert len(training_matches(held_out, NAMES)) == len(NAMES) - 1
    assert training_matches("a_vs_b", NAMES, PLAYERS) == ["d_vs_e", "f_vs_g"]
    assert training_matches("d_vs_e", NAMES, PLAYERS) == ["a_vs_b", "a_vs_c", "f_vs_g"]


def test_players_of_matches_reads_winner_and_loser_of_every_set(tmp_path):
    pd.DataFrame({"video": ["m1", "m1", "m2"], "winner": ["A", "B", "C"], "loser": ["B", "A", "A"]}).to_csv(tmp_path / "match.csv", index=False)
    assert players_of_matches(["m1", "m2"], tmp_path) == {"m1": {"A", "B"}, "m2": {"A", "C"}}


def test_held_out_rows_score_each_match_with_a_model_that_never_saw_it_even_when_tuning(recorded_fits):
    matches = {name: fake_match(name) for name in NAMES}
    rows = held_out_rows(matches, {}, tune=True, progress=lambda line: None)
    outer_fits = [fit for fit in recorded_fits if len(fit) == len(NAMES) - 1]
    inner_fits = [fit for fit in recorded_fits if len(fit) == len(NAMES) - 2]
    assert sorted(outer_fits) == sorted(tuple(sorted(set(NAMES) - {name})) for name in NAMES)
    assert len(inner_fits) == len(NAMES) * (len(NAMES) - 1)
    assert {row["thresholds"] for row in rows} == {"default", "tuned"}
    assert len(rows) == len(NAMES) * 2 * 2


def test_held_out_rows_tuning_for_one_match_uses_only_models_without_that_match(recorded_fits, monkeypatch):
    seen_by_tuning = []
    monkeypatch.setattr(hit_training, "best_thresholds", lambda matches, scores: seen_by_tuning.append({name: scores[name][0]["model"] for name in scores}) or (0.5, 0.7))
    matches = {name: fake_match(name) for name in NAMES}
    held_out_rows(matches, {}, tune=True, progress=lambda line: None)
    for held_out, inner in zip(sorted(NAMES), seen_by_tuning):
        assert held_out not in inner
        for inner_match, model in inner.items():
            assert held_out not in model and inner_match not in model


def test_held_out_rows_by_player_train_without_any_match_sharing_a_player(recorded_fits):
    matches = {name: fake_match(name) for name in NAMES}
    rows = held_out_rows(matches, {}, players=PLAYERS, progress=lambda line: None)
    assert {row["match"]: row["training_matches"] for row in rows} == {"a_vs_b": 2, "a_vs_c": 2, "d_vs_e": 3, "f_vs_g": 3}
    assert ("d_vs_e", "f_vs_g") in recorded_fits and ("a_vs_b", "a_vs_c", "f_vs_g") in recorded_fits


def test_learning_curve_subsets_come_from_the_other_matches_and_end_with_all_of_them(recorded_fits):
    matches = {name: fake_match(name) for name in NAMES}
    rows = learning_curve_rows(matches, {}, sizes=(1, 2), seed=7, progress=lambda line: None)
    assert [row["training_matches"] for row in rows if row["tolerance_s"] == 0.25] == [1, 2, 3] * len(NAMES)
    for held_out, fits in zip(sorted(NAMES), [recorded_fits[index:index + 3] for index in range(0, len(recorded_fits), 3)]):
        assert all(held_out not in fit for fit in fits)
        assert fits[-1] == tuple(sorted(set(NAMES) - {held_out}))


def test_f1_and_totals_line_on_counts():
    assert f1_of({"labelled": 10, "detected": 10, "matched": 5}) == pytest.approx(0.5)
    assert f1_of({"labelled": 0, "detected": 0, "matched": 0}) == 0.0
    rows = pd.DataFrame({"labelled": [6, 4], "detected": [5, 5], "matched": [4, 4], "hitter_correct": [4, 3], "hitter_judged": [4, 4]})
    assert totals_line("x", rows) == "x: labelled 10, detected 10, matched 8, precision 80.0%, recall 80.0%, hitter 7/8 = 87.5%"


def test_main_reports_an_error_for_a_list_without_usable_matches(tmp_path, capsys):
    listing = tmp_path / "matches.txt"
    listing.write_text("m1 " + str(tmp_path / "missing.mp4") + "\n")
    assert hit_training.main([str(listing), "--held-out", "--out", str(tmp_path / "out")]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert not (tmp_path / "out").exists()


def test_tuning_with_a_club_scores_every_training_source_with_a_model_trained_without_it(recorded_fits, monkeypatch):
    seen_by_tuning = []
    monkeypatch.setattr(hit_training, "best_thresholds", lambda sources, scores: seen_by_tuning.append(({name: scores[name][0]["model"] for name in scores}, set(sources))) or (0.5, 0.7))
    matches = {name: fake_match(name) for name in NAMES}
    clubs = {"club_lin": fake_match("club_lin")}
    held_out_rows(matches, {}, tune=True, progress=lambda line: None, club_names=["club_lin"], club_weight=0.33, clubs=clubs)
    assert len(seen_by_tuning) == len(NAMES)
    for held_out, (inner, sources) in zip(sorted(NAMES), seen_by_tuning):
        assert "club_lin" in inner and held_out not in inner
        assert set(inner) == (set(NAMES) - {held_out}) | {"club_lin"}
        for inner_source, model in inner.items():
            assert inner_source not in model and held_out not in model
        assert "club_lin" in sources
    outer_fits = [fit for fit in recorded_fits if "club_lin" in fit and len(fit) == len(NAMES)]
    assert len(outer_fits) == len(NAMES)
