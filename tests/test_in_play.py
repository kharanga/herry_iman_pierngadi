import math
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

import vision.in_play as in_play
from vision.in_play import (
    IN_PLAY_FEATURE_NAMES,
    compare_spans,
    first_row_on_serve,
    frames_for_seconds,
    in_play_spans,
    in_play_targets,
    labelled_rally_spans,
    load_in_play_model,
    main,
    rally_of_span,
    runs_of,
    save_in_play_model,
    seconds_since,
    seconds_until,
)
from vision.movement import HitCandidate, MovementSettings, detected_tables

FPS = 10.0
SETTINGS = replace(MovementSettings(), fps=FPS)


def test_frames_for_seconds_is_at_least_one():
    assert frames_for_seconds(1.0, 30.0) == 30 and frames_for_seconds(0.01, 30.0) == 1 and frames_for_seconds(0.0, 30.0) == 1


def test_seconds_since_and_until_the_nearest_peak_are_capped():
    mask = np.array([False, True, False, False, True, False])
    assert seconds_since(mask, 1.0, cap_seconds=10.0).tolist() == [10.0, 0.0, 1.0, 2.0, 0.0, 1.0]
    assert seconds_until(mask, 1.0, cap_seconds=10.0).tolist() == [1.0, 0.0, 2.0, 1.0, 0.0, 10.0]
    assert seconds_since(np.array([True] + [False] * 30), 1.0, cap_seconds=10.0)[-1] == 10.0
    assert seconds_since(np.zeros(0, dtype=bool), 1.0).tolist() == []


def test_runs_of_gives_inclusive_index_runs():
    assert runs_of([False, True, True, False, True]) == [(1, 2), (4, 4)]
    assert runs_of([True, True]) == [(0, 1)] and runs_of([]) == [] and runs_of([False]) == []


def test_targets_mark_the_first_hit_to_the_last_hit_plus_landing():
    labels = pd.DataFrame({"set_no": [1, 1, 1, 2, 2], "rally": [1, 1, 1, 1, 1], "frame": [100, 120, 140, 500, 530]})
    spans = labelled_rally_spans(labels, FPS)
    assert spans == [(100, 150), (500, 540)]
    targets = in_play_targets(np.array([99, 100, 150, 151, 499, 520, 540, 541]), spans)
    assert targets.tolist() == [0, 1, 1, 0, 0, 1, 1, 0]


def test_spans_close_short_inner_gaps_keep_edge_gaps_and_drop_short_stretches():
    probability = np.array([0.6] * 25 + [0.1] * 5 + [0.9] * 25 + [0.0] * 20 + [0.7] * 10 + [0.0] * 10 + [0.8] * 3)
    assert in_play_spans(probability, FPS) == [(0, 54)]
    long_gap = np.array([0.6] * 25 + [0.1] * 10 + [0.9] * 25)
    assert in_play_spans(long_gap, FPS) == [(0, 24), (35, 59)]
    edge_gap = np.array([0.1] * 3 + [0.9] * 25 + [0.1] * 3)
    assert in_play_spans(edge_gap, FPS) == [(3, 27)]
    assert in_play_spans(np.array([0.9]), FPS) == [] and in_play_spans(np.zeros(0), FPS) == []
    assert in_play_spans(np.array([0.5] * 20), FPS) == [(0, 19)]


def test_a_stretch_gets_an_inferred_serve_only_when_its_first_hit_is_late():
    on_time = [HitCandidate(102, "opp", 0.9), HitCandidate(115, "me", 0.9)]
    assert rally_of_span(on_time, (100, 160), SETTINGS) == on_time
    late = [HitCandidate(110, "opp", 0.9), HitCandidate(125, "me", 0.9)]
    rally = rally_of_span(late, (100, 160), SETTINGS)
    assert (rally[0].frame, rally[0].hitter) == (100, "me") and math.isnan(rally[0].score) and rally[1:] == late
    assert rally_of_span(late, (100, 160), replace(SETTINGS, infer_serves=False)) == late
    assert rally_of_span([HitCandidate(300, "me", 0.9)], (100, 160), SETTINGS) == []
    edge = rally_of_span([HitCandidate(98, "opp", 0.9)], (100, 160), SETTINGS)
    assert [candidate.frame for candidate in edge] == [98]


def test_compare_spans_counts_each_kind_once_and_times_matched_pairs():
    labelled = [(0, 99), (200, 299), (400, 499), (600, 699), (900, 999)]
    found = [(10, 89), (180, 450), (610, 640), (660, 720), (1200, 1300)]
    comparison = compare_spans(labelled, found, FPS)
    assert (comparison.labelled, comparison.found, comparison.matched) == (5, 5, 3)
    assert (comparison.merged, comparison.split, comparison.missed, comparison.extra) == (1, 1, 1, 1)
    assert comparison.start_errors == pytest.approx((-2.0, 1.0, 6.0))
    assert comparison.end_errors == pytest.approx((15.1, -1.0, 2.1))
    nothing = compare_spans([], [], FPS)
    assert (nothing.matched, nothing.missed, nothing.extra, nothing.start_errors) == (0, 0, 0, ())


def test_first_row_on_serve_counts_rallies_whose_first_row_is_near_the_labelled_serve():
    labels = pd.DataFrame({"rally": [1, 1, 2, 2, 3], "frame": [100, 130, 300, 330, 600]})
    hits = pd.DataFrame({"rally": [1, 1, 2, 2], "frame": [104, 130, 318, 330]})
    assert first_row_on_serve(hits, pd.DataFrame(), labels, tolerance=10) == (1, 3)
    assert first_row_on_serve(hits.iloc[0:0], pd.DataFrame(), labels, tolerance=10) == (0, 3)


def test_model_file_round_trip_and_refusals(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, len(IN_PLAY_FEATURE_NAMES)))
    targets = (features[:, 0] > 0).astype(int)
    model = in_play.fit_in_play_model(features, targets)
    saved = save_in_play_model(model, tmp_path / "deep" / "in_play.joblib", ["a", "b"])
    loaded = load_in_play_model(saved)
    assert np.array_equal(loaded.predict(features), model.predict(features))
    assert joblib.load(saved)["trained_on"] == ["a", "b"]
    stale = {**joblib.load(saved), "feature_names": list(IN_PLAY_FEATURE_NAMES)[:-1]}
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="other features"):
        load_in_play_model(tmp_path / "stale.joblib")
    with pytest.raises(FileNotFoundError):
        load_in_play_model(tmp_path / "missing.joblib")


def test_probability_is_smoothed_and_empty_features_give_no_probability():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(100, len(IN_PLAY_FEATURE_NAMES)))
    model = in_play.fit_in_play_model(features, (features[:, 0] > 0).astype(int))
    probability = in_play.in_play_probability(model, features, SETTINGS)
    assert probability.shape == (100,) and ((0 <= probability) & (probability <= 1)).all()
    assert np.abs(np.diff(probability)).max() < np.abs(np.diff(model.predict_proba(features)[:, 1])).max()
    assert in_play.in_play_probability(model, np.zeros((0, len(IN_PLAY_FEATURE_NAMES))), SETTINGS).tolist() == []


def test_cli_needs_held_out_or_save_and_reports_a_missing_list(tmp_path, capsys):
    assert main([str(tmp_path / "list.txt")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main([str(tmp_path / "list.txt"), "--held-out", "--out", str(tmp_path / "out")]) == 1
    assert "error:" in capsys.readouterr().err


def test_movement_without_an_in_play_model_uses_the_rule(monkeypatch):
    monkeypatch.setattr("vision.movement.detect_hits", lambda tracks, model, settings: ("rule hits", "rule rallies"))
    assert detected_tables("tracks", "model", None, SETTINGS) == ("rule hits", "rule rallies")


def test_gap_coverage_counts_frames_strictly_between_spans():
    from vision.in_play import gap_coverage
    coverage = gap_coverage([(10, 19), (40, 49), (100, 109)], [(0, 29), (35, 59), (95, 109)], 10.0)
    assert coverage["tracked_minutes"] == pytest.approx(70 / 10 / 60)
    assert coverage["out_of_play_minutes"] == pytest.approx(40 / 10 / 60)
    assert coverage["median_gap_seconds"] == pytest.approx((20 + 50) / 2 / 10)
    assert coverage["gap_tracked_share"] == pytest.approx((15 / 20 + 15 / 50) / 2)
    assert math.isnan(gap_coverage([(10, 19)], [(0, 29)], 10.0)["median_gap_seconds"])
