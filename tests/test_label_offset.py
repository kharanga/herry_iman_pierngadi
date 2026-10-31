import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vision.label_offset import (
    MIN_FINE_CLARITY,
    MIN_PEAK_CLARITY,
    OffsetResult,
    fine_offset,
    fine_offset_scores,
    is_clear,
    main,
    match_shots,
    offset_from_view,
    offset_score,
    offset_scores,
    peak_clarity,
    rally_indicator,
    rally_spans,
    score_profile,
)

REAL_SET_DIR = Path("data/CoachAI-Projects/ShuttleSet/set")
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
requires_shuttleset = pytest.mark.skipif(not (REAL_SET_DIR / AXELSEN_GINTING).exists(), reason="run ingest/fetch_shuttleset.sh first")


def broadcast_view(spans, frames, true_offset):
    return rally_indicator(spans, frames, offset=true_offset)


def write_set_csv(set_dir, match_folder, set_no, rallies_and_frames):
    folder = set_dir / match_folder
    folder.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(rallies_and_frames, columns=["rally", "frame_num"])
    rows.to_csv(folder / f"set{set_no}.csv", index=False)


def test_rally_spans_keep_rallies_of_different_sets_apart_and_round_after_scaling():
    shots = pd.DataFrame({"set_no": [1, 1, 2, 2, 2], "rally": [1, 1, 1, 1, 1], "frame_num": [10.0, 30.0, 500.0, 520.0, 510.0]})
    assert rally_spans(shots).tolist() == [[10, 30], [500, 520]]
    assert rally_spans(shots, 1.25).tolist() == [[12, 38], [625, 650]]


def test_rally_indicator_is_inclusive_at_both_ends_and_follows_a_negative_offset():
    spans = np.array([[100, 110]])
    frames = np.array([94, 95, 100, 105, 106, 110])
    assert rally_indicator(spans, frames).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert rally_indicator(spans, frames, offset=-5).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_offset_score_is_zero_when_either_signal_is_constant_and_one_at_the_true_offset():
    spans = np.array([[30, 60], [120, 150]])
    frames = np.arange(0, 200)
    view = broadcast_view(spans, frames, 4)
    assert offset_score(view, spans, frames, 4) == pytest.approx(1.0)
    assert offset_score(np.zeros(len(frames)), spans, frames, 4) == 0.0
    assert offset_score(view, np.array([[-1000, 1000]]), frames, 0) == 0.0
    assert offset_scores(view, spans, frames, [0, 4]).tolist()[1] == pytest.approx(1.0)


def test_peak_clarity_has_no_runner_up_when_every_offset_is_near_the_best():
    best, score, runner_up, clarity = peak_clarity([-2, 0, 2], [0.1, 0.5, 0.2], exclusion=10)
    assert (best, score) == (0, 0.5)
    assert math.isnan(runner_up) and math.isnan(clarity)
    assert not is_clear(OffsetResult(best, score, runner_up, clarity, 1.0))


def test_peak_clarity_ignores_the_shoulder_of_the_best_peak():
    offsets = [-20, -2, 0, 2, 20]
    assert peak_clarity(offsets, [0.3, 0.85, 0.9, 0.88, 0.4], exclusion=10) == (0, 0.9, 0.4, pytest.approx(0.5))


def test_is_clear_holds_exactly_at_the_threshold():
    assert is_clear(OffsetResult(0, 0.5, 0.4, MIN_PEAK_CLARITY, 1.0))
    assert not is_clear(OffsetResult(0, 0.5, 0.45, MIN_PEAK_CLARITY - 1e-9, 1.0))
    assert is_clear(OffsetResult(0, 0.5, 0.3, MIN_FINE_CLARITY, 1.0), MIN_FINE_CLARITY)
    assert not is_clear(OffsetResult(0, 0.5, 0.35, 0.15, 1.0), MIN_FINE_CLARITY)


def test_offset_from_view_recovers_a_late_label_and_keeps_the_frame_scale():
    spans = np.array([[300, 700], [1100, 1300], [1800, 2400], [2900, 3100]])
    frames = np.arange(0, 3600, 3)
    view = broadcast_view(spans, frames, -45)
    result = offset_from_view(view, frames, spans, max_offset=300, step=3, exclusion=150, frame_scale=1.2)
    assert result.offset_frames == -45
    assert result.label_scale == 1.2
    assert is_clear(result)


def test_offset_from_view_on_a_view_unrelated_to_the_rallies_is_unclear():
    spans = np.array([[300, 700], [1100, 1300], [1800, 2400], [2900, 3100]])
    frames = np.arange(0, 3600, 3)
    unrelated_view = (np.random.default_rng(2026).random(len(frames)) < 0.5).astype(float)
    assert not is_clear(offset_from_view(unrelated_view, frames, spans, max_offset=300, step=3, exclusion=150))


def test_fine_offset_scores_skip_hits_shifted_off_the_tracks_and_hitters_without_scores():
    frames = np.arange(100, 110)
    me = np.arange(10, dtype=float) / 10
    scores = fine_offset_scores([100, 109, 105], ["me", "me", "opp"], {"me": me}, frames, [0, 1])
    assert scores.tolist() == pytest.approx([(0.0 + 0.9) / 2, 0.1])
    assert fine_offset_scores([100], ["me"], {"me": me}, frames, [50]).tolist() == [0.0]


def test_fine_offset_reads_the_labelled_hitters_score_and_reports_a_tie_far_away_as_unclear():
    frames = np.arange(0, 400)
    me, opp = np.zeros(400), np.zeros(400)
    me[[62, 162, 262]] = 1.0
    opp[[100, 200, 300]] = 1.0
    opp[[50, 150, 250]] = 0.9
    result = fine_offset([50, 100, 150, 200, 250, 300], ["me", "opp", "me", "opp", "me", "opp"], {"me": me, "opp": opp}, frames, centre=0, half_width=20)
    assert result.offset_frames == 0
    assert result.score == pytest.approx(0.5)
    assert result.runner_up_score == pytest.approx(0.5)
    assert result.clarity == pytest.approx(0.0)
    assert not is_clear(result, MIN_FINE_CLARITY)


def test_fine_offset_searches_only_around_the_centre():
    frames = np.arange(0, 600)
    me = np.zeros(600)
    me[[110, 310]] = 1.0
    me[[60, 260]] = 0.6
    result = fine_offset([100, 300], ["me", "me"], {"me": me}, frames, centre=-40, half_width=15)
    assert result.offset_frames == -40
    assert result.score == pytest.approx(0.6)


def test_score_profile_marks_the_best_offset_on_a_flat_profile_without_dividing_by_zero():
    image = score_profile(np.array([0.5, 0.5, 0.5]), np.array([-3, 0, 3]), width=200, height=100)
    assert image.shape == (100, 200, 3)
    assert image.dtype == np.uint8


def test_match_shots_reads_every_set_present_and_drops_shots_without_a_frame(tmp_path):
    write_set_csv(tmp_path, "m", 1, [(1, 10.0), (1, 20.0)])
    write_set_csv(tmp_path, "m", 3, [(1, 30.0), (2, None)])
    shots = match_shots("m", tmp_path)
    assert shots[["set_no", "rally", "frame_num"]].values.tolist() == [[1, 1, 10.0], [1, 1, 20.0], [3, 1, 30.0]]
    with pytest.raises(FileNotFoundError):
        match_shots("other", tmp_path)


def test_main_reports_an_error_and_writes_nothing_without_a_calibrated_video(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("vision.label_offset.OFFSETS_DIR", tmp_path / "alignment")
    monkeypatch.setattr("vision.label_offset.CHECKS_DIR", tmp_path / "checks")
    assert main([str(tmp_path / "missing.mp4"), "--match", AXELSEN_GINTING, "--label-fps", "30"]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert not (tmp_path / "alignment").exists() and not (tmp_path / "checks").exists()


@requires_shuttleset
def test_axelsen_ginting_labels_give_one_span_per_rally_over_all_three_sets():
    shots = match_shots(AXELSEN_GINTING, REAL_SET_DIR)
    spans = rally_spans(shots)
    assert len(shots) == 1095
    assert sorted(shots["set_no"].unique().tolist()) == [1, 2, 3]
    assert len(spans) == shots.groupby(["set_no", "rally"]).ngroups
    assert (spans[:, 1] >= spans[:, 0]).all()


def test_tracking_stretches_pad_clamp_at_zero_and_merge_rallies_closer_than_the_merge_gap():
    from vision.label_offset import tracking_stretches
    spans = np.array([[400, 500], [10, 100], [600, 700], [1000, 1100]])
    assert tracking_stretches(spans, 25.0) == [(0, 150), (350, 750), (950, 1150)]
    assert tracking_stretches(spans, 25.0, padding_seconds=0.0, merge_seconds=0.0) == [(10, 100), (400, 500), (600, 700), (1000, 1100)]
    assert tracking_stretches(np.array([[100, 200], [250, 300]]), 50.0, padding_seconds=0.0, merge_seconds=1.0) == [(100, 300)]
    assert tracking_stretches(np.array([[100, 200], [251, 300]]), 50.0, padding_seconds=0.0, merge_seconds=1.0) == [(100, 200), (251, 300)]
    assert tracking_stretches(np.array([[100, 200], [251, 300]]), 50.0, padding_seconds=0.0, merge_seconds=0.5) == [(100, 200), (251, 300)]


def test_labelled_hitters_scale_frames_and_pixels_and_keep_shots_without_a_position():
    from vision.label_offset import labelled_hitters
    homography = np.eye(3)
    shots = pd.DataFrame({"set_no": [1, 1, 2], "rally": [1, 1, 1], "frame_num": [100.0, 125.0, 30.0], "type": ["a", "b", "c"],
                          "player_location_x": [3.0, 3.0, float("nan")], "player_location_y": [2.0, 10.0, 5.0]})
    labels = labelled_hitters(shots, homography, frame_scale=1.2, offset=-3)
    assert labels["frame"].tolist() == [117, 147, 33]
    assert labels["hitter"].tolist()[:2] == ["me", "opp"] and pd.isna(labels["hitter"].iloc[2])
    near_unless_scaled = shots.assign(player_location_y=[3.0, 3.0, 3.0])
    assert labelled_hitters(near_unless_scaled, homography, 1.0, 0)["hitter"].tolist()[:2] == ["me", "me"]
    assert labelled_hitters(near_unless_scaled, homography, 1.0, 0, pixel_scale=3.0)["hitter"].tolist()[:2] == ["opp", "opp"]


def test_pooled_scores_join_stretches_in_order():
    from vision.label_offset import pooled_scores
    frames, scores = pooled_scores([(np.array([5, 6]), {"me": np.array([0.1, 0.2]), "opp": np.array([0.3, 0.4])}), (np.array([50]), {"me": np.array([0.5]), "opp": np.array([0.6])})])
    assert frames.tolist() == [5, 6, 50]
    assert scores["me"].tolist() == [0.1, 0.2, 0.5] and scores["opp"].tolist() == [0.3, 0.4, 0.6]


def test_a_missing_record_or_stretch_list_names_the_command_that_writes_it(tmp_path):
    from vision.label_offset import read_offset_record, read_stretch_list
    with pytest.raises(FileNotFoundError, match="--label-fps"):
        read_offset_record(Path("data/match.mp4"), tmp_path)
    with pytest.raises(FileNotFoundError, match="--stretches"):
        read_stretch_list(Path("data/match.mp4"), tmp_path)


def test_main_refuses_both_steps_at_once_and_a_coarse_run_without_match_and_fps(tmp_path, capsys):
    assert main([str(tmp_path / "v.mp4"), "--stretches", "--fine"]) == 1
    assert "separate steps" in capsys.readouterr().err
    assert main([str(tmp_path / "v.mp4")]) == 1
    assert "--match and --label-fps" in capsys.readouterr().err


def test_players_stretches_skip_tracked_stretches_without_loading_a_model_and_refuse_a_bad_list(tmp_path):
    from vision.players import boxes_path, positions_path, read_stretches, track_stretches
    video = tmp_path / "v.mp4"
    for start, end in ((0, 10), (20, 30)):
        for path in (positions_path(video, start, end, tmp_path / "players"), boxes_path(video, start, end, tmp_path / "players")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frame\n")
    assert track_stretches(video, [(0, 10), (20, 30)], model_path=tmp_path / "no_weights.pt", players_dir=tmp_path / "players") == (0, 2, 0)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"start": [0], "end": [10]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        read_stretches(bad)
