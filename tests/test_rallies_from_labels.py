import argparse
import math
from dataclasses import replace

import numpy as np
import pytest

from vision.movement import HitCandidate, MovementSettings, labelled_rally, serve_hitter, tables_with_labelled_rallies

SETTINGS = MovementSettings(fps=30.0)


def test_the_server_is_opposite_the_first_chain_hit_or_the_higher_score_at_the_start():
    assert serve_hitter([HitCandidate(40, "opp", 0.9)], 10, {}) == "me"
    scores = {"me": np.array([0.1] * 20), "opp": np.array([0.4] * 20)}
    assert serve_hitter([], 10, scores) == "opp"


def test_a_labelled_rally_is_a_serve_row_then_chain_hits_after_the_same_hit_window_up_to_the_end():
    candidates = [HitCandidate(102, "me", 0.9), HitCandidate(130, "opp", 0.9), HitCandidate(160, "me", 0.9), HitCandidate(260, "opp", 0.9)]
    rally = labelled_rally(candidates, {}, 100, 200, 8, SETTINGS)
    assert rally[0].frame == 100 and math.isnan(rally[0].score) and rally[0].hitter == "me"
    assert [(hit.frame, hit.hitter) for hit in rally[1:]] == [(130, "opp"), (160, "me")]


def test_the_option_refuses_other_rally_sources_and_no_serves():
    parsed = argparse.Namespace(rallies_from=None, in_play_model=None, serve_stance=False, stretch_model=None, rallies_from_labels="r.csv", model="m.joblib")
    for field, value in (("rallies_from", "x"), ("in_play_model", "x"), ("serve_stance", True), ("stretch_model", "x")):
        with pytest.raises(ValueError, match="can't be combined"):
            tables_with_labelled_rallies(None, argparse.Namespace(**{**vars(parsed), field: value}), SETTINGS)
    with pytest.raises(ValueError, match="no-serves"):
        tables_with_labelled_rallies(None, parsed, replace(SETTINGS, infer_serves=False))
