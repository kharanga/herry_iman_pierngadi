import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vision.label_scores import ScoredRun, non_serve_table
from vision.movement import SAME_HIT_AS_SERVE_SECONDS, MovementSettings, rallies_with_newer_hits, tables_with_rallies_from

NAN = float("nan")


def hits(rows):
    return pd.DataFrame(rows, columns=["rally", "frame", "hitter", "score", "reach"])


def test_each_default_rally_keeps_its_serve_row_then_the_newer_hits_after_it():
    default_hits = hits([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.8, NAN), (2, 500, "opp", 0.7, NAN), (2, 540, "me", 0.8, NAN)])
    rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [100, 500], "end_frame": [300, 600], "hits": [3, 2]})
    newer = hits([
        (1, 95, "me", 0.9, NAN),
        (1, 110, "me", 0.9, NAN),
        (1, 116, "opp", 0.9, NAN),
        (1, 150, "opp", 0.6, NAN),
        (1, 120, "me", NAN, NAN),
        (1, 300, "me", 0.7, NAN),
        (1, 301, "opp", 0.7, NAN),
        (3, 400, "me", 0.9, NAN),
        (4, 514, "me", 0.9, NAN),
        (4, 516, "me", 0.9, NAN),
        (5, 800, "opp", 0.9, NAN),
    ])
    combined, recounted = rallies_with_newer_hits(default_hits, rallies, newer, 15)
    assert combined[["rally", "frame", "hitter"]].values.tolist() == [
        [1, 100, "me"], [1, 116, "opp"], [1, 150, "opp"], [1, 300, "me"],
        [2, 500, "opp"], [2, 516, "me"],
    ]
    assert np.isnan(combined["score"].iloc[0]) and combined["score"].iloc[4] == 0.7
    assert recounted["hits"].tolist() == [4, 2] and recounted[["start_frame", "end_frame"]].values.tolist() == [[100, 300], [500, 600]]
    assert list(combined.columns) == ["rally", "frame", "hitter", "score", "reach"]


def test_no_rallies_give_no_hits_and_a_rally_without_rows_keeps_none():
    empty, recounted = rallies_with_newer_hits(hits([]), pd.DataFrame(columns=["rally", "start_frame", "end_frame", "hits"]), hits([(1, 10, "me", 0.9, NAN)]), 15)
    assert len(empty) == 0 and len(recounted) == 0
    rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [200], "hits": [0]})
    none, recounted = rallies_with_newer_hits(hits([]), rallies, hits([(1, 150, "me", 0.9, NAN)]), 15)
    assert len(none) == 0 and recounted["hits"].tolist() == [0]


def test_the_same_hit_window_is_a_quarter_second():
    assert SAME_HIT_AS_SERVE_SECONDS == 0.25


def test_rallies_from_refuses_the_in_play_model_and_no_serves_before_loading_anything():
    parsed = argparse.Namespace(in_play_model=Path("in_play.joblib"), rallies_from=Path("missing.joblib"), model=Path("missing.joblib"), video=Path("v.mp4"), rallies_hit_score_min=0.5, rallies_end_score_min=0.7)
    with pytest.raises(ValueError, match="not both"):
        tables_with_rallies_from(None, parsed, MovementSettings())
    parsed.in_play_model = None
    with pytest.raises(ValueError, match="no-serves"):
        tables_with_rallies_from(None, parsed, replace(MovementSettings(), infer_serves=False))


def test_the_left_out_serves_table_counts_non_serves_and_those_typed_serve():
    labels = pd.DataFrame({"rally": [1, 1, 1, 1], "frame": [100, 130, 160, 190], "hitter": ["me", "opp", "me", "opp"], "type": ["serve", "net", "lift", "clear"], "detail": ["serve_short", "net", "lift", "clear"], "source": ["hand"] * 4})
    run_hits = hits([(1, 100, "me", NAN, NAN), (1, 131, "opp", 0.9, NAN), (1, 161, "me", 0.9, NAN)])
    predictions = pd.DataFrame({"frame_num": [100, 131, 161], "type_pred": ["serve", "serve", "lift"]})
    lines = non_serve_table([ScoredRun("model", run_hits, predictions)], labels, [(0.25, 5)], 20.0)
    assert lines[2] == "| model | ±0.25 s (±5 frames at 20 fps) | 2/3 = 66.7% | 1/2 = 50.0% | 1 |"
