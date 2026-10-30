import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M
from shots.compare_hit_points import compare_hit_points, landing_offset_summary, rows_predictable_with_every_feature_set, summary_lines
from shots.features import FEATURE_SET_NAMES
from shots.jitter import JITTER_OFFSETS_FRAMES, jitter_label
from shots.predict import predict_shot_types, prediction_summary
from shots.results_tables import accuracy_cell, hit_point_results_tables, predictions_table, subsets_table
from shots.train import TRAINED_GROUPS, labelled_rows_of
from tests.test_jitter import feature_rows
from tests.test_shot_classifier import GROUP_SHOT_PATTERNS, synthetic_shot_rows, write_shots_database

HIT_POINT_SECTION_TITLES = [
    "Contact vs feet",
    "Per group, contact vs feet (shared shots)",
    "Confusion matrix, contact (shared shots)",
    "Confusion matrix, feet (shared shots)",
    "Timing jitter",
    "Landing offset",
]


def rows_with_gaps() -> list[dict]:
    rows = synthetic_shot_rows(5, 4)
    first_of = lambda group: next(index for index, row in enumerate(rows) if row["type_group"] == group)
    rows[first_of("drop")] = {**rows[first_of("drop")], "landing_x": None, "landing_y": None}
    rows[first_of("clear") + 1] = {**rows[first_of("clear") + 1], "player_x": None, "player_y": None}
    return rows


@pytest.fixture(scope="module")
def synthetic_comparison(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("hit_points")
    database_path = write_shots_database(work_dir / "scout.duckdb", rows_with_gaps())
    comparison_path = work_dir / "models" / "hit_point_comparison.json"
    comparison = compare_hit_points(database_path, comparison_path, fold_count=5)
    return database_path, comparison_path, comparison


def test_shared_rows_are_those_predictable_by_both_sets(tmp_path):
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows_with_gaps())
    import duckdb
    from shots.features import load_shot_feature_rows

    connection = duckdb.connect(str(database_path), read_only=True)
    shot_rows = load_shot_feature_rows(connection)
    connection.close()
    labelled = 5 * 4 * len(GROUP_SHOT_PATTERNS)
    assert len(labelled_rows_of(shot_rows, "contact")) == labelled - 1
    assert len(labelled_rows_of(shot_rows, "feet")) == labelled - 1
    shared = rows_predictable_with_every_feature_set(labelled_rows_of(shot_rows, "contact"))
    assert len(shared) == labelled - 2
    assert shared["landing_x"].notna().all() and shared["next_player_x"].notna().all()


def test_landing_offset_is_the_distance_from_the_reply_hitters_feet_to_the_labelled_landing(tmp_path):
    shot_rows = feature_rows(tmp_path, 1, 2)
    labelled = labelled_rows_of(shot_rows, "feet")
    summary = landing_offset_summary(labelled)
    assert summary["shots"] == len(labelled)
    assert summary["p50_m"] == pytest.approx(0.0, abs=1e-9) and summary["mean_m"] == pytest.approx(0.0, abs=1e-9) and summary["p90_m"] == pytest.approx(0.0, abs=1e-9)
    shifted = labelled.copy()
    shifted["next_player_x"] = COURT_WIDTH_M - shifted["landing_x"] - 0.3
    shifted["next_player_y"] = COURT_LENGTH_M - shifted["landing_y"] + 0.4
    assert landing_offset_summary(shifted)["p50_m"] == pytest.approx(0.5)


def test_comparison_evaluates_both_sets_on_the_same_folds_with_jitter_and_writes_json(synthetic_comparison):
    database_path, comparison_path, comparison = synthetic_comparison
    assert json.loads(comparison_path.read_text()) == comparison
    labelled = 5 * 4 * len(GROUP_SHOT_PATTERNS)
    for feature_set in FEATURE_SET_NAMES:
        shared = comparison["shared_shots"][feature_set]
        every = comparison["all_shots"][feature_set]
        assert shared["shots"] == labelled - 2 and every["shots"] == labelled - 1
        assert shared["matches"] == every["matches"] == 5
        for report in (shared, every):
            assert len(report["folds"]) == 5 and all(fold["test_matches"] == 1 for fold in report["folds"])
            assert report["accuracy_mean"] > report["majority_accuracy_mean"]
            assert list(report["variants"]) == [jitter_label(offset) for offset in JITTER_OFFSETS_FRAMES]
            assert set(report["per_group"]) == set(TRAINED_GROUPS)
            assert sum(sum(row) for row in report["confusion_matrix"]["rows_true_columns_predicted"]) == report["shots"]
            assert "permutation_importance" not in report and "per_type_en" not in report
    assert comparison["landing_offset_m"]["shots"] == labelled - 2
    assert comparison["landing_offset_m"]["p50_m"] == pytest.approx(0.0, abs=1e-9)
    assert [line for line in summary_lines(comparison) if line.startswith("feet landing")] == [f"feet landing vs labelled landing: median 0.00 m on {labelled - 2} shots"]


def test_hit_point_tables_render_every_section_from_the_comparison(synthetic_comparison):
    _, _, comparison = synthetic_comparison
    tables = hit_point_results_tables(comparison)
    sections = [section.split("\n\n", 1) for section in tables.split("### ")[1:]]
    assert [title for title, _ in sections] == HIT_POINT_SECTION_TITLES
    accuracy_table = sections[0][1].splitlines()
    assert accuracy_table[0] == "| Feature set | Shared shots | Held-out accuracy | All its shots | Held-out accuracy |"
    assert f"**{comparison['shared_shots']['contact']['accuracy_mean']:.1%}**" in accuracy_table[2]
    assert f"**{comparison['all_shots']['feet']['accuracy_mean']:.1%}**" in accuracy_table[3]
    jitter = sections[4][1].splitlines()
    assert jitter[2].startswith("| exact | ") and jitter[3].startswith("| +/- 5 | ") and jitter[4].startswith("| +/- 10 | ")
    feet_five = comparison["all_shots"]["feet"]["variants"]["hit frames +/- 5"]["accuracy_mean"]
    assert jitter[3].endswith(f"| {feet_five:.1%} ({(feet_five - comparison['all_shots']['feet']['accuracy_mean']) * 100:+.1f}) |")
    assert sections[5][1].splitlines()[3] == "| Median distance | 0.00 m |"


def test_empty_subset_prints_none_predicted_and_own_matches_are_listed(synthetic_comparison, tmp_path):
    database_path, _, _ = synthetic_comparison
    assert accuracy_cell(None) == "none predicted" and accuracy_cell(0.8654) == "86.5%"
    table = subsets_table({"subsets": {"rally's last shot": {"shots": 0, "accuracy": None}, "others": {"shots": 1234, "accuracy": 0.5}}})
    assert "| rally's last shot | 0 | none predicted |" in table and "| others | 1,234 | 50.0% |" in table
    predicted_database = tmp_path / "predicted.duckdb"
    predicted_database.write_bytes(database_path.read_bytes())
    import duckdb

    connection = duckdb.connect(str(predicted_database))
    connection.execute(
        "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, player_x, player_y, opponent_x, opponent_y) VALUES "
        "('own', 'my_game', 1, 1, 1, 10, 30.0, 'me', 2.0, 1.0, 3.1, 7.5), ('own', 'my_game', 1, 1, 2, 50, 30.0, 'opp', 3.0, 5.6, 3.1, 7.9), "
        "('own', 'my_game', 2, 1, 1, 10, 30.0, 'me', 2.0, 1.0, 3.1, 7.5)"
    )
    connection.close()
    from shots.train import train_classifier

    model_path = tmp_path / "model.joblib"
    train_classifier(database_path, model_path, fold_count=5, permutation_repeats=0)
    predict_shot_types(predicted_database, model_path)
    summary = prediction_summary(predicted_database)
    assert summary["own_matches"] == {"my_game": {"shots": 3, "predicted_shots": 1, "rallies": 2}}
    assert "| Own rows of `my_game` with a prediction | 1 of 3 (2 rallies) |" in predictions_table(summary)
