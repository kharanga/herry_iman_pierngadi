import ast
import json
import re
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

import shots.train
from analytics.build_shots import SCHEMA_PATH
from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M
from shots.features import FEATURE_NAMES, FEATURE_SET_NAMES, SHIPPED_FEATURE_SET, has_hitter_and_landing_point, load_shot_feature_rows, shot_features
from shots.jitter import JITTER_OFFSETS_FRAMES, jitter_label
from shots.predict import ModelFeatureMismatchError, load_model, predict_shot_types
from shots.train import TRAINED_GROUPS, grouped_cross_validation, train_classifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "classifier_feet.md"
REAL_REPORT_PATH = PROJECT_ROOT / "data" / "models" / "shot_type_classifier_report.json"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
LABEL_OR_IDENTITY_WORDS = ("type", "group", "player", "match", "source", "set", "rally", "score", "reason", "backhand", "height", "aroundhead", "getpoint", "flaw")
GROUP_SHOT_PATTERNS = {
    "serve": ((2.0, 1.0), (4.0, 11.5), 1, 40),
    "clear": ((2.0, 0.8), (2.5, 12.8), 3, 55),
    "drop": ((4.0, 1.0), (3.5, 7.8), 3, 30),
    "smash": ((3.0, 1.5), (4.5, 10.0), 3, 15),
    "net": ((3.0, 5.5), (2.8, 7.4), 3, 25),
    "lift": ((3.0, 5.5), (3.0, 12.6), 3, 45),
    "drive": ((1.0, 4.0), (1.2, 10.0), 3, 18),
}


def synthetic_shot_rows(match_count: int, shots_per_group: int, random_seed: int = 0) -> list[dict]:
    random = np.random.default_rng(random_seed)
    rows = []
    for match_index in range(match_count):
        for group_index, (group, (hit, landing, ball_round, flight_frames)) in enumerate(GROUP_SHOT_PATTERNS.items()):
            for shot_index in range(shots_per_group):
                rally = group_index * 1000 + shot_index
                jitter = random.normal(0.0, 0.15, size=4)
                labelled_shot = (
                    {
                        "source": "shuttleset",
                        "match_id": f"match_{match_index}",
                        "set_no": 1,
                        "rally": rally,
                        "ball_round": ball_round,
                        "frame_num": 100,
                        "fps": 25.0,
                        "player": "A",
                        "type_en": group,
                        "type_group": group,
                        "hit_x": hit[0] + jitter[0],
                        "hit_y": hit[1] + jitter[1],
                        "landing_x": landing[0] + jitter[2],
                        "landing_y": landing[1] + jitter[3],
                        "player_x": hit[0] + jitter[0],
                        "player_y": hit[1] + jitter[1],
                        "opponent_x": landing[0],
                        "opponent_y": landing[1],
                        "is_cross": False,
                    }
                )
                previous_hitter = (landing[0], COURT_LENGTH_M - landing[1])
                reply_hitter = (COURT_WIDTH_M - labelled_shot["landing_x"], COURT_LENGTH_M - labelled_shot["landing_y"])
                rows.append({**labelled_shot, "ball_round": ball_round - 1 if ball_round > 1 else 0, "frame_num": 60, "hit_x": previous_hitter[0], "hit_y": previous_hitter[1], "player_x": previous_hitter[0], "player_y": previous_hitter[1], "type_group": "unknown", "type_en": "unknown"})
                rows.append(labelled_shot)
                rows.append({**labelled_shot, "ball_round": ball_round + 1, "frame_num": 100 + flight_frames, "player_x": reply_hitter[0], "player_y": reply_hitter[1], "type_group": "unknown", "type_en": "unknown"})
    return rows


SHOTS_PER_RALLY = 3
PREDICTABLE_SHOTS_PER_RALLY = SHOTS_PER_RALLY - 1


def predictable_synthetic_shots(match_count: int, shots_per_group: int) -> int:
    return match_count * shots_per_group * len(GROUP_SHOT_PATTERNS) * PREDICTABLE_SHOTS_PER_RALLY


def feature_rows_of(database_path: Path) -> pd.DataFrame:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return load_shot_feature_rows(connection)
    finally:
        connection.close()


def write_shots_database(database_path: Path, rows: list[dict]) -> Path:
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    connection.register("synthetic_rows", pd.DataFrame(rows))
    columns = ", ".join(rows[0])
    connection.execute(f"INSERT INTO shots ({columns}) SELECT {columns} FROM synthetic_rows")
    connection.unregister("synthetic_rows")
    connection.close()
    return database_path


@pytest.fixture(scope="module")
def trained_synthetic_model(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("classifier")
    database_path = write_shots_database(work_dir / "scout.duckdb", synthetic_shot_rows(5, 12))
    model_path = work_dir / "models" / "classifier.joblib"
    report = train_classifier(database_path, model_path, fold_count=5, permutation_repeats=1)
    return database_path, model_path, report


def all_rows(database_path: Path, sql: str) -> list[tuple]:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def test_feature_names_contain_no_label_or_identity_column():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 16
    for feature_name in FEATURE_NAMES:
        assert not any(word in feature_name for word in LABEL_OR_IDENTITY_WORDS), feature_name


def test_shot_features_do_not_change_when_labels_or_identities_are_shuffled(tmp_path):
    database_path = write_shots_database(tmp_path / "scout.duckdb", synthetic_shot_rows(2, 3))
    connection = duckdb.connect(str(database_path), read_only=True)
    shot_rows = load_shot_feature_rows(connection)
    connection.close()
    shuffled = shot_rows.copy()
    random = np.random.default_rng(1)
    for column_name in ("type_group", "match_id", "source"):
        shuffled[column_name] = random.permutation(shuffled[column_name].to_numpy())
    pd.testing.assert_frame_equal(shot_features(shot_rows), shot_features(shuffled))
    assert list(shot_features(shot_rows).columns) == list(FEATURE_NAMES)


def test_shipped_feature_set_is_feet_and_unknown_sets_are_refused():
    assert SHIPPED_FEATURE_SET == "feet" and set(FEATURE_SET_NAMES) == {"contact", "feet"}
    with pytest.raises(ValueError, match="unknown feature set 'shuttle'"):
        shot_features(pd.DataFrame(synthetic_shot_rows(1, 1)), "shuttle")


def test_feet_features_come_from_the_feet_and_the_reply_hitters_feet_only(tmp_path):
    rows = synthetic_shot_rows(1, 2)
    rows = [{**row, "hit_x": None, "hit_y": None, "landing_x": None, "landing_y": None, "is_cross": None} for row in rows]
    shot_rows = feature_rows_of(write_shots_database(tmp_path / "scout.duckdb", rows)).sort_values(["rally", "ball_round"])
    feet = shot_features(shot_rows, "feet")
    ordered = shot_rows.reset_index(drop=True)
    feet = feet.reset_index(drop=True)
    assert feet["hit_x"].tolist() == pytest.approx(ordered["player_x"].tolist())
    assert feet["hit_y"].tolist() == pytest.approx(ordered["player_y"].tolist())
    assert feet["receiver_x"].tolist() == pytest.approx(ordered["opponent_x"].tolist())
    for position in range(0, len(ordered), SHOTS_PER_RALLY):
        first, second, last = position, position + 1, position + 2
        assert feet.loc[first, "landing_x"] == pytest.approx(COURT_WIDTH_M - ordered.loc[second, "player_x"])
        assert feet.loc[first, "landing_y"] == pytest.approx(COURT_LENGTH_M - ordered.loc[second, "player_y"])
        assert feet.loc[second, "landing_y"] == pytest.approx(COURT_LENGTH_M - ordered.loc[last, "player_y"])
        assert np.isnan(feet.loc[last, "landing_x"]) and np.isnan(feet.loc[last, "landing_y"])
        assert np.isnan(feet.loc[first, "previous_shot_distance"]) and np.isnan(feet.loc[first, "previous_hit_y"])
        assert feet.loc[second, "previous_hit_y"] == pytest.approx(ordered.loc[first, "player_y"])
        expected_previous_distance = np.hypot(
            COURT_WIDTH_M - ordered.loc[second, "player_x"] - ordered.loc[first, "player_x"],
            COURT_LENGTH_M - ordered.loc[second, "player_y"] - ordered.loc[first, "player_y"],
        )
        assert feet.loc[second, "previous_shot_distance"] == pytest.approx(expected_previous_distance)
    assert has_hitter_and_landing_point(feet).tolist() == [True, True, False] * (len(ordered) // SHOTS_PER_RALLY)


def test_feet_is_cross_is_recomputed_from_the_two_feet_points():
    base = {
        "ball_round": 2, "frame_num": 100, "fps": 30.0, "hit_x": None, "hit_y": None, "landing_x": None, "landing_y": None,
        "opponent_x": 3.0, "opponent_y": 9.0, "is_cross": True, "previous_frame_num": None, "next_frame_num": 130,
        "previous_hit_x": None, "previous_hit_y": None, "previous_landing_x": None, "previous_landing_y": None,
        "previous_player_x": None, "previous_player_y": None,
    }
    shot_rows = pd.DataFrame(
        [
            {**base, "player_x": 1.0, "player_y": 2.0, "next_player_x": 1.0, "next_player_y": 2.0},
            {**base, "player_x": 1.0, "player_y": 2.0, "next_player_x": 5.0, "next_player_y": 2.0},
            {**base, "player_x": 2.8, "player_y": 2.0, "next_player_x": 2.6, "next_player_y": 2.0},
            {**base, "player_x": 1.0, "player_y": 2.0, "next_player_x": None, "next_player_y": None},
        ]
    )
    feet = shot_features(shot_rows, "feet")
    assert feet["landing_x"].tolist()[:3] == pytest.approx([COURT_WIDTH_M - 1.0, COURT_WIDTH_M - 5.0, COURT_WIDTH_M - 2.6])
    assert feet["is_cross"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(feet["is_cross"].iloc[3])


def test_contact_features_still_use_the_labelled_points(tmp_path):
    shot_rows = feature_rows_of(write_shots_database(tmp_path / "scout.duckdb", synthetic_shot_rows(1, 2)))
    contact = shot_features(shot_rows, "contact")
    assert contact["hit_x"].tolist() == pytest.approx(shot_rows["hit_x"].tolist())
    assert contact["landing_x"].tolist() == pytest.approx(shot_rows["landing_x"].tolist())
    assert contact["landing_y"].tolist() == pytest.approx(shot_rows["landing_y"].tolist())
    assert contact["is_cross"].tolist() == pytest.approx(shot_rows["is_cross"].astype(float).tolist())
    assert has_hitter_and_landing_point(contact).all()


def test_timing_features_come_from_neighbouring_hit_frames_only(tmp_path):
    rows = synthetic_shot_rows(1, 1)[1:3]
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows)
    connection = duckdb.connect(str(database_path), read_only=True)
    features = shot_features(load_shot_feature_rows(connection)).sort_values("ball_round")
    connection.close()
    assert features["flight_seconds"].iloc[0] == pytest.approx(40 / 25.0)
    assert np.isnan(features["flight_seconds"].iloc[1])
    assert features["seconds_since_previous_hit"].iloc[1] == pytest.approx(40 / 25.0)
    assert np.isnan(features["seconds_since_previous_hit"].iloc[0])


def test_neighbours_never_cross_source_rally_or_match(tmp_path):
    rows = synthetic_shot_rows(1, 1)[1:3]
    own_rows = [{**row, "source": "own", "player": "me", "frame_num": row["frame_num"] + 7} for row in rows]
    other_match = [{**row, "match_id": "other_match"} for row in rows]
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows + own_rows + other_match)
    connection = duckdb.connect(str(database_path), read_only=True)
    shot_rows = load_shot_feature_rows(connection)
    connection.close()
    for (source, match_id), group_rows in shot_rows.groupby(["source", "match_id"]):
        group_rows = group_rows.sort_values("ball_round")
        assert group_rows["previous_frame_num"].isna().iloc[0] and group_rows["next_frame_num"].isna().iloc[-1]
        assert group_rows["next_frame_num"].iloc[0] == group_rows["frame_num"].iloc[1]


class LeakySplitter:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, features, labels, groups):
        all_rows_index = np.arange(len(labels))
        for _ in range(self.n_splits):
            yield all_rows_index[::2], all_rows_index[1::2]


def test_cross_validation_refuses_a_fold_with_a_match_on_both_sides(monkeypatch):
    rows = pd.DataFrame(synthetic_shot_rows(5, 2))
    labelled = rows[rows["type_group"] != "unknown"].reset_index(drop=True)
    features = pd.DataFrame(np.random.default_rng(0).normal(size=(len(labelled), len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    monkeypatch.setattr(shots.train, "GroupKFold", LeakySplitter)
    with pytest.raises(RuntimeError, match="both train and test"):
        grouped_cross_validation(features, labelled["type_group"], labelled["match_id"], fold_count=2, permutation_repeats=1)


def test_real_folds_hold_out_whole_matches():
    rows = pd.DataFrame(synthetic_shot_rows(7, 2))
    labelled = rows[rows["type_group"] != "unknown"].reset_index(drop=True)
    for train_index, test_index in shots.train.GroupKFold(n_splits=5).split(labelled, labelled["type_group"], labelled["match_id"]):
        assert not set(labelled["match_id"].iloc[train_index]) & set(labelled["match_id"].iloc[test_index])


def test_training_excludes_unknown_and_own_rows_and_saves_model_and_report(trained_synthetic_model):
    database_path, model_path, report = trained_synthetic_model
    assert report["training_shots"] == 5 * 12 * len(GROUP_SHOT_PATTERNS)
    assert report["training_matches"] == 5
    assert report["feature_set"] == "feet"
    assert set(report["group_counts"]) == set(TRAINED_GROUPS)
    assert model_path.is_file() and model_path.with_name("classifier_report.json").is_file()
    assert json.loads(model_path.with_name("classifier_report.json").read_text())["accuracy_mean"] == report["accuracy_mean"]
    assert report["accuracy_mean"] > report["majority_accuracy_mean"]
    assert all(fold["test_matches"] == 1 for fold in report["folds"])
    model = joblib.load(model_path)
    assert tuple(model["feature_names"]) == FEATURE_NAMES
    assert model["feature_set"] == "feet"
    assert "unknown" not in model["groups"]


def test_training_report_scores_every_fold_on_jittered_hit_frames_too(trained_synthetic_model):
    _, _, report = trained_synthetic_model
    assert list(report["variants"]) == [jitter_label(offset) for offset in JITTER_OFFSETS_FRAMES] == ["hit frames +/- 5", "hit frames +/- 10"]
    for scores in report["variants"].values():
        assert 0.0 <= scores["accuracy_mean"] <= 1.0 and 0.0 <= scores["out_of_fold_accuracy"] <= 1.0
        assert set(scores["per_group"]) == set(TRAINED_GROUPS)
        assert sum(group["support"] for group in scores["per_group"].values()) == report["training_shots"]
    assert report["subsets"]["rally's last shot (no next hit, so no flight time)"] == {"shots": 0, "accuracy": None}
    assert report["subsets"]["all other shots"]["shots"] == report["training_shots"]


def test_training_on_the_contact_set_evaluates_the_labelled_points(trained_synthetic_model, tmp_path):
    database_path, _, feet_report = trained_synthetic_model
    report = train_classifier(database_path, tmp_path / "contact.joblib", fold_count=5, permutation_repeats=0, feature_set="contact")
    assert report["feature_set"] == "contact" and joblib.load(tmp_path / "contact.joblib")["feature_set"] == "contact"
    assert report["training_shots"] == feet_report["training_shots"]
    assert all(np.isnan(value) for value in report["permutation_importance"].values())
    assert list(report["variants"]) == list(feet_report["variants"])


def test_predict_fills_only_type_pred_and_type_conf_for_every_source(trained_synthetic_model, tmp_path):
    source_database, model_path, _ = trained_synthetic_model
    database_path = tmp_path / "predict.duckdb"
    database_path.write_bytes(source_database.read_bytes())
    connection = duckdb.connect(str(database_path))
    connection.execute(
        "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, player_x, player_y, opponent_x, opponent_y) "
        "VALUES ('own', 'my_game', 1, 1, 1, 10, 30.0, 'me', 3.0, 5.5, 3.1, 7.5), ('own', 'my_game', 1, 1, 2, 30, 30.0, 'opp', 3.0, 5.6, 3.1, 7.9), "
        "('own', 'my_game', 1, 1, 3, 60, 30.0, 'me', 2.0, 1.0, 3.0, 8.0)"
    )
    connection.execute("CREATE TABLE before_predict AS SELECT * FROM shots")
    connection.close()
    assert predict_shot_types(database_path, model_path) == predictable_synthetic_shots(5, 12) + 2
    other_columns = "* EXCLUDE (type_pred, type_conf)"
    for first_table, second_table in (("shots", "before_predict"), ("before_predict", "shots")):
        assert all_rows(database_path, f"SELECT COUNT(*) FROM (SELECT {other_columns} FROM {first_table} EXCEPT ALL SELECT {other_columns} FROM {second_table})") == [(0,)]
    assert all_rows(database_path, "SELECT COUNT(*) FROM shots WHERE (shots.type_pred IS NULL) <> (shots.type_conf IS NULL)") == [(0,)]
    last_shot_of_rally = (
        "shots.ball_round = (SELECT MAX(last_shots.ball_round) FROM shots AS last_shots WHERE last_shots.source = shots.source "
        "AND last_shots.match_id = shots.match_id AND last_shots.set_no = shots.set_no AND last_shots.rally = shots.rally)"
    )
    assert all_rows(database_path, f"SELECT COUNT(*) FROM shots WHERE (shots.type_pred IS NULL) <> ({last_shot_of_rally})") == [(0,)]
    assert all_rows(database_path, "SELECT COUNT(*) FROM shots WHERE shots.type_conf < 0 OR shots.type_conf > 1 OR shots.type_pred NOT IN ('serve', 'clear', 'drop', 'smash', 'net', 'lift', 'drive')") == [(0,)]
    assert all_rows(database_path, "SELECT shots.ball_round FROM shots WHERE shots.source = 'own' AND shots.type_pred IS NOT NULL ORDER BY shots.ball_round") == [(1,), (2,)]


def test_predict_twice_gives_the_same_table(trained_synthetic_model, tmp_path):
    source_database, model_path, _ = trained_synthetic_model
    database_path = tmp_path / "twice.duckdb"
    database_path.write_bytes(source_database.read_bytes())
    predict_shot_types(database_path, model_path)
    once = all_rows(database_path, "SELECT * FROM shots ORDER BY shots.match_id, shots.rally, shots.ball_round")
    predict_shot_types(database_path, model_path)
    assert all_rows(database_path, "SELECT * FROM shots ORDER BY shots.match_id, shots.rally, shots.ball_round") == once


def test_load_model_missing_or_with_other_features_raises(trained_synthetic_model, tmp_path):
    _, model_path, _ = trained_synthetic_model
    with pytest.raises(FileNotFoundError, match="model not found"):
        load_model(tmp_path / "missing.joblib")
    model = joblib.load(model_path)
    model["feature_names"] = FEATURE_NAMES[:-1]
    stale_model_path = tmp_path / "stale.joblib"
    joblib.dump(model, stale_model_path)
    with pytest.raises(ModelFeatureMismatchError):
        load_model(stale_model_path)


def test_load_model_refuses_a_model_trained_on_the_contact_set(trained_synthetic_model, tmp_path):
    _, model_path, _ = trained_synthetic_model
    contact_model = {**joblib.load(model_path), "feature_set": "contact"}
    contact_model_path = tmp_path / "contact.joblib"
    joblib.dump(contact_model, contact_model_path)
    with pytest.raises(ModelFeatureMismatchError, match="'contact' feature set") as raised:
        load_model(contact_model_path)
    assert raised.value.model_feature_set == "contact" and raised.value.model_feature_names == FEATURE_NAMES
    model_without_set = {key: value for key, value in joblib.load(model_path).items() if key != "feature_set"}
    joblib.dump(model_without_set, tmp_path / "old.joblib")
    with pytest.raises(ModelFeatureMismatchError):
        load_model(tmp_path / "old.joblib")
    assert load_model(model_path)["feature_set"] == "feet"


def documented_table_rows(heading: str) -> list[list[str]]:
    section = RESULTS_PATH.read_text().split(heading)[1].split("\n## ")[0]
    table_lines = [line for line in section.splitlines() if line.startswith("|")][2:]
    return [[cell.strip().strip("*") for cell in line.strip("|").split("|")] for line in table_lines]


@pytest.mark.skipif(not REAL_REPORT_PATH.exists(), reason="run python -m shots.train first")
def test_results_doc_numbers_equal_the_training_report():
    report = json.loads(REAL_REPORT_PATH.read_text())
    results_text = RESULTS_PATH.read_text()
    assert f"**{report['accuracy_mean']:.1%}**" in results_text
    assert f"{report['majority_accuracy_mean']:.1%}" in results_text
    for group, shot_count, precision, recall, f1 in documented_table_rows("## Per group (held-out)"):
        scores = report["per_group"][group]
        assert (int(shot_count.replace(",", "")), precision, recall, f1) == (
            scores["support"],
            f"{scores['precision']:.3f}",
            f"{scores['recall']:.3f}",
            f"{scores['f1']:.3f}",
        )
    documented_matrix = [[int(cell.replace(",", "")) for cell in row[1:]] for row in documented_table_rows("## Confusion matrix")]
    assert documented_matrix == report["confusion_matrix"]["rows_true_columns_predicted"]
    for feature_name, accuracy_drop in documented_table_rows("## Feature importance"):
        assert accuracy_drop == f"{report['permutation_importance'][feature_name]:.3f}", feature_name
    assert report["feature_set"] == "feet"
    assert f"**{report['training_shots']:,} shots from {report['training_matches']} matches**" in results_text
    jitter_rows = {row[0]: row for row in documented_table_rows("## Timing jitter")}
    assert jitter_rows["exact"][3] == f"{report['accuracy_mean']:.1%}"
    for max_offset in JITTER_OFFSETS_FRAMES:
        jittered_accuracy = report["variants"][jitter_label(max_offset)]["accuracy_mean"]
        assert jitter_rows[f"+/- {max_offset}"][3] == f"{jittered_accuracy:.1%} ({(jittered_accuracy - report['accuracy_mean']) * 100:+.1f})"


@pytest.mark.parametrize("source_path", sorted((PROJECT_ROOT / "shots").glob("*.py")), ids=lambda path: path.name)
def test_shots_source_has_no_comments_or_docstrings(source_path):
    source_text = source_path.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.mark.parametrize("doc_name", ["analytics/classifier.md", "results/classifier.md", "results/classifier_feet.md"])
def test_classifier_docs_have_no_task_ids(doc_name):
    assert re.findall(TASK_ID_PATTERN, (PROJECT_ROOT / "docs" / doc_name).read_text()) == []


FEET = {"player_x": 3.0, "player_y": 5.5}
NO_FEET = {"player_x": None, "player_y": None}
LABELLED_POINTS = {"hit_x": 3.0, "hit_y": 5.5, "landing_x": 2.8, "landing_y": 7.4}
NO_LABELLED_POINTS = {"hit_x": None, "hit_y": None, "landing_x": None, "landing_y": None}
GEOMETRY_CASES = {
    "feet and a reply with feet": [{**FEET, **NO_LABELLED_POINTS}, {**FEET, **NO_LABELLED_POINTS}],
    "feet and labelled points, reply with feet": [{**FEET, **LABELLED_POINTS}, {**FEET, **LABELLED_POINTS}],
    "feet, no reply": [{**FEET, **LABELLED_POINTS}],
    "feet, reply without feet": [{**FEET, **LABELLED_POINTS}, {**NO_FEET, **LABELLED_POINTS}],
    "labelled points only, reply with feet": [{**NO_FEET, **LABELLED_POINTS}, {**FEET, **LABELLED_POINTS}],
    "nothing": [{**NO_FEET, **NO_LABELLED_POINTS}, {**NO_FEET, **NO_LABELLED_POINTS}],
}
EXPECTED_PREDICTED = {
    "feet and a reply with feet": True,
    "feet and labelled points, reply with feet": True,
    "feet, no reply": False,
    "feet, reply without feet": False,
    "labelled points only, reply with feet": False,
    "nothing": False,
}


def insert_own_geometry_cases(database_path: Path) -> None:
    connection = duckdb.connect(str(database_path))
    for rally, (case_name, shots_of_case) in enumerate(GEOMETRY_CASES.items(), start=1):
        for ball_round, points in enumerate(shots_of_case, start=2):
            connection.execute(
                "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, type_pred, type_conf, "
                "hit_x, hit_y, player_x, player_y, landing_x, landing_y) VALUES ('own', ?, 1, ?, ?, ?, 30.0, 'me', 'smash', 0.99, ?, ?, ?, ?, ?, ?)",
                [case_name, rally, ball_round, 100 + 30 * ball_round, points["hit_x"], points["hit_y"], points["player_x"], points["player_y"], points["landing_x"], points["landing_y"]],
            )
    connection.close()


def test_predict_leaves_rows_without_feet_or_reply_feet_null_and_clears_stale_predictions(trained_synthetic_model, tmp_path):
    source_database, model_path, _ = trained_synthetic_model
    database_path = tmp_path / "geometry.duckdb"
    database_path.write_bytes(source_database.read_bytes())
    insert_own_geometry_cases(database_path)
    predicted_count = predict_shot_types(database_path, model_path)
    own_predictions = dict(
        (match_id, (type_pred, type_conf))
        for match_id, type_pred, type_conf in all_rows(
            database_path, "SELECT shots.match_id, shots.type_pred, shots.type_conf FROM shots WHERE shots.source = 'own' AND shots.ball_round = 2"
        )
    )
    for case_name, should_predict in EXPECTED_PREDICTED.items():
        type_pred, type_conf = own_predictions[case_name]
        assert (type_pred is not None) == should_predict, case_name
        assert (type_conf is not None) == should_predict, case_name
    assert all_rows(database_path, "SELECT COUNT(*) FROM shots WHERE shots.source = 'own' AND shots.ball_round = 3 AND shots.type_pred IS NOT NULL") == [(0,)]
    assert predicted_count == all_rows(database_path, "SELECT COUNT(shots.type_pred) FROM shots")[0][0]


def test_training_drops_labelled_shots_without_feet_or_reply_feet_but_keeps_those_without_labelled_points(tmp_path):
    rows = synthetic_shot_rows(5, 12)
    first_of = lambda group: next(index for index, row in enumerate(rows) if row["type_group"] == group)
    rows[first_of("drop")] = {**rows[first_of("drop")], "landing_x": None, "landing_y": None, "hit_x": None, "hit_y": None}
    rows[first_of("clear")] = {**rows[first_of("clear")], "player_x": None, "player_y": None}
    rows[first_of("smash") + 1] = {**rows[first_of("smash") + 1], "player_x": None, "player_y": None}
    del rows[first_of("lift") + 1]
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows)
    report = train_classifier(database_path, tmp_path / "models" / "classifier.joblib", fold_count=5, permutation_repeats=1)
    assert report["training_shots"] == 5 * 12 * len(GROUP_SHOT_PATTERNS) - 3
    assert report["group_counts"] == {**{group: 5 * 12 for group in TRAINED_GROUPS}, "clear": 5 * 12 - 1, "smash": 5 * 12 - 1, "lift": 5 * 12 - 1}


REAL_COMPARISON_PATH = PROJECT_ROOT / "data" / "models" / "feature_set_comparison.json"
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"


@pytest.mark.skipif(not REAL_REPORT_PATH.exists() or not REAL_COMPARISON_PATH.exists(), reason="run shots.train and shots.compare_features first")
def test_shipped_model_row_of_the_comparison_equals_the_training_report_fold_by_fold():
    report = json.loads(REAL_REPORT_PATH.read_text())
    comparison = json.loads(REAL_COMPARISON_PATH.read_text())
    assert comparison["shipped model"]["feature_names"] == list(FEATURE_NAMES)
    assert comparison["shipped model"]["feature_set"] == "feet"
    assert comparison["shipped model"]["fold_accuracies"] == [fold["accuracy"] for fold in report["folds"]]


REAL_HIT_POINT_COMPARISON_PATH = PROJECT_ROOT / "data" / "models" / "hit_point_comparison.json"


@pytest.mark.skipif(
    not REAL_REPORT_PATH.exists() or not REAL_HIT_POINT_COMPARISON_PATH.exists(), reason="run shots.train and shots.compare_hit_points first"
)
def test_feet_on_all_its_shots_in_the_hit_point_comparison_equals_the_training_report():
    report = json.loads(REAL_REPORT_PATH.read_text())
    comparison = json.loads(REAL_HIT_POINT_COMPARISON_PATH.read_text())
    feet = comparison["all_shots"]["feet"]
    assert feet["shots"] == report["training_shots"] and feet["matches"] == report["training_matches"]
    assert [fold["accuracy"] for fold in feet["folds"]] == [fold["accuracy"] for fold in report["folds"]]
    assert feet["confusion_matrix"] == report["confusion_matrix"]
    assert {name: scores["accuracy_mean"] for name, scores in feet["variants"].items()} == {
        name: scores["accuracy_mean"] for name, scores in report["variants"].items()
    }
    shared_counts = {feature_set: comparison["shared_shots"][feature_set]["shots"] for feature_set in FEATURE_SET_NAMES}
    assert len(set(shared_counts.values())) == 1
    assert shared_counts["feet"] <= min(comparison["all_shots"][feature_set]["shots"] for feature_set in FEATURE_SET_NAMES)
    assert comparison["landing_offset_m"]["shots"] == shared_counts["feet"]


def results_doc_sections(tables: str) -> list[tuple[str, str]]:
    return [tuple(section.split("\n\n", 1)) for section in tables.split("### ")[1:]]


@pytest.mark.skipif(
    not REAL_REPORT_PATH.exists() or not REAL_COMPARISON_PATH.exists() or not REAL_DATABASE_PATH.exists() or not REAL_HIT_POINT_COMPARISON_PATH.exists(),
    reason="run shots.train, shots.compare_features, shots.compare_hit_points and shots.predict first",
)
def test_every_generated_results_table_appears_verbatim_in_the_results_doc():
    from shots.predict import prediction_summary
    from shots.results_tables import hit_point_results_tables, results_tables

    tables = results_tables(
        json.loads(REAL_REPORT_PATH.read_text()), json.loads(REAL_COMPARISON_PATH.read_text()), prediction_summary(REAL_DATABASE_PATH)
    )
    hit_point_tables = hit_point_results_tables(json.loads(REAL_HIT_POINT_COMPARISON_PATH.read_text()))
    results_text = RESULTS_PATH.read_text()
    sections = results_doc_sections(tables)
    hit_point_sections = results_doc_sections(hit_point_tables)
    assert len(sections) == 8 and len(hit_point_sections) == 6
    for title, table in sections + hit_point_sections:
        assert table.strip() in results_text, title
