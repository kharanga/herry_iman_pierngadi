import argparse
import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from analytics.shot_types import TYPE_GROUP_BY_TYPE_DETAIL
from shots.features import FEATURE_NAMES, SHIPPED_FEATURE_SET, has_hitter_and_landing_point, load_shot_feature_rows, shot_features

DATABASE_PATH = Path("data/scout.duckdb")
MODEL_PATH = Path("data/models/shot_type_classifier.joblib")

UPDATE_PREDICTIONS_SQL = """
UPDATE shots
SET
    type_detail_pred = shot_predictions.type_detail_pred,
    type_pred = shot_predictions.type_pred,
    type_conf = shot_predictions.type_conf
FROM shot_predictions
WHERE shots.rowid = shot_predictions.row_id
"""


PREDICTION_SUMMARY_SQL = """
SELECT
    COUNT(*) AS shots,
    COUNT(shots.type_pred) AS predicted_shots,
    MIN(shots.type_conf) AS lowest_conf,
    AVG(shots.type_conf) AS mean_conf,
    AVG(CASE WHEN shots.type_pred = shots.type_group THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE shots.type_pred IS NOT NULL AND shots.type_group <> 'unknown') AS in_sample_agreement,
    AVG(CASE WHEN shots.type_detail_pred = shots.type_detail THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE shots.type_detail_pred IS NOT NULL AND shots.type_detail <> 'unknown') AS in_sample_detail_agreement
FROM shots
WHERE shots.source = 'shuttleset'
"""

OWN_PREDICTION_SUMMARY_SQL = """
SELECT
    shots.match_id,
    COUNT(*) AS shots,
    COUNT(shots.type_pred) AS predicted_shots,
    COUNT(DISTINCT (shots.set_no, shots.rally)) AS rallies
FROM shots
WHERE shots.source = 'own'
GROUP BY shots.match_id
ORDER BY shots.match_id
"""

UNKNOWN_SHOT_PREDICTIONS_SQL = """
SELECT shots.type_pred, COUNT(*) AS shot_count
FROM shots
WHERE shots.source = 'shuttleset' AND shots.type_group = 'unknown'
GROUP BY shots.type_pred
ORDER BY shot_count DESC, shots.type_pred
"""


class ModelFeatureMismatchError(RuntimeError):
    def __init__(self, model_feature_names: tuple[str, ...], model_feature_set: str | None = None):
        super().__init__(
            f"model was trained on the {model_feature_set!r} feature set with features {list(model_feature_names)}, "
            f"but shots/features.py now produces the {SHIPPED_FEATURE_SET!r} set {list(FEATURE_NAMES)}; retrain with python -m shots.train"
        )
        self.model_feature_names = model_feature_names
        self.model_feature_set = model_feature_set


class ModelLabelMismatchError(RuntimeError):
    def __init__(self, model_trained_label: str | None):
        super().__init__(
            f"model was trained on {model_trained_label or 'type_group'!r}, but shots.predict fills type_detail_pred and type_pred "
            "from a model trained on 'type_detail'; retrain with python -m shots.train"
        )
        self.model_trained_label = model_trained_label


def load_model(model_path: Path) -> dict:
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"model not found: {model_path} (run python -m shots.train first)")
    model = joblib.load(model_path)
    if tuple(model["feature_names"]) != FEATURE_NAMES or model.get("feature_set") != SHIPPED_FEATURE_SET:
        raise ModelFeatureMismatchError(tuple(model["feature_names"]), model.get("feature_set"))
    if model.get("trained_label") != "type_detail":
        raise ModelLabelMismatchError(model.get("trained_label"))
    return model


def predictions_for(model: dict, shot_rows: pd.DataFrame) -> pd.DataFrame:
    features = shot_features(shot_rows)
    is_predictable = has_hitter_and_landing_point(features).to_numpy()
    predictions = pd.DataFrame(
        {
            "row_id": shot_rows["row_id"].to_numpy(),
            "type_detail_pred": pd.Series([None] * len(shot_rows), dtype=object),
            "type_pred": pd.Series([None] * len(shot_rows), dtype=object),
            "type_conf": pd.Series([None] * len(shot_rows), dtype="Float64"),
        }
    )
    if is_predictable.any():
        probabilities = model["classifier"].predict_proba(features[is_predictable])
        predicted_details = model["classifier"].classes_[probabilities.argmax(axis=1)]
        predictions.loc[is_predictable, "type_detail_pred"] = predicted_details
        predictions.loc[is_predictable, "type_pred"] = [TYPE_GROUP_BY_TYPE_DETAIL[detail] for detail in predicted_details]
        predictions.loc[is_predictable, "type_conf"] = probabilities.max(axis=1)
    return predictions


def prediction_summary(database_path: Path = DATABASE_PATH) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cursor = connection.execute(PREDICTION_SUMMARY_SQL)
        summary = dict(zip([description[0] for description in cursor.description], cursor.fetchone()))
        summary["unknown_shot_predictions"] = dict(connection.execute(UNKNOWN_SHOT_PREDICTIONS_SQL).fetchall())
        summary["own_matches"] = {
            match_id: {"shots": shots, "predicted_shots": predicted_shots, "rallies": rallies}
            for match_id, shots, predicted_shots, rallies in connection.execute(OWN_PREDICTION_SUMMARY_SQL).fetchall()
        }
    finally:
        connection.close()
    return summary


def require_existing_database(database_path: Path) -> None:
    if not Path(database_path).is_file():
        raise FileNotFoundError(f"database not found: {database_path} (build it with python -m ingest.load_shuttleset)")


def predict_shot_types(database_path: Path = DATABASE_PATH, model_path: Path = MODEL_PATH) -> int:
    model = load_model(model_path)
    require_existing_database(database_path)
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        shot_rows = load_shot_feature_rows(connection)
        shot_predictions = predictions_for(model, shot_rows)
        connection.register("shot_predictions", shot_predictions)
        connection.execute(UPDATE_PREDICTIONS_SQL)
        connection.unregister("shot_predictions")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return int(shot_predictions["type_pred"].notna().sum())


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m shots.predict",
        description="Fill type_detail_pred, type_pred and type_conf for every shot with a hitter point and a landing point.",
    )
    parser.add_argument("database_path", nargs="?", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("model_path", nargs="?", type=Path, default=MODEL_PATH, help=f"trained model (default {MODEL_PATH})")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        predicted_count = predict_shot_types(parsed.database_path, parsed.model_path)
    except (FileNotFoundError, ModelFeatureMismatchError, ModelLabelMismatchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"type_detail_pred, type_pred and type_conf filled for {predicted_count} shots; "
          "shots without both a hitter point and a landing point are left NULL")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
