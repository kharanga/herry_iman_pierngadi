import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from shots.features import FEATURE_NAMES, has_hitter_and_landing_point, load_shot_feature_rows, shot_features

DATABASE_PATH = Path("data/scout.duckdb")
MODEL_PATH = Path("data/models/shot_type_classifier.joblib")

UPDATE_PREDICTIONS_SQL = """
UPDATE shots
SET type_pred = shot_predictions.type_pred, type_conf = shot_predictions.type_conf
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
        FILTER (WHERE shots.type_pred IS NOT NULL AND shots.type_group <> 'unknown') AS in_sample_agreement
FROM shots
WHERE shots.source = 'shuttleset'
"""

UNKNOWN_SHOT_PREDICTIONS_SQL = """
SELECT shots.type_pred, COUNT(*) AS shot_count
FROM shots
WHERE shots.source = 'shuttleset' AND shots.type_group = 'unknown'
GROUP BY shots.type_pred
ORDER BY shot_count DESC, shots.type_pred
"""


class ModelFeatureMismatchError(RuntimeError):
    def __init__(self, model_feature_names: tuple[str, ...]):
        super().__init__(
            f"model was trained on features {list(model_feature_names)}, "
            f"but shots/features.py now produces {list(FEATURE_NAMES)}; retrain with python -m shots.train"
        )
        self.model_feature_names = model_feature_names


def load_model(model_path: Path) -> dict:
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"model not found: {model_path} (run python -m shots.train first)")
    model = joblib.load(model_path)
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ModelFeatureMismatchError(tuple(model["feature_names"]))
    return model


def predictions_for(model: dict, shot_rows: pd.DataFrame) -> pd.DataFrame:
    features = shot_features(shot_rows)
    is_predictable = has_hitter_and_landing_point(features).to_numpy()
    predictions = pd.DataFrame(
        {
            "row_id": shot_rows["row_id"].to_numpy(),
            "type_pred": pd.Series([None] * len(shot_rows), dtype=object),
            "type_conf": pd.Series([None] * len(shot_rows), dtype="Float64"),
        }
    )
    if is_predictable.any():
        probabilities = model["classifier"].predict_proba(features[is_predictable])
        predictions.loc[is_predictable, "type_pred"] = model["classifier"].classes_[probabilities.argmax(axis=1)]
        predictions.loc[is_predictable, "type_conf"] = probabilities.max(axis=1)
    return predictions


def prediction_summary(database_path: Path = DATABASE_PATH) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cursor = connection.execute(PREDICTION_SUMMARY_SQL)
        summary = dict(zip([description[0] for description in cursor.description], cursor.fetchone()))
        summary["unknown_shot_predictions"] = dict(connection.execute(UNKNOWN_SHOT_PREDICTIONS_SQL).fetchall())
    finally:
        connection.close()
    return summary


def predict_shot_types(database_path: Path = DATABASE_PATH, model_path: Path = MODEL_PATH) -> int:
    model = load_model(model_path)
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


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_PATH
    print(f"type_pred and type_conf filled for {predict_shot_types(database_path, model_path)} shots; "
          "shots without both a hitter point and a landing point are left NULL")
