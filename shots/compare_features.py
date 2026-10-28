import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from shots.features import FEATURE_NAMES, shot_features
from shots.train import DATABASE_PATH, FOLD_COUNT, RANDOM_SEED, TRAINED_GROUPS, labelled_shuttleset_rows, new_classifier

COMPARISON_PATH = Path("data/models/feature_set_comparison.json")
GEOMETRY_FEATURE_NAMES = (
    "hit_x",
    "hit_y",
    "landing_x",
    "landing_y",
    "receiver_x",
    "receiver_y",
    "shot_dx",
    "shot_dy",
    "shot_distance",
    "shot_lateral_distance",
    "is_cross",
    "ball_round",
)
TIMING_FEATURE_NAMES = ("flight_seconds", "seconds_since_previous_hit")
PREVIOUS_TRUE_GROUP_FEATURE = "previous_true_group"
FEATURE_SETS = {
    "geometry only": GEOMETRY_FEATURE_NAMES,
    "+ timing": GEOMETRY_FEATURE_NAMES + TIMING_FEATURE_NAMES,
    "shipped model": FEATURE_NAMES,
    "shipped model + previous shot's true group": FEATURE_NAMES + (PREVIOUS_TRUE_GROUP_FEATURE,),
}

PREVIOUS_TRUE_GROUP_SQL = """
SELECT
    shots.rowid AS row_id,
    LAG(shots.type_group) OVER (
        PARTITION BY shots.source, shots.match_id, shots.set_no, shots.rally
        ORDER BY shots.ball_round, shots.frame_num
    ) AS previous_true_group
FROM shots
"""


def previous_true_group_codes(connection: duckdb.DuckDBPyConnection, row_ids: pd.Series) -> np.ndarray:
    previous_groups = connection.execute(PREVIOUS_TRUE_GROUP_SQL).df().set_index("row_id")["previous_true_group"]
    group_codes = {group: code for code, group in enumerate(TRAINED_GROUPS + ("unknown",))}
    return previous_groups.reindex(row_ids.to_numpy()).map(group_codes).to_numpy(float)


def grouped_accuracy(features: pd.DataFrame, labels: pd.Series, match_ids: pd.Series, fold_count: int, random_seed: int) -> list[float]:
    categorical_columns = [PREVIOUS_TRUE_GROUP_FEATURE] if PREVIOUS_TRUE_GROUP_FEATURE in features.columns else None
    fold_accuracies = []
    for train_index, test_index in GroupKFold(n_splits=fold_count).split(features, labels, match_ids):
        classifier = new_classifier(random_seed).set_params(categorical_features=categorical_columns)
        classifier.fit(features.iloc[train_index], labels.iloc[train_index])
        fold_accuracies.append(float(accuracy_score(labels.iloc[test_index], classifier.predict(features.iloc[test_index]))))
    return fold_accuracies


def compare_feature_sets(
    database_path: Path = DATABASE_PATH,
    comparison_path: Path = COMPARISON_PATH,
    fold_count: int = FOLD_COUNT,
    random_seed: int = RANDOM_SEED,
) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        labelled_rows = labelled_shuttleset_rows(connection)
        all_features = shot_features(labelled_rows)
        all_features[PREVIOUS_TRUE_GROUP_FEATURE] = previous_true_group_codes(connection, labelled_rows["row_id"])
    finally:
        connection.close()
    comparison = {}
    for set_name, feature_names in FEATURE_SETS.items():
        fold_accuracies = grouped_accuracy(
            all_features[list(feature_names)], labelled_rows["type_group"], labelled_rows["match_id"], fold_count, random_seed
        )
        comparison[set_name] = {
            "feature_names": list(feature_names),
            "fold_accuracies": fold_accuracies,
            "accuracy_mean": float(np.mean(fold_accuracies)),
        }
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    comparison_path = Path(sys.argv[2]) if len(sys.argv) > 2 else COMPARISON_PATH
    for set_name, result in compare_feature_sets(database_path, comparison_path).items():
        print(f"{set_name}: {result['accuracy_mean']:.1%}")
    print(f"saved {comparison_path}")
