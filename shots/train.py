import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from analytics.shot_types import SHOT_TYPE_GROUPS
from shots.features import FEATURE_NAMES, has_hitter_and_landing_point, load_shot_feature_rows, shot_features

DATABASE_PATH = Path("data/scout.duckdb")
MODEL_PATH = Path("data/models/shot_type_classifier.joblib")
TRAINED_GROUPS = tuple(group for group in SHOT_TYPE_GROUPS if group != "unknown")
FOLD_COUNT = 5
RANDOM_SEED = 0
PERMUTATION_REPEATS = 3


def new_classifier(random_seed: int = RANDOM_SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=random_seed)


def labelled_shuttleset_rows(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    shot_rows = load_shot_feature_rows(connection)
    is_labelled = (shot_rows["source"] == "shuttleset") & shot_rows["type_group"].isin(TRAINED_GROUPS)
    is_predictable = has_hitter_and_landing_point(shot_features(shot_rows))
    return shot_rows[is_labelled & is_predictable].reset_index(drop=True)


def labelled_shuttleset_shots(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    labelled_rows = labelled_shuttleset_rows(connection)
    return shot_features(labelled_rows), labelled_rows["type_group"], labelled_rows["match_id"]


def majority_group(labels: pd.Series) -> str:
    return labels.value_counts().sort_index().idxmax()


def per_group_scores(true_groups: pd.Series, predicted_groups: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_groups, predicted_groups, labels=list(TRAINED_GROUPS), zero_division=0
    )
    return {
        group: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
        for index, group in enumerate(TRAINED_GROUPS)
    }


def grouped_cross_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    match_ids: pd.Series,
    fold_count: int = FOLD_COUNT,
    random_seed: int = RANDOM_SEED,
    permutation_repeats: int = PERMUTATION_REPEATS,
    breakdown_rows: pd.DataFrame | None = None,
) -> dict:
    out_of_fold_predictions = np.empty(len(labels), dtype=object)
    folds = []
    importance_per_fold = []
    for fold_index, (train_index, test_index) in enumerate(GroupKFold(n_splits=fold_count).split(features, labels, match_ids)):
        train_matches = set(match_ids.iloc[train_index])
        test_matches = set(match_ids.iloc[test_index])
        if train_matches & test_matches:
            raise RuntimeError(f"fold {fold_index} has matches in both train and test")
        classifier = new_classifier(random_seed).fit(features.iloc[train_index], labels.iloc[train_index])
        predictions = classifier.predict(features.iloc[test_index])
        out_of_fold_predictions[test_index] = predictions
        baseline_group = majority_group(labels.iloc[train_index])
        folds.append(
            {
                "fold": fold_index,
                "test_matches": len(test_matches),
                "test_shots": len(test_index),
                "accuracy": float(accuracy_score(labels.iloc[test_index], predictions)),
                "majority_group": baseline_group,
                "majority_accuracy": float((labels.iloc[test_index] == baseline_group).mean()),
            }
        )
        importance = permutation_importance(
            classifier,
            features.iloc[test_index],
            labels.iloc[test_index],
            n_repeats=permutation_repeats,
            random_state=random_seed,
            scoring="accuracy",
        )
        importance_per_fold.append(importance.importances_mean)
    mean_importance = np.mean(importance_per_fold, axis=0)
    breakdowns = {} if breakdown_rows is None else out_of_fold_breakdowns(breakdown_rows, labels, out_of_fold_predictions)
    return {
        **breakdowns,
        "folds": folds,
        "accuracy_mean": float(np.mean([fold["accuracy"] for fold in folds])),
        "accuracy_std": float(np.std([fold["accuracy"] for fold in folds])),
        "majority_accuracy_mean": float(np.mean([fold["majority_accuracy"] for fold in folds])),
        "out_of_fold_accuracy": float(accuracy_score(labels, out_of_fold_predictions)),
        "per_group": per_group_scores(labels, out_of_fold_predictions),
        "confusion_matrix": {
            "labels": list(TRAINED_GROUPS),
            "rows_true_columns_predicted": confusion_matrix(labels, out_of_fold_predictions, labels=list(TRAINED_GROUPS)).tolist(),
        },
        "permutation_importance": dict(
            sorted(
                ((name, float(value)) for name, value in zip(FEATURE_NAMES, mean_importance)),
                key=lambda name_and_value: -name_and_value[1],
            )
        ),
    }


def accuracy_on(is_correct: pd.Series, mask: pd.Series) -> dict:
    return {"shots": int(mask.sum()), "accuracy": float(is_correct[mask].mean()) if mask.any() else None}


def out_of_fold_breakdowns(breakdown_rows: pd.DataFrame, labels: pd.Series, out_of_fold_predictions: np.ndarray) -> dict:
    predictions = pd.Series(out_of_fold_predictions, index=labels.index)
    is_correct = predictions == labels
    is_last_shot = breakdown_rows["next_frame_num"].isna()
    per_type_en = {}
    for type_en, type_rows in breakdown_rows.groupby("type_en"):
        type_index = type_rows.index
        wrong_predictions = predictions[type_index][~is_correct[type_index]]
        per_type_en[type_en] = {
            "group": labels[type_index].iloc[0],
            "shots": len(type_index),
            "recall": float(is_correct[type_index].mean()),
            "top_mistakes": {group: int(count) for group, count in wrong_predictions.value_counts().head(2).items()},
        }
    return {
        "subsets": {
            "excluding each rally's first shot": accuracy_on(is_correct, breakdown_rows["ball_round"] > 1),
            "rally's last shot (no next hit, so no flight time)": accuracy_on(is_correct, is_last_shot),
            "all other shots": accuracy_on(is_correct, ~is_last_shot),
        },
        "per_type_en": dict(sorted(per_type_en.items(), key=lambda type_and_scores: type_and_scores[1]["recall"])),
    }


def save_model(classifier: HistGradientBoostingClassifier, model_path: Path, training_shots: int, training_matches: int) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_names": FEATURE_NAMES,
            "groups": tuple(classifier.classes_),
            "sklearn_version": sklearn.__version__,
            "training_shots": training_shots,
            "training_matches": training_matches,
        },
        model_path,
    )


def report_path_for(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_report.json")


def train_classifier(
    database_path: Path = DATABASE_PATH,
    model_path: Path = MODEL_PATH,
    fold_count: int = FOLD_COUNT,
    random_seed: int = RANDOM_SEED,
    permutation_repeats: int = PERMUTATION_REPEATS,
) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        labelled_rows = labelled_shuttleset_rows(connection)
    finally:
        connection.close()
    features, labels, match_ids = shot_features(labelled_rows), labelled_rows["type_group"], labelled_rows["match_id"]
    report = grouped_cross_validation(
        features, labels, match_ids, fold_count, random_seed, permutation_repeats, breakdown_rows=labelled_rows
    )
    final_classifier = new_classifier(random_seed).fit(features, labels)
    save_model(final_classifier, model_path, len(labels), match_ids.nunique())
    report.update(
        {
            "training_shots": len(labels),
            "training_matches": int(match_ids.nunique()),
            "group_counts": {group: int(count) for group, count in labels.value_counts().reindex(TRAINED_GROUPS, fill_value=0).items()},
            "model_path": str(model_path),
        }
    )
    report_path_for(model_path).write_text(json.dumps(report, indent=2))
    return report


def summary_lines(report: dict) -> list[str]:
    return [
        f"trained on {report['training_shots']} shots from {report['training_matches']} matches",
        f"held-out accuracy {report['accuracy_mean']:.3f} +/- {report['accuracy_std']:.3f} over {len(report['folds'])} match folds",
        f"majority-class baseline {report['majority_accuracy_mean']:.3f}",
        f"saved {report['model_path']}",
    ]


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_PATH
    for line in summary_lines(train_classifier(database_path, model_path)):
        print(line)
