import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from analytics.shot_types import TYPE_GROUP_BY_TYPE_EN
from shots.features import SHIPPED_FEATURE_SET, shot_features
from shots.train import DATABASE_PATH, FOLD_COUNT, RANDOM_SEED, labelled_shuttleset_rows, new_classifier

COMPARISON_PATH = Path("data/models/fine_type_comparison.json")
FINE_TYPES = tuple(type_en for type_en in TYPE_GROUP_BY_TYPE_EN if type_en != "unknown")
CANDIDATE_SPLITS = {
    "push vs drive": ("push",),
    "rush vs net": ("rush",),
    "return net vs net shot": ("return net",),
    "cross-court net shot": ("cross-court net shot",),
    "wrist smash vs smash": ("wrist smash",),
    "passive drop vs drop": ("passive drop",),
    "defensive return lob vs lob": ("defensive return lob",),
    "short vs long service": ("short service", "long service"),
}
SPLIT_MIN_F1 = 0.5
SPLIT_MIN_PAIR_GAIN = 0.05


def labels_with_splits(type_en: pd.Series, split_off_types: tuple[str, ...]) -> pd.Series:
    groups = type_en.map(TYPE_GROUP_BY_TYPE_EN)
    return type_en.where(type_en.isin(split_off_types), groups)


def out_of_fold_predictions(features: pd.DataFrame, labels: pd.Series, match_ids: pd.Series, fold_count: int = FOLD_COUNT) -> tuple[np.ndarray, list[float]]:
    predictions = np.empty(len(labels), dtype=object)
    fold_accuracies = []
    for train_index, test_index in GroupKFold(n_splits=fold_count).split(features, labels, match_ids):
        if set(match_ids.iloc[train_index]) & set(match_ids.iloc[test_index]):
            raise RuntimeError("a match is in both train and test")
        classifier = new_classifier(RANDOM_SEED).fit(features.iloc[train_index], labels.iloc[train_index])
        predictions[test_index] = classifier.predict(features.iloc[test_index])
        fold_accuracies.append(float(accuracy_score(labels.iloc[test_index], predictions[test_index])))
    return predictions, fold_accuracies


def per_label_scores(labels: pd.Series, predictions: np.ndarray, label_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=label_names, zero_division=0)
    return {
        label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
        for index, label in enumerate(label_names)
    }


def as_groups(labels) -> np.ndarray:
    return np.array([TYPE_GROUP_BY_TYPE_EN.get(label, label) for label in labels], dtype=object)


def pair_scores(type_en: pd.Series, labels: pd.Series, predictions: np.ndarray, split_off_types: tuple[str, ...]) -> dict:
    group = TYPE_GROUP_BY_TYPE_EN[split_off_types[0]]
    pair_labels = sorted(set(labels[type_en.map(TYPE_GROUP_BY_TYPE_EN) == group]))
    in_group = (labels.isin(pair_labels) & pd.Series(predictions, index=labels.index).isin(pair_labels)).to_numpy()
    pair_accuracy = float(accuracy_score(labels[in_group], predictions[in_group]))
    pair_majority = float(labels[in_group].value_counts(normalize=True).max())
    return {
        "pair_labels": pair_labels,
        "pair_shots": int(in_group.sum()),
        "pair_accuracy": pair_accuracy,
        "pair_majority": pair_majority,
        "pair_gain": pair_accuracy - pair_majority,
    }


def split_holds_up(scores: dict, split_off_types: tuple[str, ...]) -> bool:
    pair_labels = scores["pair"]["pair_labels"]
    every_side_reaches_f1 = all(scores["per_label"][label]["f1"] >= SPLIT_MIN_F1 for label in pair_labels)
    return every_side_reaches_f1 and scores["pair"]["pair_gain"] >= SPLIT_MIN_PAIR_GAIN


def label_set_scores(features: pd.DataFrame, labels: pd.Series, match_ids: pd.Series) -> dict:
    predictions, fold_accuracies = out_of_fold_predictions(features, labels, match_ids)
    label_names = sorted(set(labels))
    return {
        "labels": label_names,
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "out_of_fold_accuracy": float(accuracy_score(labels, predictions)),
        "group_accuracy": float(accuracy_score(as_groups(labels), as_groups(predictions))),
        "per_label": per_label_scores(labels, predictions, label_names),
        "confusion": confusion_matrix(labels, predictions, labels=label_names).tolist(),
        "predictions": predictions,
    }


def without_predictions(scores: dict) -> dict:
    return {key: value for key, value in scores.items() if key != "predictions"}


def compare_fine_types(database_path: Path = DATABASE_PATH, comparison_path: Path = COMPARISON_PATH) -> dict:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        labelled_rows = labelled_shuttleset_rows(connection)
    features = shot_features(labelled_rows, SHIPPED_FEATURE_SET)
    type_en, match_ids = labelled_rows["type_en"], labelled_rows["match_id"]
    groups = type_en.map(TYPE_GROUP_BY_TYPE_EN)
    comparison = {
        "shots": len(labelled_rows),
        "matches": int(match_ids.nunique()),
        "feature_set": SHIPPED_FEATURE_SET,
        "groups": without_predictions(label_set_scores(features, groups, match_ids)),
        "fine_types": without_predictions(label_set_scores(features, type_en, match_ids)),
        "splits": {},
    }
    for split_name, split_off_types in CANDIDATE_SPLITS.items():
        labels = labels_with_splits(type_en, split_off_types)
        scores = label_set_scores(features, labels, match_ids)
        scores["pair"] = pair_scores(type_en, labels, scores["predictions"], split_off_types)
        scores["holds_up"] = split_holds_up(scores, split_off_types)
        comparison["splits"][split_name] = without_predictions(scores)
    kept_types = tuple(type_name for split_name, split_off_types in CANDIDATE_SPLITS.items() if comparison["splits"][split_name]["holds_up"] for type_name in split_off_types)
    comparison["middle"] = {"split_off_types": list(kept_types), **without_predictions(label_set_scores(features, labels_with_splits(type_en, kept_types), match_ids))}
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2))
    return comparison


def summary_lines(comparison: dict) -> list[str]:
    lines = [
        f"7 groups: {comparison['groups']['accuracy_mean']:.1%}",
        f"18 types: {comparison['fine_types']['accuracy_mean']:.1%} (as groups {comparison['fine_types']['group_accuracy']:.1%})",
    ]
    for split_name, scores in comparison["splits"].items():
        lines.append(f"{split_name}: pair {scores['pair']['pair_accuracy']:.1%} vs majority {scores['pair']['pair_majority']:.1%}, holds up {scores['holds_up']}")
    middle = comparison["middle"]
    lines.append(f"middle ({len(middle['labels'])} labels): {middle['accuracy_mean']:.1%} (as groups {middle['group_accuracy']:.1%})")
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m shots.fine_types",
        description="Cross-validate the feet classifier on the 18 ShuttleSet types and on single group splits; the shipped model is not touched.",
    )
    parser.add_argument("database_path", nargs="?", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("comparison_path", nargs="?", type=Path, default=COMPARISON_PATH, help=f"JSON to write (default {COMPARISON_PATH})")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    for line in summary_lines(compare_fine_types(parsed.database_path, parsed.comparison_path)):
        print(line)
    print(f"saved {parsed.comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
