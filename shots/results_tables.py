import json
import sys
from pathlib import Path

from shots.compare_features import COMPARISON_PATH
from shots.predict import DATABASE_PATH, prediction_summary
from shots.train import MODEL_PATH, report_path_for

WEAK_TYPE_RECALL_BELOW = 0.85

FEATURE_SET_DESCRIPTIONS = {
    "geometry only": "geometry only (positions, shot vector, `is_cross`, `ball_round`)",
    "+ timing": "+ timing (time to next hit, since previous hit)",
    "shipped model": "+ previous shot's length and depth (**the shipped model**)",
    "shipped model + previous shot's true group": "+ previous shot's **true** group",
}


def thousands(count: int) -> str:
    return f"{count:,}"


def accuracy_table(report: dict) -> str:
    fold_percentages = ", ".join(f"{fold['accuracy'] * 100:.1f}" for fold in report["folds"])
    return "\n".join(
        [
            "| | Held-out accuracy |",
            "|---|---|",
            f"| **Classifier** | **{report['accuracy_mean']:.1%}** (± {report['accuracy_std']:.1%} across folds; folds {fold_percentages}) |",
            f"| Majority class (`{report['folds'][0]['majority_group']}`) | {report['majority_accuracy_mean']:.1%} |",
        ]
    )


def subsets_table(report: dict) -> str:
    return "\n".join(
        ["| Shots | Count | Held-out accuracy |", "|---|---|---|"]
        + [f"| {name} | {thousands(subset['shots'])} | {subset['accuracy']:.1%} |" for name, subset in report["subsets"].items()]
    )


def weak_types_table(report: dict, recall_below: float = WEAK_TYPE_RECALL_BELOW) -> str:
    rows = [(type_en, scores) for type_en, scores in report["per_type_en"].items() if scores["recall"] < recall_below]
    return "\n".join(
        ["| Type (group) | Shots | Recall | Most common mistakes |", "|---|---|---|---|"]
        + [
            f"| {type_en} ({scores['group']}) | {thousands(scores['shots'])} | {scores['recall']:.0%} | "
            + ", ".join(f"{group} {thousands(count)}" for group, count in scores["top_mistakes"].items())
            + " |"
            for type_en, scores in rows
        ]
    )


def predictions_table(summary: dict) -> str:
    unknown_predictions = ", ".join(
        f"{'NULL' if group is None else group} {thousands(count)}" for group, count in summary["unknown_shot_predictions"].items()
    )
    return "\n".join(
        [
            "| | |",
            "|---|---|",
            f"| Shots in `shots` | {thousands(summary['shots'])} |",
            f"| With a prediction | {thousands(summary['predicted_shots'])} |",
            f"| Left NULL (no hitter point or no landing point) | {thousands(summary['shots'] - summary['predicted_shots'])} |",
            f"| `type_conf` mean / lowest | {summary['mean_conf']:.2f} / {summary['lowest_conf']:.2f} |",
            f"| In-sample agreement with `type_group` (not an accuracy) | {summary['in_sample_agreement']:.1%} |",
            f"| Shots typed `unknown`, predicted as | {unknown_predictions} |",
        ]
    )


def per_group_table(report: dict) -> str:
    rows = sorted(report["per_group"].items(), key=lambda group_and_scores: -group_and_scores[1]["f1"])
    return "\n".join(
        ["| Group | Shots | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        + [
            f"| {group} | {thousands(scores['support'])} | {scores['precision']:.3f} | {scores['recall']:.3f} | {scores['f1']:.3f} |"
            for group, scores in rows
        ]
    )


def confusion_matrix_table(report: dict) -> str:
    groups = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["rows_true_columns_predicted"]
    lines = ["| true \\ predicted | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for row_index, (group, counts) in enumerate(zip(groups, matrix)):
        cells = [f"**{thousands(count)}**" if column_index == row_index else thousands(count) for column_index, count in enumerate(counts)]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def importance_table(report: dict) -> str:
    return "\n".join(
        ["| Feature | Accuracy drop |", "|---|---|"]
        + [f"| {feature} | {accuracy_drop:.3f} |" for feature, accuracy_drop in report["permutation_importance"].items()]
    )


def feature_set_table(comparison: dict) -> str:
    return "\n".join(
        ["| Features | Held-out accuracy |", "|---|---|"]
        + [f"| {FEATURE_SET_DESCRIPTIONS[set_name]} | {result['accuracy_mean']:.1%} |" for set_name, result in comparison.items()]
    )


def results_tables(report: dict, comparison: dict, summary: dict) -> str:
    sections = [
        ("Accuracy", accuracy_table(report)),
        ("Accuracy by shot position", subsets_table(report)),
        ("Per group", per_group_table(report)),
        ("Confusion matrix", confusion_matrix_table(report)),
        ("Feature importance", importance_table(report)),
        ("Weak shot types", weak_types_table(report)),
        ("Feature sets", feature_set_table(comparison)),
        ("Predictions", predictions_table(summary)),
    ]
    return "\n\n".join(f"### {title}\n\n{table}" for title, table in sections)


if __name__ == "__main__":
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_PATH
    comparison_path = Path(sys.argv[2]) if len(sys.argv) > 2 else COMPARISON_PATH
    report = json.loads(report_path_for(model_path).read_text())
    database_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DATABASE_PATH
    comparison = json.loads(comparison_path.read_text())
    print(results_tables(report, comparison, prediction_summary(database_path)))
