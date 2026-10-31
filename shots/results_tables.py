import json
import sys
from pathlib import Path

from shots.compare_features import COMPARISON_PATH
from shots.compare_hit_points import COMPARISON_PATH as HIT_POINT_COMPARISON_PATH
from shots.features import FEATURE_SET_NAMES
from shots.jitter import JITTER_OFFSETS_FRAMES, jitter_label
from shots.predict import DATABASE_PATH, prediction_summary
from shots.train import MODEL_PATH, report_path_for

WEAK_TYPE_RECALL_BELOW = 0.85
HIT_POINT_DESCRIPTIONS = {
    "contact": "contact (labelled hit point and landing point)",
    "feet": "feet (hitter's feet; landing = the reply hitter's feet)",
}

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


def accuracy_cell(accuracy: float | None) -> str:
    return "none predicted" if accuracy is None else f"{accuracy:.1%}"


def subsets_table(report: dict) -> str:
    return "\n".join(
        ["| Shots | Count | Held-out accuracy |", "|---|---|---|"]
        + [f"| {name} | {thousands(subset['shots'])} | {accuracy_cell(subset['accuracy'])} |" for name, subset in report["subsets"].items()]
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
            f"| ShuttleSet shots in `shots` | {thousands(summary['shots'])} |",
            f"| With a prediction | {thousands(summary['predicted_shots'])} |",
            f"| Left NULL (no hitter point or no landing point) | {thousands(summary['shots'] - summary['predicted_shots'])} |",
            f"| `type_conf` mean / lowest | {summary['mean_conf']:.2f} / {summary['lowest_conf']:.2f} |",
            f"| In-sample agreement with `type_group` (not an accuracy) | {summary['in_sample_agreement']:.1%} |",
            f"| Shots typed `unknown`, predicted as | {unknown_predictions} |",
        ]
        + [
            f"| Own rows of `{match_id}` with a prediction | {thousands(counts['predicted_shots'])} of {thousands(counts['shots'])} ({counts['rallies']} rallies) |"
            for match_id, counts in summary["own_matches"].items()
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



def percent_point_gap(later: float, earlier: float) -> str:
    return f"{(later - earlier) * 100:+.1f}"


def hit_point_accuracy_table(comparison: dict) -> str:
    lines = ["| Feature set | Shared shots | Held-out accuracy | All its shots | Held-out accuracy |", "|---|---|---|---|---|"]
    for feature_set in FEATURE_SET_NAMES:
        shared = comparison["shared_shots"][feature_set]
        every = comparison["all_shots"][feature_set]
        lines.append(
            f"| {HIT_POINT_DESCRIPTIONS[feature_set]} | {thousands(shared['shots'])} | **{shared['accuracy_mean']:.1%}** (± {shared['accuracy_std']:.1%}) "
            f"| {thousands(every['shots'])} | **{every['accuracy_mean']:.1%}** (± {every['accuracy_std']:.1%}) |"
        )
    return "\n".join(lines)


def hit_point_per_group_table(comparison: dict) -> str:
    contact = comparison["shared_shots"]["contact"]["per_group"]
    feet = comparison["shared_shots"]["feet"]["per_group"]
    rows = sorted(contact, key=lambda group: -contact[group]["f1"])
    return "\n".join(
        ["| Group | Shots | contact P / R / F1 | feet P / R / F1 | F1 change |", "|---|---|---|---|---|"]
        + [
            f"| {group} | {thousands(contact[group]['support'])} "
            f"| {contact[group]['precision']:.3f} / {contact[group]['recall']:.3f} / {contact[group]['f1']:.3f} "
            f"| {feet[group]['precision']:.3f} / {feet[group]['recall']:.3f} / {feet[group]['f1']:.3f} "
            f"| {feet[group]['f1'] - contact[group]['f1']:+.3f} |"
            for group in rows
        ]
    )


def hit_point_confusion_table(comparison: dict, feature_set: str) -> str:
    return confusion_matrix_table(comparison["shared_shots"][feature_set])


def jitter_table(comparison: dict) -> str:
    columns = [(f"{feature_set}, shared shots", comparison["shared_shots"][feature_set]) for feature_set in FEATURE_SET_NAMES]
    columns.append(("feet, all its shots (the shipped model)", comparison["all_shots"]["feet"]))
    lines = ["| Hit frames | " + " | ".join(name for name, _ in columns) + " |", "|---" * (len(columns) + 1) + "|"]
    lines.append("| exact | " + " | ".join(f"**{report['accuracy_mean']:.1%}**" for _, report in columns) + " |")
    for max_offset in JITTER_OFFSETS_FRAMES:
        cells = []
        for _, report in columns:
            jittered = report["variants"][jitter_label(max_offset)]
            cells.append(f"{jittered['accuracy_mean']:.1%} ({percent_point_gap(jittered['accuracy_mean'], report['accuracy_mean'])})")
        lines.append(f"| +/- {max_offset} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metres_cell(metres: float | None) -> str:
    return "no shots" if metres is None else f"{metres:.2f} m"


def landing_offset_table(comparison: dict) -> str:
    offsets = comparison["landing_offset_m"]
    return "\n".join(
        [
            "| Reply hitter's feet vs labelled landing point | |",
            "|---|---|",
            f"| Shots | {thousands(offsets['shots'])} |",
            f"| Median distance | {metres_cell(offsets['p50_m'])} |",
            f"| 90th percentile | {metres_cell(offsets['p90_m'])} |",
            f"| Mean | {metres_cell(offsets['mean_m'])} |",
        ]
    )


def hit_point_results_tables(comparison: dict) -> str:
    sections = [
        ("Contact vs feet", hit_point_accuracy_table(comparison)),
        ("Per group, contact vs feet (shared shots)", hit_point_per_group_table(comparison)),
        ("Confusion matrix, contact (shared shots)", hit_point_confusion_table(comparison, "contact")),
        ("Confusion matrix, feet (shared shots)", hit_point_confusion_table(comparison, "feet")),
        ("Timing jitter", jitter_table(comparison)),
        ("Landing offset", landing_offset_table(comparison)),
    ]
    return "\n\n".join(f"### {title}\n\n{table}" for title, table in sections)


if __name__ == "__main__":
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_PATH
    comparison_path = Path(sys.argv[2]) if len(sys.argv) > 2 else COMPARISON_PATH
    report = json.loads(report_path_for(model_path).read_text())
    database_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DATABASE_PATH
    hit_point_comparison_path = Path(sys.argv[4]) if len(sys.argv) > 4 else HIT_POINT_COMPARISON_PATH
    comparison = json.loads(comparison_path.read_text())
    print(results_tables(report, comparison, prediction_summary(database_path)))
    if hit_point_comparison_path.is_file():
        print()
        print(hit_point_results_tables(json.loads(hit_point_comparison_path.read_text())))
