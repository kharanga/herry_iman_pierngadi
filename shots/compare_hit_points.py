import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from shots.features import FEATURE_SET_NAMES, feet_landing_columns, has_hitter_and_landing_point, shot_features
from shots.features import load_shot_feature_rows
from shots.train import DATABASE_PATH, FOLD_COUNT, RANDOM_SEED, grouped_cross_validation, jittered_feature_frames, labelled_rows_of

COMPARISON_PATH = Path("data/models/hit_point_comparison.json")
LANDING_OFFSET_QUANTILES = (0.5, 0.9)


def rows_predictable_with_every_feature_set(labelled_rows: pd.DataFrame) -> pd.DataFrame:
    is_predictable = pd.Series(True, index=labelled_rows.index)
    for feature_set in FEATURE_SET_NAMES:
        is_predictable &= has_hitter_and_landing_point(shot_features(labelled_rows, feature_set))
    return labelled_rows[is_predictable].reset_index(drop=True)


def landing_offset_summary(shot_rows: pd.DataFrame) -> dict:
    feet_landing_x, feet_landing_y = feet_landing_columns(shot_rows)
    offsets = np.hypot(feet_landing_x - shot_rows["landing_x"].to_numpy(float), feet_landing_y - shot_rows["landing_y"].to_numpy(float))
    offsets = offsets[~np.isnan(offsets)]
    if len(offsets) == 0:
        return {"shots": 0, "mean_m": None, **{f"p{int(quantile * 100)}_m": None for quantile in LANDING_OFFSET_QUANTILES}}
    return {
        "shots": int(len(offsets)),
        "mean_m": float(np.mean(offsets)),
        **{f"p{int(quantile * 100)}_m": float(np.quantile(offsets, quantile)) for quantile in LANDING_OFFSET_QUANTILES},
    }


def evaluate_feature_set(shot_rows: pd.DataFrame, labelled_rows: pd.DataFrame, feature_set: str, fold_count: int, random_seed: int) -> dict:
    report = grouped_cross_validation(
        shot_features(labelled_rows, feature_set),
        labelled_rows["type_group"],
        labelled_rows["match_id"],
        fold_count,
        random_seed,
        permutation_repeats=0,
        breakdown_rows=labelled_rows,
        scoring_variants=jittered_feature_frames(shot_rows, labelled_rows, feature_set),
    )
    report.pop("permutation_importance")
    report.pop("per_type_en")
    return {"shots": int(len(labelled_rows)), "matches": int(labelled_rows["match_id"].nunique()), **report}


def compare_hit_points(
    database_path: Path = DATABASE_PATH,
    comparison_path: Path = COMPARISON_PATH,
    fold_count: int = FOLD_COUNT,
    random_seed: int = RANDOM_SEED,
) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        shot_rows = load_shot_feature_rows(connection)
    finally:
        connection.close()
    rows_by_feature_set = {feature_set: labelled_rows_of(shot_rows, feature_set) for feature_set in FEATURE_SET_NAMES}
    shared_rows = rows_predictable_with_every_feature_set(rows_by_feature_set["contact"])
    comparison = {
        "landing_offset_m": landing_offset_summary(shared_rows),
        "shared_shots": {
            feature_set: evaluate_feature_set(shot_rows, shared_rows, feature_set, fold_count, random_seed) for feature_set in FEATURE_SET_NAMES
        },
        "all_shots": {
            feature_set: evaluate_feature_set(shot_rows, rows_by_feature_set[feature_set], feature_set, fold_count, random_seed)
            for feature_set in FEATURE_SET_NAMES
        },
    }
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2))
    return comparison


def summary_lines(comparison: dict) -> list[str]:
    lines = [
        f"feet landing vs labelled landing: median {comparison['landing_offset_m']['p50_m']:.2f} m "
        f"on {comparison['landing_offset_m']['shots']} shots"
    ]
    for subset_name, reports in (("shared shots", comparison["shared_shots"]), ("all its shots", comparison["all_shots"])):
        for feature_set, report in reports.items():
            jitter = ", ".join(f"{name} {scores['accuracy_mean']:.1%}" for name, scores in report["variants"].items())
            lines.append(f"{feature_set} on {subset_name} ({report['shots']}): {report['accuracy_mean']:.1%}; {jitter}")
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m shots.compare_hit_points",
        description="Compare the contact and feet hit points, with hit frames jittered, and write the comparison.",
    )
    parser.add_argument("database_path", nargs="?", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("comparison_path", nargs="?", type=Path, default=COMPARISON_PATH, help=f"JSON to write (default {COMPARISON_PATH})")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    for line in summary_lines(compare_hit_points(parsed.database_path, parsed.comparison_path)):
        print(line)
    print(f"saved {parsed.comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
