import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from vision.flight_types import CLASSIFIED_DETAILS, CLASSIFIER_SEED, Source, shot_rows, write_predictions

CLUB_WEIGHT_SHARE = 1 / 3
AVERAGE_PRO_SHARE = 0.5
CANDIDATES = ("club", "pro_plus_club", "average")


def club_rows(source: Source, labels_file: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_file)[["frame_num", "type_detail"]]
    rows = shot_rows(source).merge(labels, on="frame_num")
    return rows[rows["has_points"] & rows["type_detail"].isin(CLASSIFIED_DETAILS)].reset_index(drop=True)


def shuttleset_rows(database: Path) -> tuple[pd.DataFrame, pd.Series]:
    import duckdb

    from shots.features import FEATURE_NAMES, load_shot_feature_rows, shot_features
    from shots.train import TRAINED_LABEL_COLUMN, labelled_rows_of, rows_with_a_trained_detail

    connection = duckdb.connect(str(database), read_only=True)
    try:
        labelled = rows_with_a_trained_detail(labelled_rows_of(load_shot_feature_rows(connection)))
    finally:
        connection.close()
    return shot_features(labelled, "feet")[list(FEATURE_NAMES)].reset_index(drop=True), labelled[TRAINED_LABEL_COLUMN].reset_index(drop=True)


def shares_as_weights(pro_count: int, club_count: int, club_share: float = CLUB_WEIGHT_SHARE) -> np.ndarray:
    weights = np.concatenate([np.full(pro_count, (1 - club_share) / pro_count), np.full(club_count, club_share / club_count)])
    return weights * len(weights)


def feature_names() -> list[str]:
    from shots.features import FEATURE_NAMES

    return list(FEATURE_NAMES)


def fitted(features: pd.DataFrame, labels: pd.Series, sample_weight: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=CLASSIFIER_SEED).fit(features, labels, sample_weight=sample_weight)


def probabilities_over(classifier, features: pd.DataFrame, details: tuple[str, ...] = CLASSIFIED_DETAILS) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    columns = {label: index for index, label in enumerate(classifier.classes_)}
    return np.column_stack([probabilities[:, columns[detail]] if detail in columns else np.zeros(len(features)) for detail in details])


def averaged_details(pro, club, features: pd.DataFrame, pro_share: float = AVERAGE_PRO_SHARE) -> np.ndarray:
    blended = pro_share * probabilities_over(pro, features) + (1 - pro_share) * probabilities_over(club, features)
    return np.array(CLASSIFIED_DETAILS)[blended.argmax(axis=1)]


def candidate_details(models: dict, features: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "club": models["club"].predict(features),
        "pro_plus_club": models["pro_plus_club"].predict(features),
        "average": averaged_details(models["pro"], models["club"], features),
    }


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.club_classifier", description="Feet-only shot classifiers from club labels, from pro + club shots, and a pro/club probability average.")
    parser.add_argument("--train", nargs=4, action="append", required=True, metavar=("VIDEO", "LABELS", "DATABASE", "END"), help="a labelled club video, its labels CSV and a database holding its own rows built from the labelled hits (vision.flight_types --hits-out, then vision.features); END is unused here; repeatable")
    parser.add_argument("--shuttleset-database", type=Path, default=Path("data/scout.duckdb"), help="database with the labelled ShuttleSet shots (default data/scout.duckdb)")
    parser.add_argument("--pro-model", type=Path, default=Path("data/models/shot_type_classifier.joblib"), help="the shipped pro classifier (default data/models/shot_type_classifier.joblib)")
    parser.add_argument("--predict", nargs=4, action="append", default=[], metavar=("VIDEO", "DATABASE", "END", "OUT_DIR"), help="type DATABASE's own rows with each candidate, written to OUT_DIR/<candidate>.duckdb; repeatable")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    import joblib

    parsed = parse_arguments(arguments)
    try:
        club = pd.concat([club_rows(Source(Path(video), Path(database), int(end)), Path(labels)) for video, labels, database, end in parsed.train], ignore_index=True)
        pro_features, pro_labels = shuttleset_rows(parsed.shuttleset_database)
        pro_model = joblib.load(parsed.pro_model)["classifier"]
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    names = feature_names()
    models = {
        "pro": pro_model,
        "club": fitted(club[names], club["type_detail"]),
        "pro_plus_club": fitted(pd.concat([pro_features, club[names]], ignore_index=True), pd.concat([pro_labels, club["type_detail"]], ignore_index=True), shares_as_weights(len(pro_labels), len(club))),
    }
    print(f"club rows {len(club)} ({club['type_detail'].value_counts().to_dict()}); ShuttleSet rows {len(pro_labels)}; club share of the P+C weight {CLUB_WEIGHT_SHARE:.3f}")
    for video, database, end, out_dir in parsed.predict:
        source = Source(Path(video), Path(database), int(end))
        typed = shot_rows(source)
        typed = typed[typed["has_points"]]
        for name, details in candidate_details(models, typed[names]).items():
            print(f"saved {write_predictions(source, Path(out_dir) / f'{name}.duckdb', typed['row_id'], details)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
