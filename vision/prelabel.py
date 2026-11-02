import argparse
import sys
from pathlib import Path

import pandas as pd

from vision.label import LABEL_COLUMNS, RALLY_COLUMNS, REVIEW_COLUMNS, SOURCE_MODEL, label_paths, write_atomically
from vision.movement import PLAYERS

UNKNOWN_TYPE = "unknown"
OWN_PREDICTIONS_SQL = """
SELECT shots.frame_num, shots.type_pred, shots.type_detail_pred, shots.type_conf
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
"""


def read_typed_rows(database: Path, match_id: str) -> pd.DataFrame:
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(OWN_PREDICTIONS_SQL, [match_id]).df()
    finally:
        connection.close()


def model_labels(hits: pd.DataFrame, typed: pd.DataFrame) -> pd.DataFrame:
    hits = hits[hits["hitter"].isin(PLAYERS)].drop_duplicates("frame").sort_values("frame")
    typed = typed.dropna(subset=["type_pred"]).drop_duplicates("frame_num").set_index("frame_num")
    rows = []
    for hit in hits.itertuples(index=False):
        frame = int(hit.frame)
        if frame in typed.index:
            row = typed.loc[frame]
            group, detail, score = str(row["type_pred"]), str(row["type_detail_pred"]), (None if pd.isna(row["type_conf"]) else float(row["type_conf"]))
        else:
            group, detail, score = UNKNOWN_TYPE, UNKNOWN_TYPE, None
        rows.append([int(hit.rally), frame, hit.hitter, group, detail, SOURCE_MODEL, score, frame, detail])
    table = pd.DataFrame(rows, columns=LABEL_COLUMNS + REVIEW_COLUMNS)
    table["model_frame"] = table["model_frame"].astype("Int64")
    return table


def model_rallies(rallies: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    kept = rallies[rallies["rally"].isin(labels["rally"])]
    return pd.DataFrame({"rally": kept["rally"].astype(int), "start_frame": kept["start_frame"].astype(int), "end_frame": kept["end_frame"].astype(int), "bounds_checked": False})[RALLY_COLUMNS]


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.prelabel", description="Write the model's hits and shot types as a labels file for review in vision.label_ui --review.")
    parser.add_argument("video", type=Path, help="the video the labels belong to (names the labels file)")
    parser.add_argument("--hits", type=Path, required=True, help="the detector's hits CSV (its rallies_*.csv sits next to it)")
    parser.add_argument("--database", type=Path, required=True, help="a shots database with the video's own rows typed (type_pred, type_detail_pred, type_conf)")
    parser.add_argument("--match-id", required=True, help="the own rows' match id")
    parser.add_argument("--labels", type=Path, help="labels CSV to write (default data/labels/<video id>.csv)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing labels file")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    labels_file, rallies_file = label_paths(parsed.video)
    if parsed.labels is not None:
        labels_file, rallies_file = parsed.labels, parsed.labels.with_name(f"{parsed.labels.stem}_rallies.csv")
    if labels_file.exists() and not parsed.force:
        print(f"error: {labels_file} exists; it may hold reviewed labels (use --force to overwrite)", file=sys.stderr)
        return 1
    try:
        hits = pd.read_csv(parsed.hits)
        rallies = pd.read_csv(parsed.hits.with_name(parsed.hits.name.replace("hits_", "rallies_", 1)))
        labels = model_labels(hits, read_typed_rows(parsed.database, parsed.match_id))
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_atomically(labels, labels_file)
    write_atomically(model_rallies(rallies, labels), rallies_file)
    typed = int((labels["type_group"] != UNKNOWN_TYPE).sum())
    print(f"saved {labels_file} ({len(labels)} model hits, {typed} typed, {len(labels) - typed} unknown) and {rallies_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
