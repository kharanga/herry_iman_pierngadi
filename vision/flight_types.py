import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from analytics.shot_types import group_of_detail
from vision.court import calibration_path, court_to_pixels, load_calibration, video_id
from vision.flight import has_jump, shuttle_track
from vision.hits import HIT_COLUMNS, RALLY_COLUMNS
from vision.movement import PLAYERS, players_files
from vision.players import video_fps_and_width, video_last_frame

FLIGHT_FEATURE_NAMES = ("flight_peak_m", "flight_start_rising", "flight_start_speed_m_per_s", "flight_seen_share")
INITIAL_SECONDS = 0.3
START_REACH_SECONDS = 0.15
MIN_INITIAL_POINTS = 3
LAST_HIT_FLIGHT_SECONDS = 2.0
POSITION_REACH_FRAMES = 5
CLASSIFIER_SEED = 0
CLASSIFIED_DETAILS = ("serve_short", "serve_long", "net", "drive", "clear", "drop", "smash", "lift")
PLAYER_OF_ROWS_SQL = """
SELECT shots.rowid AS row_id, shots.player
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
"""
CLEAR_OWN_PREDICTIONS_SQL = """
UPDATE shots SET type_pred = NULL, type_detail_pred = NULL
WHERE shots.source = 'own' AND shots.match_id = ?
"""
CLEAR_OWN_CONFIDENCE_SQL = """
UPDATE shots SET type_conf = NULL
WHERE shots.source = 'own' AND shots.match_id = ?
"""
SHOTS_COLUMNS_SQL = "SELECT column_name FROM information_schema.columns WHERE table_name = 'shots'"
SET_PREDICTION_SQL = """
UPDATE shots SET type_pred = ?, type_detail_pred = ?
WHERE shots.rowid = ?
"""


@dataclass(frozen=True)
class Source:
    video_path: Path
    database: Path
    end_frame: int

    @property
    def match_id(self) -> str:
        return video_id(self.video_path)


def labels_as_hits(labels: pd.DataFrame) -> pd.DataFrame:
    usable = labels[labels["player"].isin(PLAYERS)]
    return pd.DataFrame({"rally": usable["rally"].astype(int), "frame": usable["frame_num"].astype(int), "hitter": usable["player"], "score": 1.0, "reach": np.nan})[list(HIT_COLUMNS)]


def labelled_rallies_table(rallies: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    counts = hits.groupby("rally").size()
    kept = rallies[rallies["rally"].isin(counts.index)]
    return pd.DataFrame({"rally": kept["rally"].astype(int), "start_frame": kept["start_frame"].astype(int), "end_frame": kept["end_frame"].astype(int), "hits": kept["rally"].map(counts).astype(int)})[list(RALLY_COLUMNS)]


def pixels_per_meter(pixel_to_court: np.ndarray, x_m: float, y_m: float) -> float:
    points = court_to_pixels(pixel_to_court, [(x_m, y_m), (x_m + 1.0, y_m), (x_m, y_m + 1.0)])
    return float(np.mean([np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[0])]))


def feet_at(positions: pd.DataFrame, player: str, frame: int) -> tuple[float, float] | None:
    rows = positions[(positions["player"] == player) & (positions["frame"].between(frame - POSITION_REACH_FRAMES, frame + POSITION_REACH_FRAMES))]
    if rows.empty:
        return None
    nearest = rows.iloc[(rows["frame"] - frame).abs().argmin()]
    return float(nearest["x_m"]), float(nearest["y_m"])


def flight_features(hit_frame: int, next_frame: int, seen: pd.DataFrame, scale: float | None, fps: float, frame_width: int) -> dict[str, float]:
    missing = dict.fromkeys(FLIGHT_FEATURE_NAMES, np.nan)
    flight = seen[(seen["frame"] > hit_frame) & (seen["frame"] < next_frame)]
    initial = flight[flight["frame"] <= hit_frame + INITIAL_SECONDS * fps]
    if scale is None or len(initial) < MIN_INITIAL_POINTS or initial["frame"].iloc[0] - hit_frame > START_REACH_SECONDS * fps or has_jump(flight, frame_width):
        return missing
    first, last = initial.iloc[0], initial.iloc[-1]
    elapsed = (last["frame"] - first["frame"]) / fps
    return {
        "flight_peak_m": float(first["y"] - flight["y"].min()) / scale,
        "flight_start_rising": float(last["y"] < first["y"]),
        "flight_start_speed_m_per_s": float(np.hypot(last["x"] - first["x"], last["y"] - first["y"]) / scale / elapsed) if elapsed > 0 else np.nan,
        "flight_seen_share": len(flight) / max(next_frame - hit_frame - 1, 1),
    }


def shot_rows(source: Source) -> pd.DataFrame:
    import duckdb

    from shots.features import FEATURE_NAMES, has_hitter_and_landing_point, load_shot_feature_rows, shot_features

    connection = duckdb.connect(str(source.database), read_only=True)
    try:
        rows = load_shot_feature_rows(connection)
        players = connection.execute(PLAYER_OF_ROWS_SQL, [source.match_id]).df()
    finally:
        connection.close()
    rows = rows[(rows["source"] == "own") & (rows["match_id"] == source.match_id)].reset_index(drop=True)
    features = shot_features(rows, "feet")[list(FEATURE_NAMES)]
    table = pd.concat([rows[["row_id", "frame_num", "next_frame_num"]], features], axis=1).merge(players, on="row_id")
    return table.assign(has_points=has_hitter_and_landing_point(features).to_numpy())


def with_flight_features(rows: pd.DataFrame, source: Source) -> pd.DataFrame:
    fps, frame_width = video_fps_and_width(source.video_path)
    track = shuttle_track(source.video_path, 0, source.end_frame)
    seen = track[(track["visibility"] == 1) & (track["inpainted"] == 0)].sort_values("frame")
    pixel_to_court = np.array(load_calibration(calibration_path(source.video_path))["pixel_to_court_homography"])
    positions = pd.read_csv(players_files(source.video_path, 0, video_last_frame(source.video_path))[0])
    fallback_frames = int(round(LAST_HIT_FLIGHT_SECONDS * fps))
    flights = []
    for row in rows.itertuples(index=False):
        next_frame = int(row.next_frame_num) if pd.notna(row.next_frame_num) else int(row.frame_num) + fallback_frames
        feet = feet_at(positions, row.player, int(row.frame_num)) if row.player in PLAYERS else None
        scale = pixels_per_meter(pixel_to_court, *feet) if feet else None
        flights.append(flight_features(int(row.frame_num), next_frame, seen, scale, fps, int(frame_width)))
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(flights, columns=list(FLIGHT_FEATURE_NAMES))], axis=1)


def training_rows(source: Source, labels_file: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_file)[["frame_num", "type_detail"]]
    rows = with_flight_features(shot_rows(source), source).merge(labels, on="frame_num")
    return rows[rows["has_points"] & rows["type_detail"].isin(CLASSIFIED_DETAILS)].reset_index(drop=True)


def feature_columns(with_flight: bool) -> list[str]:
    from shots.features import FEATURE_NAMES

    return list(FEATURE_NAMES) + (list(FLIGHT_FEATURE_NAMES) if with_flight else [])


def fitted_classifier(training: pd.DataFrame, with_flight: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=CLASSIFIER_SEED).fit(training[feature_columns(with_flight)], training["type_detail"])


def write_predictions(source: Source, output_database: Path, row_ids, details) -> Path:
    import duckdb

    output_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.database, output_database)
    connection = duckdb.connect(str(output_database))
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(CLEAR_OWN_PREDICTIONS_SQL, [source.match_id])
        if "type_conf" in {row[0] for row in connection.execute(SHOTS_COLUMNS_SQL).fetchall()}:
            connection.execute(CLEAR_OWN_CONFIDENCE_SQL, [source.match_id])
        connection.executemany(SET_PREDICTION_SQL, [[group_of_detail(detail), detail, int(row_id)] for row_id, detail in zip(row_ids, details)])
        connection.execute("COMMIT")
    finally:
        connection.close()
    return output_database


def write_hits_files(labelled_sources: list[tuple[Source, Path]], hits_dir: Path) -> list[Path]:
    written = []
    for source, labels_file in labelled_sources:
        from vision.label_scores import rallies_file_for

        labels = pd.read_csv(labels_file)
        last_frame = video_last_frame(source.video_path)
        output = hits_dir / source.match_id / f"hits_0_{last_frame}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        hits = labels_as_hits(labels[labels["frame_num"] <= source.end_frame])
        hits.to_csv(output, index=False)
        rallies_output = output.with_name(f"rallies_0_{last_frame}.csv")
        labelled_rallies_table(pd.read_csv(rallies_file_for(labels_file)), hits).to_csv(rallies_output, index=False)
        written += [output, rallies_output]
    return written


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.flight_types", description="Shot types from the feet alone against feet + shuttle flight, trained on hand-labelled club hits.")
    parser.add_argument("--train", nargs=4, action="append", required=True, metavar=("VIDEO", "LABELS", "DATABASE", "END"), help="a labelled training video: its labels CSV, a shots database holding its own rows built from the labelled hits (vision.features on the --hits-out files), and the last shuttle-tracked frame; repeatable")
    parser.add_argument("--predict", nargs=4, action="append", default=[], metavar=("VIDEO", "DATABASE", "END", "OUT_DIR"), help="a video whose own rows in DATABASE (from detected hits) get typed by both classifiers, written to OUT_DIR/feet_only.duckdb and OUT_DIR/feet_flight.duckdb; repeatable")
    parser.add_argument("--hits-out", type=Path, metavar="DIR", help="instead, write each training video's labelled hits as a hits file under DIR/<video id>/ for vision.features, and exit")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    training_sources = [(Source(Path(video), Path(database), int(end)), Path(labels)) for video, labels, database, end in parsed.train]
    try:
        if parsed.hits_out:
            for path in write_hits_files(training_sources, parsed.hits_out):
                print(f"saved {path}")
            return 0
        training = pd.concat([training_rows(source, labels) for source, labels in training_sources], ignore_index=True)
        models = {"feet_only": fitted_classifier(training, False), "feet_flight": fitted_classifier(training, True)}
        print(f"training: {len(training)} labelled hits, {int(training['flight_peak_m'].notna().sum())} with flight features; details {training['type_detail'].value_counts().to_dict()}")
        for video, database, end, out_dir in parsed.predict:
            source = Source(Path(video), Path(database), int(end))
            rows = with_flight_features(shot_rows(source), source)
            typed = rows[rows["has_points"]]
            print(f"{source.match_id}: {len(rows)} own rows, {len(typed)} with feet points, {int(typed['flight_peak_m'].notna().sum())} of those with flight features")
            for name, model in models.items():
                details = model.predict(typed[feature_columns(name == "feet_flight")])
                print(f"saved {write_predictions(source, Path(out_dir) / f'{name}.duckdb', typed['row_id'], details)}")
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
