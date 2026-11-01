import argparse
import itertools
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vision.alignment import SHUTTLESET_SET_DIR
from vision.hits import MATCH_TOLERANCE_SECONDS, match_tolerances, tolerance_frames
from vision.label_offset import MIN_FINE_CLARITY, MIN_PEAK_CLARITY, match_labels, read_offset_record, read_stretch_list
from vision.movement import (
    PLAYERS,
    MovementSettings,
    PlayerTracks,
    evaluation_summary,
    fit_hit_model,
    hit_feature_matrix,
    hit_frame_labels,
    hits_from_scores,
    load_hit_model,
    load_player_tracks,
    other_player,
    sample_hit_scores,
    save_hit_model,
    settings_for_video,
)
from vision.players import PLAYERS_DIR, video_fps_and_width
from vision.players import boxes_path as player_boxes_path
from vision.players import positions_path as player_positions_path

RESULTS_DIR = Path("data/hit_training")
DEFAULT_CLUB_WEIGHT = 1 / 3
TUNING_TOLERANCE_SECONDS = MATCH_TOLERANCE_SECONDS[0]
HIT_SCORE_GRID = (0.35, 0.45, 0.5, 0.55, 0.65)
RALLY_END_GRID = (0.5, 0.6, 0.7, 0.8)
LEARNING_CURVE_SIZES = (1, 3, 5)
LEARNING_CURVE_SEED = 2026
COUNT_COLUMNS = ("labelled", "detected", "matched", "hitter_correct", "hitter_judged")


@dataclass(frozen=True)
class MatchData:
    name: str
    video_path: Path
    fps: float
    settings: MovementSettings
    tracks: tuple[PlayerTracks, ...]
    labels: pd.DataFrame


def read_match_list(list_file: Path) -> list[tuple[str, Path]]:
    pairs = [line.split() for line in Path(list_file).read_text().splitlines() if line.strip() and not line.startswith("#")]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"{list_file}: every line needs a ShuttleSet match folder and a video path")
    return [(folder, Path(video)) for folder, video in pairs]


def exclusion_reason(record: dict) -> str | None:
    if not record.get("clarity", 0.0) >= MIN_PEAK_CLARITY:
        return f"coarse offset unclear (clarity {record.get('clarity', float('nan')):.3f})"
    if "fine_offset_frames" not in record:
        return "no fine offset yet (run python -m vision.label_offset <video> --fine)"
    if not record["fine_clarity"] >= MIN_FINE_CLARITY:
        return f"fine offset unclear (clarity {record['fine_clarity']:.3f})"
    return None


def club_labels(labels_file: Path) -> pd.DataFrame:
    table = pd.read_csv(labels_file)
    return pd.DataFrame({"rally": table["rally"].astype(int), "frame": table["frame_num"].astype(int), "hitter": table["player"], "type": table["type_group"]})


def without_doubtful(positions: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "doubtful" not in positions.columns:
        return positions, boxes
    doubtful = positions.loc[positions["doubtful"].astype(bool), ["frame", "player"]]
    keys = set(zip(doubtful["frame"], doubtful["player"]))
    keep = lambda table: table[[key not in keys for key in zip(table["frame"], table["player"])]]
    return keep(positions), keep(boxes)


def load_club(video_path: Path, labels_file: Path, players_dir: Path = PLAYERS_DIR) -> MatchData:
    fps, frame_width = video_fps_and_width(video_path)
    from vision.players import video_last_frame
    from vision.movement import player_tracks

    last_frame = video_last_frame(video_path)
    positions_file = player_positions_path(video_path, 0, last_frame, players_dir)
    boxes_file = player_boxes_path(video_path, 0, last_frame, players_dir)
    for path in (positions_file, boxes_file):
        if not path.is_file():
            raise FileNotFoundError(f"track file not found: {path}")
    positions, boxes = without_doubtful(pd.read_csv(positions_file), pd.read_csv(boxes_file))
    tracks = player_tracks(positions, boxes, 0, last_frame)
    return MatchData(f"club:{Path(video_path).stem}", Path(video_path), fps, settings_for_video(fps, frame_width), (tracks,), club_labels(labels_file))


def load_match(folder: str, video_path: Path) -> MatchData:
    record = read_offset_record(video_path)
    fps, frame_width = video_fps_and_width(video_path)
    tracks = tuple(load_player_tracks(video_path, start_frame, end_frame) for start_frame, end_frame in read_stretch_list(video_path))
    labels = match_labels(video_path, record, int(record["fine_offset_frames"]))
    return MatchData(folder, Path(video_path), fps, settings_for_video(fps, frame_width), tracks, labels)


def training_rows(match: MatchData) -> tuple[np.ndarray, np.ndarray]:
    tolerance = match.settings.label_tolerance_frames
    feature_rows, target_rows = [], []
    for tracks in match.tracks:
        for player in PLAYERS:
            detected = np.isfinite(tracks.points_of(player)[:, 0]) | np.isfinite(tracks.points_of(other_player(player))[:, 0])
            hit_frames = match.labels.loc[match.labels["hitter"] == player, "frame"]
            feature_rows.append(hit_feature_matrix(tracks, player, match.settings)[detected])
            target_rows.append(hit_frame_labels(tracks.frames, hit_frames, tolerance)[detected])
    return np.vstack(feature_rows), np.concatenate(target_rows)


def match_weights(names: list[str], rows_by_match: dict, club_names: list[str], club_weight: float) -> np.ndarray:
    pro = [name for name in names if name not in club_names]
    clubs = [name for name in names if name in club_names]
    shares = {name: (1 - club_weight) / len(pro) if clubs else 1 / len(pro) for name in pro}
    shares.update({name: club_weight / len(clubs) for name in clubs})
    weights = np.concatenate([np.full(len(rows_by_match[name][1]), shares[name] / len(rows_by_match[name][1])) for name in names])
    return weights * len(weights)


def fit_on(names: list[str], rows_by_match: dict, club_names: list[str] = (), club_weight: float = 0.0) -> object:
    features = np.vstack([rows_by_match[name][0] for name in names])
    targets = np.concatenate([rows_by_match[name][1] for name in names])
    if not any(name in club_names for name in names):
        return fit_hit_model(features, targets)
    return fit_hit_model(features, targets, match_weights(names, rows_by_match, list(club_names), club_weight))


def match_scores(model, match: MatchData) -> list[dict[str, np.ndarray]]:
    return [sample_hit_scores(tracks, model, match.settings) for tracks in match.tracks]


def detected_hits(match: MatchData, scores: list[dict[str, np.ndarray]], settings: MovementSettings) -> pd.DataFrame:
    settings = replace(settings, infer_serves=False)
    tables = [hits_from_scores(tracks, stretch_scores, settings)[0] for tracks, stretch_scores in zip(match.tracks, scores)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["frame", "hitter", "score"])


def threshold_settings(match: MatchData, thresholds: tuple[float, float]) -> MovementSettings:
    return replace(match.settings, hit_score_min=thresholds[0], rally_end_score_min=thresholds[1])


def f1_of(totals: dict) -> float:
    precision = totals["matched"] / max(totals["detected"], 1)
    recall = totals["matched"] / max(totals["labelled"], 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def best_thresholds(matches: dict[str, MatchData], scores_by_match: dict[str, list]) -> tuple[float, float]:
    best, best_f1 = (MovementSettings().hit_score_min, MovementSettings().rally_end_score_min), -1.0
    for thresholds in itertools.product(HIT_SCORE_GRID, RALLY_END_GRID):
        totals = {key: 0 for key in COUNT_COLUMNS}
        for name, scores in scores_by_match.items():
            summary = evaluation_summary(detected_hits(matches[name], scores, threshold_settings(matches[name], thresholds)), matches[name].labels, tolerance_frames(TUNING_TOLERANCE_SECONDS, matches[name].fps))
            totals = {key: totals[key] + summary[key] for key in COUNT_COLUMNS}
        if f1_of(totals) > best_f1:
            best, best_f1 = thresholds, f1_of(totals)
    return best


def players_of_matches(names: list[str], set_dir: Path = SHUTTLESET_SET_DIR) -> dict[str, set[str]]:
    table = pd.read_csv(set_dir / "match.csv")
    players = {}
    for name in names:
        rows = table[table["video"] == name]
        players[name] = set(rows["winner"]) | set(rows["loser"])
    return players


def training_matches(held_out: str, names: list[str], players: dict[str, set[str]] | None = None) -> list[str]:
    if players is None:
        return [name for name in names if name != held_out]
    return [name for name in names if name != held_out and not players[name] & players[held_out]]


def count_rows(match: MatchData, hits: pd.DataFrame, **columns) -> list[dict]:
    return [
        {"match": match.name, "fps": match.fps, **columns, "tolerance_s": window, "tolerance_frames": frames, **evaluation_summary(hits, match.labels, frames)}
        for window, frames in match_tolerances(match.fps)
    ]


def held_out_rows(
    matches: dict[str, MatchData],
    rows_by_match: dict,
    tune: bool = False,
    players: dict[str, set[str]] | None = None,
    progress=print,
    club_names: list[str] = (),
    club_weight: float = 0.0,
    clubs: dict[str, MatchData] | None = None,
) -> list[dict]:
    names = sorted(matches)
    sources = {**matches, **(clubs or {})}
    rows = []
    for held_out in names:
        training = training_matches(held_out, names, players) + list(club_names)
        match = matches[held_out]
        scores = match_scores(fit_on(training, rows_by_match, club_names, club_weight), match)
        rows += count_rows(match, detected_hits(match, scores, match.settings), thresholds="default", training_matches=len(training))
        if tune:
            inner_scores = {inner: match_scores(fit_on([name for name in training if name != inner], rows_by_match, club_names, club_weight), sources[inner]) for inner in training}
            thresholds = best_thresholds(sources, inner_scores)
            rows += count_rows(match, detected_hits(match, scores, threshold_settings(match, thresholds)), thresholds="tuned", hit_score_min=thresholds[0], rally_end_score_min=thresholds[1], training_matches=len(training))
        progress(f"held out {held_out}: trained on {len(training)} matches")
    return rows


def learning_curve_rows(matches: dict[str, MatchData], rows_by_match: dict, sizes=LEARNING_CURVE_SIZES, seed: int = LEARNING_CURVE_SEED, progress=print) -> list[dict]:
    names = sorted(matches)
    generator = random.Random(seed)
    rows = []
    for held_out in names:
        others = training_matches(held_out, names)
        for size in [size for size in sizes if size < len(others)] + [len(others)]:
            subset = generator.sample(others, size) if size < len(others) else others
            match = matches[held_out]
            rows += count_rows(match, detected_hits(match, match_scores(fit_on(subset, rows_by_match), match), match.settings), training_matches=size)
        progress(f"learning curve for {held_out} done")
    return rows


def baseline_rows(matches: dict[str, MatchData], model) -> list[dict]:
    return [row for name in sorted(matches) for row in count_rows(matches[name], detected_hits(matches[name], match_scores(model, matches[name]), matches[name].settings))]


def totals_line(name: str, rows: pd.DataFrame) -> str:
    totals = rows[list(COUNT_COLUMNS)].sum()
    precision = totals["matched"] / max(totals["detected"], 1)
    recall = totals["matched"] / max(totals["labelled"], 1)
    hitter = totals["hitter_correct"] / max(totals["hitter_judged"], 1)
    return f"{name}: labelled {totals['labelled']}, detected {totals['detected']}, matched {totals['matched']}, precision {precision:.1%}, recall {recall:.1%}, hitter {totals['hitter_correct']}/{totals['hitter_judged']} = {hitter:.1%}"


def window_text(seconds: float, rows: pd.DataFrame) -> str:
    frames = ", ".join(f"±{int(count)} frames at {fps:g} fps" for fps, count in sorted(rows.groupby("fps")["tolerance_frames"].first().items()))
    return f"±{seconds:g} s ({frames})"


def summary_lines(title: str, rows: pd.DataFrame, group_columns: tuple[str, ...] = ()) -> list[str]:
    lines = []
    for keys, group in rows.groupby([*group_columns, "tolerance_s"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = ", ".join([f"{column} {value}" for column, value in zip(group_columns, keys[:-1])] + [window_text(keys[-1], group)])
        lines.append(totals_line(f"{title}, {label}", group))
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.hit_training", description="Train and measure the movement hit scorer on whole ShuttleSet matches, each match held out in turn.")
    parser.add_argument("match_list", type=Path, help="text file, one line per match: ShuttleSet match folder and video path")
    parser.add_argument("--held-out", action="store_true", help="score each match with a model trained on the others")
    parser.add_argument("--tune", action="store_true", help="with --held-out: also tune the two thresholds on the training matches (each scored by a model without it); with --save: print the thresholds tuned that way on all the saved model's training sources")
    parser.add_argument("--by-player", action="store_true", help="with --held-out: also leave out every training match that shares a player with the held-out one")
    parser.add_argument("--learning-curve", action="store_true", help="score each match with models trained on 1, 3, 5 and all other matches")
    parser.add_argument("--baseline", type=Path, metavar="MODEL", help="score every match with this model (default thresholds)")
    parser.add_argument("--save", type=Path, metavar="MODEL", help="train on every listed match (minus --exclude) and save the model here")
    parser.add_argument("--exclude", action="append", default=[], metavar="FOLDER", help="with --save: leave this match out of the training; repeatable")
    parser.add_argument("--club", nargs=2, action="append", default=[], metavar=("VIDEO", "LABELS"), help="a fully hand-labelled club video as training data (never scored): its labels CSV and its whole-video tracks, doubtful rows dropped; repeatable")
    parser.add_argument("--club-weight", type=float, default=DEFAULT_CLUB_WEIGHT, help="share of the total sample weight carried by the club videos together (default 1/3); the ShuttleSet matches share the rest equally")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="folder for the result tables (default data/hit_training)")
    return parser.parse_args(arguments)


def loaded_matches(match_list: Path) -> dict[str, MatchData]:
    matches = {}
    for folder, video_path in read_match_list(match_list):
        reason = exclusion_reason(read_offset_record(video_path))
        if reason:
            print(f"{folder}: left out, {reason}")
            continue
        matches[folder] = load_match(folder, video_path)
    return matches


def write_table(rows: list[dict], output_path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"saved {output_path}")
    return table


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        matches = loaded_matches(parsed.match_list)
        if not matches:
            raise ValueError("no usable match in the list")
        baseline_model = load_hit_model(parsed.baseline) if parsed.baseline else None
        if not 0 <= parsed.club_weight < 1:
            raise ValueError(f"--club-weight must be in [0, 1), got {parsed.club_weight}")
        clubs = {club.name: club for club in (load_club(Path(video), Path(labels)) for video, labels in parsed.club)}
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    rows_by_match = {name: training_rows(match) for name, match in {**matches, **clubs}.items()}
    club_names = sorted(clubs)
    for name in club_names:
        print(f"{name}: {clubs[name].labels.shape[0]} labelled hits, {len(rows_by_match[name][1])} player-frames, {int(rows_by_match[name][1].sum())} hit frames, weight share {parsed.club_weight / len(club_names):.3f}")
    print(f"{len(matches)} matches: {sum(match.labels.shape[0] for match in matches.values())} labelled hits, {sum(len(rows[1]) for rows in rows_by_match.values())} player-frames, {int(sum(rows[1].sum() for rows in rows_by_match.values()))} hit frames")
    if parsed.held_out:
        table = write_table(held_out_rows(matches, rows_by_match, parsed.tune, club_names=club_names, club_weight=parsed.club_weight, clubs=clubs), parsed.out / "held_out.csv")
        for line in summary_lines("held out", table, ("thresholds",)) + summary_lines("held out", table[table["thresholds"] == "default"], ("fps",)):
            print(line)
        if parsed.by_player:
            players = players_of_matches(sorted(matches))
            table = write_table(held_out_rows(matches, rows_by_match, False, players), parsed.out / "held_out_by_player.csv")
            print("\n".join(summary_lines("held out with every match sharing a player", table)))
    if parsed.learning_curve:
        table = write_table(learning_curve_rows(matches, rows_by_match), parsed.out / "learning_curve.csv")
        print("\n".join(summary_lines("learning curve", table, ("training_matches",))))
    if baseline_model is not None:
        table = write_table(baseline_rows(matches, baseline_model), parsed.out / "baseline.csv")
        print("\n".join(summary_lines(f"baseline {parsed.baseline}", table)))
    if parsed.save:
        names = [name for name in sorted(matches) if name not in parsed.exclude] + club_names
        print(f"saved {save_hit_model(fit_on(names, rows_by_match, club_names, parsed.club_weight), parsed.save, names)} (trained on {len(names)} sources{f', club weight {parsed.club_weight:.3f}' if club_names else ''})")
        if parsed.tune:
            sources = {**matches, **clubs}
            inner_scores = {inner: match_scores(fit_on([name for name in names if name != inner], rows_by_match, club_names, parsed.club_weight), sources[inner]) for inner in names}
            hit_score_min, rally_end_score_min = best_thresholds(sources, inner_scores)
            print(f"thresholds for the saved model, tuned on its {len(names)} training sources (each scored by a model trained without it): --hit-score-min {hit_score_min:g} --rally-end-score-min {rally_end_score_min:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
