import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from vision.court import NET_Y_M
from vision.hits import MATCH_TOLERANCE_SECONDS, match_hits, match_tolerances, tolerance_frames
from vision.movement import (
    HIT_MODEL_PARAMETERS,
    PLAYERS,
    HitCandidate,
    MovementSettings,
    PlayerTracks,
    alternating_hit_chain,
    evaluation_summary,
    hit_candidates,
    local_maxima,
    motion_features,
    moving_average,
    other_player,
    tables_in_video_frames,
    tracks_on_sample_indices,
)

IN_PLAY_MODEL_PATH = Path("data/models/in_play.joblib")
RESULTS_DIR = Path("data/in_play")
LANDING_SECONDS = 1.0
WINDOW_SECONDS = (1.0, 3.0, 8.0)
PEAK_DISTANCE_CAP_SECONDS = 10.0
PROBABILITY_SMOOTHING_SECONDS = 1.0
IN_PLAY_PROBABILITY_MIN = 0.5
MIN_IN_PLAY_SECONDS = 2.0
MIN_OUT_OF_PLAY_SECONDS = 1.0
SERVE_WINDOW_SECONDS = 0.3
SPAN_PADDING_SECONDS = 0.2
TRAINING_SAMPLE_STEP = 2
IN_PLAY_MODEL_PARAMETERS = {**HIT_MODEL_PARAMETERS, "class_weight": None}
PLAYER_SIGNAL_NAMES = ("speed", "abs_acceleration", "detected", "net_distance")
PLAYER_WINDOW_STATS = (("speed", "mean"), ("speed", "max"), ("abs_acceleration", "mean"), ("detected", "mean"), ("net_distance", "mean"))
IN_PLAY_FEATURE_NAMES = (
    tuple(f"{player}_net_distance" for player in PLAYERS)
    + tuple(f"{player}_{signal}_{stat}_{seconds:g}s" for player in PLAYERS for seconds in WINDOW_SECONDS for signal, stat in PLAYER_WINDOW_STATS)
    + tuple(f"{player}_score_max_{seconds:g}s" for player in PLAYERS for seconds in WINDOW_SECONDS)
    + tuple(f"score_peaks_{seconds:g}s" for seconds in WINDOW_SECONDS)
    + ("seconds_since_peak", "seconds_to_peak")
)


def frames_for_seconds(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def player_signals(tracks: PlayerTracks, player: str, settings: MovementSettings) -> dict[str, np.ndarray]:
    motion = motion_features(tracks.points_of(player), tracks.boxes_of(player), settings)
    return {
        "speed": motion["speed"],
        "abs_acceleration": np.abs(motion["acceleration"]),
        "detected": motion["detected"],
        "net_distance": np.abs(motion["y_m"] - NET_Y_M),
    }


def centred_window(values: np.ndarray, frames: int, stat: str) -> np.ndarray:
    rolling = pd.Series(np.asarray(values, dtype=float)).rolling(frames, center=True, min_periods=1)
    return getattr(rolling, stat)().to_numpy()


def seconds_since(mask: np.ndarray, fps: float, cap_seconds: float = PEAK_DISTANCE_CAP_SECONDS) -> np.ndarray:
    indices = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, indices, -1)) if len(mask) else np.zeros(0, dtype=int)
    return np.where(last >= 0, np.minimum((indices - last) / fps, cap_seconds), cap_seconds)


def seconds_until(mask: np.ndarray, fps: float, cap_seconds: float = PEAK_DISTANCE_CAP_SECONDS) -> np.ndarray:
    return seconds_since(np.asarray(mask)[::-1], fps, cap_seconds)[::-1]


def score_peaks(scores: dict[str, np.ndarray], settings: MovementSettings) -> np.ndarray:
    combined = np.maximum(scores[PLAYERS[0]], scores[PLAYERS[1]])
    mask = np.zeros(len(combined), dtype=bool)
    mask[local_maxima(combined, settings.min_hit_gap_frames, settings.hit_score_min)] = True
    return mask


def in_play_feature_matrix(tracks: PlayerTracks, scores: dict[str, np.ndarray], settings: MovementSettings) -> np.ndarray:
    samples = tracks_on_sample_indices(tracks)
    if len(samples.frames) == 0:
        return np.zeros((0, len(IN_PLAY_FEATURE_NAMES)))
    signals = {player: player_signals(samples, player, settings) for player in PLAYERS}
    windows = [frames_for_seconds(seconds, settings.fps) for seconds in WINDOW_SECONDS]
    peaks = score_peaks(scores, settings)
    columns = [signals[player]["net_distance"] for player in PLAYERS]
    columns += [centred_window(signals[player][signal], window, stat) for player in PLAYERS for window in windows for signal, stat in PLAYER_WINDOW_STATS]
    columns += [centred_window(scores[player], window, "max") for player in PLAYERS for window in windows]
    columns += [centred_window(peaks.astype(float), window, "sum") for window in windows]
    columns += [seconds_since(peaks, settings.fps), seconds_until(peaks, settings.fps)]
    return np.column_stack(columns)


def in_play_targets(frames: np.ndarray, rally_spans: list[tuple[int, int]]) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    targets = np.zeros(len(frames), dtype=int)
    for start, end in rally_spans:
        targets[(frames >= start) & (frames <= end)] = 1
    return targets


def labelled_rally_spans(labels: pd.DataFrame, fps: float, landing_seconds: float = LANDING_SECONDS) -> list[tuple[int, int]]:
    keys = [column for column in ("set_no", "rally") if column in labels.columns]
    landing = int(round(landing_seconds * fps))
    return sorted((int(rally["frame"].min()), int(rally["frame"].max()) + landing) for _, rally in labels.groupby(keys))


def fit_in_play_model(feature_matrix: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**IN_PLAY_MODEL_PARAMETERS).fit(feature_matrix, targets, sample_weight=sample_weight)


def in_play_probability(model, feature_matrix: np.ndarray, settings: MovementSettings) -> np.ndarray:
    if len(feature_matrix) == 0:
        return np.zeros(0)
    return moving_average(model.predict_proba(feature_matrix)[:, 1], frames_for_seconds(PROBABILITY_SMOOTHING_SECONDS, settings.fps))


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    return [(int(start), int(end) - 1) for start, end in zip(edges[::2], edges[1::2])]


def in_play_spans(probability: np.ndarray, fps: float, threshold: float = IN_PLAY_PROBABILITY_MIN, min_in_play_seconds: float = MIN_IN_PLAY_SECONDS, min_gap_seconds: float = MIN_OUT_OF_PLAY_SECONDS) -> list[tuple[int, int]]:
    mask = np.asarray(probability) >= threshold
    min_gap = frames_for_seconds(min_gap_seconds, fps)
    for start, end in runs_of(~mask):
        if start > 0 and end < len(mask) - 1 and end - start + 1 < min_gap:
            mask[start : end + 1] = True
    min_length = frames_for_seconds(min_in_play_seconds, fps)
    return [(start, end) for start, end in runs_of(mask) if end - start + 1 >= min_length]


def rally_of_span(candidates: list[HitCandidate], span: tuple[int, int], settings: MovementSettings) -> list[HitCandidate]:
    start, end = span
    padding = frames_for_seconds(SPAN_PADDING_SECONDS, settings.fps)
    chain = alternating_hit_chain([candidate for candidate in candidates if start - padding <= candidate.frame <= end + padding], settings)
    if not chain or not settings.infer_serves or chain[0].frame - start <= frames_for_seconds(SERVE_WINDOW_SECONDS, settings.fps):
        return chain
    return [HitCandidate(start, other_player(chain[0].hitter), float("nan"))] + chain


def hits_in_play(tracks: PlayerTracks, scores: dict[str, np.ndarray], probability: np.ndarray, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = tracks_on_sample_indices(tracks)
    candidates = hit_candidates(samples.frames, scores, settings)
    hit_rows, rally_rows = [], []
    for span in in_play_spans(probability, settings.fps):
        rally = rally_of_span(candidates, span, settings)
        if not any(np.isfinite(candidate.score) for candidate in rally):
            continue
        number = len(rally_rows) + 1
        hit_rows.extend({"rally": number, "frame": candidate.frame, "hitter": candidate.hitter, "score": round(candidate.score, 3), "reach": np.nan} for candidate in rally)
        rally_rows.append({"rally": number, "start_frame": span[0], "end_frame": span[1], "hits": len(rally)})
    hits = pd.DataFrame(hit_rows, columns=["rally", "frame", "hitter", "score", "reach"])
    rallies = pd.DataFrame(rally_rows, columns=["rally", "start_frame", "end_frame", "hits"])
    return tables_in_video_frames(hits, rallies, tracks)


def detect_hits_in_play(tracks: PlayerTracks, scores: dict[str, np.ndarray], in_play_model, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability = in_play_probability(in_play_model, in_play_feature_matrix(tracks, scores, settings), settings)
    return hits_in_play(tracks, scores, probability, settings)


def save_in_play_model(model, model_path: Path = IN_PLAY_MODEL_PATH, trained_on: list[str] | None = None) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(IN_PLAY_FEATURE_NAMES), "landing_seconds": LANDING_SECONDS, "trained_on": trained_on or []}, model_path)
    return model_path


def load_in_play_model(model_path: Path = IN_PLAY_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"in-play model not found: {model_path} (train with: python -m vision.in_play <match list> --save {model_path})")
    saved = joblib.load(model_path)
    if list(saved["feature_names"]) != list(IN_PLAY_FEATURE_NAMES):
        raise ValueError(f"{model_path} was trained on other features; retrain it")
    return saved["model"]


@dataclass(frozen=True)
class SpanComparison:
    labelled: int
    found: int
    matched: int
    merged: int
    split: int
    missed: int
    extra: int
    start_errors: tuple[float, ...]
    end_errors: tuple[float, ...]


def overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)


def compare_spans(labelled: list[tuple[int, int]], found: list[tuple[int, int]], fps: float) -> SpanComparison:
    overlaps = [[overlap(truth, span) for span in found] for truth in labelled]
    found_per_truth = [sum(1 for value in row if value > 0) for row in overlaps]
    truths_per_found = [sum(1 for row in overlaps if row[index] > 0) for index in range(len(found))]
    pairs = sorted(((value, truth, span) for truth, row in enumerate(overlaps) for span, value in enumerate(row) if value > 0), reverse=True)
    used_truth, used_found, matched = set(), set(), []
    for _, truth, span in pairs:
        if truth not in used_truth and span not in used_found:
            used_truth.add(truth)
            used_found.add(span)
            matched.append((truth, span))
    return SpanComparison(
        labelled=len(labelled),
        found=len(found),
        matched=len(matched),
        merged=sum(1 for count in truths_per_found if count >= 2),
        split=sum(1 for count in found_per_truth if count >= 2),
        missed=sum(1 for count in found_per_truth if count == 0),
        extra=sum(1 for count in truths_per_found if count == 0),
        start_errors=tuple((found[span][0] - labelled[truth][0]) / fps for truth, span in matched),
        end_errors=tuple((found[span][1] - labelled[truth][1]) / fps for truth, span in matched),
    )


def first_row_on_serve(hits: pd.DataFrame, rallies: pd.DataFrame, labels: pd.DataFrame, tolerance: int) -> tuple[int, int]:
    keys = [column for column in ("set_no", "rally") if column in labels.columns]
    serves = [int(rally["frame"].min()) for _, rally in labels.groupby(keys)]
    firsts = [int(rally_hits["frame"].min()) for _, rally_hits in hits.groupby("rally")] if len(hits) else []
    return len(match_hits(firsts, serves, tolerance)), len(serves)


def stretch_truth(labels: pd.DataFrame, tracks: PlayerTracks, fps: float) -> list[tuple[int, int]]:
    first, last = int(tracks.frames[0]), int(tracks.frames[-1])
    return [(max(start, first), min(end, last)) for start, end in labelled_rally_spans(labels, fps) if overlap((start, end), (first, last)) > 0]


def in_play_training_rows(match, scores: list[dict[str, np.ndarray]], step: int = TRAINING_SAMPLE_STEP) -> tuple[np.ndarray, np.ndarray]:
    spans = labelled_rally_spans(match.labels, match.fps)
    feature_rows = [in_play_feature_matrix(tracks, stretch_scores, match.settings)[::step] for tracks, stretch_scores in zip(match.tracks, scores)]
    target_rows = [in_play_targets(tracks.frames, spans)[::step] for tracks in match.tracks]
    return np.vstack(feature_rows), np.concatenate(target_rows)


def hand_marked_spans(rallies: pd.DataFrame) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    from vision.label import checked_flags

    checked = pd.Series(checked_flags(rallies) if "bounds_checked" in rallies.columns else True, index=rallies.index, dtype=bool)
    as_spans = lambda table: sorted((int(start), int(end)) for start, end in zip(table["start_frame"], table["end_frame"]))
    return as_spans(rallies[checked]), as_spans(rallies[~checked])


def doubtful_frames(positions: pd.DataFrame) -> np.ndarray:
    if "doubtful" not in positions.columns:
        return np.zeros(0, dtype=int)
    return np.unique(positions.loc[positions["doubtful"].astype(bool), "frame"].astype(int))


def club_sample_mask(frames: np.ndarray, unchecked_spans: list[tuple[int, int]], doubtful: np.ndarray, step: int = TRAINING_SAMPLE_STEP) -> np.ndarray:
    sampled = np.zeros(len(frames), dtype=bool)
    sampled[::step] = True
    return sampled & ~in_play_targets(frames, unchecked_spans).astype(bool) & ~np.isin(frames, doubtful)


def club_in_play_rows(club, hit_model, rallies: pd.DataFrame, doubtful: np.ndarray, step: int = TRAINING_SAMPLE_STEP) -> tuple[np.ndarray, np.ndarray]:
    from vision.movement import sample_hit_scores

    tracks = club.tracks[0]
    in_play, unchecked = hand_marked_spans(rallies)
    features = in_play_feature_matrix(tracks, sample_hit_scores(tracks, hit_model, club.settings), club.settings)
    keep = club_sample_mask(np.asarray(tracks.frames, dtype=int), unchecked, doubtful, step)
    return features[keep], in_play_targets(tracks.frames, in_play)[keep]


def load_club_rows(video_path: Path, labels_file: Path, hit_model_path: Path) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    from vision.hit_training import load_club
    from vision.label_scores import rallies_file_for
    from vision.movement import load_hit_model, players_files
    from vision.players import video_last_frame

    club = load_club(Path(video_path), Path(labels_file))
    positions_file, _ = players_files(Path(video_path), 0, video_last_frame(Path(video_path)))
    rallies = pd.read_csv(rallies_file_for(Path(labels_file)))
    return club.name, club_in_play_rows(club, load_hit_model(hit_model_path), rallies, doubtful_frames(pd.read_csv(positions_file)))


def video_spans(spans: list[tuple[int, int]], tracks: PlayerTracks) -> list[tuple[int, int]]:
    return [(int(tracks.frames[start]), int(tracks.frames[end])) for start, end in spans]


def method_row(match, method: str, comparisons: list[SpanComparison], hits: pd.DataFrame, rallies: pd.DataFrame, frame_hits: int, frame_total: int) -> dict:
    detected = hits[hits["score"].notna()] if len(hits) else hits
    serve_first, serves = first_row_on_serve(hits, rallies, match.labels, tolerance_frames(MATCH_TOLERANCE_SECONDS[-1], match.fps))
    starts = [error for comparison in comparisons for error in comparison.start_errors]
    ends = [error for comparison in comparisons for error in comparison.end_errors]
    row = {"match": match.name, "fps": match.fps, "method": method}
    for key in ("labelled", "found", "matched", "merged", "split", "missed", "extra"):
        row[f"rallies_{key}"] = sum(getattr(comparison, key) for comparison in comparisons)
    row.update(
        start_error_median_s=float(np.median(np.abs(starts))) if starts else np.nan,
        end_error_median_s=float(np.median(np.abs(ends))) if ends else np.nan,
        start_error_mean_s=float(np.mean(starts)) if starts else np.nan,
        end_error_mean_s=float(np.mean(ends)) if ends else np.nan,
        frames_right=frame_hits,
        frames=frame_total,
        first_row_on_serve=serve_first,
        serves=serves,
    )
    for window, tolerance in match_tolerances(match.fps):
        summary = evaluation_summary(detected, match.labels, tolerance)
        row.update({f"{key}_{window:g}s": summary[key] for key in ("labelled", "detected", "matched")})
    return row


def frame_agreement(found: list[tuple[int, int]], tracks: PlayerTracks, truth: list[tuple[int, int]]) -> tuple[int, int]:
    predicted = in_play_targets(tracks.frames, found)
    return int((predicted == in_play_targets(tracks.frames, truth)).sum()), len(tracks.frames)


def held_out_match_rows(match, scores: list[dict[str, np.ndarray]], in_play_model) -> list[dict]:
    from vision.movement import hits_from_scores

    results = {"rule": ([], [], [], 0, 0), "in_play": ([], [], [], 0, 0)}
    for tracks, stretch_scores in zip(match.tracks, scores):
        if len(tracks.frames) == 0:
            continue
        truth = stretch_truth(match.labels, tracks, match.fps)
        rule_hits, rule_rallies = hits_from_scores(tracks, stretch_scores, match.settings)
        play_hits, play_rallies = detect_hits_in_play(tracks, stretch_scores, in_play_model, match.settings)
        for method, (hits, rallies) in (("rule", (rule_hits, rule_rallies)), ("in_play", (play_hits, play_rallies))):
            comparisons, hit_tables, rally_tables, right, total = results[method]
            found = [(int(start), int(end)) for start, end in zip(rallies["start_frame"], rallies["end_frame"])]
            comparisons.append(compare_spans(truth, found, match.fps))
            offset = sum(len(table) for table in rally_tables)
            hit_tables.append(hits.assign(rally=hits["rally"] + offset) if len(hits) else hits)
            rally_tables.append(rallies.assign(rally=rallies["rally"] + offset) if len(rallies) else rallies)
            frames_right, frames_total = frame_agreement(found, tracks, truth)
            results[method] = (comparisons, hit_tables, rally_tables, right + frames_right, total + frames_total)
    rows = []
    for method, (comparisons, hit_tables, rally_tables, right, total) in results.items():
        hits = pd.concat([table for table in hit_tables if len(table)], ignore_index=True) if any(len(table) for table in hit_tables) else pd.DataFrame(columns=["rally", "frame", "hitter", "score", "reach"])
        rallies = pd.concat([table for table in rally_tables if len(table)], ignore_index=True) if any(len(table) for table in rally_tables) else pd.DataFrame(columns=["rally", "start_frame", "end_frame", "hits"])
        rows.append(method_row(match, method, comparisons, hits, rallies, right, total))
    return rows


def cached_scores(match, training: list[str], rows_by_match: dict, cache_dir: Path) -> list[dict[str, np.ndarray]]:
    from vision.hit_training import fit_on, match_scores

    cache_file = Path(cache_dir) / f"{match.name}.joblib"
    if cache_file.is_file():
        cached = joblib.load(cache_file)
        if cached["training"] == sorted(training):
            return cached["scores"]
    scores = match_scores(fit_on(training, rows_by_match), match)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"training": sorted(training), "scores": scores}, cache_file)
    return scores


def gap_coverage(rally_spans: list[tuple[int, int]], stretches: list[tuple[int, int]], fps: float) -> dict:
    tracked = np.zeros(max([end for _, end in stretches] + [end for _, end in rally_spans] + [0]) + 1, dtype=bool)
    for start, end in stretches:
        tracked[start : end + 1] = True
    in_play = np.zeros_like(tracked)
    for start, end in rally_spans:
        in_play[start : end + 1] = True
    ordered = sorted(rally_spans)
    gaps = [(end + 1, next_start - 1) for (_, end), (next_start, _) in zip(ordered, ordered[1:]) if next_start - 1 >= end + 1]
    return {
        "tracked_minutes": float(tracked.sum() / fps / 60),
        "out_of_play_minutes": float((tracked & ~in_play).sum() / fps / 60),
        "out_of_play_share": float((tracked & ~in_play).sum() / max(tracked.sum(), 1)),
        "median_gap_seconds": float(np.median([(end - start + 1) / fps for start, end in gaps])) if gaps else float("nan"),
        "gap_tracked_share": float(np.mean([tracked[start : end + 1].mean() for start, end in gaps])) if gaps else float("nan"),
    }


def gap_lines(match_list: Path) -> list[str]:
    from vision.hit_training import exclusion_reason, read_match_list
    from vision.label_offset import match_labels, read_offset_record, read_stretch_list
    from vision.players import video_fps_and_width

    lines = ["| Match | fps | Tracked (min) | Out of play (min) | Out-of-play share | Median gap between rallies (s) | Tracked share of a gap, mean |", "|---|---|---|---|---|---|---|"]
    for folder, video_path in read_match_list(match_list):
        record = read_offset_record(video_path)
        if exclusion_reason(record):
            continue
        fps, _ = video_fps_and_width(video_path)
        spans = labelled_rally_spans(match_labels(video_path, record, int(record["fine_offset_frames"])), fps)
        coverage = gap_coverage(spans, read_stretch_list(video_path), fps)
        lines.append(
            f"| {folder} | {fps:g} | {coverage['tracked_minutes']:.1f} | {coverage['out_of_play_minutes']:.1f} | {coverage['out_of_play_share']:.1%} "
            f"| {coverage['median_gap_seconds']:.1f} | {coverage['gap_tracked_share']:.1%} |"
        )
    return lines


def summary_lines(table: pd.DataFrame) -> list[str]:
    lines = []
    for method, rows in table.groupby("method", sort=False):
        totals = rows.sum(numeric_only=True)
        lines.append(
            f"{method}: rallies labelled {int(totals['rallies_labelled'])}, found {int(totals['rallies_found'])}, matched {int(totals['rallies_matched'])}, "
            f"merged {int(totals['rallies_merged'])}, split {int(totals['rallies_split'])}, missed {int(totals['rallies_missed'])}, extra {int(totals['rallies_extra'])}; "
            f"frames right {totals['frames_right'] / totals['frames']:.1%}; first row on the serve {int(totals['first_row_on_serve'])}/{int(totals['serves'])}; "
            f"hits recall ±0.25 s {totals['matched_0.25s'] / totals['labelled_0.25s']:.1%}, ±0.5 s {totals['matched_0.5s'] / totals['labelled_0.5s']:.1%}, "
            f"precision ±0.5 s {totals['matched_0.5s'] / max(totals['detected_0.5s'], 1):.1%}"
        )
    return lines


def run(parsed: argparse.Namespace) -> int:
    from vision.hit_training import loaded_matches, training_rows, write_table

    matches = loaded_matches(parsed.match_list)
    if not matches:
        raise ValueError("no usable match in the list")
    names = sorted(matches)
    rows_by_match = {name: training_rows(matches[name]) for name in names}
    scores = {name: cached_scores(matches[name], [other for other in names if other != name], rows_by_match, parsed.out / "scores") for name in names}
    play_rows = {name: in_play_training_rows(matches[name], scores[name]) for name in names}
    print(f"{len(names)} matches: {sum(len(rows[1]) for rows in play_rows.values())} sampled frames, {sum(int(rows[1].sum()) for rows in play_rows.values())} in play")
    if parsed.held_out:
        table_rows = []
        for name in names:
            others = [other for other in names if other != name]
            model = fit_in_play_model(np.vstack([play_rows[other][0] for other in others]), np.concatenate([play_rows[other][1] for other in others]))
            table_rows += held_out_match_rows(matches[name], scores[name], model)
            print(f"held out {name}")
        table = write_table(table_rows, parsed.out / "held_out.csv")
        for line in summary_lines(table):
            print(line)
    if parsed.save:
        print(f"saved {save_with_clubs(play_rows, names, parsed)}")
    return 0


def save_with_clubs(play_rows: dict, names: list[str], parsed: argparse.Namespace) -> str:
    from vision.hit_training import match_weights

    clubs = dict(load_club_rows(video, labels, hit_model) for video, labels, hit_model in parsed.club)
    for name, (_, targets) in clubs.items():
        print(f"{name}: {len(targets)} sampled frames, {int(targets.sum())} in play, weight share {parsed.club_weight / len(clubs):.3f}")
    rows = {**play_rows, **clubs}
    sources = names + sorted(clubs)
    weights = match_weights(sources, rows, sorted(clubs), parsed.club_weight) if clubs else None
    model = fit_in_play_model(np.vstack([rows[name][0] for name in sources]), np.concatenate([rows[name][1] for name in sources]), weights)
    club_text = f" and {len(clubs)} club videos, club weight {parsed.club_weight:.3f}" if clubs else ""
    return f"{save_in_play_model(model, parsed.save, sources)} (trained on {len(names)} matches{club_text})"


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.in_play",
        description="Train and measure a per-frame in-play model on ShuttleSet matches (rallies = in-play stretches).",
    )
    parser.add_argument("match_list", type=Path, help="match list of vision.hit_training: ShuttleSet match folder and video path per line")
    parser.add_argument("--held-out", action="store_true", help="score each match with an in-play model trained on the others, next to the rally rule")
    parser.add_argument("--save", type=Path, metavar="MODEL", help="train on every listed match and save the in-play model")
    parser.add_argument("--gaps", action="store_true", help="only print how much out-of-play time the tracked stretches hold, per match, and exit")
    parser.add_argument("--club", nargs=3, action="append", default=[], metavar=("VIDEO", "LABELS", "HIT_MODEL"), help="with --save: add a club video; in play is between the hand-marked bounds of LABELS' rallies file (only bounds_checked rallies when it has that column; frames inside unchecked ones are left out), doubtful frames are left out, and its hit scores come from HIT_MODEL (one not trained on this video); repeatable")
    parser.add_argument("--club-weight", type=float, default=1 / 3, help="with --club: share of the total sample weight carried by the club videos together (default 1/3)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="folder for the tables and the cached hit scores (default data/in_play)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if not parsed.held_out and not parsed.save and not parsed.gaps:
        print("error: give --held-out, --save MODEL, --gaps or a mix", file=sys.stderr)
        return 2
    if parsed.club and not parsed.save:
        print("error: --club needs --save", file=sys.stderr)
        return 2
    if not 0 <= parsed.club_weight < 1:
        print(f"error: --club-weight must be in [0, 1), got {parsed.club_weight}", file=sys.stderr)
        return 2
    try:
        if parsed.gaps:
            for line in gap_lines(parsed.match_list):
                print(line)
            if not parsed.held_out and not parsed.save:
                return 0
        return run(parsed)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
