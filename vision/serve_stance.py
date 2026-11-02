import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.court import CENTRE_X_M, COURT_WIDTH_M, NET_Y_M, SHORT_SERVICE_LINE_FROM_NET_M
from vision.hits import match_hits, tolerance_frames
from vision.movement import (
    PLAYERS,
    SAME_HIT_AS_SERVE_SECONDS,
    HitCandidate,
    MovementSettings,
    PlayerTracks,
    alternating_hit_chain,
    detected_hits_only,
    filled,
    hit_candidates,
    hits_from_scores,
    motion_features,
    moving_average,
    tables_in_video_frames,
    tracks_on_sample_indices,
)

STILL_SPEED_M_PER_S = 0.5
STANCE_SECONDS = 0.5
MOVE_SPEED_M_PER_S = 0.8
MOVE_WINDOW_SECONDS = 1.5
SERVE_REACH_SECONDS = 0.5
SERVE_LEAD_SECONDS = 0.3
MIN_SERVE_GAP_SECONDS = 3.0
STANCE_GAP_SECONDS = 0.2
COURT_MARGIN_M = 0.5
STILL_SPEED_GRID = (0.3, 0.5, 0.8)
STANCE_SECONDS_GRID = (0.5, 1.0, 1.5)
MOVE_SPEED_GRID = (0.8, 1.2)
TUNING_TOLERANCE_SECONDS = 0.5
SERVE_BEFORE_STRETCH_SECONDS = 2.0
SERVE_AFTER_STRETCH_START_SECONDS = 1.0


@dataclass(frozen=True)
class StanceSettings:
    still_speed_m_per_s: float = STILL_SPEED_M_PER_S
    stance_seconds: float = STANCE_SECONDS
    move_speed_m_per_s: float = MOVE_SPEED_M_PER_S


@dataclass(frozen=True)
class ServeGuess:
    frame: int
    server: str
    stance_start: int
    stance_end: int
    from_hit: bool


def frames_of(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def smoothed(values, settings: MovementSettings) -> np.ndarray:
    return moving_average(filled(np.asarray(values, dtype=float), settings.interpolation_limit_frames), settings.position_smoothing_frames)


def in_service_court(x: np.ndarray, y: np.ndarray, margin_m: float = COURT_MARGIN_M) -> np.ndarray:
    from_net = np.abs(y - NET_Y_M)
    inside_width = (x >= -margin_m) & (x <= COURT_WIDTH_M + margin_m)
    return inside_width & (from_net >= SHORT_SERVICE_LINE_FROM_NET_M - margin_m) & (from_net <= NET_Y_M + margin_m)


def diagonal(near_x: np.ndarray, far_x: np.ndarray) -> np.ndarray:
    return (near_x - CENTRE_X_M) * (far_x - CENTRE_X_M) < 0


def runs_of(mask: np.ndarray, min_length: int, max_gap: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    runs = []
    for index in indices:
        if runs and index - runs[-1][1] <= max_gap + 1:
            runs[-1][1] = int(index)
        else:
            runs.append([int(index), int(index)])
    return [(start, end) for start, end in runs if end - start + 1 >= min_length]


def stance_mask(tracks: PlayerTracks, settings: MovementSettings, stance: StanceSettings) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    samples = tracks_on_sample_indices(tracks)
    speeds = {player: motion_features(samples.points_of(player), samples.boxes_of(player), settings)["speed"] for player in PLAYERS}
    near_x, near_y = smoothed(samples.near_points[:, 0], settings), smoothed(samples.near_points[:, 1], settings)
    far_x, far_y = smoothed(samples.far_points[:, 0], settings), smoothed(samples.far_points[:, 1], settings)
    still = (speeds["me"] < stance.still_speed_m_per_s) & (speeds["opp"] < stance.still_speed_m_per_s)
    placed = in_service_court(near_x, near_y) & in_service_court(far_x, far_y) & diagonal(near_x, far_x)
    return np.nan_to_num(still & placed, nan=0).astype(bool), speeds


def first_mover(speeds: dict[str, np.ndarray], after: int, window: int, threshold: float) -> str | None:
    best = None
    for player in PLAYERS:
        moving = np.flatnonzero(speeds[player][after + 1 : after + 1 + window] > threshold)
        if len(moving) and (best is None or moving[0] < best[0]):
            best = (int(moving[0]), player)
    return None if best is None else best[1]


def serve_hit_near(hits: pd.DataFrame, server: str, earliest: int, latest: int, stance_end: int) -> int | None:
    if not len(hits):
        return None
    near = hits[(hits["hitter"] == server) & (hits["frame"] >= earliest) & (hits["frame"] <= latest)]
    return None if near.empty else int(near["frame"].iloc[(near["frame"] - stance_end).abs().argmin()])


def later_of_close_guesses(guesses: list[ServeGuess], min_gap: int) -> list[ServeGuess]:
    kept = []
    for guess in sorted(guesses, key=lambda guess: guess.frame):
        if kept and guess.frame - kept[-1].frame < min_gap:
            kept[-1] = guess
        else:
            kept.append(guess)
    return kept


def serve_guesses(tracks: PlayerTracks, hits: pd.DataFrame, settings: MovementSettings, stance: StanceSettings = StanceSettings()) -> list[ServeGuess]:
    mask, speeds = stance_mask(tracks, settings, stance)
    frames = np.asarray(tracks.frames, dtype=int)
    reach, lead = frames_of(SERVE_REACH_SECONDS, settings.fps), frames_of(SERVE_LEAD_SECONDS, settings.fps)
    guesses = []
    for start, end in runs_of(mask, frames_of(stance.stance_seconds, settings.fps), frames_of(STANCE_GAP_SECONDS, settings.fps)):
        server = first_mover(speeds, end, frames_of(MOVE_WINDOW_SECONDS, settings.fps), stance.move_speed_m_per_s)
        if server is None:
            continue
        end_frame = int(frames[end])
        hit_frame = serve_hit_near(hits, server, int(frames[max(start, end - lead)]), int(frames[min(end + reach, len(frames) - 1)]), end_frame)
        guesses.append(ServeGuess(end_frame if hit_frame is None else hit_frame, server, int(frames[start]), end_frame, hit_frame is not None))
    return later_of_close_guesses(guesses, frames_of(MIN_SERVE_GAP_SECONDS, settings.fps))


def serve_for_stretch(guesses: list[ServeGuess], stretch_start: int, settings: MovementSettings) -> ServeGuess | None:
    earliest = stretch_start - frames_of(SERVE_BEFORE_STRETCH_SECONDS, settings.fps)
    latest = stretch_start + frames_of(SERVE_AFTER_STRETCH_START_SECONDS, settings.fps)
    near = [guess for guess in guesses if earliest <= guess.frame <= latest]
    return min(near, key=lambda guess: abs(guess.frame - stretch_start)) if near else None


def score_at(hits: pd.DataFrame, frame: int, hitter: str) -> float:
    rows = hits[(hits["frame"] == frame) & (hits["hitter"] == hitter)] if len(hits) else hits
    return float(rows["score"].iloc[0]) if len(rows) else float("nan")


def rally_after_serve(candidates: list[HitCandidate], serve: HitCandidate, stretch_end: int, settings: MovementSettings) -> list[HitCandidate]:
    from vision.in_play import SPAN_PADDING_SECONDS

    same_hit = frames_of(SAME_HIT_AS_SERVE_SECONDS, settings.fps)
    latest = stretch_end + frames_of(SPAN_PADDING_SECONDS, settings.fps)
    chain = alternating_hit_chain([candidate for candidate in candidates if serve.frame + same_hit < candidate.frame <= latest], settings)
    while chain and chain[0].hitter == serve.hitter:
        chain = chain[1:]
    return [serve] + chain


def sample_index_of(frame: int, tracks: PlayerTracks) -> int:
    frames = np.asarray(tracks.frames, dtype=int)
    return int(min(np.searchsorted(frames, frame), len(frames) - 1))


def stance_rallies(tracks: PlayerTracks, stretch_scores: dict[str, np.ndarray], hit_scores: dict[str, np.ndarray], in_play_model, settings: MovementSettings, stretch_settings: MovementSettings, stance: StanceSettings = StanceSettings()) -> tuple[pd.DataFrame, pd.DataFrame]:
    from vision.in_play import in_play_feature_matrix, in_play_probability, in_play_spans, rally_of_span

    samples = tracks_on_sample_indices(tracks)
    probability = in_play_probability(in_play_model, in_play_feature_matrix(tracks, stretch_scores, settings), settings)
    stretch_hits = detected_hits_only(hits_from_scores(tracks, stretch_scores, stretch_settings)[0])
    guesses = serve_guesses(tracks, stretch_hits, settings, stance)
    sample_guesses = [replace_frame(guess, sample_index_of(guess.frame, tracks)) for guess in guesses]
    candidates = hit_candidates(samples.frames, hit_scores, settings)
    hit_rows, rally_rows = [], []
    for start, end in in_play_spans(probability, settings.fps):
        guess = serve_for_stretch(sample_guesses, start, settings)
        if guess is None:
            rally = rally_of_span(candidates, (start, end), settings)
        else:
            serve = HitCandidate(guess.frame, guess.server, score_at(stretch_hits, int(tracks.frames[guess.frame]), guess.server) if guess.from_hit else float("nan"))
            rally = rally_after_serve(candidates, serve, end, settings)
        if not any(np.isfinite(candidate.score) for candidate in rally):
            continue
        number = len(rally_rows) + 1
        hit_rows.extend({"rally": number, "frame": candidate.frame, "hitter": candidate.hitter, "score": round(candidate.score, 3), "reach": np.nan} for candidate in rally)
        rally_rows.append({"rally": number, "start_frame": start if guess is None else guess.frame, "end_frame": end, "hits": len(rally)})
    hits = pd.DataFrame(hit_rows, columns=["rally", "frame", "hitter", "score", "reach"])
    rallies = pd.DataFrame(rally_rows, columns=["rally", "start_frame", "end_frame", "hits"])
    return tables_in_video_frames(hits, rallies, tracks)


def replace_frame(guess: ServeGuess, frame: int) -> ServeGuess:
    return ServeGuess(frame, guess.server, guess.stance_start, guess.stance_end, guess.from_hit)


def serve_scores(guesses: list[ServeGuess], labelled_serves, tolerance: int) -> tuple[int, int, int]:
    matched = len(match_hits([guess.frame for guess in guesses], list(labelled_serves), tolerance))
    return matched, len(guesses), len(labelled_serves)


def f1_of(matched: int, guessed: int, labelled: int) -> float:
    return 2 * matched / (guessed + labelled) if guessed + labelled else 0.0


def grid() -> list[StanceSettings]:
    return [StanceSettings(still, seconds, move) for still, seconds, move in itertools.product(STILL_SPEED_GRID, STANCE_SECONDS_GRID, MOVE_SPEED_GRID)]


@dataclass(frozen=True)
class TrainingVideo:
    name: str
    tracks: PlayerTracks
    hits: pd.DataFrame
    serves: tuple[int, ...]
    settings: MovementSettings
    labelled_span: tuple[int, int]


def load_training_video(video_path: Path, labels_file: Path, hit_model: Path) -> TrainingVideo:
    from vision.hit_training import load_club
    from vision.movement import detect_hits, detected_hits_only, load_hit_model

    club = load_club(Path(video_path), Path(labels_file))
    tracks = club.tracks[0]
    hits, _ = detect_hits(tracks, load_hit_model(hit_model), club.settings)
    serves = tuple(sorted(int(frame) for frame in club.labels.loc[club.labels["type"] == "serve", "frame"]))
    labelled_span = (int(club.labels["frame"].min()), int(club.labels["frame"].max()))
    return TrainingVideo(club.name, tracks, detected_hits_only(hits), serves, club.settings, labelled_span)


def guesses_inside(guesses: list[ServeGuess], span: tuple[int, int], margin: int) -> list[ServeGuess]:
    return [guess for guess in guesses if span[0] - margin <= guess.frame <= span[1] + margin]


def tuning_rows(videos: list[TrainingVideo]) -> list[dict]:
    rows = []
    for stance in grid():
        totals = [0, 0, 0]
        for video in videos:
            tolerance = tolerance_frames(TUNING_TOLERANCE_SECONDS, video.settings.fps)
            guesses = guesses_inside(serve_guesses(video.tracks, video.hits, video.settings, stance), video.labelled_span, tolerance)
            scores = serve_scores(guesses, video.serves, tolerance)
            totals = [total + score for total, score in zip(totals, scores)]
        rows.append({"still_speed_m_per_s": stance.still_speed_m_per_s, "stance_seconds": stance.stance_seconds, "move_speed_m_per_s": stance.move_speed_m_per_s, "matched": totals[0], "guessed": totals[1], "labelled": totals[2], "f1": f1_of(*totals)})
    return rows


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.serve_stance", description="Tune the serve-stance rule on labelled training videos (serves found at +-0.5 s).")
    parser.add_argument("--train", nargs=3, action="append", required=True, metavar=("VIDEO", "LABELS", "HIT_MODEL"), help="a fully labelled, fully tracked training video, its labels, and a hit model not trained on it; repeatable")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        videos = [load_training_video(Path(video), Path(labels), Path(model)) for video, labels, model in parsed.train]
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    table = pd.DataFrame(tuning_rows(videos)).sort_values(["f1", "matched"], ascending=False)
    print(table.to_string(index=False))
    best = table.iloc[0]
    print(f"best: --still-speed {best['still_speed_m_per_s']:g} --stance-seconds {best['stance_seconds']:g} --move-speed {best['move_speed_m_per_s']:g} (serves found {int(best['matched'])} of {int(best['labelled'])}, guesses {int(best['guessed'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
