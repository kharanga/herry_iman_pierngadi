import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vision.in_play import compare_spans, hand_marked_spans, in_play_targets, overlap
from vision.movement import MovementSettings, detected_tables, load_hit_model, load_player_tracks, settings_for_tracks, track_frame_stride
from vision.players import PLAYERS_DIR, video_fps_and_width, video_last_frame

RULE = "rule"


def found_spans(rallies: pd.DataFrame) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in zip(rallies["start_frame"], rallies["end_frame"])]


def outside_unchecked_only(found: list[tuple[int, int]], checked: list[tuple[int, int]], unchecked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    touches = lambda span, spans: any(overlap(span, other) > 0 for other in spans)
    return [span for span in found if touches(span, checked) or not touches(span, unchecked)]


def frames_right(found: list[tuple[int, int]], frames: np.ndarray, checked: list[tuple[int, int]], unchecked: list[tuple[int, int]]) -> tuple[int, int]:
    judged = ~in_play_targets(frames, unchecked).astype(bool)
    agree = in_play_targets(frames, found) == in_play_targets(frames, checked)
    return int((agree & judged).sum()), int(judged.sum())


def bound_row(name: str, found: list[tuple[int, int]], frames: np.ndarray, checked: list[tuple[int, int]], unchecked: list[tuple[int, int]], fps: float) -> dict:
    comparison = compare_spans(checked, outside_unchecked_only(found, checked, unchecked), fps)
    right, judged = frames_right(found, frames, checked, unchecked)
    starts, ends = np.array(comparison.start_errors), np.array(comparison.end_errors)
    return {
        "method": name,
        "frames_right": right,
        "frames_judged": judged,
        "checked_rallies": comparison.labelled,
        "found": len(found),
        "matched": comparison.matched,
        "merged": comparison.merged,
        "split": comparison.split,
        "missed": comparison.missed,
        "extra": comparison.extra,
        "start_error_median_abs_s": float(np.median(np.abs(starts))) if len(starts) else np.nan,
        "end_error_median_abs_s": float(np.median(np.abs(ends))) if len(ends) else np.nan,
        "start_error_mean_s": float(starts.mean()) if len(starts) else np.nan,
        "end_error_mean_s": float(ends.mean()) if len(ends) else np.nan,
    }


def bound_lines(rows: list[dict]) -> list[str]:
    lines = [
        "| Method | Frames right | Found | Matched | Merged | Split | Missed | Extra | Start error, median abs (s) | End error, median abs (s) | Start error, mean (s) | End error, mean (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['frames_right']}/{row['frames_judged']} = {row['frames_right'] / max(row['frames_judged'], 1):.1%} | {row['found']} | {row['matched']} of {row['checked_rallies']} "
            f"| {row['merged']} | {row['split']} | {row['missed']} | {row['extra']} | {row['start_error_median_abs_s']:.2f} | {row['end_error_median_abs_s']:.2f} "
            f"| {row['start_error_mean_s']:+.2f} | {row['end_error_mean_s']:+.2f} |"
        )
    return lines


def method_spans(video_path: Path, players_dir: Path, hit_model: Path, hit_score_min: float, rally_end_score_min: float, in_play_model: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    last_frame = video_last_frame(video_path)
    tracks = load_player_tracks(video_path, 0, last_frame, players_dir)
    base = MovementSettings(hit_score_min=hit_score_min, rally_end_score_min=rally_end_score_min)
    settings = settings_for_tracks(video_path, track_frame_stride(video_path, 0, last_frame, players_dir), base)
    _, rallies = detected_tables(tracks, load_hit_model(hit_model), None if in_play_model == RULE else Path(in_play_model), settings)
    return found_spans(rallies), np.asarray(tracks.frames, dtype=int)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.rally_bounds",
        description="Score the rallies of several detectors against a video's hand-marked rally bounds (bounds_checked rallies only).",
    )
    parser.add_argument("video", type=Path, help="the video, tracked in full")
    parser.add_argument("--rallies", type=Path, required=True, help="hand-marked rallies CSV (rally, start_frame, end_frame[, bounds_checked])")
    parser.add_argument(
        "--method",
        nargs=5,
        action="append",
        required=True,
        metavar=("NAME", "HIT_MODEL", "HIT_SCORE_MIN", "RALLY_END_SCORE_MIN", "IN_PLAY_MODEL"),
        help=f"one detector: its hit model and thresholds, and an in-play model file or '{RULE}' for the gap rule; repeatable",
    )
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files (default data/players)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        checked, unchecked = hand_marked_spans(pd.read_csv(parsed.rallies))
        fps = video_fps_and_width(parsed.video)[0]
        rows = []
        for name, hit_model, hit_score_min, rally_end_score_min, in_play_model in parsed.method:
            found, frames = method_spans(parsed.video, parsed.players_dir, Path(hit_model), float(hit_score_min), float(rally_end_score_min), in_play_model)
            rows.append(bound_row(name, found, frames, checked, unchecked, fps))
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{len(checked)} checked rallies scored; {len(unchecked)} unchecked left out (their frames are not judged, and a detected rally touching only them is not extra)")
    for line in bound_lines(rows):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
