import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vision.hits import match_hits, tolerance_frames
from vision.in_play import compare_spans, hand_marked_spans, in_play_targets, overlap
from vision.movement import MovementSettings, detected_hits_only, detected_tables, load_hit_model, load_player_tracks, settings_for_tracks, track_frame_stride
from vision.players import PLAYERS_DIR, video_fps_and_width, video_last_frame

RULE = "rule"
SERVE_TOLERANCE_SECONDS = (0.25, 0.5)


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


def method_tables(video_path: Path, players_dir: Path, hit_model: Path, hit_score_min: float, rally_end_score_min: float, in_play_model: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    last_frame = video_last_frame(video_path)
    tracks = load_player_tracks(video_path, 0, last_frame, players_dir)
    base = MovementSettings(hit_score_min=hit_score_min, rally_end_score_min=rally_end_score_min)
    settings = settings_for_tracks(video_path, track_frame_stride(video_path, 0, last_frame, players_dir), base)
    hits, rallies = detected_tables(tracks, load_hit_model(hit_model), None if in_play_model == RULE else Path(in_play_model), settings)
    return hits, rallies, np.asarray(tracks.frames, dtype=int)


def saved_tables(hits_file: Path, video_path: Path, players_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    last_frame = video_last_frame(video_path)
    tracks = load_player_tracks(video_path, 0, last_frame, players_dir)
    rallies_file = hits_file.with_name(hits_file.name.replace("hits_", "rallies_", 1))
    return pd.read_csv(hits_file), pd.read_csv(rallies_file), np.asarray(tracks.frames, dtype=int)


def up_to(end_frame: int | None, hits: pd.DataFrame, rallies: pd.DataFrame, frames: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if end_frame is None:
        return hits, rallies, frames
    return hits[hits["frame"] <= end_frame], rallies[rallies["start_frame"] <= end_frame], frames[frames <= end_frame]


def serve_frames(hits: pd.DataFrame) -> list[int]:
    return [int(frame) for frame in hits.sort_values("frame").groupby("rally")["frame"].first()] if len(hits) else []


def out_of_play_hits(hits: pd.DataFrame, checked: list[tuple[int, int]], unchecked: list[tuple[int, int]], last_judged: int) -> int:
    frames = detected_hits_only(hits)["frame"].to_numpy(dtype=int)
    inside = in_play_targets(frames, checked + unchecked).astype(bool)
    return int((~inside & (frames <= last_judged)).sum())


def serve_row(hits: pd.DataFrame, labelled_serves: list[int], checked: list[tuple[int, int]], unchecked: list[tuple[int, int]], frames: np.ndarray, fps: float) -> dict:
    guessed = serve_frames(hits)
    row = {"serves_labelled": len(labelled_serves), "serves_guessed": len(guessed), "hits_out_of_play": out_of_play_hits(hits, checked, unchecked, int(frames.max()) if len(frames) else 0)}
    for seconds in SERVE_TOLERANCE_SECONDS:
        matched = len(match_hits(guessed, labelled_serves, tolerance_frames(seconds, fps)))
        row[f"serves_found_{seconds:g}"] = matched
        row[f"serves_fake_{seconds:g}"] = len(guessed) - matched
    return row


def serve_lines(rows: list[dict]) -> list[str]:
    tolerances = " | ".join(f"Serves found ±{seconds:.2f} s | Fake serves ±{seconds:.2f} s" for seconds in SERVE_TOLERANCE_SECONDS)
    lines = [f"| Method | Serve rows | {tolerances} | Detected hits out of play |", "|---|---|" + "---|---|" * len(SERVE_TOLERANCE_SECONDS) + "---|"]
    for row in rows:
        cells = " | ".join(f"{row[f'serves_found_{seconds:g}']} of {row['serves_labelled']} | {row[f'serves_fake_{seconds:g}']}" for seconds in SERVE_TOLERANCE_SECONDS)
        lines.append(f"| {row['method']} | {row['serves_guessed']} | {cells} | {row['hits_out_of_play']} |")
    return lines


def labelled_serves_of(labels_file: Path, end_frame: int | None) -> list[int]:
    from vision.label import reviewed_rows_only

    labels = reviewed_rows_only(pd.read_csv(labels_file), labels_file)
    serves = labels.loc[labels["type_group"] == "serve", "frame_num"].astype(int)
    return sorted(int(frame) for frame in serves if end_frame is None or frame <= end_frame)


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
        default=[],
        metavar=("NAME", "HIT_MODEL", "HIT_SCORE_MIN", "RALLY_END_SCORE_MIN", "IN_PLAY_MODEL"),
        help=f"one detector: its hit model and thresholds, and an in-play model file or '{RULE}' for the gap rule; repeatable",
    )
    parser.add_argument("--found", nargs=2, action="append", default=[], metavar=("NAME", "HITS_CSV"), help="a saved detector run (python -m vision.movement); its rallies_*.csv sits next to it; repeatable")
    parser.add_argument("--end", type=int, help="last labelled frame: frames after it are not judged, and found rallies starting after it are left out")
    parser.add_argument("--labels", type=Path, help="the hand labels CSV: also score each detector's serve rows (its first row per rally) against the labelled serves, and count detected hits out of play")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files (default data/players)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if not parsed.method and not parsed.found:
        print("error: give at least one --method or --found", file=sys.stderr)
        return 1
    try:
        checked, unchecked = hand_marked_spans(pd.read_csv(parsed.rallies))
        fps = video_fps_and_width(parsed.video)[0]
        labelled_serves = labelled_serves_of(parsed.labels, parsed.end) if parsed.labels else None
        runs = [(name, method_tables(parsed.video, parsed.players_dir, Path(hit_model), float(hit_score_min), float(rally_end_score_min), in_play_model)) for name, hit_model, hit_score_min, rally_end_score_min, in_play_model in parsed.method]
        runs += [(name, saved_tables(Path(hits_file), parsed.video, parsed.players_dir)) for name, hits_file in parsed.found]
        rows, serve_rows = [], []
        for name, tables in runs:
            hits, rallies, frames = up_to(parsed.end, *tables)
            rows.append(bound_row(name, found_spans(rallies), frames, checked, unchecked, fps))
            if labelled_serves is not None:
                serve_rows.append({"method": name, **serve_row(hits, labelled_serves, checked, unchecked, frames, fps)})
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{len(checked)} checked rallies scored; {len(unchecked)} unchecked left out (their frames are not judged, and a detected rally touching only them is not extra)")
    for line in bound_lines(rows):
        print(line)
    if serve_rows:
        print()
        for line in serve_lines(serve_rows):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
