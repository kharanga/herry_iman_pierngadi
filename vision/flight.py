import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import open_video, read_frame
from vision.court import video_id
from vision.label_scores import rallies_file_for
from vision.players import video_fps_and_width
from vision.shuttle import SHUTTLE_DIR

SEEN_WINDOW_SECONDS = 0.5
USABLE_SHARE = 0.5
JUMP_SHARE_OF_WIDTH_PER_FRAME = 0.15
SHEET_HIT_COUNT = 20
SHEET_SEED = 1
SHEET_TILE_SIZE = (640, 360)
SHEET_COLUMNS = 4
CHECKS_DIR = Path("data/checks/shuttle")
TRAIL_COLOR_BGR = (0, 255, 255)
HIT_COLOR_BGR = (0, 0, 255)


def stretch_bounds(path: Path) -> tuple[int, int]:
    _, start, end = path.stem.split("_")
    return int(start), int(end)


def shuttle_track(video_path: Path, first_frame: int, last_frame: int, shuttle_dir: Path = SHUTTLE_DIR) -> pd.DataFrame:
    folder = Path(shuttle_dir) / video_id(Path(video_path))
    files = [path for path in sorted(folder.glob("shuttle_*_*.csv")) if overlaps_range(stretch_bounds(path), first_frame, last_frame)]
    if not files:
        raise FileNotFoundError(f"no shuttle track in {folder} for frames {first_frame}-{last_frame} (run: python -m vision.shuttle {video_path} --start N --end N)")
    track = pd.concat([pd.read_csv(path) for path in files], ignore_index=True).drop_duplicates("frame").sort_values("frame")
    return track[(track["frame"] >= first_frame) & (track["frame"] <= last_frame)].reset_index(drop=True)


def overlaps_range(bounds: tuple[int, int], first_frame: int, last_frame: int) -> bool:
    return bounds[0] <= last_frame and bounds[1] >= first_frame


def detected_points(track: pd.DataFrame) -> pd.DataFrame:
    seen = (track["visibility"] == 1) & (track["inpainted"] == 0)
    return track.loc[seen, ["frame", "x", "y"]].sort_values("frame").reset_index(drop=True)


def frames_seen_after(hit_frame: int, detected: np.ndarray, window_frames: int) -> int:
    return int(np.searchsorted(detected, hit_frame + window_frames, side="right") - np.searchsorted(detected, hit_frame, side="right"))


def has_jump(points: pd.DataFrame, frame_width: int) -> bool:
    if len(points) < 2:
        return False
    steps = np.hypot(np.diff(points["x"].to_numpy(dtype=float)), np.diff(points["y"].to_numpy(dtype=float)))
    frames = np.diff(points["frame"].to_numpy(dtype=float))
    return bool((steps / frames > JUMP_SHARE_OF_WIDTH_PER_FRAME * frame_width).any())


def hit_coverage(labels: pd.DataFrame, seen_points: pd.DataFrame, fps: float, frame_width: int) -> pd.DataFrame:
    window = max(1, int(round(SEEN_WINDOW_SECONDS * fps)))
    detected = seen_points["frame"].to_numpy(dtype=int)
    seen = [frames_seen_after(int(frame), detected, window) for frame in labels["frame_num"]]
    jumps = [has_jump(seen_points[(seen_points["frame"] > frame) & (seen_points["frame"] <= frame + window)], frame_width) for frame in labels["frame_num"]]
    usable = [count >= USABLE_SHARE * window for count in seen]
    return labels.assign(
        seen_frames=seen,
        window_frames=window,
        seen=[count > 0 for count in seen],
        usable=usable,
        smooth=[is_usable and not jump for is_usable, jump in zip(usable, jumps)],
    )


def in_flight_hitter(frames: np.ndarray, rally_hits: pd.DataFrame) -> np.ndarray:
    ordered = rally_hits.sort_values("frame_num")
    index = np.searchsorted(ordered["frame_num"].to_numpy(dtype=int), frames, side="right") - 1
    return ordered["player"].to_numpy()[np.maximum(index, 0)]


def rally_visibility(labels: pd.DataFrame, rallies: pd.DataFrame, detected: np.ndarray) -> pd.DataFrame:
    rows = []
    detected_set = set(detected.tolist())
    for rally in rallies.itertuples(index=False):
        rally_hits = labels[labels["rally"] == rally.rally]
        if rally_hits.empty:
            continue
        frames = np.arange(int(rally.start_frame), int(rally.end_frame) + 1)
        rows.append(pd.DataFrame({"frame": frames, "hitter": in_flight_hitter(frames, rally_hits), "seen": [frame in detected_set for frame in frames]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["frame", "hitter", "seen"])


def share_text(part: int, whole: int) -> str:
    return f"{part}/{whole} = {part / whole:.1%}" if whole else "-"


def coverage_lines(coverage: pd.DataFrame, visibility: pd.DataFrame) -> list[str]:
    lines = ["| Hitter | Labelled hits | Shuttle seen in the 0.5 s after | Usable (seen in half the frames or more) | Usable and smooth (no jump) | Rally frames with the shuttle seen |", "|---|---|---|---|---|---|"]
    groups = [("all", coverage, visibility)] + [(player, coverage[coverage["player"] == player], visibility[visibility["hitter"] == player]) for player in ("me", "opp")]
    for name, hits, frames in groups:
        lines.append(f"| {name} | {len(hits)} | {share_text(int(hits['seen'].sum()), len(hits))} | {share_text(int(hits['usable'].sum()), len(hits))} | {share_text(int(hits['smooth'].sum()), len(hits))} | {share_text(int(frames['seen'].sum()), len(frames))} |")
    return lines


def type_lines(coverage: pd.DataFrame) -> list[str]:
    lines = ["| Labelled type | Hits | Usable | Usable and smooth |", "|---|---|---|---|"]
    for group, hits in sorted(coverage.groupby("type_group"), key=lambda item: -len(item[1])):
        lines.append(f"| {group} | {len(hits)} | {share_text(int(hits['usable'].sum()), len(hits))} | {share_text(int(hits['smooth'].sum()), len(hits))} |")
    return lines


def sheet_tile(frame: np.ndarray, trail: list[tuple[int, int]], caption: str) -> np.ndarray:
    drawn = frame.copy()
    for start, end in zip(trail, trail[1:]):
        cv2.line(drawn, start, end, TRAIL_COLOR_BGR, 2, cv2.LINE_AA)
    for point in trail:
        cv2.circle(drawn, point, 4, TRAIL_COLOR_BGR, -1)
    tile = cv2.resize(drawn, SHEET_TILE_SIZE)
    cv2.putText(tile, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HIT_COLOR_BGR, 2)
    return tile


def write_sheet(video_path: Path, coverage: pd.DataFrame, track: pd.DataFrame, output_path: Path) -> Path:
    sample = coverage.sample(n=min(SHEET_HIT_COUNT, len(coverage)), random_state=SHEET_SEED).sort_values("frame_num")
    seen = track[(track["visibility"] == 1) & (track["inpainted"] == 0)].set_index("frame")
    capture = open_video(Path(video_path))
    tiles = []
    try:
        for hit in sample.itertuples(index=False):
            after = seen.loc[(seen.index > hit.frame_num) & (seen.index <= hit.frame_num + hit.window_frames)]
            trail = [(int(x), int(y)) for x, y in zip(after["x"], after["y"])]
            frame = read_frame(capture, int(hit.frame_num + hit.window_frames // 2))
            caption = f"{hit.frame_num} {hit.player} {hit.type_group}: {hit.seen_frames}/{hit.window_frames} seen{'' if hit.smooth or not hit.usable else ', jump'}"
            tiles.append(sheet_tile(frame, trail, caption))
    finally:
        capture.release()
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % SHEET_COLUMNS)
    rows = [np.hstack(tiles[index : index + SHEET_COLUMNS]) for index in range(0, len(tiles), SHEET_COLUMNS)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))
    return output_path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.flight", description="How often the shuttle tracker sees the shuttle after each labelled hit.")
    parser.add_argument("video", type=Path, help="the video, shuttle-tracked over the labelled range (python -m vision.shuttle)")
    parser.add_argument("--labels", type=Path, required=True, help="hand labels CSV (vision.label_ui); its _rallies.csv sits next to it")
    parser.add_argument("--start", type=int, default=0, help="first tracked frame (default 0)")
    parser.add_argument("--end", type=int, required=True, help="last tracked frame; labels after it are left out")
    parser.add_argument("--shuttle-dir", type=Path, default=SHUTTLE_DIR, help="root of the shuttle tracks (default data/shuttle)")
    parser.add_argument("--sheet", action="store_true", help=f"also draw {SHEET_HIT_COUNT} random hits with the shuttle seen in the 0.5 s after to data/checks/shuttle/")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        labels = pd.read_csv(parsed.labels)
        labels = labels[(labels["frame_num"] >= parsed.start) & (labels["frame_num"] <= parsed.end)].reset_index(drop=True)
        rallies = pd.read_csv(rallies_file_for(parsed.labels))
        track = shuttle_track(parsed.video, parsed.start, parsed.end, parsed.shuttle_dir)
        fps, frame_width = video_fps_and_width(parsed.video)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    seen_points = detected_points(track)
    coverage = hit_coverage(labels, seen_points, fps, int(frame_width))
    visibility = rally_visibility(labels, rallies[rallies["rally"].isin(labels["rally"])], seen_points["frame"].to_numpy(dtype=int))
    print(f"{len(labels)} labelled hits; shuttle detected in {len(seen_points)} of {len(track)} tracked frames")
    for line in coverage_lines(coverage, visibility) + [""] + type_lines(coverage):
        print(line)
    if parsed.sheet:
        print(f"saved {write_sheet(parsed.video, coverage, track, CHECKS_DIR / f'{video_id(parsed.video)}_flight_sheet.jpg')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
