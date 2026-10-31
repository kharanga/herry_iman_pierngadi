import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import open_video
from vision.court import COURT_LENGTH_M, COURT_WIDTH_M, court_line_segments, video_id
from vision.hits import HITS_DIR, output_paths
from vision.players import CHECKS_DIR, PLAYERS, PLAYERS_DIR, boxes_path, frame_stride_of, frames_in_range, require_video_file
from vision.players import positions_path as player_positions_path

PLAYER_COLOR_BGR = {"me": (255, 160, 0), "opp": (0, 0, 255)}
HIT_FLASH_COLOR_BGR = (0, 220, 0)
SERVE_FLASH_COLOR_BGR = (255, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_SHADOW_BGR = (0, 0, 0)
HIT_FLASH_FRAMES = 10
DEFAULT_OUTPUT_WIDTH_PX = 960
DEFAULT_OUTPUT_STRIDE = 2
DIAGRAM_PX_PER_M = 60
DIAGRAM_MARGIN_PX = 40
RALLY_COLORS_BGR = [(255, 160, 0), (0, 0, 255), (0, 200, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]


@dataclass(frozen=True)
class HitFlash:
    hitter: str
    frame: int
    inferred: bool


@dataclass(frozen=True)
class FrameState:
    frame_index: int
    boxes: dict
    positions: dict
    rally_label: str
    shot_count: int | None
    flash: HitFlash | None


def rows_by_player(table: pd.DataFrame, frame_index: int) -> dict:
    rows = table[table["frame"] == frame_index]
    return {player: (None if rows[rows["player"] == player].empty else rows[rows["player"] == player].iloc[0]) for player in PLAYERS}


def rally_at(frame_index: int, rallies: pd.DataFrame):
    inside = rallies[(rallies["start_frame"] <= frame_index) & (frame_index <= rallies["end_frame"])]
    return None if inside.empty else inside.iloc[0]


def hit_flash_at(frame_index: int, hits: pd.DataFrame, flash_frames: int = HIT_FLASH_FRAMES) -> HitFlash | None:
    recent = hits[(hits["frame"] <= frame_index) & (frame_index < hits["frame"] + flash_frames)]
    if recent.empty:
        return None
    latest = recent.sort_values("frame").iloc[-1]
    return HitFlash(str(latest["hitter"]), int(latest["frame"]), bool(pd.isna(latest["score"])))


def frame_state(frame_index: int, positions: pd.DataFrame, boxes: pd.DataFrame, hits: pd.DataFrame, rallies: pd.DataFrame, flash_frames: int = HIT_FLASH_FRAMES) -> FrameState:
    rally = rally_at(frame_index, rallies)
    if rally is None:
        rally_label, shot_count = "between rallies", None
    else:
        rally_label = f"rally {int(rally['rally'])} of {len(rallies)}"
        shot_count = int(((hits["rally"] == rally["rally"]) & (hits["frame"] <= frame_index)).sum())
    return FrameState(frame_index, rows_by_player(boxes, frame_index), rows_by_player(positions, frame_index), rally_label, shot_count, hit_flash_at(frame_index, hits, flash_frames))


def put_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7, color=TEXT_COLOR_BGR) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_SHADOW_BGR, max(2, int(round(3 * scale))), cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def scaled_box(box_row, scale: float) -> tuple[int, int, int, int]:
    return tuple(int(round(float(box_row[name]) * scale)) for name in ("left", "top", "right", "bottom"))


def draw_tracking_overlay(frame: np.ndarray, state: FrameState, scale: float = 1.0) -> np.ndarray:
    drawn = frame.copy()
    height = drawn.shape[0]
    for player in PLAYERS:
        box_row = state.boxes.get(player)
        color = PLAYER_COLOR_BGR[player]
        if box_row is None:
            continue
        left, top, right, bottom = scaled_box(box_row, scale)
        flashing = state.flash is not None and state.flash.hitter == player
        flash_color = SERVE_FLASH_COLOR_BGR if flashing and state.flash.inferred else HIT_FLASH_COLOR_BGR
        cv2.rectangle(drawn, (left, top), (right, bottom), flash_color if flashing else color, 6 if flashing else 2)
        cv2.circle(drawn, ((left + right) // 2, bottom), 6, color, -1)
        position = state.positions.get(player)
        label = player if position is None else f"{player} ({float(position['x_m']):.1f}, {float(position['y_m']):.1f}) m"
        put_text(drawn, label, (left, max(top - 8, 18)), 0.6, color)
        if flashing:
            kind = "serve (inferred)" if state.flash.inferred else "hit"
            put_text(drawn, f"{kind} f{state.flash.frame}", (left, min(bottom + 24, height - 8)), 0.7, flash_color)
    missing = [player for player in PLAYERS if state.boxes.get(player) is None]
    corner = f"f{state.frame_index}  {state.rally_label}" + (f"  shot {state.shot_count}" if state.shot_count is not None else "")
    put_text(drawn, corner, (12, 30), 0.8)
    if missing:
        put_text(drawn, "not detected: " + ", ".join(missing), (12, 62), 0.7, (0, 200, 255))
    return drawn


def load_overlay_inputs(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR, hits_dir: Path = HITS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions_file = player_positions_path(video_path, start_frame, end_frame, players_dir)
    boxes_file = boxes_path(video_path, start_frame, end_frame, players_dir)
    hits_file, rallies_file = output_paths(video_path, start_frame, end_frame, hits_dir)
    for path in (positions_file, boxes_file, hits_file, rallies_file):
        if not path.is_file():
            raise FileNotFoundError(f"track file not found: {path}")
    return pd.read_csv(positions_file), pd.read_csv(boxes_file), pd.read_csv(hits_file), pd.read_csv(rallies_file)


def tracking_video_path(video_path: Path, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_tracking.mp4"


def encode_h264(raw_path: Path, output_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(str(raw_path), str(output_path))
        return output_path
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path)], check=True)
    raw_path.unlink()
    return output_path


def write_tracking_video(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    players_dir: Path = PLAYERS_DIR,
    hits_dir: Path = HITS_DIR,
    output_width: int = DEFAULT_OUTPUT_WIDTH_PX,
    output_stride: int = DEFAULT_OUTPUT_STRIDE,
    flash_frames: int = HIT_FLASH_FRAMES,
) -> Path:
    require_video_file(video_path)
    positions, boxes, hits, rallies = load_overlay_inputs(video_path, start_frame, end_frame, players_dir, hits_dir)
    capture = open_video(Path(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) / output_stride
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = output_width / width
    size = (output_width, int(round(height * scale)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_stride = frame_stride_of(player_positions_path(video_path, start_frame, end_frame, players_dir))
    with tempfile.TemporaryDirectory() as scratch_dir:
        raw_path = Path(scratch_dir) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        try:
            for frame_index, frame in frames_in_range(capture, start_frame, end_frame, output_stride):
                tracked_index = frame_index - (frame_index - start_frame) % track_stride
                state = frame_state(tracked_index, positions, boxes, hits, rallies, flash_frames)
                writer.write(draw_tracking_overlay(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), state, scale))
        finally:
            writer.release()
            capture.release()
        return encode_h264(raw_path, output_path)


def court_to_diagram(points, px_per_m: float = DIAGRAM_PX_PER_M, margin: int = DIAGRAM_MARGIN_PX) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x = margin + points[:, 0] * px_per_m
    y = margin + (COURT_LENGTH_M - points[:, 1]) * px_per_m
    return np.column_stack([x, y])


def blank_court_diagram(px_per_m: float = DIAGRAM_PX_PER_M, margin: int = DIAGRAM_MARGIN_PX) -> np.ndarray:
    width = int(COURT_WIDTH_M * px_per_m + 2 * margin)
    height = int(COURT_LENGTH_M * px_per_m + 2 * margin)
    diagram = np.full((height, width, 3), (60, 120, 60), dtype=np.uint8)
    for name, (start, end) in court_line_segments().items():
        start_px, end_px = np.round(court_to_diagram([start, end], px_per_m, margin)).astype(int)
        cv2.line(diagram, tuple(start_px), tuple(end_px), (230, 230, 230), 3 if name == "net" else 2)
    return diagram


def rally_of_frame(frame_index: int, rallies: pd.DataFrame) -> int | None:
    rally = rally_at(frame_index, rallies)
    return None if rally is None else int(rally["rally"])


def court_paths_diagram(positions: pd.DataFrame, rallies: pd.DataFrame, px_per_m: float = DIAGRAM_PX_PER_M, margin: int = DIAGRAM_MARGIN_PX) -> np.ndarray:
    diagram = blank_court_diagram(px_per_m, margin)
    ordered = positions.sort_values("frame")
    for player in PLAYERS:
        player_rows = ordered[ordered["player"] == player]
        for rally in rallies.itertuples():
            in_rally = player_rows[(player_rows["frame"] >= rally.start_frame) & (player_rows["frame"] <= rally.end_frame)]
            if len(in_rally) < 2:
                continue
            path = np.round(court_to_diagram(in_rally[["x_m", "y_m"]].to_numpy(), px_per_m, margin)).astype(np.int32)
            color = RALLY_COLORS_BGR[(int(rally.rally) - 1) % len(RALLY_COLORS_BGR)]
            cv2.polylines(diagram, [path.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    put_text(diagram, f"{len(rallies)} rallies, one colour per rally (cycling); me near (bottom), opp far (top)", (8, diagram.shape[0] - 10), 0.45)
    return diagram


def paths_diagram_path(video_path: Path, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_player_paths.png"


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.overlay", description="Draw the saved tracks, hits and rallies onto the video, and the players' paths on a court diagram.")
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--start", type=int, required=True, help="first frame of the tracked range")
    parser.add_argument("--end", type=int, required=True, help="last frame of the tracked range")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files (default data/players)")
    parser.add_argument("--hits-dir", type=Path, default=HITS_DIR, help="root of the hit files (default data/hits)")
    parser.add_argument("--output", type=Path, help="output video (default data/checks/players/<video id>_tracking.mp4)")
    parser.add_argument("--width", type=int, default=DEFAULT_OUTPUT_WIDTH_PX, help="output width in pixels (default 960)")
    parser.add_argument("--every", type=int, default=DEFAULT_OUTPUT_STRIDE, help="write every Nth frame (default 2: 60 fps in, 30 fps out)")
    parser.add_argument("--no-video", action="store_true", help="skip the video")
    parser.add_argument("--diagram", action="store_true", help="also draw the paths diagram to data/checks/players/<video id>_player_paths.png")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        if not parsed.no_video:
            output = write_tracking_video(parsed.video, parsed.start, parsed.end, parsed.output or tracking_video_path(parsed.video), parsed.players_dir, parsed.hits_dir, parsed.width, parsed.every)
            print(f"saved {output}")
        if parsed.diagram:
            positions, _, _, rallies = load_overlay_inputs(parsed.video, parsed.start, parsed.end, parsed.players_dir, parsed.hits_dir)
            diagram_path = paths_diagram_path(parsed.video)
            diagram_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(diagram_path), court_paths_diagram(positions, rallies))
            print(f"saved {diagram_path}")
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
