import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ingest.label_translations import UnmappedLabelError, translate_label

SHUTTLESET_SET_DIR = Path("data/CoachAI-Projects/ShuttleSet/set")
CHECKS_DIR = Path("data/checks")
LABEL_FRAME_WIDTH = 1280
MARKER_COLOR_BGR = (0, 0, 255)
MISSING_SHOT_TYPE_CAPTION = "no shot type"
UNMAPPED_SHOT_TYPE_CAPTION = "unmapped shot type"


def set_csv_path(set_dir: Path, match_folder: str, set_no: int) -> Path:
    return set_dir / match_folder / f"set{set_no}.csv"


def check_image_dir(checks_dir: Path, match_folder: str, set_no: int) -> Path:
    return checks_dir / match_folder / f"set{set_no}"


def check_image_name(rally: int, ball_round: int, frame_num: int) -> str:
    return f"rally{rally}_shot{ball_round}_frame{frame_num}.jpg"


def label_scale(frame_width: float) -> float:
    return frame_width / LABEL_FRAME_WIDTH


def scaled_point(label_x: float, label_y: float, scale: float) -> tuple[int, int]:
    return round(label_x * scale), round(label_y * scale)


def shots_with_hit_point(shots: pd.DataFrame) -> pd.DataFrame:
    return shots.dropna(subset=["hit_x", "hit_y"]).reset_index(drop=True)


def spread_positions(row_count: int, pick_count: int) -> list[int]:
    if row_count <= 0 or pick_count <= 0:
        return []
    if row_count <= pick_count:
        return list(range(row_count))
    if pick_count == 1:
        return [0]
    return sorted({round(index * (row_count - 1) / (pick_count - 1)) for index in range(pick_count)})


def pick_spread_shots(shots: pd.DataFrame, pick_count: int = 4) -> pd.DataFrame:
    shots_to_check = shots_with_hit_point(shots)
    return shots_to_check.iloc[spread_positions(len(shots_to_check), pick_count)].reset_index(drop=True)


def english_shot_type(raw_shot_type: object) -> str:
    if not isinstance(raw_shot_type, str) or not raw_shot_type.strip():
        return MISSING_SHOT_TYPE_CAPTION
    try:
        return translate_label("type", raw_shot_type.strip())
    except UnmappedLabelError:
        return UNMAPPED_SHOT_TYPE_CAPTION


def shot_caption(rally: int, ball_round: int, shot_type: str) -> str:
    return f"rally {rally} shot {ball_round} {shot_type}"


def draw_hit_marker(frame: np.ndarray, point: tuple[int, int], caption: str) -> np.ndarray:
    marked_frame = frame.copy()
    cv2.circle(marked_frame, point, 12, MARKER_COLOR_BGR, 3)
    cv2.putText(marked_frame, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, MARKER_COLOR_BGR, 2)
    return marked_frame


def load_set_shots(set_dir: Path, match_folder: str, set_no: int) -> pd.DataFrame:
    csv_path = set_csv_path(set_dir, match_folder, set_no)
    if not csv_path.is_file():
        raise FileNotFoundError(f"set CSV not found: {csv_path} (check the match folder and set number)")
    return pd.read_csv(csv_path)


def open_video(video_path: Path) -> cv2.VideoCapture:
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {video_path}")
    return capture


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    if frame_index < 0:
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_was_read, frame = capture.read()
    return frame if frame_was_read else None


def write_alignment_checks(
    match_folder: str,
    set_no: int,
    video_path: Path,
    offset: int = 0,
    set_dir: Path = SHUTTLESET_SET_DIR,
    checks_dir: Path = CHECKS_DIR,
) -> list[Path]:
    picked_shots = pick_spread_shots(load_set_shots(set_dir, match_folder, set_no))
    capture = open_video(video_path)
    output_dir = check_image_dir(checks_dir, match_folder, set_no)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = label_scale(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"video fps={capture.get(cv2.CAP_PROP_FPS):.2f}, scale={scale:.3f}, offset={offset}")
    written_paths = []
    try:
        for shot in picked_shots.itertuples():
            rally, ball_round, frame_num = int(shot.rally), int(shot.ball_round), int(shot.frame_num)
            frame = read_frame(capture, frame_num + offset)
            if frame is None:
                print(f"could not read frame {frame_num + offset} (rally {rally} shot {ball_round})")
                continue
            marked_frame = draw_hit_marker(
                frame,
                scaled_point(shot.hit_x, shot.hit_y, scale),
                shot_caption(rally, ball_round, english_shot_type(shot.type)),
            )
            image_path = output_dir / check_image_name(rally, ball_round, frame_num)
            cv2.imwrite(str(image_path), marked_frame)
            print(f"saved {image_path}")
            written_paths.append(image_path)
    finally:
        capture.release()
    return written_paths


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.alignment",
        description="Mark ShuttleSet hit points on video frames to check label/video alignment.",
    )
    parser.add_argument("match_folder", help="ShuttleSet match folder name under set/")
    parser.add_argument("set_no", type=int, choices=[1, 2, 3], help="set number")
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--offset", type=int, default=0, help="frames added to each label frame_num")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        if pick_spread_shots(load_set_shots(SHUTTLESET_SET_DIR, parsed.match_folder, parsed.set_no)).empty:
            print("error: the set has no shots with hit_x and hit_y to check", file=sys.stderr)
            return 1
        written_paths = write_alignment_checks(parsed.match_folder, parsed.set_no, parsed.video, parsed.offset)
    except (FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not written_paths:
        print("error: no frames could be read; check the video and offset", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
