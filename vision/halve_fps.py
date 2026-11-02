import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vision.court import calibration_path
from vision.label_scores import rallies_file_for


def halved_hit_frames(frames) -> np.ndarray:
    return (np.asarray(frames, dtype=int) + 1) // 2


def halved_rally_starts(frames) -> np.ndarray:
    return np.asarray(frames, dtype=int) // 2


def halved_rally_ends(frames) -> np.ndarray:
    return (np.asarray(frames, dtype=int) + 1) // 2


def halved_labels(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.assign(frame_num=halved_hit_frames(labels["frame_num"]))


def ends_before_next_start(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    order = np.argsort(starts, kind="stable")
    next_starts = np.append(starts[order][1:], np.iinfo(int).max)
    clipped = ends.copy()
    clipped[order] = np.minimum(ends[order], next_starts - 1)
    return clipped


def halved_rallies(rallies: pd.DataFrame) -> pd.DataFrame:
    starts = halved_rally_starts(rallies["start_frame"])
    return rallies.assign(start_frame=starts, end_frame=ends_before_next_start(starts, halved_rally_ends(rallies["end_frame"])))


def overlapping_rallies(rallies: pd.DataFrame) -> list[tuple[int, int]]:
    ordered = rallies.sort_values("start_frame")
    return [(int(first.rally), int(second.rally)) for first, second in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)) if second.start_frame <= first.end_frame]


def label_problems(labels: pd.DataFrame, rallies: pd.DataFrame) -> list[str]:
    problems = [f"hits collide at frame {frame}" for frame, count in labels["frame_num"].value_counts().items() if count > 1]
    problems += [f"rallies {first} and {second} overlap" for first, second in overlapping_rallies(rallies)]
    bounds = rallies.set_index("rally")
    for hit in labels.itertuples(index=False):
        if hit.rally in bounds.index and not bounds.loc[hit.rally, "start_frame"] <= hit.frame_num <= bounds.loc[hit.rally, "end_frame"]:
            problems.append(f"hit at frame {hit.frame_num} falls outside rally {hit.rally}")
    return problems


def halve_label_files(labels_file: Path, output_file: Path) -> list[str]:
    labels = halved_labels(pd.read_csv(labels_file))
    rallies = halved_rallies(pd.read_csv(rallies_file_for(labels_file)))
    problems = label_problems(labels, rallies)
    if not problems:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(output_file, index=False)
        rallies.to_csv(rallies_file_for(output_file), index=False)
    return problems


def halve_calibration(video_path: Path, halved_video_path: Path) -> Path:
    source = calibration_path(video_path)
    target = calibration_path(halved_video_path)
    calibration = json.loads(source.read_text())
    calibration["frame_index"] = int(calibration["frame_index"]) // 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2))
    overlay = source.with_name("court_overlay.jpg")
    if overlay.is_file():
        shutil.copyfile(overlay, target.with_name("court_overlay.jpg"))
    return target


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.halve_fps",
        description="Carry a video's labels and calibration over to a copy with every second frame kept (frame k of the copy = frame 2k of the original).",
    )
    parser.add_argument("video", type=Path, help="the original video")
    parser.add_argument("halved_video", type=Path, help="the copy at half the frame rate")
    parser.add_argument("--labels", type=Path, nargs=2, metavar=("LABELS", "OUTPUT"), help="labels CSV to halve and where to write it; the rallies files sit next to each")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        print(f"saved {halve_calibration(parsed.video, parsed.halved_video)}")
        if parsed.labels:
            labels_file, output_file = parsed.labels
            problems = halve_label_files(labels_file, output_file)
            if problems:
                print("error: " + "; ".join(problems), file=sys.stderr)
                return 1
            print(f"saved {output_file} and {rallies_file_for(output_file)}")
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
