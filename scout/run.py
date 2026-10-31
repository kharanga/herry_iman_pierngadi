import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import duckdb
import pandas as pd

from analytics.build_shots import create_shots_table
from analytics.tendencies import tendencies_for
from report.build import UnknownMatchError, UnknownPlayerError
from report.serve import DEFAULT_PORT, report_path_for, serve_page, write_report
from shots.predict import MODEL_PATH, predict_shot_types
from vision import court, features, hits, players

DATABASE_PATH = Path("data/scout.duckdb")
MAX_SET_VIDEOS = 3
VIDEO_STEPS = ("court", "players", "hits")
FORCEABLE_STEPS = VIDEO_STEPS + ("features", "all")
DEVICES = (players.GPU_DEVICE, players.CPU_DEVICE)
REPORT_PLAYERS = ("me", "opp")
FINGERPRINT_BYTES = 1024 * 1024
SOURCE_RECORD_SUFFIX = ".source.json"


class StepFailedError(RuntimeError):
    def __init__(self, step: str, video_path: Path, exit_code: int):
        super().__init__(f"{step} failed for {video_path} (exit code {exit_code}); see the messages above")
        self.step = step
        self.video_path = video_path
        self.exit_code = exit_code


@dataclass(frozen=True)
class SetVideo:
    video_path: Path
    set_no: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class CalibrationInputs:
    frame: int = 0
    pick_points: bool = False
    court_points_by_set: dict[int, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingOptions:
    device: str | None = None
    batch_frames: int = players.DETECTION_BATCH_FRAMES


@dataclass
class StepLog:
    lines: list[str] = field(default_factory=list)
    seconds_by_step: dict[str, float] = field(default_factory=dict)

    def record(self, label: str, outcome: str, seconds: float) -> None:
        line = f"{label}: {outcome} ({seconds:.1f} s)"
        self.lines.append(line)
        self.seconds_by_step[label] = seconds
        print(line, flush=True)


def video_frame_count(video_path: Path) -> int:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count <= 0:
        raise ValueError(f"could not read a frame count from {video_path}")
    return frame_count


def set_videos(video_paths: list[Path], frame_range: tuple[int, int] | None = None) -> list[SetVideo]:
    if not 1 <= len(video_paths) <= MAX_SET_VIDEOS:
        raise ValueError(f"give 1 to {MAX_SET_VIDEOS} set videos, in set order; got {len(video_paths)}")
    stems = [Path(video_path).stem for video_path in video_paths]
    video_ids = [court.video_id(Path(video_path)) for video_path in video_paths]
    if len(set(stems)) != len(stems) or len(set(video_ids)) != len(video_ids):
        raise ValueError(
            f"set videos must have different file names (their outputs are keyed by vision.court.video_id): {video_ids}"
        )
    videos = []
    for set_no, video_path in enumerate(video_paths, start=1):
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        start_frame, end_frame = frame_range if frame_range is not None else (0, video_frame_count(video_path) - 1)
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"bad frame range {start_frame}-{end_frame}")
        videos.append(SetVideo(Path(video_path), set_no, start_frame, end_frame))
    return videos


def step_outputs(step: str, video: SetVideo) -> list[Path]:
    frames = (video.video_path, video.start_frame, video.end_frame)
    if step == "court":
        return [court.calibration_path(video.video_path)]
    if step == "players":
        return [players.positions_path(*frames), players.boxes_path(*frames)]
    if step == "hits":
        return list(hits.output_paths(*frames))
    raise ValueError(f"unknown step: {step!r}")


def court_arguments(video: SetVideo, calibration: CalibrationInputs) -> list[str]:
    arguments = [str(video.video_path), "--frame", str(calibration.frame)]
    if video.set_no in calibration.court_points_by_set:
        return [*arguments, "--points", *calibration.court_points_by_set[video.set_no]]
    if calibration.pick_points:
        return [*arguments, "--pick-points"]
    return arguments


def players_arguments(video: SetVideo, tracking: TrackingOptions) -> list[str]:
    arguments = [str(video.video_path), "--start", str(video.start_frame), "--end", str(video.end_frame)]
    if tracking.device is not None:
        arguments += ["--device", tracking.device]
    return [*arguments, "--batch-frames", str(tracking.batch_frames)]


def step_arguments(
    step: str, video: SetVideo, calibration: CalibrationInputs = CalibrationInputs(), tracking: TrackingOptions = TrackingOptions()
) -> list[str]:
    if step == "court":
        return court_arguments(video, calibration)
    if step == "players":
        return players_arguments(video, tracking)
    if step == "hits":
        return [str(video.video_path), "--start", str(video.start_frame), "--end", str(video.end_frame)]
    raise ValueError(f"unknown step: {step!r}")


STEP_COMMANDS = {
    "court": court.main,
    "players": players.main,
    "hits": hits.main,
}


def is_forced(step: str, forced_steps: frozenset[str]) -> bool:
    return step in forced_steps or "all" in forced_steps


def video_fingerprint(video_path: Path) -> dict:
    path = Path(video_path)
    size = path.stat().st_size
    with path.open("rb") as video_file:
        head = video_file.read(FINGERPRINT_BYTES)
        video_file.seek(max(size - FINGERPRINT_BYTES, 0))
        tail = video_file.read(FINGERPRINT_BYTES)
    return {
        "size": size,
        "head_sha256": hashlib.sha256(head).hexdigest(),
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
    }


def source_record_path(output_path: Path) -> Path:
    return Path(output_path).with_name(Path(output_path).name + SOURCE_RECORD_SUFFIX)


def write_source_record(output_path: Path, video_path: Path, fingerprint: dict) -> Path:
    record_path = source_record_path(output_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"video": str(Path(video_path).resolve()), **fingerprint}, indent=2), encoding="utf-8")
    return record_path


def read_source_record(output_path: Path) -> dict | None:
    record_path = source_record_path(output_path)
    if not record_path.is_file():
        return None
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_same_source(record: dict, fingerprint: dict) -> bool:
    return all(record.get(key) == value for key, value in fingerprint.items())


def video_frame_size(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def calibration_names_video(video: SetVideo) -> bool:
    calibration_file = court.calibration_path(video.video_path)
    try:
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        named_video = calibration.get("video")
    except (OSError, ValueError, AttributeError):
        return False
    if named_video is None or Path(named_video).resolve() != video.video_path.resolve():
        return False
    recorded_size = (calibration.get("frame_width"), calibration.get("frame_height"))
    return None in recorded_size or tuple(recorded_size) == video_frame_size(video.video_path)


def outputs_are_newer_than_video(outputs: list[Path], video_path: Path) -> bool:
    video_modified = Path(video_path).stat().st_mtime_ns
    return all(Path(path).stat().st_mtime_ns > video_modified for path in outputs)


def reuse_decision(outputs: list[Path], fingerprint: dict, may_adopt_unrecorded: bool, video_path: Path | None = None) -> tuple[bool, str]:
    if not all(path.is_file() for path in outputs):
        return False, "missing"
    records = [read_source_record(path) for path in outputs]
    if all(record is not None and is_same_source(record, fingerprint) for record in records):
        return True, "reused"
    other_videos = sorted({record["video"] for record in records if record is not None and not is_same_source(record, fingerprint)})
    if other_videos:
        return False, "made from another video (" + ", ".join(other_videos) + ")"
    if may_adopt_unrecorded and video_path is not None and outputs_are_newer_than_video(outputs, video_path):
        return True, "reused (no source record yet; this video's court.json names it, the outputs are newer than the video, record added)"
    if may_adopt_unrecorded:
        return False, "no source record, and the outputs are older than the video"
    return False, "no source record and no calibration naming this video"


def run_video_steps(
    video: SetVideo,
    forced_steps: frozenset[str],
    log: StepLog,
    calibration: CalibrationInputs = CalibrationInputs(),
    tracking: TrackingOptions = TrackingOptions(),
) -> None:
    fingerprint = video_fingerprint(video.video_path)
    court_was_reused = False
    for step in VIDEO_STEPS:
        label = f"set {video.set_no} {step}"
        outputs = step_outputs(step, video)
        may_adopt_unrecorded = calibration_names_video(video) and (step == "court" or court_was_reused)
        can_reuse, reason = reuse_decision(outputs, fingerprint, may_adopt_unrecorded, video.video_path)
        if can_reuse and not is_forced(step, forced_steps):
            for path in outputs:
                write_source_record(path, video.video_path, fingerprint)
            court_was_reused = court_was_reused or step == "court"
            log.record(label, f"{reason} " + ", ".join(str(path) for path in outputs), 0.0)
            continue
        started = time.perf_counter()
        exit_code = STEP_COMMANDS[step](step_arguments(step, video, calibration, tracking))
        if exit_code != 0:
            raise StepFailedError(step, video.video_path, exit_code)
        for path in outputs:
            write_source_record(path, video.video_path, fingerprint)
        rerun_reason = "" if is_forced(step, forced_steps) or reason == "missing" else f" (existing outputs: {reason})"
        log.record(label, "ran, wrote " + ", ".join(str(path) for path in outputs) + rerun_reason, time.perf_counter() - started)


def set_rows(video: SetVideo, match_id: str, left_handed: frozenset[str]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="scout_set_") as scratch_dir:
        return features.build_own_match(
            video.video_path,
            match_id,
            [(video.start_frame, video.end_frame, video.set_no)],
            Path(scratch_dir) / "set.duckdb",
            left_handed,
        )


def write_match_rows(database_path: Path, match_id: str, rows_by_set: list[pd.DataFrame]) -> int:
    connection = duckdb.connect(str(database_path))
    try:
        create_shots_table(connection)
    finally:
        connection.close()
    return features.write_own_shots(database_path, match_id, pd.concat(rows_by_set, ignore_index=True))


def tendencies_path(match_id: str, player: str) -> Path:
    return report_path_for(match_id).parent / f"tendencies_{player}.json"


def write_tendencies(match_id: str, database_path: Path) -> list[Path]:
    written = []
    for player in REPORT_PLAYERS:
        output_path = tendencies_path(match_id, player)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(tendencies_for(match_id, player, database_path), indent=2), encoding="utf-8")
        written.append(output_path)
    return written


def run_match(
    video_paths: list[Path],
    match_id: str,
    database_path: Path = DATABASE_PATH,
    frame_range: tuple[int, int] | None = None,
    forced_steps: frozenset[str] = frozenset(),
    left_handed: frozenset[str] = frozenset(),
    model_path: Path = MODEL_PATH,
    calibration: CalibrationInputs = CalibrationInputs(),
    tracking: TrackingOptions = TrackingOptions(),
) -> tuple[Path, StepLog]:
    log = StepLog()
    videos = set_videos(video_paths, frame_range)
    unknown_sets = set(calibration.court_points_by_set) - {video.set_no for video in videos}
    if unknown_sets:
        raise ValueError(f"calibration given for set(s) {sorted(unknown_sets)}, but only {len(videos)} video(s)")
    rows_by_set = []
    for video in videos:
        run_video_steps(video, forced_steps, log, calibration, tracking)
        started = time.perf_counter()
        rows = set_rows(video, match_id, left_handed)
        rows_by_set.append(rows)
        log.record(f"set {video.set_no} features", f"{len(rows)} shots", time.perf_counter() - started)
    started = time.perf_counter()
    written = write_match_rows(database_path, match_id, rows_by_set)
    log.record("write shots", f"{written} own rows for {match_id!r} in {database_path}", time.perf_counter() - started)
    started = time.perf_counter()
    predicted = predict_shot_types(database_path, model_path)
    log.record("classifier", f"shot types predicted for {predicted} shots (all sources)", time.perf_counter() - started)
    started = time.perf_counter()
    tendency_files = write_tendencies(match_id, database_path)
    log.record("tendencies", "wrote " + ", ".join(str(path) for path in tendency_files), time.perf_counter() - started)
    started = time.perf_counter()
    report_path = write_report(match_id, "me", database_path=database_path)
    log.record("report", f"wrote {report_path}", time.perf_counter() - started)
    return report_path, log


def parse_forced_steps(force_texts: list[str]) -> frozenset[str]:
    steps = frozenset(step.strip() for text in force_texts for step in text.split(",") if step.strip())
    unknown = steps - set(FORCEABLE_STEPS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step(s) to force: {sorted(unknown)}; choose from {', '.join(FORCEABLE_STEPS)}")
    return steps


def set_number(text: str) -> int:
    try:
        set_no = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a set number first, got {text!r}")
    if not 1 <= set_no <= MAX_SET_VIDEOS:
        raise argparse.ArgumentTypeError(f"set number must be 1 to {MAX_SET_VIDEOS}, got {set_no}")
    return set_no


def court_points_by_set(court_point_groups: list[list[str]]) -> dict[int, tuple[str, ...]]:
    points_by_set = {}
    for group in court_point_groups:
        set_no, points = set_number(group[0]), tuple(group[1:])
        if len(points) < 4:
            raise argparse.ArgumentTypeError(f"--court-points {set_no}: give at least 4 \"NAME=X,Y\" points, got {len(points)}")
        points_by_set[set_no] = points
    return points_by_set


def calibration_inputs(parsed: argparse.Namespace) -> CalibrationInputs:
    return CalibrationInputs(parsed.calibration_frame, parsed.pick_points, court_points_by_set(parsed.court_points))


def batch_frames(text: str) -> int:
    try:
        count = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of frames, got {text!r}")
    if count < 1:
        raise argparse.ArgumentTypeError(f"--batch-frames must be at least 1, got {count}")
    return count


def tracking_options(parsed: argparse.Namespace) -> TrackingOptions:
    return TrackingOptions(parsed.device, parsed.batch_frames)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scout.run",
        description="From set videos to the phone report: court, players, hits and features per video, "
        "then shot types, habit counts and the report for the match.",
    )
    parser.add_argument("videos", type=Path, nargs="+", help=f"one video per set, in set order (1 to {MAX_SET_VIDEOS}); the near player is me")
    parser.add_argument("--match-id", required=True, help="id for this match's rows and report, e.g. club_2026-11-02")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="only this frame range of every video (for testing)")
    parser.add_argument("--force", action="append", default=[], help=f"rerun these steps even if their outputs exist: {', '.join(FORCEABLE_STEPS)}")
    parser.add_argument("--device", choices=DEVICES, help="where player detection runs (default: mps when available, else cpu)")
    parser.add_argument("--batch-frames", type=batch_frames, default=players.DETECTION_BATCH_FRAMES, help=f"frames per detection call (default {players.DETECTION_BATCH_FRAMES})")
    parser.add_argument("--calibration-frame", type=int, default=0, help="frame to calibrate the court on (default 0)")
    parser.add_argument("--pick-points", action="store_true", help="court calibration: pick named court points (for a camera that can't see all 4 corners)")
    parser.add_argument(
        "--court-points", nargs="+", action="append", default=[], metavar="SET_OR_POINT",
        help='court calibration without a window for one set: SET then 4 or more "NAME=X,Y" points (names from python -m vision.court <video> --list-points); repeatable',
    )
    parser.add_argument("--left-handed", default="", help="comma-separated left-handed players: me, opp")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to serve the report on (default {DEFAULT_PORT})")
    parser.add_argument("--no-serve", action="store_true", help="write the report but don't serve it")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        forced_steps = parse_forced_steps(parsed.force)
        left_handed = frozenset(player.strip() for player in parsed.left_handed.split(",") if player.strip())
        report_path, log = run_match(
            parsed.videos,
            parsed.match_id,
            parsed.database,
            tuple(parsed.range) if parsed.range else None,
            forced_steps,
            left_handed,
            calibration=calibration_inputs(parsed),
            tracking=tracking_options(parsed),
        )
    except (argparse.ArgumentTypeError, FileNotFoundError, OSError, ValueError, StepFailedError, UnknownMatchError, UnknownPlayerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Report written: {report_path}")
    if not parsed.no_serve:
        serve_page(report_path, parsed.port)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
