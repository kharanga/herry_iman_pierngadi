import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import open_video, read_frame
from vision.court import video_id, window_was_closed
from vision.hits import HITS_DIR, output_paths
from vision.overlay import draw_tracking_overlay, frame_state, put_text
from vision.players import PLAYERS, PLAYERS_DIR, boxes_path
from vision.players import positions_path as player_positions_path

LABELS_DIR = Path("data/labels")
LABEL_COLUMNS = ["rally", "frame_num", "player", "type_group", "source"]
RALLY_COLUMNS = ["rally", "start_frame", "end_frame"]
SOURCE_HAND, SOURCE_ACCEPTED = "hand", "accepted"
TYPE_KEYS = {"s": "serve", "c": "clear", "d": "drop", "k": "smash", "n": "net", "l": "lift", "v": "drive", "?": "unknown"}
PLAYER_KEYS = {"m": "me", "o": "opp"}
STEP_KEYS = {",": -1, ".": 1, "<": -10, ">": 10}
ARROW_KEY_CODES = {2: -1, 3: 1, 81: -1, 83: 1}
NEW_RALLY_SECONDS = 2.0
PROPOSAL_MAX_DISTANCE_FRAMES = 60
LEFT_PANEL_WIDTH = 960
RIGHT_PANEL_WIDTH = 480
LOOP_MAX_FRAMES = 900
TIMELINE_HEIGHT = 48
KEY_HELP = (
    "space play/pause left   , . one frame   < > ten frames (arrows too)   [ ] previous/next rally",
    "h hit here, then m/o hitter + s serve c clear d drop k smash n net l lift v drive ? unknown",
    "a accept nearest proposal (then hitter + type)   x delete nearest label   u undo   r rally start/end here",
    "+ add rally here   - delete rally   1/2/3 loop speed   w write   q write and quit   esc quit without writing",
)
COLOR_LABEL_BGR = (0, 220, 0)
COLOR_PROPOSAL_BGR = (0, 200, 255)
COLOR_CURSOR_BGR = (255, 255, 255)
COLOR_PENDING_BGR = (255, 0, 255)


@dataclass(frozen=True)
class HitLabel:
    rally: int
    frame: int
    player: str
    type_group: str
    source: str


@dataclass(frozen=True)
class RallySpan:
    rally: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class Proposal:
    frame: int
    hitter: str


@dataclass(frozen=True)
class PendingHit:
    frame: int
    source: str
    player: str | None = None


@dataclass(frozen=True)
class LabelState:
    labels: tuple[HitLabel, ...]
    rallies: tuple[RallySpan, ...]
    proposals: tuple[Proposal, ...]
    rally_index: int
    frame: int
    last_frame: int
    fps: float
    pending: PendingHit | None = None
    history: tuple[tuple[tuple[HitLabel, ...], tuple[RallySpan, ...]], ...] = ()
    dirty: bool = False

    @property
    def rally(self) -> RallySpan | None:
        return self.rallies[self.rally_index] if self.rallies else None


def renumbered(rallies) -> tuple[RallySpan, ...]:
    ordered = sorted(rallies, key=lambda span: span.start_frame)
    return tuple(RallySpan(number, span.start_frame, span.end_frame) for number, span in enumerate(ordered, start=1))


def rally_containing(rallies, frame: int) -> int | None:
    for index, span in enumerate(rallies):
        if span.start_frame <= frame <= span.end_frame:
            return index
    return None


def labels_relabelled(labels, rallies) -> tuple[HitLabel, ...]:
    kept = []
    for label in labels:
        index = rally_containing(rallies, label.frame)
        if index is not None:
            kept.append(replace(label, rally=rallies[index].rally))
    return tuple(sorted(kept, key=lambda label: label.frame))


def remembered(state: LabelState) -> LabelState:
    return replace(state, history=state.history[-200:] + ((state.labels, state.rallies),))


def clamp_frame(frame: int, last_frame: int) -> int:
    return max(0, min(int(frame), int(last_frame)))


def step(state: LabelState, frames: int) -> LabelState:
    return replace(state, frame=clamp_frame(state.frame + frames, state.last_frame))


def go_to_rally(state: LabelState, rally_index: int) -> LabelState:
    if not state.rallies:
        return state
    index = max(0, min(rally_index, len(state.rallies) - 1))
    return replace(state, rally_index=index, frame=state.rallies[index].start_frame, pending=None)


def nearest_index(frames, frame: int) -> int | None:
    frames = list(frames)
    if not frames:
        return None
    return min(range(len(frames)), key=lambda index: (abs(frames[index] - frame), frames[index]))


def nearest_label(state: LabelState) -> HitLabel | None:
    index = nearest_index([label.frame for label in state.labels], state.frame)
    return None if index is None else state.labels[index]


def nearest_proposal(state: LabelState, max_distance: int = PROPOSAL_MAX_DISTANCE_FRAMES) -> Proposal | None:
    index = nearest_index([proposal.frame for proposal in state.proposals], state.frame)
    if index is None or abs(state.proposals[index].frame - state.frame) > max_distance:
        return None
    return state.proposals[index]


def begin_hit(state: LabelState) -> LabelState:
    return replace(state, pending=PendingHit(state.frame, SOURCE_HAND))


def accept_nearest_proposal(state: LabelState) -> LabelState:
    proposal = nearest_proposal(state)
    if proposal is None:
        return state
    return replace(state, frame=proposal.frame, pending=PendingHit(proposal.frame, SOURCE_ACCEPTED, proposal.hitter))


def set_pending_player(state: LabelState, player: str) -> LabelState:
    if state.pending is None:
        return state
    return replace(state, pending=replace(state.pending, player=player))


def with_label(state: LabelState, label: HitLabel) -> LabelState:
    others = tuple(existing for existing in state.labels if existing.frame != label.frame)
    return replace(remembered(state), labels=tuple(sorted(others + (label,), key=lambda item: item.frame)), pending=None, dirty=True)


def set_pending_type(state: LabelState, type_group: str) -> LabelState:
    if state.pending is None or state.pending.player is None:
        return state
    rally_index = rally_containing(state.rallies, state.pending.frame)
    rally_number = state.rallies[rally_index].rally if rally_index is not None else (state.rally.rally if state.rally else 0)
    return with_label(state, HitLabel(rally_number, state.pending.frame, state.pending.player, type_group, state.pending.source))


def cancel_pending(state: LabelState) -> LabelState:
    return replace(state, pending=None)


def delete_nearest_label(state: LabelState) -> LabelState:
    label = nearest_label(state)
    if label is None:
        return state
    return replace(remembered(state), labels=tuple(item for item in state.labels if item != label), dirty=True)


def undo(state: LabelState) -> LabelState:
    if not state.history:
        return state
    labels, rallies = state.history[-1]
    return replace(state, labels=labels, rallies=rallies, history=state.history[:-1], pending=None, dirty=True, rally_index=min(state.rally_index, max(len(rallies) - 1, 0)))


def set_rally_bound(state: LabelState) -> LabelState:
    span = state.rally
    if span is None:
        return state
    midpoint = (span.start_frame + span.end_frame) / 2
    updated = replace(span, start_frame=state.frame) if state.frame < midpoint else replace(span, end_frame=state.frame)
    if updated.start_frame > updated.end_frame:
        return state
    rallies = renumbered(state.rallies[: state.rally_index] + (updated,) + state.rallies[state.rally_index + 1 :])
    return replace(remembered(state), rallies=rallies, labels=labels_relabelled(state.labels, rallies), rally_index=rally_containing(rallies, updated.start_frame), dirty=True)


def add_rally(state: LabelState) -> LabelState:
    if rally_containing(state.rallies, state.frame) is not None:
        return state
    end_frame = clamp_frame(state.frame + int(round(NEW_RALLY_SECONDS * state.fps)), state.last_frame)
    rallies = renumbered(state.rallies + (RallySpan(0, state.frame, end_frame),))
    return replace(remembered(state), rallies=rallies, labels=labels_relabelled(state.labels, rallies), rally_index=rally_containing(rallies, state.frame), dirty=True)


def remove_rally(state: LabelState) -> LabelState:
    if state.rally is None:
        return state
    rallies = renumbered(state.rallies[: state.rally_index] + state.rallies[state.rally_index + 1 :])
    index = min(state.rally_index, max(len(rallies) - 1, 0))
    frame = rallies[index].start_frame if rallies else state.frame
    return replace(remembered(state), rallies=rallies, labels=labels_relabelled(state.labels, rallies), rally_index=index, frame=frame, pending=None, dirty=True)


def handle_key(state: LabelState, key: str) -> LabelState:
    if state.pending is not None:
        if key in PLAYER_KEYS:
            return set_pending_player(state, PLAYER_KEYS[key])
        if key in TYPE_KEYS:
            return set_pending_type(state, TYPE_KEYS[key])
        if key == "esc":
            return cancel_pending(state)
        return state
    if key in STEP_KEYS:
        return step(state, STEP_KEYS[key])
    if key == "[":
        return go_to_rally(state, state.rally_index - 1)
    if key == "]":
        return go_to_rally(state, state.rally_index + 1)
    actions = {"h": begin_hit, "a": accept_nearest_proposal, "x": delete_nearest_label, "u": undo, "r": set_rally_bound, "+": add_rally, "-": remove_rally}
    return actions[key](state) if key in actions else state


def labels_frame(state: LabelState) -> pd.DataFrame:
    return pd.DataFrame([(label.rally, label.frame, label.player, label.type_group, label.source) for label in state.labels], columns=LABEL_COLUMNS)


def rallies_frame(state: LabelState) -> pd.DataFrame:
    return pd.DataFrame([(span.rally, span.start_frame, span.end_frame) for span in state.rallies], columns=RALLY_COLUMNS)


def labels_from_frame(table: pd.DataFrame) -> tuple[HitLabel, ...]:
    return tuple(sorted((HitLabel(int(row.rally), int(row.frame_num), str(row.player), str(row.type_group), str(row.source)) for row in table.itertuples()), key=lambda label: label.frame))


def rallies_from_frame(table: pd.DataFrame) -> tuple[RallySpan, ...]:
    return renumbered(RallySpan(int(row.rally), int(row.start_frame), int(row.end_frame)) for row in table.itertuples())


def proposals_from_hits(hits: pd.DataFrame) -> tuple[Proposal, ...]:
    detected = hits[hits["score"].notna()] if "score" in hits.columns else hits
    return tuple(Proposal(int(row.frame), str(row.hitter)) for row in detected.sort_values("frame").itertuples())


def label_paths(video_path: Path, labels_dir: Path = LABELS_DIR) -> tuple[Path, Path]:
    stem = video_id(video_path)
    return labels_dir / f"{stem}.csv", labels_dir / f"{stem}_rallies.csv"


def write_atomically(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent, suffix=".tmp")
    os.close(handle)
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def save_labels(state: LabelState, labels_file: Path, rallies_file: Path) -> LabelState:
    write_atomically(labels_frame(state), Path(labels_file))
    write_atomically(rallies_frame(state), Path(rallies_file))
    return replace(state, dirty=False)


def initial_state(hits: pd.DataFrame, rallies: pd.DataFrame, last_frame: int, fps: float, labels: pd.DataFrame | None = None, label_rallies: pd.DataFrame | None = None) -> LabelState:
    spans = rallies_from_frame(label_rallies) if label_rallies is not None and len(label_rallies) else rallies_from_frame(rallies)
    known = labels_from_frame(labels) if labels is not None else ()
    state = LabelState(known, spans, proposals_from_hits(hits), 0, spans[0].start_frame if spans else 0, int(last_frame), float(fps))
    return state


def video_last_frame_and_fps(video_path: Path) -> tuple[int, float]:
    capture = open_video(Path(video_path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def load_tracks(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    positions_file = player_positions_path(video_path, start_frame, end_frame, players_dir)
    boxes_file = boxes_path(video_path, start_frame, end_frame, players_dir)
    if not positions_file.is_file() or not boxes_file.is_file():
        return None
    return pd.read_csv(positions_file), pd.read_csv(boxes_file)


def timeline_x(frame: int, span: RallySpan, width: int) -> int:
    length = max(span.end_frame - span.start_frame, 1)
    return int(round((frame - span.start_frame) / length * (width - 1)))


def draw_timeline(state: LabelState, width: int, height: int = TIMELINE_HEIGHT) -> np.ndarray:
    bar = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)
    span = state.rally
    if span is None:
        put_text(bar, "no rally: press + to add one here", (8, 30), 0.6)
        return bar
    cv2.line(bar, (0, height // 2), (width - 1, height // 2), (120, 120, 120), 1)
    for proposal in state.proposals:
        if span.start_frame <= proposal.frame <= span.end_frame:
            x = timeline_x(proposal.frame, span, width)
            cv2.line(bar, (x, height // 2 - 8), (x, height // 2 + 8), COLOR_PROPOSAL_BGR, 1)
    for label in state.labels:
        if span.start_frame <= label.frame <= span.end_frame:
            x = timeline_x(label.frame, span, width)
            cv2.line(bar, (x, 6), (x, height - 6), COLOR_LABEL_BGR, 2)
            put_text(bar, f"{label.player[0]}{label.type_group[0]}", (x + 2, height - 8), 0.4, COLOR_LABEL_BGR)
    if span.start_frame <= state.frame <= span.end_frame:
        x = timeline_x(state.frame, span, width)
        cv2.line(bar, (x, 0), (x, height - 1), COLOR_CURSOR_BGR, 2)
    return bar


def status_line(state: LabelState) -> str:
    span = state.rally
    rally_text = f"rally {span.rally} of {len(state.rallies)} [{span.start_frame}-{span.end_frame}]" if span else "no rally"
    inside = [label for label in state.labels if span and span.start_frame <= label.frame <= span.end_frame]
    pending = ""
    if state.pending is not None:
        pending = f"   HIT at f{state.pending.frame}: press m/o" if state.pending.player is None else f"   HIT at f{state.pending.frame} by {state.pending.player}: press a type key"
    return f"f{state.frame}  {rally_text}  labels in rally: {len(inside)}  total: {len(state.labels)}{'  (unsaved)' if state.dirty else ''}{pending}"


def compose_panels(left: np.ndarray, right: np.ndarray, timeline: np.ndarray, status: str, help_lines=KEY_HELP) -> np.ndarray:
    help_height = 22 * len(help_lines) + 12
    height = max(left.shape[0], right.shape[0]) + timeline.shape[0] + 30 + help_height
    width = left.shape[1] + right.shape[1] + 10
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + 10 : left.shape[1] + 10 + right.shape[1]] = right
    top = max(left.shape[0], right.shape[0])
    canvas[top : top + timeline.shape[0], : timeline.shape[1]] = timeline
    put_text(canvas, status, (8, top + timeline.shape[0] + 22), 0.6)
    for index, line in enumerate(help_lines):
        put_text(canvas, line, (8, top + timeline.shape[0] + 30 + 22 * (index + 1)), 0.5, (200, 200, 200))
    return canvas


def scaled(frame: np.ndarray, width: int) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def read_loop_frames(capture: cv2.VideoCapture, span: RallySpan, width: int = RIGHT_PANEL_WIDTH, max_frames: int = LOOP_MAX_FRAMES) -> tuple[list[np.ndarray], list[int]]:
    count = span.end_frame - span.start_frame + 1
    stride = max(1, int(np.ceil(count / max_frames)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, span.start_frame)
    frames, indices = [], []
    for offset in range(count):
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break
        if offset % stride == 0:
            frames.append(scaled(frame, width))
            indices.append(span.start_frame + offset)
    return frames, indices


def loop_position(indices: list[int], elapsed_seconds: float, fps: float, speed: float) -> int:
    if not indices:
        return 0
    video_seconds = (indices[-1] - indices[0] + 1) / fps
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    progressed = (elapsed_seconds * speed) % max(video_seconds, 1e-9)
    return min(int(progressed * fps / stride), len(indices) - 1)


def draw_loop_panel(frames: list[np.ndarray], indices: list[int], position: int, cursor_frame: int, speed: float) -> np.ndarray:
    if not frames:
        panel = np.zeros((270, RIGHT_PANEL_WIDTH, 3), dtype=np.uint8)
        put_text(panel, "no rally slice", (8, 24), 0.6)
        return panel
    panel = frames[position].copy()
    span = RallySpan(0, indices[0], indices[-1])
    width = panel.shape[1]
    cv2.line(panel, (0, panel.shape[0] - 6), (width - 1, panel.shape[0] - 6), (120, 120, 120), 2)
    x_cursor = timeline_x(cursor_frame, span, width) if indices[0] <= cursor_frame <= indices[-1] else None
    if x_cursor is not None:
        cv2.line(panel, (x_cursor, panel.shape[0] - 16), (x_cursor, panel.shape[0]), COLOR_CURSOR_BGR, 2)
    x_now = timeline_x(indices[position], span, width)
    cv2.circle(panel, (x_now, panel.shape[0] - 6), 4, COLOR_PROPOSAL_BGR, -1)
    put_text(panel, f"loop f{indices[position]}  x{speed:g}", (8, 22), 0.55)
    return panel


def key_name(code: int) -> str | None:
    if code in (-1, 255):
        return None
    if code == 27:
        return "esc"
    if code == 32:
        return "space"
    if code in ARROW_KEY_CODES:
        return "," if ARROW_KEY_CODES[code] < 0 else "."
    try:
        return chr(code)
    except ValueError:
        return None


def label_video(
    video_path: Path,
    labels_file: Path,
    rallies_file: Path,
    players_dir: Path = PLAYERS_DIR,
    hits_dir: Path = HITS_DIR,
    window_name: str = "label hits",
) -> int:
    last_frame, fps = video_last_frame_and_fps(video_path)
    hits_file, rallies_source = output_paths(video_path, 0, last_frame, hits_dir)
    if not hits_file.is_file() or not rallies_source.is_file():
        raise FileNotFoundError(f"hit files not found: {hits_file}, {rallies_source} (run: python -m vision.hits {video_path} --start 0 --end {last_frame})")
    labels = pd.read_csv(labels_file) if Path(labels_file).is_file() else None
    label_rallies = pd.read_csv(rallies_file) if Path(rallies_file).is_file() else None
    state = initial_state(pd.read_csv(hits_file), pd.read_csv(rallies_source), last_frame, fps, labels, label_rallies)
    tracks = load_tracks(video_path, 0, last_frame, players_dir)
    hits_table = pd.read_csv(hits_file)
    rallies_table = pd.read_csv(rallies_source)
    for line in KEY_HELP:
        print(line)
    capture = open_video(Path(video_path))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    loop_frames, loop_indices, loop_rally = [], [], None
    playing, speed, loop_started = False, 0.5, time.monotonic()
    left_cache: tuple[int, np.ndarray] | None = None
    try:
        while True:
            if state.rally != loop_rally:
                loop_rally = state.rally
                loop_frames, loop_indices = read_loop_frames(capture, loop_rally) if loop_rally else ([], [])
                loop_started = time.monotonic()
            if left_cache is None or left_cache[0] != state.frame:
                frame = read_frame(capture, state.frame)
                if frame is None:
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                if tracks is not None:
                    frame = draw_tracking_overlay(frame, frame_state(state.frame, tracks[0], tracks[1], hits_table, rallies_table))
                left_cache = (state.frame, scaled(frame, LEFT_PANEL_WIDTH))
            left = left_cache[1].copy()
            if state.pending is not None:
                put_text(left, "HIT", (left.shape[1] // 2 - 30, 40), 1.2, COLOR_PENDING_BGR)
            position = loop_position(loop_indices, time.monotonic() - loop_started, fps, speed)
            right = draw_loop_panel(loop_frames, loop_indices, position, state.frame, speed)
            cv2.imshow(window_name, compose_panels(left, right, draw_timeline(state, LEFT_PANEL_WIDTH), status_line(state)))
            code = cv2.waitKey(16 if playing else 30)
            if window_was_closed(window_name):
                return 0
            key = key_name(code & 0xFF) if code != -1 else None
            if key is None:
                if playing:
                    state = step(state, 1)
                continue
            if key == "esc" and state.pending is None:
                return 0
            if key == "space":
                playing = not playing
                continue
            if key in ("1", "2", "3") and state.pending is None:
                speed = {"1": 0.25, "2": 0.5, "3": 1.0}[key]
                loop_started = time.monotonic()
                continue
            if key in ("w", "q") and state.pending is None:
                state = save_labels(state, labels_file, rallies_file)
                print(f"saved {labels_file} and {rallies_file} ({len(state.labels)} labels, {len(state.rallies)} rallies)")
                if key == "q":
                    return 0
                continue
            state = handle_key(state, key)
    finally:
        capture.release()
        if not window_was_closed(window_name):
            cv2.destroyWindow(window_name)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.label", description="Label hits by hand, rally by rally, starting from the detector's proposals.")
    parser.add_argument("video", type=Path, help="path to the set video")
    parser.add_argument("--match-id", required=True, help="recording id (kept in the window title and the printed summary)")
    parser.add_argument("--labels", type=Path, help="labels CSV to read and write (default data/labels/<video id>.csv; the rallies file sits next to it)")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files for the overlay (default data/players)")
    parser.add_argument("--hits-dir", type=Path, default=HITS_DIR, help="root of the hits/rallies files that give the rallies and proposals (default data/hits)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    labels_file, rallies_file = label_paths(parsed.video)
    if parsed.labels is not None:
        labels_file = Path(parsed.labels)
        rallies_file = labels_file.with_name(labels_file.stem + "_rallies.csv")
    try:
        return label_video(parsed.video, labels_file, rallies_file, parsed.players_dir, parsed.hits_dir, window_name=f"label hits: {parsed.match_id}")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
