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

from analytics.shot_types import TYPE_GROUP_BY_TYPE_DETAIL
from vision.alignment import open_video, read_frame
from vision.court import video_id, window_was_closed
from vision.hits import HIT_COLUMNS, HITS_DIR, output_paths
from vision.hits import RALLY_COLUMNS as DETECTED_RALLY_COLUMNS
from vision.overlay import draw_tracking_overlay, frame_state, put_text
from vision.players import PLAYERS, PLAYERS_DIR, boxes_path
from vision.players import positions_path as player_positions_path

LABELS_DIR = Path("data/labels")
LABEL_COLUMNS = ["rally", "frame_num", "player", "type_group", "type_detail", "source"]
RALLY_COLUMNS = ["rally", "start_frame", "end_frame", "bounds_checked"]
SOURCE_HAND, SOURCE_ACCEPTED = "hand", "accepted"
SOURCE_MODEL, SOURCE_CONFIRMED, SOURCE_CORRECTED, SOURCE_MOVED, SOURCE_REJECTED = "model", "confirmed", "corrected_type", "moved", "rejected"
SOURCE_CORRECTED_HITTER = "corrected_hitter"
REVIEW_COLUMNS = ["score", "model_frame", "model_type_detail", "model_player"]
CONFIRM_KEY, MOVE_KEY = "y", "g"
REVIEW_REACH_SECONDS = 0.5
RETYPE_SOURCE = "retype"
HIT_OUTSIDE_RALLY_NOTICE = "a hit must be inside a rally: press b at the serve first, or b / e to widen the rally"
TYPE_KEYS = {"c": "clear", "d": "drop", "k": "smash", "n": "net", "l": "lift", "v": "drive", "?": "unknown"}
SERVE_GROUP = "serve"
SERVE_WITHOUT_LENGTH_TEXT = "serve ?"
SERVE_KEY = "s"
SERVE_LENGTH_KEYS = {"s": "serve_short", "l": "serve_long"}
PLAYER_KEYS = {"m": "me", "o": "opp"}
STEP_KEYS = {",": -1, ".": 1, "<": -10, ">": 10}
RALLY_START_KEY, RALLY_END_KEY = "b", "e"
DEFAULT_SEGMENT_SECONDS = 60.0
UNCHECKED_START_REACH_SECONDS = 3.0
ARROW_KEY_CODES = {2: -1, 3: 1, 81: -1, 83: 1}
NEW_RALLY_SECONDS = 2.0
PROPOSAL_MAX_DISTANCE_FRAMES = 60
LEFT_PANEL_WIDTH = 960
RIGHT_PANEL_WIDTH = 480
LOOP_MAX_FRAMES = 900
TIMELINE_HEIGHT = 48
KEY_HELP = (
    "space play/pause left   , . one frame   < > ten frames (arrows too)   [ ] previous/next rally",
    "h hit here, then m/o hitter + s serve (then s short / l long) c clear d drop k smash n net l lift v drive ? unknown",
    "a accept nearest proposal (then hitter + type)   x delete nearest label   u undo   r rally start/end here",
    "b rally starts here   e rally ends here   (with --blank: [ ] previous/next segment)",
    "+ add rally here   - delete rally   1/2/3 loop speed   w write   q write and quit   esc quit without writing",
)
COLOR_LABEL_BGR = (0, 220, 0)
COLOR_PROPOSAL_BGR = (0, 200, 255)
COLOR_CURSOR_BGR = (255, 255, 255)
COLOR_PENDING_BGR = (255, 0, 255)
COLOR_RALLY_BGR = (60, 100, 60)


@dataclass(frozen=True)
class HitLabel:
    rally: int
    frame: int
    player: str
    type_group: str
    source: str
    type_detail: str = ""
    score: float | None = field(default=None, repr=False)
    model_frame: int | None = field(default=None, repr=False)
    model_type_detail: str = field(default="", repr=False)
    model_player: str = field(default="", repr=False)

    @property
    def from_model(self) -> bool:
        return self.model_frame is not None


@dataclass(frozen=True)
class RallySpan:
    rally: int
    start_frame: int
    end_frame: int
    start_marked: bool = field(default=False, repr=False)
    end_marked: bool = field(default=False, repr=False)

    @property
    def bounds_checked(self) -> bool:
        return self.start_marked and self.end_marked


@dataclass(frozen=True)
class Proposal:
    frame: int
    hitter: str


@dataclass(frozen=True)
class PendingHit:
    frame: int
    source: str
    player: str | None = None
    choosing_serve_length: bool = False


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
    history: tuple[tuple[tuple[HitLabel, ...], tuple[RallySpan, ...], tuple[HitLabel, ...]], ...] = ()
    dirty: bool = False
    segments: tuple[RallySpan, ...] = ()
    segment_index: int = 0
    blank: bool = False
    review: bool = False
    rejected: tuple[HitLabel, ...] = ()
    notice: str = ""

    @property
    def rally(self) -> RallySpan | None:
        return self.rallies[self.rally_index] if self.rallies else None

    @property
    def view(self) -> RallySpan | None:
        return self.segments[self.segment_index] if self.segments else self.rally


def renumbered(rallies) -> tuple[RallySpan, ...]:
    ordered = sorted(rallies, key=lambda span: span.start_frame)
    return tuple(replace(span, rally=number) for number, span in enumerate(ordered, start=1))


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


def lost_labels(labels, rallies) -> list[HitLabel]:
    return [label for label in labels if rally_containing(rallies, label.frame) is None]


def newly_lost_labels(labels, old_rallies, new_rallies) -> list[HitLabel]:
    return [label for label in labels if rally_containing(old_rallies, label.frame) is not None and rally_containing(new_rallies, label.frame) is None]


def labels_kept(labels, rallies) -> tuple[HitLabel, ...]:
    orphans = tuple(label for label in labels if rally_containing(rallies, label.frame) is None)
    return tuple(sorted(labels_relabelled(labels, rallies) + orphans, key=lambda label: label.frame))


def lost_labels_notice(lost: list[HitLabel]) -> str:
    frames = ", ".join(f"f{label.frame}" for label in lost[:3]) + (" ..." if len(lost) > 3 else "")
    return f"not done: that would leave {len(lost)} hit(s) outside every rally ({frames}); move or delete them first (x)"


def unreviewed_model_rows(table: pd.DataFrame) -> int:
    return int((table["source"] == SOURCE_MODEL).sum()) if "source" in table.columns else 0


def reviewed_rows_only(table: pd.DataFrame, labels_file) -> pd.DataFrame:
    unreviewed = unreviewed_model_rows(table)
    if unreviewed:
        raise ValueError(f"{labels_file} has {unreviewed} unreviewed model labels (source = model); review them in python -m vision.label_ui --review first")
    return table


def remembered(state: LabelState) -> LabelState:
    return replace(state, history=state.history[-200:] + ((state.labels, state.rallies, state.rejected),))


def clamp_frame(frame: int, last_frame: int) -> int:
    return max(0, min(int(frame), int(last_frame)))


def segment_spans(last_frame: int, fps: float, segment_seconds: float = DEFAULT_SEGMENT_SECONDS) -> tuple[RallySpan, ...]:
    length = max(1, int(round(segment_seconds * fps)))
    return tuple(RallySpan(number, start, min(start + length, last_frame + 1) - 1) for number, start in enumerate(range(0, last_frame + 1, length), start=1))


def following_frame(state: LabelState) -> LabelState:
    if not state.segments:
        return state
    segment_index = rally_containing(state.segments, state.frame)
    rally_index = rally_containing(state.rallies, state.frame)
    return replace(
        state,
        segment_index=state.segment_index if segment_index is None else segment_index,
        rally_index=state.rally_index if rally_index is None else rally_index,
    )


def at_frame(state: LabelState, frame: int, **changes) -> LabelState:
    return following_frame(replace(state, frame=clamp_frame(frame, state.last_frame), **changes))


def step(state: LabelState, frames: int) -> LabelState:
    return at_frame(state, state.frame + frames)


def go_to_segment(state: LabelState, segment_index: int) -> LabelState:
    index = max(0, min(segment_index, len(state.segments) - 1))
    return at_frame(state, state.segments[index].start_frame, segment_index=index, pending=None)


def go_to_rally(state: LabelState, rally_index: int) -> LabelState:
    if not state.rallies:
        return state
    index = max(0, min(rally_index, len(state.rallies) - 1))
    return at_frame(state, state.rallies[index].start_frame, rally_index=index, pending=None)


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
    if rally_containing(state.rallies, state.frame) is None:
        return replace(state, notice=HIT_OUTSIDE_RALLY_NOTICE)
    return replace(state, pending=PendingHit(state.frame, SOURCE_HAND))


def accept_nearest_proposal(state: LabelState) -> LabelState:
    proposal = nearest_proposal(state)
    if proposal is None:
        return state
    if rally_containing(state.rallies, proposal.frame) is None:
        return replace(state, notice=HIT_OUTSIDE_RALLY_NOTICE)
    return at_frame(state, proposal.frame, pending=PendingHit(proposal.frame, SOURCE_ACCEPTED, proposal.hitter))


def set_pending_player(state: LabelState, player: str) -> LabelState:
    if state.pending is None:
        return state
    return replace(state, pending=replace(state.pending, player=player))


def with_label(state: LabelState, label: HitLabel) -> LabelState:
    others = tuple(existing for existing in state.labels if existing.frame != label.frame)
    return replace(remembered(state), labels=tuple(sorted(others + (label,), key=lambda item: item.frame)), pending=None, dirty=True)


def group_and_detail(type_name: str) -> tuple[str, str]:
    if type_name in TYPE_GROUP_BY_TYPE_DETAIL:
        return TYPE_GROUP_BY_TYPE_DETAIL[type_name], type_name
    return type_name, ""


def set_pending_type(state: LabelState, type_name: str) -> LabelState:
    if state.pending is None or state.pending.player is None:
        return state
    rally_index = rally_containing(state.rallies, state.pending.frame)
    rally_number = state.rallies[rally_index].rally if rally_index is not None else (state.rally.rally if state.rally and not state.segments else 0)
    type_group, type_detail = group_and_detail(type_name)
    return with_label(state, HitLabel(rally_number, state.pending.frame, state.pending.player, type_group, state.pending.source, type_detail))


def ask_serve_length(state: LabelState) -> LabelState:
    if state.pending is None or state.pending.player is None:
        return state
    return replace(state, pending=replace(state.pending, choosing_serve_length=True))


def pending_key(state: LabelState, key: str) -> LabelState:
    if key == "esc":
        return cancel_pending(state)
    if state.pending.choosing_serve_length:
        return set_pending_type(state, SERVE_LENGTH_KEYS[key]) if key in SERVE_LENGTH_KEYS else state
    if key in PLAYER_KEYS:
        return set_pending_player(state, PLAYER_KEYS[key])
    if key == SERVE_KEY:
        return ask_serve_length(state)
    if key in TYPE_KEYS:
        return set_pending_type(state, TYPE_KEYS[key])
    return state


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
    labels, rallies, rejected = state.history[-1]
    return replace(state, labels=labels, rallies=rallies, rejected=rejected, history=state.history[:-1], pending=None, dirty=True, rally_index=min(state.rally_index, max(len(rallies) - 1, 0)))


def set_rally_bound(state: LabelState) -> LabelState:
    span = state.rally
    if span is None:
        return state
    midpoint = (span.start_frame + span.end_frame) / 2
    updated = replace(span, start_frame=state.frame, start_marked=False) if state.frame < midpoint else replace(span, end_frame=state.frame, end_marked=False)
    if updated.start_frame > updated.end_frame:
        return state
    rallies = renumbered(state.rallies[: state.rally_index] + (updated,) + state.rallies[state.rally_index + 1 :])
    lost = newly_lost_labels(state.labels, state.rallies, rallies)
    if lost:
        return replace(state, notice=lost_labels_notice(lost))
    return replace(remembered(state), rallies=rallies, labels=labels_kept(state.labels, rallies), rally_index=rally_containing(rallies, updated.start_frame), dirty=True)


def add_rally(state: LabelState) -> LabelState:
    if rally_containing(state.rallies, state.frame) is not None:
        return state
    end_frame = clamp_frame(state.frame + int(round(NEW_RALLY_SECONDS * state.fps)), state.last_frame)
    rallies = renumbered(state.rallies + (RallySpan(0, state.frame, end_frame),))
    return replace(remembered(state), rallies=rallies, labels=labels_kept(state.labels, rallies), rally_index=rally_containing(rallies, state.frame), dirty=True)


def remove_rally(state: LabelState) -> LabelState:
    if state.rally is None:
        return state
    rallies = renumbered(state.rallies[: state.rally_index] + state.rallies[state.rally_index + 1 :])
    lost = newly_lost_labels(state.labels, state.rallies, rallies)
    if lost:
        return replace(state, notice=lost_labels_notice(lost))
    index = min(state.rally_index, max(len(rallies) - 1, 0))
    frame = rallies[index].start_frame if rallies and not state.segments else state.frame
    return at_frame(remembered(state), frame, rallies=rallies, labels=labels_kept(state.labels, rallies), rally_index=index, pending=None, dirty=True)


def with_rallies(state: LabelState, rallies, focus_frame: int) -> LabelState:
    rallies = renumbered(rallies)
    lost = newly_lost_labels(state.labels, state.rallies, rallies)
    if lost:
        return replace(state, notice=lost_labels_notice(lost))
    index = rally_containing(rallies, focus_frame)
    return replace(remembered(state), rallies=rallies, labels=labels_kept(state.labels, rallies), rally_index=0 if index is None else index, dirty=True)


def labels_in(state: LabelState, first_frame: int, last_frame: int) -> bool:
    return any(first_frame <= label.frame <= last_frame for label in state.labels)


def replaced_rally(state: LabelState, index: int, *spans: RallySpan) -> tuple[RallySpan, ...]:
    return state.rallies[:index] + spans + state.rallies[index + 1 :]


def mark_rally_start(state: LabelState) -> LabelState:
    frame = state.frame
    index = rally_containing(state.rallies, frame)
    if index is not None:
        span = state.rallies[index]
        if frame > span.start_frame and labels_in(state, span.start_frame, frame - 1):
            ended = replace(span, end_frame=frame - 1, end_marked=False)
            following = rally_containing(state.rallies, span.end_frame + 1)
            if following is not None and not state.rallies[following].start_marked and not labels_in(state, frame, span.end_frame):
                handed = replace(state.rallies[following], start_frame=frame, start_marked=True)
                rallies = tuple(handed if position == following else (ended if position == index else rally) for position, rally in enumerate(state.rallies))
                return with_rallies(state, rallies, frame)
            started = replace(span, start_frame=frame, start_marked=True)
            return with_rallies(state, replaced_rally(state, index, ended, started), frame)
        return with_rallies(state, replaced_rally(state, index, replace(span, start_frame=frame, start_marked=True)), frame)
    later = [position for position, span in enumerate(state.rallies) if span.start_frame > frame]
    if later:
        following = min(later, key=lambda position: state.rallies[position].start_frame)
        span = state.rallies[following]
        if not span.start_marked and span.start_frame - frame <= UNCHECKED_START_REACH_SECONDS * state.fps:
            return with_rallies(state, replaced_rally(state, following, replace(span, start_frame=frame, start_marked=True)), frame)
    later_starts = [state.rallies[position].start_frame for position in later]
    end_frame = min(later_starts) - 1 if later_starts else state.last_frame
    return with_rallies(state, state.rallies + (RallySpan(0, frame, end_frame, start_marked=True),), frame)


def mark_rally_end(state: LabelState) -> LabelState:
    frame = state.frame
    index = rally_containing(state.rallies, frame)
    if index is None:
        earlier = [position for position, span in enumerate(state.rallies) if span.end_frame < frame]
        if not earlier:
            return state
        index = max(earlier, key=lambda position: state.rallies[position].end_frame)
        return with_rallies(state, replaced_rally(state, index, replace(state.rallies[index], end_frame=frame, end_marked=True)), frame)
    span = state.rallies[index]
    previous = rally_containing(state.rallies, span.start_frame - 1)
    if previous is not None and not state.rallies[previous].end_marked and not span.start_marked and not labels_in(state, span.start_frame, frame):
        handed = replace(state.rallies[previous], end_frame=frame, end_marked=True)
        rest = replace(span, start_frame=frame + 1, start_marked=False)
        rallies = tuple(handed if position == previous else (rest if position == index else rally) for position, rally in enumerate(state.rallies))
        return with_rallies(state, tuple(rally for rally in rallies if rally.start_frame <= rally.end_frame), frame)
    if frame < span.end_frame and labels_in(state, frame + 1, span.end_frame):
        ended = replace(span, end_frame=frame, end_marked=True)
        rest = replace(span, start_frame=frame + 1, start_marked=False)
        return with_rallies(state, replaced_rally(state, index, ended, rest), frame)
    return with_rallies(state, replaced_rally(state, index, replace(span, end_frame=frame, end_marked=True)), frame)


def review_reach_frames(state: LabelState) -> int:
    return max(1, int(round(REVIEW_REACH_SECONDS * state.fps)))


def review_target(state: LabelState) -> HitLabel | None:
    candidates = [label for label in state.labels if label.from_model and abs(label.frame - state.frame) <= review_reach_frames(state)]
    if not candidates:
        return None
    return min(candidates, key=lambda label: (abs(label.frame - state.frame), label.frame))


def hitter_corrected(label: HitLabel) -> bool:
    return label.from_model and bool(label.model_player) and label.player != label.model_player


def reviewed_source(label: HitLabel) -> str:
    if label.frame != label.model_frame:
        return SOURCE_MOVED
    if label.type_detail != label.model_type_detail:
        return SOURCE_CORRECTED
    if hitter_corrected(label):
        return SOURCE_CORRECTED_HITTER
    return SOURCE_CONFIRMED


def set_hitter(state: LabelState, player: str) -> LabelState:
    target = review_target(state)
    if target is None:
        return state
    rehit = replace(target, player=player, model_player=target.model_player or target.player)
    return replaced_label(state, target, replace(rehit, source=reviewed_source(rehit)))


def replaced_label(state: LabelState, old: HitLabel, new: HitLabel) -> LabelState:
    others = tuple(label for label in state.labels if label != old)
    return replace(remembered(state), labels=tuple(sorted(others + (new,), key=lambda label: label.frame)), pending=None, dirty=True)


def confirm_label(state: LabelState) -> LabelState:
    target = review_target(state)
    if target is None:
        return state
    return replaced_label(state, target, replace(target, source=reviewed_source(target)))


def retype_label_at(state: LabelState, frame: int, type_name: str) -> LabelState:
    target = next((label for label in state.labels if label.frame == frame and label.from_model), None)
    if target is None:
        return replace(state, pending=None)
    type_group, type_detail = group_and_detail(type_name)
    retyped = replace(target, type_group=type_group, type_detail=type_detail)
    return replaced_label(state, target, replace(retyped, source=reviewed_source(retyped)))


def begin_retype(state: LabelState, key: str) -> LabelState:
    target = review_target(state)
    if target is None:
        return state
    if key == SERVE_KEY:
        return replace(state, pending=PendingHit(target.frame, RETYPE_SOURCE, target.player, choosing_serve_length=True))
    return retype_label_at(state, target.frame, TYPE_KEYS[key])


def move_label(state: LabelState) -> LabelState:
    target = review_target(state)
    if target is None or any(label.frame == state.frame and label != target for label in state.labels):
        return state
    index = rally_containing(state.rallies, state.frame)
    if index is None:
        return replace(state, notice=HIT_OUTSIDE_RALLY_NOTICE)
    moved = replace(target, frame=state.frame, rally=state.rallies[index].rally)
    return replaced_label(state, target, replace(moved, source=reviewed_source(moved)))


def reject_label(state: LabelState) -> LabelState:
    label = nearest_label(state)
    if label is None or abs(label.frame - state.frame) > review_reach_frames(state):
        return state
    if not label.from_model:
        return delete_nearest_label(state)
    rejected = replace(label, frame=label.model_frame, source=SOURCE_REJECTED)
    return replace(remembered(state), labels=tuple(item for item in state.labels if item != label), rejected=tuple(sorted(state.rejected + (rejected,), key=lambda item: item.frame)), dirty=True)


def review_key(state: LabelState, key: str) -> LabelState | None:
    if state.pending is not None and state.pending.source == RETYPE_SOURCE:
        if key == "esc":
            return cancel_pending(state)
        return retype_label_at(state, state.pending.frame, SERVE_LENGTH_KEYS[key]) if key in SERVE_LENGTH_KEYS else state
    if state.pending is not None:
        return None
    if key == CONFIRM_KEY:
        return confirm_label(state)
    if key == MOVE_KEY:
        return move_label(state)
    if key == "x":
        return reject_label(state)
    if key in PLAYER_KEYS:
        return set_hitter(state, PLAYER_KEYS[key])
    if key in TYPE_KEYS or key == SERVE_KEY:
        return begin_retype(state, key)
    return None


def review_counts(state: LabelState) -> dict[str, int]:
    reviewed = [label for label in state.labels if label.from_model and label.source != SOURCE_MODEL]
    return {
        "unreviewed": sum(label.source == SOURCE_MODEL for label in state.labels),
        "confirmed": sum(label.source == SOURCE_CONFIRMED for label in state.labels),
        "corrected_type": sum(label.type_detail != label.model_type_detail for label in reviewed),
        "corrected_hitter": sum(hitter_corrected(label) for label in reviewed),
        "moved": sum(label.frame != label.model_frame for label in reviewed),
        "rejected": len(state.rejected),
        "added": sum(not label.from_model for label in state.labels),
    }


def handle_key(state: LabelState, key: str) -> LabelState:
    if state.notice:
        state = replace(state, notice="")
    if state.review:
        reviewed = review_key(state, key)
        if reviewed is not None:
            return reviewed
    if state.pending is not None:
        return pending_key(state, key)
    if key in STEP_KEYS:
        return step(state, STEP_KEYS[key])
    if key in ("[", "]"):
        offset = -1 if key == "[" else 1
        return go_to_segment(state, state.segment_index + offset) if state.segments else go_to_rally(state, state.rally_index + offset)
    if state.segments and key in ("r", "-"):
        index = rally_containing(state.rallies, state.frame)
        if index is None:
            return state
        state = replace(state, rally_index=index)
    actions = {
        "h": begin_hit,
        "a": accept_nearest_proposal,
        "x": delete_nearest_label,
        "u": undo,
        "r": set_rally_bound,
        "+": add_rally,
        "-": remove_rally,
        RALLY_START_KEY: mark_rally_start,
        RALLY_END_KEY: mark_rally_end,
    }
    return actions[key](state) if key in actions else state


def label_rows(labels, with_review_columns: bool) -> pd.DataFrame:
    columns = LABEL_COLUMNS + (REVIEW_COLUMNS if with_review_columns else [])
    rows = []
    for label in labels:
        row = [label.rally, label.frame, label.player, label.type_group, label.type_detail, label.source]
        if with_review_columns:
            row += [label.score, label.model_frame, label.model_type_detail, label.model_player]
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    if with_review_columns:
        table["model_frame"] = table["model_frame"].astype("Int64")
    return table


def has_review_columns(state: LabelState) -> bool:
    return state.review or any(label.from_model for label in state.labels + state.rejected)


def labels_frame(state: LabelState) -> pd.DataFrame:
    return label_rows(state.labels, has_review_columns(state))


def rejected_frame(state: LabelState) -> pd.DataFrame:
    return label_rows(state.rejected, True)


def rallies_frame(state: LabelState) -> pd.DataFrame:
    return pd.DataFrame([(span.rally, span.start_frame, span.end_frame, span.bounds_checked) for span in state.rallies], columns=RALLY_COLUMNS)


def detail_of(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def detail_for(type_group: str, type_detail: str) -> str:
    if type_detail in TYPE_GROUP_BY_TYPE_DETAIL:
        return type_detail
    if type_group == SERVE_GROUP or type_group not in TYPE_GROUP_BY_TYPE_DETAIL:
        return ""
    return type_group


def needs_serve_length(label: HitLabel) -> bool:
    return label.type_group == SERVE_GROUP and label.type_detail not in SERVE_LENGTH_KEYS.values()


def type_text(label: HitLabel) -> str:
    if needs_serve_length(label):
        return SERVE_WITHOUT_LENGTH_TEXT
    return label.type_detail or label.type_group


def unfinished_counts(state: "LabelState") -> dict[str, int]:
    return {
        "serves_without_length": sum(needs_serve_length(label) for label in state.labels),
        "unchecked_rallies": sum(not span.bounds_checked for span in state.rallies),
    }


def optional_column(table: pd.DataFrame, column: str) -> pd.Series:
    return table[column] if column in table.columns else pd.Series([None] * len(table), index=table.index, dtype=object)


def labels_from_frame(table: pd.DataFrame) -> tuple[HitLabel, ...]:
    details = table["type_detail"] if "type_detail" in table.columns else pd.Series([""] * len(table), index=table.index)
    labels = (
        HitLabel(
            int(row.rally), int(row.frame_num), str(row.player), str(row.type_group), str(row.source), detail_for(str(row.type_group), detail_of(detail)),
            score=None if pd.isna(score) else float(score),
            model_frame=None if pd.isna(model_frame) else int(model_frame),
            model_type_detail=detail_of(model_detail),
            model_player=detail_of(model_player) or ("" if pd.isna(model_frame) else str(row.player)),
        )
        for row, detail, score, model_frame, model_detail, model_player in zip(
            table.itertuples(), details, optional_column(table, "score"), optional_column(table, "model_frame"), optional_column(table, "model_type_detail"), optional_column(table, "model_player")
        )
    )
    return tuple(sorted(labels, key=lambda label: label.frame))


def checked_flags(table: pd.DataFrame) -> list[bool]:
    if "bounds_checked" not in table.columns:
        return [False] * len(table)
    return [str(value).strip().lower() in ("true", "1") for value in table["bounds_checked"]]


def rallies_from_frame(table: pd.DataFrame) -> tuple[RallySpan, ...]:
    return renumbered(
        RallySpan(int(row.rally), int(row.start_frame), int(row.end_frame), start_marked=checked, end_marked=checked)
        for row, checked in zip(table.itertuples(), checked_flags(table))
    )


def proposals_from_hits(hits: pd.DataFrame) -> tuple[Proposal, ...]:
    detected = hits[hits["score"].notna()] if "score" in hits.columns else hits
    return tuple(Proposal(int(row.frame), str(row.hitter)) for row in detected.sort_values("frame").itertuples())


def rejected_path_for(labels_file: Path) -> Path:
    return Path(labels_file).with_name(f"{Path(labels_file).stem}_rejected.csv")


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
    if state.rejected or rejected_path_for(labels_file).is_file():
        write_atomically(rejected_frame(state), rejected_path_for(labels_file))
    return replace(state, dirty=False)


def initial_state(
    hits: pd.DataFrame,
    rallies: pd.DataFrame,
    last_frame: int,
    fps: float,
    labels: pd.DataFrame | None = None,
    label_rallies: pd.DataFrame | None = None,
    segment_seconds: float | None = None,
    blank: bool = False,
    review: bool = False,
    rejected: pd.DataFrame | None = None,
) -> LabelState:
    spans = rallies_from_frame(label_rallies) if label_rallies is not None and len(label_rallies) else rallies_from_frame(rallies)
    known = labels_from_frame(labels) if labels is not None else ()
    segments = segment_spans(int(last_frame), float(fps), segment_seconds) if segment_seconds else ()
    first_frame = 0 if segments else (spans[0].start_frame if spans else 0)
    known_rejected = labels_from_frame(rejected) if rejected is not None and len(rejected) else ()
    return following_frame(LabelState(known, spans, proposals_from_hits(hits), 0, first_frame, int(last_frame), float(fps), segments=segments, blank=blank, review=review, rejected=known_rejected))


def pending_prompt(pending: PendingHit) -> str:
    if pending.player is None:
        return ": press m/o"
    if pending.choosing_serve_length:
        return f" by {pending.player}: SERVE, press s for SHORT or l for LONG (esc cancels)"
    return f" by {pending.player}: press a type key"


def detector_tables(video_path: Path, last_frame: int, hits_dir: Path = HITS_DIR, blank: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if blank:
        return pd.DataFrame(columns=HIT_COLUMNS), pd.DataFrame(columns=DETECTED_RALLY_COLUMNS)
    hits_file, rallies_source = output_paths(video_path, 0, last_frame, hits_dir)
    if not hits_file.is_file() or not rallies_source.is_file():
        raise FileNotFoundError(f"hit files not found: {hits_file}, {rallies_source} (run: python -m vision.hits {video_path} --start 0 --end {last_frame}, or label with --blank)")
    return pd.read_csv(hits_file), pd.read_csv(rallies_source)


def overlay_state(frame_index: int, tracks: tuple[pd.DataFrame, pd.DataFrame], hits: pd.DataFrame, rallies: pd.DataFrame, blank: bool = False):
    state = frame_state(frame_index, tracks[0], tracks[1], hits, rallies)
    return replace(state, rally_label="", shot_count=None) if blank else state


def segment_seconds_of(blank: bool, segment_seconds: float | None) -> float | None:
    if segment_seconds is not None:
        if segment_seconds <= 0:
            raise ValueError(f"--segment-seconds must be positive, got {segment_seconds:g}")
        return segment_seconds
    return DEFAULT_SEGMENT_SECONDS if blank else None


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
    span = state.view
    if span is None:
        put_text(bar, "no rally: press + to add one here", (8, 30), 0.6)
        return bar
    if state.segments:
        for rally in state.rallies:
            if rally.end_frame >= span.start_frame and rally.start_frame <= span.end_frame:
                left_x = timeline_x(max(rally.start_frame, span.start_frame), span, width)
                right_x = timeline_x(min(rally.end_frame, span.end_frame), span, width)
                cv2.rectangle(bar, (left_x, 2), (max(right_x, left_x + 1), height - 3), COLOR_RALLY_BGR, -1)
    cv2.line(bar, (0, height // 2), (width - 1, height // 2), (120, 120, 120), 1)
    for proposal in state.proposals:
        if span.start_frame <= proposal.frame <= span.end_frame:
            x = timeline_x(proposal.frame, span, width)
            cv2.line(bar, (x, height // 2 - 8), (x, height // 2 + 8), COLOR_PROPOSAL_BGR, 1)
    for label in state.labels:
        if span.start_frame <= label.frame <= span.end_frame:
            x = timeline_x(label.frame, span, width)
            cv2.line(bar, (x, 6), (x, height - 6), COLOR_LABEL_BGR, 2)
            put_text(bar, f"{label.player[0]}{'s?' if needs_serve_length(label) else label.type_group[0]}", (x + 2, height - 8), 0.4, COLOR_LABEL_BGR)
    if span.start_frame <= state.frame <= span.end_frame:
        x = timeline_x(state.frame, span, width)
        cv2.line(bar, (x, 0), (x, height - 1), COLOR_CURSOR_BGR, 2)
    return bar


def segment_text(state: LabelState) -> str:
    segment = state.view
    index = rally_containing(state.rallies, state.frame)
    rally = f"rally {state.rallies[index].rally} [{state.rallies[index].start_frame}-{state.rallies[index].end_frame}]" if index is not None else "no rally here"
    return f"segment {segment.rally} of {len(state.segments)} [{segment.start_frame}-{segment.end_frame}]  {rally} ({len(state.rallies)} rallies)"


def status_line(state: LabelState) -> str:
    span = state.view
    rally_text = segment_text(state) if state.segments else (f"rally {span.rally} of {len(state.rallies)} [{span.start_frame}-{span.end_frame}]" if span else "no rally")
    inside = [label for label in state.labels if span and span.start_frame <= label.frame <= span.end_frame]
    pending = ""
    if state.pending is not None:
        pending = f"   HIT at f{state.pending.frame}{pending_prompt(state.pending)}"
    notice = f"   {state.notice}" if state.notice else ""
    return f"f{state.frame}  {rally_text}  labels in {'segment' if state.segments else 'rally'}: {len(inside)}  total: {len(state.labels)}{unfinished_text(state)}{'  (unsaved)' if state.dirty else ''}{pending}{notice}"


def unfinished_text(state: LabelState) -> str:
    counts = unfinished_counts(state)
    parts = []
    if counts["serves_without_length"]:
        parts.append(f"{counts['serves_without_length']} serves need short/long")
    if counts["unchecked_rallies"] and state.segments:
        parts.append(f"{counts['unchecked_rallies']} rallies unchecked")
    return "  " + ", ".join(parts) if parts else ""


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
    blank: bool = False,
    segment_seconds: float | None = None,
) -> int:
    last_frame, fps = video_last_frame_and_fps(video_path)
    hits_table, rallies_table = detector_tables(video_path, last_frame, hits_dir, blank)
    labels = pd.read_csv(labels_file) if Path(labels_file).is_file() else None
    label_rallies = pd.read_csv(rallies_file) if Path(rallies_file).is_file() else None
    state = initial_state(hits_table, rallies_table, last_frame, fps, labels, label_rallies, segment_seconds, blank)
    tracks = load_tracks(video_path, 0, last_frame, players_dir)
    for line in KEY_HELP:
        print(line)
    capture = open_video(Path(video_path))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    loop_frames, loop_indices, loop_rally = [], [], None
    playing, speed, loop_started = False, 0.5, time.monotonic()
    left_cache: tuple[int, np.ndarray] | None = None
    try:
        while True:
            if state.view != loop_rally:
                loop_rally = state.view
                loop_frames, loop_indices = read_loop_frames(capture, loop_rally) if loop_rally else ([], [])
                loop_started = time.monotonic()
            if left_cache is None or left_cache[0] != state.frame:
                frame = read_frame(capture, state.frame)
                if frame is None:
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                if tracks is not None:
                    frame = draw_tracking_overlay(frame, overlay_state(state.frame, tracks, hits_table, rallies_table, blank))
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
    parser.add_argument("--blank", action="store_true", help="show nothing the detector inferred (no detected rallies, proposals, inferred serves or hit flash); no hits file needed; mark rallies with b / e")
    parser.add_argument("--segment-seconds", type=float, help=f"move through the video in back-to-back segments of this many seconds (default {DEFAULT_SEGMENT_SECONDS:g} with --blank); never saved")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    labels_file, rallies_file = label_paths(parsed.video)
    if parsed.labels is not None:
        labels_file = Path(parsed.labels)
        rallies_file = labels_file.with_name(labels_file.stem + "_rallies.csv")
    try:
        return label_video(
            parsed.video, labels_file, rallies_file, parsed.players_dir, parsed.hits_dir, window_name=f"label hits: {parsed.match_id}",
            blank=parsed.blank, segment_seconds=segment_seconds_of(parsed.blank, parsed.segment_seconds),
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
