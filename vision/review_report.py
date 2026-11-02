import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.shot_types import group_of_detail
from vision.label import SOURCE_MODEL, rejected_path_for
from vision.label_scores import rallies_file_for
from vision.players import video_fps_and_width

RETIMED_REACH_SECONDS = 1.0
SENSITIVITY_REACH_SECONDS = (1.5, 2.0)
GROUP_ORDER = ("serve", "net", "lift", "drive", "clear", "drop", "smash", "unknown")


def model_group(detail) -> str:
    return group_of_detail(str(detail)) or "unknown" if isinstance(detail, str) and detail else "unknown"


def share_text(part: int, whole: int) -> str:
    return f"{part}/{whole} = {part / whole:.1%}" if whole else "-"


def split_rows(kept: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from_model = kept["model_frame"].notna()
    unreviewed = kept[from_model & (kept["source"] == SOURCE_MODEL)]
    reviewed = kept[from_model & (kept["source"] != SOURCE_MODEL)]
    added = kept[~from_model]
    return reviewed, added, unreviewed


def retimed_rejections(rejected: pd.DataFrame, added: pd.DataFrame, reach_frames: int) -> pd.DataFrame:
    used = set()
    rows = []
    for rejection in rejected.sort_values("frame_num").itertuples(index=False):
        candidates = added[((added["frame_num"] - rejection.frame_num).abs() <= reach_frames) & ~added.index.isin(used)]
        if candidates.empty:
            continue
        nearest = candidates.iloc[(candidates["frame_num"] - rejection.frame_num).abs().argmin()]
        used.add(nearest.name)
        rows.append({"model_frame": int(rejection.frame_num), "added_frame": int(nearest["frame_num"]), "model_type": model_group(rejection.model_type_detail), "added_type": nearest["type_group"], "same_player": rejection.player == nearest["player"]})
    return pd.DataFrame(rows, columns=["model_frame", "added_frame", "model_type", "added_type", "same_player"])


def status_lines(reviewed: pd.DataFrame, added: pd.DataFrame, unreviewed: pd.DataFrame, rejected: pd.DataFrame) -> list[str]:
    judged = len(reviewed) + len(rejected)
    unchanged = reviewed[(reviewed["frame_num"] == reviewed["model_frame"]) & (reviewed["type_detail"] == reviewed["model_type_detail"]) & (reviewed["player"] == reviewed["model_player"])]
    return [
        "| | Count |",
        "|---|---|",
        f"| Model hits reviewed | {judged} ({len(reviewed)} kept, {len(rejected)} rejected) |",
        f"| Kept exactly as the model had them (frame, type and hitter) | {share_text(len(unchanged), judged)} |",
        f"| Kept with the type corrected | {int((reviewed['type_detail'] != reviewed['model_type_detail']).sum())} |",
        f"| Kept with the hitter corrected | {int((reviewed['player'] != reviewed['model_player']).sum())} |",
        f"| Kept and moved | {int((reviewed['frame_num'] != reviewed['model_frame']).sum())} |",
        f"| Rejected | {len(rejected)} |",
        f"| Added by the user (missed by the model) | {len(added)} |",
        f"| Left unreviewed (`source = model`), left out of everything here | {len(unreviewed)} |",
    ]


def confusion_lines(reviewed: pd.DataFrame) -> list[str]:
    model = reviewed["model_type_detail"].map(model_group)
    user = reviewed["type_group"].where(reviewed["type_group"].isin(GROUP_ORDER), "unknown")
    table = pd.crosstab(model, user).reindex(index=[g for g in GROUP_ORDER if g in set(model)], columns=[g for g in GROUP_ORDER if g in set(user)], fill_value=0)
    lines = ["| model \\ user | " + " | ".join(table.columns) + " | right |", "|---|" + "---|" * (len(table.columns) + 1)]
    for group, row in table.iterrows():
        right = int(row.get(group, 0)) if group != "unknown" else 0
        lines.append(f"| {group} | " + " | ".join(str(int(value)) for value in row) + f" | {share_text(right, int(row.sum())) if group != 'unknown' else '-'} |")
    return lines


def hitter_lines(reviewed: pd.DataFrame) -> list[str]:
    lines = ["| Model hitter | Kept hits | Hitter right |", "|---|---|---|"]
    for player in ("me", "opp"):
        rows = reviewed[reviewed["model_player"] == player]
        lines.append(f"| {player} | {len(rows)} | {share_text(int((rows['player'] == player).sum()), len(rows))} |")
    lines.append(f"| all | {len(reviewed)} | {share_text(int((reviewed['player'] == reviewed['model_player']).sum()), len(reviewed))} |")
    return lines


def lag_lines(reviewed: pd.DataFrame, fps: float) -> list[str]:
    moved = reviewed[reviewed["frame_num"] != reviewed["model_frame"]]
    lags = (moved["model_frame"] - moved["frame_num"]).astype(int)
    lines = [f"Lag = model frame minus the user's frame; positive means the model's hit came after the user's ({len(moved)} moved hits):", ""]
    lines += ["| Moved hits | n | Lags (frames) | Median | Mean |", "|---|---|---|---|---|"]
    groups = [("all", moved.index)] + [(f"hitter {player}", moved.index[moved["player"] == player]) for player in ("me", "opp")]
    groups += [(f"type {group}", moved.index[moved["type_group"] == group]) for group in GROUP_ORDER if (moved["type_group"] == group).any()]
    for name, index in groups:
        values = lags.loc[index]
        if len(values):
            lines.append(f"| {name} | {len(values)} | {', '.join(f'{value:+d}' for value in sorted(values))} | {np.median(values):+.1f} f = {np.median(values) / fps:+.2f} s | {values.mean():+.1f} f = {values.mean() / fps:+.2f} s |")
    return lines


def by_type_lines(rejected: pd.DataFrame, added: pd.DataFrame, retimed: pd.DataFrame) -> list[str]:
    rejected_type = rejected["model_type_detail"].map(model_group)
    lines = ["| Type | Rejected (model's type) | of which re-added nearby | False hits (rejected, not re-added) | Added (user's type) | of which re-timed | Missed (added, not re-timed) |", "|---|---|---|---|---|---|---|"]
    for group in GROUP_ORDER:
        rejected_count = int((rejected_type == group).sum())
        retimed_model = int((retimed["model_type"] == group).sum())
        added_count = int((added["type_group"] == group).sum())
        retimed_added = int((retimed["added_type"] == group).sum())
        if rejected_count or added_count:
            lines.append(f"| {group} | {rejected_count} | {retimed_model} | {rejected_count - retimed_model} | {added_count} | {retimed_added} | {added_count - retimed_added} |")
    lines.append(f"| all | {len(rejected)} | {len(retimed)} | {len(rejected) - len(retimed)} | {len(added)} | {len(retimed)} | {len(added) - len(retimed)} |")
    return lines


def report_lines(kept: pd.DataFrame, rejected: pd.DataFrame, rallies: pd.DataFrame, fps: float) -> list[str]:
    reviewed, added, unreviewed = split_rows(kept)
    reach = int(round(RETIMED_REACH_SECONDS * fps))
    retimed = retimed_rejections(rejected, added, reach)
    lines = ["## Status", "", *status_lines(reviewed, added, unreviewed, rejected), ""]
    lines += [f"Rallies: {len(rallies)}, {int(rallies['bounds_checked'].astype(str).str.lower().isin(['true', '1']).sum())} with checked bounds.", ""]
    lines += ["## Shot type on kept model hits", "", *confusion_lines(reviewed), ""]
    lines += ["## Hitter on kept model hits", "", *hitter_lines(reviewed), ""]
    lines += ["## Moved hits", "", *lag_lines(reviewed, fps), ""]
    lines += [f"## Rejected and added, with re-timed hits (a rejection with a hit the user added within {RETIMED_REACH_SECONDS:g} s, any rally, each added hit used once)", "", *by_type_lines(rejected, added, retimed), ""]
    lines += [f"Re-timed pairs: {len(retimed)}, {int(retimed['same_player'].sum())} with the same hitter; frame gaps (added minus model): " + ", ".join(f"{gap:+d}" for gap in sorted((retimed["added_frame"] - retimed["model_frame"]).astype(int))) if len(retimed) else "Re-timed pairs: 0", ""]
    lines += ["Sensitivity, re-timed pairs at other reaches (all types / model serves):", ""]
    for seconds in (RETIMED_REACH_SECONDS, *SENSITIVITY_REACH_SECONDS):
        pairs = retimed_rejections(rejected, added, int(round(seconds * fps)))
        lines.append(f"- within {seconds:g} s: {len(pairs)} / {int((pairs['model_type'] == 'serve').sum())}; false hits {len(rejected) - len(pairs)} of {len(rejected)} rejections")
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.review_report", description="How often the model was right on a game the user reviewed in vision.label_ui --review.")
    parser.add_argument("video", type=Path, help="the reviewed video (for its fps)")
    parser.add_argument("--labels", type=Path, required=True, help="the reviewed labels CSV; its _rallies.csv and _rejected.csv sit next to it")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        kept = pd.read_csv(parsed.labels)
        rejected_file = rejected_path_for(parsed.labels)
        rejected = pd.read_csv(rejected_file) if rejected_file.is_file() else kept.iloc[:0]
        rallies = pd.read_csv(rallies_file_for(parsed.labels))
        fps = video_fps_and_width(parsed.video)[0]
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for line in report_lines(kept, rejected, rallies, fps):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
