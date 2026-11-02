import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vision.hits import MATCH_TOLERANCE_SECONDS, match_hits, match_tolerances, tolerance_frames, tolerance_label
from vision.movement import PLAYERS, detected_hits_only

UNKNOWN_TYPE = "unknown"
MISSING_GROUP_NAMES = {"hitter": "unknown hitter", "type": "no type"}
OWN_PREDICTIONS_SQL = """
SELECT shots.frame_num, shots.type_pred
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
"""
OWN_PREDICTIONS_WITH_DETAIL_SQL = """
SELECT shots.frame_num, shots.type_pred, shots.type_detail_pred
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
"""
SHOTS_COLUMNS_SQL = "SELECT column_name FROM information_schema.columns WHERE table_name = 'shots'"


@dataclass(frozen=True)
class ScoredRun:
    name: str
    hits: pd.DataFrame
    predictions: pd.DataFrame
    rallies: pd.DataFrame | None = None


def read_labels(labels_file: Path) -> pd.DataFrame:
    table = pd.read_csv(labels_file)
    return pd.DataFrame(
        {
            "rally": table["rally"].astype(int),
            "frame": table["frame_num"].astype(int),
            "hitter": table["player"].where(table["player"].isin(PLAYERS)),
            "type": table["type_group"],
            "detail": table["type_detail"].where(table["type_detail"].astype(object).map(lambda value: isinstance(value, str) and value != "")) if "type_detail" in table.columns else np.nan,
            "source": table["source"] if "source" in table.columns else "hand",
        }
    ).sort_values("frame").reset_index(drop=True)


def labelled_rally_spans(label_rallies: pd.DataFrame, labels: pd.DataFrame) -> list[tuple[int, int]]:
    labelled = label_rallies[label_rallies["rally"].isin(labels["rally"])]
    return sorted((int(start), int(end)) for start, end in zip(labelled["start_frame"], labelled["end_frame"]))


def scored_rows(hits: pd.DataFrame, with_serves: bool) -> pd.DataFrame:
    return hits if with_serves else detected_hits_only(hits)


def label_outcomes(hits: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame, tolerance: int, with_serves: bool = False) -> pd.DataFrame:
    rows = scored_rows(hits, with_serves)
    matched = dict((labelled, detected) for detected, labelled in match_hits(rows["frame"].tolist(), labels["frame"].tolist(), tolerance))
    hitter_at = dict(zip(rows["frame"], rows["hitter"]))
    inferred_at = dict(zip(rows["frame"], rows["score"].isna())) if "score" in rows.columns else {}
    prediction_at = dict(zip(predictions["frame_num"], predictions["type_pred"]))
    detail_at = dict(zip(predictions["frame_num"], predictions["type_detail_pred"])) if "type_detail_pred" in predictions.columns else {}
    labels = labels if "detail" in labels.columns else labels.assign(detail=np.nan)
    detected = [matched.get(int(frame)) for frame in labels["frame"]]
    return labels.assign(
        detected_frame=pd.array(detected, dtype="Int64"),
        detected_hitter=[hitter_at.get(frame) if frame is not None else None for frame in detected],
        inferred_serve=[bool(inferred_at.get(frame, False)) if frame is not None else False for frame in detected],
        type_pred=[prediction_at.get(frame) if frame is not None else None for frame in detected],
        detail_pred=[detail_at.get(frame) if frame is not None else None for frame in detected],
    )


def agreement_counts(outcomes: pd.DataFrame) -> dict:
    found = outcomes[outcomes["detected_frame"].notna()]
    hitter_judged = found[found["hitter"].notna()]
    type_judged = found[found["type_pred"].notna() & found["type"].notna() & (found["type"] != UNKNOWN_TYPE)]
    detail_judged = found[found["detail_pred"].notna() & found["detail"].notna() & (found["detail"] != UNKNOWN_TYPE)] if {"detail", "detail_pred"} <= set(found.columns) else found.iloc[:0]
    return {
        "labelled": len(outcomes),
        "matched": len(found),
        "hitter_right": int((hitter_judged["hitter"] == hitter_judged["detected_hitter"]).sum()),
        "hitter_judged": len(hitter_judged),
        "type_right": int((type_judged["type"] == type_judged["type_pred"]).sum()),
        "type_judged": len(type_judged),
        "detail_right": int((detail_judged["detail"] == detail_judged["detail_pred"]).sum()),
        "detail_judged": len(detail_judged),
    }


def group_names(table: pd.DataFrame, column: str) -> pd.Series:
    values = table[column].astype(object)
    return values.where(values.notna(), MISSING_GROUP_NAMES.get(column, "(none)")).map(str)


def counts_by(outcomes: pd.DataFrame, column: str) -> dict[str, dict]:
    return {name: agreement_counts(group) for name, group in outcomes.groupby(group_names(outcomes, column))}


def inside_spans(frame: int, spans: list[tuple[int, int]], tolerance: int) -> bool:
    return any(start - tolerance <= frame <= end + tolerance for start, end in spans)


def precision_in_labelled_rallies(hits: pd.DataFrame, outcomes: pd.DataFrame, spans: list[tuple[int, int]], tolerance: int, with_serves: bool = False) -> tuple[int, int]:
    rows = scored_rows(hits, with_serves)
    inside = {int(frame) for frame in rows["frame"] if inside_spans(int(frame), spans, tolerance)}
    matched = {int(frame) for frame in outcomes["detected_frame"].dropna()}
    return len(inside & matched), len(inside)


def share(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} = {numerator / denominator:.1%}" if denominator else f"{numerator}/0"


def nearest_offset(frame: int, hits: pd.DataFrame) -> int | None:
    frames = detected_hits_only(hits)["frame"]
    if frames.empty:
        return None
    offsets = frames.astype(int) - frame
    return int(offsets.iloc[offsets.abs().argmin()])


def read_predictions(database_path: Path, match_id: str) -> pd.DataFrame:
    import duckdb

    if not Path(database_path).is_file():
        raise FileNotFoundError(f"database not found: {database_path}")
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        has_detail = "type_detail_pred" in {row[0] for row in connection.execute(SHOTS_COLUMNS_SQL).fetchall()}
        return connection.execute(OWN_PREDICTIONS_WITH_DETAIL_SQL if has_detail else OWN_PREDICTIONS_SQL, [match_id]).df()
    finally:
        connection.close()


def headline_table(runs: list[ScoredRun], labels: pd.DataFrame, spans: list[tuple[int, int]], windows: list[tuple[float, int]], fps: float) -> list[str]:
    lines = [
        "| Model | Tolerance | Rows scored | Recall | Hitter right | Shot type agrees | Precision in labelled rallies (lower bound) |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for with_serves in (False, True):
            for window, tolerance in windows:
                outcomes = label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves)
                counts = agreement_counts(outcomes)
                precise, inside = precision_in_labelled_rallies(run.hits, outcomes, spans, tolerance, with_serves)
                rows_name = "detected + inferred serves" if with_serves else "detected hits"
                lines.append(
                    f"| {run.name} | {tolerance_label(window, fps)} | {rows_name} | {share(counts['matched'], counts['labelled'])} | {share(counts['hitter_right'], counts['hitter_judged'])} "
                    f"| {share(counts['type_right'], counts['type_judged'])} | {share(precise, inside)} |"
                )
    return lines


def breakdown_table(runs: list[ScoredRun], labels: pd.DataFrame, column: str, tolerance: int, with_serves: bool) -> list[str]:
    per_run = {run.name: counts_by(label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves), column) for run in runs}
    names = group_names(labels, column)
    values = sorted(names.unique(), key=lambda value: (-int((names == value).sum()), value))
    header = f"| {column} | Labelled | " + " | ".join(f"{run.name} recall | {run.name} hitter | {run.name} type" for run in runs) + " |"
    lines = [header, "|" + "---|" * (2 + 3 * len(runs))]
    for value in values:
        cells = []
        for run in runs:
            counts = per_run[run.name][value]
            cells.extend([share(counts["matched"], counts["labelled"]), share(counts["hitter_right"], counts["hitter_judged"]), share(counts["type_right"], counts["type_judged"])])
        lines.append(f"| {value} | {int((names == value).sum())} | " + " | ".join(cells) + " |")
    return lines


def type_confusion_table(run: ScoredRun, labels: pd.DataFrame, tolerance: int, with_serves: bool) -> list[str]:
    found = label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves).dropna(subset=["detected_frame"])
    found = found.assign(type_pred=found["type_pred"].fillna("(none)"))
    table = pd.crosstab(found["type"], found["type_pred"])
    lines = ["| labelled \\ predicted | " + " | ".join(table.columns) + " |", "|" + "---|" * (1 + len(table.columns))]
    lines.extend(f"| {group} | " + " | ".join(str(int(count)) for count in row) + " |" for group, row in table.iterrows())
    return lines


def missed_table(runs: list[ScoredRun], labels: pd.DataFrame, tolerance: int, with_serves: bool) -> list[str]:
    outcomes = {run.name: label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves) for run in runs}
    missed_by_any = sorted({int(frame) for result in outcomes.values() for frame in result.loc[result["detected_frame"].isna(), "frame"]})
    lines = ["| Frame | Rally | Player | Type | " + " | ".join(f"{run.name}" for run in runs) + " |", "|" + "---|" * (4 + len(runs))]
    by_frame = labels.set_index("frame")
    for frame in missed_by_any:
        label = by_frame.loc[frame]
        cells = []
        for run in runs:
            found = outcomes[run.name].set_index("frame").loc[frame, "detected_frame"]
            offset = nearest_offset(frame, run.hits)
            cells.append("found" if pd.notna(found) else f"missed (nearest {offset:+d})" if offset is not None else "missed")
        lines.append(f"| {frame} | {int(label.rally)} | {label.hitter} | {label.type} | " + " | ".join(cells) + " |")
    return lines


def serve_table(runs: list[ScoredRun], labels: pd.DataFrame, tolerance: int) -> list[str]:
    serves = labels[labels["type"] == "serve"].reset_index(drop=True)
    lines = ["| Model | Labelled serves | Found by an inferred serve row | Found by a detected hit | Missed | Shot type agrees |", "|---|---|---|---|---|---|"]
    for run in runs:
        outcomes = label_outcomes(run.hits, serves, run.predictions, tolerance, with_serves=True)
        found = outcomes[outcomes["detected_frame"].notna()]
        by_row = int(found["inferred_serve"].sum())
        counts = agreement_counts(outcomes)
        lines.append(
            f"| {run.name} | {len(serves)} | {by_row} | {len(found) - by_row} | {len(serves) - len(found)} | {share(counts['type_right'], counts['type_judged'])} |"
        )
    return lines


def non_serve_table(runs: list[ScoredRun], labels: pd.DataFrame, windows: list[tuple[float, int]], fps: float) -> list[str]:
    lines = ["| Model | Tolerance | Labelled non-serves matched | Shot type agrees | Matched non-serves typed serve |", "|---|---|---|---|---|"]
    for run in runs:
        for window, tolerance in windows:
            outcomes = label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves=True)
            non_serves = outcomes[outcomes["type"] != "serve"]
            counts = agreement_counts(non_serves)
            typed_serve = int((non_serves["detected_frame"].notna() & (non_serves["type_pred"] == "serve")).sum())
            lines.append(f"| {run.name} | {tolerance_label(window, fps)} | {share(counts['matched'], counts['labelled'])} | {share(counts['type_right'], counts['type_judged'])} | {typed_serve} |")
    return lines


def correct_shots(run: ScoredRun, labels: pd.DataFrame, spans: list[tuple[int, int]], tolerance: int) -> tuple[int, int]:
    outcomes = label_outcomes(run.hits, labels, run.predictions, tolerance, with_serves=True)
    matched_inside, inside = precision_in_labelled_rallies(run.hits, outcomes, spans, tolerance, with_serves=True)
    return agreement_counts(outcomes)["type_right"], len(labels) + inside - matched_inside


def correct_shot_table(runs: list[ScoredRun], labels: pd.DataFrame, spans: list[tuple[int, int]], windows: list[tuple[float, int]], fps: float) -> list[str]:
    lines = ["| Model | Tolerance | Correct shots | Labelled hits | False hits inside labelled rallies |", "|---|---|---|---|---|"]
    for run in runs:
        for window, tolerance in windows:
            right, total = correct_shots(run, labels, spans, tolerance)
            lines.append(f"| {run.name} | {tolerance_label(window, fps)} | {share(right, total)} | {len(labels)} | {total - len(labels)} |")
    return lines


def source_lines(runs: list[ScoredRun], labels: pd.DataFrame, spans: list[tuple[int, int]], windows: list[tuple[float, int]], fps: float) -> list[str]:
    if labels["source"].nunique() < 2:
        return ["", f"## By label source", f"every label is {labels['source'].iloc[0]}; no per-source split"]
    lines = []
    for source, source_labels in labels.groupby("source"):
        lines += ["", f"## Recall of {source} labels", *headline_table(runs, source_labels.reset_index(drop=True), spans, windows, fps)]
    return lines


def detail_lines(runs: list[ScoredRun], labels: pd.DataFrame, tolerance: int) -> list[str]:
    if "detail" not in labels.columns or labels["detail"].notna().sum() == 0:
        return ["no label has a type_detail; detail agreement not scored"]
    lines = ["| Model | Labels with a detail | Matched with a predicted detail | Detail agrees |", "|---|---|---|---|"]
    for run in runs:
        if "type_detail_pred" not in run.predictions.columns:
            lines.append(f"| {run.name} | {int(labels['detail'].notna().sum())} | no detail predictions | |")
            continue
        counts = agreement_counts(label_outcomes(run.hits, labels, run.predictions, tolerance, True))
        lines.append(f"| {run.name} | {int(labels['detail'].notna().sum())} | {counts['detail_judged']} | {share(counts['detail_right'], counts['detail_judged'])} |")
    return lines


def rallies_file_of_hits(hits_file: Path) -> Path | None:
    hits_file = Path(hits_file)
    if not hits_file.name.startswith("hits_"):
        return None
    return hits_file.with_name("rallies_" + hits_file.name[len("hits_") :])


def rallies_of_hits(hits_file: Path) -> pd.DataFrame | None:
    rallies_file = rallies_file_of_hits(hits_file)
    return pd.read_csv(rallies_file) if rallies_file is not None and rallies_file.is_file() else None


def rally_bound_lines(runs: list[ScoredRun], spans: list[tuple[int, int]], fps: float) -> list[str]:
    from vision.in_play import compare_spans

    lines = [
        "| Model | Labelled rallies | Detected rallies | Matched | Merged | Split | Missed | Extra | Start error, median abs (s) | End error, median abs (s) | Start error, mean (s) | End error, mean (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if run.rallies is None:
            lines.append(f"| {run.name} | {len(spans)} | no rallies file | | | | | | | | | |")
            continue
        comparison = compare_spans(spans, [(int(start), int(end)) for start, end in zip(run.rallies["start_frame"], run.rallies["end_frame"])], fps)
        starts, ends = np.array(comparison.start_errors), np.array(comparison.end_errors)
        seconds = lambda values, stat: f"{stat(values):+.2f}" if stat is np.mean and len(values) else (f"{stat(np.abs(values)):.2f}" if len(values) else "")
        lines.append(
            f"| {run.name} | {comparison.labelled} | {comparison.found} | {comparison.matched} | {comparison.merged} | {comparison.split} | {comparison.missed} | {comparison.extra} "
            f"| {seconds(starts, np.median)} | {seconds(ends, np.median)} | {seconds(starts, np.mean)} | {seconds(ends, np.mean)} |"
        )
    return lines


def report_lines(
    runs: list[ScoredRun],
    labels: pd.DataFrame,
    spans: list[tuple[int, int]],
    fps: float,
    seconds: tuple[float, ...] = MATCH_TOLERANCE_SECONDS,
    breakdown_seconds: float = MATCH_TOLERANCE_SECONDS[-1],
) -> list[str]:
    windows = match_tolerances(fps, seconds)
    breakdown_tolerance = tolerance_frames(breakdown_seconds, fps)
    breakdown = tolerance_label(breakdown_seconds, fps)
    lines = [f"labels: {len(labels)} in {labels['rally'].nunique()} rallies; sources: " + ", ".join(f"{source} {count}" for source, count in labels["source"].value_counts().items())]
    lines += ["", "## Headline", *headline_table(runs, labels, spans, windows, fps)]
    lines += ["", "## Correct shots: matched and typed right ÷ (labelled hits + false hits inside labelled rallies), detected + inferred serves", *correct_shot_table(runs, labels, spans, windows, fps)]
    lines += source_lines(runs, labels, spans, windows, fps)
    lines += ["", f"## Labelled serves, {breakdown}", *serve_table(runs, labels, breakdown_tolerance)]
    lines += ["", "## Labelled serves left out, detected + inferred serves", *non_serve_table(runs, labels, windows, fps)]
    for column in ("type", "hitter"):
        lines += ["", f"## By labelled {column}, {breakdown}, detected + inferred serves", *breakdown_table(runs, labels, column, breakdown_tolerance, True)]
    for run in runs:
        lines += ["", f"## Shot types on matched hits, {run.name}, {breakdown}, detected + inferred serves", *type_confusion_table(run, labels, breakdown_tolerance, True)]
    lines += ["", f"## Shot detail on matched hits, {breakdown}, detected + inferred serves", *detail_lines(runs, labels, breakdown_tolerance)]
    lines += ["", "## Rally bounds against the labelled rallies", *rally_bound_lines(runs, spans, fps)]
    lines += ["", f"## Labelled hits missed by either model, {breakdown}, detected + inferred serves", *missed_table(runs, labels, breakdown_tolerance, True)]
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.label_scores",
        description="Score hits files against partial hand labels: recall, hitter and shot type on the labelled hits, precision only inside the labelled rallies.",
    )
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV of vision.label or vision.label_ui (rally, frame_num, player, type_group, source)")
    parser.add_argument("--rallies", type=Path, help="its rallies CSV (default: <labels stem>_rallies.csv next to it)")
    parser.add_argument("--run", nargs=3, action="append", default=[], metavar=("NAME", "HITS_CSV", "DATABASE"), help="a model's hits file and the shots database holding its predicted rows; repeatable")
    parser.add_argument("--match-id", required=True, help="the own rows' match id in each DATABASE")
    parser.add_argument("--tolerance", type=float, action="append", default=[], help="matching window in seconds for the headline, turned into frames at --fps (default 0.25 and 0.5); repeatable")
    parser.add_argument("--fps", type=float, required=True, help="the video's frame rate: turns the matching windows into frames, and times the comparison of each run's rallies file (rallies_*.csv next to its hits file) with the labelled rallies")
    parser.add_argument("--breakdown-tolerance", type=float, default=MATCH_TOLERANCE_SECONDS[-1], help="window in seconds of the breakdowns, the serve table and the missed list (default 0.5)")
    parsed = parser.parse_args(arguments)
    if not parsed.run:
        parser.error("give at least one --run NAME HITS_CSV DATABASE")
    return parsed


def rallies_file_for(labels_file: Path) -> Path:
    return labels_file.with_name(f"{labels_file.stem}_rallies.csv")


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        labels = read_labels(parsed.labels)
        spans = labelled_rally_spans(pd.read_csv(parsed.rallies or rallies_file_for(parsed.labels)), labels)
        runs = [
            ScoredRun(name, pd.read_csv(hits_file), read_predictions(Path(database), parsed.match_id), rallies_of_hits(hits_file))
            for name, hits_file, database in parsed.run
        ]
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for line in report_lines(runs, labels, spans, parsed.fps, tuple(parsed.tolerance) or MATCH_TOLERANCE_SECONDS, parsed.breakdown_tolerance):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
