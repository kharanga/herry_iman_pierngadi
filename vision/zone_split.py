import argparse
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import analytics.court as court
from analytics.court import ZONE_NAMES, hit_zone, landing_zone
from vision.label import reviewed_rows_only

OLD_DEPTH_EDGES_M = (1.98, 3.50)
ZONE_ORDER = ZONE_NAMES + ("not_over_net",)
OWN_ROWS_SQL = """
SELECT shots.rowid AS row_id, shots.frame_num, shots.player, shots.hit_x, shots.hit_y, shots.landing_x, shots.landing_y
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
"""
SET_LABEL_SQL = """
UPDATE shots SET type_group = ?, type_detail = ?
WHERE shots.rowid = ?
"""
SET_ZONES_SQL = """
UPDATE shots SET hit_zone = ?, landing_zone = ?
WHERE shots.rowid = ?
"""


@contextmanager
def depth_edges(edges_m: tuple[float, float]):
    current = court.DEPTH_EDGES_M
    court.DEPTH_EDGES_M = tuple(edges_m)
    try:
        yield
    finally:
        court.DEPTH_EDGES_M = current


def point_of(x, y) -> tuple[float, float] | None:
    return None if pd.isna(x) or pd.isna(y) else (float(x), float(y))


def text_or_none(value) -> str | None:
    return value if isinstance(value, str) else None


def zones_of(rows: pd.DataFrame, edges_m: tuple[float, float]) -> pd.DataFrame:
    with depth_edges(edges_m):
        hit = [hit_zone(point_of(x, y)) for x, y in zip(rows["hit_x"], rows["hit_y"])]
        landing = [landing_zone(point_of(x, y)) for x, y in zip(rows["landing_x"], rows["landing_y"])]
    return rows.assign(hit_zone=hit, landing_zone=landing)


def zone_count_lines(old: pd.DataFrame, new: pd.DataFrame, column: str) -> list[str]:
    lines = [f"| {column} | " + " | ".join(f"{player} old | {player} new" for player in ("me", "opp")) + " |", "|---|" + "---|---|" * 2]
    for zone in ZONE_ORDER:
        cells = []
        for player in ("me", "opp"):
            cells += [str(int((old.loc[old["player"] == player, column] == zone).sum())), str(int((new.loc[new["player"] == player, column] == zone).sum()))]
        if any(cell != "0" for cell in cells):
            lines.append(f"| {zone} | " + " | ".join(cells) + " |")
    return lines


def changed_counts(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, tuple[int, int]]:
    counts = {}
    for column in ("hit_zone", "landing_zone"):
        judged = old[column].notna() & new[column].notna()
        counts[column] = (int((old.loc[judged, column] != new.loc[judged, column]).sum()), int(judged.sum()))
    return counts


def labelled_copy(database: Path, output: Path, match_id: str, labels: pd.DataFrame, edges_m: tuple[float, float]) -> pd.DataFrame:
    import duckdb

    shutil.copyfile(database, output)
    connection = duckdb.connect(str(output))
    try:
        rows = connection.execute(OWN_ROWS_SQL, [match_id]).df()
        rows = zones_of(rows, edges_m).merge(labels[["frame_num", "type_group", "type_detail"]], on="frame_num", how="left")
        connection.execute("BEGIN TRANSACTION")
        connection.executemany(SET_ZONES_SQL, [[text_or_none(row.hit_zone), text_or_none(row.landing_zone), int(row.row_id)] for row in rows.itertuples(index=False)])
        connection.executemany(SET_LABEL_SQL, [[row.type_group, text_or_none(row.type_detail), int(row.row_id)] for row in rows.itertuples(index=False) if isinstance(row.type_group, str)])
        connection.execute("COMMIT")
    finally:
        connection.close()
    return rows


def habits(database: Path, match_id: str, player: str) -> dict[str, dict]:
    from analytics.tendencies import all_observations, tendencies_from_observations

    observations = all_observations(database)
    return {tendency["id"]: tendency for tendency in tendencies_from_observations(observations, match_id, player)}


LISTED_FAMILIES = ("zone_mix", "serve_placement")


def count_text(tendency: dict | None) -> str:
    return f"{tendency['k']} of {tendency['n']}" if tendency else "-"


def changed_habit_ids(old: dict[str, dict], new: dict[str, dict]) -> list[str]:
    return sorted((habit_id for habit_id in set(old) | set(new) if count_text(old.get(habit_id)) != count_text(new.get(habit_id))), key=lambda key: (key.split(":")[0], key))


def habit_change_lines(old: dict[str, dict], new: dict[str, dict]) -> list[str]:
    changed = changed_habit_ids(old, new)
    lines = ["| Habit | Old bands | New bands |", "|---|---|---|"]
    for habit_id in changed:
        family = habit_id.split(":")[0]
        if family in LISTED_FAMILIES:
            lines.append(f"| {family}: {habit_id.split(':', 1)[1]} | {count_text(old.get(habit_id))} | {count_text(new.get(habit_id))} |")
    for family in sorted({habit_id.split(":")[0] for habit_id in set(old) | set(new)} - set(LISTED_FAMILIES)):
        total = len({habit_id for habit_id in set(old) | set(new) if habit_id.startswith(f"{family}:")})
        lines.append(f"| {family} (not listed) | {sum(habit_id.startswith(f'{family}:') for habit_id in changed)} of {total} habits change | |")
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.zone_split", description="Zones and habits of a labelled video under the old depth bands and the current ones.")
    parser.add_argument("--labels", type=Path, required=True, help="hand labels CSV of the video (read only)")
    parser.add_argument("--database", type=Path, required=True, help="a shots database holding the video's own rows built from its labelled hits (vision.features); not changed")
    parser.add_argument("--match-id", required=True, help="the own rows' match id")
    parser.add_argument("--out", type=Path, required=True, help="folder for the two labelled copies (old_bands.duckdb, new_bands.duckdb)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        labels = reviewed_rows_only(pd.read_csv(parsed.labels), parsed.labels)
        parsed.out.mkdir(parents=True, exist_ok=True)
        old_rows = labelled_copy(parsed.database, parsed.out / "old_bands.duckdb", parsed.match_id, labels, OLD_DEPTH_EDGES_M)
        new_rows = labelled_copy(parsed.database, parsed.out / "new_bands.duckdb", parsed.match_id, labels, court.DEPTH_EDGES_M)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{parsed.match_id}: {len(new_rows)} own rows from the labelled hits; old bands {OLD_DEPTH_EDGES_M}, new bands {court.DEPTH_EDGES_M}")
    for column in ("hit_zone", "landing_zone"):
        print()
        for line in zone_count_lines(old_rows, new_rows, column):
            print(line)
    print()
    for column, (changed, judged) in changed_counts(old_rows, new_rows).items():
        print(f"{column}: {changed} of {judged} shots change zone")
    for player in ("me", "opp"):
        print(f"\n### Habits that change, {player}\n")
        for line in habit_change_lines(habits(parsed.out / "old_bands.duckdb", parsed.match_id, player), habits(parsed.out / "new_bands.duckdb", parsed.match_id, player)):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
