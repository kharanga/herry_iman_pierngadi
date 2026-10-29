import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

from analytics.court import ZONE_NAMES

DATABASE_PATH = Path("data/scout.duckdb")
KNOWN_SWAPPED_RALLIES = frozenset(
    {
        ("CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal", 2, 6),
        ("Ratchanok_Intanon_Pusarla_V._Sindhu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals", 2, 1),
    }
)
FAMILIES = ("zone_mix", "response", "serve_placement", "winning_shot", "error_shot", "error_reason")
GROUP_PHRASES = {
    "serve": ("a serve", "serves"),
    "clear": ("a clear", "clears"),
    "drop": ("a drop", "drops"),
    "smash": ("a smash", "smashes"),
    "net": ("a net shot", "net shots"),
    "lift": ("a lift", "lifts"),
    "drive": ("a drive", "drives"),
}
DEPTH_PHRASES = {"front": "front court", "mid": "mid court", "rear": "back court"}
SIDE_PHRASES = {"forehand": "forehand side", "middle": "middle", "backhand": "backhand side"}
SERVE_DEPTH_PHRASES = {"front": "short", "mid": "mid-depth", "rear": "long"}
ERROR_REASON_PHRASES = {
    "out": "hit out",
    "net": "into the net",
    "not over net": "not over the net",
    "misjudged landing": "misjudged landings",
    "foul": "faults",
}
GROUP_ORDER = ("serve", "clear", "drop", "smash", "net", "lift", "drive")
CONDITION_ORDER_BY_FAMILY = {"zone_mix": ZONE_NAMES, "response": GROUP_ORDER}

TENDENCY_SHOTS_SQL = """
WITH shots_with_group AS (
    SELECT
        shots.source,
        shots.match_id,
        shots.set_no,
        shots.rally,
        shots.ball_round,
        shots.frame_num,
        shots.player,
        shots.type_en,
        shots.hit_zone,
        shots.landing_zone,
        shots.getpoint_player,
        shots.lose_reason,
        CASE
            WHEN shots.type_group IS NOT NULL AND shots.type_group <> 'unknown' THEN shots.type_group
            ELSE shots.type_pred
        END AS shot_group,
        CASE
            WHEN shots.type_group IS NOT NULL AND shots.type_group <> 'unknown' THEN 'label'
            WHEN shots.type_pred IS NOT NULL THEN 'prediction'
        END AS shot_group_source
    FROM shots
)
SELECT
    shots_with_group.*,
    LAG(shots_with_group.player) OVER rally_order AS previous_player,
    LAG(shots_with_group.shot_group) OVER rally_order AS previous_shot_group,
    LAG(shots_with_group.shot_group_source) OVER rally_order AS previous_shot_group_source,
    LEAD(shots_with_group.ball_round) OVER rally_order IS NULL AS is_last_shot,
    COUNT(*) OVER rally_window AS rally_shot_count
FROM shots_with_group
WINDOW
    rally_order AS (
        PARTITION BY shots_with_group.source, shots_with_group.match_id, shots_with_group.set_no, shots_with_group.rally
        ORDER BY shots_with_group.ball_round, shots_with_group.frame_num
    ),
    rally_window AS (
        PARTITION BY shots_with_group.source, shots_with_group.match_id, shots_with_group.set_no, shots_with_group.rally
    )
ORDER BY shots_with_group.source, shots_with_group.match_id, shots_with_group.set_no, shots_with_group.rally, shots_with_group.ball_round
"""

MATCH_PLAYERS_SQL = """
SELECT shots.source, shots.match_id, shots.player, COUNT(*) AS shot_count
FROM shots
GROUP BY shots.source, shots.match_id, shots.player
ORDER BY shots.source, shots.match_id, shots.player
"""


class UnknownMatchPlayerError(ValueError):
    def __init__(self, match_id: str, player: str):
        super().__init__(f"no shots for player {player!r} in match {match_id!r}")
        self.match_id = match_id
        self.player = player


def load_tendency_shots(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(TENDENCY_SHOTS_SQL).df()


def is_unreliable_player_rally(shot_rows: pd.DataFrame) -> pd.Series:
    is_one_shot_unknown_rally = (shot_rows["rally_shot_count"] == 1) & (shot_rows["type_en"] == "unknown")
    is_known_swapped_rally = pd.Series(
        [(match_id, set_no, rally) in KNOWN_SWAPPED_RALLIES for match_id, set_no, rally in zip(shot_rows["match_id"], shot_rows["set_no"], shot_rows["rally"])],
        index=shot_rows.index,
    )
    return is_one_shot_unknown_rally | is_known_swapped_rally


def reliable_shots(shot_rows: pd.DataFrame) -> pd.DataFrame:
    return shot_rows[~is_unreliable_player_rally(shot_rows)]


def combined_group_source(first_source: pd.Series, second_source: pd.Series) -> pd.Series:
    return first_source.where(second_source != "prediction", "prediction")


def family_observations(shot_rows: pd.DataFrame) -> pd.DataFrame:
    has_group = shot_rows["shot_group"].notna()
    hitter_won = shot_rows["getpoint_player"] == shot_rows["player"]
    hitter_lost = shot_rows["getpoint_player"].notna() & ~hitter_won
    is_serve = shot_rows["shot_group"] == "serve"
    is_reply_to_opponent = (
        shot_rows["previous_player"].notna()
        & (shot_rows["previous_player"] != shot_rows["player"])
        & shot_rows["previous_shot_group"].notna()
    )
    families = [
        ("zone_mix", has_group & ~is_serve & shot_rows["hit_zone"].notna(), shot_rows["hit_zone"], shot_rows["shot_group"], shot_rows["shot_group_source"]),
        (
            "response",
            has_group & is_reply_to_opponent,
            shot_rows["previous_shot_group"],
            shot_rows["shot_group"],
            combined_group_source(shot_rows["shot_group_source"], shot_rows["previous_shot_group_source"]),
        ),
        ("serve_placement", is_serve & shot_rows["landing_zone"].notna(), "serve", shot_rows["landing_zone"], shot_rows["shot_group_source"]),
        ("winning_shot", has_group & shot_rows["is_last_shot"] & hitter_won, "winner", shot_rows["shot_group"], shot_rows["shot_group_source"]),
        ("error_shot", has_group & shot_rows["is_last_shot"] & hitter_lost, "error", shot_rows["shot_group"], shot_rows["shot_group_source"]),
        ("error_reason", shot_rows["is_last_shot"] & hitter_lost & shot_rows["lose_reason"].notna(), "error", shot_rows["lose_reason"], "label"),
    ]
    observations = []
    for family, mask, condition, outcome, group_source in families:
        family_rows = shot_rows[mask]
        observations.append(
            pd.DataFrame(
                {
                    "family": family,
                    "source": family_rows["source"],
                    "match_id": family_rows["match_id"],
                    "set_no": family_rows["set_no"],
                    "player": family_rows["player"],
                    "condition": condition[mask] if isinstance(condition, pd.Series) else condition,
                    "outcome": outcome[mask],
                    "group_source": group_source[mask] if isinstance(group_source, pd.Series) else group_source,
                }
            )
        )
    return pd.concat(observations, ignore_index=True)


def label_source_name(prediction_count: int, n: int) -> str:
    if prediction_count == 0:
        return "labels"
    if prediction_count == n:
        return "predictions"
    return "mixed"


def zone_phrase(zone: str) -> str:
    depth, side = zone.split("_")
    return f"{DEPTH_PHRASES[depth]}, {SIDE_PHRASES[side]}"


def serve_landing_phrase(landing_zone: str) -> str:
    if landing_zone == "not_over_net":
        return "into the net"
    depth, side = landing_zone.split("_")
    side_text = "down the middle" if side == "middle" else f"to the receiver's {side}"
    return f"{SERVE_DEPTH_PHRASES[depth]} {side_text}"


def tendency_text(family: str, condition: str, outcome: str, k: int, n: int) -> str:
    evidence = f"{k} of {n} times"
    if family == "zone_mix":
        return f"From the {zone_phrase(condition)}: {GROUP_PHRASES[outcome][1]} {evidence}."
    if family == "response":
        return f"After {GROUP_PHRASES[condition][0]}: {GROUP_PHRASES[outcome][1]} {evidence}."
    if family == "serve_placement":
        return f"Serves {serve_landing_phrase(outcome)} {evidence}."
    if family == "winning_shot":
        return f"Wins rallies with {GROUP_PHRASES[outcome][1]}: {k} of {n} winners."
    if family == "error_shot":
        return f"Errors come from {GROUP_PHRASES[outcome][1]}: {k} of {n} errors."
    if family == "error_reason":
        return f"Errors {ERROR_REASON_PHRASES.get(outcome, outcome)}: {k} of {n} errors."
    raise ValueError(f"unknown tendency family: {family!r}")


def situation_counts(player_observations: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "condition", "outcome"]
    is_prediction = (player_observations["group_source"] == "prediction").rename("is_prediction")
    outcome_counts = (
        pd.concat([player_observations[keys], is_prediction], axis=1).groupby(keys).agg(k=("is_prediction", "size")).reset_index()
    )
    condition_counts = player_observations.groupby(["family", "condition"]).agg(
        n=("outcome", "size"), condition_prediction_count=("group_source", lambda sources: int((sources == "prediction").sum()))
    )
    return outcome_counts.merge(condition_counts.reset_index(), on=["family", "condition"])


def position_in(order: tuple[str, ...], value: str) -> int:
    return order.index(value) if value in order else len(order)


def plain_order_key(tendency: dict) -> tuple:
    family_order = CONDITION_ORDER_BY_FAMILY.get(tendency["family"], ())
    return (
        position_in(FAMILIES, tendency["family"]),
        position_in(family_order, tendency["condition"]),
        tendency["condition"],
        -tendency["k"],
        tendency["outcome"],
    )


def tendencies_from_observations(
    observations: pd.DataFrame,
    match_id: str,
    player: str,
    sets: tuple[int, ...] | None = None,
) -> list[dict]:
    is_player = (observations["match_id"] == match_id) & (observations["player"] == player)
    if sets is not None:
        is_player &= observations["set_no"].isin(sets)
    player_observations = observations[is_player]
    if player_observations.empty:
        return []
    played_sets = sorted(int(set_no) for set_no in player_observations["set_no"].unique())
    tendencies = []
    for situation in situation_counts(player_observations).itertuples(index=False):
        k, n = int(situation.k), int(situation.n)
        tendencies.append(
            {
                "id": f"{situation.family}:{situation.condition}:{situation.outcome}",
                "family": situation.family,
                "match_id": match_id,
                "player": player,
                "sets": played_sets,
                "condition": situation.condition,
                "outcome": situation.outcome,
                "n": n,
                "k": k,
                "rate": k / n,
                "label_source": label_source_name(int(situation.condition_prediction_count), n),
                "text": tendency_text(situation.family, situation.condition, situation.outcome, k, n),
            }
        )
    return sorted(tendencies, key=plain_order_key)


def match_players(connection: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    return [(source, match_id, player) for source, match_id, player, _ in connection.execute(MATCH_PLAYERS_SQL).fetchall()]


def all_observations(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return family_observations(reliable_shots(load_tendency_shots(connection)))
    finally:
        connection.close()


def tendencies_for(
    match_id: str,
    player: str,
    database_path: Path = DATABASE_PATH,
    sets: tuple[int, ...] | None = None,
) -> list[dict]:
    observations = all_observations(database_path)
    if not ((observations["match_id"] == match_id) & (observations["player"] == player)).any():
        raise UnknownMatchPlayerError(match_id, player)
    return tendencies_from_observations(observations, match_id, player, sets)


def parse_sets(sets_text: str | None) -> tuple[int, ...] | None:
    if not sets_text:
        return None
    return tuple(int(set_no) for set_no in sets_text.split(","))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m analytics.tendencies <match_id> <player> [sets, e.g. 1,2] [database_path]", file=sys.stderr)
        sys.exit(2)
    match_id, player = sys.argv[1], sys.argv[2]
    sets = parse_sets(sys.argv[3]) if len(sys.argv) > 3 else None
    database_path = Path(sys.argv[4]) if len(sys.argv) > 4 else DATABASE_PATH
    try:
        print(json.dumps(tendencies_for(match_id, player, database_path, sets), indent=2))
    except UnknownMatchPlayerError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
