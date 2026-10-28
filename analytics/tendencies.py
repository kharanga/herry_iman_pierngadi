import json
import math
import sys
from pathlib import Path

import duckdb
import pandas as pd

DATABASE_PATH = Path("data/scout.duckdb")
MIN_SAMPLE_SIZE = 5
CONFIDENCE_Z = 1.645
MIN_SCORE = 0.0
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


def reference_rates(observations: pd.DataFrame, excluded_match_id: str) -> pd.DataFrame:
    pro_observations = observations[(observations["source"] == "shuttleset") & (observations["match_id"] != excluded_match_id)]
    outcome_counts = pro_observations.groupby(["family", "condition", "outcome"]).size().rename("reference_k")
    condition_counts = pro_observations.groupby(["family", "condition"]).size().rename("reference_n")
    references = outcome_counts.reset_index().merge(condition_counts.reset_index(), on=["family", "condition"])
    references["reference_rate"] = references["reference_k"] / references["reference_n"]
    return references[["family", "condition", "outcome", "reference_rate", "reference_n"]]


def wilson_lower_bound(k: int, n: int, z: float = CONFIDENCE_Z) -> float:
    if n == 0:
        return 0.0
    rate = k / n
    centre = rate + z * z / (2 * n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def tendency_score(k: int, n: int, reference_rate: float, z: float = CONFIDENCE_Z) -> float:
    return wilson_lower_bound(k, n, z) - reference_rate


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


def tendency_text(family: str, condition: str, outcome: str, k: int, n: int, reference_rate: float) -> str:
    evidence = f"{k} of {n} times"
    pros = f"(pros: {reference_rate:.0%})"
    if family == "zone_mix":
        return f"From the {zone_phrase(condition)}: {GROUP_PHRASES[outcome][1]} {evidence} {pros}."
    if family == "response":
        return f"After {GROUP_PHRASES[condition][0]}: {GROUP_PHRASES[outcome][1]} {evidence} {pros}."
    if family == "serve_placement":
        return f"Serves {serve_landing_phrase(outcome)} {evidence} {pros}."
    if family == "winning_shot":
        return f"Wins rallies with {GROUP_PHRASES[outcome][1]}: {k} of {n} winners {pros}."
    if family == "error_shot":
        return f"Errors come from {GROUP_PHRASES[outcome][1]}: {k} of {n} errors {pros}."
    if family == "error_reason":
        return f"Errors {ERROR_REASON_PHRASES.get(outcome, outcome)}: {k} of {n} errors {pros}."
    raise ValueError(f"unknown tendency family: {family!r}")


def candidate_tendencies(player_observations: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "condition", "outcome"]
    is_prediction = (player_observations["group_source"] == "prediction").rename("is_prediction")
    outcome_counts = (
        pd.concat([player_observations[keys], is_prediction], axis=1)
        .groupby(keys)
        .agg(k=("is_prediction", "size"), prediction_count=("is_prediction", "sum"))
        .reset_index()
    )
    condition_counts = player_observations.groupby(["family", "condition"]).agg(
        n=("outcome", "size"), condition_prediction_count=("group_source", lambda sources: int((sources == "prediction").sum()))
    )
    reference_condition_sizes = references[["family", "condition", "reference_n"]].drop_duplicates()
    candidates = (
        outcome_counts.merge(condition_counts.reset_index(), on=["family", "condition"])
        .merge(reference_condition_sizes, on=["family", "condition"], how="inner")
        .merge(references.drop(columns="reference_n"), on=keys, how="left")
    )
    candidates["reference_rate"] = candidates["reference_rate"].fillna(0.0)
    return candidates


def tendencies_from_observations(
    observations: pd.DataFrame,
    match_id: str,
    player: str,
    sets: tuple[int, ...] | None = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    is_player = (observations["match_id"] == match_id) & (observations["player"] == player)
    if sets is not None:
        is_player &= observations["set_no"].isin(sets)
    player_observations = observations[is_player]
    if player_observations.empty:
        return []
    candidates = candidate_tendencies(player_observations, reference_rates(observations, match_id))
    tendencies = []
    for candidate in candidates.itertuples(index=False):
        k, n = int(candidate.k), int(candidate.n)
        score = tendency_score(k, n, float(candidate.reference_rate))
        if n < min_sample_size or score <= min_score:
            continue
        tendencies.append(
            {
                "id": f"{candidate.family}:{candidate.condition}:{candidate.outcome}",
                "family": candidate.family,
                "match_id": match_id,
                "player": player,
                "sets": sorted(int(set_no) for set_no in player_observations["set_no"].unique()),
                "condition": candidate.condition,
                "outcome": candidate.outcome,
                "n": n,
                "k": k,
                "rate": k / n,
                "reference_rate": float(candidate.reference_rate),
                "score": score,
                "label_source": label_source_name(int(candidate.condition_prediction_count), n),
                "text": tendency_text(candidate.family, candidate.condition, candidate.outcome, k, n, float(candidate.reference_rate)),
            }
        )
    return sorted(tendencies, key=lambda tendency: (-tendency["score"], -tendency["n"], tendency["id"]))


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
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> list[dict]:
    observations = all_observations(database_path)
    if not ((observations["match_id"] == match_id) & (observations["player"] == player)).any():
        raise UnknownMatchPlayerError(match_id, player)
    return tendencies_from_observations(observations, match_id, player, sets, min_sample_size)


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
