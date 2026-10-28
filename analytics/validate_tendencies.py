import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

from analytics.tendencies import (
    DATABASE_PATH,
    FAMILIES,
    MIN_SAMPLE_SIZE,
    candidate_tendencies,
    family_observations,
    load_tendency_shots,
    reliable_shots,
    reference_rates,
    tendencies_from_observations,
    tendency_score,
)

VALIDATION_PATH = Path("data/checks/tendencies/set_one_to_set_two.json")
MIN_FOLLOW_SAMPLE_SIZE = 5
REPORT_BULLET_COUNT = 3


def set_two_counts(observations: pd.DataFrame, match_id: str, player: str) -> tuple[dict, dict]:
    set_two = observations[(observations["match_id"] == match_id) & (observations["player"] == player) & (observations["set_no"] == 2)]
    outcome_counts = set_two.groupby(["family", "condition", "outcome"]).size().to_dict()
    condition_counts = set_two.groupby(["family", "condition"]).size().to_dict()
    return outcome_counts, condition_counts


def follow_check(family: str, condition: str, outcome: str, reference_rate: float, set_two: tuple[dict, dict]) -> dict:
    outcome_counts, condition_counts = set_two
    n = int(condition_counts.get((family, condition), 0))
    k = int(outcome_counts.get((family, condition, outcome), 0))
    is_evaluable = n >= MIN_FOLLOW_SAMPLE_SIZE
    return {
        "set_two_n": n,
        "set_two_k": k,
        "is_evaluable": is_evaluable,
        "follows": is_evaluable and k / n > reference_rate,
        "set_two_lift": k / n - reference_rate if is_evaluable else None,
    }


def set_one_candidates(observations: pd.DataFrame, match_id: str, player: str) -> pd.DataFrame:
    set_one = observations[(observations["match_id"] == match_id) & (observations["player"] == player) & (observations["set_no"] == 1)]
    candidates = candidate_tendencies(set_one, reference_rates(observations, match_id))
    candidates = candidates[candidates["n"] >= MIN_SAMPLE_SIZE].copy()
    candidates["rate"] = candidates["k"] / candidates["n"]
    candidates["score"] = [tendency_score(int(k), int(n), float(reference)) for k, n, reference in zip(candidates["k"], candidates["n"], candidates["reference_rate"])]
    return candidates


def validation_rows(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shuttleset_players = observations[observations["source"] == "shuttleset"][["match_id", "player"]].drop_duplicates()
    for match_id, player in shuttleset_players.itertuples(index=False):
        set_two = set_two_counts(observations, match_id, player)
        if not set_two[1]:
            continue
        top_ids = [tendency["id"] for tendency in tendencies_from_observations(observations, match_id, player, sets=(1,))[:REPORT_BULLET_COUNT]]
        for candidate in set_one_candidates(observations, match_id, player).itertuples(index=False):
            candidate_id = f"{candidate.family}:{candidate.condition}:{candidate.outcome}"
            rows.append(
                {
                    "match_id": match_id,
                    "player": player,
                    "family": candidate.family,
                    "id": candidate_id,
                    "set_one_n": int(candidate.n),
                    "set_one_rate": float(candidate.rate),
                    "reference_rate": float(candidate.reference_rate),
                    "score": float(candidate.score),
                    "is_tendency": float(candidate.score) > 0,
                    "is_naive_tendency": float(candidate.rate) > float(candidate.reference_rate),
                    "is_top_bullet": candidate_id in top_ids,
                    **follow_check(candidate.family, candidate.condition, candidate.outcome, float(candidate.reference_rate), set_two),
                }
            )
    return pd.DataFrame(rows)


def set_two_rates_by_situation(observations: pd.DataFrame) -> dict:
    set_two = observations[(observations["source"] == "shuttleset") & (observations["set_no"] == 2)]
    situation_sizes = set_two.groupby(["family", "condition", "match_id", "player"]).size().rename("n").reset_index()
    situation_sizes = situation_sizes[situation_sizes["n"] >= MIN_FOLLOW_SAMPLE_SIZE]
    outcome_counts = set_two.groupby(["family", "condition", "match_id", "player", "outcome"]).size().to_dict()
    rates_by_situation = {}
    for situation in situation_sizes.itertuples(index=False):
        rates_by_situation.setdefault((situation.family, situation.condition), []).append(
            (situation.match_id, situation.player, int(situation.n))
        )
    return {"players_by_situation": rates_by_situation, "outcome_counts": outcome_counts}


def matched_chance(
    family: str, condition: str, outcome: str, reference_rate: float, match_id: str, set_two_rates: dict
) -> float | None:
    other_players = [
        (other_match_id, other_player, n)
        for other_match_id, other_player, n in set_two_rates["players_by_situation"].get((family, condition), [])
        if other_match_id != match_id
    ]
    if not other_players:
        return None
    outcome_counts = set_two_rates["outcome_counts"]
    beats_pros = [
        outcome_counts.get((family, condition, other_match_id, other_player, outcome), 0) / n > reference_rate
        for other_match_id, other_player, n in other_players
    ]
    return sum(beats_pros) / len(beats_pros)


def follow_summary(rows: pd.DataFrame) -> dict:
    evaluable = rows[rows["is_evaluable"]]
    matched = evaluable["matched_chance"].dropna() if "matched_chance" in evaluable else pd.Series(dtype=float)
    follow_rate = float(evaluable["follows"].mean()) if len(evaluable) else None
    matched_chance_rate = float(matched.mean()) if len(matched) else None
    return {
        "candidates": int(len(rows)),
        "evaluable": int(len(evaluable)),
        "follows": int(evaluable["follows"].sum()),
        "follow_rate": follow_rate,
        "matched_chance": matched_chance_rate,
        "lift_over_matched_chance": None if follow_rate is None or matched_chance_rate is None else follow_rate - matched_chance_rate,
        "mean_set_two_lift": float(evaluable["set_two_lift"].mean()) if len(evaluable) else None,
    }


def with_matched_chance(rows: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    set_two_rates = set_two_rates_by_situation(observations)
    situations = zip(rows["id"], rows["reference_rate"], rows["match_id"], rows["is_evaluable"])
    matched = []
    for candidate_id, reference_rate, match_id, is_evaluable in situations:
        family, condition, outcome = candidate_id.split(":", 2)
        matched.append(matched_chance(family, condition, outcome, reference_rate, match_id, set_two_rates) if is_evaluable else None)
    return rows.assign(matched_chance=matched)


def shot_count_summary(counts: pd.Series) -> dict:
    return {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max()), "player_sets": int(len(counts))}


def per_player_set_counts(observations: pd.DataFrame, shot_rows: pd.DataFrame) -> dict:
    validated_sets = shot_rows[(shot_rows["source"] == "shuttleset") & shot_rows["set_no"].isin([1, 2])]
    rally_endings = observations[
        (observations["source"] == "shuttleset")
        & observations["set_no"].isin([1, 2])
        & observations["family"].isin(["winning_shot", "error_shot"])
    ]
    return {
        "shots": shot_count_summary(validated_sets.groupby(["match_id", "set_no", "player"]).size()),
        "rally_ending_shots": shot_count_summary(rally_endings.groupby(["match_id", "set_no", "player"]).size()),
    }


def validate_set_one_to_set_two(database_path: Path = DATABASE_PATH, validation_path: Path = VALIDATION_PATH) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        shot_rows = reliable_shots(load_tendency_shots(connection))
    finally:
        connection.close()
    observations = family_observations(shot_rows)
    rows = with_matched_chance(validation_rows(observations), observations)
    selections = {
        "tendencies (lower bound above pros)": rows["is_tendency"],
        "top 3 per player (what the report shows)": rows["is_top_bullet"],
        "naive: raw set-1 rate above pros": rows["is_naive_tendency"],
        "naive only: raw rate above pros, lower bound not": rows["is_naive_tendency"] & ~rows["is_tendency"],
        "not flagged: raw rate at or below pros": ~rows["is_naive_tendency"],
    }
    summary = {
        "players": int(rows[["match_id", "player"]].drop_duplicates().shape[0]),
        "min_sample_size": MIN_SAMPLE_SIZE,
        "min_follow_sample_size": MIN_FOLLOW_SAMPLE_SIZE,
        "per_player_set_counts": per_player_set_counts(observations, shot_rows),
        "selections": {name: follow_summary(rows[mask]) for name, mask in selections.items()},
        "tendencies_by_family": {family: follow_summary(rows[rows["is_tendency"] & (rows["family"] == family)]) for family in FAMILIES},
        "top_bullets_by_family": {family: follow_summary(rows[rows["is_top_bullet"] & (rows["family"] == family)]) for family in FAMILIES},
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    validation_path = Path(sys.argv[2]) if len(sys.argv) > 2 else VALIDATION_PATH
    summary = validate_set_one_to_set_two(database_path, validation_path)
    for name, selection in summary["selections"].items():
        print(
            f"{name}: {selection['follows']} of {selection['evaluable']} follow in set 2 ({selection['follow_rate']:.1%}), "
            f"matched chance {selection['matched_chance']:.1%}"
        )
    print(f"saved {validation_path}")
