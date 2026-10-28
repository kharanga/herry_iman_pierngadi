# `analytics/validate_tendencies.py`

Checks whether tendencies found in set 1 still hold in set 2 of the same match, across all
ShuttleSet players, and compares that with baselines. Results are in `docs/results/tendencies.md`.

## Command

```
.venv/bin/python -m analytics.validate_tendencies [database_path] [validation_path]
```

Defaults: `data/scout.duckdb`, `data/checks/tendencies/set_one_to_set_two.json`. Prints one line per
selection (`k of n follow in set 2`, and its matched chance) and writes the full summary as JSON. Takes about 10 s.

## Definitions

- **Candidate**: a (family, condition, outcome) for one player in set 1 with `n >= MIN_SAMPLE_SIZE` (5),
  scored exactly as in `analytics.tendencies` (reference = pros in all other matches).
- **Follows**: in set 2 the player is in the same situation at least `MIN_FOLLOW_SAMPLE_SIZE` (5) times
  (`is_evaluable`), and their set-2 rate for the outcome is **above the reference rate**. In other words,
  the "more than pros do" claim is still true in the next set.
- **set_two_lift**: set-2 rate minus the reference rate (evaluable candidates only).
- **Matched chance** of an evaluable candidate: among all ShuttleSet players from *other* matches who were in the
  same situation (family, condition) at least `MIN_FOLLOW_SAMPLE_SIZE` times in their set 2, the share whose set-2
  rate for the same outcome is above the candidate's reference rate. A player who never chose the outcome counts
  with rate 0. A selection's matched chance is the mean over its evaluable candidates (candidates with no such
  other player are skipped). It is the follow rate expected if set 1 told us nothing about this player.

## Selections compared

| Selection | Meaning |
|---|---|
| tendencies (lower bound above pros) | what `analytics.tendencies` reports (`score > 0`) |
| top 3 per player (what the report shows) | the 3 best-scored set-1 tendencies per player |
| naive: raw set-1 rate above pros | rate > reference, no confidence bound |
| naive only: raw rate above pros, lower bound not | candidates the naive rule flags and the score rejects |
| not flagged: raw rate at or below pros | candidates at or below the pros in set 1. Not a fair chance level: regression to the mean pulls them down (see the matched chance) |

## Functions
- `validate_set_one_to_set_two(database_path, validation_path)`: runs everything and returns the summary
  `dict`: `players`, `min_sample_size`, `min_follow_sample_size`, `per_player_set_counts`, `selections`,
  `tendencies_by_family`, `top_bullets_by_family` (the last three made of `follow_summary` dicts). Errors: as `analytics.tendencies.all_observations`.
- `validation_rows(observations)`: one row per set-1 candidate per player with a set 2, with flags
  `is_tendency`, `is_naive_tendency`, `is_top_bullet` and the `follow_check` fields.
- `set_one_candidates(observations, match_id, player)`: scored set-1 candidates with `n >= 5`.
- `set_two_counts(observations, match_id, player)`: `(outcome_counts, condition_counts)` dicts for set 2.
- `follow_check(family, condition, outcome, reference_rate, set_two)`: `set_two_n`, `set_two_k`,
  `is_evaluable`, `follows`, `set_two_lift`.
- `follow_summary(rows)`: `candidates`, `evaluable`, `follows`, `follow_rate`, `matched_chance`,
  `lift_over_matched_chance`, `mean_set_two_lift` (`None` when nothing is evaluable; `matched_chance` is only
  used when `rows` has that column).
- `with_matched_chance(rows, observations)`: `rows` plus a `matched_chance` column (`None` for non-evaluable rows).
- `set_two_rates_by_situation(observations)`: ShuttleSet set-2 data for the matched chance: which players had each
  situation at least 5 times (`players_by_situation`), and their outcome counts (`outcome_counts`).
- `matched_chance(family, condition, outcome, reference_rate, match_id, set_two_rates)`: one candidate's chance level,
  or `None` if no other player qualifies.
- `per_player_set_counts(observations, shot_rows)`: min / median / max over ShuttleSet player-sets in sets 1 and 2 of
  all shots (after the excluded rallies) and of rally-ending shots (`winning_shot` + `error_shot` observations).
- `shot_count_summary(counts)`: `min`, `median`, `max`, `player_sets` of a count series.

```python
>>> import pandas as pd
>>> from analytics.validate_tendencies import follow_check, follow_summary
>>> set_two = ({("zone_mix", "rear_backhand", "drop"): 4}, {("zone_mix", "rear_backhand"): 7})
>>> follow_check("zone_mix", "rear_backhand", "drop", 0.31, set_two)
{'set_two_n': 7, 'set_two_k': 4, 'is_evaluable': True, 'follows': True, 'set_two_lift': 0.2614285714285714}
>>> follow_check("zone_mix", "front_middle", "net", 0.5, set_two)["is_evaluable"]
False
>>> rows = pd.DataFrame({"is_evaluable": [True, True, False], "follows": [True, False, False], "set_two_lift": [0.2, -0.1, None],
...                      "matched_chance": [0.4, 0.2, None]})
>>> {key: round(value, 3) for key, value in follow_summary(rows).items()}
{'candidates': 3, 'evaluable': 2, 'follows': 1, 'follow_rate': 0.5, 'matched_chance': 0.3, 'lift_over_matched_chance': 0.2, 'mean_set_two_lift': 0.05}
>>> from analytics.validate_tendencies import matched_chance
>>> set_two_rates = {
...     "players_by_situation": {("zone_mix", "rear_backhand"): [("match_1", "A", 10), ("match_2", "B", 8), ("match_3", "A", 5)]},
...     "outcome_counts": {("zone_mix", "rear_backhand", "match_1", "A", "drop"): 5, ("zone_mix", "rear_backhand", "match_2", "B", "drop"): 1},
... }
>>> matched_chance("zone_mix", "rear_backhand", "drop", 0.3, "match_3", set_two_rates)
0.5
>>> matched_chance("zone_mix", "front_middle", "net", 0.3, "match_3", set_two_rates) is None
True

```
