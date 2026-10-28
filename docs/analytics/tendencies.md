# `analytics/tendencies.py`

Turns the `shots` table into ranked tendencies for one player in one match: what they do
more often than pro players do in the same situation, with the evidence ("8 of 10 times").
The phone report shows the top 3 per player.

## Command

```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets] [database_path]
```

- `player`: as in `shots.player`, e.g. `A`/`B` (ShuttleSet) or `me`/`opp` (own footage). Nothing is hard-coded.
- `sets` (optional): comma-separated set numbers, e.g. `1` or `1,2`. Default: all sets.
- Prints a JSON list, best first. Exit code 1 with `error: no shots for player ...` for an unknown match or player.

## Tendency families

| Family | Situation (`condition`) | What they did (`outcome`) | Shots counted |
|---|---|---|---|
| `zone_mix` | hit zone (9 zones) | shot group | every shot except serves |
| `response` | the opponent's previous shot group | shot group | every reply to an opponent's shot |
| `serve_placement` | `serve` | landing zone (or `not_over_net`) | serves |
| `winning_shot` | `winner` | shot group | the last shot of a rally the hitter won |
| `error_shot` | `error` | shot group | the last shot of a rally the hitter lost |
| `error_reason` | `error` | `lose_reason` (`out`, `net`, ...) | the same, where a reason is recorded |

A rally's last shot belongs to the winner if `getpoint_player` equals the hitter, otherwise it is an error.

**Shot group used.** The true `type_group` when there is one (ShuttleSet), otherwise the classifier's
`type_pred` (my footage, and ShuttleSet shots typed `unknown`). Each tendency says which in
`label_source`: `labels` (all true labels), `predictions` (all predicted), or `mixed`. A `response`
counts as predicted if either shot's group was predicted.

**Shots left out.** Per-player stats skip rallies where `player` is unreliable (see
`docs/results/schema.md`, "Known data issues"): one-shot rallies whose only shot is typed `unknown`,
and the 2 known rallies with A and B swapped (`KNOWN_SWAPPED_RALLIES`).

## Score and ranking

For a player, situation and outcome: `n` = times the player was in the situation, `k` = times they chose
that outcome, `rate = k / n`. The **reference rate** is how often pros chose that outcome in the same
situation: the share over all ShuttleSet matches *except the one being analysed*.

`score = wilson_lower_bound(k, n) - reference_rate`

The Wilson lower bound (one-sided 95%, z = 1.645) is the lowest rate that is still plausible given
`k` of `n`. Small samples get a low bound, so "3 of 3" (bound 0.53) can't outrank "14 of 18" (0.59).
A tendency is a candidate with `n >= MIN_SAMPLE_SIZE` (5) and `score > MIN_SCORE` (0): even the
pessimistic rate beats what pros do. Situations pros never met are skipped, since there is nothing to
compare with. Tendencies are sorted by score (then larger `n`, then `id`).

Comparing with the pros, not with the player's own overall mix, keeps out tendencies that every player
has ("plays net shots from the front court").

## Output: one tendency

| Key | Type | Meaning |
|---|---|---|
| `id` | str | `family:condition:outcome`, stable across runs |
| `family`, `condition`, `outcome` | str | see the table above |
| `match_id`, `player` | str | |
| `sets` | list[int] | sets the counts come from |
| `n`, `k` | int | evidence: `k` of `n` |
| `rate` | float | `k / n` |
| `reference_rate` | float | pros in the same situation |
| `score` | float | ranking score, > 0 |
| `label_source` | str | `labels`, `predictions` or `mixed` |
| `text` | str | plain English, e.g. `From the back court, backhand side: drops 8 of 10 times (pros: 20%).` |

## Constants
`MIN_SAMPLE_SIZE = 5`, `CONFIDENCE_Z = 1.645`, `MIN_SCORE = 0.0`, `KNOWN_SWAPPED_RALLIES`, `FAMILIES`,
the phrase tables used by `tendency_text` (`GROUP_PHRASES`, `DEPTH_PHRASES`, `SIDE_PHRASES`,
`SERVE_DEPTH_PHRASES`, `ERROR_REASON_PHRASES`), and the SQL (`TENDENCY_SHOTS_SQL`, `MATCH_PLAYERS_SQL`).

## Functions

### `tendencies_for(match_id, player, database_path=DATABASE_PATH, sets=None, min_sample_size=5)`
Loads the shots (read-only), builds observations and returns the ranked list of tendency `dict`s.
Errors: `UnknownMatchPlayerError` (a `ValueError`, with `match_id` and `player`) if the player has no
usable shots in that match; `duckdb.IOException` / `CatalogException` for a missing database or table.

### `tendencies_from_observations(observations, match_id, player, sets=None, min_sample_size=5, min_score=0.0)`
The same from an observations table (see `all_observations`), for computing many players without
reloading. Returns `[]` if the player has no observations.

### `all_observations(database_path=DATABASE_PATH)`
`family_observations(reliable_shots(load_tendency_shots(connection)))` for the whole table:
one row per (family, shot) with `family, source, match_id, set_no, player, condition, outcome, group_source`.

### Building blocks
- `load_tendency_shots(connection)`: `TENDENCY_SHOTS_SQL`, every shot with its group, group source,
  previous shot's player and group, `is_last_shot` and `rally_shot_count`.
- `is_unreliable_player_rally(shot_rows)`, `reliable_shots(shot_rows)`: the exclusion rule above.
- `family_observations(shot_rows)`: the 6 families as observation rows.
- `reference_rates(observations, excluded_match_id)`: pro rates per family, condition and outcome, without that match.
- `candidate_tendencies(player_observations, references)`: `k`, `n`, prediction counts and reference per candidate.
- `wilson_lower_bound(k, n, z=1.645)`, `tendency_score(k, n, reference_rate, z=1.645)`.
- `label_source_name(prediction_count, n)`.
- `tendency_text(family, condition, outcome, k, n, reference_rate)`, `zone_phrase(zone)`, `serve_landing_phrase(landing_zone)`.
  `tendency_text` raises `ValueError` for an unknown family.
- `match_players(connection)`: `[(source, match_id, player), ...]` for every player in `shots`.
- `parse_sets(sets_text)`: `"1,2"` -> `(1, 2)`, empty -> `None`.

```python
>>> from analytics.tendencies import *
>>> round(wilson_lower_bound(3, 3), 3), round(wilson_lower_bound(14, 18), 3), wilson_lower_bound(0, 0)
(0.526, 0.587, 0.0)
>>> round(tendency_score(14, 18, 0.3), 3)
0.287
>>> tendency_text("zone_mix", "rear_backhand", "drop", 9, 14, 0.31)
'From the back court, backhand side: drops 9 of 14 times (pros: 31%).'
>>> tendency_text("response", "smash", "lift", 8, 12, 0.4)
'After a smash: lifts 8 of 12 times (pros: 40%).'
>>> tendency_text("serve_placement", "serve", "rear_backhand", 6, 10, 0.12)
"Serves long to the receiver's backhand 6 of 10 times (pros: 12%)."
>>> tendency_text("error_reason", "error", "net", 6, 9, 0.35)
'Errors into the net: 6 of 9 errors (pros: 35%).'
>>> label_source_name(0, 10), label_source_name(10, 10), label_source_name(3, 10)
('labels', 'predictions', 'mixed')
>>> parse_sets("1,2"), parse_sets("")
((1, 2), None)

```

## Example on a small synthetic table

Two "pro" matches where the player at the back usually clears (drops 1 rally in 5), and one of my matches
where the opponent drops 8 times in 10 from the back court, backhand side:

```python
>>> import shutil, tempfile
>>> from pathlib import Path
>>> import duckdb
>>> from analytics.build_shots import SCHEMA_PATH
>>> temp_dir = Path(tempfile.mkdtemp())
>>> database_path = temp_dir / "scout.duckdb"
>>> connection = duckdb.connect(str(database_path))
>>> _ = connection.execute(SCHEMA_PATH.read_text())
>>> def rally_rows(source, match_id, rally, hitter, receiver, hitter_shot, winner):
...     serve = (source, match_id, 1, rally, 1, rally * 100, receiver, "short service", "serve", "front_middle", "front_middle", None, None)
...     reply = (source, match_id, 1, rally, 2, rally * 100 + 30, hitter, hitter_shot, hitter_shot, "rear_backhand", "front_forehand",
...              winner, "out" if winner == receiver else "opponent winner")
...     return [serve, reply]
>>> rows = []
>>> for match_id in ("pro_match_1", "pro_match_2"):
...     for rally in range(1, 21):
...         rows += rally_rows("shuttleset", match_id, rally, "B", "A", "drop" if rally % 5 == 0 else "clear", "A")
>>> for rally in range(1, 11):
...     rows += rally_rows("own", "club_2026_10_20", rally, "opp", "me", "drop" if rally <= 8 else "clear", "opp")
>>> _ = connection.executemany(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, player, type_en, type_group, hit_zone, "
...     "landing_zone, getpoint_player, lose_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
>>> connection.close()
>>> tendencies = tendencies_for("club_2026_10_20", "opp", database_path)
>>> [(tendency["id"], tendency["k"], tendency["n"], tendency["reference_rate"], round(tendency["score"], 3)) for tendency in tendencies]
[('response:serve:drop', 8, 10, 0.2, 0.341), ('zone_mix:rear_backhand:drop', 8, 10, 0.2, 0.341)]
>>> tendencies[1]["text"], tendencies[1]["label_source"], tendencies[1]["sets"]
('From the back court, backhand side: drops 8 of 10 times (pros: 20%).', 'labels', [1])
>>> try:
...     tendencies_for("club_2026_10_20", "nobody", database_path)
... except UnknownMatchPlayerError as error:
...     print(error)
no shots for player 'nobody' in match 'club_2026_10_20'
>>> shutil.rmtree(temp_dir)

```

The opponent also won all 10 rallies with their last shot, 8 of them drops. There is no `winning_shot`
tendency because the pro matches have no data for that situation: there, every rally ends with an error
by the player at the back. With nothing to compare against, the candidate is skipped rather than compared
with 0%.
