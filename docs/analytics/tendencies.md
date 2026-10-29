# `analytics/tendencies.py`

Counts one player's habits in one match from their own shots: for every situation and every choice
they made in it, one entry with a plain count ("After a net shot: lifts 7 of 9 times."). No comparison
with other players, no score, no ranking and no minimum sample size: every choice is listed, even
"1 of 9". The phone report shows all of them.

## Command

```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets] [database_path]
```

- `player`: as in `shots.player`, e.g. `A`/`B` (ShuttleSet) or `me`/`opp` (own footage). Nothing is hard-coded.
- `sets` (optional): comma-separated set numbers, e.g. `1` or `1,2`. Default: all sets.
- Prints a JSON list in plain order (below). Exit code 1 with `error: no shots for player ...` for an unknown match or player.

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

## Counting and order

For a player, situation (`family` + `condition`) and choice (`outcome`): `n` = times the player was in the
situation, `k` = times they made that choice, `rate = k / n`. The `k` of a situation's choices add up to its `n`.

Entries are ordered **family, then situation, then most times first**. Families follow `FAMILIES`; zones follow
`analytics.court.ZONE_NAMES` (front, mid, rear; forehand, middle, backhand); `response` situations follow
`GROUP_ORDER` (serve, clear, drop, smash, net, lift, drive); ties on `k` go by choice name. This is a reading order,
not a ranking.

## Output: one entry

| Key | Type | Meaning |
|---|---|---|
| `id` | str | `family:condition:outcome`, stable across runs |
| `family`, `condition`, `outcome` | str | see the table above |
| `match_id`, `player` | str | |
| `sets` | list[int] | sets the counts come from |
| `n`, `k` | int | `k` of `n` |
| `rate` | float | `k / n` |
| `label_source` | str | `labels`, `predictions` or `mixed`, over the situation's `n` shots |
| `text` | str | plain English, e.g. `From the back court, backhand side: drops 8 of 10 times.` |

## Constants
`KNOWN_SWAPPED_RALLIES`, `FAMILIES`, `GROUP_ORDER`, `CONDITION_ORDER_BY_FAMILY`, the phrase tables used by
`tendency_text` (`GROUP_PHRASES`, `DEPTH_PHRASES`, `SIDE_PHRASES`, `SERVE_DEPTH_PHRASES`, `ERROR_REASON_PHRASES`),
and the SQL (`TENDENCY_SHOTS_SQL`, `MATCH_PLAYERS_SQL`).

## Functions

### `tendencies_for(match_id, player, database_path=DATABASE_PATH, sets=None)`
Loads the shots (read-only), builds observations and returns every entry for the player, in plain order.
Errors: `UnknownMatchPlayerError` (a `ValueError`, with `match_id` and `player`) if the player has no
usable shots in that match; `duckdb.IOException` / `CatalogException` for a missing database or table.

### `tendencies_from_observations(observations, match_id, player, sets=None)`
The same from an observations table (see `all_observations`), for computing many players without
reloading. Only that player's own observations are counted. Returns `[]` if the player has none.

### `all_observations(database_path=DATABASE_PATH)`
`family_observations(reliable_shots(load_tendency_shots(connection)))` for the whole table:
one row per (family, shot) with `family, source, match_id, set_no, player, condition, outcome, group_source`.

### Building blocks
- `load_tendency_shots(connection)`: `TENDENCY_SHOTS_SQL`, every shot with its group, group source,
  previous shot's player and group, `is_last_shot` and `rally_shot_count`.
- `is_unreliable_player_rally(shot_rows)`, `reliable_shots(shot_rows)`: the exclusion rule above.
- `family_observations(shot_rows)`: the 6 families as observation rows.
- `situation_counts(player_observations)`: `k` per choice, `n` and the predicted-shot count per situation.
- `plain_order_key(tendency)`, `position_in(order, value)`: the order above.
- `label_source_name(prediction_count, n)`.
- `tendency_text(family, condition, outcome, k, n)`, `zone_phrase(zone)`, `serve_landing_phrase(landing_zone)`.
  `tendency_text` raises `ValueError` for an unknown family.
- `match_players(connection)`: `[(source, match_id, player), ...]` for every player in `shots`.
- `parse_sets(sets_text)`: `"1,2"` -> `(1, 2)`, empty -> `None`.

```python
>>> from analytics.tendencies import *
>>> tendency_text("zone_mix", "rear_backhand", "drop", 9, 14)
'From the back court, backhand side: drops 9 of 14 times.'
>>> tendency_text("response", "net", "lift", 7, 9)
'After a net shot: lifts 7 of 9 times.'
>>> tendency_text("serve_placement", "serve", "rear_backhand", 6, 10)
"Serves long to the receiver's backhand 6 of 10 times."
>>> tendency_text("error_reason", "error", "net", 6, 9)
'Errors into the net: 6 of 9 errors.'
>>> label_source_name(0, 10), label_source_name(10, 10), label_source_name(3, 10)
('labels', 'predictions', 'mixed')
>>> parse_sets("1,2"), parse_sets("")
((1, 2), None)

```

## Example on a small synthetic table

My match, where the opponent, from the back court on the backhand side, drops 8 times and clears twice,
and wins every rally with that shot. Two other matches in the same table are ignored: only the player's
own shots count.

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
>>> [(tendency["id"], tendency["k"], tendency["n"]) for tendency in tendencies]
[('zone_mix:rear_backhand:drop', 8, 10), ('zone_mix:rear_backhand:clear', 2, 10), ('response:serve:drop', 8, 10), ('response:serve:clear', 2, 10), ('winning_shot:winner:drop', 8, 10), ('winning_shot:winner:clear', 2, 10)]
>>> tendencies[1]["text"], tendencies[1]["label_source"], tendencies[1]["sets"], sorted(tendencies[0])
('From the back court, backhand side: clears 2 of 10 times.', 'labels', [1], ['condition', 'family', 'id', 'k', 'label_source', 'match_id', 'n', 'outcome', 'player', 'rate', 'sets', 'text'])
>>> [tendency["text"] for tendency in tendencies_for("club_2026_10_20", "me", database_path)]
['Serves short down the middle 10 of 10 times.']
>>> try:
...     tendencies_for("club_2026_10_20", "nobody", database_path)
... except UnknownMatchPlayerError as error:
...     print(error)
no shots for player 'nobody' in match 'club_2026_10_20'
>>> shutil.rmtree(temp_dir)

```
