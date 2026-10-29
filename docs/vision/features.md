# vision/features.py

Turns the detected hits of tracked frame ranges (`vision/hits.py`) into rows of the
`shots` table with `source = 'own'`, so the classifier, the tendency engine and the
report run on tracked footage exactly as on ShuttleSet. The columns and types come
from `analytics.build_shots.schema_columns()` (the contract in `analytics/schema.sql`)
and the geometry (normalisation so the hitter is on the near half, zones, `is_cross`)
from `analytics.court.shot_geometry`, the same function ShuttleSet rows go through.

How a hit becomes a row:
- `player` is the detected hitter, `me` (near side) or `opp` (far side).
- The **hit point** and `player_x/y` are the hitter's feet at the hit frame, from the
  player tracks (`positions_*.csv`, court meters); `opponent_x/y` the other player's
  feet at the same frame. If the exact frame has no position, the nearest within 3
  frames is used, else the point is NULL.
- The **landing point** is the next hit's hit point in the same rally: the receiver's
  feet when they reached the shuttle, which is what ShuttleSet's `landing_x/y` means
  (`docs/results/schema.md`) and what the classifier was trained on. On a rally's last
  shot it is the shuttle's floor position: the shuttle track's pixel at the rally's
  end frame, mapped through the court calibration, if the shuttle is at rest there
  for 20 frames (`vision.hits.is_at_rest`), else NULL.
- `set_no` is given per range (sets aren't detected from video), `rally` continues per
  set across ranges, `ball_round` is the hit's order in its rally, `frame_num` the hit
  frame, `fps` the video's.
- NULL on every own row: `type_en`, `type_group` (no labels; `type_pred`/`type_conf` are
  filled by `shots.predict`), `backhand`, `roundscore_me/opp`, `getpoint_player`,
  `lose_reason`.

Writing replaces the own rows of that `match_id` in one transaction and never touches
other rows. The ShuttleSet hit point is **not** the hitter's feet but the floor point
under the racket contact, about 0.9 m away on average (`docs/results/features.md`); the
contract keeps the feet, and the gap is measured there.

The per-hit logic (`feet_at`, `floor_point_at_rest`, `shot_row`, `rally_shot_rows`,
`own_shot_rows`, `schema_frame`, the comparison functions) is pure; only `video_fps`,
`load_range_tracks`, `write_own_shots`, `compare_with_shuttleset`, `build_own_match`
and `main` read files or the database.

## Command

```
.venv/bin/python -m vision.features <video> --match-id ID --range START END SET [--range ...] [--left-handed me,opp] [--database path] [--compare SHUTTLESET_MATCH]
.venv/bin/python -m vision.features <video> --match-id ID --spot-check [--compare SHUTTLESET_MATCH] [--database path]
```

| Argument | Meaning |
|---|---|
| `video` | the video; its calibration (`data/calibration/<stem>/court.json`) and, for each range, the files of `python -m vision.players`, `python -m vision.shuttle` and `python -m vision.hits` must exist |
| `--match-id` | recording id for the rows (any text; `report.serve` shows it as the title) |
| `--range START END SET` | a tracked range and its set number (1-3); repeat for several ranges, rallies are numbered on per set |
| `--left-handed` | comma-separated players who are left-handed (`me`, `opp`); zones depend on it |
| `--database` | DuckDB file (default `data/scout.duckdb`) |
| `--compare` | after writing, compare the rows with this ShuttleSet match's labelled rows (below) |
| `--spot-check` | write nothing; draw check images for the match's rows already in the database: `data/checks/features/<match-id>_spot_check_<k>.jpg` (20 random rows, 5 per sheet: the hit frame with the hit point in yellow and the opponent in blue, and the next hit's frame with the landing point in red, captioned with the row's values and, with `--compare`, the matched label) and, with `--compare`, `<match-id>_label_vs_feet.jpg` (6 matched rows: ShuttleSet's hit pixel in red, the tracked feet in yellow). Prints the `not_over_net` breakdown and the sample tally |

Then `python -m shots.predict` fills `type_pred`/`type_conf`, and
`python -m analytics.tendencies <match-id> me` and `python -m report.serve <match-id> --me me`
work as for any match.

Printed: `wrote N own shots for match 'ID' (R rallies, L with a landing point) to <database>`
(rallies counted as distinct set and rally pairs), and with `--compare` one line per key
of `comparison_summary`; with `--spot-check` the counts of `not_over_net_breakdown` and
`sample_tally` and the saved paths. Exit codes: `0`;
`1` with `error: ...` for a missing video, calibration or track file (the message
names the command that writes it), or a set number outside 1-3; `2` for invalid
arguments, including neither `--range` nor `--spot-check`.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `DATABASE_PATH` | `Path("data/scout.duckdb")` | default database |
| `SOURCE_OWN` | `"own"` | the rows' `source` |
| `FEET_MAX_FRAME_GAP` | `3` | how far from the hit frame a player position may be taken |
| `POINT_NAMES` | `("hit", "landing", "player", "opponent")` | the four point columns, in `shot_geometry`'s names |
| `END_SWITCH_SCORE` | `11` | in set 3, ends change when a player reaches this (ShuttleSet side mapping) |
| `CHECKS_DIR` | `Path("data/checks/features")` | where `--spot-check` writes |
| `SPOT_CHECK_ROWS`, `SPOT_CHECK_SEED`, `LABEL_VS_FEET_ROWS` | `20`, `1`, `6` | the samples drawn (numpy seed, so the same rows every run) |

```python
>>> from vision.features import FEET_MAX_FRAME_GAP, POINT_NAMES, SOURCE_OWN
>>> SOURCE_OWN, FEET_MAX_FRAME_GAP, POINT_NAMES
('own', 3, ('hit', 'landing', 'player', 'opponent'))

```

## Rows

The examples use a two-hit rally: `me` hits at frame 100, `opp` at 130, and the
shuttle comes to rest from frame 150. The calibration is the real broadcast one.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from vision.court import homography_from_corners
>>> homography = homography_from_corners([[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])
>>> positions = pd.DataFrame({"frame": [100, 101, 100, 130, 131], "player": ["me", "me", "opp", "opp", "me"],
...                           "x_m": [3.0, 3.1, 3.0, 2.5, 2.0], "y_m": [2.0, 2.1, 11.0, 10.0, 4.0], "track_id": 1, "confidence": 0.9})
>>> hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 130], "hitter": ["me", "opp"], "score": 0.9, "reach": 0.0})
>>> rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [150], "hits": [2]})
>>> resting = pd.DataFrame({"frame": range(150, 170), "x": 640, "y": 500, "visibility": 1, "inpainted": 0})

```

### `feet_by_frame(positions) -> dict`, `feet_at(feet, player, frame, max_gap=3) -> (x, y) | None`
Index the player positions table by player and frame; look one up, taking the
nearest frame within `max_gap`.

```python
>>> from vision.features import feet_at, feet_by_frame
>>> feet = feet_by_frame(positions)
>>> feet["opp"]
{100: (3.0, 11.0), 130: (2.5, 10.0)}
>>> feet_at(feet, "me", 102), feet_at(feet, "me", 110), feet_at(feet, "opp", 128)
((3.1, 2.1), None, (2.5, 10.0))

```

### `other_player(player) -> str`

```python
>>> from vision.features import other_player
>>> other_player("me"), other_player("opp")
('opp', 'me')

```

### `floor_point_at_rest(shuttle, end_frame, homography, rest_frame_count=20) -> (x, y) | None`
The shuttle's floor position in court meters at `end_frame`, if the shuttle track is
visible and at rest for the next `rest_frame_count` frames; else `None`.

```python
>>> from vision.features import floor_point_at_rest
>>> tuple(round(value, 2) for value in floor_point_at_rest(resting, 150, homography))
(3.07, 4.58)
>>> floor_point_at_rest(resting.iloc[:10], 150, homography) is None
True

```

### `shot_row(match_id, set_no, rally, ball_round, frame, fps, hitter, hit, landing, opponent, left_handed) -> dict`
One row with every schema column. `hit` doubles as the hitter's position (`player_x/y`).
`shot_geometry` normalises the points so the hitter is on the near half and fills
the zones and `is_cross`.

```python
>>> from vision.features import shot_row
>>> row = shot_row("club_1", 1, 1, 1, 100, 30.0, "me", (3.0, 2.0), (2.5, 10.0), (3.0, 11.0), frozenset())
>>> {key: row[key] for key in ("player", "hit_x", "hit_y", "landing_x", "landing_y", "opponent_y", "hit_zone", "landing_zone", "is_cross")}
{'player': 'me', 'hit_x': 3.0, 'hit_y': 2.0, 'landing_x': 2.5, 'landing_y': 10.0, 'opponent_y': 11.0, 'hit_zone': 'rear_middle', 'landing_zone': 'mid_forehand', 'is_cross': False}
>>> len(row), row["source"], row["type_group"], row["roundscore_me"]
(28, 'own', None, None)

```

A far-side hitter is rotated by a half turn, `(x, y) -> (6.1 - x, 13.4 - y)`:

```python
>>> far = shot_row("club_1", 1, 1, 2, 130, 30.0, "opp", (2.5, 10.0), (2.0, 4.0), (2.0, 4.0), frozenset())
>>> {key: round(far[key], 1) for key in ("hit_x", "hit_y", "landing_x", "landing_y", "opponent_x", "opponent_y")}
{'hit_x': 3.6, 'hit_y': 3.4, 'landing_x': 4.1, 'landing_y': 9.4, 'opponent_x': 4.1, 'opponent_y': 9.4}
>>> far["player"], far["hit_zone"], far["landing_zone"]
('opp', 'mid_forehand', 'mid_backhand')

```

### `rally_shot_rows(rally_hits, feet, last_landing, match_id, set_no, rally, fps, left_handed) -> list[dict]`, `own_shot_rows(hits, rallies, positions, shuttle, homography, match_id, set_no, fps, first_rally=1, left_handed=frozenset()) -> list[dict]`
All rows of one rally (hits in frame order, `ball_round` from 1, each landing = the
next hit's hit point, the last = `last_landing`), and of a whole range (the last
landing of each rally from `floor_point_at_rest` at the rally's `end_frame`).

```python
>>> from vision.features import own_shot_rows
>>> rows = own_shot_rows(hits, rallies, positions, resting, homography, "club_1", 1, 30.0)
>>> [(row["ball_round"], row["player"], round(row["landing_x"], 2), round(row["landing_y"], 2)) for row in rows]
[(1, 'me', 2.5, 10.0), (2, 'opp', 3.03, 8.82)]

```

The second shot's landing is the resting shuttle's floor point, rotated to the
receiver's (far) half because its hitter was on the far side.

### `pandas_dtype_for(duckdb_type) -> str`, `schema_frame(rows) -> DataFrame`
The rows as a data frame with exactly the schema's columns, in its order, as nullable
pandas types (`Int64`, `float64`, `boolean`, `object`).

```python
>>> from vision.features import pandas_dtype_for, schema_frame
>>> pandas_dtype_for("INTEGER"), pandas_dtype_for("DOUBLE"), pandas_dtype_for("BOOLEAN"), pandas_dtype_for("VARCHAR")
('Int64', 'float64', 'boolean', 'object')
>>> frame = schema_frame(rows)
>>> frame.shape, list(frame.columns[:4])
((2, 28), ['source', 'match_id', 'set_no', 'rally'])
>>> frame.dtypes[["set_no", "fps", "player", "backhand", "is_cross"]].astype(str).to_dict()
{'set_no': 'Int64', 'fps': 'float64', 'player': 'object', 'backhand': 'boolean', 'is_cross': 'boolean'}

```

### `write_own_shots(database_path, match_id, shots_frame) -> int`
In one transaction: create the table if needed (`analytics.build_shots.create_shots_table`),
delete the own rows with that `match_id`, insert the frame; returns the number of own
rows of that `match_id` afterwards. Raises (and rolls back) on any constraint violation.

### `build_own_match(video_path, match_id, ranges, database_path=DATABASE_PATH, left_handed=frozenset()) -> DataFrame`
The whole thing for a list of `(start_frame, end_frame, set_no)`: loads each range's
track files (`load_range_tracks`; `FileNotFoundError` naming the command for a missing
one), builds the rows, writes them, returns the frame.

## Comparison with ShuttleSet

### `shuttleset_player_side(player, set_no, roundscore_a, roundscore_b, downcourt) -> str`, `with_labelled_sides(labelled, downcourt) -> DataFrame`
Which side (`me` near, `opp` far) a ShuttleSet `A`/`B` hitter was on: `downcourt = 0`
means A starts near, ends change every set and at 11 points in set 3
(`docs/results/schema.md`).

```python
>>> from vision.features import shuttleset_player_side
>>> [shuttleset_player_side("A", 1, 0, 0, 1), shuttleset_player_side("B", 1, 0, 0, 1), shuttleset_player_side("A", 2, 3, 2, 1)]
['opp', 'me', 'me']
>>> shuttleset_player_side("A", 3, 5, 9, 1), shuttleset_player_side("A", 3, 8, 11, 1), shuttleset_player_side("A", 1, 0, 0, 0)
('opp', 'me', 'me')

```

### `matched_shot_pairs(own, labelled, tolerance=5) -> DataFrame`
One-to-one matching of own rows to labelled rows of the same set within `tolerance`
frames (closest first); the result has the own columns prefixed `own_` and the
labelled ones `label_`.

```python
>>> from vision.features import matched_shot_pairs, with_labelled_sides
>>> own = pd.DataFrame({"set_no": [1, 1], "frame_num": [100, 131], "player": ["me", "opp"], "type_pred": ["net", "lift"],
...                     "type_conf": [0.9, 0.8], "hit_x": [3.0, 3.1], "hit_y": [2.0, 2.4], "landing_x": [2.5, 2.0], "landing_y": [10.0, 9.4],
...                     "player_x": [3.0, 3.1], "player_y": [2.0, 2.4], "hit_zone": ["rear_middle", "mid_middle"],
...                     "landing_zone": ["mid_middle", "mid_middle"], "is_cross": [False, False]})
>>> labelled = pd.DataFrame({"set_no": [1, 1, 1], "frame_num": [102, 128, 200], "player": ["B", "A", "B"], "roundscore_me": 0,
...                          "roundscore_opp": 0, "type_group": ["net", "drive", "net"], "hit_x": [3.5, 3.1, 1.0], "hit_y": [2.0, 2.4, 1.0],
...                          "landing_x": [2.5, 2.0, 1.0], "landing_y": [10.0, 9.4, 9.0], "player_x": [3.0, 3.1, 1.0], "player_y": [2.0, 2.4, 1.0],
...                          "hit_zone": ["rear_middle", "mid_middle", "rear_backhand"], "landing_zone": "mid_middle", "is_cross": False})
>>> pairs = matched_shot_pairs(own, with_labelled_sides(labelled, downcourt=1))
>>> pairs[["own_frame_num", "label_frame_num", "own_player", "label_player", "label_side"]]
   own_frame_num  label_frame_num own_player label_player label_side
0            100              102         me            B         me
1            131              128        opp            A        opp

```

### `point_errors(pairs, name) -> Series`, `comparison_summary(pairs, own_count, labelled_count) -> dict`, `confusion(pairs) -> DataFrame`
Distance in meters between the own and labelled point `name` per pair; the summary
(counts, hitter agreement against `label_side`, `type_pred` accuracy against
`type_group` on pairs with both and a group other than `unknown`, median and mean
point errors for hit, player and landing points, zone and `is_cross` agreement); and
the labelled-vs-predicted group cross table.

```python
>>> from vision.features import comparison_summary, confusion, point_errors
>>> point_errors(pairs, "hit").round(2).tolist()
[0.5, 0.0]
>>> summary = comparison_summary(pairs, len(own), len(labelled))
>>> summary["matched_pairs"], summary["hitter_agreement"], summary["type_pred_accuracy"], summary["type_pred_judged"]
(2, 1.0, 0.5, 2)
>>> round(summary["hit_point_median_error_m"], 2), summary["player_point_median_error_m"], summary["hit_zone_agreement"]
(0.25, 0.0, 1.0)
>>> confusion(pairs)
predicted  lift  net
labelled            
drive         1    0
net           0    1

```

### `compare_with_shuttleset(database_path, own_match_id, shuttleset_match_id, tolerance=5) -> (dict, DataFrame)`
Reads both matches' rows and the ShuttleSet match's `downcourt` from the database and
returns `comparison_summary` and the pairs.

## Checks

### `rally_count(shots_frame) -> int`
Distinct `(set_no, rally)` pairs (rally numbers restart per set).

```python
>>> from vision.features import rally_count
>>> rally_count(pd.DataFrame({"set_no": [1, 1, 2, 2], "rally": [1, 1, 1, 2]}))
3

```

### `not_over_net_breakdown(own) -> dict`
How many rows have `landing_zone = 'not_over_net'` and why: `followed_by_same_player`
(the next hit in the rally is by the same player, so the landing is their own feet; an
extra or missed hit in between), `rally_end_rest_point` (a last shot whose floor point
is on the hitter's side), `other`.

```python
>>> from vision.features import not_over_net_breakdown
>>> own = pd.DataFrame({"set_no": 1, "rally": [1, 1, 1, 2], "ball_round": [1, 2, 3, 1], "player": ["me", "me", "opp", "me"],
...                     "landing_zone": ["not_over_net", "mid_middle", "not_over_net", "front_middle"]})
>>> not_over_net_breakdown(own)
{'rows': 4, 'not_over_net': 2, 'followed_by_same_player': 1, 'rally_end_rest_point': 1, 'other': 0}

```

### `sample_tally(sample) -> dict`
For rows with `type_pred` and `label_type_group` columns: how many predictions are
right, wrong, have no label, or no prediction.

```python
>>> from vision.features import sample_tally
>>> sample = pd.DataFrame({"type_pred": ["net", "lift", "net", None], "label_type_group": ["net", "drive", None, "net"]})
>>> sample_tally(sample)
{'rows': 4, 'right': 1, 'wrong': 1, 'without_label': 1, 'without_prediction': 1}

```

### `unnormalised_point(x, y, player) -> (x, y) | None`
Undoes the half-turn normalisation for drawing: a far-side hitter's (`opp`) stored
points are rotated back to where they are in the video.

```python
>>> from vision.features import unnormalised_point
>>> tuple(round(value, 2) for value in unnormalised_point(3.6, 3.4, "opp")), unnormalised_point(3.0, 2.0, "me"), unnormalised_point(None, 2.0, "me")
((2.5, 10.0), (3.0, 2.0), None)

```

### `spot_check_sample(own, row_count=20, seed=1) -> DataFrame`
The rows drawn for the sheets, in table order (numpy's generator with `seed`).

```python
>>> from vision.features import spot_check_sample
>>> spot_check_sample(pd.DataFrame({"frame_num": range(100, 110)}), row_count=3, seed=1)["frame_num"].tolist()
[103, 104, 107]

```

### `own_rows_with_labels(database_path, match_id, shuttleset_match_id, tolerance=5) -> DataFrame`, `write_spot_check_sheets(...)`, `write_label_vs_feet_sheet(...)`, `spot_check(video_path, match_id, database_path=DATABASE_PATH, shuttleset_match_id=None, checks_dir=CHECKS_DIR) -> dict`
Read the match's own rows (with the matched label's group and player when a ShuttleSet
match is given), draw the sheets, and return the two breakdowns, the sampled frames and
the paths. Errors: `ValueError` if the match has no own rows; the usual
`FileNotFoundError` for the video or calibration.

### `parse_range(text) -> (int, int, int)`, `main(arguments) -> int`

```python
>>> from vision.features import parse_range
>>> parse_range(["10", "20", "2"])
(10, 20, 2)
>>> parse_range(["10", "20", "4"])
Traceback (most recent call last):
ValueError: set number must be 1, 2 or 3, got 4

```
