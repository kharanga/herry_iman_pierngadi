# vision/features.py

Turns the detected hits of tracked frame ranges (`vision/hits.py`) into rows of the
`shots` table with `source = 'own'`, so the classifier, the tendency engine and the
report run on tracked footage exactly as on ShuttleSet. The columns and types come
from `analytics.build_shots.schema_columns()` (the contract in `analytics/schema.sql`)
and the geometry (normalisation so the hitter is on the near half, zones, `is_cross`)
from `analytics.court.shot_geometry`, the same function ShuttleSet rows go through.

How a hit becomes a row:
- `player` is the detected hitter, `me` (near side) or `opp` (far side).
- The **hit point** (`hit_x/y`) is the hitter's feet at the hit frame, from the player
  tracks (`positions_*.csv`, court meters), and `player_x/y` is the same point;
  `opponent_x/y` is the other player's feet at the same frame. If the exact frame has no
  position, the nearest within 3 frames (scaled with the frame rate) is used, else the
  point is NULL. The pipeline tracks the players only, so the racket contact (the
  ShuttleSet convention, about 1 m in front of the feet on a reaching shot) is not
  available; the shipped classifier is trained on feet (`docs/results/classifier_feet.md`).
- The **landing point** is the next hit's hit point in the same rally (the reply
  hitter's feet), which is ShuttleSet's convention: each shot's landing equals the next
  shot's hit point (`docs/results/schema.md`). A rally's last shot has no next hit and
  gets a NULL landing (the classifier leaves it unpredicted).
- A rally's first row is usually an **inferred serve** written by `vision.hits`: the
  movement detector can't see a serve, so without it the first detected hit, the
  return, would become `ball_round` 1, and the classifier calls `ball_round` 1 a serve
  almost every time (`docs/results/no_shuttle.md`: 5 of 7 returns on the broadcast
  ranges). So `vision.hits` adds a row for the server 0.76 s before the first detected
  hit (`score` NaN in `hits_*.csv`). Here it is a hit like any other: the server's feet
  are its hit point, the return its landing, and the detected hits follow at
  `ball_round` 2 onward.
- `set_no` is given per range (sets aren't detected from video), `rally` continues per
  set across ranges, `ball_round` is the hit's order in its rally, `frame_num` the hit
  frame, `fps` the video's. Tracks made with `vision.players --frame-stride N` (the
  tracking record next to the positions file says so) are a video at `fps / N`: the
  row's `fps` is that effective rate and `frame_num` is the video frame divided by `N`,
  so times between hits come out right and the rows look like those of a video at the
  effective rate.
- NULL on every own row: `type_en`, `type_group` (no labels; `type_pred`/`type_conf` are
  filled by `shots.predict`), `backhand`, `roundscore_me/opp`, `getpoint_player`,
  `lose_reason`.

Writing replaces the own rows of that `match_id` in one transaction and never touches
other rows. A rebuild leaves `type_pred` NULL until `shots.predict` runs again.

The per-hit logic (`feet_at`, `shot_row`, `rally_shot_rows`,
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
| `video` | the video; for each range, the files of `python -m vision.players` and `python -m vision.hits` must exist (the calibration, `data/calibration/<video id>/court.json`, is needed by `--spot-check` to draw) |
| `--match-id` | recording id for the rows (any text; `report.serve` shows it as the title) |
| `--range START END SET` | a tracked range and its set number (1-3); repeat for several ranges, rallies are numbered on per set |
| `--left-handed` | comma-separated players who are left-handed (`me`, `opp`); zones and the hitting shoulder depend on it |
| `--keep-other-sets` | replace only the match's rows of the sets given in `--range` and keep its other sets, so one match can be written one video (set) at a time; without it every own row of the match id is replaced |
| `--database` | DuckDB file (default `data/scout.duckdb`) |
| `--players-dir`, `--hits-dir` | roots of the player tracking files and the hit files (defaults `data/players`, `data/hits`), for tracks of another configuration kept next to the default ones |
| `--compare` | after writing, compare the rows with this ShuttleSet match's labelled rows (below) |
| `--spot-check` | write nothing; draw check images for the match's rows already in the database: `data/checks/features/<match-id>_spot_check_<k>.jpg` (20 random rows, 5 per sheet: the hit frame with the hit point (the feet) in yellow and the opponent in blue, and the next hit's frame with the landing point in red, captioned with the row's values and, with `--compare`, the matched label) and, with `--compare`, `<match-id>_label_vs_feet.jpg` (6 matched rows: ShuttleSet's hit pixel in red, the tracked feet, `player_x/y`, in yellow). Prints the `not_over_net` breakdown and the sample tally |

| `--spot-check-first N` | write nothing; draw the match's first N rows in order (set, rally, ball_round), 10 per sheet, to `data/checks/features/<match-id>_first<N>_<k>.jpg`, the same tile pair as `--spot-check` (hit frame with the hit point in yellow and the opponent in blue; the next hit's frame with the landing point in red, or the black "last shot, landing NULL" tile). The left caption is `s<set> r<rally> #<ball_round> [inferred serve] f<frame> <hitter> pred <type> (<confidence>)` (an inferred serve is a row whose frame has `score` NaN in the match's hits files under `--hits-dir`), the right one `next hit f<frame> by <player>: <hit_zone> -> <landing_zone> straight|cross` |

Then `python -m shots.predict` fills `type_pred`/`type_conf`, and
`python -m analytics.tendencies <match-id> me` and `python -m report.serve <match-id> --me me`
work as for any match.

Printed: `wrote N own shots for match 'ID' (R rallies, L with a landing point) to <database>`
(rallies counted as distinct set and rally pairs), then with `--compare` one line per key
of `comparison_summary`; with `--spot-check` the counts of `not_over_net_breakdown` and
`sample_tally` and the saved paths. Exit codes: `0`;
`1` with `error: ...` for a missing video or track file (the message names the command
that writes it), or a set number outside 1-3; `2` for invalid
arguments, including neither `--range` nor `--spot-check`.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `DATABASE_PATH` | `Path("data/scout.duckdb")` | default database |
| `SOURCE_OWN` | `"own"` | the rows' `source` |
| `FEET_MAX_FRAME_GAP` | `3` | how far from the hit frame a player position may be taken |
| `POINT_NAMES` | `("hit", "landing", "player", "opponent")` | the four point columns, in `shot_geometry`'s names |
| `END_SWITCH_SCORE` | `11` | in set 3, ends change when a player reaches this (ShuttleSet side mapping) |
| `HIT_POINT_SOURCES` | `("feet",)` | the only hit point `build_own_match` accepts (the racket contact left the pipeline with the shuttle) |
| `CHECKS_DIR` | `Path("data/checks/features")` | where `--spot-check` writes |
| `SPOT_CHECK_ROWS`, `SPOT_CHECK_SEED`, `LABEL_VS_FEET_ROWS` | `20`, `1`, `6` | the samples drawn (numpy seed, so the same rows every run) |
| `ORDERED_SHEET_ROWS` | `10` | rows per sheet of `--spot-check-first` |

```python
>>> from vision.features import FEET_MAX_FRAME_GAP, POINT_NAMES, SOURCE_OWN
>>> SOURCE_OWN, FEET_MAX_FRAME_GAP, POINT_NAMES
('own', 3, ('hit', 'landing', 'player', 'opponent'))

```

## Rows

The examples use a two-hit rally: `me` hits at frame 100, `opp` at 130.

```python
>>> import numpy as np
>>> import pandas as pd
>>> positions = pd.DataFrame({"frame": [100, 101, 100, 130, 131], "player": ["me", "me", "opp", "opp", "me"],
...                           "x_m": [3.0, 3.1, 3.0, 2.5, 2.0], "y_m": [2.0, 2.1, 11.0, 10.0, 4.0], "track_id": 1, "confidence": 0.9})
>>> hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 130], "hitter": ["me", "opp"], "score": 0.9, "reach": 0.0})

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

### `shot_row(match_id, set_no, rally, ball_round, frame, fps, hitter, hit, landing, opponent, left_handed, feet=None) -> dict`
One row with every schema column. `feet` is the hitter's position (`player_x/y`); when
it is `None`, `hit` doubles as it.
`shot_geometry` normalises the points so the hitter is on the near half and fills
the zones and `is_cross`.

```python
>>> from vision.features import shot_row
>>> row = shot_row("club_1", 1, 1, 1, 100, 30.0, "me", (3.0, 2.0), (2.5, 10.0), (3.0, 11.0), frozenset())
>>> {key: row[key] for key in ("player", "hit_x", "hit_y", "landing_x", "landing_y", "opponent_y", "hit_zone", "landing_zone", "is_cross")}
{'player': 'me', 'hit_x': 3.0, 'hit_y': 2.0, 'landing_x': 2.5, 'landing_y': 10.0, 'opponent_y': 11.0, 'hit_zone': 'rear_backhand', 'landing_zone': 'mid_forehand', 'is_cross': False}
>>> len(row), row["source"], row["type_group"], row["roundscore_me"]
(30, 'own', None, None)

```

With a hit point and the feet given separately (not what the pipeline does now; it
passes the feet as the hit point), the zone comes from the hit point:

```python
>>> with_feet = shot_row("club_1", 1, 1, 1, 100, 30.0, "me", (3.4, 3.0), (2.5, 10.0), (3.0, 11.0), frozenset(), (3.0, 2.0))
>>> {key: with_feet[key] for key in ("hit_x", "hit_y", "player_x", "player_y", "hit_zone")}
{'hit_x': 3.4, 'hit_y': 3.0, 'player_x': 3.0, 'player_y': 2.0, 'hit_zone': 'mid_forehand'}

```

A far-side hitter is rotated by a half turn, `(x, y) -> (6.1 - x, 13.4 - y)`:

```python
>>> far = shot_row("club_1", 1, 1, 2, 130, 30.0, "opp", (2.5, 10.0), (2.0, 4.0), (2.0, 4.0), frozenset())
>>> {key: round(far[key], 1) for key in ("hit_x", "hit_y", "landing_x", "landing_y", "opponent_x", "opponent_y")}
{'hit_x': 3.6, 'hit_y': 3.4, 'landing_x': 4.1, 'landing_y': 9.4, 'opponent_x': 4.1, 'opponent_y': 9.4}
>>> far["player"], far["hit_zone"], far["landing_zone"]
('opp', 'mid_forehand', 'front_backhand')

```

### `effective_frame_numbers(table, frame_stride) -> DataFrame`, `rally_shot_rows(rally_hits, feet, match_id, set_no, rally, fps, left_handed, feet_max_gap=3) -> list[dict]`, `own_shot_rows(hits, positions, match_id, set_no, fps, first_rally=1, left_handed=frozenset(), frame_stride=1) -> list[dict]`
All rows of one rally (hits in frame order, `ball_round` from 1; hit point = the
hitter's feet; each landing = the next hit's hit point, the last `None`), and of a
whole range (rallies numbered from `first_rally` in the order of the hits file). The
feet gap is scaled with `fps` (`vision.hits.frames_for_fps`: 3 frames at 30 fps, 6 at
60). With a `frame_stride`, `effective_frame_numbers` divides the hits' and positions'
frame numbers by it first, and `fps` should be the effective rate.

```python
>>> from vision.features import effective_frame_numbers, own_shot_rows
>>> effective_frame_numbers(pd.DataFrame({"frame": [100, 102, 105]}), 2)["frame"].tolist()
[50, 51, 52]
>>> strided_hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 130], "hitter": ["me", "opp"], "score": 0.9, "reach": np.nan})
>>> [(row["frame_num"], row["fps"]) for row in own_shot_rows(strided_hits, positions, "club_1", 1, 30.0, frame_stride=2)]
[(50, 30.0), (65, 30.0)]

```

```python
>>> from vision.features import own_shot_rows
>>> rows = own_shot_rows(hits, positions, "club_1", 1, 30.0)
>>> [(row["ball_round"], row["player"], round(row["hit_x"], 1), round(row["hit_y"], 1), row["landing_x"], row["landing_y"]) for row in rows]
[(1, 'me', 3.0, 2.0, 2.5, 10.0), (2, 'opp', 3.6, 3.4, None, None)]

```

The second (last) shot has no landing; its hit point is its feet rotated to the near
half because its hitter was on the far side. An inferred serve row in the hits file
(`score` NaN) is treated like any other hit:

```python
>>> with_serve = pd.concat([pd.DataFrame({"rally": [1], "frame": [77], "hitter": ["opp"], "score": [np.nan], "reach": [np.nan]}), hits])
>>> served = pd.concat([pd.DataFrame({"frame": [77], "player": ["opp"], "x_m": [3.0], "y_m": [11.0], "track_id": 1, "confidence": 0.9}), positions])
>>> [(row["ball_round"], row["player"], row["frame_num"], row["landing_x"] and round(row["landing_x"], 1), row["landing_y"]) for row in own_shot_rows(with_serve, served, "club_1", 1, 30.0)]
[(1, 'opp', 77, 3.1, 11.4), (2, 'me', 100, 2.5, 10.0), (3, 'opp', 130, None, None)]

```

### `pandas_dtype_for(duckdb_type) -> str`, `schema_frame(rows) -> DataFrame`
The rows as a data frame with exactly the schema's columns, in its order, as nullable
pandas types (`Int64`, `float64`, `boolean`, `object`).

```python
>>> from vision.features import pandas_dtype_for, schema_frame
>>> pandas_dtype_for("INTEGER"), pandas_dtype_for("DOUBLE"), pandas_dtype_for("BOOLEAN"), pandas_dtype_for("VARCHAR")
('Int64', 'float64', 'boolean', 'object')
>>> frame = schema_frame(rows)
>>> frame.shape, list(frame.columns[:4])
((2, 30), ['source', 'match_id', 'set_no', 'rally'])
>>> frame.dtypes[["set_no", "fps", "player", "backhand", "is_cross"]].astype(str).to_dict()
{'set_no': 'Int64', 'fps': 'float64', 'player': 'object', 'backhand': 'boolean', 'is_cross': 'boolean'}

```

### `write_own_shots(database_path, match_id, shots_frame, replace_sets=None) -> int`
In one transaction: create the table if needed (`analytics.build_shots.create_shots_table`),
delete the own rows with that `match_id` (only those of the set numbers in
`replace_sets`, if given), insert the frame; returns the number of own rows of that
`match_id` afterwards. Raises (and rolls back) on any constraint violation.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from vision.features import write_own_shots
>>> database = Path(tempfile.mkdtemp()) / "scout.duckdb"
>>> def set_rows(set_no, count):
...     return schema_frame([shot_row("club_1", set_no, 1, index + 1, 100 + index, 60.0, "me", (3.0, 2.0), None, None, frozenset())
...                          for index in range(count)])
>>> write_own_shots(database, "club_1", set_rows(1, 3)), write_own_shots(database, "club_1", set_rows(2, 2), replace_sets=[2])
(3, 5)
>>> write_own_shots(database, "club_1", set_rows(2, 4), replace_sets=[2]), write_own_shots(database, "club_1", set_rows(1, 1))
(7, 1)

```

### `load_range_tracks(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR, hits_dir=HITS_DIR) -> (hits, rallies, positions)`, `range_frame_stride(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> int`
The three track files of a range as data frames (`FileNotFoundError` naming the
command that writes a missing one) and the stride of the range's tracking record.

### `build_own_match(video_path, match_id, ranges, database_path=DATABASE_PATH, left_handed=frozenset(), hit_point="feet", keep_other_sets=False, players_dir=PLAYERS_DIR, hits_dir=HITS_DIR) -> DataFrame`
With `keep_other_sets=True` only the sets in `ranges` are replaced (one call per set video).
The whole thing for a list of `(start_frame, end_frame, set_no)`: loads each range's
track files, builds the rows with the range's effective fps and frame numbers, writes
them, returns the frame. Errors: `ValueError` for
a `hit_point` other than `"feet"` (the parameter stays so callers that named it keep
working), `FileNotFoundError` for a missing video or track file.

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
...                     "player_x": [3.0, 3.1], "player_y": [2.0, 2.4], "hit_zone": ["rear_backhand", "mid_forehand"],
...                     "landing_zone": ["mid_backhand", "mid_backhand"], "is_cross": [False, False]})
>>> labelled = pd.DataFrame({"set_no": [1, 1, 1], "frame_num": [102, 128, 200], "player": ["B", "A", "B"], "roundscore_me": 0,
...                          "roundscore_opp": 0, "type_group": ["net", "drive", "net"], "hit_x": [3.5, 3.1, 1.0], "hit_y": [2.0, 2.4, 1.0],
...                          "landing_x": [2.5, 2.0, 1.0], "landing_y": [10.0, 9.4, 9.0], "player_x": [3.0, 3.1, 1.0], "player_y": [2.0, 2.4, 1.0],
...                          "hit_zone": ["rear_backhand", "mid_forehand", "rear_backhand"], "landing_zone": "mid_backhand", "is_cross": False})
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
...                     "landing_zone": ["not_over_net", "mid_backhand", "not_over_net", "front_forehand"]})
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

### `direction_word(is_cross) -> str`, `frame_centre(frame) -> (int, int)`, `inferred_serve_frames(video_path, hits_dir=HITS_DIR) -> set[int]`, `first_rows(own, row_count) -> DataFrame`, `ordered_sheet_path(match_id, row_count, sheet_index, checks_dir=CHECKS_DIR) -> Path`
Helpers of the ordered sheets: `cross` / `straight` / `no direction` for `is_cross`; the
crop centre when a row has no hit point; the frames of the inferred serves in the
match's hits files; the first rows in `(set_no, rally, ball_round)` order; the sheet
file name.

### `frame_stride_of_row(row, video_fps) -> int`, `video_frame_of_row(row, video_fps) -> int`, `is_inferred_serve(row, serve_frames, video_fps) -> bool`
The sheets read frames from the video, but rows of tracks made with a frame stride hold
`frame_num` = video frame // stride and the effective `fps`. The stride comes back as
the video's fps over the row's `fps` (1 when either is missing). The tile reads video
frame `frame_num x stride` and its caption shows that frame. For tracks that start off
the stride grid, that is one sampled frame early. A row is an inferred serve when its
`frame_num` is one of the hits files' serve frames divided by the same stride.

```python
>>> from vision.features import frame_stride_of_row, is_inferred_serve, video_frame_of_row
>>> strided, plain = pd.DataFrame({"frame_num": [50, 100], "fps": [30.0, 60.0]}).itertuples()
>>> frame_stride_of_row(strided, 60.0), video_frame_of_row(strided, 60.0), is_inferred_serve(strided, {100, 161}, 60.0)
(2, 100, True)
>>> frame_stride_of_row(plain, 60.0), video_frame_of_row(plain, 60.0), is_inferred_serve(plain, {101}, 60.0)
(1, 100, False)

```

```python
>>> from vision.features import direction_word, first_rows, frame_centre, ordered_sheet_path
>>> direction_word(True), direction_word(False), direction_word(None)
('cross', 'straight', 'no direction')
>>> frame_centre(np.zeros((720, 1280, 3), dtype=np.uint8))
(640, 360)
>>> first_rows(pd.DataFrame({"set_no": [1, 1, 1], "rally": [2, 1, 1], "ball_round": [1, 2, 1]}), 2)[["rally", "ball_round"]].values.tolist()
[[1, 1], [1, 2]]
>>> ordered_sheet_path("stand_in_set1", 100, 3).as_posix()
'data/checks/features/stand_in_set1_first100_3.jpg'

```

### `own_rows_with_labels(database_path, match_id, shuttleset_match_id, tolerance=5) -> DataFrame`, `write_spot_check_sheets(...)`, `write_ordered_spot_check_sheets(video_path, homography, own, row_count, match_id, serve_frames, checks_dir=CHECKS_DIR, rows_per_sheet=10) -> list[Path]`, `ordered_spot_check(video_path, match_id, row_count, database_path=DATABASE_PATH, shuttleset_match_id=None, checks_dir=CHECKS_DIR, hits_dir=HITS_DIR) -> list[Path]`, `write_label_vs_feet_sheet(...)`, `spot_check(video_path, match_id, database_path=DATABASE_PATH, shuttleset_match_id=None, checks_dir=CHECKS_DIR) -> dict`
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
