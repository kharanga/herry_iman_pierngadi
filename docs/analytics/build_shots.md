# `analytics/build_shots.py` and `analytics/schema.sql`

Examples need: data/CoachAI-Projects/ShuttleSet/set (run ./ingest/fetch_shuttleset.sh)

Builds the `shots` table: one row per shot, the same columns whether the shot came
from ShuttleSet or from my own footage. Everything downstream (classifier,
tendencies, report) reads only this table.

`analytics/schema.sql` is the contract. It defines the table; this module fills the
ShuttleSet rows. The vision track writes `source = 'own'` rows against the same
schema and uses `analytics.court.shot_geometry` for positions, zones and `is_cross`.

## Command

```
.venv/bin/python -m ingest.load_shuttleset
.venv/bin/python -m analytics.build_shots [database_path]
```

- `database_path` (optional): default `data/scout.duckdb`. It must already hold
  `raw_shots`, `matches` and `homography` (built by `ingest.load_shuttleset`).
- Run from the project root. Prints:

```
shots: 36483 shuttleset rows from 36484 raw shots (1 exact duplicates dropped)
```

## The `shots` table

Created by `analytics/schema.sql` with `CREATE TABLE IF NOT EXISTS`. A rebuild replaces
only the `source = 'shuttleset'` rows, so `own` rows survive it.

| Column | Type | Null | Meaning |
|---|---|---|---|
| source | VARCHAR | no | `shuttleset` or `own` |
| match_id | VARCHAR | no | ShuttleSet folder name, or own recording id |
| set_no | INTEGER | no | 1-3 |
| rally | INTEGER | no | rally number within the set |
| ball_round | INTEGER | no | shot number within the rally, from 1 |
| frame_num | INTEGER | yes | video frame of the hit |
| fps | DOUBLE | yes | frame rate of the match video |
| player | VARCHAR | no | hitter: `A`/`B` (ShuttleSet) or `me`/`opp` (own) |
| type_en | VARCHAR | yes | English shot type |
| type_detail | VARCHAR | yes | shot label: `serve_short, serve_long, net, drive, clear, drop, smash, lift, unknown` (see `shot_types.md`); NULL on own rows until labelled |
| type_group | VARCHAR | yes | `serve, clear, drop, smash, net, lift, drive, unknown`, the group of `type_detail` |
| type_detail_pred | VARCHAR | yes | classifier's predicted label; NULL until the classifier runs |
| type_pred | VARCHAR | yes | the group of `type_detail_pred` |
| type_conf | DOUBLE | yes | classifier confidence; NULL until the classifier runs |
| backhand | BOOLEAN | yes | hit on the backhand |
| hit_x, hit_y | DOUBLE | yes | hitter's feet at the hit, court meters, normalized |
| landing_x, landing_y | DOUBLE | yes | where the shot ended, court meters, normalized (see below) |
| player_x, player_y | DOUBLE | yes | hitter's position at the hit, normalized |
| opponent_x, opponent_y | DOUBLE | yes | receiver's position at the hit, normalized |
| hit_zone | VARCHAR | yes | zone of the hit, from the hitter's view: one of the 6 zones below |
| landing_zone | VARCHAR | yes | zone of the landing, from the receiver's view: one of the 6 zones, or `not_over_net` |
| is_cross | BOOLEAN | yes | cross-court (see `court.md`) |
| roundscore_me | INTEGER | yes | score of `A` (ShuttleSet) or `me` (own) **before** the rally |
| roundscore_opp | INTEGER | yes | score of `B` or `opp` before the rally |
| getpoint_player | VARCHAR | yes | rally winner, on the rally's last shot only |
| lose_reason | VARCHAR | yes | why the rally ended, on the last shot only |

CHECK constraints reject any other value of `source`, `player`, `getpoint_player`,
`set_no`, `type_detail`, `type_group`, `type_detail_pred`, `type_pred`, `hit_zone` and `landing_zone`.

Zones (`analytics.court.ZONE_NAMES`): each half is 3 depth bands x 2 sides = 6 zones, with the
band edges on the painted lines.

| Depth (from the net) | Side | Zones |
|---|---|---|
| front 0-1.98 m (to the short service line), mid 1.98-3.50 m, rear 3.50 m and beyond | forehand / backhand, split at the centre line | `front_forehand`, `front_backhand`, `mid_forehand`, `mid_backhand`, `rear_forehand`, `rear_backhand` |

Forehand and backhand swap for a left-handed player. `landing_zone` can also be `not_over_net`.
`is_cross` is TRUE when the hit and landing points are on opposite sides of the centre line, NULL
for a `not_over_net` landing. Details in `court.md`.

Positions are in court meters (origin near-left corner, `x` 0-6.1 across, `y` 0-13.4
along, net at 6.7; see `court.md`), turned so that the **hitter is on the near half**.

How the ShuttleSet columns are filled:
- Exact duplicate rows of `raw_shots` are dropped first (`SELECT DISTINCT`).
- `ball_round`, `rally` become INTEGER. `backhand` is TRUE where ShuttleSet has 1, FALSE where it is blank.
- `fps` is `matches.fps` (25 or 30).
- Pixels -> meters with the match's own homography (`court.shuttleset_pixel_to_court_homography`).
  `player_x/y` come from `player_location_x/y`, `opponent_x/y` from `opponent_location_x/y`.
- In ShuttleSet each shot's hit point equals the previous shot's landing point.
  So "landing" means where the receiver reached the shuttle (their feet), not where it
  would have hit the floor. Only on a rally's last shot is it where the rally ended.
- Hit point is NULL for most serves. `hit_zone` and `is_cross` then use the server's position.
- `roundscore_A/B` in ShuttleSet are the score **after** the rally. The table stores the score
  before it: the rally winner (`getpoint_player`, or else the side whose score went up)
  loses one point. This works even where a rally is missing from the source.
- `me` is player `A`, the match winner in all 44 ShuttleSet matches; `opp` is `B`.
- Handedness: players named in `LEFT_HANDED_PLAYER_NAMES` (matched against
  `matches.winner` for `A` and `matches.loser` for `B`) get left-handed zones.

## Constants

| Name | Value |
|---|---|
| `DATABASE_PATH` | `data/scout.duckdb` |
| `SCHEMA_PATH` | `analytics/schema.sql` (absolute, next to the module) |
| `LEFT_HANDED_PLAYER_NAMES` | `frozenset({"Carolina MARIN", "Kento MOMOTA"})` |
| `POSITION_COLUMNS` | `raw_shots` pixel columns for `hit`, `landing`, `player`, `opponent` |
| `SHUTTLESET_SHOT_ROWS_SQL`, `MATCH_HOMOGRAPHIES_SQL`, `RAW_SHOT_COUNT_SQL`, `SHOTS_COLUMNS_SQL`, `SHOTS_CONSTRAINTS_SQL`, `DELETE_SHUTTLESET_SHOTS_SQL`, `INSERT_SHUTTLESET_SHOTS_SQL` | the SQL of each step |

## `build_shots(database_path, left_handed_player_names=LEFT_HANDED_PLAYER_NAMES)`

Recreates `shots` from `schema.sql` with the ShuttleSet rows, in one transaction, and keeps every
`own` row: the rows are read before the table is dropped and written back with their `hit_zone`,
`landing_zone` and `is_cross` re-derived from their stored (normalized) coordinates by
`court.shot_geometry`, so a zone-definition change reaches own rows too without re-tracking.
Own rows are assumed right-handed when re-derived (handedness is not stored). Their `rowid`s change.

- `database_path` (`Path`): a database with `raw_shots`, `matches`, `homography`.
- `left_handed_player_names` (iterable of `str`): player names as in `matches.winner` / `matches.loser`.
- Returns a `dict`: `raw_shots` (row count), `shuttleset_shots` (rows written),
  `duplicates_dropped` (the difference), `own_shots` (rows kept).
- `FileNotFoundError` if `database_path` is not a file, checked before opening it, so a mistyped path creates no
  empty database; the command prints it and exits 1.
- Errors (the transaction rolls back, so the database keeps its previous `shots`):
  - `duckdb.CatalogException` if `raw_shots`, `matches` or `homography` is missing.
  - `ShotsSchemaMismatchError` if an existing `shots` table with own rows lacks a column of `schema.sql` other than
    `type_detail` and `type_detail_pred` (its own rows cannot be read back in full). A table from before those two
    columns keeps its own rows, with both new columns NULL; a table with the current columns but older CHECKs is
    simply recreated. Own rows are written back by column name, and their `type_group` / `type_pred` are re-derived
    from `type_detail` / `type_detail_pred` where those are set. Retired labels in those two columns are rewritten
    first (`TYPE_DETAIL_BY_RETIRED_TYPE_DETAIL`: `return_net` -> `net`, `push` -> `drive`), so own rows labelled or
    predicted before the 8-label change pass the CHECKs.
  - `analytics.shot_types.UnmappedShotTypeError` if a shot type has no group.
  - `duckdb.IOException` if the database cannot be opened, e.g. its folder does not exist.

Build into a temp folder (never `data/scout.duckdb`):

```python
>>> import shutil, tempfile
>>> from pathlib import Path
>>> import duckdb
>>> from ingest.load_shuttleset import SHUTTLESET_SET_DIR, build_database
>>> from analytics.build_shots import *
>>> temp_dir = Path(tempfile.mkdtemp())
>>> database_path = temp_dir / "scout.duckdb"
>>> build_database(SHUTTLESET_SET_DIR, database_path)
>>> build_shots(database_path)
{'raw_shots': 36484, 'shuttleset_shots': 36483, 'duplicates_dropped': 1, 'own_shots': 0}
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute(
...     "SELECT type_group, COUNT(*) FROM shots GROUP BY type_group ORDER BY COUNT(*) DESC LIMIT 3"
... ).fetchall()
[('net', 11793), ('lift', 5631), ('drive', 4572)]
>>> connection.execute("SELECT roundscore_me, roundscore_opp FROM shots WHERE rally = 1 GROUP BY ALL").fetchall()
[(0, 0)]

```

Own rows survive a rebuild, with their zones re-derived from their coordinates (the second row's
stored `hit_zone` is wrong on purpose):

```python
>>> _ = connection.execute(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player) VALUES ('own', 'club_2026_10_20', 1, 1, 1, 'me')"
... )
>>> _ = connection.execute(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player, hit_x, hit_y, landing_x, landing_y, player_x, player_y, hit_zone, landing_zone, is_cross) "
...     "VALUES ('own', 'club_2026_10_20', 1, 1, 2, 'opp', 1.0, 1.0, 5.0, 12.0, 1.0, 1.0, 'front_forehand', 'front_forehand', FALSE)"
... )
>>> connection.close()
>>> summary = build_shots(database_path)
>>> summary["shuttleset_shots"], summary["own_shots"]
(36483, 2)
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute("SELECT source, COUNT(*) FROM shots GROUP BY source ORDER BY source").fetchall()
[('own', 2), ('shuttleset', 36483)]
>>> connection.execute("SELECT ball_round, hit_zone, landing_zone, is_cross FROM shots WHERE source = 'own' ORDER BY ball_round").fetchall()
[(1, None, None, None), (2, 'rear_backhand', 'rear_backhand', True)]

```

## `schema_columns()`, `shots_columns(connection)`, `schema_columns_and_constraints()`, `shots_constraints(connection)`

- `schema_columns()` returns `[(column_name, data_type), ...]` from running `schema.sql` in an
  in-memory database: the contract's columns in order.
- `shots_columns(connection)` returns the same for the `shots` table behind `connection`
  (an empty list if there is no `shots` table).
- `shots_constraints(connection)` returns the table's NOT NULL and CHECK constraints as
  `[(constraint_type, column_names_tuple, expression_or_None), ...]`, sorted.
- `schema_columns_and_constraints()` returns `(schema_columns(), constraints)` from `schema.sql`.

```python
>>> shots_columns(connection) == schema_columns()
True
>>> len(schema_columns()), schema_columns()[:3]
(30, [('source', 'VARCHAR'), ('match_id', 'VARCHAR'), ('set_no', 'INTEGER')])
>>> shots_constraints(connection) == schema_columns_and_constraints()[1]
True
>>> [constraint for constraint in shots_constraints(connection) if constraint[1] == ("hit_zone",)][0][2].count("_middle")
0

```

## `create_shots_table(connection)`

Runs `schema.sql` (`CREATE TABLE IF NOT EXISTS`), then checks the table's columns and
constraints. `CREATE TABLE IF NOT EXISTS` leaves an older table as it is, so this is what
catches a table built from an older `schema.sql`, for example one whose zone CHECK still
allows the earlier 9 zones with a middle column. `build_shots` recreates such a table itself.
Errors: `ShotsSchemaMismatchError` if the columns or the constraints differ from `schema.sql`.

## `ShotsSchemaMismatchError(existing_columns, expected_columns, existing_constraints=None, expected_constraints=None)`

A `RuntimeError` with `existing_columns`, `expected_columns`, `existing_constraints` and
`expected_constraints` attributes (constraints default to `[]`). The message lists the
constraints that differ. To move to a new schema, run `build_shots`: it recreates the table
and keeps the `own` rows (below). The error is for the other writers (`shots.predict`,
`vision.features`), which must not touch a stale table.

```python
>>> _ = connection.execute("ALTER TABLE shots ADD COLUMN speed DOUBLE")
>>> try:
...     create_shots_table(connection)
... except ShotsSchemaMismatchError as error:
...     error.existing_columns[-1], len(error.expected_columns)
(('speed', 'DOUBLE'), 30)
>>> _ = connection.execute("ALTER TABLE shots DROP COLUMN speed")

```

A table with the right columns but the older zone CHECK (the 9 zones with a middle column) is
also rejected:

```python
>>> nine_zone_schema_sql = SCHEMA_PATH.read_text().replace("'front_backhand', ", "'front_middle', 'front_backhand', ").replace("'mid_backhand', ", "'mid_middle', 'mid_backhand', ").replace("'rear_backhand'", "'rear_middle', 'rear_backhand'")
>>> old_connection = duckdb.connect(":memory:")
>>> _ = old_connection.execute(nine_zone_schema_sql)
>>> shots_columns(old_connection) == schema_columns()
True
>>> try:
...     create_shots_table(old_connection)
... except ShotsSchemaMismatchError as error:
...     sorted(set(error.existing_constraints) ^ set(error.expected_constraints))[0][1]
('hit_zone',)
>>> old_connection.close()

```

## `pixel_to_court_homography_by_match(connection)`

Returns `{match_id: 3x3 np.ndarray}`, each match's pixel -> court-meter homography,
from the `homography` table.

```python
>>> homographies = pixel_to_court_homography_by_match(connection)
>>> axelsen_ginting = homographies["Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"]
>>> from analytics.court import apply_homography
>>> (apply_homography(axelsen_ginting, [(289.4, 659.6), (864.2, 301.6)]).round(2) + 0.0).tolist()
[[0.0, 0.0], [6.1, 13.4]]

```

## `shuttleset_shots_frame(connection, left_handed_player_names)`

Returns the ShuttleSet rows ready to insert as a `pandas.DataFrame`: deduplicated,
converted to meters, normalized, with zones, `is_cross`, `type_group` and scores
before the rally. It uses three helpers, each also public:

- `positions_in_court_meters(shot_rows, homography_by_match)`: a DataFrame with
  `hit_x, hit_y, landing_x, ...` in meters (not yet normalized), same index as `shot_rows`.
- `geometry_columns(shot_rows, meters, left_handed_player_names)`: runs
  `court.shot_geometry` on every row; returns normalized positions, `hit_zone`, `landing_zone`, `is_cross`.
- `point_or_none(x, y)`: `(x, y)` as floats, or `None` if either is NaN.

```python
>>> frame = shuttleset_shots_frame(connection, LEFT_HANDED_PLAYER_NAMES)
>>> len(frame), bool((frame["player_y"].dropna() <= 6.7).all())
(36483, True)
>>> point_or_none(1, 2), point_or_none(float("nan"), 2)
((1.0, 2.0), None)
>>> connection.close()
>>> shutil.rmtree(temp_dir)

```
