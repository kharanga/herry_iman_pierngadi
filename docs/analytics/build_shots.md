# `analytics/build_shots.py` and `analytics/schema.sql`

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
| type_group | VARCHAR | yes | `serve, clear, drop, smash, net, lift, drive, unknown` (see `shot_types.md`) |
| type_pred | VARCHAR | yes | classifier's predicted group; NULL until the classifier runs |
| type_conf | DOUBLE | yes | classifier confidence; NULL until the classifier runs |
| backhand | BOOLEAN | yes | hit on the backhand |
| hit_x, hit_y | DOUBLE | yes | hitter's feet at the hit, court meters, normalized |
| landing_x, landing_y | DOUBLE | yes | where the shot ended, court meters, normalized (see below) |
| player_x, player_y | DOUBLE | yes | hitter's position at the hit, normalized |
| opponent_x, opponent_y | DOUBLE | yes | receiver's position at the hit, normalized |
| hit_zone | VARCHAR | yes | zone of the hit, from the hitter's view: one of the 9 zones below |
| landing_zone | VARCHAR | yes | zone of the landing, from the receiver's view: one of the 9 zones, or `not_over_net` |
| is_cross | BOOLEAN | yes | cross-court (see `court.md`) |
| roundscore_me | INTEGER | yes | score of `A` (ShuttleSet) or `me` (own) **before** the rally |
| roundscore_opp | INTEGER | yes | score of `B` or `opp` before the rally |
| getpoint_player | VARCHAR | yes | rally winner, on the rally's last shot only |
| lose_reason | VARCHAR | yes | why the rally ended, on the last shot only |

CHECK constraints reject any other value of `source`, `player`, `getpoint_player`,
`set_no`, `type_group`, `type_pred`, `hit_zone` and `landing_zone`.

Zones (`analytics.court.ZONE_NAMES`): each half is 3 depths x 3 sides = 9 zones.

| Depth (from the net) | Side | Zones |
|---|---|---|
| front 0-2.23 m, mid 2.23-4.47 m, rear beyond | forehand / middle / backhand | `front_forehand`, `front_middle`, `front_backhand`, `mid_forehand`, `mid_middle`, `mid_backhand`, `rear_forehand`, `rear_middle`, `rear_backhand` |

The middle column is `MIDDLE_COLUMN_WIDTH_M` = 1.0 m wide, centred on the centre line
(x 2.55-3.55 m, edges count as middle). Forehand and backhand swap for a left-handed
player; middle does not. `landing_zone` can also be `not_over_net`. Details in `court.md`.

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

Creates `shots` if needed and replaces its ShuttleSet rows, in one transaction.

- `database_path` (`Path`): a database with `raw_shots`, `matches`, `homography`.
- `left_handed_player_names` (iterable of `str`): player names as in `matches.winner` / `matches.loser`.
- Returns a `dict`: `raw_shots` (row count), `shuttleset_shots` (rows written),
  `duplicates_dropped` (the difference).
- Errors (the transaction rolls back, so the database keeps its previous `shots`):
  - `duckdb.CatalogException` if `raw_shots`, `matches` or `homography` is missing.
  - `ShotsSchemaMismatchError` if an existing `shots` table has different columns or types than `schema.sql`.
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
{'raw_shots': 36484, 'shuttleset_shots': 36483, 'duplicates_dropped': 1}
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute(
...     "SELECT type_group, COUNT(*) FROM shots GROUP BY type_group ORDER BY COUNT(*) DESC LIMIT 3"
... ).fetchall()
[('net', 11793), ('lift', 5631), ('drive', 4572)]
>>> connection.execute("SELECT roundscore_me, roundscore_opp FROM shots WHERE rally = 1 GROUP BY ALL").fetchall()
[(0, 0)]

```

Own rows survive a rebuild:

```python
>>> _ = connection.execute(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player) VALUES ('own', 'club_2026_10_20', 1, 1, 1, 'me')"
... )
>>> connection.close()
>>> build_shots(database_path)["shuttleset_shots"]
36483
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute("SELECT source, COUNT(*) FROM shots GROUP BY source ORDER BY source").fetchall()
[('own', 1), ('shuttleset', 36483)]

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
(28, [('source', 'VARCHAR'), ('match_id', 'VARCHAR'), ('set_no', 'INTEGER')])
>>> shots_constraints(connection) == schema_columns_and_constraints()[1]
True
>>> [constraint for constraint in shots_constraints(connection) if constraint[1] == ("hit_zone",)][0][2].count("_middle")
3

```

## `create_shots_table(connection)`

Runs `schema.sql` (`CREATE TABLE IF NOT EXISTS`), then checks the table's columns and
constraints. `CREATE TABLE IF NOT EXISTS` leaves an older table as it is, so this is what
catches a table built from an older `schema.sql`, for example one whose zone CHECK still
allows only the old 6 zones.
Errors: `ShotsSchemaMismatchError` if the columns or the constraints differ from `schema.sql`.

## `ShotsSchemaMismatchError(existing_columns, expected_columns, existing_constraints=None, expected_constraints=None)`

A `RuntimeError` with `existing_columns`, `expected_columns`, `existing_constraints` and
`expected_constraints` attributes (constraints default to `[]`). The message lists the
constraints that differ. To move to a new schema, drop the `shots` table and run
`build_shots` again. Copy any `own` rows out first and re-insert them afterwards:
dropping the table deletes them.

```python
>>> _ = connection.execute("ALTER TABLE shots ADD COLUMN speed DOUBLE")
>>> try:
...     create_shots_table(connection)
... except ShotsSchemaMismatchError as error:
...     error.existing_columns[-1], len(error.expected_columns)
(('speed', 'DOUBLE'), 28)
>>> _ = connection.execute("ALTER TABLE shots DROP COLUMN speed")

```

A table with the right columns but an older zone CHECK (the 6 zones without middle) is
also rejected:

```python
>>> six_zone_schema_sql = SCHEMA_PATH.read_text().replace("'front_middle', ", "").replace("'mid_middle', ", "").replace("'rear_middle', ", "")
>>> old_connection = duckdb.connect(":memory:")
>>> _ = old_connection.execute(six_zone_schema_sql)
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
