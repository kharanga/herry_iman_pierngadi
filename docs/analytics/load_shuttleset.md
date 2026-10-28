# `ingest/load_shuttleset.py`

Loads the ShuttleSet CSVs into a DuckDB database: one row per shot in `raw_shots`,
one row per match in `matches` (with its frame rate), and one row per match in
`homography`. Run `ingest/fetch_shuttleset.sh` first (see `fetch_shuttleset.md`).

The database holds no Chinese: `raw_shots.type`, `raw_shots.lose_reason` and
`raw_shots.win_reason` are translated to English at ingest with the maps in
`ingest/label_translations.py` (see `label_translations.md`).

## Command

```
.venv/bin/python -m ingest.load_shuttleset [database_path]
```

- `database_path` (optional): where to write the database. Default `data/scout.duckdb`.
- Reads from `data/CoachAI-Projects/ShuttleSet/set` (`SHUTTLESET_SET_DIR`).
- Runs `build_database`, then `print_summary`. Run from the project root, as a module (`-m`),
  so `ingest.label_translations` can be imported.

Output on the full dataset:

```
raw_shots: 36484 shots, 44 matches, 104 sets
fps 25: 19 matches (estimates 25.00-25.00)
fps 30: 25 matches (estimates 29.97-30.00)
```

## Constants

| Name | Type | Value |
|---|---|---|
| `SHUTTLESET_SET_DIR` | `Path` | `data/CoachAI-Projects/ShuttleSet/set` |
| `DATABASE_PATH` | `Path` | `data/scout.duckdb` |
| `RAW_SHOTS_SQL`, `MATCHES_SQL`, `HOMOGRAPHY_SQL`, `FRAME_RATE_SQL`, `ATTACH_FPS_TO_MATCHES_SQL` | `str` | the SQL each build step runs |
| `LABEL_TRANSLATIONS_TABLE_SQL`, `INSERT_LABEL_TRANSLATION_SQL`, `RAW_SHOTS_COLUMN_NAMES_SQL`, `DROP_LABEL_TRANSLATIONS_SQL` | `str` | SQL for the temporary `label_translations` table used during the build |
| `UNMAPPED_LABELS_SQL_TEMPLATE`, `TRANSLATE_COLUMN_SQL_TEMPLATE` | `str` | per-column SQL; `{column_name}` is filled from `ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN` only |

## `build_database(set_dir, database_path)`

Builds (or rebuilds) `raw_shots`, `matches` and `homography` in one transaction.
Every table is `CREATE OR REPLACE`, so re-running gives identical tables.

Inputs:
- `set_dir` (`Path`): the ShuttleSet `set/` folder. Must contain `match.csv`,
  `homography.csv` and `<match_folder>/set{1,2,3}.csv`.
- `database_path` (`Path`): DuckDB file to write. Parent folders are created.

Output: `None`. The tables are written to `database_path`.

How it works:
- `match_id` and `set_no` are parsed from each CSV's path (`set/<match_id>/set<set_no>.csv`).
- `time` (`H:MM:SS` or `HH:MM:SS`) becomes `time_seconds`.
- `type`, `lose_reason` and `win_reason` are translated to English in the same
  transaction (`translate_raw_shots_to_english`). NULLs stay NULL; labels that are
  already English stay as they are. A column missing from the CSVs is skipped.
- `fps_estimate` is the least-squares slope of `frame_num` over `time_seconds`
  for all shots in a match; `fps` is that estimate rounded.

Errors:
- `ingest.label_translations.UnmappedLabelError` (a `ValueError`) if `type`, `lose_reason`
  or `win_reason` holds a label with no English translation. The message names the column
  and the label. The transaction is rolled back, so nothing silently becomes NULL.
- `duckdb.IOException` if no `set*.csv` files match under `set_dir`, or if
  `match.csv` / `homography.csv` is missing. The transaction is rolled back, so
  an existing database keeps its old tables; a new `database_path` is left as an
  empty database file.
- `duckdb.IOException` if `database_path` is already open for writing by another process.

Example (builds into a temp folder, never `data/scout.duckdb`):

```python
>>> import tempfile
>>> from pathlib import Path
>>> import duckdb
>>> from ingest.load_shuttleset import SHUTTLESET_SET_DIR, build_database, print_summary
>>> temp_dir = Path(tempfile.mkdtemp())
>>> database_path = temp_dir / "scout.duckdb"
>>> build_database(SHUTTLESET_SET_DIR, database_path)
>>> connection = duckdb.connect(str(database_path), read_only=True)
>>> connection.execute("SELECT COUNT(*) FROM raw_shots").fetchone()[0]
36484
>>> connection.execute(
...     "SELECT fps, ROUND(fps_estimate, 2) FROM matches WHERE match_id = ?",
...     ["Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"],
... ).fetchone()
(30, 30.0)
>>> connection.execute(
...     "SELECT raw_shots.type, COUNT(*) FROM raw_shots GROUP BY raw_shots.type ORDER BY COUNT(*) DESC LIMIT 3"
... ).fetchall()
[('net shot', 6290), ('lob', 5331), ('return net', 3620)]
>>> connection.execute(
...     "SELECT raw_shots.lose_reason, raw_shots.win_reason, COUNT(*) FROM raw_shots "
...     "GROUP BY raw_shots.lose_reason, raw_shots.win_reason ORDER BY COUNT(*) DESC LIMIT 3"
... ).fetchall()
[(None, None, 32975), ('opponent winner', 'winner', 1189), ('out', 'opponent out', 1048)]
>>> connection.close()

```

Missing input:

```python
>>> try:
...     build_database(temp_dir / "no_such_set_dir", temp_dir / "empty.duckdb")
... except duckdb.IOException as error:
...     print(str(error).split(" match the pattern")[0])
IO Error: No files found that

```

Unmapped label (a synthetic `set1.csv` with a shot type that is in no map):

```python
>>> from ingest.label_translations import UnmappedLabelError
>>> unmapped_set_dir = temp_dir / "unmapped_set"
>>> (unmapped_set_dir / "Synthetic_Match").mkdir(parents=True)
>>> _ = (unmapped_set_dir / "Synthetic_Match" / "set1.csv").write_text(
...     "rally,ball_round,time,frame_num,player,type,lose_reason,win_reason\n"
...     "1,1,0:00:01,25,A,殺球,,\n"
...     "1,2,0:00:02,50,B,飛球,出界,對手出界\n"
... )
>>> _ = (unmapped_set_dir / "match.csv").write_text("id,video,set\n1,Synthetic_Match,1\n")
>>> _ = (unmapped_set_dir / "homography.csv").write_text(
...     'id,video,homography_matrix\n1,Synthetic_Match,"[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]"\n'
... )
>>> try:
...     build_database(unmapped_set_dir, temp_dir / "unmapped.duckdb")
... except UnmappedLabelError as error:
...     print(error)
raw_shots.type has a label with no English translation: '飛球'

```

## `translate_raw_shots_to_english(connection)`

Translates `raw_shots.type`, `raw_shots.lose_reason` and `raw_shots.win_reason` in place.
Called by `build_database` inside its transaction, right after `raw_shots` is created.
Loads every accepted label into a temporary `label_translations` table
(`load_label_translations`), then for each of the three columns present in `raw_shots`
(`translated_columns_present`) checks for unmapped labels (`fail_on_unmapped_labels`)
and replaces each non-NULL value with its English label. The temporary table is dropped at the end.

Inputs:
- `connection` (`duckdb.DuckDBPyConnection`): an open, writable connection with a `raw_shots` table.

Output: `None`. `raw_shots` is updated; its row count, row order and other columns do not change.

Errors:
- `UnmappedLabelError` from `fail_on_unmapped_labels` for the first unmapped label
  (alphabetically) in the first column that has one, checked in the order `type`, `lose_reason`, `win_reason`.

```python
>>> from ingest.load_shuttleset import translate_raw_shots_to_english
>>> memory_connection = duckdb.connect()
>>> _ = memory_connection.execute("CREATE TABLE raw_shots (ball_round INTEGER, type VARCHAR, lose_reason VARCHAR)")
>>> _ = memory_connection.execute("INSERT INTO raw_shots VALUES (1, '挑球', NULL), (2, '過渡切球', '掛網')")
>>> translate_raw_shots_to_english(memory_connection)
>>> memory_connection.execute("SELECT * FROM raw_shots ORDER BY ball_round").fetchall()
[(1, 'lob', None), (2, 'passive drop', 'net')]
>>> memory_connection.close()

```

## `print_summary(database_path)`

Prints shot, match and set counts from `raw_shots`, then how many matches have
each `fps` with the range of their `fps_estimate`.

Inputs:
- `database_path` (`Path`): a database built by `build_database`. Opened read-only.

Output: `None`. Prints one line for `raw_shots` and one line per distinct `fps`.

Errors:
- `duckdb.IOException` if `database_path` does not exist.
- `duckdb.CatalogException` if the database has no `raw_shots` or `matches` table.

Example (uses the database built above):

```python
>>> print_summary(database_path)
raw_shots: 36484 shots, 44 matches, 104 sets
fps 25: 19 matches (estimates 25.00-25.00)
fps 30: 25 matches (estimates 29.97-30.00)

```

## Tables

### `raw_shots` (36,484 rows)
One row per shot, every column of `set/*/set*.csv` plus `match_id`, `set_no`, `time_seconds`.
Ordered by `match_id, set_no, rally, ball_round`. Coordinates are pixels in a 1280x720 frame.

| Column | Type | Notes |
|---|---|---|
| match_id | VARCHAR | match folder name (from the path) |
| set_no | INTEGER | 1-3 (from the file name) |
| frame_num | INTEGER | hit frame |
| time_seconds | INTEGER | `time` in seconds |
| rally | BIGINT | |
| ball_round | DOUBLE | shot order within rally |
| time | VARCHAR | original `H:MM:SS` text |
| roundscore_A | BIGINT | |
| roundscore_B | BIGINT | |
| player | VARCHAR | `A` or `B` |
| server | BIGINT | |
| type | VARCHAR | shot type, English label (`net shot`, `lob`, ..., `unknown`; see `label_translations.md`) |
| aroundhead | DOUBLE | |
| backhand | DOUBLE | |
| hit_height | DOUBLE | |
| hit_area | BIGINT | |
| hit_x | DOUBLE | hitter's feet, pixels |
| hit_y | DOUBLE | hitter's feet, pixels |
| landing_height | DOUBLE | |
| landing_area | BIGINT | |
| landing_x | DOUBLE | pixels |
| landing_y | DOUBLE | pixels |
| lose_reason | VARCHAR | why the rally's last shot lost, English (`out`, `net`, ...); only set on a rally's last shot, and not always: 3,508 of 3,683 rallies have one (3,509 rows, because one rally's last shot appears twice in the source CSV) |
| win_reason | VARCHAR | the same event from the winner's side, English (`winner`, `opponent out`, ...); set on the same rows as `lose_reason` |
| getpoint_player | VARCHAR | |
| flaw | BIGINT | |
| player_location_area | BIGINT | |
| player_location_x | DOUBLE | pixels |
| player_location_y | DOUBLE | pixels |
| opponent_location_area | BIGINT | |
| opponent_location_x | DOUBLE | pixels |
| opponent_location_y | DOUBLE | pixels |
| db | BIGINT | |

### `matches` (44 rows)
`match.csv` plus `match_id` and the frame-rate columns. Ordered by `id`.

| Column | Type | Notes |
|---|---|---|
| match_id | VARCHAR | same as `video` |
| id | BIGINT | |
| video | VARCHAR | match folder name |
| tournament | VARCHAR | |
| round | VARCHAR | |
| year | BIGINT | |
| month | BIGINT | |
| day | BIGINT | |
| set | BIGINT | number of sets played |
| duration | BIGINT | |
| winner | VARCHAR | |
| loser | VARCHAR | |
| downcourt | BIGINT | which player starts on the near side |
| url | VARCHAR | YouTube link |
| fps_estimate | DOUBLE | slope of `frame_num` over `time_seconds` |
| fps_fit_r2 | DOUBLE | R² of that fit |
| fps | INTEGER | `fps_estimate` rounded (25 or 30) |

### `homography` (44 rows)
`homography.csv` plus `match_id`, with the matrix parsed. Ordered by `id`.

| Column | Type | Notes |
|---|---|---|
| match_id | VARCHAR | same as `video` |
| homography_matrix | DOUBLE[3][3] | pixel -> court mapping |
| id | BIGINT | |
| video | VARCHAR | match folder name |
| upleft_x | DOUBLE | court corner, pixels |
| upright_x | DOUBLE | |
| downleft_x | DOUBLE | |
| downright_x | DOUBLE | |
| upleft_y | DOUBLE | |
| upright_y | DOUBLE | |
| downleft_y | DOUBLE | |
| downright_y | DOUBLE | |

The tables above match the code:

```python
>>> connection = duckdb.connect(str(database_path), read_only=True)
>>> [len(connection.execute(f"DESCRIBE {table}").fetchall()) for table in ("raw_shots", "matches", "homography")]
[33, 17, 12]
>>> connection.execute("DESCRIBE homography").fetchall()[1][:2]
('homography_matrix', 'DOUBLE[3][3]')
>>> connection.close()
>>> import shutil
>>> shutil.rmtree(temp_dir)

```
