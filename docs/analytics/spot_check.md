# `analytics/spot_check.py`

Examples need: data/CoachAI-Projects/ShuttleSet/set (run ./ingest/fetch_shuttleset.sh)

Draws a few random ShuttleSet shots from the `shots` table on a top-down court
diagram, to check by eye that positions, normalization, zones and `is_cross` make sense.

## Command

```
.venv/bin/python -m analytics.spot_check [database_path] [output_path]
```

- `database_path` (optional): default `data/scout.duckdb`, must contain `shots`.
- `output_path` (optional): default `data/checks/shots/random_shots.png`.
- Prints one line per shot and the saved path.

The image shows the court in meters with the hitter's half at the bottom: white lines
are the court lines, thin gray lines the depth bands (front/mid/rear) and the
edges of the middle column, the black line the net. Each shot is a numbered yellow dot (hit)
with a red arrow to its landing; blue dots are the receivers at the hit. The legend
lists each shot's type, group, hitter, zones, straight/cross and source rally.

## Constants

| Name | Value |
|---|---|
| `DATABASE_PATH` | `data/scout.duckdb` |
| `SPOT_CHECK_PATH` | `data/checks/shots/random_shots.png` |
| `SPOT_CHECK_SEED` | 7 |
| `SPOT_CHECK_SHOT_COUNT` | 10 |
| `PIXELS_PER_METER`, `MARGIN_PIXELS`, `LEGEND_WIDTH_PIXELS` | 60, 50, 720 (image layout) |
| `SINGLES_SIDELINE_INSET_M`, `SHORT_SERVICE_LINE_FROM_NET_M`, `DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M` | 0.46, 1.98, 0.76 |
| `SPOT_CHECK_CANDIDATES_SQL` | ShuttleSet shots that have both a hit and a landing point |

## `draw_spot_check(database_path, output_path=SPOT_CHECK_PATH, shot_count=10, seed=7)`

- `database_path` (`Path`), `output_path` (`Path`, folders are created), `shot_count` (`int`), `seed` (`int`).
- Returns the drawn shots as a `list[dict]` (column name -> value), in table order.
  The same seed gives the same shots for the same table.
- Errors: `duckdb.IOException` if the database does not exist; `duckdb.CatalogException`
  if it has no `shots` table; `ValueError` if fewer than `shot_count` shots qualify.

## Helpers

- `random_shots(connection, shot_count, seed)`: the sampled shots, as returned above.
- `court_to_image(point_m)`: court meters -> image pixel `(column, row)`.
- `court_lines_m()`, `zone_boundaries_m()`: line segments in meters (`[(start, end), ...]`).
- `blank_court_image()`: the empty diagram as a BGR `np.ndarray`.
- `draw_shot(image, shot_number, shot)`, `draw_legend(image, shots)`: draw onto `image` in place.
- `shot_legend_line(shot_number, shot)`, `shot_source_line(shot)`: legend text.

```python
>>> from analytics.spot_check import blank_court_image, court_to_image, shot_legend_line
>>> court_to_image((0.0, 0.0)), court_to_image((6.1, 13.4))
((50, 854), (416, 50))
>>> blank_court_image().shape
(904, 1186, 3)
>>> shot_legend_line(1, {"type_en": "clear", "type_group": "clear", "player": "B",
...                      "hit_zone": "rear_backhand", "landing_zone": "rear_forehand", "is_cross": False})
'1. clear (clear) by B: rear_backhand -> rear_forehand, straight'

```

On a database built in a temp folder:

```python
>>> import shutil, tempfile
>>> from pathlib import Path
>>> from ingest.load_shuttleset import SHUTTLESET_SET_DIR, build_database
>>> from analytics.build_shots import build_shots
>>> from analytics.spot_check import draw_spot_check
>>> temp_dir = Path(tempfile.mkdtemp())
>>> build_database(SHUTTLESET_SET_DIR, temp_dir / "scout.duckdb")
>>> _ = build_shots(temp_dir / "scout.duckdb")
>>> shots = draw_spot_check(temp_dir / "scout.duckdb", temp_dir / "checks" / "random_shots.png")
>>> len(shots), (temp_dir / "checks" / "random_shots.png").exists()
(10, True)
>>> shots[0]["type_en"], shots[0]["hit_zone"], shots[0]["landing_zone"], shots[0]["is_cross"]
('clear', 'rear_backhand', 'rear_middle', True)
>>> shutil.rmtree(temp_dir)

```
