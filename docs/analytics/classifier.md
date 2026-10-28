# `shots/`: shot-type classifier

Predicts a shot's `type_group` (serve, clear, drop, smash, net, lift, drive) from geometry
and timing only, so the same model runs on my own footage, where nobody labels shot types.

- `shots/features.py`: turns `shots` rows into model features.
- `shots/train.py`: trains a gradient-boosted tree (`HistGradientBoostingClassifier`)
  on ShuttleSet, evaluates it on held-out matches, saves the model under `data/models/`.
- `shots/predict.py`: fills `shots.type_pred` and `shots.type_conf` for every row that has a hitter
  point and a landing point, any source.
- `shots/compare_features.py`: re-runs the same match folds for several feature sets (the
  feature-set table in the results doc).
- `shots/results_tables.py`: prints the results-doc tables from the saved JSON, so the doc and
  the code can't drift.

Measured accuracy and the per-group breakdown are in `docs/results/classifier.md`.

## Commands

```
.venv/bin/python -m shots.train [database_path] [model_path]
.venv/bin/python -m shots.predict [database_path] [model_path]
.venv/bin/python -m shots.compare_features [database_path] [comparison_path]
.venv/bin/python -m shots.results_tables [model_path] [comparison_path]
```

Defaults: `data/scout.duckdb` and `data/models/shot_type_classifier.joblib`. Training also
writes `data/models/shot_type_classifier_report.json` (every number in the results doc).
`train` opens the database read-only; `predict` changes only `type_pred` and `type_conf`.

## Features

All positions are in court meters, normalized so the hitter is on the near half
(see `court.md`). Each feature can be produced by the tracking pipeline from my own video:

| Feature | Meaning | Available from tracking because |
|---|---|---|
| `hit_x`, `hit_y` | hitter's feet at the hit (the hit point, else the hitter's position) | player tracking gives the hitter's feet at the hit frame |
| `landing_x`, `landing_y` | where the shot ended (where the receiver reached it) | the next hit's position, or where shuttle tracking loses the shuttle |
| `receiver_x`, `receiver_y` | receiver's position at the hit | player tracking gives the other player at the hit frame |
| `shot_dx`, `shot_dy` | landing minus hit, sideways and lengthways | arithmetic on the two points above |
| `shot_distance` | length of that vector | arithmetic |
| `shot_lateral_distance` | sideways distance, absolute | arithmetic |
| `is_cross` | cross-court (see `court.md`) | derived from the hit and landing points |
| `ball_round` | shot number in the rally (1 = serve) | hit detection counts hits within a rally |
| `flight_seconds` | time until the next hit (NaN on a rally's last shot) | hit detection gives hit frames; `fps` is known |
| `seconds_since_previous_hit` | time since the previous hit (NaN on the first shot) | same |
| `previous_shot_distance` | length of the previous shot | previous shot's hit and landing points |
| `previous_hit_y` | how deep the previous hitter stood, in their own frame | previous shot's hit point |

Not used, because only a human labeller knows them: `type_en` itself, `backhand`,
and ShuttleSet's `hit_height`, `aroundhead`, `landing_height`, `hit_area`. Not used, so the model
doesn't learn who is playing: `player`, `match_id`, scores. Not used: the previous shot's
true group. On my footage only a *predicted* previous group would exist, and the true one
adds only 0.3 points of accuracy (see the feature-set table in the results).

Missing values stay NaN; the gradient-boosted tree handles them natively.

**Which shots get a prediction.** Only shots with both a hitter point (`hit_x/hit_y`, or the hitter's
position) and a landing point. Without them there is nothing to classify: on ShuttleSet, the 7 labelled
shots that have only one of the two were predicted right 1 time in 7. Other shots keep `type_pred` and
`type_conf` NULL. Training and evaluation use the same rule, so the reported accuracy is for exactly
the shots that get a prediction.

## `shots/features.py`

### Constants
- `FEATURE_NAMES` (`tuple[str, ...]`): the 16 features above, in model column order.
- `SHOT_FEATURE_ROWS_SQL`: every `shots` row with its `rowid` (`row_id`) and the previous and
  next shot in the same rally (`LAG`/`LEAD` over `source, match_id, set_no, rally`, ordered by
  `ball_round, frame_num`).

### `load_shot_feature_rows(connection)`
Runs `SHOT_FEATURE_ROWS_SQL`. Returns a `pandas.DataFrame`, one row per `shots` row.
Errors: `duckdb.CatalogException` if there is no `shots` table.

### `shot_features(shot_rows)`
- `shot_rows` (`DataFrame` with the columns of `SHOT_FEATURE_ROWS_SQL`).
- Returns a `DataFrame` with exactly `FEATURE_NAMES` as columns, same index, floats (NaN if missing).
- Errors: `KeyError` if a needed column is missing.

### `hitter_point_columns(shot_rows)`
Returns `(x, y)` arrays: the hit point when present and on the near half, else the hitter's
position. This is the same rule as `analytics.court.hitter_point_for_zone`.

### `has_hitter_and_landing_point(features)`
`True` per row when `hit_x`, `hit_y`, `landing_x` and `landing_y` are all present in a `shot_features`
result. This decides which shots are trained on and predicted.

### `seconds_between(later_frames, earlier_frames, fps)`
`(later - earlier) / fps` as a float array, NaN where any value is missing.

A clear followed by a reply 45 frames later at 30 fps, then a serve with no hit point:

```python
>>> import math
>>> import pandas as pd
>>> from shots.features import FEATURE_NAMES, shot_features
>>> shot_rows = pd.DataFrame({
...     "ball_round": [2, 1], "frame_num": [1000, 2000], "fps": [30.0, 30.0],
...     "hit_x": [1.0, None], "hit_y": [1.0, None], "player_x": [1.1, 3.4], "player_y": [1.2, 4.5],
...     "landing_x": [1.5, 1.5], "landing_y": [12.8, 8.9], "opponent_x": [3.0, 2.5], "opponent_y": [9.5, 8.6],
...     "is_cross": [False, True], "previous_frame_num": [955, None], "next_frame_num": [1045, None],
...     "previous_hit_x": [2.0, None], "previous_hit_y": [4.0, None],
...     "previous_landing_x": [5.1, None], "previous_landing_y": [12.4, None],
... })
>>> features = shot_features(shot_rows)
>>> list(features.columns) == list(FEATURE_NAMES)
True
>>> features.loc[0, ["hit_x", "shot_distance", "flight_seconds", "seconds_since_previous_hit", "previous_shot_distance"]].round(2).tolist()
[1.0, 11.81, 1.5, 1.5, 8.95]
>>> features.loc[1, ["hit_x", "hit_y", "is_cross"]].tolist(), math.isnan(features.loc[1, "flight_seconds"])
([3.4, 4.5, 1.0], True)
>>> from shots.features import has_hitter_and_landing_point
>>> has_hitter_and_landing_point(features).tolist()
[True, True]

```

## `shots/train.py`

### Constants

| Name | Value |
|---|---|
| `DATABASE_PATH` | `data/scout.duckdb` |
| `MODEL_PATH` | `data/models/shot_type_classifier.joblib` |
| `TRAINED_GROUPS` | the 7 groups except `unknown` |
| `FOLD_COUNT` | 5 |
| `RANDOM_SEED` | 0 |
| `PERMUTATION_REPEATS` | 3 |

### `train_classifier(database_path=DATABASE_PATH, model_path=MODEL_PATH, fold_count=5, random_seed=0, permutation_repeats=3)`

1. Takes the ShuttleSet rows whose `type_group` is not `unknown` (shots typed `unknown` are left
   out of training and evaluation: their label carries no group) and that have a hitter point
   and a landing point.
2. Grouped cross-validation over **matches** (`GroupKFold`): each fold's test matches never
   appear in its training data, and every shot gets a prediction from a model that never
   saw its match. Raises `RuntimeError` if a fold ever had a match on both sides.
3. Trains the final model on all labelled matches and saves it.

- Returns the report `dict`: `folds` (per fold: test matches, shots, accuracy, majority group and its
  accuracy), `accuracy_mean`, `accuracy_std`, `majority_accuracy_mean`, `out_of_fold_accuracy`,
  `per_group` (precision, recall, f1, support), `confusion_matrix`, `permutation_importance`
  (mean accuracy drop on the test folds, largest first), `training_shots`, `training_matches`,
  `group_counts`, `model_path`. The same dict is written as JSON next to the model.
- The saved model is a joblib `dict`: `classifier`, `feature_names`, `groups`, `sklearn_version`,
  `training_shots`, `training_matches`.
- Errors: `duckdb.IOException` if the database does not exist; `duckdb.CatalogException` if it
  has no `shots` table; `ValueError` from scikit-learn if there are fewer matches than folds,
  or if a feature is missing on every training shot.

### Helpers
- `labelled_shuttleset_rows(connection)`: the training rows (columns of `SHOT_FEATURE_ROWS_SQL`).
- `labelled_shuttleset_shots(connection)`: `(features, labels, match_ids)` for training.
- `grouped_cross_validation(features, labels, match_ids, fold_count, random_seed, permutation_repeats)`: the evaluation part, returns the report without the training fields.
- `new_classifier(random_seed)`: `HistGradientBoostingClassifier(random_state=random_seed)`, default settings.
- `majority_group(labels)`: most common label (ties: alphabetical first).
- `per_group_scores(true_groups, predicted_groups)`: precision/recall/f1/support per group.
- `save_model(classifier, model_path, training_shots, training_matches)`, `report_path_for(model_path)`, `summary_lines(report)`.

## `shots/predict.py`

### `predict_shot_types(database_path=DATABASE_PATH, model_path=MODEL_PATH)`
Sets `type_pred` (the most likely group) and `type_conf` (its probability, 0-1) on every `shots`
row that has a hitter point and a landing point: both sources, and `unknown` shots too. Every other
row gets NULL in both columns, which also clears an older prediction. Matched by `rowid`, in one
transaction; no other column changes.

- Returns the number of rows that got a prediction (`int`).
- Errors: `FileNotFoundError` if the model file is missing; `ModelFeatureMismatchError` if the model
  was trained on a different `FEATURE_NAMES` (retrain); `duckdb.IOException` / `CatalogException` for a
  missing database or table. On any error the transaction rolls back.

### Helpers
- `load_model(model_path)`: loads and checks the joblib `dict`.
- `predictions_for(model, shot_rows)`: `DataFrame` of `row_id`, `type_pred`, `type_conf`, with
  `None` / `<NA>` for rows without a hitter point and a landing point.
- `ModelFeatureMismatchError(model_feature_names)`: a `RuntimeError` with `model_feature_names`.

## `shots/compare_features.py`

`compare_feature_sets(database_path=DATABASE_PATH, comparison_path=COMPARISON_PATH, fold_count=5, random_seed=0)`
re-runs the same match folds and classifier as training for each entry of `FEATURE_SETS`: geometry
only, + timing, the shipped `FEATURE_NAMES`, and the shipped set plus the previous shot's **true**
group (a categorical feature, `previous_true_group`, from `PREVIOUS_TRUE_GROUP_SQL`). That last set
is an experiment only: on my own footage the true group is never known.

- Returns `{set_name: {"feature_names", "fold_accuracies", "accuracy_mean"}}` and writes it as JSON to
  `comparison_path` (default `data/models/feature_set_comparison.json`).
- Errors: as `train_classifier`.
- Helpers: `previous_true_group_codes(connection, row_ids)` (group codes 0-7, NaN on a rally's first shot),
  `grouped_accuracy(features, labels, match_ids, fold_count, random_seed)` (accuracy per fold).

## `shots/results_tables.py`

Prints the markdown tables of `docs/results/classifier.md` from the training report and the comparison
JSON: `accuracy_table(report)`, `per_group_table(report)` (sorted by F1), `confusion_matrix_table(report)`,
`importance_table(report)` (3 decimals), `feature_set_table(comparison)`, and `results_tables(report, comparison)`
for all of them. Every function returns a `str`. Errors: `KeyError` for a report without the expected fields.

```python
>>> from shots.results_tables import importance_table
>>> print(importance_table({"permutation_importance": {"landing_y": 0.28512, "flight_seconds": 0.13846, "shot_dy": 0.06248}}))
| Feature | Accuracy drop |
|---|---|
| landing_y | 0.285 |
| flight_seconds | 0.138 |
| shot_dy | 0.062 |

```

## Example: train and predict on a small synthetic table

Six matches of clear -> net shot -> smash rallies (no ShuttleSet needed):

```python
>>> import shutil, tempfile
>>> from pathlib import Path
>>> import duckdb
>>> from analytics.build_shots import SCHEMA_PATH
>>> from shots.train import train_classifier
>>> from shots.predict import ModelFeatureMismatchError, load_model, predict_shot_types
>>> temp_dir = Path(tempfile.mkdtemp())
>>> database_path = temp_dir / "scout.duckdb"
>>> connection = duckdb.connect(str(database_path))
>>> _ = connection.execute(SCHEMA_PATH.read_text())
>>> shot_shapes = {"clear": (1.0, 1.0, 1.5, 12.8), "net": (3.0, 5.5, 3.2, 7.4), "smash": (4.5, 1.5, 4.0, 10.0)}
>>> rows = []
>>> for match_number in range(6):
...     for rally in range(1, 21):
...         for ball_round, group in enumerate(["clear", "net", "smash"], start=1):
...             hit_x, hit_y, landing_x, landing_y = shot_shapes[group]
...             jitter = ((match_number * 7 + rally * 3 + ball_round) % 10) / 20
...             rows.append(("shuttleset", f"match_{match_number}", 1, rally, ball_round, rally * 1000 + ball_round * 60,
...                          30.0, "A", group, group, hit_x + jitter, hit_y, landing_x - jitter, landing_y,
...                          hit_x + jitter, hit_y, landing_x, landing_y, group == "smash"))
>>> _ = connection.executemany(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, type_en, type_group, "
...     "hit_x, hit_y, landing_x, landing_y, player_x, player_y, opponent_x, opponent_y, is_cross) "
...     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
>>> _ = connection.execute(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player, hit_x, hit_y, landing_x, landing_y) "
...     "VALUES ('own', 'club_2026_10_20', 1, 1, 2, 'me', 1.1, 1.0, 1.4, 12.7), "
...     "('own', 'club_2026_10_20', 1, 2, 1, 'me', NULL, NULL, NULL, NULL)")
>>> connection.close()
>>> report = train_classifier(database_path, temp_dir / "models" / "classifier.joblib", fold_count=3, permutation_repeats=1)
>>> report["training_shots"], report["training_matches"], report["accuracy_mean"], round(report["majority_accuracy_mean"], 2)
(360, 6, 1.0, 0.33)
>>> sorted(load_model(temp_dir / "models" / "classifier.joblib")["groups"])
['clear', 'net', 'smash']
>>> predict_shot_types(database_path, temp_dir / "models" / "classifier.joblib")
361
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute("SELECT rally, type_pred, type_conf IS NULL FROM shots WHERE source = 'own' ORDER BY rally").fetchall()
[(1, 'clear', False), (2, None, True)]
>>> connection.execute("SELECT COUNT(*) FROM shots WHERE type_pred = type_group").fetchone()[0]
360
>>> connection.close()
>>> try:
...     predict_shot_types(database_path, temp_dir / "no_model.joblib")
... except FileNotFoundError as error:
...     str(error).endswith("no_model.joblib (run python -m shots.train first)")
True
>>> shutil.rmtree(temp_dir)

```
