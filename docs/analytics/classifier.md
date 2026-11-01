# `shots/`: shot-type classifier

Predicts a shot's label `type_detail` (`serve_short`, `serve_long`, `net`, `drive`,
`clear`, `drop`, `smash`, `lift`; see `shot_types.md`) and with it its `type_group` (serve, clear, drop,
smash, net, lift, drive) from the two
players' feet and the hit frames only, so the same model runs on my own footage, where the
pipeline tracks the players (not the shuttle) and nobody labels shot types.

- `shots/features.py`: turns `shots` rows into model features. Two feature sets share the
  same 16 column names: **`feet`** (the shipped one: hitter's feet, the reply hitter's feet
  as the landing) and `contact` (ShuttleSet's labelled hit and landing points, kept for comparison).
- `shots/jitter.py`: shifts every hit frame by a random offset, to measure what imprecise hit
  timing costs.
- `shots/train.py`: trains a gradient-boosted tree (`HistGradientBoostingClassifier`)
  on ShuttleSet, evaluates it on held-out matches (exact frames and jittered frames), saves the
  model under `data/models/`.
- `shots/predict.py`: fills `shots.type_detail_pred`, `shots.type_pred` and `shots.type_conf` for every row that has a hitter
  point and a landing point, any source.
- `shots/compare_features.py`: re-runs the same match folds for several feature subsets (the
  feature-set table in the results doc).
- `shots/compare_hit_points.py`: the `contact` and `feet` feature sets side by side on the same
  match folds, with the jitter table.
- `shots/fine_types.py`: the same feet model and match folds trained on the 18 ShuttleSet types and on
  the 7 groups with one type split off at a time (a measurement only; the shipped model keeps the groups).
- `shots/results_tables.py`: prints the results-doc tables from the saved JSON, so the doc and
  the code can't drift.

Measured accuracy and the per-group breakdown are in `docs/results/classifier_feet.md`
(the shipped feet model) and `docs/results/classifier.md` (the earlier contact-point model).

## Commands

```
.venv/bin/python -m shots.train [database_path] [model_path]
.venv/bin/python -m shots.predict [database_path] [model_path]
.venv/bin/python -m shots.compare_features [database_path] [comparison_path]
.venv/bin/python -m shots.compare_hit_points [database_path] [comparison_path]
.venv/bin/python -m shots.fine_types [database_path] [comparison_path]
.venv/bin/python -m shots.results_tables [model_path] [comparison_path] [database_path] [hit_point_comparison_path]
```

Defaults: `data/scout.duckdb` and `data/models/shot_type_classifier.joblib`. Training also
writes `data/models/shot_type_classifier_report.json` (every number in the results doc).
`train` opens the database read-only; `predict` changes only `type_detail_pred`, `type_pred` and `type_conf`.

## Features

All positions are in court meters, normalized so the hitter is on the near half
(see `court.md`). The shipped `feet` set uses only what player tracking gives:

| Feature | Meaning (`feet` set) | Available from tracking because |
|---|---|---|
| `hit_x`, `hit_y` | the hitter's feet at the hit (`player_x`, `player_y`) | player tracking gives the hitter's feet at the hit frame |
| `landing_x`, `landing_y` | the **next** shot's hitter's feet, turned into this hitter's frame | the reply hitter's feet at the next hit frame |
| `receiver_x`, `receiver_y` | the opponent's feet at the hit (`opponent_x`, `opponent_y`) | player tracking gives the other player at the hit frame |
| `shot_dx`, `shot_dy` | landing minus hit, sideways and lengthways | arithmetic on the two points above |
| `shot_distance` | length of that vector | arithmetic |
| `shot_lateral_distance` | sideways distance, absolute | arithmetic |
| `is_cross` | cross-court, from the two feet points (same rule as `court.md`) | arithmetic |
| `ball_round` | shot number in the rally (1 = serve) | hit detection counts hits within a rally |
| `flight_seconds` | time until the next hit (NaN on a rally's last shot) | hit detection gives hit frames; `fps` is known |
| `seconds_since_previous_hit` | time since the previous hit (NaN on the first shot) | same |
| `previous_shot_distance` | previous hitter's feet to this hitter's feet (in the previous hitter's frame) | previous and current hit frames |
| `previous_hit_y` | how deep the previous hitter stood, in their own frame | previous hit frame |

The `contact` set has the same columns from ShuttleSet's labelled points instead: `hit_x/hit_y`
(the floor point under the racket, else the hitter's feet), the labelled `landing_x/landing_y`,
the stored `is_cross`, and the previous shot's labelled hit and landing. It can't be produced
from player tracking; it is kept so the results doc can show what the feet-only features cost.

Not used, because only a human labeller knows them: `type_en` itself, `backhand`,
and ShuttleSet's `hit_height`, `aroundhead`, `landing_height`, `hit_area`. Not used, so the model
doesn't learn who is playing: `player`, `match_id`, scores. Not used: the previous shot's
true group (on my footage only a *predicted* previous group would exist).

Missing values stay NaN; the gradient-boosted tree handles them natively.

**Which shots get a prediction.** Only shots with both a hitter point and a landing point in
the shipped feature set. With `feet` that means the hitter's feet are known **and the next shot's
hitter's feet are known**: a rally's last shot has no reply, so it gets no prediction, and neither
does a shot whose reply has no hitter position. Other shots keep `type_detail_pred`, `type_pred` and `type_conf` NULL.
Training and evaluation use the same rule, so the reported accuracy is for exactly the shots
that get a prediction.

## `shots/features.py`

### Constants
- `FEATURE_SET_NAMES` (`("contact", "feet")`) and `SHIPPED_FEATURE_SET` (`"feet"`).
- `FEATURE_NAMES` (`tuple[str, ...]`): the 16 features above, in model column order, the same
  for both feature sets.
- `SHOT_FEATURE_ROWS_SQL`: every `shots` row with its `rowid` (`row_id`) and the previous and
  next shot in the same rally (`LAG`/`LEAD` over `source, match_id, set_no, rally`, ordered by
  `ball_round, frame_num`): `previous_frame_num`, `next_frame_num`, `previous_hit_x/y`,
  `previous_landing_x/y`, `previous_player_x/y`, `next_player_x/y`.

### `load_shot_feature_rows(connection)`
Runs `SHOT_FEATURE_ROWS_SQL`. Returns a `pandas.DataFrame`, one row per `shots` row.
Errors: `duckdb.CatalogException` if there is no `shots` table.

### `shot_features(shot_rows, feature_set=SHIPPED_FEATURE_SET)`
- `shot_rows` (`DataFrame` with the columns of `SHOT_FEATURE_ROWS_SQL`).
- `feature_set` (`str`): `"feet"` or `"contact"`.
- Returns a `DataFrame` with exactly `FEATURE_NAMES` as columns, same index, floats (NaN if missing).
- Errors: `ValueError` for an unknown feature set; `KeyError` if a needed column is missing.

### Point builders
- `feet_shot_points(shot_rows)` / `contact_shot_points(shot_rows)`: `dict` of the arrays that differ
  between the two sets (`hit_x`, `hit_y`, `landing_x`, `landing_y`, `is_cross`, `previous_shot_distance`,
  `previous_hit_y`). `SHOT_POINT_BUILDERS` maps the set name to the builder.
- `hitter_point_columns(shot_rows)`: `(x, y)` arrays for the `contact` set: the hit point when present
  and on the near half, else the hitter's position (the rule of `analytics.court.hitter_point_for_zone`).
- `rotated_into_other_frame(x, y)`: `(6.1 - x, 13.4 - y)`, the half turn that moves a point from
  one player's frame into the other's.
- `feet_landing_columns(shot_rows)`: the next shot's `player_x/player_y` rotated into this hitter's frame.
- `is_cross_court_columns(hit_x, landing_x)`: the `court.md` cross-court rule on arrays (1.0 / 0.0, NaN if a point is missing).

### `has_hitter_and_landing_point(features)`
`True` per row when `hit_x`, `hit_y`, `landing_x` and `landing_y` are all present in a `shot_features`
result. This decides which shots are trained on and predicted.

### `seconds_between(later_frames, earlier_frames, fps)`
`(later - earlier) / fps` as a float array, NaN where any value is missing.

A clear whose reply comes 45 frames later at 30 fps, hit from (1.1, 1.2) by a player whose
opponent then hits from (2.2, 1.0) in their own frame; then a rally's last shot:

```python
>>> import math
>>> import pandas as pd
>>> from shots.features import FEATURE_NAMES, has_hitter_and_landing_point, shot_features
>>> shot_rows = pd.DataFrame({
...     "ball_round": [2, 3], "frame_num": [1000, 1045], "fps": [30.0, 30.0],
...     "hit_x": [1.0, 2.0], "hit_y": [1.0, 1.1], "player_x": [1.1, 2.2], "player_y": [1.2, 1.0],
...     "landing_x": [4.0, 1.5], "landing_y": [12.5, 8.9], "opponent_x": [3.0, 2.5], "opponent_y": [9.5, 8.6],
...     "is_cross": [True, False], "previous_frame_num": [955, 1000], "next_frame_num": [1045, None],
...     "previous_hit_x": [2.0, 1.0], "previous_hit_y": [4.0, 1.0],
...     "previous_landing_x": [5.1, 4.0], "previous_landing_y": [12.4, 12.5],
...     "previous_player_x": [2.1, 1.1], "previous_player_y": [4.2, 1.2],
...     "next_player_x": [2.2, None], "next_player_y": [1.0, None],
... })
>>> feet = shot_features(shot_rows)
>>> list(feet.columns) == list(FEATURE_NAMES)
True
>>> feet.loc[0, ["hit_x", "hit_y", "landing_x", "landing_y", "is_cross", "flight_seconds"]].round(2).tolist()
[1.1, 1.2, 3.9, 12.4, 1.0, 1.5]
>>> feet.loc[0, ["previous_shot_distance", "previous_hit_y"]].round(2).tolist()
[8.51, 4.2]
>>> has_hitter_and_landing_point(feet).tolist()
[True, False]
>>> contact = shot_features(shot_rows, "contact")
>>> contact.loc[0, ["hit_x", "landing_x", "landing_y", "is_cross"]].tolist(), math.isnan(contact.loc[1, "flight_seconds"])
([1.0, 4.0, 12.5, 1.0], True)
>>> has_hitter_and_landing_point(contact).tolist()
[True, True]
>>> shot_features(shot_rows, "shuttle")
Traceback (most recent call last):
    ...
ValueError: unknown feature set 'shuttle'; choose from contact, feet

```

## `shots/jitter.py`

### `jitter_hit_frames(shot_rows, max_offset_frames, random_seed=JITTER_SEED)`
Returns a copy of `shot_rows` (the columns of `SHOT_FEATURE_ROWS_SQL`, all rows of the table)
where every `frame_num` is shifted by an integer drawn uniformly from `-max_offset_frames` to
`+max_offset_frames` (`numpy.random.default_rng(random_seed)`, so the same seed gives the same
shifts), and `previous_frame_num` / `next_frame_num` are recomputed from the shifted frames of
the neighbouring shots in the same rally (same rally order as the SQL: `ball_round`, then the
original `frame_num`). Positions are untouched: only the timing features change.

Pass the whole table, not a subset: a shot's neighbour may be a row the model never trains on
(a serve typed `unknown`), and it must still be there to supply the time features.

- `JITTER_OFFSETS_FRAMES` (`(5, 10)`): the offsets the results doc reports. `JITTER_SEED` is 0.
- `jitter_label(max_offset_frames)`: the report key, e.g. `"hit frames +/- 5"`.
- `uniform_frame_offsets(shot_count, max_offset_frames, random_seed)`: the offsets themselves.
- `rally_order_index(shot_rows)`: the row index in rally order.

```python
>>> import pandas as pd
>>> from shots.jitter import jitter_hit_frames, jitter_label, uniform_frame_offsets
>>> shot_rows = pd.DataFrame({
...     "source": ["shuttleset"] * 3, "match_id": ["m"] * 3, "set_no": [1] * 3, "rally": [1] * 3,
...     "ball_round": [1, 2, 3], "frame_num": [100, 130, 170],
...     "previous_frame_num": [None, 100, 130], "next_frame_num": [130, 170, None],
... })
>>> jittered = jitter_hit_frames(shot_rows, 5, random_seed=0)
>>> offsets = jittered["frame_num"] - shot_rows["frame_num"]
>>> bool(offsets.abs().le(5).all()), bool((offsets == uniform_frame_offsets(3, 5, random_seed=0)).all())
(True, True)
>>> jittered["next_frame_num"].tolist()[:2] == jittered["frame_num"].tolist()[1:]
True
>>> jitter_hit_frames(shot_rows, 0)[["previous_frame_num", "next_frame_num"]].to_numpy().tolist()
[[nan, 130.0], [100.0, 170.0], [130.0, nan]]
>>> jitter_label(10)
'hit frames +/- 10'

```

## `shots/train.py`

### Constants

| Name | Value |
|---|---|
| `DATABASE_PATH` | `data/scout.duckdb` |
| `MODEL_PATH` | `data/models/shot_type_classifier.joblib` |
| `TRAINED_GROUPS` | the 7 groups except `unknown` |
| `TRAINED_DETAILS` | the 8 labels except `unknown` |
| `TRAINED_LABEL_COLUMN` | `type_detail`, the column the model learns |
| `FOLD_COUNT` | 5 |
| `RANDOM_SEED` | 0 |
| `PERMUTATION_REPEATS` | 3 |

### `train_classifier(database_path=DATABASE_PATH, model_path=MODEL_PATH, fold_count=5, random_seed=0, permutation_repeats=3, feature_set=SHIPPED_FEATURE_SET)`

1. Takes the ShuttleSet rows whose `type_group` is not `unknown` (shots typed `unknown` are left
   out of training and evaluation: their label carries no group), that have a hitter point
   and a landing point in `feature_set`, and whose `type_detail` is one of `TRAINED_DETAILS`
   (`rows_with_a_trained_detail`; a row with a group but no label is left out). The model learns `type_detail`.
2. Grouped cross-validation over **matches** (`GroupKFold`): each fold's test matches never
   appear in its training data, and every shot gets a prediction from a model that never
   saw its match. Raises `RuntimeError` if a fold ever had a match on both sides. Each fold's
   model also scores its test shots with the hit frames jittered by ±5 and ±10 frames
   (`shots/jitter.py`), the `variants` of the report.
3. Trains the final model on all labelled matches and saves it.

- Scores are at **group level** (both the label and the prediction renamed to their group), so they compare
  directly with a model trained on the groups; the same out-of-fold predictions at label level are under `detail`.
- Returns the report `dict`: `feature_set`, `trained_label`, `detail` (label level: `labels`, `accuracy_mean`,
  `accuracy_std`, `out_of_fold_accuracy`, `per_detail`, `confusion_matrix`), `folds` (per fold: test matches,
  shots, group accuracy, majority group and its accuracy), `accuracy_mean`, `accuracy_std`, `majority_accuracy_mean`, `out_of_fold_accuracy`,
  `per_group` (precision, recall, f1, support), `confusion_matrix`, `permutation_importance`
  (mean accuracy drop on the test folds, largest first), `variants` (per jitter label: `accuracy_mean`,
  `accuracy_std`, `out_of_fold_accuracy`, `per_group`), `subsets`, `per_type_en`, `training_shots`,
  `training_matches`, `group_counts`, `detail_counts`, `model_path`. The same dict is written as JSON next to the model.
  `permutation_importance` is measured on the label the model predicts (`type_detail`).
- The saved model is a joblib `dict`: `classifier`, `feature_set`, `feature_names`, `trained_label`, `labels` (the
  classifier's classes), `groups` (their groups), `sklearn_version`,
  `training_shots`, `training_matches`.
- Errors: `duckdb.IOException` if the database does not exist; `duckdb.CatalogException` if it
  has no `shots` table; `ValueError` from scikit-learn if there are fewer matches than folds,
  or if a feature is missing on every training shot.

### Helpers
- `labelled_rows_of(shot_rows, feature_set)`: the training rows out of a `load_shot_feature_rows` frame.
- `labelled_shuttleset_rows(connection, feature_set)`: the same, read from the database.
- `labelled_shuttleset_shots(connection, feature_set)`: `(features, labels, match_ids)` for training.
- `jittered_feature_frames(shot_rows, labelled_rows, feature_set, random_seed)`: `{jitter label: features}`
  for the labelled rows, with the whole table jittered first.
- `grouped_cross_validation(features, labels, match_ids, fold_count, random_seed, permutation_repeats, breakdown_rows, scoring_variants, group_of_label=None)`:
  the evaluation part, returns the report without the training fields. With `group_of_label` (a label -> group
  `dict`, e.g. `TYPE_GROUP_BY_TYPE_DETAIL`) the scores are at group level and the label level goes under `detail`;
  without it the labels are the groups and there is no `detail`. `permutation_repeats=0` skips the
  permutation importance (every value NaN); `scoring_variants` (`{name: features}`, same index as `features`)
  are scored by each fold's model on its test shots and reported under `variants`.
- `new_classifier(random_seed)`: `HistGradientBoostingClassifier(random_state=random_seed)`, default settings.
- `majority_group(labels)`: most common label (ties: alphabetical first).
- `per_group_scores(true_groups, predicted_groups)`: precision/recall/f1/support per group;
  `per_label_scores(true_labels, predicted_labels, label_names)` the same for any labels.
- `as_groups(labels, group_of_label)`: the labels renamed to their groups (unchanged without a map).
- `detail_scores(labels, predictions, fold_accuracies)`: the `detail` entry of the report.
- `variant_scores(labels, predictions, fold_accuracies)`: the per-variant entry of the report.
- `save_model(classifier, model_path, training_shots, training_matches, feature_set)`, `report_path_for(model_path)`, `summary_lines(report)`.

## `shots/predict.py`

### `predict_shot_types(database_path=DATABASE_PATH, model_path=MODEL_PATH)`
Sets `type_detail_pred` (the most likely label), `type_pred` (that label's group) and `type_conf` (the
label's probability, 0-1; the group's own probability is at least this) on every `shots`
row that has a hitter point and a landing point in the shipped feature set: both sources, and
`unknown` shots too. Every other row gets NULL in both columns, which also clears an older
prediction. Matched by `rowid`, in one transaction; no other column changes.

- Returns the number of rows that got a prediction (`int`).
- Errors: `FileNotFoundError` if the model file is missing; `ModelFeatureMismatchError` if the model
  was trained on a different `FEATURE_NAMES` or a different feature set than `SHIPPED_FEATURE_SET`
  (retrain); `ModelLabelMismatchError` if the model was not trained on `type_detail` (a model from before the
  labels; retrain); `FileNotFoundError` for a missing database (checked before opening it, so a mistyped path
  creates no file; the command prints it and exits 1); `CatalogException` for a missing table. On any error
  the transaction rolls back.

### Helpers
- `load_model(model_path)`: loads and checks the joblib `dict`.
- `prediction_summary(database_path)`: counts for the results doc: ShuttleSet rows (`shots`, `predicted_shots`,
  `lowest_conf`, `mean_conf`, `in_sample_agreement`, `in_sample_detail_agreement`, `unknown_shot_predictions`) and, under `own_matches`,
  per own match id `shots`, `predicted_shots` and `rallies`.
- `predictions_for(model, shot_rows)`: `DataFrame` of `row_id`, `type_detail_pred`, `type_pred`, `type_conf`, with
  `None` / `<NA>` for rows without a hitter point and a landing point.
- `ModelFeatureMismatchError(model_feature_names, model_feature_set)`: a `RuntimeError` with both attributes.
- `ModelLabelMismatchError(model_trained_label)`: a `RuntimeError`; `model_trained_label` is `None` for a model
  saved before labels existed.

## `shots/compare_features.py`

`compare_feature_sets(database_path=DATABASE_PATH, comparison_path=COMPARISON_PATH, fold_count=5, random_seed=0, feature_set=SHIPPED_FEATURE_SET)`
re-runs the same match folds and classifier as training for each entry of `FEATURE_SETS`: geometry
only, + timing, the shipped `FEATURE_NAMES`, and the shipped set plus the previous shot's **true**
group (a categorical feature, `previous_true_group`, from `PREVIOUS_TRUE_GROUP_SQL`). That last set
is an experiment only: on my own footage the true group is never known. All columns come from
`feature_set` (default: the shipped `feet` set). Like training, each set learns `type_detail` on the shots that
have one and is scored on the groups of its labels, so the shipped set's folds equal the training report's.

- Returns `{set_name: {"feature_set", "feature_names", "fold_accuracies", "accuracy_mean"}}` and writes it as JSON to
  `comparison_path` (default `data/models/feature_set_comparison.json`).
- Errors: as `train_classifier`.
- Helpers: `previous_true_group_codes(connection, row_ids)` (group codes 0-7, NaN on a rally's first shot),
  `grouped_accuracy(features, labels, match_ids, fold_count, random_seed)` (group accuracy per fold for detail labels).

## `shots/compare_hit_points.py`

`compare_hit_points(database_path=DATABASE_PATH, comparison_path=COMPARISON_PATH, fold_count=5, random_seed=0)`
evaluates the `contact` and `feet` feature sets with the same match folds, classifier and jitter
as training (no permutation importance), twice each: on the **shared shots** (labelled shots that
have a hitter point and a landing point in both sets, so the two accuracies are on identical
shots) and on **all its shots** (every shot that set can predict; for `feet` that is the shipped model's
evaluation, fold by fold). Both sets learn `type_detail` and are scored as groups, like training.

- Returns and writes as JSON (default `data/models/hit_point_comparison.json`):
  `landing_offset_m` (`shots`, `mean_m`, `p50_m`, `p90_m`: distance between the reply hitter's feet and the
  labelled landing point on the shared shots), `shared_shots` and `all_shots` (per feature set: `shots`,
  `matches`, `folds`, `accuracy_mean`, `accuracy_std`, `majority_accuracy_mean`, `out_of_fold_accuracy`,
  `per_group`, `confusion_matrix`, `subsets`, `variants`, `detail`).
- Errors: as `train_classifier`.
- Helpers: `rows_predictable_with_every_feature_set(labelled_rows)`, `landing_offset_summary(shot_rows)` (with no
  shot that has both a labelled landing and a reply hitter: `shots` 0 and `None` for every distance, and the results
  table prints "no shots"), `evaluate_feature_set(shot_rows, labelled_rows, feature_set, fold_count, random_seed)`,
  `summary_lines(comparison)`.

```python
>>> import pandas as pd
>>> from shots.compare_hit_points import landing_offset_summary
>>> from shots.results_tables import landing_offset_table
>>> rows = pd.DataFrame({"landing_x": [1.0, 2.0], "landing_y": [12.0, 11.0], "next_player_x": [5.0, 4.4], "next_player_y": [1.4, 2.0]})
>>> {key: round(value, 2) for key, value in landing_offset_summary(rows).items()}
{'shots': 2, 'mean_m': 0.3, 'p50_m': 0.3, 'p90_m': 0.46}
>>> landing_offset_summary(rows.iloc[:0])
{'shots': 0, 'mean_m': None, 'p50_m': None, 'p90_m': None}
>>> print(landing_offset_table({"landing_offset_m": landing_offset_summary(rows.iloc[:0])}))
| Reply hitter's feet vs labelled landing point | |
|---|---|
| Shots | 0 |
| Median distance | no shots |
| 90th percentile | no shots |
| Mean | no shots |

```

## `shots/fine_types.py`

`compare_fine_types(database_path=DATABASE_PATH, comparison_path=COMPARISON_PATH)` takes the same labelled
ShuttleSet rows as training (the 7 groups, `unknown` left out, hitter and landing point present), the shipped
`feet` features, the same classifier and `GroupKFold(5)` by match, and cross-validates these label sets:

- `groups`: the 7 groups (the shipped labels, as a reference).
- `fine_types`: the 18 `type_en` values (`FINE_TYPES`).
- `splits`: for each entry of `CANDIDATE_SPLITS`, the 7 groups with only those types split off
  (`labels_with_splits(type_en, split_off_types)`; `short vs long service` splits `serve` into both).
- `middle`: the 7 groups with every split that holds up taken off at once.

Each label set reports `accuracy_mean`/`accuracy_std` over the folds, `out_of_fold_accuracy`, `group_accuracy`
(predictions and labels both mapped back to the 7 groups, so a finer model can be compared with the shipped one),
`per_label` precision/recall/F1/support and `confusion` (rows true, columns predicted, in `labels` order).
Each split also has `pair`: among shots whose true and predicted labels are both in the split group, the
accuracy of telling the sides apart (`pair_accuracy`) against always picking that group's commonest side
(`pair_majority`). A split **holds up** (`split_holds_up`) when every side of it reaches F1 >= `SPLIT_MIN_F1`
(0.5) and `pair_accuracy` beats `pair_majority` by at least `SPLIT_MIN_PAIR_GAIN` (5 points). Both thresholds
were set before the run.

- Writes the comparison as JSON to `comparison_path` (default `data/models/fine_type_comparison.json`); opens the
  database read-only and never touches `data/models/shot_type_classifier.joblib`.

```python
>>> import pandas as pd
>>> from shots.fine_types import labels_with_splits, FINE_TYPES
>>> len(FINE_TYPES), "unknown" in FINE_TYPES
(18, False)
>>> labels_with_splits(pd.Series(["push", "drive", "rush", "short service"]), ("push",)).tolist()
['push', 'drive', 'net', 'serve']
>>> labels_with_splits(pd.Series(["short service", "long service", "lob"]), ("short service", "long service")).tolist()
['short service', 'long service', 'lift']

```

## `shots/results_tables.py`

Prints the markdown tables of the results docs from the training report and the comparison
JSONs: `accuracy_table(report)`, `subsets_table(report)`, `per_group_table(report)` (sorted by F1),
`confusion_matrix_table(report)`, `importance_table(report)` (3 decimals), `weak_types_table(report)`,
`feature_set_table(comparison)`, `predictions_table(summary)`, and `results_tables(report, comparison, summary)`
for all of them. From the hit-point comparison: `hit_point_accuracy_table`, `hit_point_per_group_table`,
`hit_point_confusion_table(comparison, feature_set)`, `jitter_table`, `landing_offset_table` and
`hit_point_results_tables(comparison)` for all of them. Every function returns a `str`.
Errors: `KeyError` for a report without the expected fields.

```python
>>> from shots.results_tables import importance_table, jitter_table
>>> print(importance_table({"permutation_importance": {"landing_y": 0.28512, "flight_seconds": 0.13846, "shot_dy": 0.06248}}))
| Feature | Accuracy drop |
|---|---|
| landing_y | 0.285 |
| flight_seconds | 0.138 |
| shot_dy | 0.062 |
>>> scores = lambda exact, five, ten: {"accuracy_mean": exact, "variants": {"hit frames +/- 5": {"accuracy_mean": five}, "hit frames +/- 10": {"accuracy_mean": ten}}}
>>> print(jitter_table({"shared_shots": {"contact": scores(0.87, 0.85, 0.83), "feet": scores(0.80, 0.78, 0.75)}, "all_shots": {"feet": scores(0.81, 0.79, 0.76)}}))
| Hit frames | contact, shared shots | feet, shared shots | feet, all its shots (the shipped model) |
|---|---|---|---|
| exact | **87.0%** | **80.0%** | **81.0%** |
| +/- 5 | 85.0% (-2.0) | 78.0% (-2.0) | 79.0% (-2.0) |
| +/- 10 | 83.0% (-4.0) | 75.0% (-5.0) | 76.0% (-5.0) |

```

## Example: train and predict on a small synthetic table

Six matches of clear -> net shot -> smash rallies (no ShuttleSet needed). With the `feet` set a
shot's landing is the next hitter's feet, so the last shot of each rally (the smash) has no
landing and is neither trained on nor predicted:

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
>>> hitter_feet = {"clear": (1.0, 1.0), "net": (3.0, 5.5), "smash": (4.5, 1.5)}
>>> rows = []
>>> for match_number in range(6):
...     for rally in range(1, 21):
...         for ball_round, group in enumerate(["clear", "net", "smash"], start=1):
...             player_x, player_y = hitter_feet[group]
...             jitter = ((match_number * 7 + rally * 3 + ball_round) % 10) / 20
...             rows.append(("shuttleset", f"match_{match_number}", 1, rally, ball_round, rally * 1000 + ball_round * 60,
...                          30.0, "A", group, group, group, player_x + jitter, player_y, 3.0, 10.0))
>>> _ = connection.executemany(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, type_en, type_detail, type_group, "
...     "player_x, player_y, opponent_x, opponent_y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
>>> _ = connection.execute(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, fps, player, player_x, player_y) "
...     "VALUES ('own', 'club_2026_10_20', 1, 1, 1, 100, 30.0, 'me', 1.1, 1.0), "
...     "('own', 'club_2026_10_20', 1, 1, 2, 160, 30.0, 'opp', 3.0, 5.5), "
...     "('own', 'club_2026_10_20', 1, 2, 1, 400, 30.0, 'me', NULL, NULL)")
>>> connection.close()
>>> report = train_classifier(database_path, temp_dir / "models" / "classifier.joblib", fold_count=3, permutation_repeats=1)
>>> report["feature_set"], report["training_shots"], report["training_matches"], report["accuracy_mean"], round(report["majority_accuracy_mean"], 2)
('feet', 240, 6, 1.0, 0.5)
>>> sorted(report["variants"]), report["variants"]["hit frames +/- 5"]["accuracy_mean"]
(['hit frames +/- 10', 'hit frames +/- 5'], 1.0)
>>> model = load_model(temp_dir / "models" / "classifier.joblib")
>>> model["trained_label"], sorted(model["labels"]), sorted(model["groups"])
('type_detail', ['clear', 'net'], ['clear', 'net'])
>>> report["detail"]["accuracy_mean"], sorted(report["detail_counts"].items())[:2]
(1.0, [('clear', 120), ('drive', 0)])
>>> predict_shot_types(database_path, temp_dir / "models" / "classifier.joblib")
241
>>> connection = duckdb.connect(str(database_path))
>>> connection.execute("SELECT rally, ball_round, type_pred, type_conf IS NULL FROM shots WHERE source = 'own' ORDER BY rally, ball_round").fetchall()
[(1, 1, 'clear', False), (1, 2, None, True), (2, 1, None, True)]
>>> connection.execute("SELECT COUNT(*) FROM shots WHERE type_pred = type_group").fetchone()[0]
240
>>> connection.execute("SELECT COUNT(*) FROM shots WHERE type_detail_pred = type_detail").fetchone()[0]
240
>>> connection.close()
>>> try:
...     predict_shot_types(database_path, temp_dir / "no_model.joblib")
... except FileNotFoundError as error:
...     str(error).endswith("no_model.joblib (run python -m shots.train first)")
True
>>> shutil.rmtree(temp_dir)

```
