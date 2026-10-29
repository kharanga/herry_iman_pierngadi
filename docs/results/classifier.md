# Shot-type classifier: results

## How to run
```
.venv/bin/python -m shots.train
.venv/bin/python -m shots.predict
.venv/bin/python -m shots.compare_features
.venv/bin/python -m shots.results_tables
```
`train` takes about 80 s on the dev MacBook Air (5 folds + permutation importance + the final fit);
`compare_features` about 2 minutes. Every table below is printed by `shots.results_tables` from
`data/models/shot_type_classifier_report.json` (written by `train`),
`data/models/feature_set_comparison.json` (written by `compare_features`) and the `shots` table.
Numbers in the text come from the same files; gaps between feature sets are computed from the unrounded values.

## Setup
- Model: scikit-learn 1.9.1 `HistGradientBoostingClassifier`, default settings, `random_state=0`.
- Shots used: ShuttleSet shots with one of the 7 real groups **and** both a hitter point and a landing
  point: **35,056 shots from 44 matches**. Left out: the 1,407 shots typed `unknown`
  (their label has no group), and 20 labelled shots without a hitter point or a landing point. `predict` leaves those
  NULL too, so the accuracy below is for exactly the shots that get a prediction.
- Features: 16, from geometry and timing only: hit, landing and receiver positions, the shot vector,
  `is_cross`, `ball_round`, time to the next and since the previous hit, the previous shot's length and depth.
  The list, with why each is available from tracking, is in `docs/analytics/classifier.md`. No human-only
  labels (`backhand`, `hit_height`, `aroundhead`, ...) and nothing that identifies a player or match.
- Evaluation: 5-fold `GroupKFold` over **matches**. Each fold tests on 8-9 matches the model never saw,
  and the code refuses a fold with a match on both sides. Every shot gets exactly one held-out prediction,
  so the per-group numbers and the confusion matrix below are over all 35,056 shots, each predicted
  by a model that did not train on its match.
- Baseline: the majority class of each fold's training data (always `net`), scored on the test fold.

## Accuracy

| | Held-out accuracy |
|---|---|
| **Classifier** | **86.8%** (± 0.8% across folds; folds 85.7, 87.9, 86.0, 87.5, 86.7) |
| Majority class (`net`) | 33.6% |

| Shots | Count | Held-out accuracy |
|---|---|---|
| excluding each rally's first shot | 32,621 | 85.8% |
| rally's last shot (no next hit, so no flight time) | 3,508 | 79.5% |
| all other shots | 31,548 | 87.6% |

Serves are easy: the first shot of a rally is nearly always one (`ball_round` = 1). A rally's last shot is
harder, because it has no next hit and so no flight time, the third most important feature.

## Per group (held-out)

| Group | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve | 2,415 | 0.997 | 1.000 | 0.998 |
| net | 11,790 | 0.953 | 0.964 | 0.959 |
| clear | 2,922 | 0.948 | 0.943 | 0.945 |
| drop | 3,500 | 0.876 | 0.866 | 0.871 |
| smash | 4,234 | 0.848 | 0.873 | 0.860 |
| lift | 5,627 | 0.772 | 0.822 | 0.796 |
| drive | 4,568 | 0.634 | 0.554 | 0.592 |

## Confusion matrix (held-out; rows = true group, columns = predicted)

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,414** | 0 | 0 | 1 | 0 | 0 | 0 |
| clear | 0 | **2,754** | 12 | 58 | 2 | 40 | 56 |
| drop | 0 | 7 | **3,030** | 287 | 130 | 2 | 44 |
| smash | 2 | 51 | 264 | **3,697** | 50 | 6 | 164 |
| net | 2 | 1 | 74 | 17 | **11,365** | 28 | 303 |
| lift | 0 | 39 | 5 | 3 | 61 | **4,626** | 893 |
| drive | 3 | 53 | 75 | 296 | 315 | 1,294 | **2,532** |

## Weak groups, and why
Recall per original shot type (held-out), for every type below 85%:

| Type (group) | Shots | Recall | Most common mistakes |
|---|---|---|---|
| driven flight (drive) | 65 | 20% | net 51, smash 1 |
| back-court drive (drive) | 473 | 32% | smash 229, drop 69 |
| push (drive) | 2,924 | 54% | lift 1,236, net 77 |
| defensive return lob (lift) | 301 | 54% | drive 104, clear 23 |
| rush (net) | 512 | 60% | drive 181, lift 19 |
| defensive return drive (drive) | 406 | 67% | net 69, lift 46 |
| drive (drive) | 700 | 74% | net 118, smash 45 |
| wrist smash (smash) | 1,648 | 83% | drop 151, drive 93 |
| lob (lift) | 5,326 | 84% | drive 789, net 54 |
| drop (drop) | 2,144 | 85% | smash 206, net 98 |

- **drive vs lift is the main error** (1,294 drives predicted lift, 893 lifts predicted drive).
  It is mostly `push` vs `lob` (1,236 pushes predicted lift): both are hit from the front or mid court and
  reach the receiver in the mid or rear court. What separates them is **height** (a push is flat, a lob goes
  high), which floor positions and timing don't capture. Flight time helps (a lob takes longer), but the ranges overlap.
- `back-court drive` (flat, from the rear) looks like a smash on the floor: same start, same end, fast
  (229 predicted smash).
- `driven flight` (65 shots) is too rare to learn and looks like a net exchange.
- `drop` vs `smash` (287 + 264): both go from the rear court to the front or mid court; the smash is faster.

What would help, later: the shuttle's height or apex from shuttle tracking (a pixel-height or flight-arc
feature). It is not a court-floor position, so it is not in the `shots` table today.

## Feature importance
Permutation importance: the drop in held-out accuracy when one feature is shuffled, averaged over
the 5 test folds (3 repeats each). Features are correlated, so a feature can look unimportant while
another one carries the same information (`shot_dy` vs `landing_y`). `-0.000` means a tiny negative mean:
shuffling that feature made no difference.

| Feature | Accuracy drop |
|---|---|
| landing_y | 0.289 |
| hit_y | 0.191 |
| flight_seconds | 0.140 |
| ball_round | 0.092 |
| shot_dy | 0.070 |
| seconds_since_previous_hit | 0.043 |
| shot_distance | 0.007 |
| receiver_y | 0.004 |
| landing_x | 0.003 |
| hit_x | 0.003 |
| previous_shot_distance | 0.003 |
| shot_lateral_distance | 0.002 |
| previous_hit_y | 0.001 |
| shot_dx | 0.001 |
| receiver_x | 0.000 |
| is_cross | -0.000 |

Depth (how far from the net the shot starts and ends) and timing carry the model; sideways
position barely matters for the shot type. Tracking must get depth and hit frames right.

## Feature-set experiments (same 5 match folds, same shots)

| Features | Held-out accuracy |
|---|---|
| geometry only (positions, shot vector, `is_cross`, `ball_round`) | 83.4% |
| + timing (time to next hit, since previous hit) | 86.9% |
| + previous shot's length and depth (**the shipped model**) | 86.8% |
| + previous shot's **true** group | 87.1% |

Run by `shots.compare_features` with the shipped `shots/features.py` and classifier; the "shipped model"
row is the same 5 folds as the Accuracy table and matches it exactly.
- Timing adds 3.5 points (83.38% -> 86.93%).
- The previous shot's length and depth change accuracy by -0.2 points (86.93% -> 86.76%): no measurable help
  (folds vary by ± 0.8%). They are kept for now, since removing them changes nothing measurable either.
- The previous shot's **true** group adds 0.3 points (86.76% -> 87.07%). On my footage it would have to be the
  *predicted* group, so it is left out.

## Predictions in `shots`

| | |
|---|---|
| ShuttleSet shots in `shots` | 36,483 |
| With a prediction | 35,293 |
| Left NULL (no hitter point or no landing point) | 1,190 |
| `type_conf` mean / lowest | 0.89 / 0.25 |
| In-sample agreement with `type_group` (not an accuracy) | 93.2% |
| Shots typed `unknown`, predicted as | NULL 1,170, serve 199, smash 16, net 12, lift 6, drive 4 |

- `predict` changed only `type_pred` and `type_conf` (`EXCEPT ALL` on every other column, both ways, 0 rows),
  and a second run gives the same table.
- Shots without both a hitter point and a landing point get NULL in both columns: a guess without
  geometry is meaningless. Most of them are shots typed `unknown` with no positions at all.
- `type_conf` is the model's probability for its chosen group. It is not calibrated or validated as a probability yet.
- The in-sample agreement is on shots the final model trained on, so it is **not** an accuracy figure.
  Use the held-out 86.8%.

## Notes
- Player identity is not a feature, so the unreliable `player` labels on one-shot `unknown` rallies
  (`docs/results/schema.md`, "Known data issues") don't affect the classifier. Those shots are excluded
  from training anyway.
- The model file, report and comparison live under `data/models/` (gitignored, never committed).
- A feature that is missing on every training shot makes scikit-learn's binning fail (`ValueError`).
  This can't happen with ShuttleSet, where every feature has values.
