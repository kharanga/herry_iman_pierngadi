# Feet-only shot-type classifier: results

The pipeline tracks the two players, not the shuttle, so the classifier now learns from exactly
what player tracking gives: the hitter's feet at the hit, the opponent's feet at the hit, and
the hit frames. The landing point is where the opponent hits the reply (their feet at the next
hit). This page has the same evaluation as `docs/results/classifier.md` (the earlier model, trained on
ShuttleSet's labelled hit and landing points), the two side by side, and what imprecise hit
timing costs. The numbers are for the six-zone court (`docs/results/schema.md`): `is_cross`, one
of the 16 features, is now "hit and landing on opposite sides of the centre line" instead of the
earlier 1 m-sideways rule. Retraining on it moved the feet model from 86.5% to 86.6%, inside the
fold spread, and every other number by a few tenths at most.

## How to run
```
.venv/bin/python -m shots.train
.venv/bin/python -m shots.compare_hit_points
.venv/bin/python -m shots.compare_features
.venv/bin/python -m shots.predict
.venv/bin/python -m shots.results_tables
```
On the dev MacBook Air: `train` 58 s, `compare_hit_points` 1 min 52 s, `compare_features` 43 s,
`predict` 2 s. Every table below is printed by `shots.results_tables` from
`data/models/shot_type_classifier_report.json` (written by `train`),
`data/models/hit_point_comparison.json` (`compare_hit_points`),
`data/models/feature_set_comparison.json` (`compare_features`) and the `shots` table.
Numbers in the text come from the same files.

## Setup
- Model: scikit-learn `HistGradientBoostingClassifier`, default settings, `random_state=0`, the
  same as before. Only the features changed.
- Feature set `feet` (`shots/features.py`, the shipped one): hit point = `player_x/player_y` (the
  hitter's feet); landing = the **next shot's** `player_x/player_y` turned half a turn into this
  hitter's frame (the reply hitter's feet); receiver = `opponent_x/opponent_y` at the hit; `is_cross`
  (opposite sides of the centre line) and the shot vector from those two feet points; `ball_round`; timing from `frame_num` and `fps` as
  before; the previous shot's length and depth from the previous hitter's feet. Same 16 column
  names as the `contact` set, so everything downstream is unchanged.
- Shots used: ShuttleSet shots with one of the 7 real groups **and** a hitter position **and** a
  reply with a hitter position: **31,544 shots from 44 matches**. A rally's last shot has no reply,
  so it has no landing point and gets no prediction (3,508 labelled shots; the earlier model
  predicted them at 79.5%). Left out as before: 1,407 shots typed `unknown`, and labelled shots
  without positions.
- Evaluation: 5-fold `GroupKFold` over **matches**, as before: each fold tests on 8-9 matches the
  model never saw, every shot gets one held-out prediction, and the code refuses a fold with a match
  on both sides. Baseline: each fold's majority training class (`net`).
- Timing jitter: each fold's model also scores its test shots with every hit frame in the table
  shifted by an integer drawn uniformly from -5..+5 and from -10..+10 frames (`numpy` seed 0;
  neighbours' frames are shifted too, so both time features move). At ShuttleSet's 25 and 30 fps
  that is up to 0.17-0.20 s and 0.33-0.40 s; at the Pi's 60 fps the same frame offsets are half the time.

## Accuracy

| | Held-out accuracy |
|---|---|
| **Classifier** | **86.6%** (± 0.7% across folds; folds 85.5, 87.0, 87.2, 87.0, 86.0) |
| Majority class (`net`) | 33.1% |

| Shots | Count | Held-out accuracy |
|---|---|---|
| excluding each rally's first shot | 29,131 | 85.5% |
| rally's last shot (no next hit, so no flight time) | 0 | none predicted |
| all other shots | 31,544 | 86.6% |

Serves stay easy (`ball_round` = 1). The "last shot" row is empty by construction: with feet-only
features a shot without a reply has no landing point, so it is neither trained on nor predicted.

## Per group (held-out)

| Group | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve | 2,393 | 0.996 | 1.000 | 0.998 |
| net | 10,458 | 0.957 | 0.962 | 0.959 |
| clear | 2,734 | 0.946 | 0.931 | 0.938 |
| drop | 3,204 | 0.867 | 0.848 | 0.857 |
| smash | 3,328 | 0.834 | 0.871 | 0.852 |
| lift | 5,259 | 0.771 | 0.841 | 0.804 |
| drive | 4,168 | 0.632 | 0.545 | 0.585 |

## Confusion matrix (held-out; rows = true group, columns = predicted)

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,393** | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | **2,545** | 33 | 50 | 0 | 63 | 43 |
| drop | 0 | 31 | **2,717** | 247 | 142 | 2 | 65 |
| smash | 1 | 36 | 228 | **2,898** | 15 | 2 | 148 |
| net | 3 | 1 | 66 | 14 | **10,058** | 12 | 304 |
| lift | 0 | 40 | 4 | 3 | 28 | **4,422** | 762 |
| drive | 5 | 38 | 86 | 262 | 270 | 1,236 | **2,271** |

## Contact points vs feet, side by side

`shots.compare_hit_points` runs both feature sets through the same 5 match folds, classifier and
jitter. "Shared shots" are the 31,543 labelled shots that both sets can predict, so those two
accuracies are on identical shots; "all its shots" is each set on everything it can predict (for
`feet` that is the shipped model's own evaluation, fold by fold; for `contact` the 86.8% of
`docs/results/classifier.md`).

| Feature set | Shared shots | Held-out accuracy | All its shots | Held-out accuracy |
|---|---|---|---|---|
| contact (labelled hit point and landing point) | 31,543 | **87.5%** (± 0.9%) | 35,056 | **86.8%** (± 0.8%) |
| feet (hitter's feet; landing = the reply hitter's feet) | 31,543 | **86.8%** (± 1.1%) | 31,544 | **86.6%** (± 0.7%) |

| Group | Shots | contact P / R / F1 | feet P / R / F1 | F1 change |
|---|---|---|---|---|
| serve | 2,393 | 0.996 / 1.000 / 0.998 | 0.997 / 1.000 / 0.999 | +0.001 |
| net | 10,457 | 0.965 / 0.965 / 0.965 | 0.957 / 0.961 / 0.959 | -0.006 |
| clear | 2,734 | 0.957 / 0.955 / 0.956 | 0.944 / 0.933 / 0.938 | -0.018 |
| drop | 3,204 | 0.890 / 0.885 / 0.887 | 0.874 / 0.847 / 0.860 | -0.027 |
| smash | 3,328 | 0.848 / 0.882 / 0.864 | 0.833 / 0.873 / 0.852 | -0.012 |
| lift | 5,259 | 0.780 / 0.837 / 0.807 | 0.775 / 0.846 / 0.809 | +0.002 |
| drive | 4,168 | 0.641 / 0.564 / 0.600 | 0.639 / 0.554 / 0.594 | -0.007 |

**Feet cost 0.8 points on the same shots (87.5% -> 86.8%)**, inside one fold's spread (± 1.1%).
The reply hitter's feet are a good stand-in for the labelled landing point:

| Reply hitter's feet vs labelled landing point | |
|---|---|
| Shots | 31,543 |
| Median distance | 0.82 m |
| 90th percentile | 1.53 m |
| Mean | 0.90 m |

The median 0.82 m is the same distance ShuttleSet's labelled hit point sits from the hitter's feet
(it marks the floor under the racket, not the feet), so swapping both ends of the shot for feet moves
every shot by a similar, mostly lengthways amount, which the tree re-learns. What the feet lose is
precision at the ends: `drop` (-0.027 F1) and `clear` (-0.018) are the groups separated by **where
exactly the shuttle comes down** (a drop just over the net vs a net shot; a clear to the baseline vs
a lift), and a receiver who reaches a drop late, or takes a clear a step inside the baseline, blurs that.
`drive` vs `lift` (1,213 drives predicted lift, 733 lifts predicted drive) stays the main error, as
before: it is a matter of height, which neither feature set has.

The two confusion matrices on the shared shots:

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,392** | 0 | 0 | 1 | 0 | 0 | 0 |
| clear | 0 | **2,611** | 16 | 31 | 1 | 30 | 45 |
| drop | 0 | 4 | **2,834** | 205 | 112 | 2 | 47 |
| smash | 2 | 27 | 190 | **2,934** | 13 | 2 | 160 |
| net | 3 | 1 | 78 | 7 | **10,090** | 7 | 271 |
| lift | 0 | 43 | 3 | 1 | 17 | **4,401** | 794 |
| drive | 4 | 42 | 64 | 281 | 222 | 1,203 | **2,352** |

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,393** | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | **2,550** | 27 | 52 | 1 | 59 | 45 |
| drop | 0 | 35 | **2,715** | 250 | 138 | 2 | 64 |
| smash | 1 | 38 | 219 | **2,904** | 16 | 2 | 148 |
| net | 3 | 1 | 63 | 15 | **10,045** | 14 | 316 |
| lift | 0 | 40 | 4 | 2 | 30 | **4,450** | 733 |
| drive | 2 | 37 | 80 | 262 | 263 | 1,213 | **2,311** |

## Timing jitter

| Hit frames | contact, shared shots | feet, shared shots | feet, all its shots (the shipped model) |
|---|---|---|---|
| exact | **87.5%** | **86.8%** | **86.6%** |
| +/- 5 | 84.6% (-3.0) | 83.4% (-3.4) | 83.2% (-3.3) |
| +/- 10 | 80.9% (-6.6) | 78.9% (-7.9) | 78.8% (-7.8) |

- Hit timing matters about as much for both feature sets: **±5 frames costs 3.4 points, ±10 frames
  7.8 points** on the shipped model. Time to the next hit is the third most important feature and
  the main thing that separates a smash from a drop and a lift from a drive; a 10-frame error at 30 fps
  is a third of a second on flights that take 0.3-1.5 s.
- On my footage the hit frames come from player movement (no shuttle), so the frame error will be
  larger than ShuttleSet's hand-labelled hits. The table says what that costs: the hit detector should
  be judged by how many of its hits land within ±5 frames of the real contact, not just by whether a
  hit was found.
- This jitter moves only the time features. On real footage a wrong hit frame also samples the feet
  at the wrong moment; that extra cost is not measured here (it needs labelled own footage).

## Feature importance (shipped feet model)

Permutation importance: the drop in held-out accuracy when one feature is shuffled, averaged over
the 5 test folds (3 repeats each). Features are correlated, so a feature can look unimportant while
another carries the same information.

| Feature | Accuracy drop |
|---|---|
| landing_y | 0.311 |
| hit_y | 0.170 |
| flight_seconds | 0.160 |
| ball_round | 0.099 |
| seconds_since_previous_hit | 0.056 |
| shot_dy | 0.046 |
| shot_distance | 0.018 |
| landing_x | 0.006 |
| hit_x | 0.005 |
| previous_hit_y | 0.004 |
| receiver_y | 0.004 |
| previous_shot_distance | 0.003 |
| receiver_x | 0.003 |
| shot_lateral_distance | 0.001 |
| shot_dx | 0.001 |
| is_cross | 0.000 |

Depth (`landing_y`, `hit_y`, `shot_dy`) and timing still carry the model; sideways position
barely matters. Player tracking must get the feet's depth and the hit frames right; a lateral
error is cheap.

## Weak shot types, and why
Recall per original shot type (held-out), for every type below 85%:

| Type (group) | Shots | Recall | Most common mistakes |
|---|---|---|---|
| driven flight (drive) | 63 | 17% | net 51, smash 1 |
| back-court drive (drive) | 442 | 30% | smash 214, drop 80 |
| rush (net) | 218 | 30% | drive 141, smash 10 |
| push (drive) | 2,682 | 53% | lift 1,184, net 37 |
| defensive return drive (drive) | 353 | 60% | net 81, lift 45 |
| defensive return lob (lift) | 274 | 61% | drive 69, clear 18 |
| drive (drive) | 628 | 77% | net 100, smash 36 |
| wrist smash (smash) | 1,407 | 82% | drop 142, drive 87 |
| drop (drop) | 1,924 | 83% | smash 176, net 112 |

Compared with the contact model: `rush` (net) drops from 60% to 30% recall, but on a different set
of shots: 294 of the 512 rushes end their rally (a kill at the net) and are no longer predicted, and
141 of the 218 that remain are predicted `drive`, the group a rush that gets returned looks like from
the feet (both players mid-court, a short flight). `push` vs `lob` (1,184 pushes predicted lift) and
`back-court drive` vs `smash` (214) are the same height problem as before. The other types move by
a few points either way.

## Feature-set experiments (feet features, same 5 match folds, same shots)

| Features | Held-out accuracy |
|---|---|
| geometry only (positions, shot vector, `is_cross`, `ball_round`) | 83.7% |
| + timing (time to next hit, since previous hit) | 86.4% |
| + previous shot's length and depth (**the shipped model**) | 86.6% |
| + previous shot's **true** group | 86.9% |

Timing adds 2.7 points on feet features (83.7% -> 86.4%); the previous shot's length and depth add
0.2; the previous shot's **true** group would add 0.4 points but is never known on my footage.

## Predictions in `shots`

| | |
|---|---|
| ShuttleSet shots in `shots` | 36,483 |
| With a prediction | 31,701 |
| Left NULL (no hitter point or no landing point) | 4,782 |
| `type_conf` mean / lowest | 0.89 / 0.31 |
| In-sample agreement with `type_group` (not an accuracy) | 93.4% |
| Shots typed `unknown`, predicted as | NULL 1,250, serve 128, lift 8, clear 6, drive 5, smash 5, net 4, drop 1 |
| Own rows of `match_tracked` with a prediction | 119 of 132 (7 rallies) |
| Own rows of `stand_in_set1` with a prediction | 251 of 287 (36 rallies) |

- `predict` scores both sources with the one feet model. On ShuttleSet, 31,701 rows get a prediction;
  the 4,782 left NULL are the rally-ending shots (no reply), shots without positions, and most
  `unknown`-typed rows.
- The own rows of `match_tracked` (the tracked broadcast rallies, rebuilt from the movement hits with an inferred
  serve per rally, `docs/results/no_shuttle.md`): 119 of 132 predicted; the 13 NULL rows are the last shot of each
  of the 7 rallies and 6 serve rows without a hitter position (the broadcast cuts to a close-up at the serve).
- The own rows of `stand_in_set1` (the full stand-in club set, `docs/results/end_to_end.md`): 251 of 287 predicted;
  the 36 NULL rows are the last shot of each of its 36 rallies. Every serve row has the server's feet on that fixed camera.
- `type_conf` is the model's probability for its chosen group, not calibrated.
- The in-sample agreement is on shots the final model trained on, so it is **not** an accuracy figure.
  Use the held-out 86.6%.

## Notes
- A model file trained on the `contact` set is refused by `predict` (`ModelFeatureMismatchError`
  names the model's feature set): retrain with `shots.train`.
- The earlier contact-point results stay in `docs/results/classifier.md` for reference.
- The model file, report and comparisons live under `data/models/` (gitignored, never committed).
