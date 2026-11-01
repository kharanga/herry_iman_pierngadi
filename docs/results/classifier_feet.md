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

The model now learns 8 labels instead of the 7 groups: `serve` splits into short and long serves, and
every other group is its own label. I label return nets as net shots and don't use push on my own footage,
so those two splits (which `docs/results/classifier_fine_types.md` found to hold up on ShuttleSet) are not
used. Each label belongs to one group, so every group-level number below is the label model's prediction
mapped to its group, directly comparable with the earlier group-trained 86.6%: it is **86.3%**, and the
8 labels themselves are right **86.3%** of the time too (a wrong serve kind is rare enough not to show in
the first decimal).

## How to run
```
.venv/bin/python -m shots.train
.venv/bin/python -m shots.compare_hit_points
.venv/bin/python -m shots.compare_features
.venv/bin/python -m shots.predict
.venv/bin/python -m shots.results_tables
```
On the dev MacBook Air: `train` 29 s, `compare_hit_points` 32 s, `compare_features` 23 s,
`predict` 2 s. Every table below is printed by `shots.results_tables` from
`data/models/shot_type_classifier_report.json` (written by `train`),
`data/models/hit_point_comparison.json` (`compare_hit_points`),
`data/models/feature_set_comparison.json` (`compare_features`) and the `shots` table.
Numbers in the text come from the same files.

## Setup
- Model: scikit-learn `HistGradientBoostingClassifier`, default settings, `random_state=0`, the
  same as before. It is trained on `type_detail` (the 8 labels); each prediction's group is the
  group of its label.
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
| **Classifier** | **86.3%** (± 0.7% across folds; folds 85.3, 87.1, 86.9, 86.6, 85.7) |
| Majority class (`net`) | 33.1% |
| Same model, scored on its 8 labels | 86.3% (± 0.7% across folds) |

| Shots | Count | Held-out accuracy |
|---|---|---|
| excluding each rally's first shot | 29,131 | 85.3% |
| rally's last shot (no next hit, so no flight time) | 0 | none predicted |
| all other shots | 31,544 | 86.3% |

Serves stay easy (`ball_round` = 1). The "last shot" row is empty by construction: with feet-only
features a shot without a reply has no landing point, so it is neither trained on nor predicted.

## Per group (held-out)

| Group | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve | 2,393 | 0.995 | 1.000 | 0.997 |
| net | 10,458 | 0.957 | 0.961 | 0.959 |
| clear | 2,734 | 0.947 | 0.931 | 0.939 |
| drop | 3,204 | 0.869 | 0.842 | 0.855 |
| smash | 3,328 | 0.825 | 0.871 | 0.847 |
| lift | 5,259 | 0.767 | 0.837 | 0.801 |
| drive | 4,168 | 0.625 | 0.539 | 0.579 |

## Confusion matrix (held-out; rows = true group, columns = predicted)

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,392** | 0 | 0 | 0 | 0 | 1 | 0 |
| clear | 0 | **2,546** | 34 | 52 | 1 | 58 | 43 |
| drop | 0 | 30 | **2,697** | 273 | 141 | 3 | 60 |
| smash | 1 | 32 | 220 | **2,899** | 14 | 2 | 160 |
| net | 5 | 1 | 71 | 12 | **10,047** | 13 | 309 |
| lift | 2 | 42 | 4 | 3 | 28 | **4,404** | 776 |
| drive | 3 | 38 | 78 | 275 | 265 | 1,262 | **2,247** |

## Per label (held-out)

| Label | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve_short | 2,033 | 0.996 | 0.998 | 0.997 |
| serve_long | 360 | 0.978 | 0.994 | 0.986 |
| net | 10,458 | 0.957 | 0.961 | 0.959 |
| clear | 2,734 | 0.947 | 0.931 | 0.939 |
| drop | 3,204 | 0.869 | 0.842 | 0.855 |
| smash | 3,328 | 0.825 | 0.871 | 0.847 |
| lift | 5,259 | 0.767 | 0.837 | 0.801 |
| drive | 4,168 | 0.625 | 0.539 | 0.579 |

## Confusion matrix, labels (held-out; rows = true label, columns = predicted)

| true \ predicted | serve_short | serve_long | net | drive | clear | drop | smash | lift |
|---|---|---|---|---|---|---|---|---|
| serve_short | **2,029** | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| serve_long | 1 | **358** | 0 | 0 | 0 | 0 | 0 | 1 |
| net | 5 | 0 | **10,047** | 309 | 1 | 71 | 12 | 13 |
| drive | 1 | 2 | 265 | **2,247** | 38 | 78 | 275 | 1,262 |
| clear | 0 | 0 | 1 | 43 | **2,546** | 34 | 52 | 58 |
| drop | 0 | 0 | 141 | 60 | 30 | **2,697** | 273 | 3 |
| smash | 0 | 1 | 14 | 160 | 32 | 220 | **2,899** | 2 |
| lift | 1 | 1 | 28 | 776 | 42 | 4 | 3 | **4,404** |

- Short vs long serve is almost free: 5 of 2,393 serves get the wrong kind (4 short serves called long,
  1 long called short), and 1 long serve is called a lift.
- Every other label is a whole group, so its row is the group's row above.

## Contact points vs feet, side by side

`shots.compare_hit_points` runs both feature sets through the same 5 match folds, classifier and
jitter. "Shared shots" are the 31,543 labelled shots that both sets can predict, so those two
accuracies are on identical shots; "all its shots" is each set on everything it can predict (for
`feet` that is the shipped model's own evaluation, fold by fold). Both sets are trained on the 8 labels and
scored as groups, like the shipped model; `contact` on all its shots is 86.8% that way, the same as the
group-trained contact model of `docs/results/classifier.md`.

| Feature set | Shared shots | Held-out accuracy | All its shots | Held-out accuracy |
|---|---|---|---|---|
| contact (labelled hit point and landing point) | 31,543 | **87.7%** (± 0.9%) | 35,056 | **86.8%** (± 0.9%) |
| feet (hitter's feet; landing = the reply hitter's feet) | 31,543 | **86.7%** (± 1.0%) | 31,544 | **86.3%** (± 0.7%) |

| Group | Shots | contact P / R / F1 | feet P / R / F1 | F1 change |
|---|---|---|---|---|
| serve | 2,393 | 0.997 / 1.000 / 0.998 | 0.997 / 1.000 / 0.998 | -0.000 |
| net | 10,457 | 0.965 / 0.967 / 0.966 | 0.957 / 0.960 / 0.959 | -0.007 |
| clear | 2,734 | 0.960 / 0.954 / 0.957 | 0.946 / 0.931 / 0.939 | -0.018 |
| drop | 3,204 | 0.893 / 0.884 / 0.889 | 0.872 / 0.848 / 0.860 | -0.029 |
| smash | 3,328 | 0.849 / 0.879 / 0.864 | 0.831 / 0.873 / 0.852 | -0.012 |
| lift | 5,259 | 0.781 / 0.843 / 0.811 | 0.777 / 0.842 / 0.808 | -0.003 |
| drive | 4,168 | 0.648 / 0.569 / 0.606 | 0.634 / 0.555 / 0.592 | -0.014 |

**Feet cost 1.0 point on the same shots (87.7% -> 86.7%)**, inside one fold's spread (± 1.0%).
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
precision at the ends: `drop` (-0.029 F1) and `clear` (-0.018) are the groups separated by **where
exactly the shuttle comes down** (a drop just over the net vs a net shot; a clear to the baseline vs
a lift), and a receiver who reaches a drop late, or takes a clear a step inside the baseline, blurs that.
`drive` vs `lift` (1,193 drives predicted lift, 763 lifts predicted drive) stays the main error, as
before: it is a matter of height, which neither feature set has.

The two confusion matrices on the shared shots:

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,393** | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | **2,607** | 14 | 35 | 1 | 37 | 40 |
| drop | 0 | 4 | **2,833** | 202 | 115 | 1 | 49 |
| smash | 2 | 27 | 189 | **2,926** | 14 | 3 | 167 |
| net | 4 | 1 | 65 | 6 | **10,108** | 8 | 265 |
| lift | 1 | 40 | 3 | 2 | 15 | **4,433** | 765 |
| drive | 1 | 38 | 68 | 275 | 222 | 1,194 | **2,370** |

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,392** | 0 | 0 | 0 | 0 | 1 | 0 |
| clear | 0 | **2,546** | 31 | 51 | 0 | 58 | 48 |
| drop | 0 | 34 | **2,718** | 250 | 136 | 2 | 64 |
| smash | 1 | 38 | 217 | **2,905** | 15 | 2 | 150 |
| net | 4 | 1 | 66 | 17 | **10,042** | 14 | 313 |
| lift | 1 | 36 | 3 | 3 | 27 | **4,426** | 763 |
| drive | 2 | 35 | 83 | 268 | 274 | 1,193 | **2,313** |

## Timing jitter

| Hit frames | contact, shared shots | feet, shared shots | feet, all its shots (the shipped model) |
|---|---|---|---|
| exact | **87.7%** | **86.7%** | **86.3%** |
| +/- 5 | 84.6% (-3.1) | 83.2% (-3.4) | 83.2% (-3.2) |
| +/- 10 | 80.9% (-6.8) | 78.7% (-8.0) | 78.6% (-7.7) |

- Hit timing matters about as much for both feature sets: **±5 frames costs 3.2 points, ±10 frames
  7.7 points** on the shipped model. Time to the next hit is the third most important feature and
  the main thing that separates a smash from a drop and a lift from a drive; a 10-frame error at 30 fps
  is a third of a second on flights that take 0.3-1.5 s.
- On my footage the hit frames come from player movement (no shuttle), so the frame error will be
  larger than ShuttleSet's hand-labelled hits. The table says what that costs: the hit detector should
  be judged by how many of its hits land within ±5 frames of the real contact, not just by whether a
  hit was found.
- This jitter moves only the time features. On real footage a wrong hit frame also samples the feet
  at the wrong moment; that extra cost is not measured here (it needs labelled own footage).

## Feature importance (shipped feet model)

Permutation importance: the drop in held-out label accuracy when one feature is shuffled, averaged over
the 5 test folds (3 repeats each). Features are correlated, so a feature can look unimportant while
another carries the same information.

| Feature | Accuracy drop |
|---|---|
| landing_y | 0.336 |
| hit_y | 0.166 |
| flight_seconds | 0.161 |
| ball_round | 0.102 |
| seconds_since_previous_hit | 0.054 |
| shot_dy | 0.041 |
| shot_distance | 0.018 |
| hit_x | 0.004 |
| landing_x | 0.004 |
| previous_hit_y | 0.003 |
| receiver_y | 0.003 |
| previous_shot_distance | 0.002 |
| receiver_x | 0.001 |
| shot_dx | 0.000 |
| is_cross | 0.000 |
| shot_lateral_distance | -0.000 |

Depth (`landing_y`, `hit_y`, `shot_dy`) and timing still carry the model; sideways position
barely matters. Player tracking must get the feet's depth and the hit frames right; a lateral
error is cheap.

## Weak shot types, and why
Recall per original shot type (held-out), for every type below 85%:

| Type (group) | Shots | Recall | Most common mistakes |
|---|---|---|---|
| driven flight (drive) | 63 | 22% | net 48, smash 1 |
| rush (net) | 218 | 29% | drive 146, smash 8 |
| back-court drive (drive) | 442 | 31% | smash 215, drop 74 |
| push (drive) | 2,682 | 52% | lift 1,204, net 47 |
| defensive return drive (drive) | 353 | 58% | net 83, lift 51 |
| defensive return lob (lift) | 274 | 60% | drive 68, clear 20 |
| drive (drive) | 628 | 78% | net 85, smash 45 |
| drop (drop) | 1,924 | 82% | smash 190, net 108 |
| wrist smash (smash) | 1,407 | 82% | drop 136, drive 93 |

Compared with the contact model: `rush` (net) drops from 60% to 29% recall, but on a different set
of shots: 294 of the 512 rushes end their rally (a kill at the net) and are no longer predicted, and
146 of the 218 that remain are predicted `drive`, the group a rush that gets returned looks like from
the feet (both players mid-court, a short flight). `push` vs `lob` (1,204 pushes predicted lift) and
`back-court drive` vs `smash` (215) are the same height problem as before. The other types move by
a few points either way.

## Feature-set experiments (feet features, same 5 match folds, same shots)

| Features | Held-out accuracy |
|---|---|
| geometry only (positions, shot vector, `is_cross`, `ball_round`) | 83.5% |
| + timing (time to next hit, since previous hit) | 86.4% |
| + previous shot's length and depth (**the shipped model**) | 86.3% |
| + previous shot's **true** group | 87.0% |

Timing adds 2.9 points on feet features (83.5% -> 86.4%). The previous shot's length and depth now cost
0.1 (86.4% -> 86.3%), inside the fold spread; they added 0.2-0.3 in earlier runs, so they stay. The previous
shot's **true** group would add 0.7 points but is never known on my footage.

## Predictions in `shots`

| | |
|---|---|
| ShuttleSet shots in `shots` | 36,483 |
| With a prediction | 31,701 |
| Left NULL (no hitter point or no landing point) | 4,782 |
| `type_conf` mean / lowest | 0.89 / 0.29 |
| In-sample agreement with `type_group` (not an accuracy) | 92.5% |
| In-sample agreement with `type_detail` (not an accuracy) | 92.5% |
| Shots typed `unknown`, predicted as | NULL 1,250, serve 128, lift 13, clear 7, smash 4, drive 2, net 2, drop 1 |
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
- `type_detail_pred` is the chosen label and `type_pred` its group; `type_conf` is the model's probability
  for the chosen label, not calibrated.
- The in-sample agreement is on shots the final model trained on, so it is **not** an accuracy figure.
  Use the held-out 86.3% (also 86.3% for the labels).

## Notes
- A model file trained on the `contact` set is refused by `predict` (`ModelFeatureMismatchError`
  names the model's feature set): retrain with `shots.train`.
- A model file trained on the 7 groups is refused too (`ModelLabelMismatchError`): retrain with
  `shots.train`.
- The earlier contact-point results stay in `docs/results/classifier.md` for reference.
- The model file, report and comparisons live under `data/models/` (gitignored, never committed).
