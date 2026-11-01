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

The model now learns 10 labels instead of the 7 groups: `serve` splits into short and long serves,
`net` into net shots and return nets, and `drive` into pushes and drives (`push` vs `drive`, `return_net`
vs `net` and `serve_short` vs `serve_long` are the splits `docs/results/classifier_fine_types.md` found
to hold up). Each label belongs to one group, so every group-level number below is the label model's
prediction mapped to its group, directly comparable with the earlier group-trained 86.6%: it is
**86.5%**, and the 10 labels themselves are right **82.8%** of the time.

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
  same as before. It is trained on `type_detail` (the 10 labels); each prediction's group is the
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
| **Classifier** | **86.5%** (± 0.6% across folds; folds 85.5, 86.8, 87.0, 86.9, 86.1) |
| Majority class (`net`) | 33.1% |
| Same model, scored on its 10 labels | 82.8% (± 1.2% across folds) |

| Shots | Count | Held-out accuracy |
|---|---|---|
| excluding each rally's first shot | 29,131 | 85.4% |
| rally's last shot (no next hit, so no flight time) | 0 | none predicted |
| all other shots | 31,544 | 86.5% |

Serves stay easy (`ball_round` = 1). The "last shot" row is empty by construction: with feet-only
features a shot without a reply has no landing point, so it is neither trained on nor predicted.

## Per group (held-out)

| Group | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve | 2,393 | 0.997 | 1.000 | 0.998 |
| net | 10,458 | 0.953 | 0.961 | 0.957 |
| clear | 2,734 | 0.946 | 0.931 | 0.938 |
| drop | 3,204 | 0.864 | 0.852 | 0.858 |
| smash | 3,328 | 0.832 | 0.877 | 0.854 |
| lift | 5,259 | 0.768 | 0.841 | 0.803 |
| drive | 4,168 | 0.638 | 0.534 | 0.581 |

## Confusion matrix (held-out; rows = true group, columns = predicted)

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,393** | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | **2,544** | 42 | 45 | 2 | 80 | 21 |
| drop | 0 | 25 | **2,730** | 244 | 132 | 3 | 70 |
| smash | 0 | 32 | 210 | **2,918** | 18 | 2 | 148 |
| net | 4 | 1 | 85 | 20 | **10,050** | 17 | 281 |
| lift | 0 | 48 | 3 | 5 | 38 | **4,423** | 742 |
| drive | 4 | 38 | 89 | 274 | 302 | 1,237 | **2,224** |

## Per label (held-out)

| Label | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| serve_short | 2,033 | 0.996 | 0.998 | 0.997 |
| serve_long | 360 | 0.981 | 0.994 | 0.988 |
| clear | 2,734 | 0.946 | 0.931 | 0.938 |
| net | 7,253 | 0.910 | 0.904 | 0.907 |
| drop | 3,204 | 0.864 | 0.852 | 0.858 |
| smash | 3,328 | 0.832 | 0.877 | 0.854 |
| lift | 5,259 | 0.768 | 0.841 | 0.803 |
| return_net | 3,205 | 0.766 | 0.796 | 0.781 |
| push | 2,682 | 0.585 | 0.485 | 0.530 |
| drive | 1,486 | 0.553 | 0.468 | 0.507 |

## Confusion matrix, labels (held-out; rows = true label, columns = predicted)

| true \ predicted | serve_short | serve_long | net | return_net | drive | push | clear | drop | smash | lift |
|---|---|---|---|---|---|---|---|---|---|---|
| serve_short | **2,028** | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| serve_long | 2 | **358** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| net | 3 | 0 | **6,559** | 487 | 54 | 99 | 0 | 29 | 15 | 7 |
| return_net | 1 | 0 | 452 | **2,552** | 122 | 6 | 1 | 56 | 5 | 10 |
| drive | 0 | 0 | 87 | 150 | **696** | 110 | 22 | 85 | 268 | 68 |
| push | 2 | 2 | 55 | 10 | 116 | **1,302** | 16 | 4 | 6 | 1,169 |
| clear | 0 | 0 | 0 | 2 | 12 | 9 | **2,544** | 42 | 45 | 80 |
| drop | 0 | 0 | 32 | 100 | 66 | 4 | 25 | **2,730** | 244 | 3 |
| smash | 0 | 0 | 13 | 5 | 142 | 6 | 32 | 210 | **2,918** | 2 |
| lift | 0 | 0 | 12 | 26 | 51 | 691 | 48 | 3 | 5 | **4,423** |

- Short vs long serve is almost free: 7 of 2,393 serves get the wrong kind.
- `return_net` vs `net` is right about 4 times in 5 (452 return nets called net shots, 487 net shots
  called return nets): both are played at the net, and what separates them is whether the shot before
  was a net shot, which the model only sees through the previous shot's length and depth.
- `push` is the weakest label (recall 0.485): 1,169 pushes are predicted `lift`, the same height problem
  as `drive` vs `lift` at group level. Inside the `drive` group, push vs drive is mostly right
  (116 pushes called drives, 110 drives called pushes).

## Contact points vs feet, side by side

`shots.compare_hit_points` runs both feature sets through the same 5 match folds, classifier and
jitter. "Shared shots" are the 31,543 labelled shots that both sets can predict, so those two
accuracies are on identical shots; "all its shots" is each set on everything it can predict (for
`feet` that is the shipped model's own evaluation, fold by fold). Both sets are trained on the 10 labels and
scored as groups, like the shipped model; `contact` on all its shots is 86.7% that way, against 86.8% for the
group-trained contact model of `docs/results/classifier.md`.

| Feature set | Shared shots | Held-out accuracy | All its shots | Held-out accuracy |
|---|---|---|---|---|
| contact (labelled hit point and landing point) | 31,543 | **87.6%** (± 1.0%) | 35,056 | **86.7%** (± 1.0%) |
| feet (hitter's feet; landing = the reply hitter's feet) | 31,543 | **86.5%** (± 1.2%) | 31,544 | **86.5%** (± 0.6%) |

| Group | Shots | contact P / R / F1 | feet P / R / F1 | F1 change |
|---|---|---|---|---|
| serve | 2,393 | 0.995 / 1.000 / 0.997 | 0.992 / 0.995 / 0.994 | -0.004 |
| net | 10,457 | 0.964 / 0.968 / 0.966 | 0.955 / 0.961 / 0.958 | -0.008 |
| clear | 2,734 | 0.959 / 0.952 / 0.955 | 0.943 / 0.933 / 0.938 | -0.017 |
| drop | 3,204 | 0.889 / 0.889 / 0.889 | 0.863 / 0.853 / 0.858 | -0.031 |
| smash | 3,328 | 0.847 / 0.881 / 0.864 | 0.825 / 0.872 / 0.848 | -0.016 |
| lift | 5,259 | 0.773 / 0.848 / 0.809 | 0.768 / 0.844 / 0.805 | -0.004 |
| drive | 4,168 | 0.655 / 0.547 / 0.596 | 0.646 / 0.535 / 0.585 | -0.011 |

**Feet cost 1.1 points on the same shots (87.6% -> 86.5%)**, inside one fold's spread (± 1.2%).
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
precision at the ends: `drop` (-0.031 F1) and `clear` (-0.017) are the groups separated by **where
exactly the shuttle comes down** (a drop just over the net vs a net shot; a clear to the baseline vs
a lift), and a receiver who reaches a drop late, or takes a clear a step inside the baseline, blurs that.
`drive` vs `lift` (1,238 drives predicted lift, 725 lifts predicted drive) stays the main error, as
before: it is a matter of height, which neither feature set has.

The two confusion matrices on the shared shots:

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,393** | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | **2,602** | 15 | 34 | 2 | 53 | 28 |
| drop | 0 | 5 | **2,848** | 201 | 108 | 2 | 40 |
| smash | 2 | 29 | 192 | **2,932** | 12 | 4 | 157 |
| net | 4 | 2 | 71 | 9 | **10,124** | 5 | 242 |
| lift | 1 | 41 | 3 | 2 | 17 | **4,461** | 734 |
| drive | 5 | 35 | 73 | 284 | 243 | 1,248 | **2,280** |

| true \ predicted | serve | clear | drop | smash | net | lift | drive |
|---|---|---|---|---|---|---|---|
| serve | **2,382** | 0 | 0 | 1 | 0 | 7 | 3 |
| clear | 4 | **2,551** | 36 | 48 | 2 | 74 | 19 |
| drop | 0 | 33 | **2,733** | 257 | 121 | 4 | 56 |
| smash | 2 | 37 | 233 | **2,901** | 18 | 2 | 135 |
| net | 5 | 1 | 82 | 21 | **10,049** | 15 | 284 |
| lift | 5 | 46 | 4 | 3 | 35 | **4,441** | 725 |
| drive | 3 | 37 | 80 | 285 | 296 | 1,238 | **2,229** |

## Timing jitter

| Hit frames | contact, shared shots | feet, shared shots | feet, all its shots (the shipped model) |
|---|---|---|---|
| exact | **87.6%** | **86.5%** | **86.5%** |
| +/- 5 | 84.4% (-3.2) | 83.0% (-3.5) | 83.0% (-3.5) |
| +/- 10 | 80.7% (-6.9) | 78.2% (-8.3) | 78.3% (-8.2) |

- Hit timing matters about as much for both feature sets: **±5 frames costs 3.5 points, ±10 frames
  8.2 points** on the shipped model. Time to the next hit is the third most important feature and
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
| landing_y | 0.312 |
| hit_y | 0.169 |
| flight_seconds | 0.164 |
| shot_dy | 0.116 |
| ball_round | 0.102 |
| seconds_since_previous_hit | 0.074 |
| shot_distance | 0.011 |
| previous_shot_distance | 0.007 |
| landing_x | 0.007 |
| receiver_y | 0.006 |
| previous_hit_y | 0.005 |
| hit_x | 0.005 |
| shot_lateral_distance | 0.004 |
| receiver_x | 0.003 |
| shot_dx | 0.002 |
| is_cross | 0.000 |

Depth (`landing_y`, `hit_y`, `shot_dy`) and timing still carry the model; sideways position
barely matters. Player tracking must get the feet's depth and the hit frames right; a lateral
error is cheap.

## Weak shot types, and why
Recall per original shot type (held-out), for every type below 85%:

| Type (group) | Shots | Recall | Most common mistakes |
|---|---|---|---|
| driven flight (drive) | 63 | 21% | net 49, smash 1 |
| back-court drive (drive) | 442 | 30% | smash 216, drop 81 |
| rush (net) | 218 | 42% | drive 109, smash 14 |
| push (drive) | 2,682 | 53% | lift 1,169, net 65 |
| defensive return drive (drive) | 353 | 54% | net 89, lift 57 |
| defensive return lob (lift) | 274 | 62% | drive 51, clear 25 |
| drive (drive) | 628 | 75% | net 99, smash 42 |
| wrist smash (smash) | 1,407 | 83% | drop 128, drive 89 |
| drop (drop) | 1,924 | 83% | smash 173, net 102 |

Compared with the contact model: `rush` (net) drops from 60% to 42% recall, but on a different set
of shots: 294 of the 512 rushes end their rally (a kill at the net) and are no longer predicted, and
109 of the 218 that remain are predicted `drive`, the group a rush that gets returned looks like from
the feet (both players mid-court, a short flight). `push` vs `lob` (1,169 pushes predicted lift) and
`back-court drive` vs `smash` (216) are the same height problem as before. The other types move by
a few points either way.

## Feature-set experiments (feet features, same 5 match folds, same shots)

| Features | Held-out accuracy |
|---|---|
| geometry only (positions, shot vector, `is_cross`, `ball_round`) | 83.6% |
| + timing (time to next hit, since previous hit) | 86.2% |
| + previous shot's length and depth (**the shipped model**) | 86.5% |
| + previous shot's **true** group | 86.9% |

Timing adds 2.6 points on feet features (83.6% -> 86.2%); the previous shot's length and depth add
0.3; the previous shot's **true** group would add 0.4 points but is never known on my footage.

## Predictions in `shots`

| | |
|---|---|
| ShuttleSet shots in `shots` | 36,483 |
| With a prediction | 31,701 |
| Left NULL (no hitter point or no landing point) | 4,782 |
| `type_conf` mean / lowest | 0.86 / 0.26 |
| In-sample agreement with `type_group` (not an accuracy) | 92.9% |
| In-sample agreement with `type_detail` (not an accuracy) | 90.7% |
| Shots typed `unknown`, predicted as | NULL 1,250, serve 121, clear 9, lift 8, net 8, smash 6, drive 3, drop 2 |
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
  Use the held-out 86.5% (82.8% for the labels).

## Notes
- A model file trained on the `contact` set is refused by `predict` (`ModelFeatureMismatchError`
  names the model's feature set): retrain with `shots.train`.
- A model file trained on the 7 groups is refused too (`ModelLabelMismatchError`): retrain with
  `shots.train`.
- The earlier contact-point results stay in `docs/results/classifier.md` for reference.
- The model file, report and comparisons live under `data/models/` (gitignored, never committed).
