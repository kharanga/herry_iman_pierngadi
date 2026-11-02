# How often the model was right: a reviewed club game

The model labelled a new club game and the user confirmed or corrected every shot in
`vision.label_ui --review`. This page counts how often the model was right as it
labelled.

- **Video:** `n7MqoAVqEW4` (CAM-AM hall, low camera at the near corner), as a 30 fps
  copy. The first 12:05 (21,728 frames) holds the first set.
- **Model:** the current best pipeline:
  - hits and rallies from the combined detector (the default model's rallies and
    serves, plus the nine + Lin + Ding tuned hits at 0.55 / 0.6);
  - shot types from the averaged pro / club classifier.
- **Files:** the model's files are kept untouched in `data/club_review/model_prelabels/`.
  The user's reviewed files were copied to `data/club_review/reviewed/` and read from
  there.

**What this measures.** This is agreement with the model's own proposals. The user
started from the model's frame for each hit and moved it only when it was clearly off,
so the hit-timing numbers lean in the model's favour. The shot types, hitters, false
hits and missed hits don't lean that way: the user judged each one.

**Left out:** 2 hits the user didn't review (`source = model`, frames 1423 and 3121) are
left out of every count below.

## Status

| | Count |
|---|---|
| Model hits reviewed | 292 (213 kept, 79 rejected) |
| Kept exactly as the model had them (frame, type and hitter) | 167/292 = 57.2% |
| Kept with the type corrected | 43 |
| Kept with the hitter corrected | 1 |
| Kept and moved | 6 |
| Rejected | 79 |
| Added by the user (missed by the model) | 40 |
| Left unreviewed (`source = model`), left out of everything here | 2 |

Rallies: 48, 9 with checked bounds.

## Shot type on kept model hits

| model \ user | serve | net | lift | drive | clear | drop | smash | unknown | right |
|---|---|---|---|---|---|---|---|---|---|
| serve | 21 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 21/24 = 87.5% |
| net | 2 | 43 | 4 | 1 | 0 | 1 | 0 | 0 | 43/51 = 84.3% |
| lift | 1 | 0 | 38 | 0 | 1 | 0 | 0 | 0 | 38/40 = 95.0% |
| drive | 1 | 1 | 0 | 15 | 0 | 0 | 1 | 0 | 15/18 = 83.3% |
| clear | 0 | 0 | 0 | 0 | 21 | 1 | 0 | 0 | 21/22 = 95.5% |
| drop | 0 | 0 | 0 | 0 | 3 | 18 | 0 | 0 | 18/21 = 85.7% |
| smash | 0 | 2 | 0 | 2 | 1 | 0 | 9 | 0 | 9/14 = 64.3% |
| unknown | 1 | 1 | 3 | 0 | 3 | 4 | 6 | 5 | - |

## Hitter on kept model hits

| Model hitter | Kept hits | Hitter right |
|---|---|---|
| me | 108 | 108/108 = 100.0% |
| opp | 105 | 104/105 = 99.0% |
| all | 213 | 212/213 = 99.5% |

## Moved hits

Lag = model frame minus the user's frame; positive means the model's hit came after the user's (6 moved hits):

| Moved hits | n | Lags (frames) | Median | Mean |
|---|---|---|---|---|
| all | 6 | +1, +3, +6, +8, +12, +12 | +7.0 f = +0.23 s | +7.0 f = +0.23 s |
| hitter me | 1 | +3 | +3.0 f = +0.10 s | +3.0 f = +0.10 s |
| hitter opp | 5 | +1, +6, +8, +12, +12 | +8.0 f = +0.27 s | +7.8 f = +0.26 s |
| type serve | 1 | +6 | +6.0 f = +0.20 s | +6.0 f = +0.20 s |
| type net | 1 | +12 | +12.0 f = +0.40 s | +12.0 f = +0.40 s |
| type lift | 1 | +1 | +1.0 f = +0.03 s | +1.0 f = +0.03 s |
| type drive | 1 | +12 | +12.0 f = +0.40 s | +12.0 f = +0.40 s |
| type drop | 1 | +3 | +3.0 f = +0.10 s | +3.0 f = +0.10 s |
| type smash | 1 | +8 | +8.0 f = +0.27 s | +8.0 f = +0.27 s |

## Rejected and added, with re-timed hits (a rejection with a hit the user added within 1 s, any rally, each added hit used once)

| Type | Rejected (model's type) | of which re-added nearby | False hits (rejected, not re-added) | Added (user's type) | of which re-timed | Missed (added, not re-timed) |
|---|---|---|---|---|---|---|
| serve | 16 | 3 | 13 | 11 | 1 | 10 |
| net | 19 | 2 | 17 | 5 | 1 | 4 |
| lift | 4 | 0 | 4 | 7 | 1 | 6 |
| drive | 2 | 1 | 1 | 1 | 1 | 0 |
| clear | 2 | 0 | 2 | 7 | 3 | 4 |
| drop | 4 | 0 | 4 | 3 | 2 | 1 |
| smash | 2 | 1 | 1 | 5 | 0 | 5 |
| unknown | 30 | 2 | 28 | 1 | 0 | 1 |
| all | 79 | 9 | 70 | 40 | 9 | 31 |

Re-timed pairs: 9, 6 with the same hitter; frame gaps (added minus model): -13, -11, -4, -3, -2, +20, +20, +25, +27

Sensitivity, re-timed pairs at other reaches (all types / model serves):

- within 1 s: 9 / 3; false hits 70 of 79 rejections
- within 1.5 s: 10 / 3; false hits 69 of 79 rejections
- within 2 s: 10 / 3; false hits 69 of 79 rejections

## What it says

- **More than half of the model's hits were kept exactly as labelled.** 167 of 292
  (57.2%) kept the model's frame, type and hitter. 213 of 292 (72.9%) were kept at all,
  with or without a fix.
- **Hitters are nearly always right:** 212 of 213 kept hits (99.5%).
- **Shot types on kept hits are right 85-95% of the time,** except smashes (9 of 14):
  - smashes were relabelled as net (2), drive (2) or clear (1);
  - the 23 kept hits the model left untyped (each rally's last shot, which has no
    landing point) were mostly smashes (6), drops (4), clears (3) and lifts (3); the
    user left 5 of them unknown;
  - serves typed serve: 21 of 24.
- **Timing:** the 6 moved hits were all late, by 1-12 frames (median +7 frames, 0.23 s).
  The opponent's hits were later (median +8 frames) than the user's own (one hit, +3).
  Six hits is too few for a firm number, and the model-favouring bias above means real
  lags may be larger.
- **False hits: 79 rejected, of which 9 were re-timed.** A re-timed hit is one where the
  user added a hit within 1 s instead, so the model had the shot at the wrong frame.
  That leaves 70 false hits: 28 untyped rally-end rows, 17 net shots and 13 serves,
  among others.
  - **Rejected serves aren't mistimed serves.** None of the 16 was re-added as a serve.
    3 had a different shot added within 1 s (a drop, a drop and a lift), so the model
    put its serve row on a mid-rally hit. The other 13 are rally starts that aren't
    rallies.
  - Widening the reach to 1.5 s or 2 s adds 1 more re-timed pair, so 69 false hits.
- **Missed hits: 40 added, of which 9 were re-timed and 31 genuinely missed.** Of those
  31, 10 are serves: 1 of the 11 added serves replaced a rejected untyped row 0.8 s
  earlier, and the other 10 are serves the model never had. The rest are 6 lifts,
  5 smashes, 4 net shots, 4 clears, a drop and one the user left unknown.
- **Rallies:** the user kept 48 rallies, against the model's 52, but checked the bounds
  of only 9. So rally bounds aren't scored here.

## How to run

```
.venv/bin/python -m vision.review_report data/recordings/club/n7MqoAVqEW4_30.mp4 --labels data/club_review/reviewed/club_n7MqoAVqEW4_30.csv
```

The pipeline that wrote the model's labels:

```
V=data/recordings/club/n7MqoAVqEW4_30.mp4; M=club_n7MqoAVqEW4_30; E=21727; D=data/club_review
.venv/bin/python -m vision.movement $V --start 0 --end $E --rallies-from data/models/movement_hits.joblib \
  --model data/models/movement_hits_nine_plus_lin_ding_tuned.joblib --hit-score-min 0.55 --rally-end-score-min 0.6 --hits-dir $D/hits
cp data/scout.duckdb $D/scout.duckdb
.venv/bin/python -m vision.features $V --match-id $M --range 0 $E 1 --database $D/scout.duckdb --hits-dir $D/hits
.venv/bin/python -m vision.club_classifier --train <Lin> --train <Ding> --predict $V $D/scout.duckdb $E $D/typed   # as in shot_classifier_club.md
.venv/bin/python -m vision.prelabel $V --hits $D/hits/$M/hits_0_$E.csv --database $D/typed/average.duckdb --match-id $M
```
