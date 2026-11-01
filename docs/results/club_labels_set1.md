# Hit detection on the club video, scored against hand labels (stand-in set 1, first labels)

**Superseded.** This page scores the first 153 labels of set 1. The user has since
relabelled set 1 in full (172 hits, 34 rallies, serves short / long, checked rally
bounds), and `docs/results/hits_club.md` scores four models against that relabel. This
page stays as the record of the first labels.

The test copy of the first labels is `data/club_labels/set1_labels_153/`. Scores use
second-based windows; the earlier frame-based (±5 / ±10 / ±20 frames) figures are in
the git history.

The user labelled 153 hits in 25 rallies of `data/recordings/stand_in/set1.mp4` (60 fps,
26,235 frames) with the browser labelling tool. Two movement hit models were run on the
existing set-1 tracks and scored against these labels:

- **five ranges**: the shipped default, `data/models/movement_hits.joblib`, trained on 128
  hits in five ranges of one broadcast match (`docs/results/hits_movement.md`);
- **nine matches**: `data/models/movement_hits_ten_matches.joblib`, trained on about 9,400
  shots of nine ShuttleSet matches (`docs/results/hits_ten_matches.md`).

The default model is not changed.

## How the labels are scored

The labelling is partial: some real hits in the video have no label. So:

- recall, hitter and shot type are measured on the 153 labelled hits only;
- a detection with no label is not counted as a false positive;
- precision is counted only for detections inside the 25 labelled rallies (their start and
  end frames, padded by the window). It is a **lower bound**: a detection there can be a
  real hit the user did not label.

Matching is one to one, closest first, as on ShuttleSet. The windows are **±0.25 s**
(±15 frames at 60 fps) and **±0.50 s** (±30 frames), the same time windows as on the 25
and 30 fps broadcasts (`vision.hits.MATCH_TOLERANCE_SECONDS`).

The rows "detected + inferred serves" also count the serve row the pipeline adds 0.76 s
before each rally's first detected hit (`docs/vision/movement.md`).

Shot types are the feet classifier's `type_pred` (the 8-label model, scored by group).
For each model, the rows are built from that model's hits and predicted inside a copy of
the database (`data/club_labels/<model>/scout.duckdb`), so the shipped `stand_in_set1`
rows are untouched. A labelled hit is compared with the prediction at its matched row;
`unknown` labels and rows with no prediction are left out.

**Labels by source.** All 153 labels are `hand`; none was accepted from a detector
proposal (`a`). So no recall number here is flattered by the old model's own frames.

One small bias remains. The labelling tool showed the five-range model's proposals on the
timeline while the user picked frames, which may pull a few hand frames toward that
model's detections.

| Labelled type | net | lift | serve | drop | drive | clear | smash | unknown | total |
|---|---|---|---|---|---|---|---|---|---|
| hits | 43 | 27 | 24 | 19 | 19 | 17 | 2 | 2 | 153 |

`me` 80, `opp` 73. Each labelled rally holds 3 to 13 hits (median 5).

## Headline

| Model | Tolerance | Rows scored | Recall | Hitter right | Shot type agrees | Precision in labelled rallies (lower bound) |
|---|---|---|---|---|---|---|
| five ranges | ±0.25 s (±15 frames at 60 fps) | detected hits | 105/153 = 68.6% | 101/105 = 96.2% | 68/97 = 70.1% | 105/171 = 61.4% |
| five ranges | ±0.5 s (±30 frames at 60 fps) | detected hits | 120/153 = 78.4% | 110/120 = 91.7% | 79/110 = 71.8% | 120/173 = 69.4% |
| five ranges | ±0.25 s (±15 frames at 60 fps) | detected + inferred serves | 118/153 = 77.1% | 114/118 = 96.6% | 81/110 = 73.6% | 118/195 = 60.5% |
| five ranges | ±0.5 s (±30 frames at 60 fps) | detected + inferred serves | 137/153 = 89.5% | 126/137 = 92.0% | 94/127 = 74.0% | 137/199 = 68.8% |
| nine matches | ±0.25 s (±15 frames at 60 fps) | detected hits | 135/153 = 88.2% | 131/135 = 97.0% | 89/127 = 70.1% | 135/195 = 69.2% |
| nine matches | ±0.5 s (±30 frames at 60 fps) | detected hits | 145/153 = 94.8% | 136/145 = 93.8% | 94/137 = 68.6% | 145/202 = 71.8% |
| nine matches | ±0.25 s (±15 frames at 60 fps) | detected + inferred serves | 137/153 = 89.5% | 133/137 = 97.1% | 91/129 = 70.5% | 137/204 = 67.2% |
| nine matches | ±0.5 s (±30 frames at 60 fps) | detected + inferred serves | 148/153 = 96.7% | 139/148 = 93.9% | 96/140 = 68.6% | 148/212 = 69.8% |

Both models detect over the whole set, and many detections fall outside the labelled
rallies (padded by the ±0.50 s window), where precision is not judged:

- five ranges: 251 detected hits, 78 of them outside the labelled rallies;
- nine matches: 361 detected hits, 159 of them outside the labelled rallies.

Without the padding it's 85 and 184.

## Serves (±0.5 s (±30 frames at 60 fps))

| Model | Labelled serves | Found by an inferred serve row | Found by a detected hit | Missed | Shot type agrees |
|---|---|---|---|---|---|
| five ranges | 24 | 16 | 3 | 5 | 16/19 = 84.2% |
| nine matches | 24 | 2 | 18 | 4 | 2/20 = 10.0% |

The nine-match model often detects the serve itself as a hit. But it does not end rallies
between points: it finds 27 rallies where the five-range model finds 36, and it keeps
detecting hits while the players walk back between points. So a real serve sits deep
inside a detected rally. Of the rows matched to a labelled serve, only 2 are at
`ball_round` 1; the others are at 2 to 34 (five at 4, three at 14). The classifier relies
on `ball_round` 1 for serves, so it calls them net (12), drive (4), lift (2) and
serve (2).

The five-range model ends its rallies earlier, so its inferred serve row lands on the
labelled serve: 16 of its 19 matched serves are at `ball_round` 1, all predicted `serve`.

## By labelled type (±0.5 s (±30 frames at 60 fps), detected + inferred serves)

| type | Labelled | five ranges recall | five ranges hitter | five ranges type | nine matches recall | nine matches hitter | nine matches type |
|---|---|---|---|---|---|---|---|
| net | 43 | 40/43 = 93.0% | 40/40 = 100.0% | 34/37 = 91.9% | 43/43 = 100.0% | 40/43 = 93.0% | 40/42 = 95.2% |
| lift | 27 | 27/27 = 100.0% | 23/27 = 85.2% | 18/26 = 69.2% | 27/27 = 100.0% | 26/27 = 96.3% | 21/25 = 84.0% |
| serve | 24 | 18/24 = 75.0% | 17/18 = 94.4% | 15/18 = 83.3% | 20/24 = 83.3% | 19/20 = 95.0% | 2/20 = 10.0% |
| drive | 19 | 18/19 = 94.7% | 15/18 = 83.3% | 6/18 = 33.3% | 19/19 = 100.0% | 16/19 = 84.2% | 5/18 = 27.8% |
| drop | 19 | 17/19 = 89.5% | 17/17 = 100.0% | 12/17 = 70.6% | 19/19 = 100.0% | 19/19 = 100.0% | 15/19 = 78.9% |
| clear | 17 | 14/17 = 82.4% | 12/14 = 85.7% | 9/9 = 100.0% | 16/17 = 94.1% | 16/16 = 100.0% | 13/14 = 92.9% |
| smash | 2 | 2/2 = 100.0% | 1/2 = 50.0% | 0/2 = 0.0% | 2/2 = 100.0% | 1/2 = 50.0% | 0/2 = 0.0% |
| unknown | 2 | 1/2 = 50.0% | 1/1 = 100.0% | 0/0 | 2/2 = 100.0% | 2/2 = 100.0% | 0/0 |

## By player (±0.5 s (±30 frames at 60 fps), detected + inferred serves)

| hitter | Labelled | five ranges recall | five ranges hitter | five ranges type | nine matches recall | nine matches hitter | nine matches type |
|---|---|---|---|---|---|---|---|
| me | 80 | 69/80 = 86.2% | 63/69 = 91.3% | 51/67 = 76.1% | 78/80 = 97.5% | 72/78 = 92.3% | 45/74 = 60.8% |
| opp | 73 | 68/73 = 93.2% | 63/68 = 92.6% | 43/60 = 71.7% | 70/73 = 95.9% | 67/70 = 95.7% | 51/66 = 77.3% |

## Shot types on matched hits (±0.5 s (±30 frames at 60 fps), detected + inferred serves)

Five ranges:

| labelled \ predicted | (none) | clear | drive | drop | lift | net | serve | smash |
|---|---|---|---|---|---|---|---|---|
| clear | 5 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| drive | 0 | 4 | 6 | 2 | 2 | 1 | 1 | 2 |
| drop | 0 | 1 | 1 | 12 | 0 | 2 | 0 | 1 |
| lift | 1 | 1 | 4 | 0 | 18 | 2 | 1 | 0 |
| net | 3 | 0 | 1 | 0 | 1 | 34 | 0 | 1 |
| serve | 0 | 0 | 1 | 1 | 0 | 1 | 15 | 0 |
| smash | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| unknown | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

Nine matches:

| labelled \ predicted | (none) | clear | drive | drop | lift | net | serve | smash |
|---|---|---|---|---|---|---|---|---|
| clear | 2 | 13 | 0 | 0 | 0 | 0 | 1 | 0 |
| drive | 1 | 3 | 5 | 2 | 2 | 2 | 0 | 4 |
| drop | 0 | 1 | 0 | 15 | 1 | 2 | 0 | 0 |
| lift | 2 | 1 | 2 | 0 | 21 | 1 | 0 | 0 |
| net | 1 | 0 | 1 | 0 | 1 | 40 | 0 | 0 |
| serve | 0 | 0 | 4 | 0 | 2 | 12 | 2 | 0 |
| smash | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| unknown | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

Rows with no prediction (`nan`) are last shots of a detected rally, which have no
landing.

## Labelled hits missed by either model (±0.5 s (±30 frames at 60 fps), detected + inferred serves)

The last two columns say whether each model found the hit. For a miss, `nearest` is the
signed frame offset of that model's nearest detected hit; positive means later.

| Frame | Rally | Player | Type | five ranges | nine matches |
|---|---|---|---|---|---|
| 473 | 1 | me | clear | missed (nearest -64) | found |
| 4263 | 5 | me | serve | found | missed (nearest +44) |
| 4491 | 5 | opp | clear | found | missed (nearest -32) |
| 4804 | 5 | me | drop | missed (nearest +36) | found |
| 5117 | 5 | me | clear | missed (nearest -59) | found |
| 6080 | 7 | me | net | missed (nearest -32) | found |
| 6745 | 8 | opp | serve | missed (nearest +39) | found |
| 7694 | 9 | me | net | missed (nearest -31) | found |
| 8894 | 10 | me | clear | missed (nearest -84) | found |
| 8963 | 10 | opp | drive | missed (nearest -153) | found |
| 10337 | 11 | me | serve | missed (nearest +42) | missed (nearest -41) |
| 11021 | 12 | me | serve | missed (nearest +78) | found |
| 11877 | 13 | me | serve | missed (nearest +76) | found |
| 11969 | 13 | opp | unknown | missed (nearest -16) | found |
| 13841 | 15 | opp | drop | missed (nearest -31) | found |
| 14724 | 16 | me | net | missed (nearest -54) | found |
| 16597 | 19 | opp | serve | missed (nearest +45) | missed (nearest +45) |
| 24711 | 25 | me | serve | missed (nearest +119) | found |
| 25417 | 26 | opp | serve | found | missed (nearest -32) |

## What it says

- **Finding hits.** The nine-match model finds more of the labelled hits:
  - 88.2% against 68.6% at ±0.25 s (detected hits);
  - 94.8% against 78.4% at ±0.50 s.

  Its precision inside the labelled rallies is also higher (69.2% against 61.4% at
  ±0.25 s, lower bounds).
- **Hitter.** 96-97% right for both at ±0.25 s.
- **Shot type.** About the same overall (69-74%), for different reasons. The nine-match
  model loses the serves and the shots right after them, because its rallies run across
  points (above). It gains the lifts (84.0% against 69.2%) and drops (78.9% against
  70.6%).
- **What to fix before switching:** the nine-match model's weak point on club footage is
  rally segmentation between points, not hit detection. This led to the in-play model
  (`docs/results/rally_in_play.md`) and the club-trained scorer
  (`docs/results/hits_club.md`).

## How to run

```
V=data/recordings/stand_in/set1.mp4
for run in five_ranges:data/models/movement_hits.joblib nine_matches:data/models/movement_hits_ten_matches.joblib; do
  name=${run%%:*}; model=${run#*:}
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --model $model --hits-dir data/club_labels/$name/hits
  cp data/scout.duckdb data/club_labels/$name/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id stand_in_set1 --range 0 26234 1 --database data/club_labels/$name/scout.duckdb --hits-dir data/club_labels/$name/hits
  .venv/bin/python -m shots.predict data/club_labels/$name/scout.duckdb
done
D=data/club_labels
.venv/bin/python -m vision.label_scores --labels $D/set1_labels_153/stand_in_set1.csv --match-id stand_in_set1 --fps 60 \
  --run five_ranges $D/five_ranges/hits/stand_in_set1/hits_0_26234.csv $D/five_ranges/scout.duckdb \
  --run nine_matches $D/nine_matches/hits/stand_in_set1/hits_0_26234.csv $D/nine_matches/scout.duckdb
```

Detection takes about 4 s per model on the CPU. The five-range hits file is byte for byte
the shipped `data/hits/stand_in_set1/hits_0_26234.csv`.
