# Hit detection on the club video, scored against hand labels (stand-in set 1)

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
  end frames in `data/labels/stand_in_set1_rallies.csv`, padded by the tolerance). It is
  a **lower bound**: a detection there can be a real hit the user did not label.

Matching is one to one, closest first, as on ShuttleSet. The tolerances are in video
frames at 60 fps:

- ±5 = 83 ms;
- ±10 = 167 ms, the same time as ±5 at ShuttleSet's 30 fps;
- ±20 = 333 ms, the same time as ±10 at 30 fps.

The rows "detected + inferred serves" also count the serve row the pipeline adds 0.76 s
before each rally's first detected hit (`docs/vision/movement.md`).

Shot types are the feet classifier's `type_pred`. For each model, the rows are built from
that model's hits and predicted inside a copy of the database
(`data/club_labels/<model>/scout.duckdb`), so the shipped `stand_in_set1` rows are untouched.
A labelled hit is compared with the prediction at its matched row; `unknown` labels and
rows with no prediction are left out.

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
| five ranges | ±5 | detected hits | 65/153 = 42.5% | 64/65 = 98.5% | 46/61 = 75.4% | 65/168 = 38.7% |
| five ranges | ±10 | detected hits | 98/153 = 64.1% | 94/98 = 95.9% | 66/91 = 72.5% | 98/169 = 58.0% |
| five ranges | ±20 | detected hits | 110/153 = 71.9% | 105/110 = 95.5% | 74/101 = 73.3% | 110/171 = 64.3% |
| five ranges | ±5 | detected + inferred serves | 76/153 = 49.7% | 75/76 = 98.7% | 57/72 = 79.2% | 76/192 = 39.6% |
| five ranges | ±10 | detected + inferred serves | 111/153 = 72.5% | 107/111 = 96.4% | 79/104 = 76.0% | 111/193 = 57.5% |
| five ranges | ±20 | detected + inferred serves | 126/153 = 82.4% | 121/126 = 96.0% | 89/117 = 76.1% | 126/195 = 64.6% |
| nine matches | ±5 | detected hits | 91/153 = 59.5% | 87/91 = 95.6% | 64/85 = 75.3% | 91/179 = 50.8% |
| nine matches | ±10 | detected hits | 121/153 = 79.1% | 117/121 = 96.7% | 80/113 = 70.8% | 121/186 = 65.1% |
| nine matches | ±20 | detected hits | 141/153 = 92.2% | 134/141 = 95.0% | 87/133 = 65.4% | 141/197 = 71.6% |
| nine matches | ±5 | detected + inferred serves | 92/153 = 60.1% | 88/92 = 95.7% | 65/86 = 75.6% | 92/188 = 48.9% |
| nine matches | ±10 | detected + inferred serves | 123/153 = 80.4% | 119/123 = 96.7% | 82/115 = 71.3% | 123/195 = 63.1% |
| nine matches | ±20 | detected + inferred serves | 143/153 = 93.5% | 136/143 = 95.1% | 89/135 = 65.9% | 143/206 = 69.4% |

Both models detect over the whole set, and many detections fall outside the labelled
rallies (padded by 10 frames, as in the ±10 precision), where precision is not judged:

- five ranges: 251 detected hits, 82 of them outside the labelled rallies;
- nine matches: 361 detected hits, 175 of them outside the labelled rallies.

Without the padding it's 85 and 184.

## Serves

| Model | Labelled serves | Found by an inferred serve row | Found by a detected hit | Missed | Shot type agrees |
|---|---|---|---|---|---|
| five ranges | 24 | 13 | 2 | 9 | 13/15 = 86.7% |
| nine matches | 24 | 2 | 7 | 15 | 2/9 = 22.2% |

At ±20, the five-range model finds 18 serves (15 by an inferred serve row) and the
nine-match model 20 (18 by a detected hit). Its shot type agrees on 2 of 20.

The nine-match model often detects the serve itself as a hit. But it does not end rallies
between points: it finds 27 rallies where the five-range model finds 36, and it keeps
detecting hits while the players walk back between points. So a real serve sits deep
inside a detected rally. The serves it matched at ±20 were at `ball_round` 4 (five times),
14 (three times), and up to 34; only 2 were at `ball_round` 1. The classifier relies on
`ball_round` 1 for serves, so it calls them net (12), drive (4), lift (2) and serve (2).

The five-range model ends its rallies earlier, so its inferred serve row lands on the
labelled serve: 15 of its 18 matched serves are at `ball_round` 1, all predicted `serve`.

## By labelled type (±10, detected + inferred serves)

| type | Labelled | five ranges recall | five ranges hitter | five ranges type | nine matches recall | nine matches hitter | nine matches type |
|---|---|---|---|---|---|---|---|
| net | 43 | 35/43 = 81.4% | 35/35 = 100.0% | 30/32 = 93.8% | 37/43 = 86.0% | 36/37 = 97.3% | 33/36 = 91.7% |
| lift | 27 | 21/27 = 77.8% | 20/21 = 95.2% | 13/20 = 65.0% | 25/27 = 92.6% | 24/25 = 96.0% | 18/23 = 78.3% |
| serve | 24 | 15/24 = 62.5% | 14/15 = 93.3% | 13/15 = 86.7% | 9/24 = 37.5% | 9/9 = 100.0% | 2/9 = 22.2% |
| drive | 19 | 14/19 = 73.7% | 12/14 = 85.7% | 5/14 = 35.7% | 16/19 = 84.2% | 14/16 = 87.5% | 3/15 = 20.0% |
| drop | 19 | 16/19 = 84.2% | 16/16 = 100.0% | 12/16 = 75.0% | 18/19 = 94.7% | 18/18 = 100.0% | 14/18 = 77.8% |
| clear | 17 | 8/17 = 47.1% | 8/8 = 100.0% | 6/6 = 100.0% | 15/17 = 88.2% | 15/15 = 100.0% | 12/13 = 92.3% |
| smash | 2 | 1/2 = 50.0% | 1/1 = 100.0% | 0/1 = 0.0% | 1/2 = 50.0% | 1/1 = 100.0% | 0/1 = 0.0% |
| unknown | 2 | 1/2 = 50.0% | 1/1 = 100.0% | 0/0 | 2/2 = 100.0% | 2/2 = 100.0% | 0/0 |

## By player (±10, detected + inferred serves)

| hitter | Labelled | five ranges recall | five ranges hitter | five ranges type | nine matches recall | nine matches hitter | nine matches type |
|---|---|---|---|---|---|---|---|
| me | 80 | 57/80 = 71.2% | 55/57 = 96.5% | 42/56 = 75.0% | 61/80 = 76.2% | 58/61 = 95.1% | 34/57 = 59.6% |
| opp | 73 | 54/73 = 74.0% | 52/54 = 96.3% | 37/48 = 77.1% | 62/73 = 84.9% | 61/62 = 98.4% | 48/58 = 82.8% |

## Shot types on matched hits (±10, detected + inferred serves)

Five ranges:

| labelled \ predicted | (none) | clear | drive | drop | lift | net | serve | smash |
|---|---|---|---|---|---|---|---|---|
| clear | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| drive | 0 | 4 | 5 | 2 | 0 | 1 | 0 | 2 |
| drop | 0 | 0 | 0 | 12 | 0 | 2 | 0 | 2 |
| lift | 1 | 1 | 4 | 0 | 13 | 2 | 0 | 0 |
| net | 3 | 0 | 1 | 0 | 1 | 30 | 0 | 0 |
| serve | 0 | 0 | 0 | 1 | 0 | 1 | 13 | 0 |
| smash | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| unknown | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

Nine matches:

| labelled \ predicted | (none) | clear | drive | drop | lift | net | serve | smash |
|---|---|---|---|---|---|---|---|---|
| clear | 2 | 12 | 0 | 0 | 0 | 0 | 0 | 1 |
| drive | 1 | 4 | 3 | 2 | 0 | 2 | 0 | 4 |
| drop | 0 | 1 | 0 | 14 | 0 | 3 | 0 | 0 |
| lift | 2 | 1 | 3 | 0 | 18 | 1 | 0 | 0 |
| net | 1 | 0 | 2 | 0 | 1 | 33 | 0 | 0 |
| serve | 0 | 0 | 1 | 0 | 0 | 6 | 2 | 0 |
| smash | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| unknown | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

Rows with `(none)` have no prediction: the last shot of a detected rally has no landing.

## Labelled hits missed by either model (±10, detected + inferred serves)

The last two columns say whether each model found the hit. For a miss, `nearest` is the
signed frame offset of that model's nearest detected hit; positive means later.

| Frame | Rally | Player | Type | five ranges | nine matches |
|---|---|---|---|---|---|
| 169 | 1 | me | net | missed (nearest -13) | missed (nearest -18) |
| 388 | 1 | opp | clear | missed (nearest +21) | found |
| 473 | 1 | me | clear | missed (nearest -64) | found |
| 783 | 2 | me | net | found | missed (nearest -12) |
| 840 | 2 | opp | net | missed (nearest -24) | found |
| 1831 | 3 | opp | serve | missed (nearest +17) | missed (nearest +13) |
| 2219 | 3 | opp | drive | missed (nearest +11) | missed (nearest +12) |
| 3546 | 4 | me | serve | found | missed (nearest +17) |
| 3608 | 4 | opp | lift | missed (nearest -16) | missed (nearest -14) |
| 4263 | 5 | me | serve | found | missed (nearest +44) |
| 4491 | 5 | opp | clear | missed (nearest -30) | missed (nearest -32) |
| 4564 | 5 | me | clear | missed (nearest +12) | found |
| 4804 | 5 | me | drop | missed (nearest +36) | missed (nearest +30) |
| 5039 | 5 | opp | clear | missed (nearest +19) | missed (nearest +74) |
| 5117 | 5 | me | clear | missed (nearest -59) | found |
| 5274 | 6 | opp | drive | missed (nearest +12) | found |
| 6080 | 7 | me | net | missed (nearest -32) | missed (nearest -14) |
| 6745 | 8 | opp | serve | missed (nearest +39) | found |
| 6807 | 8 | me | lift | missed (nearest -23) | found |
| 7694 | 9 | me | net | missed (nearest -31) | found |
| 7750 | 9 | opp | net | missed (nearest -19) | found |
| 8502 | 10 | opp | serve | found | missed (nearest -13) |
| 8629 | 10 | me | drop | missed (nearest +18) | found |
| 8894 | 10 | me | clear | missed (nearest -84) | found |
| 8963 | 10 | opp | drive | missed (nearest -153) | found |
| 9515 | 11 | me | serve | missed (nearest +64) | missed (nearest -11) |
| 9843 | 11 | opp | drive | missed (nearest +22) | missed (nearest +21) |
| 10337 | 11 | me | serve | missed (nearest +42) | missed (nearest -41) |
| 11021 | 12 | me | serve | missed (nearest +78) | missed (nearest -12) |
| 11275 | 12 | me | net | found | missed (nearest -28) |
| 11877 | 13 | me | serve | missed (nearest +76) | missed (nearest -17) |
| 11924 | 13 | opp | drive | missed (nearest +29) | missed (nearest -11) |
| 11969 | 13 | opp | unknown | missed (nearest -16) | found |
| 12683 | 14 | me | serve | found | missed (nearest -13) |
| 12793 | 14 | me | lift | found | missed (nearest -14) |
| 12971 | 14 | me | net | missed (nearest -25) | missed (nearest +25) |
| 13026 | 14 | opp | net | found | missed (nearest +17) |
| 13621 | 15 | me | serve | found | missed (nearest -16) |
| 13841 | 15 | opp | drop | missed (nearest -31) | found |
| 14502 | 16 | me | serve | found | missed (nearest -14) |
| 14724 | 16 | me | net | missed (nearest -54) | found |
| 15845 | 18 | opp | lift | missed (nearest -11) | found |
| 16597 | 19 | opp | serve | missed (nearest +45) | missed (nearest +45) |
| 16724 | 19 | me | smash | missed (nearest -23) | missed (nearest -20) |
| 18653 | 21 | me | serve | found | missed (nearest -12) |
| 19027 | 21 | me | clear | missed (nearest +11) | found |
| 20618 | 22 | me | serve | missed (nearest +65) | found |
| 20837 | 22 | opp | clear | missed (nearest +11) | found |
| 22211 | 23 | me | serve | found | missed (nearest -13) |
| 23200 | 23 | opp | net | missed (nearest -27) | found |
| 24711 | 25 | me | serve | missed (nearest +119) | found |
| 24758 | 25 | me | lift | missed (nearest +72) | found |
| 25417 | 26 | opp | serve | found | missed (nearest -32) |
| 25519 | 26 | me | lift | missed (nearest -30) | found |
| 25752 | 26 | me | lift | missed (nearest -21) | found |

## What it says

- **Finding hits.** The nine-match model finds more of the labelled hits at every
  tolerance:
  - 79.1% vs 64.1% at ±10 (detected hits);
  - 92.2% vs 71.9% at ±20.

  Its gain is largest on clears (15 vs 8 of 17), lifts and drops: shots traded from the
  back, which the five-range model was known to miss. Its precision inside the labelled
  rallies is also higher (65.1% vs 58.0% at ±10, lower bounds).
- **Hitter.** About 96% right for both models.
- **Shot type.** Overall agreement is about the same, 71% vs 76% at ±10, but for
  different reasons:
  - The nine-match model loses the serves and the shots right after them. Its rallies
    run across points, as described under Serves.
  - It gains the clears.
  - Its `me` type agreement falls to 59.6%, because 18 of the 24 labelled serves are
    `me`'s.
- **What to fix before switching.** The nine-match model's weak point on club footage is
  rally segmentation between points, not hit detection. Two cheap things to try before
  making it the default:
  - its rally-end threshold;
  - running it with `--no-serves`.
- **Timing.** Many misses at ±10 have a detection 11-20 frames away. At 60 fps the
  detector's timing is looser than ±10 frames, which is why ±20 is reported too.

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
.venv/bin/python -m vision.label_scores --labels data/labels/stand_in_set1.csv --match-id stand_in_set1 \
  --run five_ranges $D/five_ranges/hits/stand_in_set1/hits_0_26234.csv $D/five_ranges/scout.duckdb \
  --run nine_matches $D/nine_matches/hits/stand_in_set1/hits_0_26234.csv $D/nine_matches/scout.duckdb \
  --tolerance 5 --tolerance 10 --tolerance 20
```

Detection takes about 4 s per model on the CPU. The five-range hits file comes out byte
for byte equal to the shipped `data/hits/stand_in_set1/hits_0_26234.csv`. `--breakdown-tolerance 20`
gives the ±20 serve numbers above.
