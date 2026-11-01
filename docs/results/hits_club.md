# Hit scorer trained with a club match (Lin v Yeung)

The movement hit scorer was trained on nine ShuttleSet broadcast matches
(`docs/results/hits_ten_matches.md`). This adds the user's fully hand-labelled club
match, Lin v Yeung (`q3pfhnvKPig`, 24 fps, 421 hits in 67 rallies), as a training
source. Lin is training data only and is never scored. The result is shipped as its own
model files; the default scorer is unchanged.

All numbers on this page use:

- Lin's tracks after the player-choice fix (`docs/results/club_tracks_clean.md`: rebuilt
  from the saved detections);
- the second-based scoring windows, **±0.25 s** and **±0.50 s**, turned into frames at
  each video's fps (`vision.hits.MATCH_TOLERANCE_SECONDS`).

Earlier versions of this page used Lin's tracks from before that fix and frame-based
windows (±5 / ±10 frames); they are in the git history.

## How Lin is used

- **Tracks:** Lin's whole-video tracks, with the rows flagged `doubtful` dropped.
- **Hit frames:** frames within the scaled `label_tolerance_frames` (2 frames at 24 fps)
  of a labelled hit of that player.
- **Negatives:** every other tracked frame of the whole video, since Lin is labelled in
  full.
- **Rows:** 57,530 player-frames, 2,105 of them hit frames (3.7%; in the nine broadcast
  matches it is 7.8%, because only their rallies are tracked).
- **Weighting, fixed before any test:** Lin carries one third of the total sample weight,
  and the nine matches share the other two thirds equally. Each source's weight is spread
  evenly over its rows (`vision.hit_training --club ... --club-weight`).
- **Sensitivity**, reported only and not used to choose: weights of 1/6 and 1/2.
- **Thresholds, tuned:** chosen nested, on the training sources only, never on set 1 or
  on the held-out match:
  - the grid of `--tune` (hit score 0.35-0.65 x rally-end score 0.5-0.8), by F1 at
    ±0.25 s;
  - each training source (the other eight matches and Lin) scored by a model trained
    without it;
  - Lin's 421 hits are pooled with the matches' hits in the F1, so Lin has a small say
    in the choice.

## ShuttleSet, held out by match (9 matches, 9,424 hits)

Each match is scored by a model trained on the other eight plus Lin.

| Model | Precision ±0.25 s | Recall ±0.25 s | Precision ±0.50 s | Recall ±0.50 s | Hitter ±0.25 s |
|---|---|---|---|---|---|
| Nine matches (`hits_ten_matches.md`) | 85.2% | **84.6%** | 89.8% | 89.2% | 98.5% |
| Nine + Lin, weight 1/3, shipped thresholds (0.5 / 0.7) | **88.3%** | 81.1% | 92.9% | 85.3% | 98.5% |
| Nine + Lin, weight 1/6 (sensitivity) | 87.8% | 82.9% | 92.1% | 87.1% | 98.5% |
| Nine + Lin, weight 1/2 (sensitivity) | 87.9% | 78.5% | 93.1% | 83.1% | 98.5% |
| **Nine + Lin, weight 1/3, tuned thresholds** | 85.9% | **85.2%** | 90.8% | 90.0% | 98.6% |

By frame rate (±0.25 s, precision / recall):

| Model | 25 fps matches (±6 frames) | 30 fps matches (±8 frames) |
|---|---|---|
| Nine matches | 84.5% / 87.3% | 85.7% / 82.7% |
| Nine + Lin, 1/3, shipped thresholds | 87.5% / 83.3% | 88.9% / 79.6% |
| Nine + Lin, 1/3, tuned | 85.3% / 86.7% | 86.4% / 84.2% |

**Effect of the thresholds:**

- **Shipped thresholds:** adding Lin trades recall for precision (precision +3.1 points,
  recall -3.5). That misses the acceptance bar of no more than about 1 point lost. The
  more weight Lin gets, the fewer hits are reported: 9,354 detections at weight 0,
  8,905 at 1/6, 8,658 at 1/3, 8,416 at 1/2.
- **Tuned:** the bar is met, with both above the nine-match model (85.9% / 85.2% against
  85.2% / 84.6%). Every held-out match chose hit score **0.55** (shipped default 0.5) and
  rally-end score **0.5** (default 0.7). Tuned on all ten sources for the shipped model,
  it is also 0.55 / 0.5.
- **Where recall comes back:** mostly from trimming fewer hits at rally ends, not from
  accepting weaker hits.

## Stand-in set 1 after the user's relabel (172 hits, 34 rallies, 28 with checked bounds)

Set 1 is never trained on.

- **Labels:** the user relabelled it in blank mode: every hit by hand, serves split into
  short and long, rally bounds set with `b` / `e`. It now has 172 hits, 1 serve without
  short / long, 34 rallies, 28 of them with both bounds checked, and 2 holding no hits.
- **Test files:** a read-only copy, `data/club_labels/set1_labels_relabel/`. A second
  copy keeps only the 28 checked rallies and their 136 hits
  (`set1_labels_relabel_checked/`).
- **Tracks:** set 1's current tracks (unchanged by the cleanup, see
  `club_tracks_clean.md`).
- **Rows:** built and predicted in copies of the database, with the 8-label classifier.

The set is labelled in full, so precision inside the labelled rallies is a real precision
there.

| | Five ranges (default) | Nine matches | Nine + Lin (1/3), shipped thresholds | **Nine + Lin (1/3), tuned 0.55 / 0.5** |
|---|---|---|---|---|
| Recall ±0.25 s, detected hits | 113/172 = 65.7% | 152/172 = **88.4%** | 115/172 = 66.9% | 139/172 = 80.8% |
| Recall ±0.50 s, detected hits | 128/172 = 74.4% | 162/172 = **94.2%** | 126/172 = 73.3% | 151/172 = 87.8% |
| Recall ±0.25 s, with inferred serves | 130/172 = 75.6% | 154/172 = **89.5%** | 128/172 = 74.4% | 142/172 = 82.6% |
| Hitter right on matched hits (±0.25 s, detected) | 106/113 = 93.8% | 143/152 = 94.1% | 111/115 = **96.5%** | 131/139 = 94.2% |
| Precision inside the labelled rallies, ±0.25 s, detected | 113/176 = 64.2% | 152/229 = 66.4% | 115/153 = **75.2%** | 139/186 = 74.7% |
| Precision inside the labelled rallies, ±0.50 s, detected | 128/182 = 70.3% | 162/240 = 67.5% | 126/157 = **80.3%** | 151/192 = 78.6% |
| Shot type agrees, group (±0.50 s, with serves) | 103/139 = **74.1%** | 103/158 = 65.2% | 90/128 = 70.3% | 97/144 = 67.4% |
| Shot type agrees, 8-label detail (±0.50 s, with serves) | 101/139 = **72.7%** | 103/157 = 65.6% | 89/128 = 69.5% | 94/144 = 65.3% |
| Labelled serves (31): found / typed `serve` | 23 / **20** | 27 / 3 | 16 / 11 | 24 / 5 |

Rally bounds against the user's rallies, as `vision.label_scores` prints them. It scores
the 32 rallies that hold labelled hits. The other 2 of the 34 hold no hits and are left
out: rally 1 (frames 11-33) and rally 7 (frames 4190-4208), at 0.4 s and 0.3 s, both
unchecked, each ending one frame before the next rally starts. They look like leftovers of
the relabel and are worth a look. "Extra" counts detected rallies that overlap none of the
user's; with the whole set labelled, these are rallies the detector found between points.

| | Five ranges | Nine matches | Nine + Lin | Nine + Lin, tuned |
|---|---|---|---|---|
| 32 rallies with hits: detected / matched | 36 / 29 | 27 / 22 | 25 / 25 | 40 / **30** |
| 32: merged / split / missed / extra | 2 / 2 / 1 / 5 | **10** / 3 / 0 / 4 | 4 / 1 / 4 / **0** | 2 / 4 / 1 / 7 |
| 32: start / end error, median abs | 1.00 s / 2.90 s | 1.80 s / 3.18 s | 1.43 s / 1.82 s | **0.82 s / 1.62 s** |
| 28 checked: matched | 25 | 19 | 23 | **27** |
| 28 checked: merged / split / missed / extra | 2 / 1 / 1 / 10 | **8** / 2 / 0 / 7 | 2 / 1 / 4 / 2 | **0** / 3 / 1 / 10 |
| 28 checked: start / end error, median abs | 1.15 s / 2.90 s | 1.70 s / 3.27 s | 1.43 s / 1.58 s | **0.63 s / 1.58 s** |

Sensitivity, at the shipped thresholds (reported only):

| | Nine + Lin, 1/6 | Nine + Lin, 1/2 |
|---|---|---|
| Recall ±0.25 s / ±0.50 s, detected | 75.6% / 79.7% | 58.7% / 66.3% |
| Precision inside rallies ±0.25 s / ±0.50 s | 71.0% / 71.7% | 72.1% / 80.3% |
| Serves found / typed `serve` | 18 / 8 | 10 / 8 |
| 32 rallies with hits: merged / split / missed / extra | 4 / 2 / 1 / 3 | 3 / 1 / 6 / 0 |

## What it says

- **On the broadcast matches, held out:**
  - adding Lin at the shipped thresholds makes the scorer more precise but finds fewer
    hits (88.3% / 81.1%);
  - with the thresholds tuned on the training sources it beats the nine-match model on
    both (85.9% / 85.2% against 85.2% / 84.6%).
- **On set 1, against the relabel, the four models trade off:**
  - **Nine matches** finds the most hits (88.4% at ±0.25 s). But it merges 10 of the 32
    rallies, and only 3 of its 27 matched serves are typed a serve.
  - **Nine + Lin, shipped thresholds,** is the most precise inside rallies (75.2%) with
    the best hitter (96.5%), and finds no rallies between points. It finds as few hits as
    the default (66.9%) and misses 4 rallies.
  - **Nine + Lin, tuned,** keeps most of that precision (74.7%) and gets recall back to
    80.8%. It matches 27 of the 28 checked rallies with none merged, and has the closest
    rally starts and ends (0.63 s / 1.58 s on the checked ones). But it finds 7 rallies
    between points, and types only 5 of 24 matched serves as serves.
  - **The default (five ranges)** still types serves best (20 of 23) and has the best
    shot-type agreement (74.1%), because its rallies start on the serve. It finds the
    fewest hits.
- **Serve typing:** the classifier needs the serve at `ball_round` 1, and only the
  default's rally starts land there reliably. That is a rally-start problem, the one the
  in-play model was built for (`docs/results/rally_in_play.md`), not a hit-detection
  one.
- **The default is unchanged.** The models are:
  - `data/models/movement_hits_nine_plus_lin_w0.33.joblib` (and `_w0.16`, `_w0.5`);
  - the same 1/3 fit as `_w0.33_tuned.joblib`, used with
    `--hit-score-min 0.55 --rally-end-score-min 0.5`.

## What next

- **Ding** (Chong v Ding), once re-tracked and labelled by the user, would be a second
  club source and a second club test.
- **The in-play model refit** on Lin's 67 rallies plus set 1's 28 checked ones, for the
  serve problem above (planned, not started).

## How to run

```
CLUB="--club data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv"
for w in 0.3333333333 0.1666666667 0.5; do tag=$(echo $w | cut -c1-4)
  .venv/bin/python -m vision.hit_training data/hit_training/matches.txt --held-out $CLUB --club-weight $w \
    --out data/hit_training_club/weight_$tag --save data/models/movement_hits_nine_plus_lin_w$tag.joblib
done
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --held-out --tune $CLUB --club-weight 0.3333333333 \
  --out data/hit_training_club/weight_0.33_tuned --save data/models/movement_hits_nine_plus_lin_w0.33_tuned.joblib   # 28 min CPU
V=data/recordings/stand_in/set1.mp4; D=data/club_labels
for run in nine_plus_lin:w0.33:0.5:0.7 lin_w0.16:w0.16:0.5:0.7 lin_w0.5:w0.5:0.5:0.7 nine_plus_lin_tuned:w0.33_tuned:0.55:0.5; do
  IFS=: read name w h r <<< "$run"
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --model data/models/movement_hits_nine_plus_lin_$w.joblib \
    --hit-score-min $h --rally-end-score-min $r --hits-dir $D/$name/hits
done
for name in five_ranges nine_matches nine_plus_lin nine_plus_lin_tuned lin_w0.16 lin_w0.5; do
  cp data/scout.duckdb $D/$name/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id stand_in_set1 --range 0 26234 1 --database $D/$name/scout.duckdb --hits-dir $D/$name/hits
  .venv/bin/python -m shots.predict $D/$name/scout.duckdb
done
for sub in relabel relabel_checked; do
  .venv/bin/python -m vision.label_scores --labels $D/set1_labels_$sub/stand_in_set1.csv --match-id stand_in_set1 --fps 60 \
    --run five_ranges $D/five_ranges/hits/stand_in_set1/hits_0_26234.csv $D/five_ranges/scout.duckdb \
    --run nine_matches $D/nine_matches/hits/stand_in_set1/hits_0_26234.csv $D/nine_matches/scout.duckdb \
    --run nine_plus_lin $D/nine_plus_lin/hits/stand_in_set1/hits_0_26234.csv $D/nine_plus_lin/scout.duckdb \
    --run nine_plus_lin_tuned $D/nine_plus_lin_tuned/hits/stand_in_set1/hits_0_26234.csv $D/nine_plus_lin_tuned/scout.duckdb \
    --run lin_w0.16 $D/lin_w0.16/hits/stand_in_set1/hits_0_26234.csv $D/lin_w0.16/scout.duckdb \
    --run lin_w0.5 $D/lin_w0.5/hits/stand_in_set1/hits_0_26234.csv $D/lin_w0.5/scout.duckdb
done
```

The five-range and nine-match hits on set 1 are those of `docs/results/club_labels_set1.md`.
Every rally row comes from the `label_scores` runs above: the 32-rally rows from the
`relabel` copy, the 28-checked rows from `relabel_checked`.
