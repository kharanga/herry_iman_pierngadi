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
- **A later labelling rule:** a drive that directly follows a clear in the same rally is
  a smash. It retypes 2 of set 1's 172 labels (frames 4657 and 8963, drive to smash) and
  leaves the hit frames, hitters and rallies alone. Every set-1 number on this page was
  scored before that rule. To reproduce them, use the copies kept from then:
  `data/club_labels/set1_labels_relabel_2026-11-01/` and
  `set1_labels_relabel_checked_2026-11-01/`. The undated copies now follow the rule.
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

## Lin's weight at 1/2, 0.6 and 0.7 (set 1 only)

The nine + Lin scorer is refitted with Lin's share of the sample weight at 0.5, 0.6 and 0.7:
- each is one fit on the nine matches plus Lin;
- its thresholds are tuned on those 10 training sources, each scored by a model trained
  without it (`--save --tune`), never on set 1;
- there is no pro held-out run for these.

The 1/3 tuned model is the reference.

| Model | Tuned thresholds (hit / rally end) |
|---|---|
| `movement_hits_nine_plus_lin_w0.33_tuned.joblib` | 0.55 / 0.5 |
| `movement_hits_nine_plus_lin_w0.5_tuned.joblib` | 0.55 / 0.5 |
| `movement_hits_nine_plus_lin_w0.6_tuned.joblib` | 0.55 / 0.5 |
| `movement_hits_nine_plus_lin_w0.7_tuned.joblib` | 0.5 / 0.5 |

The `_tuned` names keep `_w0.5.joblib`, the 1/2 sensitivity model above at the shipped
thresholds, as it was.

Set 1 against the relabel (172 hits). Detected hits unless the row says otherwise:

| | 1/3, tuned | 1/2 | 0.6 | 0.7 |
|---|---|---|---|---|
| Recall ±0.25 s / ±0.50 s | **80.8%** / **87.8%** | 76.7% / 84.9% | 73.8% / 82.0% | 76.2% / 84.3% |
| Precision inside rallies ±0.25 s / ±0.50 s | 74.7% / 78.6% | 75.4% / 81.1% | **79.4%** / **86.0%** | 74.0% / 80.6% |
| F1 ±0.25 s | **77.6** | 76.0 | 76.5 | 75.1 |
| Hitter right ±0.25 s | 94.2% | 93.9% | **96.9%** | 94.7% |
| Shot type, with serves (±0.50 s, detected + inferred serves) | 97/144 = 67.4% | 92/139 = 66.2% | 89/132 = 67.4% | 95/141 = 67.4% |
| Shot type, labelled serves left out ±0.25 s / ±0.50 s | **78.7%** / **76.7%** | 75.2% / 73.7% | 74.7% / 73.9% | 74.8% / 73.3% |
| 8-label detail, with serves (±0.50 s) | 65.3% | 64.7% | 66.7% | 65.2% |
| Labelled serves (31): found / typed `serve` | 24 / 5 | 21 / 5 | 21 / 7 | 25 / **10** |
| 32 rallies with hits: detected / matched | 40 / 30 | 36 / 27 | 34 / 27 | 39 / 29 |
| 32: merged / split / missed / extra | 2 / 4 / 1 / 7 | 2 / 3 / 3 / 6 | 2 / 4 / 4 / 4 | 3 / 5 / 1 / 6 |
| 28 checked: matched; merged / split / missed / extra | 27; 0 / 3 / 1 / 10 | 25; 0 / 3 / 3 / 8 | 24; 0 / 3 / 4 / 7 | 27; 1 / 4 / 1 / 9 |
| 28 checked: start / end error, median abs | **0.63 s** / 1.58 s | 0.75 s / 1.40 s | 0.73 s / **1.04 s** | 1.07 s / 1.25 s |

- A heavier Lin buys precision and hitter accuracy (0.6 is the most precise) at the cost of
  recall. None beats the 1/3 tuned model on F1 or on shot type with the serves left out.
- The best of the three by F1 at ±0.25 s (the tuning's own criterion) is 0.6, so 0.6 is
  the one used in the combined detector below.
- Picking the weight this way uses set 1 for a decision, so set 1 is no longer a clean test
  of that choice. Ding, once labelled, is.

## Default rallies and serves, newer hits (`--rallies-from`, set 1 only)

The combined detector takes the rallies and serve rows from the default five-range model:
its gap rule, its inferred serve, and its rally bounds. A newer scorer's detected hits fill
each rally after the serve row:
- a newer hit within ±0.25 s of the serve row is the serve itself;
- newer hits outside every default rally are dropped.

The rally rows are therefore the default's for all three: 36 detected, 29 of the 32
matched, 2 / 2 / 1 / 5, and 25 of the 28 checked.

| | Default (five ranges) | Rallies from the default + nine matches | + nine + Lin 1/3 tuned | + nine + Lin 0.6 tuned |
|---|---|---|---|---|
| Hits found ±0.25 s / ±0.50 s (detected) | 65.7% / 74.4% | **75.6%** / **79.7%** | 70.3% / 77.3% | 66.3% / 74.4% |
| Hits found, detected + inferred serves ±0.25 s / ±0.50 s | 75.6% / 86.6% | **84.9%** / **91.3%** | 79.7% / 89.0% | 76.2% / 86.6% |
| Precision inside rallies ±0.25 s / ±0.50 s (detected) | 64.2% / 70.3% | 71.4% / 72.9% | 76.1% / 82.1% | **78.1%** / **87.1%** |
| Hitter right ±0.25 s | 93.8% | 96.2% | 95.0% | **97.4%** |
| Shot type, with serves ±0.25 s / ±0.50 s | 73.8% / 74.1% | 74.5% / 73.5% | **77.0%** / 75.5% | 76.5% / **75.8%** |
| Shot type, labelled serves left out ±0.25 s / ±0.50 s | 70.9% / 71.8% | 72.0% / 70.6% | **75.5%** / 73.5% | 75.0% / **73.9%** |
| 8-label detail, with serves (±0.50 s) | 72.7% | 72.8% | **74.8%** | 74.2% |
| Labelled serves (31): found / typed `serve` | 23 / 20 | 22 / **20** | 22 / 19 | 21 / 18 |
| Matched non-serves typed `serve` (±0.25 s / ±0.50 s) | 0 / 2 | 0 / 2 | 0 / 1 | 0 / 1 |

- Every combination keeps the default's serve typing (18 to 20 of the matched serves, where
  the newer scorers alone type 3 to 10) and adds hits or precision on top.
- **Nine matches** inside the default rallies finds the most hits. Its merged rallies are
  gone, because the rallies are the default's.
- **Nine + Lin 1/3 tuned** has the best shot type with and without serves (77.0% at
  ±0.25 s with serves), at 76.1% precision.
- **0.6** is the most precise and has the best hitter, and finds about as many hits as the
  default.
- The cost: the rally bounds stay the default's, whose ends are the furthest off (2.90 s
  median against 1.58 s for the 1/3 tuned scorer alone). The combination doesn't get
  those back.
- The ±0.25 s same-hit window was set beforehand, not tuned on set 1.

## Nine + Lin + Ding (set 1 only)

The scorer is refitted with the user's Chong v Ding labels added:
- the labels are 414 hits in 47 rallies, every bound checked;
- the tracks are Ding's re-tracked ones, with doubtful player rows dropped as for Lin;
- Lin and Ding together carry 1/3 of the sample weight, 1/6 each, fixed in advance;
- there is one fit, with thresholds tuned on its 11 training sources (`--save --tune`):
  `--hit-score-min 0.55 --rally-end-score-min 0.6`;
- there is no pro held-out run.

The fit read the labels after the user's fix of rally 10, where frame 6094 is now
`opp`, `serve_short`. The model is
`data/models/movement_hits_nine_plus_lin_ding_tuned.joblib`.

Set 1 against the relabel (172 hits), next to nine + Lin 1/3 tuned, alone and inside the
combined detector. Detected hits unless the row says otherwise:

| | Nine + Lin, alone | Nine + Lin + Ding, alone | Combined + nine + Lin | Combined + nine + Lin + Ding |
|---|---|---|---|---|
| Recall ±0.25 s / ±0.50 s | **80.8% / 87.8%** | 75.0% / 83.7% | 70.3% / 77.3% | 67.4% / 75.0% |
| Precision inside rallies ±0.25 s / ±0.50 s | 74.7% / 78.6% | 71.3% / 77.4% | **76.1% / 82.1%** | 73.0% / 79.6% |
| Hitter right ±0.25 s | 94.2% | 94.6% | 95.0% | **95.7%** |
| Shot type, with serves (detected + inferred serves) ±0.25 s / ±0.50 s | 67.7% / 67.4% | 71.4% / 68.8% | **77.0% / 75.5%** | **77.0%** / 75.0% |
| Shot type, labelled serves left out ±0.25 s / ±0.50 s | **78.7% / 76.7%** | 77.6% / 73.9% | 75.5% / 73.5% | 74.8% / 72.2% |
| 8-label detail, with serves (±0.50 s) | 65.3% | 67.4% | **74.8%** | 72.8% |
| Labelled serves (31): found / typed `serve` | 24 / 5 | 23 / 9 | 22 / **19** | 21 / **19** |
| Matched non-serves typed `serve` ±0.25 s / ±0.50 s | 0 / 1 | 0 / 0 | 0 / 1 | 0 / 1 |
| 32 rallies: matched; merged / split / missed / extra | 30; 2 / 4 / 1 / 7 | 28; 3 / 3 / 2 / **3** | 29; 2 / 2 / 1 / 5 | 29; 2 / 2 / 1 / 5 |
| 28 checked: matched; start / end error, median abs | **27; 0.63 s / 1.58 s** | 26; 0.88 s / 1.62 s | 25; 1.15 s / 2.90 s | 25; 1.15 s / 2.90 s |

- **On set 1, adding Ding doesn't help the hits.** Alone, it finds fewer hits (75.0%
  against 80.8% at ±0.25 s) at a lower precision. Inside the combined detector it
  trails by about 3 points on both.
- **It is better on a few things.** It finds fewer rallies between points (3 extra
  against 7), types more serves when alone (9 against 5), and has a slightly better
  hitter. The shot type with serves is level inside the combined detector (77.0% each).
- **Set 1 is one club video.** Ding is a different hall and camera from set 1. The test
  games (`docs/results/club_games.md`) are the next clean check of whether Ding's labels
  carry over.
- Nothing was chosen on set 1. The weight split and the threshold tuning were fixed
  beforehand or used the training sources only.

## Club-only scorers: Lin + Ding, Lin, Ding (set 1 only)

Three scorers trained on club footage only, with no ShuttleSet match. Features, model and
the doubtful-row rule are the same as for the others (`vision.hit_training --clubs-only`):

| | Trained on | Tuned thresholds (hit / rally end) | How tuned |
|---|---|---|---|
| F | Lin + Ding, equal weight | 0.65 / 0.7 | each club scored by a model trained on the other |
| G | Lin | 0.65 / 0.7 | **in-sample**: Lin scored by the model trained on Lin |
| H | Ding | 0.65 / 0.8 | **in-sample**: Ding scored by the model trained on Ding |

The models are `data/models/movement_hits_club_{lin_ding,lin,ding}_tuned.joblib`. All
three hit-score thresholds are 0.65, the top of the tuning grid, so a higher one might
have scored better on the training clubs.

Set 1 against the relabel (172 hits). B is the combined detector with nine + Lin 1/3
tuned, and C the combined detector with nine + Lin + Ding. "Combined" means the default
model's rallies and serves with the scorer's hits after the serve.

| | Recall ±0.25 / ±0.50 s | Precision in rallies ±0.25 / ±0.50 s | Hitter ±0.25 s | Shot type with serves ±0.25 / ±0.50 s | Shot type, serves left out ±0.25 / ±0.50 s | 8-label detail ±0.50 s | Serves found / typed `serve` | Non-serves typed `serve` ±0.25 / ±0.50 s | 32 rallies: matched; merged / split / missed / extra |
|---|---|---|---|---|---|---|---|---|---|
| D: nine + Lin 1/3 tuned, alone | **80.8% / 87.8%** | 74.7% / 78.6% | 94.2% | 67.7% / 67.4% | 78.7% / 76.7% | 65.3% | 24 / 5 | 0 / 1 | 30; 2 / 4 / 1 / 7 |
| F alone | 66.9% / 72.7% | 82.1% / 88.0% | 94.8% | 71.8% / 73.3% | 79.3% / 78.9% | 71.6% | 21 / 10 | 4 / 4 | 28; 0 / 3 / 4 / 3 |
| G alone | 48.3% / 51.2% | 85.6% / 89.8% | 98.8% | 70.7% / 72.6% | 75.4% / 75.7% | 71.1% | 14 / 8 | 4 / 5 | 24; 1 / 4 / 8 / 1 |
| H alone | 41.9% / 47.7% | 75.0% / 85.4% | 98.6% | 70.6% / 69.9% | 69.4% / 68.0% | 69.9% | 8 / 7 | 7 / 10 | 21; 1 / 2 / 11 / 0 |
| B: combined + nine + Lin | 70.3% / 77.3% | 76.1% / 82.1% | 95.0% | 77.0% / 75.5% | 75.5% / 73.5% | 74.8% | 22 / 19 | 0 / 1 | 29; 2 / 2 / 1 / 5 |
| C: combined + nine + Lin + Ding | 67.4% / 75.0% | 73.0% / 79.6% | 95.7% | 77.0% / 75.0% | 74.8% / 72.2% | 72.8% | 21 / 19 | 0 / 1 | 29; 2 / 2 / 1 / 5 |
| F combined | 59.3% / 64.5% | 81.6% / 88.1% | 96.1% | 83.0% / 81.1% | 80.7% / 78.3% | 79.3% | 20 / 19 | 0 / 2 | 29; 2 / 2 / 1 / 5 |
| G combined | 45.3% / 48.3% | **86.7% / 91.2%** | **98.7%** | **84.4% / 85.7%** | **82.0% / 83.3%** | **82.1%** | 18 / 17 | 0 / 0 | 29; 2 / 2 / 1 / 5 |
| H combined | 41.3% / 46.5% | 75.5% / 85.1% | 98.6% | 80.3% / 81.5% | 75.9% / 77.3% | 76.5% | 15 / 15 | 0 / 1 | 29; 2 / 2 / 1 / 5 |

- **The club-only scorers are precise but find far fewer hits.** F finds 67% at ±0.25 s,
  G 48% and H 42%, against 81% for D. Their precision is up to 86% (G).
- **Their shot types look better, but on fewer hits.** The agreement is counted on
  matched hits only. G combined agrees on 85.7% at ±0.50 s, but over 84 hits, where B
  agrees on 75.5% over 139. A scorer that finds only the clearest hits leaves the
  classifier the easier shots. So the shot-type columns don't compare like with like
  across rows that find very different numbers of hits.
- **F (Lin + Ding) is the most balanced of the three:**
  - fewer between-point rallies than D (3 extra against 7);
  - the closest rally ends of the scorers alone (1.00 s median);
  - inside the combined detector, 83.0% shot type with serves at ±0.25 s, at 59% of hits.
- **G and H are tuned in-sample**, so their thresholds fit their own video. H is trained
  on Ding, a different hall, and finds the fewest hits on set 1.
- Nothing was chosen on set 1. These are candidates F, G and H for the final test on the
  two held-out games (`docs/results/club_games.md`, "Two test games"), each alone and combined.

## What next

- **The two test games** (`docs/results/club_games.md`), once labelled, test nine + Lin
  and nine + Lin + Ding on footage that trained and tuned nothing.
- **Serve rows on in-play rallies**: the in-play refit is done
  (`docs/results/rally_in_play.md`); its rally starts still rarely carry the serve.

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

The weights and the combined detector (set 1 only):

```
for w in 0.5 0.6 0.7; do
  .venv/bin/python -m vision.hit_training data/hit_training/matches.txt --tune $CLUB --club-weight $w \
    --save data/models/movement_hits_nine_plus_lin_w${w}_tuned.joblib   # about 3 min CPU each, prints the tuned thresholds
done
for run in lin_w0.5_tuned:w0.5_tuned:0.55:0.5 lin_w0.6_tuned:w0.6_tuned:0.55:0.5 lin_w0.7_tuned:w0.7_tuned:0.5:0.5; do
  IFS=: read name w h r <<< "$run"
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --model data/models/movement_hits_nine_plus_lin_$w.joblib \
    --hit-score-min $h --rally-end-score-min $r --hits-dir $D/$name/hits
done
for run in combined_nine_matches:movement_hits_ten_matches:0.5:0.7 combined_nine_plus_lin_tuned:movement_hits_nine_plus_lin_w0.33_tuned:0.55:0.5 combined_lin_w0.6_tuned:movement_hits_nine_plus_lin_w0.6_tuned:0.55:0.5; do
  IFS=: read name model h r <<< "$run"
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --rallies-from data/models/movement_hits.joblib \
    --model data/models/$model.joblib --hit-score-min $h --rally-end-score-min $r --hits-dir $D/$name/hits
done
```

Then `vision.features` and `shots.predict` on a copy of `data/scout.duckdb` per run, and
`vision.label_scores` on both label copies, as above. The weight runs are scored with
`--run` for `five_ranges nine_matches nine_plus_lin_tuned lin_w0.5_tuned lin_w0.6_tuned lin_w0.7_tuned`,
and the combined runs with `five_ranges combined_nine_matches combined_nine_plus_lin_tuned combined_lin_w0.6_tuned`.
The "labelled serves left out" rows are `label_scores`' table of that name: matched with
the inferred serve rows in, then scored on the labels whose group is not `serve`. The
reports are `data/club_labels/report_vis028_{weights,combined}{,_checked}.md`.

Nine + Lin + Ding (set 1 only):

```
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --tune $CLUB \
  --club data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv --club-weight 0.3333333333 \
  --save data/models/movement_hits_nine_plus_lin_ding_tuned.joblib   # about 5 min CPU
M=data/models/movement_hits_nine_plus_lin_ding_tuned.joblib
.venv/bin/python -m vision.movement $V --start 0 --end 26234 --model $M --hit-score-min 0.55 --rally-end-score-min 0.6 --hits-dir $D/lin_ding_tuned/hits
.venv/bin/python -m vision.movement $V --start 0 --end 26234 --rallies-from data/models/movement_hits.joblib --model $M \
  --hit-score-min 0.55 --rally-end-score-min 0.6 --hits-dir $D/combined_lin_ding_tuned/hits
```

Then `vision.features` and `shots.predict` per run, and `vision.label_scores` on both
label copies with `--run` for
`nine_plus_lin_tuned lin_ding_tuned combined_nine_plus_lin_tuned combined_lin_ding_tuned`
(reports `data/club_labels/report_vis030{,_checked}.md`).

Club-only scorers (set 1 only):

```
LIN="--club data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv"
DING="--club data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv"
.venv/bin/python -m vision.hit_training --clubs-only $LIN $DING --tune --save data/models/movement_hits_club_lin_ding_tuned.joblib   # about 2 min CPU
.venv/bin/python -m vision.hit_training --clubs-only $LIN --tune --save data/models/movement_hits_club_lin_tuned.joblib
.venv/bin/python -m vision.hit_training --clubs-only $DING --tune --save data/models/movement_hits_club_ding_tuned.joblib
for run in club_lin_ding:0.65:0.7 club_lin:0.65:0.7 club_ding:0.65:0.8; do
  IFS=: read name h r <<< "$run"; M=data/models/movement_hits_${name}_tuned.joblib
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --model $M --hit-score-min $h --rally-end-score-min $r --hits-dir $D/$name/hits
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --rallies-from data/models/movement_hits.joblib --model $M \
    --hit-score-min $h --rally-end-score-min $r --hits-dir $D/combined_$name/hits
done
```

Then `vision.features` and `shots.predict` per run, and `vision.label_scores` on both
label copies with `--run` for
`nine_plus_lin_tuned club_lin_ding club_lin club_ding combined_nine_plus_lin_tuned combined_lin_ding_tuned combined_club_lin_ding combined_club_lin combined_club_ding`
(reports `data/club_labels/report_vis033{,_checked}.md`).
