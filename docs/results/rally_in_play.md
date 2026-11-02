# Learning when a rally is in play

The hit scorer's rally rule ends a rally when there is no hit-like candidate for
`max_hit_gap_frames` and the end score is at least 0.7. With the hit scorer trained on
nine matches, the rule rarely closes a rally on club footage: stand-in set 1 gets 27
rallies, where the user labelled 25 and the old five-range scorer found 36. The next
serve then gets a mid-rally `ball_round` and is classified as a net shot or a drive
(`docs/results/club_labels_set1.md`). This task replaces the rule with a learned
per-frame in-play model (`docs/vision/in_play.md`), shipped as an option next to it. The
default is unchanged.

## Training data: how much out-of-play time the tracks hold

The nine matches of `docs/results/hits_ten_matches.md` are tracked only around their
labelled rallies, 2 s before the first hit and after the last. A frame is in play from a
rally's first hit (the serve) to its last hit plus 1.0 s for the landing.

| | Tracked | Out of play | Out-of-play share |
|---|---|---|---|
| 9 matches | 189.0 min | 43.8 min | 23.2% |

Per match (`python -m vision.in_play data/hit_training/matches.txt --gaps`):

- a **gap** is the frames between one labelled rally's span (first hit to last hit +
  1.0 s) and the next rally's first hit, across the whole match, sets in order;
- the **tracked share of a gap** is the share of its frames inside a tracked stretch,
  averaged over the match's gaps.

| Match | fps | Tracked (min) | Out of play (min) | Out-of-play share | Median gap between rallies (s) | Tracked share of a gap, mean |
|---|---|---|---|---|---|---|
| Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals | 30 | 21.9 | 5.3 | 24.4% | 19.4 | 14.9% |
| Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals | 25 | 32.2 | 5.7 | 17.5% | 22.0 | 13.7% |
| Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final | 25 | 18.1 | 3.9 | 21.6% | 23.9 | 12.9% |
| Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals | 25 | 26.2 | 5.6 | 21.4% | 24.2 | 13.0% |
| Carolina_Marin_Supanida_Katethong_YONEX_Thailand_Open_2021_QuarterFinals | 30 | 13.4 | 3.7 | 27.7% | 22.5 | 12.8% |
| Anders_Antonsen_Sameer_Verma_TOYOTA_THAILAND_OPEN_2021_QuarterFinals | 30 | 23.8 | 5.8 | 24.4% | 25.3 | 12.3% |
| Carolina_Marin_Neslihan_Yigit_TOYOTA_THAILAND_OPEN_2021_QuarterFinals | 30 | 11.2 | 3.1 | 28.0% | 20.8 | 15.4% |
| Ng_Ka_Long_Angus_Kidambi_Srikanth_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals | 30 | 23.9 | 5.6 | 23.5% | 18.5 | 15.8% |
| Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals | 30 | 18.3 | 5.0 | 27.5% | 22.5 | 13.9% |

The out-of-play share is 17.5-28.0%. But the stretches cover only 12.3-15.8% of each
gap between rallies (the median gap is 18.5-25.3 s). So the model sees the first second
or so after a landing and the last second or so before a serve, and almost none of the
walking back, picking up the shuttle and resetting in between.

Broadcast video often cuts to replays and close-ups in those gaps anyway, so they
couldn't all be tracked. Wider padding wasn't tracked; see "What next".

## The model

The model:

- **Features:** 43 per frame. For each player: speed, acceleration, detection and
  distance from the net, over 1, 3 and 8 s windows; the max hit score per player per
  window; the number of hit-score peaks per window; and the seconds since and until the
  nearest peak.
- **Model family:** the hit scorer's (gradient boosting).
- **Training scores:** the hit scores come from a hit model trained without the match.
- **From probability to rallies:**
  - the probability is smoothed over 1 s, and frames at 0.5 or more are in play;
  - gaps under 1 s are closed, and stretches under 2 s are dropped;
  - each stretch is a rally whose hits are the hit chain inside it;
  - the inferred serve sits at the stretch's start, unless the first detected hit is
    within 0.3 s of it.

The shipped model, trained on all nine matches, is
`data/models/in_play_nine_matches.joblib` (never committed).

## Held out by match, ShuttleSet (9 matches, 876 rallies, 9,424 hits)

Each match is scored by an in-play model trained on the other eight, on the same
held-out hit scores as the rule. Rally bounds are compared with the labelled
first-hit-to-landing spans inside the tracked stretches.

| | Rule | In-play model |
|---|---|---|
| Rallies found (876 labelled) | 849 | 838 |
| Matched one to one | 832 | 836 |
| Merged (one found rally over two labelled) | 0 | 0 |
| Split (two found rallies over one labelled) | 15 | 2 |
| Missed / extra | 44 / 2 | 40 / 0 |
| Start error, mean (s); median per match of the median abs error | -0.01; 0.25 | +0.06; 0.16 |
| End error, mean (s); median per match of the median abs error | +1.11; 1.20 | +0.14; 0.24 |
| Frames called rightly in or out of play | 86.6% | **94.8%** |
| Rally's first row on the labelled serve (±0.50 s) | 545 of 876 = 62.2% | **724 of 876 = 82.6%** |
| Hit recall ±0.25 s / ±0.50 s | 84.6% / 89.2% | 87.7% / **92.6%** |
| Hit precision ±0.50 s | 89.8% | 89.9% |

Merging can't show up here, because each tracked stretch holds about one rally. The
club video below is the test of merging.

| Match | fps | Rallies | Rule: found / merged / split | In-play: found / merged / split | Frames right, rule / in-play | First row on the serve, rule / in-play | Hit recall ±0.50 s, rule / in-play |
|---|---|---|---|---|---|---|---|
| Anders Antonsen Sameer Verma TOYOTA THAILAND OPE | 30 | 116 | 109 / 0 / 2 | 108 / 0 / 0 | 85.2% / 94.5% | 92 / 96 of 116 | 84.5% / 87.5% |
| Anders Antonsen Viktor Axelsen HSBC BWF WORLD TO | 30 | 101 | 94 / 0 / 0 | 93 / 0 / 0 | 80.7% / 92.3% | 62 / 83 of 101 | 84.0% / 90.6% |
| Carolina Marin Neslihan Yigit TOYOTA THAILAND OP | 30 | 63 | 61 / 0 / 1 | 62 / 0 / 0 | 87.1% / 95.4% | 47 / 57 of 63 | 86.7% / 94.8% |
| Carolina Marin Supanida Katethong YONEX Thailand | 30 | 74 | 72 / 0 / 0 | 72 / 0 / 0 | 88.5% / 95.0% | 60 / 64 of 74 | 89.7% / 92.7% |
| Kento MOMOTA CHOU Tien Chen Denmark Open 2018 Fi | 25 | 112 | 110 / 0 / 3 | 106 / 0 / 0 | 88.0% / 95.5% | 18 / 91 of 112 | 95.6% / 96.6% |
| Kento MOMOTA CHOU Tien Chen Fuzhou Open 2019 Fin | 25 | 113 | 117 / 0 / 7 | 112 / 0 / 2 | 89.0% / 95.7% | 29 / 92 of 113 | 92.7% / 95.4% |
| Kento MOMOTA CHOU Tien Chen KOREA OPEN 2019 Fina | 25 | 78 | 77 / 0 / 0 | 76 / 0 / 0 | 88.7% / 95.9% | 68 / 69 of 78 | 91.8% / 94.2% |
| Ng Ka Long Angus Kidambi Srikanth HSBC BWF WORLD | 30 | 112 | 110 / 0 / 1 | 107 / 0 / 0 | 89.4% / 95.4% | 98 / 97 of 112 | 90.0% / 93.7% |
| Viktor Axelsen Anthony Sinisuka Ginting YONEX Th | 30 | 107 | 99 / 0 / 1 | 102 / 0 / 0 | 82.7% / 93.2% | 71 / 75 of 107 | 83.3% / 87.2% |

**Choosing the serve rule, held out.** The rule picks between taking the first
detected hit of a stretch as the serve and adding an inferred serve row at the start.
It was chosen on these held-out matches, never on the club video:

| Serve rule | First row on the serve | Inferred serve rows |
|---|---|---|
| **first hit within 0.3 s of the start is the serve (shipped)** | **724 / 876 = 82.6%** | 279 |
| within 0.5 s | 709 = 80.9% | 213 |
| within 1.0 s | 613 = 70.0% | 28 |
| within 1.5 s | 597 = 68.2% | 3 |
| within 2.0 s | 596 = 68.0% | 0 |
| no inferred serve row | 596 = 68.0% | 0 |

The rule was chosen when serves were matched within ±10 frames; rescored at ±0.50 s the
ranking is the same.

## Stand-in set 1 (club, 60 fps), against the user's 153 labels and 25 rallies

All three runs score the same labels. Their rows are built and predicted in copies of
the database, with the classifier trained on the eight shot labels. The hit models are:

- **old (five ranges):** the shipped default hit scorer, with the rule;
- **nine matches:** the nine-match hit scorer, with the rule;
- **nine matches + in-play:** the nine-match hit scorer, with the in-play model.

| | Old (five ranges) | Nine matches | Nine matches + in-play |
|---|---|---|---|
| Rallies found (25 labelled) | 36 | 27 | 33 |
| Matched / merged / split | 25 / 0 / 0 | 20 / **8** / 4 | 23 / **2** / 2 |
| Extra (rallies outside the labelled ones) | 11 | 6 | 8 |
| Start error, median abs; mean (s) | 0.00; -0.08 | 2.97; -3.98 | 0.57; -0.72 |
| End error, median abs; mean (s) | 1.12; +1.91 | 3.13; +5.59 | 1.25; +1.73 |
| Hit recall ±0.25 s, detected hits | 68.6% | 88.2% | **88.9%** |
| Hit recall ±0.50 s, detected hits | 78.4% | 94.8% | **96.1%** |
| Hitter right on matched hits (±0.25 s) | 96.2% | 97.0% | 95.6% |
| Precision inside the labelled rallies (±0.25 s, lower bound) | 61.4% | 69.2% | 70.5% |
| **Labelled serves typed `serve`, ±0.50 s** (target: about 15 of 18 or better) | **16 of 19** | 2 of 20 | **11 of 22** |
| Labelled serves found by an inferred row / a detected hit / missed | 16 / 3 / 5 | 2 / 18 / 4 | 2 / 20 / 2 |
| `ball_round` 1 rows predicted (detail) | serve_short 24, serve_long 10, lift 1, net 1 | serve_short 18, serve_long 7, smash 1, net 1 | serve_short 30, serve_long 3 |

Notes on the set-1 numbers:

- **Old model's starts:** 14 of the 25 labelled rally starts are the old model's own
  frames, because the user kept them. That flatters the old model's start error.
- **Detail level:** set 1 was labelled before the ten shot labels existed, so its labels
  have no `type_detail` and detail agreement can't be scored. The last row shows only
  the predicted detail of each rally's first row.
- **Noise:** set 1 has 24 labelled serves, of which each run matches 19-22 at ±0.50 s,
  so a difference of one or two is noise. 11 of 22 against 16 of 19 is not.
- **Superseded labels:** these rows use set 1's first 153 labels (copy in
  `data/club_labels/set1_labels_153/`). The user has since relabelled set 1; the hit
  models are scored against the relabel in `docs/results/hits_club.md`. The frame-based
  figures (±5 / ±10 / ±20 frames) of earlier versions are in the git history.

## What it says

- **On ShuttleSet** the in-play model is better than the rule on every count:
  - 94.8% of frames rightly in or out of play, against 86.6%;
  - rally ends 0.24 s off, against 1.2 s;
  - the first row on the serve in 83% of rallies, against 62%;
  - 3 points more hit recall at the same precision.
- **On the club video** it fixes most of the rule's merging: 2 merged and 2 split
  rallies, against 8 and 4. Starts are 0.57 s off, against 2.97 s. Hit recall holds:
  88.9% against 88.2% at ±0.25 s.
- **Serve typing misses its target: 11 of 22, against the old model's 16 of 19.** The
  in-play starts on the club video come 0.72 s early on average (on ShuttleSet, +0.06
  s). So every rally's first row is an inferred serve 0.7 s before the real one, and it
  is predicted a serve (30 serve_short, 3 serve_long). But the real serve, which the
  nine-match scorer usually detects as a hit, then follows as a second row at
  `ball_round` 2 and is typed as a return. The serve window that would absorb this
  (1 s or more) is the worst choice on ShuttleSet, so the rule can't fix it.
- **Why club starts are early:** before a club serve the players walk to position and
  settle in view of the camera. In the broadcast stretches that time is mostly cut away
  or not tracked, so the model never learned it.
- **The in-play model is shipped as an option** (`--in-play-model`). The default stays
  the old five-range scorer with the rule, until the user decides.

## Refit with the user's club rallies (set 1 only)

The in-play model is refitted on the nine matches plus Lin v Yeung's 67 hand-marked
rallies. For Lin, the whole video counts: in play between the user's start and end,
everything else not in play. Two kinds of frames are left out:
- frames where a player is flagged doubtful;
- every other frame, as for the pro matches.

That leaves 13,157 Lin frames, 6,588 of them in play. Lin carries 1/3 of the sample
weight, fixed in advance.

Lin's hit scores come from the nine-match hit model, which wasn't trained on Lin, just as
each pro match is scored by a model trained without it. Nothing was tuned: the
probability threshold and the minimum lengths are the existing constants.

The model is `data/models/in_play_nine_plus_lin.joblib`. A second one,
`data/models/in_play_nine_plus_lin_set1.joblib`, adds set 1's 28 checked rallies at a
share of 1/6, the same as Lin's, with set 1's hit scores from the 1/3 nine + Lin scorer.
It is saved for the clean test on Ding once the user has labelled it, and is not scored
here.

### Rally bounds against set 1's 28 checked rallies

These come from `vision.rally_bounds`. The 6 unchecked rallies are left out: their
frames are not judged, and a detected rally touching only them is not extra.

| Method (hit scorer, thresholds) | Frames right | Found | Matched of 28 | Merged / split / missed / extra | Start / end error, median abs | Start / end error, mean |
|---|---|---|---|---|---|---|
| Gap rule, default five-range scorer (0.5 / 0.7) | 62.5% | 36 | 25 | 2 / 1 / 1 / 5 | 1.15 s / 2.90 s | +0.00 s / +2.70 s |
| Gap rule, nine + Lin 1/3 tuned (0.55 / 0.5) | 71.2% | 40 | **27** | 0 / 3 / 1 / 7 | 0.63 s / 1.58 s | +0.10 s / +1.50 s |
| Current in-play, nine + Lin 1/3 tuned | 78.1% | 26 | 22 | 0 / 0 / 6 / **0** | 1.36 s / 0.82 s | +1.61 s / -0.34 s |
| **Refit in-play, nine + Lin 1/3 tuned** | 78.7% | 26 | 22 | 0 / 0 / 6 / **0** | 1.72 s / 0.73 s | +1.78 s / -0.63 s |
| Current in-play, nine-match scorer (0.5 / 0.7) | 74.8% | 33 | **27** | 0 / 0 / 1 / 4 | 0.73 s / 0.68 s | -0.08 s / +1.39 s |
| **Refit in-play, nine-match scorer (0.5 / 0.7)** | **84.2%** | 28 | 25 | 0 / 0 / 3 / 1 | **0.53 s / 0.47 s** | +0.59 s / +0.07 s |

- **With the nine-match scorer**, the refit is the best rally finder on set 1:
  - the most frames right (84.2%);
  - the closest starts and ends (about half a second each), with 1 extra rally.
  - The current model with that scorer starts on time on average, but ends late
    (+1.39 s).
- **With the nine + Lin 1/3 tuned scorer**, which the spec pairs it with, the refit
  changes almost nothing. Both in-play models miss 6 of the 28 rallies and start
  rallies late (about +1.7 s mean). The tuned scorer at the default 0.5 threshold does
  the same (79.6% frames right, 22 matched, 6 missed), so the cause is that scorer's
  scores, not its threshold.
- The in-play model reads the hit scores as features, and it was trained on scores from
  pro-only hit models. A scorer trained with Lin gives scores it hasn't seen.

### The pipeline, next to the combined detector

These come from `vision.label_scores` against the relabel (172 hits). The combined
columns are the default's rallies and serves with the newer hits
(`docs/results/hits_club.md`). Detected hits at ±0.25 s, unless the row says otherwise:

| | Combined + nine matches | Combined + 1/3 tuned | Refit in-play + 1/3 tuned | Refit in-play + nine matches |
|---|---|---|---|---|
| Hits found ±0.25 s / ±0.50 s | 75.6% / 79.7% | 70.3% / 77.3% | 64.5% / 71.5% | **80.8% / 87.8%** |
| Precision inside rallies ±0.25 s / ±0.50 s | 71.4% / 72.9% | **76.1%** / 82.1% | 75.5% / **83.1%** | 70.2% / 74.0% |
| Hitter right | **96.2%** | 95.0% | 92.8% | 94.2% |
| Shot type, with serves (detected + inferred serves) ±0.25 s / ±0.50 s | 74.5% / 73.5% | **77.0% / 75.5%** | 64.7% / 63.2% | 65.9% / 64.8% |
| Shot type, labelled serves left out ±0.25 s / ±0.50 s | 72.0% / 70.6% | **75.5% / 73.5%** | 64.6% / 62.6% | 70.6% / 70.4% |
| 8-label detail, with serves (±0.50 s) | 72.8% | **74.8%** | 63.7% | 64.6% |
| Labelled serves (31): found / typed `serve` | 22 / **20** | 22 / 19 | 7 / 5 | 20 / 6 |
| Matched non-serves typed `serve` ±0.25 s / ±0.50 s | 0 / 2 | 0 / 1 | 12 / 15 | 5 / 5 |
| 32 rallies: matched; merged / split / missed / extra | 29; 2 / 2 / 1 / 5 | 29; 2 / 2 / 1 / 5 | 25; 2 / 1 / 6 / 0 | 27; 2 / 0 / 3 / 1 |
| 32 rallies: start / end error, median abs | 1.00 s / 2.90 s | 1.00 s / 2.90 s | 1.82 s / 0.75 s | **0.53 s / 0.60 s** |

- **Refit in-play + nine matches** finds the most hits and has the best rally bounds.
  But its rallies rarely start on an inferred serve row: only 9 of 28 have one, because
  a detected hit lies within 0.3 s of the stretch start. Only 6 of its 20 matched
  serves are typed `serve`, and 5 non-serves are typed `serve`. So its shot types trail
  the combined detector's by 1 to 11 points.
- **Refit in-play + 1/3 tuned** is worse on hits, serves and shot type than both
  combined detectors.
- **The combined detector stays the best for shot type and serves.** The refit in-play
  model with the nine-match scorer is the best for rally bounds and hit recall. Getting
  both would need the in-play rally starts to carry the serve, which is the serve rule's
  problem, not the refit's.
- Set 1 was only scored here, never trained on (the saved second model aside). The
  nine-match pairing was added after the tuned pairing failed on set 1, so it is a
  set-1 finding for Ding to confirm.

## What next

- **Next: serve rows on in-play rallies.** With the nine-match scorer, the refit's rallies
  rarely start on an inferred serve, so serves are seldom typed `serve`. Fixing the
  serve rule for in-play stretches would combine its rally bounds with the combined
  detector's serve typing.
- **Test the refit on Ding** once the user has labelled it: the refit model (Lin only)
  and `in_play_nine_plus_lin_set1.joblib`, each with the nine-match scorer and the 1/3
  tuned one. That confirms or rejects the set-1 finding that the in-play model needs a
  pro-only scorer.
- **Wider padding on the broadcast matches** (for example 8 s after the last hit, about
  2.5 h on the GPU) would add some between-point frames. But broadcast gaps are partly
  cut to replays, so the club labels are the better fix.

## How to run

```
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --gaps
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --held-out --save data/models/in_play_nine_matches.joblib
V=data/recordings/stand_in/set1.mp4; D=data/club_labels
.venv/bin/python -m vision.movement $V --start 0 --end 26234 --model data/models/movement_hits_ten_matches.joblib --in-play-model data/models/in_play_nine_matches.joblib --hits-dir $D/nine_in_play/hits
for name in five_ranges nine_matches nine_in_play; do
  cp data/scout.duckdb $D/$name/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id stand_in_set1 --range 0 26234 1 --database $D/$name/scout.duckdb --hits-dir $D/$name/hits
  .venv/bin/python -m shots.predict $D/$name/scout.duckdb
done
.venv/bin/python -m vision.label_scores --labels data/labels/stand_in_set1.csv --match-id stand_in_set1 --fps 60 \
  --run five_ranges $D/five_ranges/hits/stand_in_set1/hits_0_26234.csv $D/five_ranges/scout.duckdb \
  --run nine_matches $D/nine_matches/hits/stand_in_set1/hits_0_26234.csv $D/nine_matches/scout.duckdb \
  --run nine_in_play $D/nine_in_play/hits/stand_in_set1/hits_0_26234.csv $D/nine_in_play/scout.duckdb
```

The five_ranges and nine_matches hits are made as in `docs/results/club_labels_set1.md`.
The held-out run takes about 2 minutes on the CPU (with the hit scores cached in
`data/in_play/scores/`). The serve-rule comparison was run with a scratch script that
changes only `SERVE_WINDOW_SECONDS` and `infer_serves` on the same held-out
probabilities.

The refit (set 1 only):

```
LIN="--club data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/models/movement_hits_ten_matches.joblib"
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --save data/models/in_play_nine_plus_lin.joblib $LIN   # about 25 s CPU
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --save data/models/in_play_nine_plus_lin_set1.joblib $LIN \
  --club data/recordings/stand_in/set1.mp4 data/club_labels/set1_labels_relabel/stand_in_set1.csv data/models/movement_hits_nine_plus_lin_w0.33_tuned.joblib
V=data/recordings/stand_in/set1.mp4; D=data/club_labels
T=data/models/movement_hits_nine_plus_lin_w0.33_tuned.joblib; N=data/models/movement_hits_ten_matches.joblib
.venv/bin/python -m vision.rally_bounds $V --rallies $D/set1_labels_relabel/stand_in_set1_rallies.csv \
  --method default_gap_rule data/models/movement_hits.joblib 0.5 0.7 rule --method tuned_gap_rule $T 0.55 0.5 rule \
  --method tuned_in_play_current $T 0.55 0.5 data/models/in_play_nine_matches.joblib --method tuned_in_play_refit $T 0.55 0.5 data/models/in_play_nine_plus_lin.joblib \
  --method nine_in_play_current $N 0.5 0.7 data/models/in_play_nine_matches.joblib --method nine_in_play_refit $N 0.5 0.7 data/models/in_play_nine_plus_lin.joblib \
  --method tuned_at_0.5_in_play_refit $T 0.5 0.5 data/models/in_play_nine_plus_lin.joblib
for run in in_play_refit_tuned:$T:0.55:0.5 in_play_refit_nine:$N:0.5:0.7; do
  IFS=: read name model h r <<< "$run"
  .venv/bin/python -m vision.movement $V --start 0 --end 26234 --model $model --in-play-model data/models/in_play_nine_plus_lin.joblib \
    --hit-score-min $h --rally-end-score-min $r --hits-dir $D/$name/hits
done
```

Then `vision.features` and `shots.predict` on a copy of `data/scout.duckdb` per run, and
`vision.label_scores` on the relabel with `--run` for
`five_ranges combined_nine_matches combined_nine_plus_lin_tuned combined_lin_w0.6_tuned in_play_refit_tuned in_play_refit_nine`
(reports `data/club_labels/report_vis029_pipeline{,_checked}.md`). Those shot types were
scored before the drive-after-clear-is-a-smash rule retyped 2 of set 1's labels; to
reproduce them, use the label copies dated `_2026-11-01` (`docs/results/hits_club.md`,
"Test files"). The rally bounds and the in-play training don't use shot types, so they
reproduce from either copy.
