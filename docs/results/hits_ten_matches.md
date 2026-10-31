# Movement hit scorer trained on nine ShuttleSet matches: results

The movement hit detector (`vision/movement.py`, `docs/results/hits_movement.md`) was
trained on 128 hand-checked hits from five ranges of one match. This page trains the
same scorer on every labelled hit of nine ShuttleSet matches (9,424 hits, 45,594 hit
frames among 581,694 player-frames) and measures it with each match held out in turn.
The new model is a separate file, `data/models/movement_hits_ten_matches.joblib`; the
default `data/models/movement_hits.joblib` is unchanged.

## Headline (each match held out, ±5 frames, same thresholds as the shipped model)

| Model | Scored on | Labelled | Detected | Matched | Precision | Recall | Hitter |
|---|---|---|---|---|---|---|---|
| Shipped (5 ranges of Axelsen-Ginting) | the 8 other matches | 8,329 | 7,332 | 5,909 | 80.6% | 70.9% | 97.2% |
| Trained on the other 8 matches | the same 8 matches, each held out | 8,329 | 8,332 | 6,866 | **82.4%** | **82.4%** | **98.7%** |
| Trained on the other 8 matches | all 9 matches, each held out | 9,424 | 9,354 | 7,717 | 82.5% | 81.9% | 98.8% |
| Trained on the other 8 matches, ±10 | all 9 matches, each held out | 9,424 | 9,354 | 8,212 | 87.8% | 87.1% | 98.4% |

More training data mostly adds recall: +11.5 points on matches neither model saw, for
+1.8 points of precision. Hitter accuracy is over matched hits whose label has a
player position.

**Not memorising the players.** Folds are by match, but players recur: Momota - Chou is
three of the nine matches, Marin and Antonsen play in two each, and Axelsen plays in
Antonsen - Axelsen too. Leaving out of the training *every* match that shares a player
with the held-out one (`--by-player`; 6 to 8 training matches) gives
**81.7% / 80.9%** at ±5 over all nine (87.5% / 86.6% at ±10, hitter 98.4%), against
82.5% / 81.9%. About 1 point comes from having seen the same players; the gain over the
shipped model stays at about 10 points of recall. Axelsen - Ginting alone:
82.1% / 77.7% (vs 83.3% / 77.7%).

**On the original five ranges** (the 128 hits of `hits_movement.md`), the model trained
on the eight matches *without* Axelsen-Ginting scores 122 detected, 113 matched:
**92.6% precision, 88.3% recall** at ±5 (94.3% / 89.8% at ±10), hitter 113/113, against
85.8% / 80.5% for the five-range model with each range held out, and 90.5% / 89.1% for
the old shuttle-based detector. Per range at ±5: 80.8% / 80.8%, 96.2% / 92.6%,
92.3% / 88.9%, 100% / 94.4%, 96.3% / 86.7%. The nine-match model (which has seen these
hits) gets 88.9% / 87.5%; that is in-sample and not an estimate.

## Data

| Match | fps | Coarse offset (clarity) | Fine offset (clarity) | Stretches | Frames tracked | Labelled hits |
|---|---|---|---|---|---|---|
| Axelsen - Ginting, Thailand Open 2021 SF | 30 | +48 (0.59) | 0 (0.40) | 107 | 39,391 | 1,095 |
| Momota - Chou, Fuzhou Open 2019 F | 25 | +3 (0.57) | -2 (0.32) | 113 | 48,311 | 1,644 |
| Momota - Chou, Korea Open 2019 F | 25 | +42 (0.55) | -1 (0.31) | 78 | 27,143 | 928 |
| Momota - Chou, Denmark Open 2018 F | 25 | -3 (0.49) | -1 (0.33) | 112 | 39,323 | 1,311 |
| Marin - Katethong, Thailand Open 2021 QF | 30 | +36 (0.55) | +1 (0.33) | 74 | 24,050 | 629 |
| Antonsen - Verma, Thailand Open 2021 QF | 30 | +51 (0.54) | 0 (0.36) | 116 | 42,779 | 1,204 |
| Marin - Yigit, Thailand Open 2021 QF | 30 | +30 (0.40) | +1 (0.34) | 63 | 20,228 | 518 |
| Ng - Srikanth, World Tour Finals 2020 QF | 30 | +48 (0.58) | +1 (0.38) | 112 | 42,976 | 1,200 |
| Antonsen - Axelsen, World Tour Finals 2020 F | 30 | +57 (0.38) | 0 (0.34) | 101 | 33,010 | 895 |
| **Total** | | | | **876** | **317,211** | **9,424** |

- **Choice of matches.** Ten matches were picked whose labels and YouTube video share
  the frame rate (no scaling), 1280x720 H.264, singles, with a main broadcast camera.
  Eight were downloaded with yt-dlp; Axelsen-Ginting was already in `data/`.
- **Excluded: Chou - Christie, Sudirman Cup 2019 QF.** Its coarse offset was unclear
  (correlation 0.07, clarity 0.03, below the 0.1 threshold): the copy online does not
  follow the labelled rallies. The replacement (Chou - Christie, Indonesia Open 2019)
  could not be downloaded, because YouTube asked for a signed-in session. I did not
  pass browser cookies to the downloader. Nine matches remain.
- **Alignment** (`vision/label_offset.py`, `docs/vision/label_offset.md`). The coarse
  offset comes from broadcast camera cuts and is only good to about 2 s; it sits
  +30 to +57 frames late on most matches. The fine offset is the shift within ±3 s
  of it at which the shipped model's hitter score peaks on the labelled hits. All
  nine fine offsets are within 2 frames of 0, and all nine fine clarities
  (0.31-0.40) are above the 0.2 threshold, so no match was excluded at that step.
  Axelsen-Ginting, the one match whose offset was known by hand (0), comes out at 0.
- **What was tracked.** Only the labelled rallies: each rally from its first to its
  last label, padded by 2 s each side, with stretches less than 1 s apart merged. The
  player tracker (YOLO11s + ByteTrack on MPS) ran once per stretch, with one model
  loaded for the whole match (`track_players(..., tracker=...)`). The positions and
  boxes files sit under `data/players/` like any other range.
- **Tracking time.** 6 h 26 min for 317,211 frames (about 14 frames/s), one GPU job at
  a time, with no NMS time-limit drops.
- **Restart at batch 4.** The first two starts were stopped by the swap guard. The
  first was a one-poll spike to 5.6 GB (from 2.9 GB), so the guard now needs two polls
  in a row above the limit. The second was a sustained rise to 5.2 GB at 8 frames per
  call, so the restart used 4 frames per call: same speed, half the GPU memory, swap
  steady at about 3 GB for the rest of the night. Nothing was tracked twice and nothing
  is missing. A stretch's files are written only once the stretch is complete, so the
  stretch that was cut off was tracked again from its first frame. A restart skips
  stretches that already have both files. The 3 stretches (923 frames) finished
  before the restart were kept. All 876 stretches have both files, and no two overlap.

## Setup
- **Labels.** Every ShuttleSet shot in a tracked rally, at
  `frame_num x label_scale + fine offset`, with the hitter from `player_location`
  through the match's homography (as in `hits_movement.md`). Serves are included in the
  labelled count; the inferred serve rows are left out of the detections.
- **Training targets.** As for the shipped model: frames within ±2 of a hit of that
  player, on frames where at least one player was detected.
- **Each match held out.** The scorer is trained on the other eight matches and scores
  the held-out one. Matching is one to one, closest first, within ±5 frames (±10 also
  reported).
- **Thresholds.** The headline uses the shipped thresholds (hit score 0.5, rally-end
  score 0.7). I also tuned them for each held-out match (`--tune`). The grid is hit
  score 0.35-0.65 x rally-end score 0.5-0.8, picked by best F1 at ±5 over the eight
  training matches. Each training match is scored by a model trained on the other seven,
  so the held-out match plays no part in the choice. Tuning gives 83.9% / 81.2%
  (89.1% / 86.2% at ±10): +1.4 points of precision for -0.7 of recall. The chosen
  hit score is 0.55 for seven matches and 0.65 for the two with Antonsen (rally-end
  0.5-0.7), always a little above the shipped 0.5. The thresholds stay where they are.
- **Settings by fps.** As shipped: `settings_for_video` scales the frame windows and
  gaps by fps, so the 25 fps matches use the 30 fps settings converted to time.

## Held out, per match (±5 / ±10, shipped thresholds)

| Match | fps | Labelled | Detected | Precision ±5 | Recall ±5 | Precision ±10 | Recall ±10 | Hitter ±5 | Shipped model ±5 |
|---|---|---|---|---|---|---|---|---|---|
| Axelsen - Ginting | 30 | 1,095 | 1,022 | 83.3% | 77.7% | 87.7% | 81.8% | 845/850 | 82.9% / 73.7% (partly in-sample) |
| Momota - Chou, Fuzhou | 25 | 1,644 | 1,668 | 83.8% | 85.0% | 89.9% | 91.2% | 1373/1397 | 82.3% / 72.5% |
| Momota - Chou, Korea | 25 | 928 | 954 | 81.2% | 83.5% | 86.9% | 89.3% | 760/774 | 78.0% / 70.5% |
| Momota - Chou, Denmark | 25 | 1,311 | 1,388 | 82.4% | 87.3% | 88.5% | 93.7% | 1129/1144 | 78.1% / 70.6% |
| Marin - Katethong | 30 | 629 | 622 | 82.2% | 81.2% | 88.3% | 87.3% | 510/511 | 82.8% / 71.1% |
| Antonsen - Verma | 30 | 1,204 | 1,144 | 79.4% | 75.4% | 85.5% | 81.2% | 885/903 | 80.4% / 67.6% |
| Marin - Yigit | 30 | 518 | 505 | 83.2% | 81.1% | 86.7% | 84.6% | 417/419 | 79.9% / 69.7% |
| Ng - Srikanth | 30 | 1,200 | 1,170 | 87.5% | 85.3% | 90.9% | 88.7% | 1016/1022 | 86.0% / 75.7% |
| Antonsen - Axelsen | 30 | 895 | 881 | 78.0% | 76.8% | 83.0% | 81.7% | 676/687 | 75.8% / 67.9% |

## 25 vs 30 fps (each match held out, ±5)

| fps | Matches | Labelled | Precision | Recall | ±10 precision / recall | Hitter |
|---|---|---|---|---|---|---|
| 25 | 3 (all Momota - Chou) | 3,883 | 82.7% | 85.4% | 88.7% / 91.6% | 98.4% |
| 30 | 6 | 5,541 | 82.4% | 79.4% | 87.1% / 84.0% | 99.0% |

The fps scaling works: 25 fps is not worse. Recall is 6 points *higher* at 25 fps, but
all three 25 fps matches are the same two players, so this is as likely Momota and
Chou's style (long, regular rallies) as the frame rate; with these matches the two
can't be told apart. Among the 30 fps matches the spread is 75-85% recall, and the two
matches with Antonsen are the lowest.

## Learning curve (each match held out, ±5, shipped thresholds)

For each held-out match the scorer is trained on 1, 3, 5 or all 8 of the other matches
(a random subset, seed 2026), then all nine held-out results are added up.

| Training matches | Detected | Precision | Recall | ±10 precision / recall | Hitter |
|---|---|---|---|---|---|
| 1 | 8,818 | 83.2% | 77.8% | 88.8% / 83.1% | 98.5% |
| 3 | 9,333 | 81.7% | 80.9% | 87.5% / 86.7% | 98.5% |
| 5 | 9,295 | 82.3% | 81.2% | 88.2% / 87.0% | 98.7% |
| 8 | 9,354 | 82.5% | 81.9% | 87.8% / 87.1% | 98.8% |

Most of the gain is from one whole match (about 1,000 hits) over five ranges (128 hits):
recall 70.9% to 77.8% on unseen matches. Three matches add 3 more points, and five
to eight add under 1. The scorer is near its ceiling with these 20 movement features;
more ShuttleSet matches won't move it much. What's left is the rhythm step's misses and
the features themselves.

**What the sample bears.** 9,424 hits from nine matches and seven pairs of players:
about ±0.8 points at 95% confidence on the totals if hits were independent; they are
not (hits in a rally share a tracking quality), so match-to-match spread (75-87% recall)
is the honest error bar for a new broadcast match. None of this is my club footage:
the camera angle there is different, and these numbers don't transfer to it without
labelled club hits.

## How to run

Per match (videos from the `url` in ShuttleSet's `match.csv`, saved as
`data/shuttleset_videos/<folder>.mp4`; Axelsen - Ginting is `data/match.mp4`):
```
M=Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final
V=data/shuttleset_videos/$M.mp4
.venv/bin/python -m vision.court $V --shuttleset-match $M                                  # calibration from the homography
.venv/bin/python -m vision.label_offset $V --match $M --label-fps 25                       # coarse offset
.venv/bin/python -m vision.label_offset $V --stretches                                     # labelled rallies to track
.venv/bin/python -m vision.players $V --stretches data/alignment/shuttleset_videos_${M}_stretches.csv --batch-frames 4   # GPU, resumable
.venv/bin/python -m vision.label_offset $V --fine                                          # fine offset
```
Then all matches together, from a list file with one `folder video` pair per line (the
ten lines are the nine matches above plus Chou - Christie, which the command leaves out
for its coarse clarity):
```
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --held-out --tune --by-player --learning-curve \
    --baseline data/models/movement_hits.joblib --save data/models/movement_hits_ten_matches.joblib       # 17 min, CPU
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --save data/hit_training/model_without_axelsen_ginting.joblib \
    --exclude Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.movement data/match.mp4 --start 14201 --end 14947 \
    --evaluate Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals 1 \
    --model data/hit_training/model_without_axelsen_ginting.joblib --no-serves --hits-dir /tmp/ten_matches   # and the other four ranges
```
Every number on this page comes out of these commands. Their tables are in
`data/hit_training/` (`held_out.csv`, `held_out_by_player.csv`, `learning_curve.csv`,
`baseline.csv`). The shipped-model rows of the headline are `baseline.csv` without
Axelsen - Ginting. Run from the repo, all nine fine offsets and clarities, the stretch
lists, the held-out counts, the learning curve, the baseline and the five-range figures
are equal to the first run's.

## Decisions
- Nine matches, not ten: Chou - Christie dropped (no alignment) and no replacement
  downloadable without cookies.
- Only labelled rallies are tracked, padded 2 s: everything outside them has no labels
  and would only add negatives the rally step never sees.
- Tracking at 4 frames per GPU call for this run (memory); the module default stays 8.
- Thresholds unchanged. Tuning, nested so the held-out match plays no part, moved about
  1 point between precision and recall.
- The new model is written to its own file. Whether it should become the default is the
  manager's call. On every broadcast measurement it is better: +11 points of recall
  on unseen matches, and +7.8 precision / +7.8 recall on the five Axelsen ranges.
  Its effect on the club stand-in video is not measured here.
