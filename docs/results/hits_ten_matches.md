# Movement hit scorer trained on nine ShuttleSet matches: results

The movement hit detector (`vision/movement.py`, `docs/results/hits_movement.md`) was
trained on 128 hand-checked hits from five ranges of one match. This page trains the
same scorer on every labelled hit of nine ShuttleSet matches (9,424 hits, 45,594 hit
frames among 581,694 player-frames) and measures it with each match held out in turn.
The new model is a separate file, `data/models/movement_hits_ten_matches.joblib`; the
default `data/models/movement_hits.joblib` is unchanged.

**Scoring windows.** A detection matches a labelled hit within **±0.25 s** (also
reported: ±0.50 s). The window is turned into frames at each match's own fps: ±6 / ±13
frames at 25 fps, ±8 / ±15 at 30 fps (`vision.hits.MATCH_TOLERANCE_SECONDS`). The
earlier figures on this page, scored at ±5 / ±10 frames whatever the fps, are in the
git history.

## Headline (each match held out, ±0.25 s, same thresholds as the shipped model)

| Model | Scored on | Labelled | Detected | Matched | Precision | Recall | Hitter |
|---|---|---|---|---|---|---|---|
| Shipped (5 ranges of Axelsen-Ginting) | the 8 other matches | 8,329 | 7,332 | 6,144 | 83.8% | 73.8% | 96.7% |
| Trained on the other 8 matches | the same 8 matches, each held out | 8,329 | 8,332 | 7,091 | **85.1%** | **85.1%** | **98.4%** |
| Trained on the other 8 matches | all 9 matches, each held out | 9,424 | 9,354 | 7,970 | 85.2% | 84.6% | 98.5% |
| Trained on the other 8 matches, ±0.50 s | all 9 matches, each held out | 9,424 | 9,354 | 8,403 | 89.8% | 89.2% | 98.2% |

More training data mostly adds recall: +11.3 points on matches neither model saw, for
+1.3 points of precision. Hitter accuracy is over matched hits whose label has a
player position.

**Not memorising the players.** Folds are by match, but players recur: Momota - Chou is
three of the nine matches, Marin and Antonsen play in two each, and Axelsen plays in
Antonsen - Axelsen too. Leaving out of the training *every* match that shares a player
with the held-out one (`--by-player`; 6 to 8 training matches) gives
**84.6% / 83.8%** at ±0.25 s over all nine (89.4% / 88.5% at ±0.50 s, hitter 98.1%),
against 85.2% / 84.6%. Under 1 point comes from having seen the same players; the gain
over the shipped model stays at about 10 points of recall. Axelsen - Ginting alone:
84.9% / 80.4% (against 86.0% / 80.3%).

**On the original five ranges** (the 128 hits of `hits_movement.md`), the model trained
on the eight matches *without* Axelsen-Ginting scores 122 detected, 114 matched:
**93.4% precision, 89.1% recall** at ±0.25 s (95.9% / 91.4% at ±0.50 s), hitter
114/114.

- **Five-range model, each range held out** (`--cross-validate`): 88.3% / 82.8%.
- **Old shuttle-based detector:** 90.5% / 89.1%. That figure was scored at ±5 frames and
  isn't re-run here: the shuttle tracker is out of the pipeline.
- **Per range at ±0.25 s:** 80.8% / 80.8%, 100% / 96.3%, 92.3% / 88.9%, 100% / 94.4%,
  96.3% / 86.7%.
- **The nine-match model, which has seen these hits:** 91.3% / 89.8%. That is in-sample
  and not an estimate.

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
  the held-out one. Matching is one to one, closest first, within ±0.25 s (±0.50 s also
  reported), in frames at the match's fps.
- **Thresholds.** The headline uses the shipped thresholds (hit score 0.5, rally-end
  score 0.7). I also tuned them for each held-out match (`--tune`). The grid is hit
  score 0.35-0.65 x rally-end score 0.5-0.8, picked by best F1 at ±0.25 s over the
  eight training matches. Each training match is scored by a model trained on the other
  seven, so the held-out match plays no part in the choice. Tuning gives 85.0% / 85.0%
  (89.6% / 89.6% at ±0.50 s): -0.2 points of precision for +0.4 of recall. The chosen
  hit score is 0.55 for all nine matches (rally-end 0.5-0.7), a little above the
  shipped 0.5. The thresholds stay where they are. `--save --tune` gives 0.55 / 0.6 for
  the shipped nine-match model, tuned on all nine.
- **Settings by fps.** As shipped: `settings_for_video` scales the frame windows and
  gaps by fps, so the 25 fps matches use the 30 fps settings converted to time.

## Held out, per match (±0.25 s / ±0.50 s, shipped thresholds)

| Match | fps | Labelled | Detected | Precision ±0.25 s | Recall ±0.25 s | Precision ±0.50 s | Recall ±0.50 s | Hitter ±0.25 s | Shipped model ±0.25 s |
|---|---|---|---|---|---|---|---|---|---|
| Axelsen - Ginting | 30 | 1,095 | 1,022 | 86.0% | 80.3% | 89.2% | 83.3% | 871/878 | 86.6% / 77.0% (partly in-sample) |
| Momota - Chou, Fuzhou | 25 | 1,644 | 1,668 | 85.4% | 86.7% | 91.4% | 92.7% | 1399/1425 | 84.3% / 74.3% |
| Momota - Chou, Korea | 25 | 928 | 954 | 83.1% | 85.5% | 89.3% | 91.8% | 776/792 | 80.1% / 72.3% |
| Momota - Chou, Denmark | 25 | 1,311 | 1,388 | 84.4% | 89.3% | 90.3% | 95.6% | 1154/1171 | 81.7% / 73.8% |
| Marin - Katethong | 30 | 629 | 622 | 86.3% | 85.4% | 90.7% | 89.7% | 533/537 | 85.7% / 73.6% |
| Antonsen - Verma | 30 | 1,204 | 1,144 | 84.0% | 79.8% | 88.9% | 84.5% | 933/956 | 83.8% / 70.4% |
| Marin - Yigit | 30 | 518 | 505 | 85.3% | 83.2% | 88.9% | 86.7% | 427/430 | 83.2% / 72.6% |
| Ng - Srikanth | 30 | 1,200 | 1,170 | 90.1% | 87.8% | 92.3% | 90.0% | 1039/1052 | 90.3% / 79.5% |
| Antonsen - Axelsen | 30 | 895 | 881 | 81.6% | 80.3% | 85.4% | 84.0% | 706/719 | 80.3% / 72.0% |

## 25 vs 30 fps (each match held out)

| fps | Matches | Labelled | Precision ±0.25 s | Recall ±0.25 s | ±0.50 s precision / recall | Hitter |
|---|---|---|---|---|---|---|
| 25 (±6 / ±13 frames) | 3 (all Momota - Chou) | 3,883 | 84.5% | 87.3% | 90.5% / 93.5% | 98.3% |
| 30 (±8 / ±15 frames) | 6 | 5,541 | 85.7% | 82.7% | 89.3% / 86.2% | 98.6% |

The fps scaling works: 25 fps is not worse. At the same time window, recall is 4.6
points *higher* at 25 fps. But all three 25 fps matches are the same two players, so
this is as likely Momota and Chou's style (long, regular rallies) as the frame rate;
with these matches the two can't be told apart. Among the 30 fps matches the spread is
80-88% recall, and the two matches with Antonsen are the lowest.

## Learning curve (each match held out, ±0.25 s, shipped thresholds)

For each held-out match the scorer is trained on 1, 3, 5 or all 8 of the other matches
(a random subset, seed 2026), then all nine held-out results are added up.

| Training matches | Detected | Precision ±0.25 s | Recall ±0.25 s | ±0.50 s precision / recall | Hitter |
|---|---|---|---|---|---|
| 1 | 8,818 | 86.3% | 80.8% | 91.1% / 85.3% | 98.3% |
| 3 | 9,333 | 84.7% | 83.8% | 89.3% / 88.4% | 98.3% |
| 5 | 9,295 | 85.5% | 84.3% | 89.9% / 88.7% | 98.3% |
| 8 | 9,354 | 85.2% | 84.6% | 89.8% / 89.2% | 98.5% |

Most of the gain is from one whole match (about 1,000 hits) over five ranges (128 hits):
recall 73.8% to 80.8% on unseen matches. Three matches add 3 more points, and five
to eight add under 1. The scorer is near its ceiling with these 20 movement features;
more ShuttleSet matches won't move it much. What's left is the rhythm step's misses and
the features themselves.

**What the sample bears:**

- 9,424 hits from nine matches and seven pairs of players. If hits were independent,
  the totals would be good to about ±0.8 points at 95% confidence. They aren't (hits
  in a rally share a tracking quality), so the match-to-match spread (80-89% recall) is
  the honest error bar for a new broadcast match.
- None of this is club footage: the camera angle there is different, and these numbers
  don't transfer to it without labelled club hits (`docs/results/hits_club.md`).

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
    --baseline data/models/movement_hits.joblib --save data/models/movement_hits_ten_matches.joblib       # 20 min, CPU
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --save data/hit_training/model_without_axelsen_ginting.joblib \
    --exclude Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.movement data/match.mp4 --start 14201 --end 14947 \
    --evaluate Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals 1 \
    --model data/hit_training/model_without_axelsen_ginting.joblib --no-serves --hits-dir /tmp/ten_matches   # and the other four ranges
.venv/bin/python -m vision.movement data/match.mp4 --cross-validate --match Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals \
    --range 14201 14947 1 --range 16551 17343 1 --range 81360 82116 2 --range 107560 108103 3 --range 114673 116641 3
```
The in-sample five-range figure is the same `--evaluate` with
`--model data/models/movement_hits_ten_matches.joblib`. Every number on this page comes
out of these commands. Their tables are in
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
