# Shuttle flight as extra shot-type evidence: an experiment

The feet-only shot classifier mixes up lifts and drives, and smashes with drops and
clears, on club footage. The shuttle's flight shows high against flat and steep against
soft, so this experiment adds it as evidence. The players still decide where the hits
are; the shuttle only adds features to the shot type. Nothing here is shipped, and no
default changed.

Everything below is at 30 fps (`docs/vision/halve_fps.md`), except the two training
videos (Lin v Yeung and Chong v Ding), which are 24 fps.

## Stage A: does the tracker see the shuttle?

The existing TrackNetV3 tracker (`vision/shuttle.py`) runs on the Mac's GPU (`--device
mps`), in 600-frame stretches. It uses `--low-memory-background`, a background median
taken at the model's input size: the full-size median needs several GB at 1080p and
tripped the swap guard. On a 600-frame stretch the two backgrounds gave the same
positions, with a median difference of 0 px and 15 of 592 frames more than 5 px apart.

Three rules, all set before the results were read (`vision/flight.py`):
- **Seen:** at least one real detection (visible, not filled in by InpaintNet) in the
  0.5 s after a labelled hit.
- **Usable:** real detections in at least half the frames of that 0.5 s.
- **Smooth:** no step between detections of more than 15% of the frame width per frame.

The smooth rule was added after the first sample sheet of test game 1. There, about 5 of
20 tracks jumped to a different shuttle: on a neighbouring court, or in the pile on the
floor.

| Video | Labelled hits | Seen | Usable | **Usable and smooth** | Near (me) / far (opp), smooth | Rally frames with the shuttle seen |
|---|---|---|---|---|---|---|
| Test game 1 (`UHKzdJqgFwk_30`, labelled range, 720p) | 110 | 100% | 86.4% | **72.7%** | 69.8% / 75.4% | 67.9% |
| Stand-in set 1 (`set1_30`, 1080p) | 172 | 100% | 99.4% | **97.7%** | 95.6% / 100% | 79.9% |

By labelled type, usable and smooth:

| Type | Test game 1 | Set 1 |
|---|---|---|
| net | 18/21 | 46/46 |
| lift | 19/27 | 29/29 |
| serve | 9/13 | 31/31 |
| clear | 10/10 | 17/18 |
| drop | 6/11 | 20/23 |
| drive | 4/6 | 18/18 |
| smash | 10/16 | 5/5 |

- **On set 1 the tracker sees almost every flight.** Its camera is behind the near
  baseline, high, at 1080p, with a plain wall behind the far court. On the sample sheet,
  19 of 20 tracks are clean arcs.
- **On test game 1 it sees about three in four,** passing the 70% gate by a little. The
  low wide-angle view takes in other courts, so stray shuttles cause the losses, not the
  far half (the far player is no worse). Smashes and drops, the fast and the short
  shots, are seen least.
- **Time:** GPU minutes per minute of footage, with `nonoverlap`:
  - 2.4 for the 24 fps 720p training videos;
  - 3.2 for test game 1 (30 fps, 720p);
  - 3.9 for set 1 (30 fps, 1080p).
- Sample sheets: `data/checks/shuttle/<video id>_flight_sheet.jpg`.

## Stage B: feet only against feet + flight

**Flight features** (`vision/flight_types.py`) come from the smooth detections between a
hit and the next hit. The image distances are turned into metres using the court's
pixel size at the hitter's feet (from the calibration):
- `flight_peak_m`: how far the shuttle rises above its first detection before the next
  hit;
- `flight_start_rising`: whether it goes up over the first 0.3 s;
- `flight_start_speed_m_per_s`: its speed over the first 0.3 s;
- `flight_seen_share`: the share of frames until the next hit with the shuttle seen.

Flight time is already a feet feature. A hit without a smooth, usable flight (fewer than
3 detections in its first 0.3 s, the first one later than 0.15 s, or a jump) gets
missing flight features. The classifier then works from its feet features alone.

**Classifiers.** These are the same model family as the shipped shot classifier, a
gradient-boosted tree (seed 0), on the same feet features, over 8 labels. They are
trained on the user's club labels: Lin first, then Lin + Ding. The training rows are
the labelled hits run through `vision.features`, as if they were detections:

| Training | Labelled hits with feet points | With smooth flight features |
|---|---|---|
| Lin | 354 (lift 91, clear 64, drop 47, short serve 46, net 45, smash 28, long serve 21, drive 12) | 239 |
| Lin + Ding | 704 (lift 180, net 122, clear 112, short serve 90, drop 71, smash 64, drive 43, long serve 22) | 372 |

**Test.** The hits come from the nine + Lin + Ding tuned scorer, alone and inside the
combined detector (`docs/results/club_test.md`, "At 30 fps"). Only the shot types change
between rows. The reference is the shipped pro-trained feet-only classifier on the same
rows. Scores are at ±0.50 s against the user's labels.

### Test game 1 (110 hits; 73 of 247 typed rows have smooth flight features)

Correct shots are matched and typed right, divided by labelled hits plus false hits
inside the labelled rallies:

| Classifier | Hits alone | Hits combined |
|---|---|---|
| Pro-trained feet (shipped) | 49/125 = 39.2% | 48/123 = 39.0% |
| Club feet, Lin | **64/125 = 51.2%** | **62/123 = 50.4%** |
| Club feet + flight, Lin | 63/125 = 50.4% | 60/123 = 48.8% |
| Club feet, Lin + Ding | 62/125 = 49.6% | 60/123 = 48.8% |
| Club feet + flight, Lin + Ding | 62/125 = 49.6% | 61/123 = 49.6% |

Shot type on matched hits, Lin + Ding classifiers, hits alone (combined):

| | Pro feet | Club feet | Club feet + flight |
|---|---|---|---|
| With serves | 52.1% (55.2%) | 66.0% (69.0%) | 66.0% (70.1%) |
| Labelled serves left out | 53.7% (51.2%) | 69.5% (66.2%) | 69.5% (67.5%) |
| lift | 10/24 (10/24) | 21/24 (20/24) | 21/24 (20/24) |
| drive | 1/5 (1/5) | 0/5 (0/5) | 0/5 (0/5) |
| smash | 3/12 (3/12) | 5/12 (5/12) | 5/12 (5/12) |
| drop | 6/11 (5/11) | 7/11 (7/11) | 7/11 (7/11) |
| clear | 9/10 (8/8) | 9/10 (7/8) | 8/10 (7/8) |

- The pro classifier types 10 of 24 lifts as drives; the club one types 1. The club
  classifiers are wrong on drives instead: 3 of 5 drives become smashes.
- The serve rows decide the serves, so all three classifiers type the same serves.

### Stand-in set 1 (172 hits; 137 of 152 typed rows have smooth flight features)

Lin + Ding classifiers:

| Classifier | Correct shots, hits alone | Correct shots, combined | Shot type with serves, alone (combined) |
|---|---|---|---|
| Pro-trained feet (shipped) | **102/214 = 47.7%** | **98/201 = 48.8%** | **73.9% (76.6%)** |
| Club feet | 94/214 = 43.9% | 92/201 = 45.8% | 68.1% (71.9%) |
| Club feet + flight | 96/214 = 44.9% | 95/201 = 47.3% | 69.6% (74.2%) |

Per group, hits alone (pro, then club feet, then club feet + flight):
- lift: 20/24, 23/24, 23/24;
- drop: 17/22, 12/22, 12/22 (the club classifiers type 6-7 drops as smashes);
- drive: 3/15, 1/15, 0/15;
- net: 38/39, 35/39, 36/39;
- clear: 12/14, 11/14, 12/14.

## What it says

- **Club labels are the big gain on club footage.** On test game 1, a feet-only
  classifier trained on the user's club labels gets about 10-12 points more correct
  shots than the shipped pro-trained one. Most of that is the lift/drive mix-up going
  away.
- **The gain depends on the camera.** On set 1, filmed from high behind the baseline
  like the broadcast matches, the pro-trained classifier stays best, by 1.5-4 points. The
  club classifiers lose drops to smash there. The club training videos and test game 1
  are all low corner views.
- **Flight adds little.** On test game 1 it adds -2 to +1 correct shots. Only 73 of 247
  typed rows have smooth flight there, partly because rows after frame 7,802 have no
  shuttle track (none of them is scored). On set 1, where 137 of 152 rows have it, it
  adds 2-3 shots over club feet, about one each on clear, smash and net. It never beats
  the best feet-only classifier for the view.
- **Drives stay the weakest type** for every classifier: 0-3 of 15 on set 1, 0-1 of 5 on
  test game 1.
- **Small numbers:** 110 and 172 hits, two games, two training videos. A difference of
  1-3 shots is within noise.

## How to run

```
S=data/recordings; T=data/club_test
for job in club/UHKzdJqgFwk_30:0:7802 stand_in/set1_30:0:13117 club/q3pfhnvKPig:0:29710 club/E6SRRmvJ6dI:0:31799; do
  IFS=: read video first last <<< "$job"
  for start in $(seq $first 600 $last); do end=$(( start + 599 > last ? last : start + 599 ))
    .venv/bin/python -m vision.shuttle $S/$video.mp4 --start $start --end $end --ensemble nonoverlap --device mps --low-memory-background
  done
done
.venv/bin/python -m vision.flight $S/club/UHKzdJqgFwk_30.mp4 --labels data/labels/club_UHKzdJqgFwk_30.csv --end 7802 --sheet
.venv/bin/python -m vision.flight $S/stand_in/set1_30.mp4 --labels data/labels/stand_in_set1_30.csv --end 13117 --sheet
LIN="--train $S/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/flight/club_q3pfhnvKPig/scout.duckdb 29710"
DING="--train $S/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv data/flight/club_E6SRRmvJ6dI/scout.duckdb 31799"
.venv/bin/python -m vision.flight_types ${=LIN} ${=DING} --hits-out data/flight/hits
for pair in $S/club/q3pfhnvKPig.mp4:club_q3pfhnvKPig:29710 $S/club/E6SRRmvJ6dI.mp4:club_E6SRRmvJ6dI:31799; do
  IFS=: read video match end <<< "$pair"; mkdir -p data/flight/$match; cp data/scout.duckdb data/flight/$match/scout.duckdb
  .venv/bin/python -m vision.features $video --match-id $match --range 0 $end 1 --database data/flight/$match/scout.duckdb --hits-dir data/flight/hits
done
.venv/bin/python -m vision.flight_types ${=LIN} ${=DING} \
  --predict $S/club/UHKzdJqgFwk_30.mp4 $T/UHKzdJqgFwk_30/lin_ding_alone/scout.duckdb 7802 data/flight/lin_ding/alone \
  --predict $S/club/UHKzdJqgFwk_30.mp4 $T/UHKzdJqgFwk_30/lin_ding_combined/scout.duckdb 7802 data/flight/lin_ding/combined
```

The Lin-only run is the same without `$DING` (output `data/flight/lin`), and set 1 uses
`$S/stand_in/set1_30.mp4`, `$T/stand_in_set1_30/...` and `13117` (output
`data/flight/lin_ding_set1`). Each output folder is scored with `vision.label_scores
--fps 30` against the video's 30 fps labels. Each mode gets three `--run`s on the same
hits file: the original database (`<mode>_pro_feet`), `feet_only.duckdb` and
`feet_flight.duckdb`. The reports are `data/flight/{lin,lin_ding,lin_ding_set1}/report.md`.
