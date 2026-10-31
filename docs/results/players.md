# Player tracking: results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.players data/match.mp4 --shuttleset-rally $M 1 4 --overlay-frame 14700
.venv/bin/python -m vision.players data/match.mp4 --shuttleset-rally $M 1 6 --overlay-frame 16900 --overlay-frame 17294
.venv/bin/python -m vision.players data/match.mp4 --shuttleset-rally $M 2 22 --overlay-frame 81700 --overlay-frame 82048
```
Each run writes `data/players/match/positions_<start>_<end>.csv` and the overlays to
`data/checks/players/`. It needs `data/calibration/match/court.json` (court calibration)
and `data/models/yolo11s.pt`.

## Setup
- Video: `data/match.mp4`, Axelsen vs Ginting (Thailand Open 2021 SF), 1280 x 720, 30 fps.
  Broadcast camera behind one baseline; ShuttleSet labels align at offset 0.
- Model: YOLO11s (person class), ByteTrack with Ultralytics' default settings,
  inference at 1280 px. The accuracy runs below were made on the CPU, one frame per
  call (about 0.26 s per frame); detection now runs on the Mac's GPU, 8 frames per call,
  with the same output (see "Speed" below).
- Three rallies, chosen as the longest in their set: set 1 rally 4 (26 shots) and
  rally 6 (27 shots), and set 2 rally 22 (27 shots). In set 2 the players have
  changed ends: Ginting is near (`me`) in set 1 and far (`opp`) in set 2.
- Measured frames: main-view frames from the rally's first labelled hit to its
  last. A frame where a player has no row, or a different track id than that
  player's most common one, counts as a failure.

## Results

| Rally | In-rally frames | Measured (main view) | `me`: one id | `opp`: one id | `me` in near half | `opp` in far half |
|---|---|---|---|---|---|---|
| set 1, rally 4 | 687 | 687 | 687/687 = 100% | 687/687 = 100% | 100% | 100% |
| set 1, rally 6 | 733 | 722 | 722/722 = 100% | 722/722 = 100% | 100% | 100% |
| set 2, rally 22 | 697 | 686 | 686/686 = 100% | 686/686 = 100% | 100% | 100% |

Target was at least 95% of frames with one id; all six player-rallies reach 100%,
each with a single ByteTrack id for the whole rally and a detection in every
measured frame.

Position ranges (meters, whole tracked range including padding):

| Rally | `me` x | `me` y | `opp` x | `opp` y |
|---|---|---|---|---|
| set 1, rally 4 | 1.58 to 4.81 | -0.32 to 4.11 | 1.83 to 4.80 | 7.72 to 12.97 |
| set 1, rally 6 | 1.18 to 4.46 | -0.26 to 3.88 | 0.99 to 4.83 | 8.33 to 14.49 |
| set 2, rally 22 | 1.23 to 4.60 | 0.34 to 4.91 | 1.66 to 5.22 | 8.74 to 13.81 |

Players stay between the singles sidelines (0.46 to 5.64) and well clear of the net.
`me` slightly behind the near baseline (-0.3 m) is real: players step behind the
line to return deep shots. `opp` at 14.49 is a jump smash (see below).

## Fixes made while measuring

**Duplicate box on the far player (set 2, rally 22).** The first run gave `opp` two
track ids: 647/686 = 94.3%, below target. At frame 82047 YOLO returned two
overlapping boxes on the far player, a narrow one (the existing track) and a wider one
that included the player's outstretched arm. The wide box started a new track, and on the
next frame it was the only box left, so the player continued under the new id.
Lowering the non-maximum-suppression overlap threshold from YOLO's default 0.7
to 0.5 removes the duplicate box; the rerun keeps one id for 686/686 frames, and
the other two rallies are unchanged at 100%.

**Jumping far player dropped (set 1, rally 6).** With a 1 m margin all round,
`opp` had no row for frames 17293-17296 (718/722 = 99.4%). The player was in the air for
a jump smash near the back line. The calibration maps image points as if they
were on the floor, so raised feet map further back: the foot point went from
13.4 m to beyond 14.4 m. Behind the far baseline the margin is now 2 m (the
nearest bystanders there map beyond 17 m); the rerun keeps the player in every frame.
A jumping near player rises in the image towards the net, so the near end keeps
the 1 m margin. Positions of an airborne player are too far back by up to about
1 m, which matters for later shot features but not for tracking.

## Skipped frames (not the main camera view)

| Range | Frames | Main view | Skipped |
|---|---|---|---|
| set 1, rally 4 (with 30-frame padding) | 747 | 718 | 29 |
| set 1, rally 6 (with padding) | 793 | 752 | 41 |
| set 2, rally 22 (with padding) | 757 | 716 | 41 |
| all of set 1, first to last labelled hit (frames 12337 to 54360, view check only) | 42,024 | 17,435 | 24,589 |

The view check finds 97-99% of the court-line samples on the main view and
1-26% on close-ups and replays, so the 0.6 threshold has a wide gap on both sides.

In set 1, 58% of the broadcast between the first and last hit is not the main
view (replays, close-ups and crowd shots between rallies). Inside rallies (first
to last labelled hit) only 265 of 11,286 frames (2.3%) are skipped, in 21 of the 40
rallies, and **all 265 are at the start of a rally**: the broadcast is still on a
close-up of the server when the serve is labelled and cuts to the main view 3 to
32 frames (up to about 1 s) later. Frames 16581-16591 (rally 6) and 81390-81400
(rally 22) are examples; the first is a close-up of the server's racket. No rally
loses main-view frames in the middle. For hit detection this means the serve is
often not visible in broadcast footage. My own fixed-camera footage won't have
cuts.

## Overlays (looked at)
`data/checks/players/`, `me` blue, `opp` red with track id and meters, everyone else grey:
- `match_players_frame14700.jpg` (set 1): Ginting near, Axelsen far, labelled
  correctly; the umpire, service judge, line judges, coaches and people by the
  boards are all grey.
- `match_players_frame16900.jpg` (set 1, rally 6): same.
- `match_players_frame17294.jpg` (set 1, rally 6): Axelsen in the air for a jump
  smash, still `opp` at (1.6, 14.5) m; the true position is just behind the far baseline.
- `match_players_frame81700.jpg`, `match_players_frame82048.jpg` (set 2, rally 22):
  ends changed, Axelsen near (`me`), Ginting far (`opp`); at 82048, the frame of
  the former duplicate box, there is one box and the same track id.

## Speed: GPU, batches and prefetching
Default now: YOLO on the Mac's GPU (`mps`, Apple Metal), 8 frames per `track()` call,
with the next chunks decoded and view-checked on a background thread while the GPU works.
`--device cpu` and `--batch-frames N` select the other rows below.

Measured on the dev machine (MacBook Air, M2, 8 GB, fanless), wall clock of the whole
`vision.players` tracking step (decode, view check, YOLO, labelling), each run alone on
the machine:

| Video | Device, frames per call | Wall clock | fps | vs CPU one frame per call |
|---|---|---|---|---|
| `match.mp4` 14201-14947 (747 frames, 1280x720, 30 fps) | CPU, 1 | 197.6 s | 3.8 | 1.0x |
| | CPU, 1, prefetch | 160.7 s | 4.6 | 1.2x |
| | CPU, 8 | 258.3 s | 2.9 | 0.8x |
| | GPU, 1 | 102.9 s | 7.3 | 1.9x |
| | GPU, 4 | 83.8 s | 8.9 | 2.4x |
| | GPU, 8, no prefetch | 92.4 s | 8.1 | 2.1x |
| | **GPU, 8, prefetch (default)** | **48.4 s** | **15.5** | **4.1x** |
| `stand_in/set1.mp4` 0-3599 (3600 frames, 1920x1080, 60 fps) | CPU, 1 | 1087.2 s | 3.3 | 1.0x |
| | CPU, 8 | 1056.3 s | 3.4 | 1.0x |
| | GPU, 4, no prefetch | 629.6 s | 5.7 | 1.7x |
| | GPU, 8, no prefetch | 574.4 s | 6.3 | 1.9x |
| | GPU, 8, warm-up, no prefetch | 414.8 s | 8.7 | 2.6x |
| | **GPU, 8, warm-up, prefetch (default)** | **358.4 s** | **10.0** | **3.0x** |

The first minute of the stand-in video went from 22 min (the earlier run noted in
`docs/results/stand_in.md`) to 6 min; a whole 7-minute set should take about 45 min
instead of 2 h 40 min. The aim was 3x over the CPU; the broadcast rally reaches 4.1x,
the 60 fps club video 3.0x.

**Same output.** Against the CPU one-frame-per-call run of the same code, every GPU and
batched run gives the same set of main-view frames, `me`/`opp` on 100% of frames (718 of
718 and 3590 of 3590), positions within 0.001 m and boxes within 0.1 px (float32 on the
GPU vs the CPU). The GPU run also reproduces the position file shipped with the first
version of this page to within 0.001 m. ByteTrack sees the batch frame by frame in order,
so the ids are the same.

**Why not 30 frames per call.** YOLO alone on 60 frames of `match.mp4` (frames
pre-decoded, after warm-up), GPU memory as Metal reports it:

| Frames per call | GPU fps | GPU memory | CPU fps |
|---|---|---|---|
| 1 | 11.9 | 1.2 GB | 5.0 |
| 4 | 18.8 | 1.2 GB | 4.5 |
| 8 | 17.9 (10.2 in a second run) | 2.3 GB | 3.1 |
| 15 | 5.8 | 4.5 GB | |
| 30 | 0.21 (282 s for 60 frames) | | |

Each frame at 1280 px needs about 150 MB of activations on the GPU. The GPU shares the
8 GB of unified memory with everything else, so 15 frames already spill, and 30 frames
need more than the machine has: Metal pages to disk and the run is 18x slower than the
CPU. On the CPU a batch brings nothing (the same work, more memory), so `--device cpu`
is best with `--batch-frames 1`. The 1920-wide stand-in video is detected at 1280 px
(`MAX_DETECTION_WIDTH_PX`): its GPU memory and YOLO fps per frame equal the 1280-wide
broadcast video's (1.18 GB vs 1.18 GB at 1 frame per call), so the cap is what keeps it
there; the remaining gap (10 vs 15.5 fps overall) is decoding 1080p60 and the view check
on the larger frames.

**Warm-up.** The first GPU call compiles Metal kernels and takes seconds. Ultralytics
drops the boxes of a batch whose post-processing runs past its time limit (2.0 s plus
0.05 s per frame): in the first stand-in run that batch was frames 96-103, which came
back with no players (5 of 3590 frames differed from the CPU). `track_players` now runs
the detector once on a black frame before the first chunk; after that the runs are
identical.

**Throttling.** The 60 fps run slows down as the fanless machine heats up: 11.5 fps over
the first 600 frames, 12.1 at 1200, 9.0 at 2400, 8.7 at 3600 (run without prefetch).
Whole-set numbers will be nearer the end of that curve than the start.

**Prefetch.** Without it, the GPU waits while the CPU decodes and view-checks the next
chunk: 146 s of the 415 s stand-in run. Reading two chunks ahead on a thread overlaps
the two, which is the last 2.6x to 3.0x step. It helps the CPU path too (3.8 to 4.6 fps
on the broadcast rally), since decoding and the view check release the interpreter lock.

## Limitations
- Accuracy tested on 3 rallies of one broadcast match, about 2,100 measured frames,
  plus the first minute of the club stand-in video (`docs/results/stand_in.md`).
- Speed depends on the machine's memory and cooling: the batch size was chosen for an
  8 GB M2 (`DETECTION_BATCH_FRAMES`); a bigger GPU share allows more frames per call, a
  smaller one fewer.
- Track ids restart at every run and after any gap longer than ByteTrack's
  30-frame buffer; `me`/`opp` don't depend on them.
- A track's side is fixed when first seen on court. Two players overlapping in the
  image (far player hidden behind the near one) never happened in these rallies; if
  ByteTrack swapped their ids, the swap would show as a player in the wrong half.
- Airborne players' positions are too far back (far player) or too close to the net
  (near player).
