# Two club games, tracked for labelling by hand

Two club singles videos from YouTube were tracked in full, so the user can label every
rally and hit by hand in the browser tool's blank mode. The rally bounds the user marks
there (start = serve contact, end = landing) are the club training and test data for
the in-play model (`docs/vision/in_play.md`), and the hits and shot types score the hit
detector and the classifier on club footage. Nothing from these videos is committed:
the footage, the check images and the labels stay under `data/`.

A third video, `IyJwoiSfYBw` (59 min at 60 fps, a club with players on the courts next
to it in view), was downloaded but dropped by the user before tracking. It stays in
`data/recordings/club/`.

## Videos

Downloaded with `yt-dlp` at 1280x720 (video only, no sign-in, no cookies) into
`data/recordings/club/<id>.mp4`:

| Video | Match | Length | fps | Frames | Codec | Camera |
|---|---|---|---|---|---|---|
| `E6SRRmvJ6dI` | Kelvin Chong (SJSU) v Leyang Ding (UC Davis), NorCal Collegiate | 22:05 | 24 | 31,800 | VP9 | one fixed view, low, from beside the court at a corner; players on the next courts in view |
| `q3pfhnvKPig` | Ella Lin v Jasmine Yeung, Open WS final, 2025 Bay Badminton Championships | 20:38 | 24 | 29,711 | AV1 | one fixed view, low, from beside the court at a corner; umpire, line judges and spectators in view |

Eight frames spread over each video (`data/checks/club/<id>_views.jpg`) show the camera
never moves, so each video has one calibration.

## Calibration

Both calibrations were clicked by the user with
`python -m vision.court <video> --frame N --pick-points`, because court corners are
outside the frame or hidden. The court was projected from the clicked reference points:

| Video | Frame | Points clicked | Overlay |
|---|---|---|---|
| `E6SRRmvJ6dI` | 9494 | 21 | `data/calibration/club_E6SRRmvJ6dI/court_overlay.jpg` |
| `q3pfhnvKPig` | 8870 | 16 | `data/calibration/club_q3pfhnvKPig/court_overlay.jpg` |

On both overlays the drawn court lies on the painted lines. Most points are within 1-4
px of the fit, and the largest is 6.4 px (a short-service-line point on Lin v Yeung). The
far half is compressed in both views, so far-court positions are less precise than
near-court ones.

**Near and far.** "Near" is the half closer to the camera. In both videos the players
switch ends between games. As in the rest of the pipeline, `me` is always the near
player and `opp` the far one, so in these videos `me` and `opp` mean **near and far**,
not two fixed people. In Lin v Yeung, the black-shirted player is near in the first
game and the red-shirted one after the change of ends.

## Tracking

Every frame of both videos, YOLO11s on the Mac's GPU (`mps`), 4 frames per call, in
back-to-back 3-minute stretches that are joined into one file per video. It ran alone on
the GPU with the swap guard on.

| Video | Tracking time | Frames with `me` | Frames with `opp` | Frames with both | "NMS time limit" warnings |
|---|---|---|---|---|---|
| `E6SRRmvJ6dI` | 39 min (2,341 s) | 29,895 (94.0%) | 28,454 (89.5%) | 27,756 (87.3%) | 1 |
| `q3pfhnvKPig` | 34 min (2,041 s) | 29,699 (100.0%) | 29,699 (100.0%) | 29,699 (100.0%) | 0 |

A first run on `E6SRRmvJ6dI` was stopped by the swap guard after 3 minutes, while two
full test runs from other sessions shared the machine. Nothing was written, and the
rerun started once nothing heavy was running. The single "NMS time limit" warning means
one batch may have lost a frame to the time limit under load.

**Are `me` and `opp` the two players?** Checked by eye on 12 frames spread over each
video: rallies, between-point gaps, the change of ends and the game break. The check
sheets are `data/checks/club/<id>_tracking_check.jpg`. For the whole video, the table
counts rows on the wrong half and rows off the court.

| Video | Frames right by eye | `me` on the far half | `opp` on the near half | Frames with the two swapped | Rows off the court |
|---|---|---|---|---|---|
| `E6SRRmvJ6dI` | 9 of 12 | 7.3% of `me` rows | 7.2% of `opp` rows | 1,223 (0.8 min) | 0.9% |
| `q3pfhnvKPig` | 11 of 12 (the 12th is the game break) | 0.4% | 0.9% | 0 | 14.3% |

- **Lin v Yeung:** "100% of frames with both players" is too good to be true. The
  players are right in rallies and at the change of ends, but when a player is out of
  view the tracker labels someone else on that side, standing in a fixed spot:
  - 15% of the `opp` rows (14.8%) are right of the court (x > 6.6 m), with a median
    position of (6.7, 12.8) m, outside the far right corner;
  - 11% of the `me` rows (11.0%) are left of the court (x < -0.5 m), with a median
    position of (-0.8, 4.9) m, beside the near left sideline;
  - in the game-break frame (16000), `me` is the red player standing off court.

  These fixed spots point to a line judge or someone courtside. They come and go
  through the whole video, which fits between-point gaps.
- **Chong v Ding:** the near and far players are right in rallies, with three kinds of
  error:
  - frame 14300: a person crossing close to the camera takes `me`, and the near player
    becomes `opp`;
  - frame 17000: in a break, someone off court becomes `opp`;
  - frame 19700: `me` and `opp` are swapped, which happens on 1,223 frames in all.

  Players on the neighbouring courts are not picked up as `me`/`opp` in the sample.

For labelling by hand this only means a wrong box now and then, because the user sees
the video. For any use of these tracks as features (the in-play model, the hit
detector) the swapped and stand-in rows have to be dealt with first. The user's rally
bounds will show how much of the stand-in time is between points.

## Detection

- `E6SRRmvJ6dI`: hits and rallies with the default (five-range) scorer, before the user
  chose blank labelling: 87 rallies, 697 detected hits plus 87 inferred serves, in
  `data/hits/club_E6SRRmvJ6dI/`. Blank mode doesn't read or show them; they're kept to
  score the detector against the labels later.
- `q3pfhnvKPig`: tracking only; the hits run was skipped (user's decision).

## Labelling

```
.venv/bin/python -m vision.label_ui data/recordings/club/E6SRRmvJ6dI.mp4 --match-id club_E6SRRmvJ6dI --blank --port 8766
.venv/bin/python -m vision.label_ui data/recordings/club/q3pfhnvKPig.mp4 --match-id club_q3pfhnvKPig --blank --port 8767
```

**Blank mode** (`docs/vision/label_ui.md`, "Blank mode"):

- nothing from the detector is shown; the tracking boxes stay;
- the video is cut into 60 s segments (23 for Ding, 21 for Lin), with `[` / `]` between
  them;
- `b` starts a rally at the current frame and `e` ends it.

The labels go to `data/labels/club_<id>.csv` and `data/labels/club_<id>_rallies.csv`,
with the usual columns and video frame numbers.

The rules for the rally bounds (`docs/vision/label.md`, "Rally bounds"):

- **start** = the frame where the serve's racket meets the shuttle;
- **end** = the frame where the shuttle lands (on the floor, or the net for a net error);
- a gap counts as not in play only when the rallies on both sides are labelled, which
  they are when the whole video is labelled;
- mark a rally's start before its hits, because a hit's rally is the user's rally that
  holds it.

## Two test games

Two more club videos are held out as the final test of the hit detector and the shot
classifier (`docs/results/club_test.md`, once run). The user labels them in blank mode.
They are never used for training, tuning or any choice before that test, and their labels
are not read until then.

| Video | Length | fps | Frames | Codec | Camera | Labelled and tracked |
|---|---|---|---|---|---|---|
| `UHKzdJqgFwk` | 35:15 | 60 | 126,887 | AV1 | one fixed view, low, behind the near court at a corner; other courts and players in view | first 10 min: frames 0-36,299 (10 min plus 5 s) |
| `xBClOgrP1Hs` | 10:37 | 59.95 | 38,183 | VP9 | one fixed view, low, beside the near court at a corner; other courts and players in view | the whole video |

Both were downloaded like the others: 1280x720, video only, no sign-in, no cookies.

**`xBClOgrP1Hs` is an irregular file.** Read in order it has 38,179 frames, against the
38,183 in its header. OpenCV's seeking also lands 1-2 frames past in-order reading later
in the video (+1 at frame 5,000, +2 by 20,000). So frame numbers on this 60 fps file
depend on how it is read. It is not used for scoring. The user labels its regular 30 fps
copy instead (`docs/vision/halve_fps.md`).

**Calibration.** The user clicked both with `--pick-points`; neither view shows the near
baseline's far end.

| Video | Frame | Points clicked | Error, max / RMS | Overlay |
|---|---|---|---|---|
| `UHKzdJqgFwk` | 8400 | 22 | 8.7 px / 3.6 px | `data/calibration/club_UHKzdJqgFwk/court_overlay.jpg` |
| `xBClOgrP1Hs` | 21212 | 18 | 6.2 px / 3.0 px | `data/calibration/club_xBClOgrP1Hs/court_overlay.jpg` |

On both overlays the drawn court lies on the painted lines. `UHKzdJqgFwk` has a wide lens,
so at the near-left the drawn sideline drifts off the painted one by a few pixels. That is
where its largest point errors are.

**Tracking.** This used the cleaned player choice, with the raw detections saved, on the
GPU (`mps`, 4 frames per call) in 3-minute stretches, alone, with the swap guard on. No
hit detection was run.
- `UHKzdJqgFwk`: 4 stretches over frames 0-36,299, 28 min plus about 11 min for a first
  stretch, paused once so another job could run. The merged file keeps the whole-video
  name (`positions_0_126886.csv`); its tracking record lists the stretches, and frames
  after 36,299 have no rows.
- `xBClOgrP1Hs`: the whole video, 40 min.
- No "NMS time limit" warnings, and swap use stayed flat.

| Video | Tracked frames | Frames with `me` / `opp` | Doubtful rows | Still stand-ins | Passers-by | me/opp swapped |
|---|---|---|---|---|---|---|
| `UHKzdJqgFwk` | 36,300 | 36,267 / 36,275 (99.9%) | 2,722 (3.8%) | 1 run, 175 player-frames | 4 runs, 79 player-frames | 0 |
| `xBClOgrP1Hs` | 38,183 | 38,065 / 37,844 (99.7% / 99.1%) | 198 (0.3%) | 0 | 1 run, 15 player-frames | 0 |

The counts come from `vision.track_check` without labels. Gaps inside rallies need rally
bounds, so they are not counted: there is no detector run, and the labels stay closed
until the test.

**Labelling.** The user labels in blank mode on ports 8769 and 8770 (commands below).
For `UHKzdJqgFwk` they stop after the tenth 60-second segment. The tracking boxes show
for the tracked part only.

## At 30 fps

The three 60 fps videos have 30 fps copies made by keeping every second frame
(`docs/vision/halve_fps.md`):
- `data/recordings/stand_in/set1_30.mp4`: 13,118 frames;
- `data/recordings/club/UHKzdJqgFwk_30.mp4`: 63,444 frames;
- `data/recordings/club/xBClOgrP1Hs_30.mp4`: 19,090 frames at 29.97 fps.

Spot checks show 30 fps frame k matching the original's frame 2k, in decode order.

The copies were tracked with the cleaned player choice, raw detections saved, alone on
the GPU with the swap guard on. There were no "NMS time limit" warnings.

| Video | Tracked | Time | Still stand-ins | Passers-by | me/opp swapped | Gaps of 0.5 s or more in labelled rallies | Labelled hits with the hitter tracked and moving |
|---|---|---|---|---|---|---|---|
| `set1_30` | all 13,118 frames | 14 min | 0 | 0 | 0 | 2 runs, 31 player-frames | 171 of 172 |
| `UHKzdJqgFwk_30` | frames 0-18,149 | 21 min | 1 run, 122 player-frames, outside rallies | 2 runs, 35 player-frames | 0 | 0 | 110 of 110 |
| `xBClOgrP1Hs_30` | all 19,090 frames | 20 min | 0 | 0 | 0 | not labelled yet | not labelled yet |

The user labels `xBClOgrP1Hs_30` directly:

```
.venv/bin/python -m vision.label_ui data/recordings/club/xBClOgrP1Hs_30.mp4 --match-id club_xBClOgrP1Hs_30 --blank --port 8770
```

## How to reproduce

```
for id in E6SRRmvJ6dI q3pfhnvKPig; do
  .venv/bin/yt-dlp --no-playlist -f "bv*[height<=720][ext=mp4]/bv*[height<=720]" --remux-video mp4 -o "data/recordings/club/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id"
done
.venv/bin/python -m vision.court data/recordings/club/E6SRRmvJ6dI.mp4 --frame 9494 --pick-points
.venv/bin/python -m vision.court data/recordings/club/q3pfhnvKPig.mp4 --frame 8870 --pick-points
for id in E6SRRmvJ6dI q3pfhnvKPig; do
  .venv/bin/python -m vision.players data/recordings/club/$id.mp4 --plan-stretches data/club_games/${id}_stretches.csv
  .venv/bin/python -m vision.players data/recordings/club/$id.mp4 --stretches data/club_games/${id}_stretches.csv --merge --batch-frames 4
done
.venv/bin/python -m vision.hits data/recordings/club/E6SRRmvJ6dI.mp4 --start 0 --end 31799
```

The two test games:

```
for id in UHKzdJqgFwk xBClOgrP1Hs; do
  .venv/bin/yt-dlp --no-playlist -f "bv*[height<=720][ext=mp4]/bv*[height<=720]" --remux-video mp4 -o "data/recordings/club/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id"
done
.venv/bin/python -m vision.court data/recordings/club/UHKzdJqgFwk.mp4 --frame 8400 --pick-points
.venv/bin/python -m vision.court data/recordings/club/xBClOgrP1Hs.mp4 --frame 21212 --pick-points
printf "start_frame,end_frame\n0,10799\n10800,21599\n21600,32399\n32400,36299\n" > data/club_games/UHKzdJqgFwk_stretches.csv
.venv/bin/python -m vision.players data/recordings/club/xBClOgrP1Hs.mp4 --plan-stretches data/club_games/xBClOgrP1Hs_stretches.csv
for id in UHKzdJqgFwk xBClOgrP1Hs; do
  .venv/bin/python -m vision.players data/recordings/club/$id.mp4 --stretches data/club_games/${id}_stretches.csv --merge --batch-frames 4
done
.venv/bin/python -m vision.track_check data/recordings/club/UHKzdJqgFwk.mp4 --end 126886 --sheets --tag test
.venv/bin/python -m vision.track_check data/recordings/club/xBClOgrP1Hs.mp4 --end 38182 --sheets --tag test
.venv/bin/python -m vision.label_ui data/recordings/club/UHKzdJqgFwk.mp4 --match-id club_UHKzdJqgFwk --blank --port 8769
.venv/bin/python -m vision.label_ui data/recordings/club/xBClOgrP1Hs.mp4 --match-id club_xBClOgrP1Hs --blank --port 8770
```
