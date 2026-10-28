# Shuttle tracking: results

## How to run
```
./vision/fetch_tracknetv3.sh
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.shuttle data/match.mp4 --shuttleset-rally $M 1 4 --overlay
.venv/bin/python -m vision.shuttle data/match.mp4 --shuttleset-rally $M 1 4 --ensemble nonoverlap
```
Output: `data/shuttle/match/shuttle_14201_14947.csv` (frames 14201-14947: rally 4 of
set 1, first hit 14231, last hit 14917, plus 30 frames each side). For the comparison
the two runs are kept in `data/shuttle/match/weighted/` and `data/shuttle/match/nonoverlap/`.

## Setup
- Video `data/match.mp4`, Axelsen vs Ginting, Thailand Open 2021 SF, 1280 x 720, 30 fps.
- Pretrained TrackNetV3 checkpoints (TrackNet + InpaintNet) from the official
  repository, run on the CPU (no GPU): `weighted` took 19 min 22 s for 747 frames
  (about 1.6 s per frame), `nonoverlap` 3 min 7 s (about 0.25 s per frame).
- Result: 727 of 747 frames have a position (667 of the 687 in-rally frames); InpaintNet
  filled 79 frames (47 in-rally). The 29 cutaway frames before the serve (a close-up)
  also get "positions", which are meaningless; frames outside the main camera view
  have to be removed with the view check from player tracking.

## Detection rate on a hand-labelled sample

**Sample.** 60 frames drawn at random (numpy seed 0) from the 687 in-rally frames
(14231-14917), before looking at any prediction. All are main-view frames.

**How they were labelled.** Finding a 4-6 px shuttle in a full 1280 x 720 frame by
eye was not reliable (first attempt, abandoned), so each frame was judged against the
prediction:
1. A 3x crop (120 x 120 px of video) centred on the predicted position, with the
   prediction marked by an open crosshair that doesn't cover the centre. Verdict
   "correct" when the shuttle is visibly within about 10 px (video pixels) of the
   centre. 10 px at 1280 x 720 corresponds to TrackNet's usual 4 px tolerance at
   512 x 288.
2. Every frame that wasn't clear in step 1 (21 frames: shuttle over white clothing,
   over the white net tape, motion-blurred, or predicted not visible) was checked on
   a strip of 2x crops of frames f-4, f-2, f, f+2, f+4, so the shuttle's path through
   the neighbouring frames shows where it must be in frame f.
3. For the 3 frames predicted "not visible" the full frame was checked as well.

The judge saw the model's marker, so this is a verification, not a blind
labelling; it can't find a shuttle the model and the judge both missed. Frames
where the shuttle couldn't be made out are counted as uncertain and left out of
the rates. Labels: `data/shuttle/match/hand_labels_14231_14917.csv` (frame,
prediction, verdict, note for each frame).

**Results** (`weighted`, the default):

| Verdict | Frames |
|---|---|
| Shuttle visible, predicted within ~10 px | 50 |
| Shuttle visible, predicted at the wrong place (15, 18 and 35 px off) | 3 |
| Shuttle visible, predicted not visible | 0 |
| Shuttle not visible (above the picture during a high lift), predicted not visible | 3 |
| Shuttle not visible, predicted visible | 0 |
| Uncertain (excluded) | 4 |

- **Detection rate** (shuttle visible and found within ~10 px): **50 of 53 = 94%**.
- **Not-visible frames:** 3 of 3 correct. No false detections in the sample.
- **Overall:** 53 of 56 judged frames correct (95%).
- **Uncertain:** 3 frames of the short serve that starts the rally (the shuttle
  crosses the net right over the white net tape, 14232-14236) and one motion-blurred frame (14284).

Where it fails: 2 of the 3 wrong positions are frames that InpaintNet filled
(14425, 14430: the shuttle was on the racket in front of the player's white shirt,
and InpaintNet's guess was 15 and 35 px away). Of the 6 inpainted frames in the
sample: 3 correct, 2 wrong, 1 uncertain. The other wrong one (14749) is a frame at the
moment of a hit, 18 px from the shuttle. Frames at a hit, with the shuttle in front
of a player, are the hard case; they are also the frames hit detection needs.

**`nonoverlap` on the same frames.** Its position is within 10 px of `weighted` in
57 of the 60 frames, and visibility agrees in all 60. The 3 that differ: at 14430
and 14749 `nonoverlap` is right (3 and 5 px) where `weighted` is wrong, and at 14592
it is 15 px off where `weighted` is right. So `nonoverlap` gets 51 of 53 = 96%:
on this sample the two are equal within one frame, and `nonoverlap` is 6 times
faster. The default stays `weighted` (the upstream default); the sample is too
small to prefer one.

## Overlay of one full rally
- `data/checks/shuttle/match_shuttle_14201_14947.mp4`: the whole rally plus padding
  (747 frames, 30 fps), the last 8 visible positions as a yellow trail and the
  current position as a red circle.
- `data/checks/shuttle/match_shuttle_14201_14947_strip.jpg`: 8 frames from it.

Looked at the strip and at the context strips used for labelling: the trail follows
the shuttle through clears, drops and net shots across the whole rally, ends at the
shuttle, and disappears while the shuttle is above the top of the picture (frame
14801). The circle drifts away from the shuttle in some frames where the shuttle is on
the racket (as in the sample's wrong frames).

## Limitations
- One rally of one broadcast match, 60 sampled frames; the margin of error at 94% on 53
  frames is several percentage points.
- CPU only: a whole match (about 140,000 frames) would take about 60 hours with
  `weighted` and 10 hours with `nonoverlap`. A GPU, a shorter frame range (rallies
  only), or running on the Pi's AI HAT are the options.
- Positions are pixels. The shuttle is in the air, so the floor calibration can't turn
  them into meters; landing points need the frame where the shuttle reaches the floor.
- Cutaway frames get meaningless detections (above).

## Training data and the production camera

TrackNetV3's README reports its accuracy on the test split of the Shuttlecock
Trajectory Dataset and its training instructions use that dataset, so the published
checkpoints were presumably trained on its training split (the README doesn't say so
in one sentence). The dataset page lists 26 training videos at 1280 x 720 and 30 fps:
23 professional matches from international tournament broadcasts (the same kind of
view as this video: one camera high behind one baseline, looking down the middle of
the court) and 3 amateur videos (the page doesn't say where their camera stood). The 3
test matches are broadcasts too. The README mentions one training match (match 16)
with a "camera angle discrepancy", so there is a little variety, but nearly all
training frames show the court from the centre behind one end, with the shuttle
against broadcast boards, lights and a dark ceiling.

The production camera will be different: at the rear-right corner behind the near
baseline, about 2.5-3 m high, so a diagonal view of the court, lower than a
broadcast camera. Likely effects:
- **Shuttle appearance and paths change.** Shots travel diagonally across the image
  instead of mostly up and down, the court is foreshortened differently, and the
  shuttle seen from lower down crosses more of the far wall and less of the floor.
  TrackNet uses the shape of the motion over 8 frames, so detection is likely to drop
  for unfamiliar paths, especially cross-court shots that run nearly towards or away
  from the camera.
- **Background.** A club hall has its own walls, windows, lights and other courts
  (other people's shuttles moving on neighbouring courts are false detections the
  broadcast never had). The median background helps with static clutter, not with
  other games.
- **InpaintNet** learned typical broadcast paths in image coordinates; its gap filling
  (already the weakest part above) will probably be less accurate on diagonal paths.
  Its rule that a gap is the shuttle leaving the picture only when it reappears near
  the top edge is a broadcast assumption; from a lower camera the shuttle can leave at
  the top on most high clears, and from the side as well.
- **Frame rate.** The model saw 30 fps. If the Pi records at 60 or 120 fps, 8 frames
  cover a quarter or an eighth of the time, and the shuttle moves less per frame than
  in training; feeding every 2nd or 4th frame keeps the 30 fps spacing.
- **What stays the same.** A fixed camera has no cutaways, and a median background
  over a whole match is clean.

How much the rate drops can only be measured on footage from that camera: the same
labelling method on 60 frames of a recorded game gives a number to compare with the
94% here. Whether to fine-tune on hand-labelled club footage (the TrackNetV3 repository
has the training code; the dataset's label format is one CSV per rally with frame,
visibility, x, y) is a separate decision to make after that measurement.
