# Hit detection: results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
for range in "14201 14947 1" "16551 17343 1" "81360 82116 2" "107560 108103 3" "114673 116641 3"; do
  set -- $range
  .venv/bin/python -m vision.players data/match.mp4 --start $1 --end $2
  .venv/bin/python -m vision.shuttle data/match.mp4 --start $1 --end $2 --ensemble nonoverlap
  .venv/bin/python -m vision.hits data/match.mp4 --start $1 --end $2 --evaluate $M $3
done
```
Outputs under `data/hits/match/` (`hits_*.csv`, `rallies_*.csv`). The tracking runs
take about 3 min per 750 frames each on the CPU; hit detection itself takes seconds
(plus the view check, about 12 ms per frame).

## Setup
- Video `data/match.mp4`, Axelsen vs Ginting (Thailand Open 2021 SF), 30 fps,
  ShuttleSet labels aligned at offset 0; `frame_num` is the labelled hit frame.
- Five frame ranges covering **7 rallies, 128 labelled hits**: set 1 rallies 4 and 6,
  set 2 rally 22, set 3 rally 14 (before the change of ends at 11 points) and set 3
  rallies 20-22 in one range (after the change of ends; the range includes the two
  breaks between them). Each range has 30 frames of padding around the labelled hits.
  The shuttle was tracked with the `nonoverlap` ensemble (6 times faster; equal on
  the sample measured for shuttle tracking).
- A detection counts as a labelled hit if it is within ±5 frames (one-to-one, closest
  first). Hitter accuracy is over matched hits whose label has the hitter's position
  (serves often don't). ShuttleSet's `player_location` is mapped through the court
  calibration to near/far, which handles the change of ends.
- **Thresholds were chosen on sets 1 and 2 (80 hits) and set 3 (48 hits) was held
  out**: a grid over the score threshold, the minimum speed after the hit, the reach
  limit and the minimum gap, picking the best F1 on sets 1-2. The default of each
  was kept where the grid was flat.

## Results

| Range | Set | Labelled | Detected | Matched | Precision | Recall | Hitter correct |
|---|---|---|---|---|---|---|---|
| 14201-14947 (rally 4) | 1 | 26 | 25 | 24 | 96.0% | 92.3% | 24/24 |
| 16551-17343 (rally 6) | 1 | 27 | 28 | 26 | 92.9% | 96.3% | 26/26 |
| 81360-82116 (rally 22) | 2 | 27 | 22 | 22 | 100% | 81.5% | 19/22 |
| 107560-108103 (rally 14) | 3 | 18 | 18 | 16 | 88.9% | 88.9% | 15/16 |
| 114673-116641 (rallies 20-22) | 3 | 30 | 33 | 26 | 78.8% | 86.7% | 23/26 |
| **Tuning (sets 1-2)** | | 80 | 75 | 72 | **96.0%** | **90.0%** | 69/72 = 95.8% |
| **Held out (set 3)** | | 48 | 51 | 42 | **82.4%** | **87.5%** | 38/42 = 90.5% |
| **All** | | 128 | 126 | 114 | **90.5%** | **89.1%** | 107/114 = 93.9% |

Frame offset of matched detections (detected minus labelled): median −1 in every
range, mean −0.9 to −1.4, 96% within ±2. The detector fires one frame before the
labelled contact because the score window spans the hit; the ±5 tolerance is not
what makes the matches.

At a stricter ±2-frame tolerance (measured by the validator): all 86.5% / 85.2%,
sets 1-2 92.0% / 86.2%, set 3 78.4% / 83.3% (precision / recall).

Before tuning (score ≥ 0.5, no speed test, reach ≤ 0.75), the same ranges gave
precision 73.9% and recall 90.6%: the speed test removed most false positives
without costing recall. The shipped thresholds are near, not at, the sets 1-2
optimum: on that grid a minimum gap of 10 frames instead of 8 would have scored
F1 94.1 against the shipped 92.9. They were not retuned, to keep the held-out
comparison honest and because a 10-frame gap loses more of the fast net exchanges.

**What the sample bears.** 128 hits from 7 rallies of one match, one camera, two
players. At these rates the 95% confidence interval is about ±5 percentage points
on all 128 and ±10 on the held-out 48, and the rallies were chosen for length (to
get many hits per tracking run), so they are not a random sample of shots.
Numbers from another match, or from the club camera, can differ by more than that.

## Rally segmentation

| Range | Labelled rallies (first hit - last hit) | Detected rallies (first hit - end) |
|---|---|---|
| 14201-14947 | 14231-14917 | 14256-14947 (range end, shuttle still flying) |
| 16551-17343 | 16581-17313 | 16597-17341 (last tracked frame) |
| 81360-82116 | 81390-82086 | 81406-82116 (range end) |
| 107560-108103 | 107590-108073 | 107626-108103 (range end) |
| 114673-116641 | 114703-115082, 115589-115795, 116493-116611 | 114694-115121 (rest), **115152-115162 (1 hit)**, 115605-115872 (rest), 116525-116641 (range end) |

All 7 labelled rallies are found, none is split and none merged across a break.
One extra 1-hit "rally" is the near player picking the shuttle up 30 frames after it
came to rest; the rest rule separates it from the real rally (whose end is then the
correct frame, 115121), and the next task can drop rallies with fewer than 2 hits.
Starts are 16-36 frames after the labelled serve because the serve is not visible
(below). Ends: where the shuttle comes to rest on the floor when that is inside the
range; otherwise the last tracked frame (the 30-frame padding is shorter than the
last shot's flight in 3 of the 5 ranges).

## Error analysis

12 false positives and 14 misses out of 128. The causes are assigned by
`error_breakdown` and printed by `--evaluate` (so they are re-derived on every run);
the per-frame lines are summarised here:

| Kind | Cause (from `error_breakdown`) | Frames |
|---|---|---|
| miss | serve, frame not in the main view (broadcast close-up) | 4 |
| miss | serve, in the main view but no candidate passes the score or speed test (shuttle in the hand, or only 3 frames after the cut) | 2 |
| miss | score below 0.4: shot right after the cut (14232), a drop (81982), a far-court smash where a descending lob and the smash both move *down* in the image (115082) | 3 |
| miss | speed after the hit below 8 px/frame: a net return (81488), a flat drive (116516), and a smash where the tracker stuck on the player (81624) | 3 |
| miss | reach above 0.75: a lunging net return, racket far outside the box (108043) | 1 |
| miss | suppressed by a stronger candidate within 8 frames: a drop and a net kill 6 frames apart (81705) | 1 |
| false positive | after the last hit of a rally: floor bounce (2), net cord (1), pickup (1) | 4 |
| false positive | before the first labelled hit: filled shuttle track while the server stands still (114694) | 1 |
| false positive | within a rally on an InpaintNet-filled frame (the filled path has a kink) | 4 |
| false positive | within a rally, no labelled hit nearby (scores 0.40-0.43) | 3 |

So **6 of the 7 serves are missed**, and **6 of the 12 false positives are on
InpaintNet-filled frames** (the 4 within rallies, plus 114694 and one of the
post-rally ones, 115856); 13 of the 114 matched hits are on filled frames too (the
shuttle hidden by the hitter's body), so filled frames can't simply be excluded.

Wrong hitter (7 of 114): all at the net, shuttle `y` between 305 and 375 px, where
the near player's box (in front) and the far player's box overlap in the image and
both reaches are small (0.0-0.6). A rule that picks the player the shuttle approached
over the 8 frames before the hit fixed 4 of these and broke 4 others (the far
player's box is small and the shuttle crosses it after a far-side hit), so the
simpler nearest-box rule is kept. Hits strictly alternate in singles; enforcing that
along a rally could fix net cases once recall is high enough, and is left to the
shot-features task.

How the known traps show up:
- **Close-up at the serve** (first 3-32 frames of most broadcast rallies): 4 of the 7
  serves are in a close-up, and the two served in the main view are missed as well
  (one 3 frames after the cut, before the velocity window is defined; one with the
  shuttle still in the hand, where the track is InpaintNet filler). Rallies therefore
  start at shot 2 or 3. A fixed club camera has no cuts, but the serve itself stays a
  weak case: the shuttle starts from the hand at speed 0.
- **Jumps**: no effect. Reach uses the box, not the foot position, and a jumping
  player's box still contains the racket.
- **Shuttle leaving the top of the picture** (high lifts): the shuttle is invisible
  for 10-20 frames; hits aren't affected (the next hit comes well after it
  reappears), but the rally end rule had to search past such gaps, otherwise a lift
  right after a hit ended the rally.
- **InpaintNet fills**: 6 of the 12 false positives and 13 of the 114 matched hits
  are on filled frames (above).

## Visual check
`data/checks/hits/match_hits_14201_14947.jpg`: a crop around the shuttle at each of
the 25 detected hits of set 1 rally 4, with both players' boxes (`me` blue, `opp` red)
and the hitter and the matched label in the caption. In all 24 matched frames the
shuttle is at the hitter's racket; the one unmatched detection is the near player at
the net 8 frames after a labelled net shot.

## Own footage
The criterion "the same metrics on one hand-labelled game of mine" is pending: no
recording from the club camera exists yet. The same command works on it once the
range is tracked; the labels would be hit frames marked by hand (a 60-frame-per-game
check as for shuttle tracking is not enough here, every hit of the game must be
labelled). Expect the serve to be visible there, the far-left corner weaker, and
more net-side hitter errors from the diagonal view (the players overlap more).
