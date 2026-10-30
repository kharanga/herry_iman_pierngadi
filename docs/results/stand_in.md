# Stand-in club footage: results

Until the Pi records, a club singles video from YouTube stands in for my own footage:
`data/recordings/singles_youtube.mp4` ("Singles in 90 degree heat", 1920 x 1080, VP9 at
60.00 fps, 34 min 06 s; local only, never committed). It is filmed with a phone from
behind one baseline, low (the camera comes out at 1.53 m) and just outside the right
sideline, so the view is diagonal, close to the rear-corner camera planned for the Pi.
No labels exist; every check here is by eye.

## How to run
```
V=data/recordings/stand_in/set1.mp4
.venv/bin/python -m vision.court $V --frame 600 --points "near_left=217,762" \
  "near_long_service_line x left_doubles_sideline=383,736" "near_long_service_line x centre_line=574,889" \
  "near_baseline x centre_line=356,960" "near_short_service_line x centre_line=1140,718" \
  "near_short_service_line x right_singles_sideline=1584,810" "near_short_service_line x right_doubles_sideline=1714,837"
.venv/bin/python -m vision.camera $V --net-posts 1004,496 1748,500
.venv/bin/python -m vision.players $V --start 0 --end 3599 --overlay-frame 600 --overlay-frame 1800 --overlay-frame 3000
.venv/bin/python -m vision.shuttle $V --start 0 --end 3599 --ensemble nonoverlap --overlay
.venv/bin/python -m vision.hits $V --start 0 --end 3599
.venv/bin/python -m vision.contact $V --range 0 3599 1 --hit-spot-check 20
.venv/bin/python -m vision.features $V --match-id stand_in_set1 --range 0 3599 1 --database <copy of data/scout.duckdb>
```
All outputs go to folders named `stand_in_set1` (the video id, `docs/vision/court.md`).
The shot rows were written to a copy of the database, not `data/scout.duckdb`, so the
stand-in doesn't change the numbers of the analytics results.

## Stretches
Cut points were found by measuring the image shift between frames one second apart
on the walls and ceiling (phase correlation), then per frame around each jump, and by
frame differences and 1-second contact sheets for the player change and the end switch
(`data/checks/stand_in/contact_sheet_30s.jpg`, one frame every 30 s). The two camera
moves and the player change are edit cuts (one frame to the next); the end switch is a
real walk-over.

| Set video | Original frames | Original time | Players (near = `me`) | Why it ends |
|---|---|---|---|---|
| `stand_in/set1.mp4` | 0 - 26234 | 0:00 - 7:17 | black vest and teal shorts near, grey shirt far | camera shifted 18 px (edit at frame 26235) |
| `stand_in/set2.mp4` | 26488 - 52847 | 7:21 - 14:40 | the same two, same ends | edit at 52849: other players |
| `stand_in/set3.mp4` | 52976 - 76859 | 14:42 - 21:20 | blue shirt near, pink shirt far | players change ends (21:21 - 21:26) |
| `stand_in/set4.mp4` | 77308 - 102104 | 21:28 - 28:21 | pink near, blue far | edit at 102106: camera at the other end of the hall |
| `stand_in/set5.mp4` | 102106 - 122749 | 28:22 - 34:06 | blue near, other player far; camera at the other end | end of video |

Cut with `ffmpeg -ss <start> -i ... -t <length> -c copy -an` (no re-encoding). VP9 here
has a keyframe every 5.13 s, so a copy-cut can only start on a keyframe: sets 2, 3 and 4
start at the first keyframe after the boundary and lose 4.2, 2.1 and 2.5 s (between
rallies in every case); set 5 starts exactly at its edit, which is a keyframe. A `-t` copy-cut can also end one
frame early (the last packet before the cut time is dropped), so sets 2 and 4 end one
frame before their boundary; the ends listed are the cut files' last frames. Checked:
each set's first frame is identical to the original frame listed (mean pixel difference
0.0), and set 1's last frame is the one before the camera shift.

## Calibration
From 7 named court points (the near baseline only enters at the left, the near-right
corner is below the frame); details in `docs/results/court.md`, "Calibration with a
hidden corner". Line check median 1.0 px; reference errors up to 8 px (the phone's wide
lens bends lines near the frame's edges). Camera: 0.65 m behind the near baseline,
x = 6.46 m (outside the right sideline), 1.53 m high, focal length 892 px (94°).
`data/checks/stand_in/stand_in_set1_court_overlay.jpg` and `..._camera_check.jpg`.

## 60 fps
`vision.hits.settings_for_video` scales the 30 fps / 1280 px tuning: at 60 fps and
1920 px the velocity window is 6 frames, the minimum hit gap 16, the rally gap 240, the
rest 40 frames within 12 px, and speeds per frame are 0.75 of the 30 fps values
(half the time per frame, 1.5 times the pixels). Contact and feature look-ups (2 and 3
frames at 30 fps) become 4 and 6. Results at 30 fps are byte-identical to before.
TrackNet itself runs on every 60 fps frame unchanged (its 8-frame window now covers
0.13 s instead of 0.27 s).

## Run on set 1, first minute (frames 0 - 3599)
CPU only: players 22 min, shuttle (`nonoverlap`) 33 min, hits 2 min, contacts about 1 min
(pose on 49 hits), features seconds.

**View check.** The fixed threshold (60% of the court-line samples found) skipped all
3600 frames: from this camera only about 50% of the in-frame samples are ever found
(near lines are much wider than the search window, far lines are thin and behind the
net). The view check now sizes each sample's search window to the painted line's width
and calls a frame the main view at 60% of the share found on the calibration frame
(0.515 here, so 0.309). Result: 3590 main view, 10 skipped (frames 2370-2379, a player
walking right in front of the camera). Broadcast results unchanged.

**Players.** `me` detected in 3590 of 3590 main-view frames with one track id, `opp` in
3586 with one id; both 100% in their own half. `me` y from 0.1 to 5.6 m (median 2.6),
`opp` from 7.2 to 14.6 m (median 10.0). The overlays
(`data/checks/stand_in/stand_in_set1_players_frame{600,1800,3000}.jpg`) show both boxed
correctly and the people along the walls left out.

**Shuttle.** A position in 3450 of 3600 frames (95.8%), but 1553 of those (43%) come from
InpaintNet filling gaps, against 9.9% on the broadcast. TrackNet loses the shuttle far
more often here (below).

**Hits.** 49 hits in 9 rallies; the detected rallies cover 2434 of the 3600 frames. 19 of
the 49 hits are followed, in the same rally, by another hit from the same side, which
can't happen in singles: each is a missed hit, an extra hit or a wrong hitter.

**Features.** 49 rows, 48 with a landing point. `landing_zone = not_over_net` on 25 rows,
the same symptom (consecutive hits by one side). Shot types predicted: net 21, serve 8,
drop 6, clear 4, lift 4, drive 3, smash 2. 8 serves in a minute with about 3 rallies is
another sign that between-rally activity is being read as play.

## Spot check of 20 hits
`python -m vision.contact ... --hit-spot-check 20` draws 20 random hits (numpy seed 1),
each as three frames 6 frames (30 fps; 12 at 60 fps) before, at and after the hit,
cropped around the shuttle, with the shuttle's path (yellow), both players' boxes (`me`
blue, `opp` red) and the estimated contact floor point (green):
`data/checks/stand_in/stand_in_set1_hit_spot_check_0_3599.jpg`. Looked at all 20:

| Frame | Detected | Hit? | Hitter | Note |
|---|---|---|---|---|
| 102 | opp | yes | right | |
| 295 | opp | yes | **wrong** | near player lunges at the net; the far player's box overlaps in the image |
| 387 | opp | yes | right | |
| 708 | opp | yes | right | |
| 839 | opp | yes | right | |
| 894 | opp | yes | unsure | both players at the net; 839 was opp too, so one of the two is wrong |
| 1044 | me | yes | right | near-side overhead |
| 1127 | opp | yes | right | far-side overhead |
| 1186 | opp | yes | right | |
| 1278 | me | unsure | | shuttle at the near player's feet, player bent low: a low lift or a pick-up |
| 1938 | me | yes | right | near-side overhead |
| 1998 | opp | yes | right | far-side lunge |
| 2217 | me | yes | unsure | both players at the net tape |
| 2288 | me | **no** | | the shuttle falls straight onto the floor (bounce) |
| 2500 | me | **no** | | a "shuttle" on the shutter door, no player near: between rallies |
| 2519 | opp | **no** | | shuttle sliding on the floor at the far player's feet |
| 3127 | me | **no** | | the shuttle tracker is on the near player's white shoe sole |
| 3165 | me | **no** | | the near player scoops the shuttle off the floor after a rally |
| 3247 | me | yes | right | a serve |
| 3544 | me | **no** | | false track on the shutter door |

**13 real hits, 6 not hits, 1 unsure.** Of the 13 real hits the hitter is right on 10,
wrong on 1 and unsure on 2, all three at the net. Hit points: where the contact marker is
in the crop (the low and mid-height hits), it is under or next to the shuttle at the
racket; on overheads the floor point is outside the crop (centred on the shuttle high
above), so they couldn't be judged from this sheet. All 6 non-hits are between rallies:
floor bounces, pick-ups and false shuttle tracks. During continuous play (rally 1, frames
78-1409) every sampled hit was real.

## What breaks from this angle
- **False shuttle tracks between rallies.** TrackNet, trained on broadcasts, locks onto a
  white shoe sole near the camera and onto features of the shutter door at the back of
  the hall, and follows shuttles lying on or tumbling over the floor. Those tracks make
  extra hits and 1-hit "rallies" (rallies 3, 5 and 9 have one hit each). On the broadcast
  the cutaways between rallies hid most of this. A rule that a rally needs both players to
  hit, or ignoring hits whose shuttle is near the floor and slow, would remove most.
- **Shuttle lost more often.** 43% of positions are InpaintNet fills (10% on the
  broadcast): the shuttle crosses the bright window and the white wall behind the far
  court, and near the camera it moves many pixels per frame and smears.
- **Far half squashed.** The whole far half, 6.7 m from the net to the far baseline, spans
  43 px of image height on the left sideline and 112 px on the right (net post bases at
  y 639 and 752, far corners at 596 and 640), while the near half fills the rest of the
  frame below the net and runs off its bottom edge: the
  camera is 1.5 m high and close to the near court. A pixel there is 5-15 cm along the court. Far-player positions and far landing points are the least
  accurate; far court lines are mostly not found by the line check.
- **Player overlap at the net.** From a low camera the far player at the net stands
  right behind the near player in the image; both boxes overlap and the nearest-box
  hitter rule fails (the 3 net cases above). The broadcast's high camera separates them.
- **Blur at 60 fps.** Less than at 30 fps per frame, but near the camera the shuttle still
  smears over 10-20 px on fast shots; TrackNet's 8-frame window covers half the time it was
  trained on.
- **Lens distortion.** Calibration residuals of 4-8 px near the frame's edges; a homography
  can't model the wide lens. It hasn't shown up as a problem in the tracks yet.

What works: calibration from any visible court points, player tracking (one id per
player, correct halves, bystanders out), hits during continuous play, contact points on
the hits that could be judged.

## Not done
- Stretches 2-5 and the rest of set 1 were not run (about 1 h of CPU per tracked minute
  for players and shuttle together); their calibrations would each need their own clicks
  (the camera moved).
- No labels, so no precision or recall; the spot check is the measure.
