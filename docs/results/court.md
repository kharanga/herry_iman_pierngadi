# Court calibration: results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.court data/match.mp4 --frame 12300 --shuttleset-match $M
```
Writes `data/calibration/match/court.json` and `data/calibration/match/court_overlay.jpg`.
On my own footage the same command without `--shuttleset-match` opens the click window,
or takes `--corners X,Y X,Y X,Y X,Y` (near_left near_right far_right far_left).

## Input
Video `data/match.mp4` (Axelsen vs Ginting, Thailand Open 2021 SF), 1280 x 720, 30 fps.
Corners from ShuttleSet `homography.csv` (row 27), the outer doubles corners:

| Corner | Pixel (x, y) | Court (m) |
|---|---|---|
| near_left (`downleft`) | 289.4, 659.6 | 0, 0 |
| near_right (`downright`) | 987.2, 657.4 | 6.1, 0 |
| far_right (`upright`) | 864.2, 301.6 | 6.1, 13.4 |
| far_left (`upleft`) | 412.6, 303.0 | 0, 13.4 |

## Corner reprojection error
Max 1.6e-13 px over the 4 corners. With exactly 4 points the homography is exact,
so this only confirms the solver; the line check below is the real test.

## Agreement with ShuttleSet's homography
ShuttleSet's `homography_matrix` does **not** map to meters. Applying it to the 4
corner pixels gives (27.4, 810), (327.6, 810), (327.6, 150), (27.4, 150): a court
diagram 300.2 x 660 units with its origin top-left, x to the right, y pointing
towards the camera (near baseline at y = 810, far baseline at y = 150), about
49.2 units per meter (49.21 across, 49.25 along). Their README describes it as
"real-world" coordinates, and says `p = H^-1 p'` takes them back to the camera.

To compare, I converted court meters to that diagram
(`x' = 27.4 + x * 300.2 / 6.1`, `y' = 810 - y * 660 / 13.4`), mapped them back to
pixels with ShuttleSet's inverse, and compared with `court_to_pixels` from my
homography on a 13 x 27 grid covering the whole court (0.5 m steps):

| Measure | Pixels |
|---|---|
| Max difference | 8.6e-5 |
| Mean difference | 1.9e-5 |

Equivalently, ShuttleSet's matrix with the diagram -> meter conversion folded in
matches mine entry by entry to 5-6 significant digits. Both are fitted to the same
4 corners, so this agreement is expected; it confirms the corner order, the axis
directions (my y = 13.4 is their y = 150) and the corner semantics are the same.

## Overlay: do the reprojected lines sit on the painted lines?
Overlays (red = reprojected lines, 1 px):
- `data/calibration/match/court_overlay.jpg` (frame 12300, set 1, with the corners marked)
- `data/checks/court/court_overlay_frame12300.jpg` (set 1)
- `data/checks/court/court_overlay_frame95659.jpg` (set 3)
- `data/checks/court/court_overlay_frame126000.jpg` (set 3, late)

What they show: all 12 floor lines (baselines, doubles and singles sidelines,
long and short service lines, centre lines) sit on the white painted lines across
the whole court, near and far, in all three frames. The camera does not move
between sets. In a 4x zoom of the near-left corner, the red line runs along
the outer edge of the white baseline and doubles sideline, as it should: the
official 6.1 x 13.4 m is measured to the outer edges, and the corners are clicked
there. The red `net` line lies on the floor between the net post bases; the white
net tape is drawn higher because it is 1.55 m above the floor, so the net line is
not expected to match the tape (a homography only covers the floor plane).

Measured: for each floor line, 40 points between 5% and 95% of its length, and at
each point the brightest pixel within ±8 px across the line (skipped where the
profile has no clear white peak, for example under a player).
`line_offsets_to_bright_pixels` does this.

| Frame | Samples | Median abs offset | Mean abs offset | 95th percentile |
|---|---|---|---|---|
| 12300 (set 1) | 475 | 2.0 px | 2.4 px | 6.0 px |
| 95659 (set 3) | 473 | 2.0 px | 2.3 px | 5.0 px |
| 126000 (set 3) | 473 | 2.5 px | 2.6 px | 6.0 px |

Per line on frame 12300 (signed median, positive = painted line lies to the
right of the line's direction, which for a left-to-right line means lower in the image):

| Line | Median | Mean abs |
|---|---|---|
| near baseline | -5.0 | 5.0 |
| far baseline | 0.0 | 1.5 |
| left doubles sideline | +2.8 | 2.8 |
| right doubles sideline | -3.0 | 3.4 |
| left singles sideline | +3.0 | 3.1 |
| right singles sideline | -3.8 | 3.7 |
| near doubles long service line | -4.0 | 3.8 |
| far doubles long service line | 0.0 | 0.8 |
| near short service line | -2.0 | 1.6 |
| far short service line | -2.0 | 2.0 |
| near centre line | +1.0 | 1.0 |
| far centre line | 0.0 | 0.0 |

This compares the drawn line with the *centre* of the painted line, so a
perfect calibration still shows about half a line width. The sidelines
show it clearly: left lines are offset +3 and right lines −3, so the paint centre
is inward of the drawn line on both sides. The near baseline and near long service
line (−4 to −5 px, the paint is above the drawn line) are the largest offsets.
That is the side nearest the camera, where lines are widest in pixels. The far
lines are 0 to 2 px. The worst sample is 8 px, which is the search radius, so it
is a stray bright pixel and not a line. Overall the reprojected lines are within
a few pixels of the painted ones. That meets the acceptance criterion; for
meters, 5 px at the near baseline is about 0.1 m along the court.

## Things to know for later tasks
- The broadcast video cuts to other cameras: close-ups (e.g. frame 58704) and
  replays from a low side angle (e.g. frame 80214). There the homography is
  meaningless; the line check gives few, random samples (89 and 123 samples,
  medians 5-6 px) and the overlay floats over players and boards. Player and
  shuttle tracking must skip frames that aren't the main view; the
  share of found line samples is a cheap test for that.
- My own Pi footage has one fixed camera, so this doesn't apply there. The frame used
  for clicking needs 4 or more known court points, not necessarily the corners (see
  "Calibration with a hidden corner" below).

## Calibration with a hidden corner

### Broadcast video without the near-left corner
To check that a calibration from other known points is as good, `data/match.mp4` was
calibrated again without the near-left corner: the other 3 corners plus 2 crossings on
the left doubles sideline, read off an 8x zoom of frame 12300 at the sideline's outer
edge (the corner convention) and the middle of the line across:

```
.venv/bin/python -m vision.court data/match.mp4 --frame 12300 --output-dir <temp> --points \
  "near_right=987.2,657.4" "far_right=864.2,301.6" "far_left=412.6,303.0" \
  "near_long_service_line x left_doubles_sideline=301,629.5" \
  "near_short_service_line x left_doubles_sideline=347.5,495"
```

| | 4 corners | 3 corners + 2 crossings |
|---|---|---|
| Line check, frame 12300 (median / mean / p95, px) | 2.0 / 2.41 / 6.0 | 2.0 / 2.24 / 6.3 |
| frame 95659 | 2.0 / 2.33 / 5.0 | 2.0 / 2.16 / 5.0 |
| frame 126000 | 2.5 / 2.56 / 6.0 | 2.0 / 2.36 / 6.0 |
| All three frames | median 2.0, mean 2.43 | median 2.0, mean 2.25 |
| Reference point errors | 0 (exact with 4) | 0.95 to 1.49 px |
| Hidden near-left corner, where the homography puts it | (289.4, 659.6) clicked | (289.6, 659.2), 0.50 px away |

The median line error is the same, within the criterion's 2 px; the mean is slightly
lower, because 5 points average out part of each click's error.

### Stand-in club video
`data/recordings/stand_in/set1.mp4` (1920 x 1080, phone wide lens, camera low behind
the near baseline and outside the right sideline): the near baseline enters the frame
only at the left; the near-right corner is below the frame and the far corners are
behind the net and players. Calibrated from 7 points on frame 600, identified from a
mask of the white lines (the centre line ends on the short service line and on the
baseline; the short service line ends at the outer of the two right sidelines) and
read off 5x zooms:

| Point | Pixel | Error (px) |
|---|---|---|
| `near_left` | 217, 762 | 6.3 |
| `near_long_service_line x left_doubles_sideline` | 383, 736 | 5.0 |
| `near_long_service_line x centre_line` | 574, 889 | 6.3 |
| `near_baseline x centre_line` | 356, 960 | 8.0 |
| `near_short_service_line x centre_line` | 1140, 718 | 0.2 |
| `near_short_service_line x right_singles_sideline` | 1584, 810 | 4.0 |
| `near_short_service_line x right_doubles_sideline` | 1714, 837 | 4.9 |

RMS 5.4 px. Leaving any one point out, the other 6 predict it within 1 to 26 px (the
worst is `near_baseline x centre_line`, closest to the frame's edge), so every point is
identified correctly; the residuals are larger than on the broadcast because the phone's
wide lens bends straight lines near the frame's edges, which a homography can't follow.
The left net post base, not used in the fit, comes out at (1000, 639), where the post
stands. The corners that weren't clicked are projected (`corner_pixels_source`), the
near-right one at (1202, 2325), far below the frame.

Line check on frames 600, 6000 and 20000: 164 to 174 samples found of the 367 inside the
frame, median 1.0 px, mean 1.5 to 1.7 px, 95th percentile 4.8 to 8.0 px. The overlay
`data/calibration/set1/court_overlay.jpg` and the camera check
`data/calibration/set1/camera_check.jpg` (looked at) show the reprojected lines on the
painted ones in both halves, including the far half behind the net, and the net top
line on the net tape.

The camera (`python -m vision.camera ... --net-posts 1004,496 1748,500`, post tops read
off 6x zooms) puts the lens at x 6.46, y −0.65, z 1.53 m: 0.65 m behind the near
baseline, just outside the right sideline, at chest height, as filmed; focal length
892 px (a 94° horizontal field of view, a phone's wide lens); reference errors 3.3 to
8.5 px.

Only about 45% of the in-frame line samples are found here (98-99% on the broadcast):
the near lines are wider than the 8 px search window and the far ones are thin and
partly behind the net. That matters for the view check (below 0.6 means "not the main
view") and is taken up with the stand-in footage.
