# Racket contact point: results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
R="--range 14201 14947 1 --range 16551 17343 1 --range 81360 82116 2 --range 107560 108103 3 --range 114673 116641 3"
.venv/bin/python -m vision.camera data/match.mp4 --net-posts 364.5,308.5 910.0,306.5
.venv/bin/python -m vision.contact data/match.mp4 $R --compare $M
.venv/bin/python -m vision.features data/match.mp4 --match-id match_tracked $R
.venv/bin/python -m shots.predict
.venv/bin/python -m vision.features data/match.mp4 --match-id match_tracked --spot-check --compare $M
```
The five ranges are the hit-detection ranges (`docs/results/hits.md`); their player,
shuttle and hit files must exist. The pose model runs once per hit (about 1 s each on
the CPU, 2 minutes for the 126 hits) and is cached under `data/contact/match/`.

## Camera
The net-post tops were read off a 4x zoom of frame 12300 (the court calibration's
frame): left (364.5, 308.5), right (910.0, 306.5). Each sits vertically above its post's
base as the floor calibration places it ((364.2, 443.1) and (912.5, 441.4)).

| | Value |
|---|---|
| Focal length | 2661 px (horizontal field of view 27°) |
| Camera position | x 3.08, y −21.9, z 8.46 m: 21.9 m behind the near baseline, 8.5 m up, on the centre line |
| Reference point errors | corners 0.9, 1.3, 2.4, 2.0 px; post tops 2.7, 3.5 px (RMS 2.3) |
| Court lines (473 samples, 12 floor lines, as in `docs/results/court.md`) | median 1.0 px, 95th percentile 7.0 px |

The floor calibration alone gives median 2.0 px on the same samples; the camera
model's floor is slightly better because it spreads the corner error over 6 points
instead of fitting 4 exactly. `data/calibration/match/camera_check.jpg` (looked at): the
reprojected floor lines lie on the painted lines, and the reprojected net top line
lies on the white net tape from post to post, with both posts on the blue posts. The
camera is fitted at the post tops to within 3.5 px, which at the net is about 4 cm.

## Contact estimate vs feet
110 detected hits match a labelled ShuttleSet hit with a hit point (±5 frames). The
labelled hit pixel is mapped to meters with the court calibration, and both estimates
are compared with it.

**Tuning on sets 1-2 (70 hits).** Median / mean distance to the labelled hit point (m):

| Radius (m) | toward the net | nearer the camera | farther from the camera |
|---|---|---|---|
| 0.6 | 0.55 / 1.04 | 0.55 / 1.08 | 0.55 / 1.03 |
| 0.8 | 0.60 / 1.00 | 0.64 / 1.18 | **0.51** / 0.97 |
| 1.0 | 0.73 / 1.00 | 0.82 / 1.35 | **0.51 / 0.92** |
| 1.2 | 0.78 / 1.10 | 1.07 / 1.58 | 0.68 / 0.98 |
| 1.4 | 0.98 / 1.22 | 1.33 / 1.81 | 0.84 / 1.10 |
| 1.6 | 1.20 / 1.34 | 1.58 / 2.06 | 1.01 / 1.21 |
| 1.8 | 1.47 / 1.50 | 1.80 / 2.29 | 1.31 / 1.36 |

Shipped: radius 1.0 m, the intersection farther from the camera (tied with 0.8 m on
the median, lower mean). The feet on the same hits: median 1.14, mean 1.44.

**Results** (shipped settings):

| | Hits | Contact median | Contact mean | Feet median | Feet mean | Contact closer |
|---|---|---|---|---|---|---|
| Sets 1-2 (tuning) | 70 | 0.51 m | 0.92 m | 1.14 m | 1.44 m | 57 of 70 |
| **Set 3 (held out)** | 40 | **0.71 m** | 1.11 m | **1.19 m** | 1.60 m | 31 of 40 |
| All | 110 | 0.58 m | 0.99 m | 1.14 m | 1.50 m | 88 of 110 |

On the held-out set the contact estimate is 0.48 m closer to the labelled hit point
than the feet at the median and 0.49 m at the mean, and closer on 31 of 40 hits: it
clearly beats the feet, so it is shipped. (Radius 0.8 m would have given 0.62 m on
set 3; it wasn't chosen because it didn't win on sets 1-2.)

Quartiles of the distance on set 3: contact 0.28 / 0.71 / 1.25 m, feet 0.77 / 1.19 /
1.81 m.

By hitter (all 110): far player (`opp`) contact median 0.49 m vs feet 1.01 m; near player
(`me`) 0.76 m vs 1.56 m. The near player is seen from behind, so the line of sight
through the shuttle runs almost along the court and the depth along it (which one
camera can't see) moves the floor point mostly along `y`; for the far player the same
depth error moves it less.

**Fallbacks:** none. Every one of the 126 hits had a visible shuttle within 2 frames
and a pose matching the hitter's box with a confident hitting-side shoulder. 74 hits
took a sphere intersection, 52 the closest point to the sphere (the ray misses the
1.0 m sphere: the shuttle is further from the shoulder than the radius in the image,
often because the detected hit frame is a frame off or the shoulder keypoint is
off). Median error by method on set 3: sphere 0.65 m (26 hits), closest point 0.77 m
(14).

The worst set-3 cases (5 to 8 m) are hits that hit detection gave to the wrong player
(114901, 116525, 107722 in `docs/results/hits.md`): the contact is then estimated
around the wrong player's shoulder; the feet are equally wrong there.

## Shot types before and after
`vision.features` now writes the contact floor point as `hit_x/hit_y` (and the next
hit's contact as `landing_x/y`, which is ShuttleSet's convention); `player_x/y` keep
the feet. Classifier accuracy on the matched tracked rows (`type_pred` vs ShuttleSet's
`type_group`), before = feet (the earlier rows, rebuilt with `--hit-point feet` into a
copy of the database), after = contact:

| | Before (feet) | After (contact) |
|---|---|---|
| Sets 1-2 | 48 of 69 = 69.6% | 50 of 69 = 72.5% |
| **Set 3 (held out)** | 25 of 41 = 61.0% | **24 of 41 = 58.5%** |
| All | 73 of 110 = 66.4% | 74 of 110 = 67.3% |

| | Before (feet) | After (contact) |
|---|---|---|
| Hit point, median distance to the label | 1.14 m | 0.58 m |
| Landing point, median distance to the label | 1.23 m | 0.72 m |
| `hit_zone` agreement | 59.6% | 67.5% |
| `landing_zone` agreement | 52.6% | 57.9% |
| `is_cross` agreement | 88.2% | 97.3% |

Per group (all 110, before -> after): clear 7 -> 7 of 7, smash 12 -> 12 of 14, net 29 ->
28 of 36, lift 12 -> 15 of 19, drop 8 -> 8 of 15, drive 5 -> 4 of 18, serve 0 -> 0 of 1.

**The geometry improved clearly; the shot types barely did** (+1 shot overall, −1 on
the held-out set, both within the ±9 points of 110 shots). Where the remaining loss
is, measured by giving the classifier the tracked rows with one ingredient replaced by
ShuttleSet's labelled value (same 110 shots):

| Rows given to the classifier | Correct |
|---|---|
| tracked rows as written (contact) | 74 |
| with labelled hit frames (timing) | 75 |
| with labelled receiver positions | 77 |
| with labelled hit points | 79 |
| with labelled landing points | **85** |
| ShuttleSet's own labelled rows | 101 |

The landing point matters most: replacing it alone gains 11 shots, the hit point 5.
The landing is the next hit's contact, so it inherits that hit's 0.6-0.7 m error and
also every hit-detection error (a missed or extra hit makes a landing on the wrong
player). The 101 of 110 on labelled rows is optimistic: this match may be in the
classifier's training split (`docs/results/classifier.md`). The tracked pipeline's
shot types will not get close to the labelled accuracy until the landing point is
better, which is a hit-detection and contact-depth problem, not a classifier one.

## Spot check
`data/checks/contact/match_contact_spot_check.jpg`, drawn by `--compare`: 10 random
matched hits (numpy seed 1) with the shuttle pixel (yellow), the contact floor point
(green, with a line up to the estimated contact in the air), the hitter's feet (blue)
and the labelled hit point (red), captioned with both distances.

What it shows (looked at): on far-side hits (14347, 14675, 16708, 81544) the green point
is on or next to the red one, while the feet are up to 1.2 m away (on the overhead at
81544 feet, contact and label coincide); on near-side reaches
(16894, 114811, 115743) the green point lies between the feet and the label, short of
the label along the line of sight: the depth error described above. 17297 (a far-side
clear from behind the baseline) is off by 1.8 m, about as much as the feet: the ray
misses the sphere and the closest point is at the wrong depth. The contact line goes
up from the green point to the shuttle pixel in every tile, so the camera and the ray
are consistent.

`data/checks/features/match_tracked_spot_check_{0..3}.jpg` (regenerated) now show the
contact floor point as the hit marker: under the racket on reaching shots instead of
at the feet.

## Limitations and the club camera
- One broadcast match, 110 matched hits (40 held out): the set-3 median carries roughly
  ±0.15 m of sampling error.
- The radius and the intersection rule absorb systematic errors (detected hit frame one
  frame early, the shoulder keypoint seen from behind); on another camera they should be
  re-checked against labelled hits.
- From the rear-right corner the lines of sight cross the court diagonally, so the depth
  ambiguity is diagonal too and the near player is no longer seen from straight behind;
  the near-side bias above should shrink, and the far-left corner will be the weak spot.
  The net-post tops must be visible on the calibration frame (both, including the far
  one); `python -m vision.camera <video>` opens the click window for them.
