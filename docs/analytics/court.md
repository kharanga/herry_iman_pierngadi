# `analytics/court.py`

Court geometry shared by every shot source: pixel -> court-meter conversion,
normalizing a shot so the hitter is on the near half, zones and straight vs cross-court.
Pure functions (numbers in, numbers out), so the vision track can call them for its
own shots and they can later be ported to C++.

## Court frame

All positions are in **court meters**: origin at the near-left outer corner
(seen from behind the near baseline), `x` across the court (0 to 6.1), `y` along it
(0 to 13.4). The net is at `y = 6.7`, the centre line at `x = 3.05`.
After normalization the hitter is always on the near half (`y < 6.7`) and the
receiver on the far half.

Normalizing turns the whole shot half a turn around the court centre
(`x -> 6.1 - x`, `y -> 13.4 - y`). That is a rotation, not a mirror, so a player's
left stays their left.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `COURT_WIDTH_M` | 6.1 | doubles width, outer edges |
| `COURT_LENGTH_M` | 13.4 | length, outer edges |
| `NET_Y_M` | 6.7 | net position |
| `CENTRE_X_M` | 3.05 | centre line |
| `DEPTH_BAND_M` | 2.233 | a half is split into 3 equal depth bands of this length, measured from the net |
| `CROSS_COURT_MIN_LATERAL_M` | 1.0 | minimum sideways travel for a cross-court shot |
| `DEPTH_NAMES` | `("front", "mid", "rear")` | |
| `ZONE_NAMES` | `front_forehand`, `front_backhand`, `mid_forehand`, `mid_backhand`, `rear_forehand`, `rear_backhand` | |
| `NOT_OVER_NET_ZONE` | `"not_over_net"` | landing zone of a shot that stayed on the hitter's half |

`Point` is a `tuple[float, float]` in meters. Any function that takes a `Point | None`
treats `None` or a NaN coordinate as missing.

```python
>>> from analytics.court import *
>>> round(DEPTH_BAND_M, 3), ZONE_NAMES[:2]
(2.233, ('front_forehand', 'front_backhand'))

```

## `apply_homography(homography, points)`

Applies a 3x3 homography to points.

- `homography` (3x3 array-like), `points` (array-like of `(x, y)`, any shape that reshapes to `(-1, 2)`).
- Returns `np.ndarray` of shape `(n, 2)`. NaN in, NaN out.
- Errors: `ValueError` if `points` does not reshape to pairs.

```python
>>> apply_homography([[2, 0, 1], [0, 2, 0], [0, 0, 1]], [(1, 1), (0, 3)]).tolist()
[[3.0, 2.0], [1.0, 6.0]]

```

## `diagram_to_court_homography(diagram_near_left, diagram_far_right)`

ShuttleSet's `homography_matrix` maps video pixels to a court **diagram**, not to meters.
In the diagram, x runs left to right and y points towards the camera, so the far
baseline has the smaller y. The corners are not identical across matches: the left
sideline is at x = 25 in some matches and x = 27.4 in others (y is always 150 to 810).
This function builds the scaling from one match's diagram to court meters.

- `diagram_near_left`, `diagram_far_right` (`Point`): the near-left and far-right
  court corners in diagram units.
- Returns a 3x3 `np.ndarray` (scale and offset only).
- Errors: `ZeroDivisionError` if the two corners share an x or a y.

```python
>>> diagram_to_meters = diagram_to_court_homography((27.4, 810.0), (327.6, 150.0))
>>> apply_homography(diagram_to_meters, [(27.4, 810.0), (327.6, 150.0), (177.5, 480.0)]).round(3).tolist()
[[0.0, 0.0], [6.1, 13.4], [3.05, 6.7]]

```

## `shuttleset_pixel_to_court_homography(shuttleset_homography, near_left_pixel, far_right_pixel)`

One match's pixel -> court-meter homography: ShuttleSet's matrix followed by
`diagram_to_court_homography` of that match's corners.

- `shuttleset_homography` (3x3): the match's `homography.homography_matrix`.
- `near_left_pixel`, `far_right_pixel` (`Point`): the match's `downleft` and `upright`
  corner pixels from `homography.csv`.
- Returns a 3x3 `np.ndarray`.
- Errors: as `diagram_to_court_homography`.

Example with Axelsen vs Ginting (Thailand Open 2021 SF). The 4 corner pixels map to the 4 court corners:

```python
>>> shuttleset_matrix = [
...     [1.2454713360219938, 0.5086861333343398, -616.7379978086947],
...     [0.017899391044319535, 5.78722378892984, -1480.2094549869073],
...     [8.602641647220215e-06, 0.0028640753448057883, 1.0],
... ]
>>> apply_homography(shuttleset_matrix, [(289.4, 659.6), (864.2, 301.6)]).round(1).tolist()
[[27.4, 810.0], [327.6, 150.0]]
>>> pixels_to_meters = shuttleset_pixel_to_court_homography(shuttleset_matrix, (289.4, 659.6), (864.2, 301.6))
>>> corners = [(289.4, 659.6), (987.2, 657.4), (864.2, 301.6), (412.6, 303.0)]
>>> (apply_homography(pixels_to_meters, corners).round(1) + 0.0).tolist()
[[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]]

```

## `is_missing(point)`, `present_or_none(point)`

- `is_missing` returns `True` if `point` is `None` or has a `None` or NaN coordinate.
- `present_or_none` returns the point as two floats, or `None` if it is missing.

```python
>>> is_missing(None), is_missing((1.0, float("nan"))), is_missing((1.0, 2.0))
(True, True, False)
>>> present_or_none((1, 2)), present_or_none((None, 2.0))
((1.0, 2.0), None)

```

## `is_on_far_half(point)`, `rotate_half_turn(point)`

- `is_on_far_half` returns `True` if `y > 6.7`. The point must not be missing.
- `rotate_half_turn` returns the point turned half a turn around the court centre,
  or `None` for a missing point.

```python
>>> is_on_far_half((1.0, 6.7)), is_on_far_half((1.0, 6.71))
(False, True)
>>> tuple(round(value, 2) for value in rotate_half_turn((1.0, 12.0)))
(5.1, 1.4)
>>> rotate_half_turn(None) is None
True

```

## `hitter_is_on_far_half(player, hit, opponent, landing)`

Decides which half the hitter is on, from the first position that is present, in
this order: the hitter's position (`player`), the hit point, then the receiver's
position (`opponent`) and the landing point, which are on the other half.

- All arguments are `Point | None`, in court meters before normalization.
- Returns `bool`, or `None` if all four are missing.

```python
>>> hitter_is_on_far_half(player=(3.0, 11.0), hit=None, opponent=None, landing=None)
True
>>> hitter_is_on_far_half(player=None, hit=None, opponent=(3.0, 11.0), landing=None)
False
>>> hitter_is_on_far_half(None, None, None, None) is None
True

```

## `depth_name(distance_from_net_m)`

`"front"` under 2.233 m from the net, `"mid"` under 4.467 m, `"rear"` beyond
(including behind the baseline). Negative distances count as `"front"`.

```python
>>> depth_name(0.5), depth_name(3.0), depth_name(6.5), depth_name(8.0)
('front', 'mid', 'rear', 'rear')

```

## `zone_in_near_half(point, left_handed=False)`

Zone of a point on the near half, from the near player's view. That player faces
the far baseline, so their right is `x > 3.05`. Forehand is the right side for a
right-handed player and the left side for a left-handed one.

- `point` (`Point`, not missing), `left_handed` (`bool`).
- Returns one of `ZONE_NAMES`.

```python
>>> zone_in_near_half((5.0, 1.0)), zone_in_near_half((5.0, 1.0), left_handed=True)
('rear_forehand', 'rear_backhand')
>>> zone_in_near_half((1.0, 6.0))
'front_backhand'

```

## `hit_zone(hit, hitter_left_handed=False)`

Zone of a normalized hit point, from the hitter's view. Returns `None` if `hit` is missing.

```python
>>> hit_zone((4.0, 3.5)), hit_zone(None)
('mid_forehand', None)

```

## `landing_zone(landing, receiver_left_handed=False)`

Zone of a normalized landing point, from the **receiver's** view. The receiver faces
the near baseline, so their right is `x < 3.05`. A landing on the hitter's own half
returns `"not_over_net"`. Returns `None` if `landing` is missing.

```python
>>> landing_zone((1.0, 12.5)), landing_zone((1.0, 12.5), receiver_left_handed=True)
('rear_forehand', 'rear_backhand')
>>> landing_zone((3.0, 6.0)), landing_zone(None)
('not_over_net', None)

```

## `is_cross_court(hit, landing)`

`True` when the hit point and the landing point are on opposite sides of the centre
line **and** the landing is at least 1.0 m sideways from the hit. A shot from the
middle that drifts just over the centre line stays straight. Returns `None` if
either point is missing.

```python
>>> is_cross_court((1.0, 1.0), (5.0, 12.0)), is_cross_court((2.8, 1.0), (3.3, 12.0)), is_cross_court((1.0, 1.0), (1.5, 12.0))
(True, False, False)
>>> is_cross_court(None, (1.0, 12.0)) is None
True

```

## `hitter_point_for_zone(hit, player)`

The point used for `hit_zone` and `is_cross`: the hit point when it is present and on
the near half, otherwise the hitter's position (`player`), otherwise the hit point.
In ShuttleSet the hit point is missing for most serves, and on a few shots it was
labelled on the wrong half. Both points must already be normalized.

```python
>>> hitter_point_for_zone((1.0, 2.0), (1.5, 2.5)), hitter_point_for_zone(None, (1.5, 2.5))
((1.0, 2.0), (1.5, 2.5))
>>> hitter_point_for_zone((1.0, 9.0), (1.5, 2.5))
(1.5, 2.5)

```

## `shot_geometry(hit, landing, player, opponent, hitter_left_handed=False, receiver_left_handed=False)`

Everything the `shots` table needs from one shot's positions. Use this for every
source so all rows follow the same definitions.

- `hit` (`Point | None`): the hitter's feet at the hit, in court meters.
- `landing` (`Point | None`): where the shot ended. In ShuttleSet this is where the receiver reached it.
- `player`, `opponent` (`Point | None`): hitter's and receiver's positions at the hit.
- `hitter_left_handed`, `receiver_left_handed` (`bool`).
- Returns a `dict` with `hit`, `landing`, `player`, `opponent` (normalized `Point | None`),
  and `hit_zone`, `landing_zone` (`str | None`) and `is_cross` (`bool | None`).
  If all four points are missing, nothing is rotated and every value is `None`.

A shot hit from the far half is turned so the hitter is near:

```python
>>> geometry = shot_geometry(hit=(5.0, 12.0), landing=(4.0, 2.0), player=(5.2, 12.3), opponent=(3.0, 3.0))
>>> {name: tuple(round(value, 2) for value in geometry[name]) for name in ("hit", "landing", "player", "opponent")}
{'hit': (1.1, 1.4), 'landing': (2.1, 11.4), 'player': (0.9, 1.1), 'opponent': (3.1, 10.4)}
>>> geometry["hit_zone"], geometry["landing_zone"], geometry["is_cross"]
('rear_backhand', 'rear_forehand', False)

```

A serve with no hit point uses the server's position:

```python
>>> serve = shot_geometry(hit=None, landing=(1.5, 8.9), player=(3.4, 4.5), opponent=(2.5, 8.6))
>>> serve["hit"], serve["hit_zone"], serve["landing_zone"], serve["is_cross"]
(None, 'front_forehand', 'front_forehand', True)

```
