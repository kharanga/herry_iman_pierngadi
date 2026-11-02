# Shots table: results

## How to run
```
.venv/bin/python -m ingest.load_shuttleset
.venv/bin/python -m analytics.build_shots
.venv/bin/python -m analytics.spot_check
```

## Output
`shots` in `data/scout.duckdb`: 36,483 ShuttleSet rows, 28 columns, exactly as
`analytics/schema.sql` (column list in `docs/analytics/build_shots.md`).
The spot-check image is `data/checks/shots/random_shots.png`.

| Check | Result |
|---|---|
| Columns and types equal `schema.sql` | yes, checked on every build; a different table raises `ShotsSchemaMismatchError` |
| Every `type_en` has a `type_group` | 19 of 19 types mapped, 0 NULL `type_group`; `passive drop` -> `drop`; 1,407 shots are the explicit `unknown` |
| Exact duplicates dropped | **1** (36,484 `raw_shots` -> 36,483 `shots`): Momota v Chou, Fuzhou Open 2018 Final, set 3 rally 10, last shot listed twice |
| Hitter normalized to the near half | every shot with a hitter position has `player_y` ≤ 6.7 |
| Landing inside the court, non-error shots | **99.86%** (31,744 of 31,790) inside 6.1 x 13.4 m; 99.60% inside the receiver's half |
| `hit_zone`, `landing_zone`, `is_cross` | set on every shot that has the points they need (below) |
| Rebuild idempotent | a second build gives 0 differing rows (`EXCEPT ALL` both ways), and `own` rows are kept |

Group sizes: net 11,793, lift 5,631, drive 4,572, smash 4,234, drop 3,500, clear 2,922,
serve 2,424, unknown 1,407.

## Pixels to meters
ShuttleSet's `homography_matrix` maps pixels to a court diagram, not to meters. The diagram
is not the same in every match: applying each match's matrix to its own corner pixels gives a left
sideline at x = 25 in some matches and x = 27.4 in others (right sideline 325 or 327.6, baselines
always y = 150 far and 810 near). So the diagram -> meter scaling is computed per match from its
own corners, and the 4 corner pixels of each match land exactly on (0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4).

## What "landing" means
In ShuttleSet, each shot's hit point equals the previous shot's landing point
(31,679 of 31,692 pairs), and the hit point is the hitter's **feet**.
A pair is a shot and the shot before it in the same match, set and rally, ordered by
`ball_round`, after dropping exact duplicates. Both the hit point and the previous landing
point must be present. Equal means both pixel coordinates match within 1e-6.
All 31,692 pairs have consecutive `ball_round`. Counted on `raw_shots`:

```sql
WITH distinct_raw_shots AS (
    SELECT DISTINCT * FROM raw_shots
),
shot_pairs AS (
    SELECT
        distinct_raw_shots.hit_x,
        distinct_raw_shots.hit_y,
        LAG(distinct_raw_shots.landing_x) OVER rally_order AS previous_landing_x,
        LAG(distinct_raw_shots.landing_y) OVER rally_order AS previous_landing_y
    FROM distinct_raw_shots
    WINDOW rally_order AS (
        PARTITION BY distinct_raw_shots.match_id, distinct_raw_shots.set_no, distinct_raw_shots.rally
        ORDER BY distinct_raw_shots.ball_round
    )
)
SELECT
    COUNT(*) AS pairs,
    COUNT(*) FILTER (
        WHERE ABS(shot_pairs.hit_x - shot_pairs.previous_landing_x) < 1e-6
        AND ABS(shot_pairs.hit_y - shot_pairs.previous_landing_y) < 1e-6
    ) AS hit_equals_previous_landing
FROM shot_pairs
WHERE shot_pairs.hit_x IS NOT NULL AND shot_pairs.previous_landing_x IS NOT NULL
```
Result: `pairs` 31,692, `hit_equals_previous_landing` 31,679.

So `landing_x/y` is where the receiver was standing when they reached the shuttle,
not where it would have hit the floor. Only on a rally's last shot is it where the
rally ended.

That's why the "landing inside the court" check uses **non-error shots**: every shot
except a rally's last shot with a `lose_reason`. For those, the landing is a player's
feet, and players sometimes stand just outside the lines; 46 of 31,790 do. Rally-ending
shots are reported separately, since a landing outside is often the point of them:

| `lose_reason` | Shots with a landing | Inside the receiver's half |
|---|---|---|
| opponent winner | 1,189 | 99.2% |
| out | 1,046 | 57.3% (the rest outside 6.1 x 13.4) |
| net | 841 | 13.0% (mostly on the hitter's half: `not_over_net`) |
| not over net | 376 | 16.8% |
| misjudged landing | 44 | 93.2% |
| misjudged opponent landing | 7 | 100% |
| foul | 4 | 100% |

57% of `out` shots landing inside 6.1 x 13.4 is expected for singles: those are inside the
doubles lines but outside the singles sidelines (0.46 m in) or the singles long service line.

## Which half is the hitter on?
Players change ends between sets and at 11 in set 3. The hitter's half comes from
the hitter's own position (`player_location`, else the hit point, else the receiver
or the landing on the other half). As a check, the rule "`downcourt` = 0 means A starts
near, ends swap every set and at 11 in set 3" agrees for 35,177 of 35,354 shots (99.5%),
taking the hitter's position or, where it is missing, the hit point. (With `player_location`
only, as in "Known data issues" below, it is 35,153 of 35,314.)
The 177 disagreements are not near the net (median 3.9 m from it). In all 161 of them
that have a receiver position, the receiver is on the other half from the hitter, so the
positions agree with each other and the position is used. In 86 shots the hit point
and the hitter's position are on opposite halves. The end-swap rule sides with the
hitter's position in all 86, so for `hit_zone` and `is_cross` the hitter's position is
used there instead of the hit point.

## Zones and cross-court
Zones: each half has 3 depth bands (front: net to 2.75 m, just behind the short service line at 1.98 m;
mid: 2.75-4.00 m; rear: 4.00 m to the baseline and beyond) x 2 sides
(forehand / backhand, split at the centre line, x = 3.05 m; `DEPTH_EDGES_M`, `SIDE_NAMES`).
A point exactly on a band edge belongs to the band behind it; a point exactly on the centre line
is the left side. There is no middle column. `hit_zone` is seen from the hitter, `landing_zone`
from the receiver, who faces the other way. Players are right-handed except
`LEFT_HANDED_PLAYER_NAMES` (Carolina Marin, Kento Momota; 11 of 44 matches involve one of them);
for them forehand and backhand swap. Player `A` is the match winner in all 44 matches, which is
how `A`/`B` are matched to names.

| Zone | `hit_zone` | `landing_zone` |
|---|---|---|
| front_forehand | 8,228 | 7,956 |
| front_backhand | 7,262 | 7,066 |
| mid_forehand | 3,205 | 2,257 |
| mid_backhand | 3,419 | 2,461 |
| rear_forehand | 6,634 | 7,167 |
| rear_backhand | 6,606 | 7,259 |
| not_over_net | n/a | 1,131 |
| NULL (no position) | 1,129 | 1,186 |

### What moved from the 1.98 / 3.50 m edges
The bands used to end on the painted short service line (1.98 m) and at 3.50 m. Both edges moved back
(2.75 m and 4.00 m from the net) after looking at the zone overlay on club footage. Sides are unchanged,
so a shot only moves one band towards the net. 7,356 of 35,354 `hit_zone`s (20.8%) and 6,579 of 35,297
`landing_zone`s (18.6%) changed:

| Before -> after | `hit_zone` | `landing_zone` |
|---|---|---|
| mid_forehand -> front_forehand (2.75 m edge instead of 1.98) | 2,659 | 2,269 |
| mid_backhand -> front_backhand | 2,407 | 2,102 |
| rear_backhand -> mid_backhand (4.00 m edge instead of 3.50) | 1,192 | 1,176 |
| rear_forehand -> mid_forehand | 1,098 | 1,032 |

The front band grew most (hits 10,424 -> 15,490): shots played just behind the short service line, up to
2.75 m from the net, now count as front. The rear band shrank (hits 15,530 -> 13,240), because the
players' base position 3.5-4.0 m from the net is now mid. The `not_over_net` and NULL rows didn't change.

`is_cross`: TRUE when the hit and landing points are on opposite sides of the centre line, NULL
for a landing on the hitter's own half (`not_over_net`, 1,131 shots; the earlier build gave those a
value). 13,774 of 34,162 shots (40%) are cross; 1,675 shots changed value against the earlier
1.0 m-sideways rule, all of them shots that crossed the centre line by less than 1 m. The labelled
shot types back the rule up:

| Type | Cross-court |
|---|---|
| cross-court net shot | 98.4% |
| long service | 98.9% (serves must go diagonally) |
| short service | 98.8% |
| push | 48.3% |
| lob | 46.9% |
| drop | 44.3% |
| drive | 34.4% |
| smash | 33.4% |
| clear | 30.9% |
| net shot | 17.8% |
| back-court drive | 11.8% |

Serves come out 99% cross (every serve is diagonal), where the 1.0 m rule had long serves at 87.7%:
a serve from next to the centre line to just across it travels less than 1 m sideways. Net shots at
17.8% (13.9% before) are the shots that drift just over the centre line, now counted as cross.

## Coverage
- 35,354 shots have a hitter point (hit or hitter position) and a `hit_zone`. 1,129 have neither.
- The hit point is NULL on 3,646 of 3,684 first shots, because there is no previous landing.
  The server's position is used for their `hit_zone`, so 2,710 first shots have one.
- 35,297 shots have a landing and a `landing_zone`; 34,162 have both points, a landing over the net and an `is_cross`
  (1,131 `not_over_net` landings keep `is_cross` NULL).
- 1,125 shots have no position at all.

## Scores
ShuttleSet's `roundscore_A/B` are the score **after** the rally (rally 1 already shows 1-0).
`roundscore_me/opp` are the score before it: the rally winner loses one point. The winner is
`getpoint_player`, or for the 174 rallies without one, the side whose score went up. Check:
for all 3,545 pairs of consecutive rallies, the next rally's before-score is this rally's
before-score plus exactly one point. Every set's rally 1 is 0-0.

## Spot check
`data/checks/shots/random_shots.png` (seed 7). 10 random shots with a hit and a landing, on a
top-down court with the hitter's half at the bottom; gray lines mark the band edges (2.75 and
4.00 m from the net) and the centre line through the front bands. I checked each against its numbers:

| # | Shot | Hit (m) | Landing (m) | Zones | Looks right? |
|---|---|---|---|---|---|
| 1 | clear by Christie (B) | 0.78, 1.12 | 3.10, 11.72 | rear_backhand -> rear_backhand, cross | yes: rear left corner, deep, to just right of the centre line, which from the receiver's view is their backhand; cross by 0.05 m |
| 2 | net shot by Chen Long | 2.65, 4.70 | 3.07, 7.56 | front_backhand -> front_backhand, cross | yes: 2.00 m from the net, just behind the short service line but inside the 2.75 m edge, so "front"; lands 0.02 m across the centre, so cross |
| 3 | net shot by Chou | 4.93, 4.25 | 4.85, 7.31 | front_forehand -> front_backhand, straight | yes: 2.45 m from the net; the receiver's left |
| 4 | push by Chou (B) to Momota | 4.53, 4.33 | 1.56, 12.54 | front_forehand -> rear_backhand, cross | yes: 2.37 m from the net; left-handed Momota's backhand is on his right (small x) |
| 5 | net shot by Momota | 1.10, 5.14 | 1.36, 7.40 | front_forehand -> front_forehand, straight | yes: left side is a left-hander's forehand |
| 6 | back-court drive by Momota | 1.11, 0.66 | 0.93, 10.90 | rear_forehand -> rear_forehand, straight | yes: deep, straight down the line; the receiver takes it 4.2 m from the net, just past the 4.00 m edge, so "rear" |
| 7 | passive drop by Kidambi (B) | 1.23, 1.26 | 6.13, 9.90 | rear_backhand -> mid_backhand, cross | yes: receiver reached it just outside the sideline, 3.2 m from the net |
| 8 | drop by Intanon | 2.15, 1.04 | 5.11, 8.60 | rear_backhand -> front_backhand, cross | yes |
| 9 | lob by Axelsen | 0.81, 5.10 | 1.19, 12.86 | front_backhand -> rear_forehand, straight | yes: from the net to the back line |
| 10 | net shot by Axelsen | 1.23, 4.28 | 2.86, 7.88 | front_backhand -> front_forehand, straight | yes: 2.42 m from the net; lands 0.19 m left of the centre line, the same side as the hit, so straight |

Shot types agree with distances: the four net shots travel 2.3-4.0 m and stay in front.
The clear and the back-court drive travel 10.2-10.9 m, and the lob from the net 7.8 m. Receivers (blue) cluster around the far
mid-court base position.
