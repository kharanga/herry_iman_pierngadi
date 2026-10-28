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
Zones: each half has 3 equal depth bands from the net (2.23 m each: front, mid, rear) x 3 sides
(forehand / middle / backhand). The middle column is 1.0 m wide around the centre line
(x 2.55-3.55 m, `MIDDLE_COLUMN_WIDTH_M`), and a point exactly on its edge counts as middle.
`hit_zone` is seen from the hitter, `landing_zone` from the receiver, who faces the other way.
Players are right-handed except `LEFT_HANDED_PLAYER_NAMES` (Carolina Marin, Kento Momota;
11 of 44 matches involve one of them); for them forehand and backhand swap, middle does not.
Player `A` is the match winner in all 44 matches, which is how `A`/`B` are matched to names.

| Zone | `hit_zone` | `landing_zone` |
|---|---|---|
| front_forehand | 5,775 | 5,848 |
| front_middle | 1,601 | 1,569 |
| front_backhand | 5,034 | 5,106 |
| mid_forehand | 3,958 | 3,946 |
| mid_middle | 3,566 | 1,063 |
| mid_backhand | 4,086 | 4,215 |
| rear_forehand | 5,605 | 6,068 |
| rear_middle | 365 | 406 |
| rear_backhand | 5,364 | 5,945 |
| not_over_net | n/a | 1,131 |
| NULL (no position) | 1,129 | 1,186 |

### Shots moved into the middle column
Compared with the earlier 6-zone build (forehand / backhand only), joined shot by shot on
(`match_id`, `set_no`, `rally`, `ball_round`, `frame_num`), which is unique:

| Before -> after | `hit_zone` | `landing_zone` |
|---|---|---|
| front_forehand -> front_middle | 814 | 801 |
| front_backhand -> front_middle | 787 | 768 |
| mid_forehand -> mid_middle | 1,751 | 525 |
| mid_backhand -> mid_middle | 1,815 | 538 |
| rear_forehand -> rear_middle | 164 | 192 |
| rear_backhand -> rear_middle | 201 | 214 |
| **total moved** | **5,532 of 35,354 (15.6%)** | **3,038 of 35,297 (8.6%)** |

No other change: every changed zone moved into `middle` at the same depth, from forehand and
backhand in about equal numbers. All other columns, `is_cross` included, are identical
row by row (`EXCEPT ALL` on every column except the two zones gives 0 rows). The middle column
fills most in `mid` for hits (3,566), where players stand on their base position between
shots, and least at the rear (365 hits, 406 landings), where shots go to the corners.

`is_cross`: the hit and landing are on opposite sides of the centre line and at least 1.0 m apart
sideways. 12,395 of 35,293 shots (35%). The labelled shot types back this up:

| Type | Cross-court |
|---|---|
| cross-court net shot | 97.7% |
| long service | 87.7% (serves must go diagonally) |
| clear | 30.7% |
| smash | 32.6% |
| net shot | 13.9% |
| back-court drive | 12.1% |

The 1.0 m minimum was picked from these numbers. Requiring only opposite sides gives
98% / 17% for cross-court net shot / net shot; 1.0 m gives 98% / 14%; 2.0 m drops
cross-court net shots to 95%. Limitation: a shot from a sideline to just past the centre
counts as cross (spot-check shot 1).

## Coverage
- 35,354 shots have a hitter point (hit or hitter position) and a `hit_zone`. 1,129 have neither.
- The hit point is NULL on 3,646 of 3,684 first shots, because there is no previous landing.
  The server's position is used for their `hit_zone`, so 2,710 first shots have one.
- 35,297 shots have a landing and a `landing_zone`; 35,293 have both points and an `is_cross`.
- 1,125 shots have no position at all.

## Scores
ShuttleSet's `roundscore_A/B` are the score **after** the rally (rally 1 already shows 1-0).
`roundscore_me/opp` are the score before it: the rally winner loses one point. The winner is
`getpoint_player`, or for the 174 rallies without one, the side whose score went up. Check:
for all 3,545 pairs of consecutive rallies, the next rally's before-score is this rally's
before-score plus exactly one point. Every set's rally 1 is 0-0.

## Spot check
`data/checks/shots/random_shots.png` (seed 7). 10 random shots with a hit and a landing, on a
top-down court with the hitter's half at the bottom; gray lines mark the depth bands and the middle column. I checked each against its numbers:

| # | Shot | Hit (m) | Landing (m) | Zones | Looks right? |
|---|---|---|---|---|---|
| 1 | clear by Christie (B) | 0.78, 1.12 | 3.10, 11.72 | rear_backhand -> rear_middle, cross | yes: rear left corner, deep, to the middle of the back line. It lands 0.05 m past centre, so "cross" is borderline |
| 2 | net shot by Chen Long | 2.65, 4.70 | 3.07, 7.56 | front_middle -> front_middle | yes: from the middle at the net to just over it, down the centre |
| 3 | net shot by Chou | 4.93, 4.25 | 4.85, 7.31 | mid_forehand -> front_backhand | yes: 2.45 m from the net is just in "mid"; the receiver's left |
| 4 | push by Chou (B) to Momota | 4.53, 4.33 | 1.56, 12.54 | mid_forehand -> rear_backhand, cross | yes: left-handed Momota's backhand is on his right (small x) |
| 5 | net shot by Momota | 1.10, 5.14 | 1.36, 7.40 | front_forehand -> front_forehand | yes: left side is a left-hander's forehand |
| 6 | back-court drive by Momota | 1.11, 0.66 | 0.93, 10.90 | rear_forehand -> mid_forehand | yes: deep, straight down the line |
| 7 | passive drop by Kidambi (B) | 1.23, 1.26 | 6.13, 9.90 | rear_backhand -> mid_backhand, cross | yes: receiver reached it just outside the sideline |
| 8 | drop by Intanon | 2.15, 1.04 | 5.11, 8.60 | rear_backhand -> front_backhand, cross | yes |
| 9 | lob by Axelsen | 0.81, 5.10 | 1.19, 12.86 | front_backhand -> rear_forehand | yes: from the net to the back line |
| 10 | net shot by Axelsen | 1.23, 4.28 | 2.86, 7.88 | mid_backhand -> front_middle | yes: lands 0.19 m from the centre line, inside the middle column |

Shot types agree with distances: the four net shots travel 2.3-4.0 m and stay in front.
The clear and the back-court drive travel 10.2-10.9 m, and the lob from the net 7.8 m. Receivers (blue) cluster around the far
mid-court base position.

## Known data issues
**`player` (A/B) disagrees with the court position in 54 rallies.** The test is the end-swap
rule: `downcourt` = 0 means A starts near, ends swap each set and at 11 in set 3. For 35,153 of
35,314 shots with a hitter position (99.54%), the hitter's half from `player_location` matches
the half the rule gives that `player`. 152 rallies have at least one shot that disagrees. In 54
of them, every shot with a hitter position disagrees (63 shots, 9 matches; 23 in set 1, 24 in set 2,
7 in set 3). The receiver's position is on the other half in every one of those shots, so the
positions agree with each other and only the A/B label looks wrong.

- **52 of the 54 are one-shot rallies** whose only shot is an `unknown` first shot. Of the 113
  one-shot rallies that have a hitter position, 52 disagree: about a coin flip. So `player` on a
  one-shot `unknown` rally should not be trusted (182 one-shot rallies in all, 160 of them `unknown`).
- **2 are full rallies where A and B look swapped:** `CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal`
  set 2 rally 6 (5 shots), and `Ratchanok_Intanon_Pusarla_V._Sindhu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals`
  set 2 rally 1 (6 shots).

Effect: geometry is correct, because normalization follows positions, not `player`. But
`player`, `roundscore_me/opp` attribution and `getpoint_player` follow the source labels, so
per-player tendencies count those shots for the wrong player. The table is left as the source has it.
The tendency engine should skip one-shot `unknown` rallies (`ball_round` = 1 is the rally's only
shot and `type_en = 'unknown'`) for anything per-player.

How to find them: convert `player_location` to meters (pixels -> meters as above, before
normalizing), take the hitter's half (`y` > 6.7 = far), and compare it with the rule's half for
`player`. The set 3 swap is after the first rally whose after-score reaches 11 for either player.
A rally is flagged when every shot with a `player_location` disagrees.

## Data notes
- 1,240 of 3,684 first shots of a rally are typed `unknown` in ShuttleSet; only 2,424 are
  typed as a serve. Serve tendencies should use `type_group = 'serve'` and note the gap.
- One key repeats without being an exact duplicate: Ginting v Gemke (Thailand Open 2021 QF)
  set 3 rally 25 has two different first shots (frames 128176 and 129140, both `unknown`).
  Both are kept, so (`match_id`, `set_no`, `rally`, `ball_round`) is not a unique key.
- Some rallies are missing from the source (17 sets have gaps in rally numbers); the score
  handling above does not depend on rallies being consecutive.
