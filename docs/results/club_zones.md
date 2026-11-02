# Zones on club footage: the old depth bands against the new

The user moved the depth bands from 1.98 m and 3.50 m from the net (the short service
line and a mid edge) to 2.75 m and 4.00 m (`analytics/court.py`, `DEPTH_EDGES_M`). This
page shows what that does to the zones and habits of the two labelled club videos at
30 fps:
- stand-in set 1: 172 labelled hits;
- test game 1 (`UHKzdJqgFwk`, labelled range): 110 labelled hits.

**How.** Each video's labelled hits are run through `vision.features` as if they were
detections. The hit point is the hitter's tracked feet, and the landing point is the
receiver's feet at the next labelled hit. These are the club footage's positions, not
the labelled contact points. The labels themselves are only read.

`vision.zone_split` then gives every row its zones under both band settings, with the
same `analytics.court` functions the pipeline uses. It puts the labelled shot types on
the rows and builds the habits with `analytics.tendencies` twice, once per band setting.
`me` is the near player and `opp` the far one.

## Stand-in set 1

| hit_zone | me old | me new | opp old | opp new |
|---|---|---|---|---|
| front_forehand | 5 | 14 | 7 | 19 |
| front_backhand | 1 | 9 | 12 | 19 |
| mid_forehand | 29 | 25 | 19 | 8 |
| mid_backhand | 15 | 7 | 17 | 12 |
| rear_forehand | 18 | 13 | 10 | 9 |
| rear_backhand | 22 | 22 | 17 | 15 |

| landing_zone | me old | me new | opp old | opp new |
|---|---|---|---|---|
| front_forehand | 7 | 19 | 5 | 14 |
| front_backhand | 12 | 19 | 1 | 9 |
| mid_forehand | 16 | 5 | 12 | 5 |
| mid_backhand | 13 | 8 | 11 | 3 |
| rear_forehand | 9 | 8 | 12 | 10 |
| rear_backhand | 16 | 14 | 21 | 21 |
| not_over_net | 5 | 5 | 0 | 0 |

hit_zone: 44 of 172 shots change zone
landing_zone: 41 of 140 shots change zone

### Habits that change, me

| Habit | Old bands | New bands |
|---|---|---|
| serve_placement: serve_short:front_backhand | - | 6 of 19 |
| serve_placement: serve_short:front_forehand | - | 8 of 19 |
| serve_placement: serve_short:mid_backhand | 9 of 19 | 3 of 19 |
| serve_placement: serve_short:mid_forehand | 8 of 19 | - |
| zone_mix: front_backhand:drop | - | 1 of 9 |
| zone_mix: front_backhand:lift | - | 3 of 9 |
| zone_mix: front_backhand:net | 1 of 1 | 5 of 9 |
| zone_mix: front_forehand:lift | 2 of 5 | 6 of 14 |
| zone_mix: front_forehand:net | 3 of 5 | 8 of 14 |
| zone_mix: mid_backhand:drop | 1 of 11 | - |
| zone_mix: mid_backhand:lift | 5 of 11 | 2 of 3 |
| zone_mix: mid_backhand:net | 5 of 11 | 1 of 3 |
| zone_mix: mid_forehand:drive | - | 1 of 8 |
| zone_mix: mid_forehand:lift | 6 of 13 | 3 of 8 |
| zone_mix: mid_forehand:net | 6 of 13 | 3 of 8 |
| zone_mix: mid_forehand:smash | 1 of 13 | 1 of 8 |
| zone_mix: rear_forehand:clear | 2 of 15 | 2 of 11 |
| zone_mix: rear_forehand:drive | 5 of 15 | 4 of 11 |
| zone_mix: rear_forehand:drop | 5 of 15 | 5 of 11 |
| zone_mix: rear_forehand:lift | 1 of 15 | - |
| zone_mix: rear_forehand:net | 2 of 15 | - |
| response (not listed) | 58 of 75 habits change | |

### Habits that change, opp

| Habit | Old bands | New bands |
|---|---|---|
| zone_mix: front_backhand:lift | 1 of 12 | 4 of 19 |
| zone_mix: front_backhand:net | 11 of 12 | 15 of 19 |
| zone_mix: front_forehand:drive | 1 of 7 | 1 of 19 |
| zone_mix: front_forehand:lift | - | 7 of 19 |
| zone_mix: front_forehand:net | 6 of 7 | 11 of 19 |
| zone_mix: mid_backhand:clear | - | 2 of 7 |
| zone_mix: mid_backhand:drive | 2 of 12 | 2 of 7 |
| zone_mix: mid_backhand:lift | 5 of 12 | 2 of 7 |
| zone_mix: mid_backhand:net | 5 of 12 | 1 of 7 |
| zone_mix: mid_forehand:drive | - | 1 of 4 |
| zone_mix: mid_forehand:drop | 1 of 15 | 1 of 4 |
| zone_mix: mid_forehand:lift | 7 of 15 | - |
| zone_mix: mid_forehand:net | 7 of 15 | 2 of 4 |
| zone_mix: rear_backhand:clear | 7 of 17 | 5 of 15 |
| zone_mix: rear_backhand:drive | 4 of 17 | 4 of 15 |
| zone_mix: rear_backhand:drop | 5 of 17 | 5 of 15 |
| zone_mix: rear_backhand:smash | 1 of 17 | 1 of 15 |
| zone_mix: rear_forehand:clear | 3 of 10 | 3 of 9 |
| zone_mix: rear_forehand:drive | 2 of 10 | 1 of 9 |
| zone_mix: rear_forehand:drop | 4 of 10 | 4 of 9 |
| zone_mix: rear_forehand:smash | 1 of 10 | 1 of 9 |
| response (not listed) | 68 of 85 habits change | |

## Test game 1 (`UHKzdJqgFwk`, labelled range)

| hit_zone | me old | me new | opp old | opp new |
|---|---|---|---|---|
| front_forehand | 0 | 1 | 3 | 13 |
| front_backhand | 1 | 2 | 7 | 15 |
| mid_forehand | 8 | 17 | 17 | 8 |
| mid_backhand | 6 | 11 | 10 | 2 |
| rear_forehand | 18 | 8 | 8 | 7 |
| rear_backhand | 20 | 14 | 12 | 12 |

| landing_zone | me old | me new | opp old | opp new |
|---|---|---|---|---|
| front_forehand | 3 | 12 | 0 | 1 |
| front_backhand | 7 | 12 | 1 | 2 |
| mid_forehand | 13 | 5 | 5 | 14 |
| mid_backhand | 7 | 2 | 6 | 8 |
| rear_forehand | 8 | 7 | 18 | 8 |
| rear_backhand | 12 | 12 | 17 | 14 |

hit_zone: 37 of 110 shots change zone
landing_zone: 30 of 97 shots change zone

### Habits that change, me

| Habit | Old bands | New bands |
|---|---|---|
| serve_placement: serve_short:front_backhand | 2 of 6 | 3 of 6 |
| serve_placement: serve_short:front_forehand | - | 3 of 6 |
| serve_placement: serve_short:mid_backhand | 1 of 6 | - |
| serve_placement: serve_short:mid_forehand | 3 of 6 | - |
| zone_mix: front_backhand:lift | 1 of 1 | 2 of 2 |
| zone_mix: front_forehand:lift | - | 1 of 1 |
| zone_mix: mid_backhand:drop | - | 1 of 8 |
| zone_mix: mid_backhand:lift | 4 of 6 | 3 of 8 |
| zone_mix: mid_backhand:net | 2 of 6 | 4 of 8 |
| zone_mix: mid_forehand:clear | - | 2 of 14 |
| zone_mix: mid_forehand:drive | - | 1 of 14 |
| zone_mix: mid_forehand:lift | 4 of 5 | 6 of 14 |
| zone_mix: mid_forehand:net | 1 of 5 | 4 of 14 |
| zone_mix: mid_forehand:smash | - | 1 of 14 |
| zone_mix: rear_backhand:clear | 4 of 15 | 4 of 12 |
| zone_mix: rear_backhand:drive | 2 of 15 | 2 of 12 |
| zone_mix: rear_backhand:drop | 2 of 15 | 1 of 12 |
| zone_mix: rear_backhand:net | 3 of 15 | 1 of 12 |
| zone_mix: rear_backhand:smash | 4 of 15 | 4 of 12 |
| zone_mix: rear_forehand:clear | 2 of 18 | - |
| zone_mix: rear_forehand:drive | 3 of 18 | 2 of 8 |
| zone_mix: rear_forehand:lift | 5 of 18 | 2 of 8 |
| zone_mix: rear_forehand:net | 4 of 18 | 1 of 8 |
| zone_mix: rear_forehand:smash | 4 of 18 | 3 of 8 |
| response (not listed) | 59 of 67 habits change | |

### Habits that change, opp

| Habit | Old bands | New bands |
|---|---|---|
| serve_placement: serve_long:mid_forehand | - | 1 of 1 |
| serve_placement: serve_long:rear_forehand | 1 of 1 | - |
| zone_mix: front_backhand:drive | - | 1 of 12 |
| zone_mix: front_backhand:drop | 1 of 7 | 1 of 12 |
| zone_mix: front_backhand:lift | 2 of 7 | 6 of 12 |
| zone_mix: front_backhand:net | 2 of 7 | 2 of 12 |
| zone_mix: front_backhand:smash | 2 of 7 | 2 of 12 |
| zone_mix: front_forehand:lift | 1 of 2 | 5 of 11 |
| zone_mix: front_forehand:net | 1 of 2 | 5 of 11 |
| zone_mix: front_forehand:smash | - | 1 of 11 |
| zone_mix: mid_backhand:drive | 1 of 7 | - |
| zone_mix: mid_backhand:lift | 4 of 7 | - |
| zone_mix: mid_backhand:net | 2 of 7 | 2 of 2 |
| zone_mix: mid_forehand:lift | 5 of 12 | 2 of 4 |
| zone_mix: mid_forehand:net | 6 of 12 | 2 of 4 |
| zone_mix: mid_forehand:smash | 1 of 12 | - |
| zone_mix: rear_forehand:clear | 2 of 7 | 2 of 6 |
| zone_mix: rear_forehand:drop | 3 of 7 | 3 of 6 |
| zone_mix: rear_forehand:lift | 1 of 7 | - |
| zone_mix: rear_forehand:smash | 1 of 7 | 1 of 6 |
| response (not listed) | 55 of 65 habits change | |

The "response" family (a reply to the opponent's shot from a given zone, landing in a
given zone) has a zone in its condition, so most of those habits change name with the
bands. Almost all of them rest on 1-3 shots each, so they aren't listed one by one; the
reports have them in full.

## What it says

- **About a quarter of the shots change zone:**
  - hit zones: 44 of 172 on set 1 and 37 of 110 on test game 1;
  - landing zones: 41 of 140 and 30 of 97.
  Every one of them moves one band towards the net, mid to front or rear to mid, and
  none changes side: both edges moved away from the net.
- **The front zones grow and mid shrinks.** Everything between 1.98 m and 2.75 m from the
  net is now front court rather than mid. On set 1:
  - front-zone hits grow from 6 to 23 for `me` and from 19 to 38 for `opp`;
  - landings in the front zones from 19 to 38 for `me`.
- **Short serves now land "front" rather than "mid".** A short serve lands just past the
  short service line. Under the old bands that was mid, under the new it's front. On set
  1, `me`'s 19 short serves go from 17 landing in mid (9 backhand, 8 forehand) to 14 in
  front (6 backhand, 8 forehand) and 3 in mid.
- **Zone mixes shift in the same direction.** For example, `opp`'s net shots on set 1:
  - front-backhand: 11 of 12, against 15 of 19;
  - front-forehand: 6 of 7, against 11 of 19;
  - mid-forehand: 7 of 15, against 2 of 4.
  The front zones now take in shots played a little further from the net.
- **Small numbers:** the habits rest on 2-19 shots each, on one stand-in set and one
  test game.

## How to run

```
D=data/club_zones; S1=data/recordings/stand_in/set1_30.mp4; U=data/recordings/club/UHKzdJqgFwk_30.mp4
.venv/bin/python -m vision.flight_types --train $S1 data/labels/stand_in_set1_30.csv $D/stand_in_set1_30/scout.duckdb 13117 \
  --train $U data/labels/club_UHKzdJqgFwk_30.csv $D/club_UHKzdJqgFwk_30/scout.duckdb 18149 --hits-out $D/hits
for pair in "$S1:stand_in_set1_30:13117" "$U:club_UHKzdJqgFwk_30:63443"; do IFS=: read V M E <<< "$pair"
  mkdir -p $D/$M; cp data/scout.duckdb $D/$M/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id $M --range 0 $E 1 --database $D/$M/scout.duckdb --hits-dir $D/hits
  .venv/bin/python -m vision.zone_split --labels data/labels/$M.csv --database $D/$M/scout.duckdb --match-id $M --out $D/$M
done
```

`vision.zone_split` writes `old_bands.duckdb` and `new_bands.duckdb` into `--out` and
prints the tables above (`data/club_zones/<match>/report.md`).
