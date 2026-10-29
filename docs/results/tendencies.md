# Habit counts: results

## How to run
```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets]
```
How habits are counted: `docs/analytics/tendencies.md`. Each player's habits are counted only from their own
shots; there is no comparison with other players, no score, no ranking and no minimum sample size.

## Runs on every match
All 88 ShuttleSet players (44 matches x 2) and the 2 players of my tracked match run without error.
Loading the shots and building the observations takes 0.2 s; counting all 90 players then takes 1.0 s.

| | Entries per player |
|---|---|
| ShuttleSet, whole match | 126 to 333 (median 245) |
| `match_tracked`, `me` | 68 |
| `match_tracked`, `opp` | 63 |

- One entry per (situation, choice), 21,794 in total over 9,202 situations.
- In every situation the choices' `k` add up to its `n`: checked for all 9,202 situations (9,202 hold).
- Nothing is filtered: 11,792 entries are a single shot ("1 of n").
- Entries per family: zone_mix 2,924, response 17,230, serve_placement 538, winning_shot 351, error_shot 482, error_reason 269.
- Label source: ShuttleSet 21,285 `labels`, 295 `mixed` and
  83 `predictions` (21,663 in all). A situation is `mixed` when some of its shots were typed `unknown`
  in ShuttleSet and got the classifier's group, and `predictions` when all of them were: the 83 `predictions`
  entries (83 `response`) come from situations so small that every shot in them was `unknown`, e.g.
  `response:serve|mid_forehand|mid_middle:drive cross, 1 of 1`; my tracked match 131 `predictions`
  and 0 `mixed`, since nobody labels my shots.

Replies count the other player's previous shot by its type, where it was hit from and where it landed, and the
reply by its type and direction: 31,705 counted replies, 20,924 straight,
10,778 cross and 3 with no direction (`is_cross` missing).

Example, Ginting (B) in Axelsen vs Ginting (Thailand Open 2021 SF), in his most frequent reply situation:

- After a lift from the other player's front backhand to the rear backhand: smashes straight 7 of 25 times.
- After a lift from the other player's front backhand to the rear backhand: smashes cross 6 of 25 times.
- After a lift from the other player's front backhand to the rear backhand: clears straight 4 of 25 times.
- After a lift from the other player's front backhand to the rear backhand: drops straight 4 of 25 times.
- After a lift from the other player's front backhand to the rear backhand: drops cross 3 of 25 times.
- After a lift from the other player's front backhand to the rear backhand: clears cross 1 of 25 times.

The choices add up: 7 + 6 + 4 + 4 + 3 + 1 = 25.

## Limits
- Counts describe what happened in this match; small `n` means little, and the page shows `n` so the reader can judge.
- Player labels on one-shot `unknown` rallies are unreliable in ShuttleSet, so those rallies (and the 2 known A/B-swapped
  rallies) are left out of per-player counts.
- On my footage every shot type is a classifier prediction (held-out accuracy on ShuttleSet in
  `docs/results/classifier.md`); positions and timing come from tracking.
- A set 1 -> set 2 check of how well habits carry over was removed with the comparison to pros; it can come back as its own task.
