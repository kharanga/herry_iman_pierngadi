# Tendencies: results

## How to run
```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets]
.venv/bin/python -m analytics.validate_tendencies
```
Numbers below come from `data/checks/tendencies/set_one_to_set_two.json` (written by
`analytics.validate_tendencies`) and from running `analytics.tendencies` on every player in `data/scout.duckdb`.
How tendencies are found and scored: `docs/analytics/tendencies.md`.

## Runs on every match
All 88 ShuttleSet players (44 matches x 2) run without error. Tendencies per player over a whole
match: 6 to 32 (median 14); every one has n >= 5.
Label source: 1,204 tendencies use only true labels and 90 are `mixed`
(a ShuttleSet shot typed `unknown` got the classifier's group). None are predictions only, as expected
for ShuttleSet.

Families in the top 3 per player (what the report shows): zone_mix 87, response 111, serve_placement 40, winning_shot 10, error_shot 10, error_reason 6.

Example, Axelsen vs Ginting (Thailand Open 2021 SF), top 3 per player over the whole match:

Viktor AXELSEN (A):
1. Errors hit out: 21 of 28 errors (pros: 46%). (score 0.141, `error_reason:error:out`)
2. After a net shot: lifts 88 of 148 times (pros: 39%). (score 0.134, `response:net:lift`)
3. From the front court, backhand side: lifts 42 of 71 times (pros: 37%). (score 0.122, `zone_mix:front_backhand:lift`)

Anthony Sinisuka GINTING (B):
1. From the mid court, backhand side: drives 25 of 77 times (pros: 18%). (score 0.065, `zone_mix:mid_backhand:drive`)
2. Serves mid-depth down the middle 3 of 12 times (pros: 4%). (score 0.065, `serve_placement:serve:mid_middle`)
3. From the front court, forehand side: drives 19 of 63 times (pros: 16%). (score 0.056, `zone_mix:front_forehand:drive`)

## Set 1 -> set 2
For each of the 88 players: find tendencies in set 1 only, then check set 2 of the same match.
A tendency **follows** when, in set 2, the player is in the same situation at least
5 times and still chooses the outcome **more often than pros do** (set-2 rate above the
reference rate). The reference never includes the match itself.

**Matched chance.** "Above the pros' rate" can come true in set 2 by chance, and how often depends on the
situation and outcome. So each evaluable candidate gets its own chance level. Take all *other* ShuttleSet players
(other matches) who were in the same situation at least 5 times in their set 2; the chance level is the
share of them whose set-2 rate for the same outcome is above the same reference rate. A selection's matched chance
is the mean over its evaluable candidates. It answers: how often would this claim come true for a player we know
nothing about? Computed by `analytics.validate_tendencies.with_matched_chance`.

| Selection | Evaluable in set 2 | Follow | Follow rate | Matched chance | Lift over matched chance (points) |
|---|---|---|---|---|---|
| tendencies (lower bound above pros) | 906 | 519 | 57.3% | 38.4% | +18.9 |
| top 3 per player (what the report shows) | 246 | 171 | 69.5% | 42.9% | +26.6 |
| naive: raw set-1 rate above pros | 2,831 | 1,463 | 51.7% | 42.3% | +9.4 |
| naive only: raw rate above pros, lower bound not | 1,925 | 944 | 49.0% | 44.1% | +4.9 |
| not flagged: raw rate at or below pros | 2,157 | 855 | 39.6% | 46.8% | -7.2 |

What this says:
- **The top 3 per player are predictive**: 69.5% follow, against a matched chance of 42.9%
  (+26.6 points). These are the bullets the report shows.
- **All tendencies together**: 57.3% against 38.4% (+18.9 points).
- **The confidence bound matters**: the candidates the naive rule adds but the bound rejects follow 49.0% against
  44.1% (+4.9 points), a much smaller lift than the tendencies the bound keeps.
- **The old "chance level" was biased.** An earlier version of this page used the unflagged candidates (39.6%) as
  chance. Those were picked *because* their set-1 rate was at or below the pros, so regression to the mean pulls
  them down in set 2, and they cover a different mix of situations. Against their own matched chance
  (46.8%) they are -7.2 points. Comparing other selections with them overstated every lift.

## By family

All tendencies (set 1):

| Family | Evaluable | Follow rate | Matched chance | Lift (points) |
|---|---|---|---|---|
| `zone_mix` | 350 | 55.7% | 38.8% | +16.9 |
| `response` | 361 | 62.3% | 39.8% | +22.6 |
| `serve_placement` | 98 | 59.2% | 34.0% | +25.1 |
| `winning_shot` | 24 | 41.7% | 28.2% | +13.5 |
| `error_shot` | 49 | 44.9% | 40.3% | +4.6 |
| `error_reason` | 24 | 37.5% | 35.1% | +2.4 |

Top 3 per player only:

| Family | Evaluable | Follow rate | Matched chance | Lift (points) |
|---|---|---|---|---|
| `zone_mix` | 97 | 62.9% | 43.6% | +19.3 |
| `response` | 95 | 78.9% | 45.1% | +33.8 |
| `serve_placement` | 30 | 70.0% | 34.3% | +35.7 |
| `winning_shot` | 5 | 60.0% | 31.3% | +28.7 |
| `error_shot` | 11 | 72.7% | 44.6% | +28.1 |
| `error_reason` | 8 | 37.5% | 46.4% | -8.9 |

- Every family follows more often than matched chance, but by very different margins: largest for
  `serve_placement` (+25.1) and `response` (+22.6), then `zone_mix`
  (+16.9) and `winning_shot` (+13.5, on only 24 evaluable).
  `error_shot` (+4.6) and `error_reason` (+2.4) are barely above chance.
- In the top 3, `error_reason` is below its matched chance (-8.9 points, on only
  8 evaluable); the other families' top-3 bullets are well above it.
- The rally-ending families rest on few rallies: a player hits 4 to 29 rally-ending shots in a set
  (median 17), counted as `winning_shot` + `error_shot` observations in sets 1 and 2, over all
  176 player-sets.

## Recommendation for the report
- Show only the top 3 per player: that is where the evidence is strongest.
- Trust `serve_placement` and `response` bullets most. Show `error_shot` and `error_reason` bullets, if they reach the
  top 3, as weaker evidence: they are few and barely beat chance.
- Always show the "k of n" evidence and the pros' rate, as the text already does.

## Limits
- One set is short: a player hits 41 to 293 shots in a set (median 162.5). That counts every shot in
  sets 1 and 2 except the excluded rallies below, over all 176 player-sets. Spread over 9 zones and
  7 groups, many situations have under 5 shots and produce no tendency at all.
- Set 2 is a single, small follow-up check, and the rally-ending families have few evaluable tendencies.
- Pros adapt between sets, and a club opponent may behave differently. The reference is pro play, so on my
  footage a tendency means "more than pros do", not "more than club players do".
- `player` labels on one-shot `unknown` rallies are unreliable, so those rallies (and the 2 known A/B-swapped
  rallies) are left out of per-player stats.
