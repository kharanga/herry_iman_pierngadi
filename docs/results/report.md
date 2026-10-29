# Phone report: results

## How to run
```
.venv/bin/python -m report.serve match_tracked --me me
.venv/bin/python -m report.serve <shuttleset match_id> [--me A|B] [--no-serve]
.venv/bin/python -m report.preview data/reports/<match_id>/index.html
```
The first command writes `data/reports/match_tracked/index.html`, serves it and prints
`Open on your phone: http://192.168.x.x:8765/`. How the page is built: `docs/analytics/report.md`.

## What it looks like
Screenshots at 390 px (the width of an iPhone 14/15 screen), in `data/checks/report/`:
- `match_tracked_390px.png`: my tracked match, me = `me`.
- `Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals_390px.png`: a ShuttleSet match, me = Axelsen (A).

The page, top to bottom: title "<opponent>: shot counts", the match and the sets; **Opponent** with every one of
their habits; **Me** with every one of mine; a footer saying what "k of n" means, that every choice is listed with no
comparison or ranking, and whether shot types are labels or predictions. Inside each player, bullets sit under one
sub-heading per family ("Shots by court zone", "Replies to the other player's shot", "Serves", "Winning shots",
"Shots that ended in an error", "How errors happened"). Each bullet is the counted sentence in bold
("From the front court, forehand side: lifts 6 of 7 times."), the evidence line (`6 of 7 for Opponent`) and, for
zone and serve bullets, a 54 px court diagram with the zone shaded (me at the bottom). Reply bullets name the other
player ("After a clear from my mid backhand to the rear backhand: smashes straight 1 of 2 times.") and their diagram
shows that player's shot: the hit-from zone in yellow, the landing zone in red and an arrow between them, still with
me at the bottom. Bullets whose shot types
came from the classifier carry a "predicted shot types" tag.

For `match_tracked`: 63 bullets for the opponent and 68 for me, under
Shots by court zone, Replies to the other player's shot, Serves (opponent) and Shots by court zone, Replies to the other player's shot, Serves (me);
174.1 KB of HTML; 131 bullets from predictions only, 0 mixed.

## Checks

| Check | Result |
|---|---|
| Builds for my tracked match | `report.serve match_tracked --me me --no-serve` writes the page |
| Works for every ShuttleSet match | 88 pages built (44 matches x me = A and me = B) without error, 17.2 s in one process |
| Every habit shown, no cap | each page lists every entry of both players: 259 to 658 bullets a page (median 492) |
| Every bullet has a count | all 43,457 bullets (ShuttleSet pages + `match_tracked`) have `evidence` matching `k of n`, and it appears in the HTML (0 without) |
| No comparison with pros | 0 of 89 pages mention pros (0 found) |
| Diagrams | 6,845 bullets (zone and serve placement) get a shaded zone; 34,373 reply bullets get hit-from and landing zones with an arrow |
| Reply bullets name the other player | 34,373 of 34,373 reply bullets name the player; 0 say "the other player" |
| Page size | ShuttleSet 320 to 851 KB (median 627 KB), nothing external (inline CSS and SVG, no scripts) |
| Only the report is served | the server serves a temporary folder holding a copy of the page and nothing else |

## Readable at 390 px
Checked by rendering the page with headless Chrome inside a 390 px wide `<iframe>` (`report.preview`) and looking at
the screenshot: no text is cut off, no horizontal scrollbar, the diagrams sit beside the text. Bullet text is 17 px,
the smallest text (the tag and the footer) 13 px. The `viewport` meta tag is set. With every habit listed, the page is
long: it is meant to be scrolled, one family at a time.

## Decisions
- Every (situation, choice) from `analytics.tendencies` becomes a bullet, in its plain order (family, situation, most
  times first). Nothing is ranked, filtered, capped or compared with other players.
- No advice lines: the page states what was counted, nothing more.
- No LLM. `report.build.rewrite_bullets` stays as the seam; a later pass may reword `text` only.
- Diagrams where a zone is the point: hit-zone and serve-placement bullets (one zone) and replies (the other
  player's shot, from and to, with an arrow). With me at the bottom on every diagram, my replies show the other
  player's shot coming from the far half to the near half; the opponent's replies show my shot going the other way.
- Handedness for the diagrams comes from `LEFT_HANDED_PLAYER_NAMES` (Marin, Momota); own footage assumes
  right-handed players.

## Limits
- ShuttleSet pages are long (658 bullets at most): every rare choice is listed. Ranking or filtering can
  come back as its own task.
- The server is Python's `http.server`: fine for one phone on the home network, not for the internet.
- The page is static: re-run the command after new shots are loaded.
