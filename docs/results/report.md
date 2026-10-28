# Phone report: results

## How to run
```
.venv/bin/python -m report.serve Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m report.serve <match_id> --me B --no-serve
.venv/bin/python -m report.preview data/reports/<match_id>/index.html
```
The first command writes `data/reports/<match_id>/index.html`, serves it and prints
`Open on your phone: http://192.168.x.x:8765/`. How the page is built: `docs/analytics/report.md`.

## What it looks like
Screenshots at 390 px (the width of an iPhone 14/15 screen), in `data/checks/report/`:
- `Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals_390px.png`: me = Axelsen (A, default), opponent Ginting.
- `..._me_B_390px.png`: the same match with me = Ginting.

The page, top to bottom: title "Anthony Sinisuka GINTING scouting report", the match and the sets; **Opponent**
with 3 bullets; **Me** with 3 bullets; a footer explaining "k of n", "pros", the tags and the diagrams.
Each bullet is the engine's sentence in bold, a one-line reading under it, then the evidence line
(`25 of 77 for Anthony Sinisuka GINTING · pros 18%`) and, for zone and serve bullets, a 54 px wide court
diagram on the right with the zone shaded (me at the bottom). Error-based bullets carry a yellow
"weaker evidence" tag. For Axelsen vs Ginting the opponent bullets are:

1. From the mid court, backhand side: drives 25 of 77 times (pros: 18%). "From their mid court, backhand side, expect drives."
2. Serves mid-depth down the middle 3 of 12 times (pros: 4%). "Be ready for a serve mid-depth down the middle."
3. From the front court, forehand side: drives 19 of 63 times (pros: 16%). "From their front court, forehand side, expect drives."

and the Me bullets: errors hit out 21 of 28 (weaker evidence); after a net shot, lifts 88 of 148; from the front
court, backhand side, lifts 42 of 71. Reading the whole page takes well under a minute: 136 words of bullets (70 in the 6 bold sentences,
66 in the 6 reading lines, split on whitespace) plus the footer.

## Checks

| Check | Result |
|---|---|
| Works for every ShuttleSet match | 88 pages built (44 matches x me = A and me = B) without error, 15.3 s in one process (0.17 s a page after the first) |
| At most 3 bullets per section | 528 bullets over 176 sections: every section has exactly 3 (no player has fewer than 3 tendencies over a whole match; the "no clear tendency" text is covered by the doc example) |
| Every bullet has a sample count | all 528 have `evidence` matching `k of n` with n >= 5, and the same `k of n` string appears in the HTML |
| Diagrams | 250 bullets (zone and serve placement) get one; every shaded rectangle lies inside the 6.1 x 13.4 m court |
| Families shown | response 222, zone_mix 174, serve_placement 80, winning_shot 20, error_shot 20, error_reason 12 |
| Shot-type source | 490 bullets from labels, 38 `mixed` (tagged "predicted shot types"), 0 predictions only |
| Page size | 5.6 to 9.3 KB of HTML, nothing external (inline CSS and SVG, no scripts, no fonts, no images) |
| One command, served on the LAN | `report.serve` wrote the page and printed `http://192.168.1.8:8797/`; fetching it returned HTTP 200, 8,465 bytes, in 5 ms |
| Only the report is served | the server serves a temporary folder holding a copy of the page and nothing else: with `--out` into a folder that also held `secret.txt`, `/secret.txt` returned 404 while `/` returned the page |
| Build time for one match | about 1.1 s wall clock for `report.serve --no-serve`, most of it Python start-up and loading the shots table |

## Readable at 390 px
Checked by rendering the page with headless Chrome inside a 390 px wide `<iframe>` (`report.preview`;
Chrome's own window won't go below about 500 px, which is why the frame is needed) and looking at the
screenshot. At 390 px: no text is cut off, no element is wider than the frame (the frame shows no horizontal
scrollbar), the diagrams sit beside the text, the bullet text is 17 px and the smallest text (the tags and the footer) 13 px.
The CSS has no fixed widths except the 54 px diagram; long tokens (match ids with underscores) wrap with
`overflow-wrap: anywhere`. The `viewport` meta tag is set, so a phone renders it at its real width rather than
zoomed out.

## Decisions
- No LLM in v1. `report.build.rewrite_bullets` is the seam: it receives the bullets and returns them unchanged.
  A later pass may reword `text` and `hint` only; the numbers stay the engine's.
- The "reading" line under each bullet is a fixed template per tendency family, so it never introduces a number.
- Error-based bullets (`error_shot`, `error_reason`) are shown when they reach the top 3, tagged as weaker evidence,
  following the set 1 -> set 2 results in `docs/results/tendencies.md`.
- Diagrams only where the zone is the point: hit-zone and serve-placement bullets. Response and rally-ending
  bullets are text only.
- Handedness for the diagrams comes from `LEFT_HANDED_PLAYER_NAMES` (Marin, Momota); own footage assumes
  right-handed players.

## Limits
- The server is Python's `http.server`: fine for one phone on the home network, not for the internet. It serves a
  temporary folder containing only the report, whatever `--out` points to.
- The page is static: re-run the command after new shots are loaded.
- On ShuttleSet, 490 of the 528 bullets use labelled shot types and 38 are tagged "predicted shot types": those
  involve a shot typed `unknown` in ShuttleSet, which got the classifier's group. On own footage every bullet will
  carry that tag.
