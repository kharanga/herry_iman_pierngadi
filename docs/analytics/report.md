# `report/`: the phone report

One page per match: the opponent's top tendencies and my own, at most 3 bullets each, every
bullet with its "k of n" evidence and the pros' rate, served on the local network so it opens on
my phone. The numbers all come from `analytics.tendencies`; the page only arranges them.

- `report/build.py`: picks the players, runs the tendency engine, turns the top tendencies into bullets.
- `report/page.py`: renders the bullets as one self-contained HTML page (inline CSS, no scripts, no external files).
- `report/court_svg.py`: the tiny inline-SVG court with one zone shaded.
- `report/serve.py`: the command: write the page, serve it, print the URL.
- `report/preview.py`: renders the page at phone width with headless Chrome, for checking the layout.

What the page looks like and how it was checked: `docs/results/report.md`.

## Command

```
.venv/bin/python -m report.serve <match_id> [--me A|B|me] [--sets 1,2] [--out file.html] [--port 8765] [--no-serve] [--database path]
```

- Writes `data/reports/<match_id>/index.html` (or `--out`) and, unless `--no-serve`, serves a copy of that one file
  (as `index.html` in a temporary folder holding nothing else, so no other file on the Mac is exposed, whatever
  `--out` points to) on all interfaces and prints `Open on your phone: http://<this Mac's LAN address>:8765/`. Ctrl-C stops it.
- `--me`: which player I am. ShuttleSet matches have `A` (the match winner) and `B`; default `A`. Own footage
  has `me` and `opp`; default `me`. The other player is the opponent.
- `--sets`: limit the evidence to those sets (e.g. `1` for a between-sets report later).
- Exit 1 with `error: ...` for an unknown match or a player not in the match.

```
.venv/bin/python -m report.preview <report.html> [output.png]
```
Screenshots the page inside a 390 px wide frame with headless Chrome (`CHROME_PATH`, the Mac's Chrome app).
Default output: `data/checks/report/<report folder name>_390px.png`. Raises `FileNotFoundError` without Chrome.

## The page

- Header: "<opponent> scouting report", the match (tournament, round, year for ShuttleSet; the recording
  id otherwise), who is who, and the sets used.
- **Opponent**: their top 3 tendencies, each with the engine's sentence, a one-line reading of it
  ("After your lifts, expect drops."), the evidence (`k of n`), the pros' rate, and a court diagram
  for zone and serve-placement bullets.
- **Me**: my top 3, read as habits ("you play lifts more than pros do: an opponent can read it").
- Tags: `weaker evidence` on `error_shot` / `error_reason` bullets (they carry over between sets least,
  see `docs/results/tendencies.md`); `predicted shot types` on bullets whose shot types came from the
  classifier rather than labels (`label_source` not `labels`).
- A section with no tendency says so instead of showing nothing.
- Footer: what "k of n", "pros", the tags and the diagrams mean, and whether shot types are labels or predictions.
- Court diagrams always show **me at the bottom** and the opponent at the top, whichever player the bullet
  is about. Forehand and backhand sides follow the player's handedness (`LEFT_HANDED_PLAYER_NAMES` from
  `analytics.build_shots`, or `left_handed_names`); own footage assumes right-handed unless `me` / `opp`
  is passed in that set.

**Rewording seam.** `report.build.rewrite_bullets(bullets)` is called on every section's bullets before
rendering and currently returns them unchanged. A later LLM pass would replace only `text` and `hint`;
`evidence`, `k`, `n`, `rate`, `pros_rate` and the tags come from the engine and must stay as they are.

## `report/build.py`

### `build_report(match_id, me=None, database_path=DATABASE_PATH, sets=None, bullet_count=3, left_handed_names=LEFT_HANDED_PLAYER_NAMES)`

Returns the report as a `dict`:

| Key | Meaning |
|---|---|
| `match_id`, `title` | title is "tournament round, year" for ShuttleSet, else the match id |
| `sets` | the sets asked for, or `None` for the whole match |
| `opponent`, `me` | sections: `player`, `name`, `bullets`, `tendency_count` (before the cut to 3), `sets` (sets with evidence) |

Each bullet: `id`, `family`, `text` (the engine's sentence), `hint`, `evidence` (`"k of n"`), `k`, `n`,
`rate`, `pros_rate`, `label_source`, `weaker_evidence` (bool), `zone_rect` (court-meter rectangle to
shade, me at the bottom, or `None`), `sets`.

Errors: `UnknownMatchError` (no shots for that match), `UnknownPlayerError` (`--me` not in the match;
both are `ValueError`s with the details as attributes), `ValueError` if the match has other than 2 players,
`duckdb.IOException` / `CatalogException` for a missing database or table.

### Helpers
- `match_players(connection, match_id)`: players in the match, sorted; raises `UnknownMatchError`.
- `match_title(connection, match_id)`: `(title, {player: display name})`; own matches (not in `matches`, or a database
  with no `matches` table at all) get the match id as title and `Me` / `Opponent` as names. `has_matches_table(connection)`.
- `default_me(players)`: `me` if present, else `A`, else the first player. `opponent_of(me, players)`.
- `opponent_hint(tendency)`, `me_hint(tendency)`: the one-line readings, by family.
- `bullet_from_tendency(tendency, hitter_is_me, me_left_handed, opponent_left_handed)`, `evidence_text(tendency)`.
- `player_section(...)`, `rewrite_bullets(bullets)` (the seam above).
- Constants: `REPORT_BULLET_COUNT = 3`, `WEAKER_EVIDENCE_FAMILIES`, `OWN_PLAYER_NAMES`, `MATCH_SQL`, `MATCH_PLAYERS_SQL`.

```python
>>> from report.build import *
>>> tendency = {"family": "response", "condition": "lift", "outcome": "drop", "k": 43, "n": 98, "rate": 43 / 98,
...             "reference_rate": 0.31, "id": "response:lift:drop", "label_source": "labels", "sets": [1, 2, 3],
...             "text": "After a lift: drops 43 of 98 times (pros: 31%)."}
>>> opponent_hint(tendency), me_hint(tendency)
('After your lifts, expect drops.', 'After their lifts, you play drops more than pros do: vary it.')
>>> opponent_hint({"family": "serve_placement", "condition": "serve", "outcome": "rear_backhand"})
'Be ready for a serve long to your backhand.'
>>> bullet = bullet_from_tendency(tendency, hitter_is_me=False, me_left_handed=False, opponent_left_handed=False)
>>> bullet["evidence"], bullet["weaker_evidence"], bullet["zone_rect"]
('43 of 98', False, None)
>>> default_me(["A", "B"]), default_me(["me", "opp"]), opponent_of("B", ["A", "B"])
('A', 'me', 'A')

```

## `report/court_svg.py`

- `zone_rect_in_near_half(zone, left_handed=False)`: `(x0, y0, x1, y1)` in court meters of a zone on the
  near half, from the near player's view (forehand on the right for a right-hander).
- `rotate_rect_half_turn(rect)`: the same zone seen from the other end.
- `bullet_zone_rect(hitter_is_me, family, zone, me_left_handed, opponent_left_handed)`: the rectangle to
  shade with **me at the bottom**: the hitter's zone for `zone_mix`, the receiver's zone for
  `serve_placement`; `None` for other families and for `not_over_net`.
- `court_svg(shaded_rect, label)`: the SVG string, `SVG_WIDTH_PX` (54) wide, with the court lines from
  `analytics.spot_check.court_lines_m`, the net, and the shaded rectangle.

```python
>>> from report.court_svg import *
>>> tuple(round(value, 2) for value in zone_rect_in_near_half("rear_backhand"))
(0.0, 0.0, 2.55, 2.23)
>>> tuple(round(value, 2) for value in zone_rect_in_near_half("rear_backhand", left_handed=True))
(3.55, 0.0, 6.1, 2.23)
>>> tuple(round(value, 2) for value in bullet_zone_rect(False, "zone_mix", "rear_backhand"))
(3.55, 11.17, 6.1, 13.4)
>>> tuple(round(value, 2) for value in bullet_zone_rect(False, "serve_placement", "front_middle"))
(2.55, 4.47, 3.55, 6.7)
>>> bullet_zone_rect(True, "response", "drop") is None, bullet_zone_rect(True, "serve_placement", "not_over_net") is None
(True, True)
>>> svg = court_svg(bullet_zone_rect(True, "zone_mix", "front_forehand"), "example")
>>> svg.startswith('<svg class="court"'), svg.count("<line") , '<rect x="35.5" y="67" width="25.5" height="22.3" class="zone"/>' in svg
(True, 13, True)

```

## `report/page.py`

- `render_page(report)`: the full HTML page (`str`) for a `build_report` dict.
- `section_html(section, heading, lead)`, `bullet_html(bullet, hitter_name)`, `tags_html(bullet)`,
  `footer_html(report)`, `sets_text(sets)`. All text is HTML-escaped. `PAGE_CSS` is the stylesheet.

```python
>>> from report.page import sets_text, tags_html
>>> sets_text(None), sets_text([1]), sets_text([1, 2])
('whole match', 'set 1', 'set 1, 2')
>>> tags_html({"label_source": "predictions", "weaker_evidence": True})
'<span class="tag predicted">predicted shot types</span><span class="tag weak">weaker evidence</span>'

```

## `report/serve.py`

- `write_report(match_id, me=None, output_path=None, database_path=DATABASE_PATH, sets=None)`: builds and
  writes the page, returns its path. `report_path_for(match_id)` is the default path.
- `serve_page(page_path, port=8765)`: copies the page into a fresh temporary folder as `index.html` and serves
  that folder (nothing else) with the standard library's `http.server` on all interfaces until Ctrl-C; the folder is
  removed afterwards. `serve_directory(directory, port)` is the underlying server. `local_network_ip()` is the address printed (`127.0.0.1` if offline).
- `main(arguments)`: the command; returns the exit code.

## Example on a small synthetic table

Two pro matches where the back-court player mostly clears, and my match where the opponent drops 8 of 10:

```python
>>> import shutil, tempfile
>>> from pathlib import Path
>>> import duckdb
>>> from analytics.build_shots import SCHEMA_PATH
>>> from report.build import build_report
>>> from report.page import render_page
>>> from report.serve import write_report
>>> temp_dir = Path(tempfile.mkdtemp())
>>> database_path = temp_dir / "scout.duckdb"
>>> connection = duckdb.connect(str(database_path))
>>> _ = connection.execute(SCHEMA_PATH.read_text())
>>> def rally_rows(source, match_id, rally, hitter, receiver, hitter_shot, winner):
...     serve = (source, match_id, 1, rally, 1, rally * 100, receiver, "short service", "serve", "front_middle", "front_middle", None, None)
...     reply = (source, match_id, 1, rally, 2, rally * 100 + 30, hitter, hitter_shot, hitter_shot, "rear_backhand", "front_forehand",
...              winner, "out" if winner == receiver else "opponent winner")
...     return [serve, reply]
>>> rows = []
>>> for match_id in ("pro_match_1", "pro_match_2"):
...     for rally in range(1, 21):
...         rows += rally_rows("shuttleset", match_id, rally, "B", "A", "drop" if rally % 5 == 0 else "clear", "A")
>>> for rally in range(1, 11):
...     rows += rally_rows("own", "club_2026_10_20", rally, "opp", "me", "drop" if rally <= 8 else "clear", "opp")
>>> _ = connection.executemany(
...     "INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, player, type_en, type_group, hit_zone, "
...     "landing_zone, getpoint_player, lose_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
>>> connection.close()
>>> report = build_report("club_2026_10_20", database_path=database_path)
>>> report["title"], report["me"]["name"], report["opponent"]["name"], report["sets"]
('club_2026_10_20', 'Me', 'Opponent', None)
>>> [(bullet["evidence"], bullet["hint"]) for bullet in report["opponent"]["bullets"]]
[('8 of 10', 'After your serves, expect drops.'), ('8 of 10', 'From their back court, backhand side, expect drops.')]
>>> report["me"]["bullets"], report["me"]["tendency_count"]
([], 0)
>>> page = render_page(report)
>>> page.count("<li>"), "8 of 10" in page, "No clear tendency yet" in page, "<script" in page
(2, True, True, False)
>>> output_path = write_report("club_2026_10_20", output_path=temp_dir / "report" / "index.html", database_path=database_path)
>>> output_path.read_text(encoding="utf-8") == page
True
>>> from report.build import UnknownPlayerError
>>> try:
...     build_report("club_2026_10_20", me="A", database_path=database_path)
... except UnknownPlayerError as error:
...     print(error)
player 'A' is not in match 'club_2026_10_20'; players are ['me', 'opp']
>>> shutil.rmtree(temp_dir)

```
