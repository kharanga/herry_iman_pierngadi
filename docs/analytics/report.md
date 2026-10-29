# `report/`: the phone report

One page per match: every habit of the opponent and every habit of mine, as plain counts from each
player's own shots ("After a net shot: lifts 7 of 9 times."), grouped by family, served on the local
network so it opens on my phone. No comparison, no ranking, no cap on the number of bullets. The numbers
all come from `analytics.tendencies`; the page only arranges them.

- `report/build.py`: picks the players, runs the tendency engine, turns every entry into a bullet, groups them by family.
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

- Header: "<opponent>: shot counts", the match (tournament, round, year for ShuttleSet; the recording
  id otherwise), who is who, and the sets used.
- **Opponent** and **Me**: every entry from `analytics.tendencies` for that player, in its plain order, under
  one sub-heading per family (`FAMILY_HEADINGS`: "Shots by court zone", "Replies to the other player's shot",
  "Serves", "Winning shots", "Shots that ended in an error", "How errors happened"). Each bullet has the
  engine's sentence, the evidence line (`k of n` for the player) and, for zone and serve-placement bullets, a court
  diagram with the zone shaded. Reply bullets name the other player instead of "the other player" ("my" when that
  is me, else "<name>'s") and get a diagram of that player's shot: the zone it was hit from (yellow), the zone it
  landed in (red) and an arrow between them. Winners and errors have no zone, so no diagram.
- Tag: `predicted shot types` on bullets whose shot types came from the classifier rather than labels
  (`label_source` not `labels`).
- A player with no counted shots gets "No shots counted for this player yet." instead of nothing.
- Footer: what "k of n" means, that every choice is listed with no comparison or ranking, whether shot types are
  labels or predictions, and how to read the diagrams.
- Court diagrams always show **me at the bottom** and the opponent at the top, whichever player the bullet
  is about. Forehand and backhand sides follow the player's handedness (`LEFT_HANDED_PLAYER_NAMES` from
  `analytics.build_shots`, or `left_handed_names`); own footage assumes right-handed unless `me` / `opp`
  is passed in that set.

**Rewording seam.** `report.build.rewrite_bullets(bullets)` is called on every section's bullets before
rendering and currently returns them unchanged. A later LLM pass would replace only `text`;
`evidence`, `k`, `n`, `rate` and the tag come from the engine and must stay as they are.

## `report/build.py`

### `build_report(match_id, me=None, database_path=DATABASE_PATH, sets=None, left_handed_names=LEFT_HANDED_PLAYER_NAMES)`

Returns the report as a `dict`:

| Key | Meaning |
|---|---|
| `match_id`, `title` | title is "tournament round, year" for ShuttleSet, else the match id |
| `sets` | the sets asked for, or `None` for the whole match |
| `opponent`, `me` | sections: `player`, `name`, `bullets` (all, in plain order), `groups` (`[{family, heading, bullets}]`, families in `FAMILIES` order, empty families left out), `sets` (sets with evidence) |

Each bullet: `id`, `family`, `text` (the engine's sentence), `evidence` (`"k of n"`), `k`, `n`,
`rate`, `label_source`, `reply_rects` (for replies: `(hit-from rectangle, landing rectangle)`, me at the bottom;
else `None`), `zone_rect` (court-meter rectangle to
shade, me at the bottom, or `None`), `sets`.

Errors: `UnknownMatchError` (no shots for that match), `UnknownPlayerError` (`--me` not in the match;
both are `ValueError`s with the details as attributes), `ValueError` if the match has other than 2 players,
`duckdb.IOException` / `CatalogException` for a missing database or table.

### Helpers
- `match_players(connection, match_id)`: players in the match, sorted; raises `UnknownMatchError`.
- `match_title(connection, match_id)`: `(title, {player: display name})`; own matches (not in `matches`, or a database
  with no `matches` table at all) get the match id as title and `Me` / `Opponent` as names. `has_matches_table(connection)`.
- `default_me(players)`: `me` if present, else `A`, else the first player. `opponent_of(me, players)`.
- `bullet_from_tendency(tendency, hitter_is_me, me_left_handed, opponent_left_handed, other_player_name="the other player")`,
  `evidence_text(tendency)`, `named_text(text, other_player_name, other_player_is_me)`, `other_player_possessive(...)`
  (`my` or `<name>'s`), `OTHER_PLAYER_PHRASE`.
- `player_section(...)`, `family_groups(bullets)`, `rewrite_bullets(bullets)` (the seam above).
- Constants: `OWN_PLAYER_NAMES`, `FAMILY_HEADINGS`, `MATCH_SQL`, `HAS_MATCHES_TABLE_SQL`, `MATCH_PLAYERS_SQL`.

```python
>>> from report.build import *
>>> tendency = {"family": "response", "condition": "lift|front_forehand|rear_backhand", "outcome": "drop cross", "k": 4, "n": 9,
...             "rate": 4 / 9, "id": "response:lift|front_forehand|rear_backhand:drop cross", "label_source": "labels",
...             "sets": [1, 2, 3], "text": "After a lift from the other player's front forehand to the rear backhand: drops cross 4 of 9 times."}
>>> bullet = bullet_from_tendency(tendency, hitter_is_me=False, me_left_handed=False, opponent_left_handed=False, other_player_name="Me")
>>> bullet["text"]
'After a lift from my front forehand to the rear backhand: drops cross 4 of 9 times.'
>>> bullet_from_tendency(tendency, hitter_is_me=True, me_left_handed=False, opponent_left_handed=False, other_player_name="Viktor AXELSEN")["text"]
"After a lift from Viktor AXELSEN's front forehand to the rear backhand: drops cross 4 of 9 times."
>>> bullet["evidence"], bullet["zone_rect"], sorted(bullet)
('4 of 9', None, ['evidence', 'family', 'id', 'k', 'label_source', 'n', 'rate', 'reply_rects', 'sets', 'text', 'zone_rect'])
>>> [tuple(round(value, 2) for value in rect) for rect in bullet["reply_rects"]]
[(3.55, 4.47, 6.1, 6.7), (3.55, 11.17, 6.1, 13.4)]
>>> [(group["family"], len(group["bullets"])) for group in family_groups([bullet, dict(bullet, family="zone_mix"), bullet])]
[('zone_mix', 1), ('response', 2)]
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
- `reply_zone_rects(hitter_is_me, condition, me_left_handed, opponent_left_handed)`: for a reply, the other player's
  hit zone (on their half) and the replier's landing zone (on the replier's half), both with **me at the bottom**: for
  my replies the hit-from zone is on the far half and the landing zone on the near half; for the opponent's replies
  it is the same picture seen from my end.
- `rect_centre(rect)`, `rect_svg(rect, css_class)`, `arrowhead_points(from_point, to_point)`, `arrow_svg(from_rect, to_rect)`:
  the arrow is a line between the two zone centres plus a small triangle (no shared SVG ids, so many diagrams can sit on one page).
- `court_svg(shaded_rect, label, reply_rects=None)`: the SVG string, `SVG_WIDTH_PX` (54) wide, with the court lines from
  `analytics.spot_check.court_lines_m`, the net, the shaded rectangle, or for replies the two zones and the arrow.

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
>>> bullet_zone_rect(True, "response", "drop cross") is None, bullet_zone_rect(True, "serve_placement", "not_over_net") is None
(True, True)
>>> svg = court_svg(bullet_zone_rect(True, "zone_mix", "front_forehand"), "example")
>>> svg.startswith('<svg class="court"'), svg.count("<line") , '<rect x="35.5" y="67" width="25.5" height="22.3" class="zone"/>' in svg
(True, 13, True)
>>> [tuple(round(value, 2) for value in rect) for rect in reply_zone_rects(True, "net|front_forehand|front_backhand")]
[(0.0, 6.7, 2.55, 8.93), (0.0, 4.47, 2.55, 6.7)]
>>> reply_svg = court_svg(None, "example", reply_zone_rects(True, "net|front_forehand|front_backhand"))
>>> reply_svg.count('class="zone from"'), reply_svg.count('class="zone"'), reply_svg.count("<polygon"), "id=" in reply_svg
(1, 1, 1, False)

```

## `report/page.py`

- `render_page(report)`: the full HTML page (`str`) for a `build_report` dict.
- `section_html(section, heading, lead)`, `group_html(group, hitter_name)`, `bullet_html(bullet, hitter_name)`, `tags_html(bullet)`,
  `footer_html(report)`, `sets_text(sets)`. All text is HTML-escaped. `PAGE_CSS` is the stylesheet.

```python
>>> from report.page import sets_text, tags_html
>>> sets_text(None), sets_text([1]), sets_text([1, 2])
('whole match', 'set 1', 'set 1, 2')
>>> tags_html({"label_source": "predictions"}), tags_html({"label_source": "labels"})
('<span class="tag predicted">predicted shot types</span>', '')

```

## `report/serve.py`

- `write_report(match_id, me=None, output_path=None, database_path=DATABASE_PATH, sets=None)`: builds and
  writes the page, returns its path. `report_path_for(match_id)` is the default path.
- `serve_page(page_path, port=8765)`: copies the page into a fresh temporary folder as `index.html` and serves
  that folder (nothing else) with the standard library's `http.server` on all interfaces until Ctrl-C; the folder is
  removed afterwards. `serve_directory(directory, port)` is the underlying server. `local_network_ip()` is the address printed (`127.0.0.1` if offline).
- `main(arguments)`: the command; returns the exit code.

## Example on a small synthetic table

My match, where the opponent drops 8 times and clears twice from the back court and wins every rally; two other
matches in the same table don't count, since every player is counted only from their own shots:

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
>>> [(group["heading"], [bullet["evidence"] for bullet in group["bullets"]]) for group in report["opponent"]["groups"]]
[('Shots by court zone', ['8 of 10', '2 of 10']), ("Replies to the other player's shot", ['8 of 10', '2 of 10']), ('Winning shots', ['8 of 10', '2 of 10'])]
>>> [bullet["text"] for bullet in report["me"]["bullets"]]
['Serves short down the middle 10 of 10 times.']
>>> page = render_page(report)
>>> page.count("<li>"), page.count("<h3>"), "2 of 10" in page, "<script" in page
(7, 4, True, False)
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
