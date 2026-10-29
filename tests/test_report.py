import json
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analytics.build_shots import SCHEMA_PATH
from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, ZONE_NAMES
from analytics.tendencies import DATABASE_PATH, FAMILIES, all_observations, tendencies_from_observations
from report.build import (
    FAMILY_HEADINGS,
    UnknownMatchError,
    UnknownPlayerError,
    build_report,
    bullet_from_tendency,
    default_me,
    family_groups,
    opponent_of,
    rewrite_bullets,
)
from report.court_svg import DIAGRAM_FAMILIES, SVG_WIDTH_PX, bullet_zone_rect, court_svg, rotate_rect_half_turn, zone_rect_in_near_half
from report.page import PAGE_CSS, render_page
from report.preview import CHROME_PATH, PHONE_WIDTH_PX, PREVIEW_HEIGHT_PX, WINDOW_PADDING_PX, screenshot_at_width
from report.serve import main, write_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "report.md"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
TRACKED_MATCH_ID = "match_tracked"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
EVIDENCE_PATTERN = r"\d+ of \d+"
PROS_PATTERN = r"(?i)\bpros?\b|professional|more than|weaker evidence|class=\"hint\""
BULLET_KEYS = {"id", "family", "text", "evidence", "k", "n", "rate", "label_source", "zone_rect", "sets"}
SECTION_KEYS = {"player", "name", "bullets", "groups", "sets"}
EMPTY_SECTION_TEXT = "No shots counted for this player yet."
VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
HOSTILE_MATCH_ID = 'club<game>&"1'
HOSTILE_WINNER = "Ann <A> & Co"
HOSTILE_LOSER = 'Bob "B"'
HOSTILE_TOURNAMENT = "Open <Cup> & Co"
OVERFLOW_HARNESS = """<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">
<iframe id="page" src="%s" style="width:%dpx;height:1500px;border:0;display:block"></iframe><pre id="out"></pre>
<script>
document.getElementById('page').addEventListener('load', function () {
  var frameDocument = this.contentDocument, frameWindow = this.contentWindow, width = %d;
  var result = {scrollWidth: frameDocument.documentElement.scrollWidth, clientWidth: frameDocument.documentElement.clientWidth,
                viewport: !!frameDocument.querySelector('meta[name=viewport]'), wide: [], clipped: [], scripts: frameDocument.scripts.length};
  frameDocument.querySelectorAll('body *').forEach(function (element) {
    var box = element.getBoundingClientRect(), style = frameWindow.getComputedStyle(element);
    if (box.right > width + 0.5 || box.left < -0.5) result.wide.push(element.tagName + ' ' + box.left + '-' + box.right);
    if (element.children.length === 0 && element.textContent.trim() && element.scrollWidth > element.clientWidth + 1 && style.overflow !== 'visible')
      result.clipped.push(element.tagName + ' ' + element.textContent.slice(0, 40));
  });
  document.getElementById('out').textContent = 'RESULT' + JSON.stringify(result) + 'END';
});
</script></body></html>"""


class PageStructure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []
        self.texts = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend((tag, key, value) for key, value in attrs)

    def handle_data(self, data):
        self.texts.append(data)


def parsed(page: str) -> PageStructure:
    structure = PageStructure()
    structure.feed(page)
    return structure


def shot_row(source: str, match_id: str, rally: int, ball_round: int, player: str, **columns) -> dict:
    return {"source": source, "match_id": match_id, "set_no": 1, "rally": rally, "ball_round": ball_round, "frame_num": 100 * rally + ball_round, "player": player, **columns}


def pro_rally(match_id: str, rally: int) -> list[dict]:
    reply_group = "clear" if rally % 2 else "lift"
    return [
        shot_row("shuttleset", match_id, rally, 1, "A", type_en="short service", type_group="serve", landing_zone="front_middle"),
        shot_row("shuttleset", match_id, rally, 2, "B", type_en=reply_group, type_group=reply_group, hit_zone="rear_forehand", getpoint_player="A", lose_reason="out"),
    ]


def own_rally(match_id: str, rally: int) -> list[dict]:
    return [
        shot_row("own", match_id, rally, 1, "opp", type_pred="serve", landing_zone="front_forehand"),
        shot_row("own", match_id, rally, 2, "me", type_pred="drop", hit_zone="rear_forehand", getpoint_player="opp", lose_reason="out"),
    ]


def labelled_rally(match_id: str, rally: int) -> list[dict]:
    reply_group = "clear" if rally == 20 else "drop"
    winner = "B" if rally % 5 == 0 else "A"
    return [
        shot_row("shuttleset", match_id, rally, 1, "A", type_en="short service", type_group="serve", landing_zone="rear_backhand" if rally % 4 == 0 else "front_forehand"),
        shot_row("shuttleset", match_id, rally, 2, "B", type_en=reply_group, type_group=reply_group, hit_zone="rear_forehand", getpoint_player=winner,
                 lose_reason="out" if winner == "A" else "opponent winner"),
    ]


def write_database(database_path: Path, rows: list[dict], matches: list[tuple] | None = None) -> Path:
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    frame = pd.DataFrame(rows)
    connection.register("synthetic_rows", frame)
    columns = ", ".join(frame.columns)
    connection.execute(f"INSERT INTO shots ({columns}) SELECT {columns} FROM synthetic_rows")
    if matches is not None:
        connection.execute("CREATE TABLE matches (match_id VARCHAR, tournament VARCHAR, round VARCHAR, year BIGINT, winner VARCHAR, loser VARCHAR)")
        connection.executemany("INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)", matches)
    connection.close()
    return database_path


def pro_rows() -> list[dict]:
    return [row for match_index in range(3) for rally in range(1, 21) for row in pro_rally(f"pro_{match_index}", rally)]


@pytest.fixture(scope="module")
def own_only_database():
    with tempfile.TemporaryDirectory() as temp_dir:
        rows = pro_rows() + [row for rally in range(1, 21) for row in own_rally(HOSTILE_MATCH_ID, rally)]
        yield write_database(Path(temp_dir) / "scout.duckdb", rows)


@pytest.fixture(scope="module")
def labelled_database():
    with tempfile.TemporaryDirectory() as temp_dir:
        rows = pro_rows() + [row for rally in range(1, 21) for row in labelled_rally(HOSTILE_MATCH_ID, rally)]
        matches = [(HOSTILE_MATCH_ID, HOSTILE_TOURNAMENT, "Final", 2021, HOSTILE_WINNER, HOSTILE_LOSER)]
        yield write_database(Path(temp_dir) / "scout.duckdb", rows, matches)


def section_pages(page: str) -> list[str]:
    return re.findall(r"<section>.*?</section>", page)


def group_parts(section_page: str) -> list[tuple[str, list[str]]]:
    return [(unescape(heading), re.findall(r"<li>.*?</li>", body)) for heading, body in re.findall(r'<h3>(.*?)</h3><ol class="bullets">(.*?)</ol>', section_page)]


def assert_section_shows_every_engine_entry_grouped(section: dict, section_page: str, engine: list[dict]) -> None:
    assert set(section) == SECTION_KEYS
    assert len(section["bullets"]) == len(engine)
    for bullet, tendency in zip(section["bullets"], engine, strict=True):
        assert set(bullet) == BULLET_KEYS
        assert (bullet["id"], bullet["family"], bullet["k"], bullet["n"], bullet["rate"], bullet["text"], bullet["label_source"], bullet["sets"]) == (
            tendency["id"], tendency["family"], tendency["k"], tendency["n"], tendency["rate"], tendency["text"], tendency["label_source"], tendency["sets"])
        assert bullet["evidence"] == f"{tendency['k']} of {tendency['n']}"
    present = [family for family in FAMILIES if any(tendency["family"] == family for tendency in engine)]
    assert [group["family"] for group in section["groups"]] == present
    assert [group["heading"] for group in section["groups"]] == [FAMILY_HEADINGS[family] for family in present]
    assert [bullet for group in section["groups"] for bullet in group["bullets"]] == section["bullets"]
    parts = group_parts(section_page)
    assert [heading for heading, _ in parts] == [group["heading"] for group in section["groups"]]
    assert section_page.count("<h3>") == len(parts) and section_page.count("<li>") == len(engine)
    for (_, items), group in zip(parts, section["groups"], strict=True):
        assert len(items) == len(group["bullets"])
        for item, bullet in zip(items, group["bullets"], strict=True):
            assert f'<p class="claim">{escape(bullet["text"])}</p>' in item
            assert f'<span class="evidence">{bullet["evidence"]}</span> for ' in item
            assert ('<svg class="court"' in item) == (bullet["zone_rect"] is not None)
            assert ("predicted shot types" in item) == (bullet["label_source"] != "labels")


def test_rewrite_bullets_is_the_identity_in_version_one():
    bullets = [{"text": "a", "evidence": "5 of 9", "k": 5, "n": 9}]
    assert rewrite_bullets(bullets) is bullets
    assert rewrite_bullets([]) == []


def test_player_defaults_and_opponent():
    assert default_me(["A", "B"]) == "A"
    assert default_me(["me", "opp"]) == "me"
    assert default_me(["opp", "x"]) == "opp"
    assert opponent_of("A", ["A", "B"]) == "B"
    with pytest.raises(ValueError):
        opponent_of("A", ["A", "B", "C"])


def test_own_footage_only_database_without_a_matches_table(own_only_database):
    report = build_report(HOSTILE_MATCH_ID, database_path=own_only_database)
    assert report["title"] == HOSTILE_MATCH_ID
    assert (report["me"]["player"], report["me"]["name"]) == ("me", "Me")
    assert (report["opponent"]["player"], report["opponent"]["name"]) == ("opp", "Opponent")
    bullets = report["opponent"]["bullets"] + report["me"]["bullets"]
    assert {bullet["label_source"] for bullet in bullets if bullet["family"] != "error_reason"} == {"predictions"}
    assert [bullet["label_source"] for bullet in bullets if bullet["family"] == "error_reason"] == ["labels"]
    assert [bullet["id"] for bullet in report["opponent"]["bullets"]] == ["serve_placement:serve:front_forehand"]
    assert [bullet["id"] for bullet in report["me"]["bullets"]] == ["zone_mix:rear_forehand:drop", "response:serve:drop", "error_shot:error:drop", "error_reason:error:out"]
    page = render_page(report)
    assert page.count("predicted shot types</span>") == len(bullets) - 1
    assert "use the classifier's predicted types; the others use labelled types" in page
    assert build_report(HOSTILE_MATCH_ID, me="opp", database_path=own_only_database)["opponent"]["player"] == "me"


def test_page_shows_every_engine_entry_for_both_players_grouped_by_family(labelled_database):
    observations = all_observations(labelled_database)
    for me in ("A", "B"):
        report = build_report(HOSTILE_MATCH_ID, me=me, database_path=labelled_database)
        page = render_page(report)
        opponent_page, me_page = section_pages(page)
        for section, section_page in ((report["opponent"], opponent_page), (report["me"], me_page)):
            assert_section_shows_every_engine_entry_grouped(section, section_page, tendencies_from_observations(observations, HOSTILE_MATCH_ID, section["player"]))
    report = build_report(HOSTILE_MATCH_ID, me="A", database_path=labelled_database)
    assert [(bullet["id"], bullet["evidence"]) for bullet in report["opponent"]["bullets"]] == [
        ("zone_mix:rear_forehand:drop", "19 of 20"), ("zone_mix:rear_forehand:clear", "1 of 20"),
        ("response:serve:drop", "19 of 20"), ("response:serve:clear", "1 of 20"),
        ("winning_shot:winner:drop", "3 of 4"), ("winning_shot:winner:clear", "1 of 4"),
        ("error_shot:error:drop", "16 of 16"), ("error_reason:error:out", "16 of 16"),
    ]
    assert [(bullet["id"], bullet["evidence"]) for bullet in report["me"]["bullets"]] == [
        ("serve_placement:serve:front_forehand", "15 of 20"), ("serve_placement:serve:rear_backhand", "5 of 20")]
    assert [group["heading"] for group in report["opponent"]["groups"]] == [
        "Shots by court zone", "Replies to the other player's shot", "Winning shots", "Shots that ended in an error", "How errors happened"]
    assert [group["heading"] for group in report["me"]["groups"]] == ["Serves"]


def test_family_groups_follow_the_family_order_and_skip_empty_families():
    bullets = [{"family": family, "id": index} for index, family in enumerate(["error_reason", "zone_mix", "serve_placement", "zone_mix"])]
    assert [(group["family"], group["heading"], [bullet["id"] for bullet in group["bullets"]]) for group in family_groups(bullets)] == [
        ("zone_mix", "Shots by court zone", [1, 3]), ("serve_placement", "Serves", [2]), ("error_reason", "How errors happened", [0])]
    assert family_groups([]) == []
    assert set(FAMILY_HEADINGS) == set(FAMILIES)


def test_a_player_with_no_counted_shots_shows_the_empty_text(labelled_database):
    report = build_report(HOSTILE_MATCH_ID, me="A", database_path=labelled_database, sets=(2,))
    assert report["opponent"]["bullets"] == [] and report["me"]["groups"] == []
    page = render_page(report)
    assert page.count("<h2>") == 2 and page.count("<li>") == 0 and page.count(EMPTY_SECTION_TEXT) == 2 and "<h3>" not in page
    report = build_report(HOSTILE_MATCH_ID, me="A", database_path=labelled_database)
    report["me"]["bullets"], report["me"]["groups"] = [], []
    page = render_page(report)
    assert page.count(EMPTY_SECTION_TEXT) == 1 and page.count("<li>") == 8 and page.count("<h3>") == 5


def test_no_pros_comparison_hint_or_weaker_evidence_remains(labelled_database, own_only_database):
    for database, me in ((labelled_database, "A"), (labelled_database, "B"), (own_only_database, None)):
        report = build_report(HOSTILE_MATCH_ID, me=me, database_path=database)
        page = render_page(report)
        assert not re.search(PROS_PATTERN, page) and "weak" not in page and "hint" not in page
        assert "tendency_count" not in report["opponent"] and "tendency_count" not in report["me"]
        for bullet in report["opponent"]["bullets"] + report["me"]["bullets"]:
            assert not {"pros_rate", "hint", "weaker_evidence", "reference_rate", "score"} & set(bullet)
    assert not re.search(r"(?i)pros|weak|hint", PAGE_CSS)
    build_source = (PROJECT_ROOT / "report" / "build.py").read_text() + (PROJECT_ROOT / "report" / "page.py").read_text()
    for removed in ("REPORT_BULLET_COUNT", "WEAKER_EVIDENCE_FAMILIES", "opponent_hint", "me_hint", "pros_rate", "weaker_evidence", "tendency_count", "bullet_count", "reference_rate"):
        assert removed not in build_source, removed
    page = render_page(build_report(HOSTILE_MATCH_ID, me="B", database_path=labelled_database))
    assert page.count("predicted shot types") == 0 and "labelled types from the match data" in page
    assert "Every choice is listed, however rare, with no comparison and no ranking." in page


def test_hostile_names_and_ids_are_escaped_and_the_page_stays_well_formed(labelled_database, own_only_database):
    labelled = render_page(build_report(HOSTILE_MATCH_ID, me="A", database_path=labelled_database))
    own = render_page(build_report(HOSTILE_MATCH_ID, database_path=own_only_database))
    for page in (labelled, own):
        structure = parsed(page)
        assert "b" not in structure.tags and "a" not in structure.tags
        assert all(value is None or "<" not in value for _, _, value in structure.attributes)
    labelled_text = "".join(parsed(labelled).texts)
    assert HOSTILE_WINNER in labelled_text and HOSTILE_LOSER in labelled_text and HOSTILE_TOURNAMENT in labelled_text
    assert HOSTILE_WINNER not in labelled and HOSTILE_TOURNAMENT not in labelled
    assert f"<title>{HOSTILE_LOSER}: shot counts" not in labelled and "<title>Bob &quot;B&quot;: shot counts</title>" in labelled
    assert HOSTILE_MATCH_ID in "".join(parsed(own).texts) and HOSTILE_MATCH_ID not in own


def test_pages_are_self_contained_with_a_viewport_meta(labelled_database, own_only_database):
    for database, me in ((labelled_database, "A"), (labelled_database, "B"), (own_only_database, None)):
        page = render_page(build_report(HOSTILE_MATCH_ID, me=me, database_path=database))
        structure = parsed(page)
        assert not {"script", "link", "img", "iframe", "object"} & set(structure.tags)
        assert not [item for item in structure.attributes if item[1] in ("src", "href", "onload", "onclick")]
        assert not re.search(r"https?://|url\(|@import|<!--|/\*", page)
        assert VIEWPORT_META in page and page.startswith("<!doctype html>") and page.count("<style>") == 1
        assert page.count('<svg class="court"') == sum(
            bullet["zone_rect"] is not None for bullet in build_report(HOSTILE_MATCH_ID, me=me, database_path=database)["opponent"]["bullets"] + build_report(HOSTILE_MATCH_ID, me=me, database_path=database)["me"]["bullets"])


def test_unknown_match_and_player_raise(labelled_database):
    with pytest.raises(UnknownMatchError):
        build_report("nobody_played_this", database_path=labelled_database)
    with pytest.raises(UnknownPlayerError) as error:
        build_report(HOSTILE_MATCH_ID, me="me", database_path=labelled_database)
    assert error.value.players == ["A", "B"]


@pytest.mark.parametrize("zone", ZONE_NAMES)
def test_zone_rect_sides_follow_handedness_and_depth_bands(zone):
    depth, side = zone.split("_")
    right_handed = zone_rect_in_near_half(zone)
    left_handed = zone_rect_in_near_half(zone, left_handed=True)
    band_bottom = {"front": NET_Y_M - NET_Y_M / 3, "mid": NET_Y_M / 3, "rear": 0.0}[depth]
    assert right_handed[1] == pytest.approx(band_bottom) and right_handed[3] == pytest.approx(band_bottom + NET_Y_M / 3)
    if side == "middle":
        assert right_handed == left_handed and right_handed[0] == pytest.approx(2.55) and right_handed[2] == pytest.approx(3.55)
    else:
        expected_right = (3.55, COURT_WIDTH_M) if side == "forehand" else (0.0, 2.55)
        assert (right_handed[0], right_handed[2]) == pytest.approx(expected_right)
        assert (left_handed[0], left_handed[2]) == pytest.approx((COURT_WIDTH_M - expected_right[1], COURT_WIDTH_M - expected_right[0]))


def test_me_at_the_bottom_opponent_at_the_top_whoever_the_zone_belongs_to():
    mine = bullet_zone_rect(True, "zone_mix", "front_forehand")
    theirs = bullet_zone_rect(False, "zone_mix", "front_forehand")
    assert mine == pytest.approx((3.55, NET_Y_M - NET_Y_M / 3, COURT_WIDTH_M, NET_Y_M))
    assert theirs == pytest.approx((0.0, NET_Y_M, 2.55, NET_Y_M + NET_Y_M / 3))
    assert theirs == pytest.approx(rotate_rect_half_turn(mine))
    assert bullet_zone_rect(False, "zone_mix", "front_forehand", opponent_left_handed=True) == pytest.approx((3.55, NET_Y_M, COURT_WIDTH_M, NET_Y_M + NET_Y_M / 3))
    assert bullet_zone_rect(True, "zone_mix", "front_forehand", me_left_handed=True) == pytest.approx((0.0, NET_Y_M - NET_Y_M / 3, 2.55, NET_Y_M))
    assert bullet_zone_rect(False, "serve_placement", "rear_backhand") == bullet_zone_rect(True, "zone_mix", "rear_backhand")
    assert bullet_zone_rect(True, "serve_placement", "rear_backhand") == bullet_zone_rect(False, "zone_mix", "rear_backhand")
    assert bullet_zone_rect(True, "serve_placement", "rear_backhand", me_left_handed=True) == bullet_zone_rect(True, "serve_placement", "rear_backhand")
    assert bullet_zone_rect(True, "serve_placement", "rear_backhand", opponent_left_handed=True) == bullet_zone_rect(False, "zone_mix", "rear_backhand", opponent_left_handed=True)
    for family in ("response", "winning_shot", "error_shot", "error_reason"):
        assert bullet_zone_rect(True, family, "drop") is None
    assert bullet_zone_rect(False, "serve_placement", "not_over_net") is None


@pytest.mark.parametrize("hitter_is_me", [True, False])
@pytest.mark.parametrize("family", DIAGRAM_FAMILIES)
@pytest.mark.parametrize("zone", ZONE_NAMES)
def test_every_diagram_rectangle_lies_in_the_correct_half_of_the_court(hitter_is_me, family, zone):
    for me_left, opponent_left in ((False, False), (True, False), (False, True), (True, True)):
        x0, y0, x1, y1 = bullet_zone_rect(hitter_is_me, family, zone, me_left, opponent_left)
        assert 0 <= x0 < x1 <= COURT_WIDTH_M + 1e-9 and 0 <= y0 < y1 <= COURT_LENGTH_M + 1e-9
        zone_is_mine = hitter_is_me == (family == "zone_mix")
        assert (y1 <= NET_Y_M + 1e-9) if zone_is_mine else (y0 >= NET_Y_M - 1e-9)


def test_svg_shades_the_rectangle_with_my_baseline_at_the_bottom():
    svg = court_svg(bullet_zone_rect(True, "zone_mix", "rear_forehand"), "label")
    rect = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" class="zone"/>', svg)
    x, y, width, height = (float(value) for value in rect.groups())
    assert (x, width) == (35.5, 25.5) and y + height == pytest.approx(COURT_LENGTH_M * 10) and y == pytest.approx(111.7, abs=0.05)
    far = court_svg(bullet_zone_rect(False, "zone_mix", "rear_forehand"), "label")
    far_rect = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" class="zone"/>', far)
    assert float(far_rect.group(1)) == 0.0 and float(far_rect.group(2)) == 0.0
    assert f'width="{SVG_WIDTH_PX}"' in svg and 'class="net"' in svg and svg.count("<line") == 13
    assert 'class="zone"' not in court_svg(None, "label")


def test_bullet_from_tendency_copies_the_numbers_unchanged():
    tendency = {"id": "zone_mix:front_backhand:lift", "family": "zone_mix", "condition": "front_backhand", "outcome": "lift", "k": 42, "n": 71, "match_id": "m", "player": "A",
                "rate": 42 / 71, "label_source": "mixed", "sets": [1, 2], "text": "From the front court, backhand side: lifts 42 of 71 times."}
    bullet = bullet_from_tendency(tendency, True, False, False)
    assert set(bullet) == BULLET_KEYS
    assert (bullet["k"], bullet["n"], bullet["rate"], bullet["evidence"], bullet["text"], bullet["sets"]) == (42, 71, 42 / 71, "42 of 71", tendency["text"], [1, 2])
    assert bullet["zone_rect"] == bullet_zone_rect(True, "zone_mix", "front_backhand")
    serve = dict(tendency, family="serve_placement", condition="serve", outcome="rear_backhand")
    assert bullet_from_tendency(serve, False, False, False)["zone_rect"] == bullet_zone_rect(False, "serve_placement", "rear_backhand")
    assert bullet_from_tendency(dict(tendency, family="response", condition="net"), True, False, False)["zone_rect"] is None


def free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_no_serve_writes_the_file_and_exits_zero(labelled_database, tmp_path):
    output = tmp_path / "reports" / "m" / "index.html"
    assert main([HOSTILE_MATCH_ID, "--me", "B", "--no-serve", "--out", str(output), "--database", str(labelled_database), "--sets", "1"]) == 0
    assert output.read_text(encoding="utf-8") == render_page(build_report(HOSTILE_MATCH_ID, me="B", database_path=labelled_database, sets=(1,)))
    assert "· set 1<" in output.read_text(encoding="utf-8")


def test_unknown_match_or_player_exit_one_without_writing(labelled_database, tmp_path):
    for arguments in (["nobody", "--database", str(labelled_database)], [HOSTILE_MATCH_ID, "--me", "C", "--database", str(labelled_database)]):
        result = subprocess.run([sys.executable, "-m", "report.serve", *arguments, "--no-serve", "--out", str(tmp_path / "x.html")], cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 1 and result.stderr.startswith("error: ") and not (tmp_path / "x.html").exists()


def fetch(port: int, path: str) -> tuple[int, bytes]:
    request = socket.create_connection(("127.0.0.1", port), timeout=5)
    request.sendall(f"GET {path} HTTP/1.0\r\n\r\n".encode())
    chunks = []
    while chunk := request.recv(65536):
        chunks.append(chunk)
    request.close()
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), body


def served_folders() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("badminton_report_*"))


def test_server_serves_only_a_copy_of_the_report_and_removes_it_on_ctrl_c(labelled_database, tmp_path):
    output = tmp_path / "reports" / "m" / "index.html"
    output.parent.mkdir(parents=True)
    (output.parent / "other.txt").write_text("outside")
    (tmp_path / "reports" / "sibling.txt").write_text("outside")
    (tmp_path / "secret.txt").write_text("outside")
    folders_before = served_folders()
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "report.serve", HOSTILE_MATCH_ID, "--database", str(labelled_database), "--out", str(output), "--port", str(port)],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                status, body = fetch(port, "/")
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not answer")
        assert status == 200 and body == output.read_bytes()
        assert fetch(port, "/index.html")[0] == 200
        for path in ("/other.txt", "/../sibling.txt", "/../../secret.txt", "/%2e%2e/sibling.txt", "/..%2f..%2fsecret.txt", "/missing"):
            status, body = fetch(port, path)
            assert status == 404 and b"outside" not in body
        new_folders = served_folders() - folders_before
        assert len(new_folders) == 1 and [entry.name for entry in next(iter(new_folders)).iterdir()] == ["index.html"]
    finally:
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=10)
    assert process.returncode == 0 and served_folders() - folders_before == set()
    assert f"Report written: {output}" in stdout and re.search(rf"Open on your phone: http://[\d.]+:{port}/", stdout)


def test_text_sizes_match_the_results_doc():
    doc = RESULTS_PATH.read_text()
    tag_rule = re.search(r"\.tag \{[^}]*font-size: (\d+)px", PAGE_CSS).group(1)
    footer_rule = re.search(r"footer \{[^}]*font-size: (\d+)px", PAGE_CSS).group(1)
    body_rule = re.search(r"body \{ font: (\d+)px", PAGE_CSS).group(1)
    smallest = min(int(size) for size in re.findall(r"font-size: (\d+)px", PAGE_CSS))
    assert int(tag_rule) == int(footer_rule) == smallest
    assert f"Bullet text is {body_rule} px,\nthe smallest text (the tag and the footer) {tag_rule} px" in doc


def test_serve_without_arguments_exits_two():
    assert subprocess.run([sys.executable, "-m", "report.serve"], cwd=PROJECT_ROOT, capture_output=True).returncode == 2


chrome_required = pytest.mark.skipif(not CHROME_PATH.is_file(), reason="Chrome is not installed")


@chrome_required
def test_preview_screenshots_the_page_in_a_phone_width_frame(labelled_database, tmp_path):
    from PIL import Image

    page_path = write_report(HOSTILE_MATCH_ID, me="A", output_path=tmp_path / "m" / "index.html", database_path=labelled_database)
    png_path = screenshot_at_width(page_path, tmp_path / "shot.png")
    image = Image.open(png_path).convert("RGB")
    assert image.size == (PHONE_WIDTH_PX + WINDOW_PADDING_PX, PREVIEW_HEIGHT_PX)
    assert image.getpixel((PHONE_WIDTH_PX - 1, 100)) != (136, 136, 136) and image.getpixel((PHONE_WIDTH_PX, 100)) == (136, 136, 136)


def test_preview_without_chrome_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        screenshot_at_width(tmp_path / "page.html", tmp_path / "shot.png", chrome_path=tmp_path / "no-chrome")


def rendered_layout(page_path: Path, width_px: int = PHONE_WIDTH_PX) -> dict:
    with tempfile.TemporaryDirectory() as harness_dir:
        harness = Path(harness_dir) / "harness.html"
        harness.write_text(OVERFLOW_HARNESS % (page_path.resolve().as_uri(), width_px, width_px), encoding="utf-8")
        dom = subprocess.run(
            [str(CHROME_PATH), "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--virtual-time-budget=3000", "--window-size=600,1600", "--dump-dom", harness.resolve().as_uri()],
            capture_output=True, text=True, check=True).stdout
    return json.loads(dom.split("RESULT", 1)[1].split("END", 1)[0])


@chrome_required
def test_page_has_no_horizontal_overflow_or_cut_off_text_at_phone_width(labelled_database, own_only_database, tmp_path):
    for index, (database, me) in enumerate(((labelled_database, "A"), (labelled_database, "B"), (own_only_database, None))):
        page_path = write_report(HOSTILE_MATCH_ID, me=me, output_path=tmp_path / str(index) / "index.html", database_path=database)
        layout = rendered_layout(page_path)
        assert layout["scrollWidth"] <= layout["clientWidth"] <= PHONE_WIDTH_PX
        assert layout["wide"] == [] and layout["clipped"] == [] and layout["viewport"] and layout["scripts"] == 0


@pytest.fixture(scope="module")
def real_pages():
    if not REAL_DATABASE_PATH.exists():
        pytest.skip("data/scout.duckdb is not built")
    connection = duckdb.connect(str(REAL_DATABASE_PATH), read_only=True)
    match_ids = [row[0] for row in connection.execute("SELECT DISTINCT shots.match_id FROM shots WHERE shots.source = 'shuttleset' ORDER BY shots.match_id").fetchall()]
    connection.close()
    reports = {(match_id, me): build_report(match_id, me=me) for match_id in match_ids for me in ("A", "B")}
    return {key: (report, render_page(report)) for key, (report) in reports.items()}


@pytest.fixture(scope="module")
def tracked_page():
    if not REAL_DATABASE_PATH.exists():
        pytest.skip("data/scout.duckdb is not built")
    connection = duckdb.connect(str(REAL_DATABASE_PATH), read_only=True)
    has_tracked = connection.execute("SELECT COUNT(*) FROM shots WHERE shots.match_id = 'match_tracked'").fetchone()[0] > 0
    connection.close()
    if not has_tracked:
        pytest.skip("run vision.features on the tracked match first")
    report = build_report(TRACKED_MATCH_ID, me="me")
    return report, render_page(report)


def page_bullets(report: dict) -> list[dict]:
    return report["opponent"]["bullets"] + report["me"]["bullets"]


def assert_page_shows_every_entry(report: dict, page: str, observations: pd.DataFrame) -> None:
    assert page.count("<h2>") == 2 and VIEWPORT_META in page and "<script" not in page and not re.search(PROS_PATTERN, page)
    for section, section_page in zip((report["opponent"], report["me"]), section_pages(page), strict=True):
        assert_section_shows_every_engine_entry_grouped(section, section_page, tendencies_from_observations(observations, report["match_id"], section["player"]))
    for bullet in page_bullets(report):
        assert re.fullmatch(EVIDENCE_PATTERN, bullet["evidence"])
        if bullet["zone_rect"] is not None:
            x0, y0, x1, y1 = bullet["zone_rect"]
            assert 0 <= x0 < x1 <= COURT_WIDTH_M + 1e-9 and 0 <= y0 < y1 <= COURT_LENGTH_M + 1e-9
    assert page.count('<svg class="court"') == sum(bullet["zone_rect"] is not None for bullet in page_bullets(report))


def test_every_shuttleset_page_shows_every_entry_of_both_players_grouped_by_family(real_pages):
    observations = all_observations(DATABASE_PATH)
    assert len(real_pages) == 88
    for report, page in real_pages.values():
        assert_page_shows_every_entry(report, page, observations)
        assert min(entry["n"] for entry in page_bullets(report)) >= 1
    assert sum(bullet["k"] == 1 for report, _ in real_pages.values() for bullet in page_bullets(report)) > 0


def test_tracked_match_page_shows_every_entry_and_matches_the_shipped_page(tracked_page):
    report, page = tracked_page
    assert_page_shows_every_entry(report, page, all_observations(DATABASE_PATH))
    assert (len(report["opponent"]["bullets"]), len(report["me"]["bullets"])) == (39, 40)
    assert {bullet["label_source"] for bullet in page_bullets(report)} == {"predictions"}
    assert page.count("predicted shot types</span>") == 79 and "<title>Opponent: shot counts</title>" in page
    reference = PROJECT_ROOT / "data" / "reports" / TRACKED_MATCH_ID / "index.html"
    if reference.exists():
        assert reference.read_text(encoding="utf-8") == page


def test_results_doc_counts_match_the_generated_pages(real_pages, tracked_page):
    doc = RESULTS_PATH.read_text()
    tracked_report, tracked_html = tracked_page
    reports = [report for report, _ in real_pages.values()] + [tracked_report]
    pages = [page for _, page in real_pages.values()] + [tracked_html]
    bullets = [bullet for report in reports for bullet in page_bullets(report)]
    per_page = sorted(len(page_bullets(report)) for report, _ in real_pages.values())
    sizes = sorted(len(page.encode()) for _, page in real_pages.values())
    assert "88 pages built (44 matches x me = A and me = B) without error" in doc
    assert f"{per_page[0]} to {per_page[-1]} bullets a page (median {(per_page[43] + per_page[44]) / 2:g})" in doc
    without_count = sum(f'<span class="evidence">{bullet["evidence"]}</span>' not in page for report, page in zip(reports, pages) for bullet in page_bullets(report))
    assert f"all {len(bullets):,} bullets (ShuttleSet pages + `match_tracked`)" in doc and f"in the HTML ({without_count} without)" in doc and without_count == 0
    mentioning = sum(bool(re.search(r"(?i)\bpros?\b", page)) for page in pages)
    assert f"{mentioning} of {len(pages)} pages mention pros ({mentioning} found)" in doc and mentioning == 0
    assert f"{sum(bullet['zone_rect'] is not None for bullet in bullets):,} bullets (zone and serve placement) get one" in doc
    median_size = (sizes[43] + sizes[44]) / 2
    assert f"ShuttleSet {sizes[0] / 1000:.0f} to {sizes[-1] / 1000:.0f} KB (median {median_size / 1000:.0f} KB)" in doc
    assert f"ShuttleSet pages are long ({per_page[-1]} bullets at most)" in doc
    opponent, me = tracked_report["opponent"], tracked_report["me"]
    assert f"For `match_tracked`: {len(opponent['bullets'])} bullets for the opponent and {len(me['bullets'])} for me" in doc
    assert ", ".join(group["heading"] for group in opponent["groups"]) + " (opponent) and " + ", ".join(group["heading"] for group in me["groups"]) + " (me)" in doc
    sources = Counter(bullet["label_source"] for bullet in page_bullets(tracked_report))
    assert f"{len(tracked_html.encode()) / 1000:.1f} KB of HTML; {sources['predictions']} bullets from predictions only, {sources['mixed']} mixed" in doc
    first = opponent["bullets"][0]
    assert f'("{first["text"]}")' in doc and f"(`{first['evidence']} for {opponent['name']}`)" in doc
    assert ", ".join(f'"{heading}"' for heading in FAMILY_HEADINGS.values()) in re.sub(r"\s+", " ", doc)


def test_results_doc_axelsen_ginting_page_matches(real_pages):
    report, page = real_pages[(AXELSEN_GINTING, "A")]
    assert report["opponent"]["name"] == "Anthony Sinisuka GINTING" and "<title>Anthony Sinisuka GINTING: shot counts</title>" in page
    assert [group["heading"] for group in report["opponent"]["groups"]] == list(FAMILY_HEADINGS.values())
    reference = PROJECT_ROOT / "data" / "reports" / AXELSEN_GINTING / "index.html"
    if reference.exists():
        assert reference.read_text(encoding="utf-8") == page


@pytest.mark.parametrize("source_name", ["build.py", "page.py", "court_svg.py", "serve.py", "preview.py", "__init__.py"])
def test_report_sources_have_no_comments_or_docstrings(source_name):
    source = (PROJECT_ROOT / "report" / source_name).read_text()
    triple_quoted_blocks = source.split('"""')[1::2]
    assert all(block.lstrip().startswith(("SELECT", ":root")) for block in triple_quoted_blocks)
    assert not re.search(r"(?m)^\s*#", source) and "'''" not in source
    assert "<!--" not in source and "/*" not in source


@pytest.mark.parametrize("doc_name", ["docs/analytics/report.md", "docs/results/report.md", "docs/analytics/README.md"])
def test_report_docs_have_no_task_ids(doc_name):
    assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc_name).read_text())
