import json
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analytics.build_shots import SCHEMA_PATH
from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, ZONE_NAMES
from analytics.tendencies import DATABASE_PATH, all_observations, tendencies_from_observations
from report.build import (
    REPORT_BULLET_COUNT,
    UnknownMatchError,
    UnknownPlayerError,
    build_report,
    bullet_from_tendency,
    default_me,
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
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
EVIDENCE_PATTERN = r"\d+ of \d+"
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
    return [
        shot_row("shuttleset", match_id, rally, 1, "A", type_en="short service", type_group="serve", landing_zone="front_forehand"),
        shot_row("shuttleset", match_id, rally, 2, "B", type_en="drop", type_group="drop", hit_zone="rear_forehand", getpoint_player="A", lose_reason="out"),
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


def test_rewrite_bullets_is_the_identity_in_version_one():
    bullets = [{"text": "a", "hint": "b", "evidence": "5 of 9", "k": 5, "n": 9}]
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
    assert bullets and {bullet["label_source"] for bullet in bullets} == {"predictions"}
    assert len(report["opponent"]["bullets"]) == 1 and len(report["me"]["bullets"]) == 3
    page = render_page(report)
    assert page.count("predicted shot types") == len(bullets)
    assert "predicted by the classifier" in page
    assert build_report(HOSTILE_MATCH_ID, me="opp", database_path=own_only_database)["opponent"]["player"] == "me"


def test_page_bullets_equal_the_engine_top_three_and_carry_their_evidence(labelled_database):
    observations = all_observations(labelled_database)
    for me in ("A", "B"):
        report = build_report(HOSTILE_MATCH_ID, me=me, database_path=labelled_database)
        page = render_page(report)
        for section in (report["opponent"], report["me"]):
            engine = tendencies_from_observations(observations, HOSTILE_MATCH_ID, section["player"])
            assert section["tendency_count"] == len(engine)
            assert len(section["bullets"]) == min(REPORT_BULLET_COUNT, len(engine))
            for bullet, tendency in zip(section["bullets"], engine, strict=True):
                assert (bullet["id"], bullet["k"], bullet["n"], bullet["rate"], bullet["pros_rate"], bullet["text"], bullet["label_source"]) == (
                    tendency["id"], tendency["k"], tendency["n"], tendency["rate"], tendency["reference_rate"], tendency["text"], tendency["label_source"])
                assert bullet["evidence"] == f"{tendency['k']} of {tendency['n']}" and tendency["n"] >= 5
                assert f'<span class="evidence">{bullet["evidence"]}</span>' in page
                pros_in_text = re.search(r"\(pros: (\d+%)\)", tendency["text"]).group(1)
                assert f" · pros {pros_in_text}<" in page


def test_fewer_than_three_tendencies_shows_fewer_bullets_and_none_shows_the_empty_text(labelled_database):
    report = build_report(HOSTILE_MATCH_ID, me="A", database_path=labelled_database)
    assert len(report["me"]["bullets"]) == 1 and report["me"]["tendency_count"] == 1
    page = render_page(report)
    assert page.count("<h2>") == 2 and page.count("<li>") == 4 and "No clear tendency yet" not in page
    report["me"]["bullets"] = []
    page = render_page(report)
    assert page.count("<h2>") == 2 and page.count("<li>") == 3 and "No clear tendency yet" in page


def test_weaker_evidence_tag_only_on_error_families(labelled_database):
    report = build_report(HOSTILE_MATCH_ID, me="B", database_path=labelled_database)
    bullets = report["opponent"]["bullets"] + report["me"]["bullets"]
    assert {bullet["family"] for bullet in bullets} >= {"error_shot", "zone_mix"}
    for bullet in bullets:
        assert bullet["weaker_evidence"] == (bullet["family"] in ("error_shot", "error_reason"))
    page = render_page(report)
    assert page.count("weaker evidence</span>") == sum(bullet["weaker_evidence"] for bullet in bullets)
    assert page.count("predicted shot types") == 0 and "labelled types from the match data" in page


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
    assert f"<title>{HOSTILE_LOSER} scouting report" not in labelled and "Bob &quot;B&quot; scouting report</title>" in labelled
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
    tendency = {"id": "zone_mix:front_backhand:lift", "family": "zone_mix", "condition": "front_backhand", "outcome": "lift", "k": 42, "n": 71,
                "rate": 42 / 71, "reference_rate": 0.3712, "label_source": "mixed", "sets": [1, 2], "text": "From the front court, backhand side: lifts 42 of 71 times (pros: 37%)."}
    bullet = bullet_from_tendency(tendency, True, False, False)
    assert (bullet["k"], bullet["n"], bullet["rate"], bullet["pros_rate"], bullet["evidence"]) == (42, 71, 42 / 71, 0.3712, "42 of 71")
    assert bullet["zone_rect"] == bullet_zone_rect(True, "zone_mix", "front_backhand") and bullet["weaker_evidence"] is False
    assert "you play lifts" in bullet["hint"] and "42" not in bullet["hint"]


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


def test_tag_font_size_and_bullet_word_count_match_the_results_doc(real_pages):
    doc = RESULTS_PATH.read_text()
    tag_rule = re.search(r"\.tag \{[^}]*font-size: (\d+)px", PAGE_CSS).group(1)
    footer_rule = re.search(r"footer \{[^}]*font-size: (\d+)px", PAGE_CSS).group(1)
    assert tag_rule == footer_rule and f"smallest text (the tags and the footer) {tag_rule} px" in doc
    _, page = real_pages[(AXELSEN_GINTING, "A")]
    sentence_words = sum(len(unescape(text).split()) for text in re.findall(r'<p class="claim">(.*?)</p>', page))
    reading_words = sum(len(unescape(text).split()) for text in re.findall(r'<p class="hint">(.*?)</p>', page))
    assert f"{sentence_words + reading_words} words of bullets ({sentence_words} in the 6 bold sentences" in doc
    assert f"{reading_words} in the 6 reading lines" in doc


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


def test_every_shuttleset_match_builds_with_at_most_three_evidenced_bullets_per_section(real_pages):
    observations = all_observations(DATABASE_PATH)
    assert len(real_pages) == 88
    bullet_count = 0
    for (match_id, me), (report, page) in real_pages.items():
        assert page.count("<h2>") == 2 and VIEWPORT_META in page and "<script" not in page
        for section in (report["opponent"], report["me"]):
            engine = tendencies_from_observations(observations, match_id, section["player"])
            assert 0 < len(section["bullets"]) <= 3 and section["tendency_count"] == len(engine)
            for bullet, tendency in zip(section["bullets"], engine, strict=False):
                bullet_count += 1
                assert (bullet["id"], bullet["k"], bullet["n"], bullet["rate"], bullet["pros_rate"]) == (tendency["id"], tendency["k"], tendency["n"], tendency["rate"], tendency["reference_rate"])
                assert re.fullmatch(EVIDENCE_PATTERN, bullet["evidence"]) and bullet["n"] >= 5
                assert f'<span class="evidence">{bullet["evidence"]}</span>' in page
                assert f" · pros {re.search(r'\(pros: (\d+%)\)', tendency['text']).group(1)}<" in page
                if bullet["zone_rect"] is not None:
                    x0, y0, x1, y1 = bullet["zone_rect"]
                    assert 0 <= x0 < x1 <= COURT_WIDTH_M + 1e-9 and 0 <= y0 < y1 <= COURT_LENGTH_M + 1e-9
    assert bullet_count == 528


def test_results_doc_counts_match_the_generated_pages(real_pages):
    doc = RESULTS_PATH.read_text()
    bullets = [bullet for report, _ in real_pages.values() for bullet in report["opponent"]["bullets"] + report["me"]["bullets"]]
    families = pd.Series([bullet["family"] for bullet in bullets]).value_counts()
    sizes = [len(page.encode()) for _, page in real_pages.values()]
    assert "88 pages built" in doc and f"{len(bullets)} bullets over {len(real_pages) * 2} sections" in doc
    assert f"{sum(bullet['zone_rect'] is not None for bullet in bullets)} bullets (zone and serve placement)" in doc
    assert ", ".join(f"{family} {families[family]}" for family in ("response", "zone_mix", "serve_placement", "winning_shot", "error_shot", "error_reason")) in doc
    sources = pd.Series([bullet["label_source"] for bullet in bullets]).value_counts()
    assert f"{sources['labels']} bullets from labels, {sources['mixed']} `mixed`" in doc and "predictions" not in sources
    assert f"{min(sizes) / 1000:.1f} to {max(sizes) / 1000:.1f} KB" in doc


def test_results_doc_axelsen_ginting_page_matches(real_pages):
    doc = RESULTS_PATH.read_text()
    report, page = real_pages[(AXELSEN_GINTING, "A")]
    assert report["opponent"]["name"] == "Anthony Sinisuka GINTING" and "<title>Anthony Sinisuka GINTING scouting report</title>" in page
    claims = [re.sub(r"\s+", " ", claim) for claim in re.findall(r'<p class="claim">(.*?)</p>', page)]
    assert claims[:3] == [
        "From the mid court, backhand side: drives 25 of 77 times (pros: 18%).",
        "Serves mid-depth down the middle 3 of 12 times (pros: 4%).",
        "From the front court, forehand side: drives 19 of 63 times (pros: 16%).",
    ]
    assert [bullet["evidence"] for bullet in report["me"]["bullets"]] == ["21 of 28", "88 of 148", "42 of 71"]
    assert report["me"]["bullets"][0]["weaker_evidence"] and f"{len(page.encode()):,} bytes" in doc
    for claim in claims[:3]:
        assert claim[:-1] in doc
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
