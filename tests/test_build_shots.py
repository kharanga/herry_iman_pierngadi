import ast
import math
import re
import shutil
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pytest

from analytics.build_shots import ShotsSchemaMismatchError, build_shots, pixel_to_court_homography_by_match, schema_columns
from analytics.court import (
    CENTRE_X_M,
    CROSS_COURT_MIN_LATERAL_M,
    MIDDLE_COLUMN_LEFT_EDGE_X_M,
    MIDDLE_COLUMN_RIGHT_EDGE_X_M,
    MIDDLE_COLUMN_WIDTH_M,
    ZONE_NAMES,
    DEPTH_BAND_M,
    NOT_OVER_NET_ZONE,
    apply_homography,
    diagram_to_court_homography,
    hit_zone,
    hitter_is_on_far_half,
    is_cross_court,
    landing_zone,
    rotate_half_turn,
    shot_geometry,
    zone_in_near_half,
)
from analytics.shot_types import SHOT_TYPE_GROUPS, TYPE_GROUP_BY_TYPE_EN, UnmappedShotTypeError, type_group_for
from ingest.label_translations import SHOT_TYPE_ENGLISH_BY_CHINESE
from ingest.load_shuttleset import SHUTTLESET_SET_DIR, build_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_SET_DIR = PROJECT_ROOT / SHUTTLESET_SET_DIR
MATCH_CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
SCHEMA_RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "schema.md"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
DUPLICATED_SHOT_MATCH = "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"

CONTRACT_COLUMNS = [
    ("source", "VARCHAR"),
    ("match_id", "VARCHAR"),
    ("set_no", "INTEGER"),
    ("rally", "INTEGER"),
    ("ball_round", "INTEGER"),
    ("frame_num", "INTEGER"),
    ("fps", "DOUBLE"),
    ("player", "VARCHAR"),
    ("type_en", "VARCHAR"),
    ("type_group", "VARCHAR"),
    ("type_pred", "VARCHAR"),
    ("type_conf", "DOUBLE"),
    ("backhand", "BOOLEAN"),
    ("hit_x", "DOUBLE"),
    ("hit_y", "DOUBLE"),
    ("landing_x", "DOUBLE"),
    ("landing_y", "DOUBLE"),
    ("player_x", "DOUBLE"),
    ("player_y", "DOUBLE"),
    ("opponent_x", "DOUBLE"),
    ("opponent_y", "DOUBLE"),
    ("hit_zone", "VARCHAR"),
    ("landing_zone", "VARCHAR"),
    ("is_cross", "BOOLEAN"),
    ("roundscore_me", "INTEGER"),
    ("roundscore_opp", "INTEGER"),
    ("getpoint_player", "VARCHAR"),
    ("lose_reason", "VARCHAR"),
]

CONTRACT_TYPE_GROUPS = {
    "serve": {"short service", "long service"},
    "clear": {"clear"},
    "drop": {"drop", "passive drop"},
    "smash": {"smash", "wrist smash"},
    "net": {"net shot", "return net", "cross-court net shot", "rush"},
    "lift": {"lob", "defensive return lob"},
    "drive": {"drive", "driven flight", "back-court drive", "push", "defensive return drive"},
    "unknown": {"unknown"},
}

COURT_CORNER_METERS = np.array([[0.0, 13.4], [6.1, 13.4], [0.0, 0.0], [6.1, 0.0]])

CONTRACT_ZONES = [f"{depth}_{side}" for depth in ("front", "mid", "rear") for side in ("forehand", "middle", "backhand")]

requires_shuttleset = pytest.mark.skipif(not REAL_SET_DIR.exists(), reason="run ingest/fetch_shuttleset.sh first")


def test_schema_sql_matches_contract_columns_and_types_in_order():
    assert schema_columns() == CONTRACT_COLUMNS
    assert "type_raw" not in [column_name for column_name, _ in schema_columns()]


@pytest.mark.parametrize(
    "column_name, bad_value",
    [
        ("source", "youtube"),
        ("set_no", 4),
        ("player", "C"),
        ("type_group", "flick"),
        ("hit_zone", "front"),
        ("hit_zone", "not_over_net"),
        ("hit_zone", "middle_front"),
        ("landing_zone", "out"),
        ("landing_zone", "front_centre"),
    ],
)
def test_schema_check_constraints_reject_values_outside_the_contract(column_name, bad_value):
    connection = duckdb.connect(":memory:")
    connection.execute((PROJECT_ROOT / "analytics" / "schema.sql").read_text())
    good_row = {"source": "own", "match_id": "m", "set_no": 1, "rally": 1, "ball_round": 1, "player": "me"}
    connection.execute(
        f"INSERT INTO shots ({', '.join(good_row)}) VALUES ({', '.join('?' * len(good_row))})", list(good_row.values())
    )
    bad_row = {**good_row, column_name: bad_value}
    with pytest.raises(duckdb.ConstraintException):
        connection.execute(
            f"INSERT INTO shots ({', '.join(bad_row)}) VALUES ({', '.join('?' * len(bad_row))})", list(bad_row.values())
        )
    connection.close()


def test_type_group_map_equals_contract_groups_and_covers_every_ingest_label():
    groups_from_map = {group: {type_en for type_en, mapped in TYPE_GROUP_BY_TYPE_EN.items() if mapped == group} for group in SHOT_TYPE_GROUPS}
    assert groups_from_map == CONTRACT_TYPE_GROUPS
    assert set(SHOT_TYPE_ENGLISH_BY_CHINESE.values()) == set(TYPE_GROUP_BY_TYPE_EN)
    assert type_group_for("passive drop") == "drop"
    assert type_group_for(None) is None


@pytest.mark.parametrize("type_en", ["flick", "Smash", "smash ", "", "未知球種"])
def test_type_group_for_unmapped_label_raises(type_en):
    with pytest.raises(UnmappedShotTypeError) as raised:
        type_group_for(type_en)
    assert raised.value.type_en == type_en


def test_diagram_to_court_homography_maps_diagram_corners_and_net():
    homography = diagram_to_court_homography((27.4, 810.0), (327.6, 150.0))
    np.testing.assert_allclose(
        apply_homography(homography, [[27.4, 810.0], [327.6, 810.0], [327.6, 150.0], [27.4, 150.0], [27.4, 480.0]]),
        [[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4], [0.0, 6.7]],
        atol=1e-12,
    )


@pytest.mark.parametrize("point", [(0.0, 0.0), (1.2, 9.5), (6.1, 13.4)])
def test_rotate_half_turn_swaps_halves_and_is_its_own_inverse(point):
    rotated = rotate_half_turn(point)
    assert (rotated[1] > 6.7) != (point[1] > 6.7) or point[1] == 6.7
    np.testing.assert_allclose(rotate_half_turn(rotated), point)
    assert rotate_half_turn(None) is None
    assert rotate_half_turn((math.nan, 1.0)) is None


def test_hitter_is_on_far_half_uses_hitter_points_before_receiver_points():
    assert hitter_is_on_far_half((3.0, 10.0), (3.0, 2.0), None, None) is True
    assert hitter_is_on_far_half(None, (3.0, 2.0), (3.0, 2.0), None) is False
    assert hitter_is_on_far_half(None, None, (3.0, 2.0), None) is True
    assert hitter_is_on_far_half(None, None, None, (3.0, 11.0)) is False
    assert hitter_is_on_far_half(None, None, None, None) is None


@pytest.mark.parametrize(
    "point, left_handed, expected_zone",
    [
        ((4.0, 6.7 - DEPTH_BAND_M + 0.01), False, "front_forehand"),
        ((4.0, 6.7 - DEPTH_BAND_M - 0.01), False, "mid_forehand"),
        ((2.0, 6.7 - 2 * DEPTH_BAND_M - 0.01), False, "rear_backhand"),
        ((2.0, -0.8), False, "rear_backhand"),
        ((2.0, 3.0), True, "mid_forehand"),
        ((4.0, 3.0), True, "mid_backhand"),
        ((3.05, 3.0), False, "mid_middle"),
        ((3.05, 3.0), True, "mid_middle"),
        ((2.55, 1.0), False, "rear_middle"),
        ((3.55, 1.0), True, "rear_middle"),
        ((2.5499, 5.0), False, "front_backhand"),
        ((2.5499, 5.0), True, "front_forehand"),
        ((3.5501, 5.0), False, "front_forehand"),
        ((3.5501, 5.0), True, "front_backhand"),
    ],
)
def test_zone_in_near_half_depth_bands_and_handedness(point, left_handed, expected_zone):
    assert zone_in_near_half(point, left_handed) == expected_zone


def test_landing_zone_is_seen_from_the_receiver_and_short_landings_are_not_over_net():
    assert landing_zone((5.5, 12.9)) == "rear_backhand"
    assert landing_zone((0.5, 12.9)) == "rear_forehand"
    assert landing_zone((0.5, 12.9), receiver_left_handed=True) == "rear_backhand"
    assert landing_zone((3.0, 6.7)) == NOT_OVER_NET_ZONE
    assert landing_zone((3.0, 2.0)) == NOT_OVER_NET_ZONE
    assert landing_zone(None) is None
    assert hit_zone(None) is None


def test_is_cross_court_needs_opposite_sides_and_minimum_lateral_distance():
    assert is_cross_court((1.0, 2.0), (1.2, 12.0)) is False
    assert is_cross_court((2.9, 2.0), (3.2, 12.0)) is False
    assert is_cross_court((2.5, 2.0), (2.5 + CROSS_COURT_MIN_LATERAL_M, 12.0)) is True
    assert is_cross_court((1.0, 2.0), (5.0, 12.0)) is True
    assert is_cross_court(None, (5.0, 12.0)) is None


def test_shot_geometry_normalizes_a_far_hitter_and_keeps_points_consistent():
    geometry = shot_geometry((1.0, 11.0), (5.0, 2.0), (1.0, 11.2), (5.2, 2.5))
    assert geometry["player"][1] <= 6.7 and geometry["hit"][1] <= 6.7
    assert geometry["landing"][1] > 6.7 and geometry["opponent"][1] > 6.7
    np.testing.assert_allclose(geometry["hit"], (5.1, 2.4))
    assert geometry["hit_zone"] == "mid_forehand"
    assert geometry["is_cross"] is True
    near_geometry = shot_geometry((5.1, 2.4), (1.1, 11.4), (5.1, 2.2), (0.9, 10.9))
    for derived_name in ("hit_zone", "landing_zone", "is_cross"):
        assert near_geometry[derived_name] == geometry[derived_name]
    for point_name in ("hit", "landing"):
        np.testing.assert_allclose(near_geometry[point_name], geometry[point_name])


def test_shot_geometry_with_no_points_returns_all_none():
    assert set(shot_geometry(None, None, None, None).values()) == {None}


@pytest.fixture(scope="module")
def built_database(tmp_path_factory):
    if not REAL_SET_DIR.exists():
        pytest.skip("run ingest/fetch_shuttleset.sh first")
    database_path = tmp_path_factory.mktemp("shots") / "scout.duckdb"
    build_database(REAL_SET_DIR, database_path)
    summary = build_shots(database_path)
    return database_path, summary


def fetch_all(database_path: Path, sql: str, parameters: list | None = None) -> list[tuple]:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return connection.execute(sql, parameters or []).fetchall()
    finally:
        connection.close()


@requires_shuttleset
def test_shots_table_has_contract_columns_and_one_row_per_distinct_raw_shot(built_database):
    database_path, summary = built_database
    assert [(name, column_type) for name, column_type, *_ in fetch_all(database_path, "DESCRIBE shots")] == CONTRACT_COLUMNS
    assert summary == {"raw_shots": 36484, "shuttleset_shots": 36483, "duplicates_dropped": 1}
    assert fetch_all(database_path, "SELECT COUNT(*) FROM shots WHERE shots.match_id = ? AND shots.set_no = 3 AND shots.rally = 10 AND shots.ball_round = 29", [DUPLICATED_SHOT_MATCH]) == [(1,)]
    assert fetch_all(database_path, "SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw_shots)") == [(36483,)]


@requires_shuttleset
def test_every_type_has_a_group_and_only_unknown_is_unknown(built_database):
    database_path, _ = built_database
    assert fetch_all(database_path, "SELECT COUNT(*) FROM shots WHERE shots.type_group IS NULL OR shots.type_en IS NULL") == [(0,)]
    assert fetch_all(database_path, "SELECT DISTINCT shots.type_en FROM shots WHERE shots.type_group = 'unknown'") == [("unknown",)]
    assert fetch_all(database_path, "SELECT DISTINCT shots.type_group FROM shots WHERE shots.type_en = 'passive drop'") == [("drop",)]


@requires_shuttleset
def test_backhand_and_ball_round_types_follow_the_source(built_database):
    database_path, _ = built_database
    assert fetch_all(database_path, "SELECT COUNT(*) FROM shots WHERE shots.backhand") == fetch_all(
        database_path, "SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw_shots) WHERE backhand = 1"
    )
    assert fetch_all(database_path, "SELECT COUNT(*) FROM shots WHERE shots.backhand IS NULL") == [(0,)]


def four_corner_homography(corner_row) -> np.ndarray:
    upleft_x, upright_x, downleft_x, downright_x, upleft_y, upright_y, downleft_y, downright_y = corner_row
    corner_pixels = np.float32([[upleft_x, upleft_y], [upright_x, upright_y], [downleft_x, downleft_y], [downright_x, downright_y]])
    return cv2.getPerspectiveTransform(corner_pixels, np.float32(COURT_CORNER_METERS))


@requires_shuttleset
def test_pixel_to_meters_agrees_with_an_independent_four_corner_homography_for_every_match(built_database):
    database_path, _ = built_database
    connection = duckdb.connect(str(database_path), read_only=True)
    analytics_homographies = pixel_to_court_homography_by_match(connection)
    corner_rows = connection.execute(
        "SELECT homography.match_id, homography.upleft_x, homography.upright_x, homography.downleft_x, homography.downright_x, "
        "homography.upleft_y, homography.upright_y, homography.downleft_y, homography.downright_y FROM homography"
    ).fetchall()
    connection.close()
    assert len(corner_rows) == 44
    for match_id, *corner_row in corner_rows:
        independent = four_corner_homography(corner_row)
        upleft_x, upright_x, downleft_x, downright_x, upleft_y, upright_y, downleft_y, downright_y = corner_row
        pixel_grid = np.array([[x, y] for x in np.linspace(downleft_x, downright_x, 7) for y in np.linspace(upleft_y, downleft_y, 7)])
        np.testing.assert_allclose(apply_homography(analytics_homographies[match_id], pixel_grid), apply_homography(independent, pixel_grid), atol=1e-4)


@pytest.mark.skipif(not MATCH_CALIBRATION_PATH.exists() or not REAL_SET_DIR.exists(), reason="needs ShuttleSet and data/calibration/match/court.json")
def test_net_line_of_the_video_calibration_maps_to_6_7_meters(built_database):
    from vision.court import court_to_pixels, load_calibration

    database_path, _ = built_database
    calibration = load_calibration(MATCH_CALIBRATION_PATH)
    net_pixels = court_to_pixels(calibration["pixel_to_court_homography"], [[0.0, 6.7], [3.05, 6.7], [6.1, 6.7]])
    connection = duckdb.connect(str(database_path), read_only=True)
    analytics_homography = pixel_to_court_homography_by_match(connection)[AXELSEN_GINTING]
    connection.close()
    np.testing.assert_allclose(apply_homography(analytics_homography, net_pixels), [[0.0, 6.7], [3.05, 6.7], [6.1, 6.7]], atol=0.01)


@requires_shuttleset
def test_hitter_is_always_on_the_near_half_and_receiver_on_the_far_half(built_database):
    database_path, _ = built_database
    assert fetch_all(database_path, "SELECT COUNT(*) FROM shots WHERE shots.player_y > 6.7") == [(0,)]
    near_hits, all_hits = fetch_all(database_path, "SELECT COUNT(*) FILTER (WHERE shots.hit_y <= 6.7), COUNT(shots.hit_y) FROM shots")[0]
    assert near_hits / all_hits > 0.99
    far_receivers, all_receivers = fetch_all(database_path, "SELECT COUNT(*) FILTER (WHERE shots.opponent_y > 6.7), COUNT(shots.opponent_y) FROM shots")[0]
    assert far_receivers / all_receivers > 0.99


def leading_score_before_each_rally(database_path: Path) -> dict[tuple[str, int, int], int]:
    rally_scores = fetch_all(
        database_path,
        "SELECT raw_shots.match_id, raw_shots.set_no, raw_shots.rally, GREATEST(MAX(raw_shots.roundscore_A), MAX(raw_shots.roundscore_B)) "
        "FROM raw_shots GROUP BY raw_shots.match_id, raw_shots.set_no, raw_shots.rally "
        "ORDER BY raw_shots.match_id, raw_shots.set_no, raw_shots.rally",
    )
    leading_before = {}
    previous_set, previous_leading_after = None, 0
    for match_id, set_no, rally, leading_after in rally_scores:
        if (match_id, set_no) != previous_set:
            previous_set, previous_leading_after = (match_id, set_no), 0
        leading_before[(match_id, set_no, rally)] = previous_leading_after
        previous_leading_after = leading_after
    return leading_before


@requires_shuttleset
def test_normalization_follows_end_switches_between_sets_and_mid_set_3(built_database):
    database_path, _ = built_database
    leading_before = leading_score_before_each_rally(database_path)
    rows = fetch_all(
        database_path,
        "SELECT raw_shots.match_id, raw_shots.set_no, raw_shots.rally, raw_shots.player, raw_shots.player_location_x, raw_shots.player_location_y, "
        "matches.downcourt, homography.upleft_x, homography.upright_x, homography.downleft_x, homography.downright_x, "
        "homography.upleft_y, homography.upright_y, homography.downleft_y, homography.downright_y "
        "FROM raw_shots JOIN matches ON matches.match_id = raw_shots.match_id JOIN homography ON homography.match_id = raw_shots.match_id "
        "WHERE raw_shots.player_location_x IS NOT NULL",
    )
    homography_by_match = {row[0]: four_corner_homography(row[7:]) for row in rows}
    agreeing = 0
    for match_id, set_no, rally, player, pixel_x, pixel_y, downcourt, *_ in rows:
        position_is_near = apply_homography(homography_by_match[match_id], [[pixel_x, pixel_y]])[0, 1] <= 6.7
        end_swaps = (set_no - 1) + int(set_no == 3 and leading_before[(match_id, set_no, rally)] >= 11)
        a_is_near = (downcourt == 0) == (end_swaps % 2 == 0)
        agreeing += position_is_near == (a_is_near == (player == "A"))
    assert agreeing / len(rows) > 0.99


@requires_shuttleset
def test_landing_inside_court_for_at_least_95_percent_of_non_error_shots_as_documented(built_database):
    database_path, _ = built_database
    inside, non_error = fetch_all(
        database_path,
        "SELECT COUNT(*) FILTER (WHERE shots.landing_x BETWEEN 0 AND 6.1 AND shots.landing_y BETWEEN 0 AND 13.4), COUNT(*) "
        "FROM shots WHERE shots.landing_x IS NOT NULL AND shots.lose_reason IS NULL",
    )[0]
    assert inside / non_error >= 0.95
    assert f"({inside:,} of {non_error:,})" in SCHEMA_RESULTS_PATH.read_text()


@requires_shuttleset
def test_documented_hit_equals_previous_landing_pair_count(built_database):
    database_path, _ = built_database
    equal_pairs, all_pairs = fetch_all(
        database_path,
        "SELECT COUNT(*) FILTER (WHERE ABS(hit_x - previous_landing_x) < 1e-6 AND ABS(hit_y - previous_landing_y) < 1e-6), COUNT(*) FROM ("
        "SELECT distinct_raw_shots.hit_x, distinct_raw_shots.hit_y, "
        "LAG(distinct_raw_shots.landing_x) OVER (PARTITION BY distinct_raw_shots.match_id, distinct_raw_shots.set_no, distinct_raw_shots.rally ORDER BY distinct_raw_shots.ball_round) AS previous_landing_x, "
        "LAG(distinct_raw_shots.landing_y) OVER (PARTITION BY distinct_raw_shots.match_id, distinct_raw_shots.set_no, distinct_raw_shots.rally ORDER BY distinct_raw_shots.ball_round) AS previous_landing_y "
        "FROM (SELECT DISTINCT * FROM raw_shots) AS distinct_raw_shots) "
        "WHERE hit_x IS NOT NULL AND previous_landing_x IS NOT NULL",
    )[0]
    assert equal_pairs / all_pairs > 0.99
    assert f"{equal_pairs:,} of {all_pairs:,} pairs" in SCHEMA_RESULTS_PATH.read_text()


@requires_shuttleset
def test_zones_and_is_cross_set_on_every_shot_with_the_points_they_need(built_database):
    database_path, _ = built_database
    assert fetch_all(
        database_path,
        "SELECT COUNT(*) FILTER (WHERE (shots.hit_x IS NOT NULL OR shots.player_x IS NOT NULL) AND shots.hit_zone IS NULL), "
        "COUNT(*) FILTER (WHERE shots.landing_x IS NOT NULL AND shots.landing_zone IS NULL), "
        "COUNT(*) FILTER (WHERE shots.landing_x IS NOT NULL AND (shots.hit_x IS NOT NULL OR shots.player_x IS NOT NULL) AND shots.is_cross IS NULL), "
        "COUNT(*) FILTER (WHERE shots.hit_x IS NULL AND shots.player_x IS NULL AND shots.hit_zone IS NOT NULL) FROM shots",
    ) == [(0, 0, 0, 0)]


@requires_shuttleset
def test_scores_are_before_the_rally_and_step_one_point_per_rally(built_database):
    database_path, _ = built_database
    assert fetch_all(
        database_path,
        "SELECT COUNT(*) FROM (SELECT shots.match_id, shots.set_no, MIN(shots.rally) AS first_rally FROM shots GROUP BY ALL) AS first_rallies "
        "JOIN shots ON shots.match_id = first_rallies.match_id AND shots.set_no = first_rallies.set_no AND shots.rally = first_rallies.first_rally "
        "WHERE shots.roundscore_me <> 0 OR shots.roundscore_opp <> 0",
    ) == [(0,)]
    step_sizes = fetch_all(
        database_path,
        "SELECT DISTINCT next_total - this_total, COUNT(*) FROM ("
        "SELECT rally_scores.rally, rally_scores.total AS this_total, "
        "LEAD(rally_scores.total) OVER (PARTITION BY rally_scores.match_id, rally_scores.set_no ORDER BY rally_scores.rally) AS next_total, "
        "LEAD(rally_scores.rally) OVER (PARTITION BY rally_scores.match_id, rally_scores.set_no ORDER BY rally_scores.rally) AS next_rally "
        "FROM (SELECT shots.match_id, shots.set_no, shots.rally, MAX(shots.roundscore_me + shots.roundscore_opp) AS total FROM shots GROUP BY ALL) AS rally_scores) "
        "WHERE next_rally = rally + 1 GROUP BY ALL",
    )
    assert step_sizes == [(1, 3545)]


@requires_shuttleset
def test_rebuild_is_idempotent_and_keeps_own_rows(built_database, tmp_path):
    database_path, _ = built_database
    rebuilt_path = tmp_path / "rebuilt.duckdb"
    shutil.copy(database_path, rebuilt_path)
    connection = duckdb.connect(str(rebuilt_path))
    connection.execute("INSERT INTO shots (source, match_id, set_no, rally, ball_round, player) VALUES ('own', 'my_game', 1, 1, 1, 'me')")
    connection.close()
    build_shots(rebuilt_path)
    build_shots(rebuilt_path)
    connection = duckdb.connect()
    connection.execute(f"ATTACH '{database_path}' AS first_build (READ_ONLY)")
    connection.execute(f"ATTACH '{rebuilt_path}' AS rebuilt (READ_ONLY)")
    try:
        for first_schema, second_schema in (("first_build", "rebuilt"), ("rebuilt", "first_build")):
            assert connection.execute(
                f"SELECT COUNT(*) FROM (SELECT * FROM {first_schema}.shots WHERE source = 'shuttleset' "
                f"EXCEPT ALL SELECT * FROM {second_schema}.shots WHERE source = 'shuttleset')"
            ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM rebuilt.shots WHERE source = 'own'").fetchone()[0] == 1
    finally:
        connection.close()


@requires_shuttleset
def test_unmapped_type_fails_build_and_keeps_previous_shots(built_database, tmp_path):
    database_path, _ = built_database
    broken_path = tmp_path / "broken.duckdb"
    shutil.copy(database_path, broken_path)
    connection = duckdb.connect(str(broken_path))
    connection.execute("UPDATE raw_shots SET type = 'flick' WHERE raw_shots.match_id = ? AND raw_shots.set_no = 1 AND raw_shots.rally = 1", [AXELSEN_GINTING])
    connection.close()
    with pytest.raises(UnmappedShotTypeError):
        build_shots(broken_path)
    assert fetch_all(broken_path, "SELECT COUNT(*) FROM shots") == [(36483,)]


def test_old_shots_table_with_other_columns_raises_schema_mismatch(tmp_path):
    database_path = tmp_path / "old.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE shots (source VARCHAR, match_id VARCHAR, type_raw VARCHAR)")
    connection.close()
    with pytest.raises(ShotsSchemaMismatchError):
        build_shots(database_path)
    assert fetch_all(database_path, "SELECT column_name FROM information_schema.columns WHERE table_name = 'shots' ORDER BY ordinal_position") == [
        ("source",),
        ("match_id",),
        ("type_raw",),
    ]


@pytest.mark.parametrize("source_path", sorted((PROJECT_ROOT / "analytics").glob("*.py")), ids=lambda path: path.name)
def test_analytics_source_has_no_comments_or_docstrings(source_path):
    source_text = source_path.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.mark.parametrize(
    "doc_name", ["analytics/build_shots.md", "analytics/court.md", "analytics/shot_types.md", "analytics/spot_check.md", "results/schema.md"]
)
def test_shots_docs_exist_and_have_no_task_ids(doc_name):
    doc_text = (PROJECT_ROOT / "docs" / doc_name).read_text()
    assert re.findall(TASK_ID_PATTERN, doc_text) == []


@requires_shuttleset
def test_documented_pair_sql_reproduces_its_documented_result(built_database):
    database_path, _ = built_database
    results_text = SCHEMA_RESULTS_PATH.read_text()
    documented_sql = re.search(r"```sql\n(.*?)```", results_text, re.DOTALL).group(1)
    pairs, equal_pairs = fetch_all(database_path, documented_sql)[0]
    assert f"Result: `pairs` {pairs:,}, `hit_equals_previous_landing` {equal_pairs:,}." in results_text
    assert (pairs, equal_pairs) == (31692, 31679)


def test_zone_names_are_the_nine_contract_zones():
    assert sorted(ZONE_NAMES) == sorted(CONTRACT_ZONES)


@pytest.mark.parametrize("zone", CONTRACT_ZONES)
def test_schema_accepts_every_contract_zone(zone):
    connection = duckdb.connect(":memory:")
    connection.execute((PROJECT_ROOT / "analytics" / "schema.sql").read_text())
    connection.execute(
        "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player, hit_zone, landing_zone) VALUES ('own', 'm', 1, 1, 1, 'me', ?, ?)",
        [zone, zone],
    )
    connection.execute(
        "INSERT INTO shots (source, match_id, set_no, rally, ball_round, player, hit_zone, landing_zone) VALUES ('own', 'm', 1, 1, 2, 'me', ?, 'not_over_net')",
        [zone],
    )
    connection.close()


def test_middle_column_edges_derive_from_the_one_width_constant():
    assert MIDDLE_COLUMN_WIDTH_M == 1.0
    assert MIDDLE_COLUMN_LEFT_EDGE_X_M == CENTRE_X_M - MIDDLE_COLUMN_WIDTH_M / 2 == 2.55
    assert MIDDLE_COLUMN_RIGHT_EDGE_X_M == CENTRE_X_M + MIDDLE_COLUMN_WIDTH_M / 2 == 3.55
    for source_path in (PROJECT_ROOT / "analytics").glob("*.py"):
        assert re.search(r"\b(2\.55|3\.55)\b", source_path.read_text()) is None, source_path.name
    spot_check_source = (PROJECT_ROOT / "analytics" / "spot_check.py").read_text()
    assert "MIDDLE_COLUMN_LEFT_EDGE_X_M" in spot_check_source and "MIDDLE_COLUMN_RIGHT_EDGE_X_M" in spot_check_source


@pytest.mark.parametrize("width_m", [0.6, 2.0])
def test_shot_geometry_follows_the_middle_column_width_constant(monkeypatch, width_m):
    import analytics.court

    monkeypatch.setattr(analytics.court, "MIDDLE_COLUMN_LEFT_EDGE_X_M", CENTRE_X_M - width_m / 2)
    monkeypatch.setattr(analytics.court, "MIDDLE_COLUMN_RIGHT_EDGE_X_M", CENTRE_X_M + width_m / 2)
    inside_x = CENTRE_X_M + width_m / 2 - 0.01
    outside_x = CENTRE_X_M + width_m / 2 + 0.01
    assert analytics.court.shot_geometry((inside_x, 3.0), (6.1 - inside_x, 10.0), None, None)["hit_zone"] == "mid_middle"
    assert analytics.court.shot_geometry((outside_x, 3.0), (6.1 - outside_x, 10.0), None, None)["hit_zone"] == "mid_forehand"
    assert analytics.court.shot_geometry((3.0, 3.0), (6.1 - inside_x, 10.0), None, None)["landing_zone"] == "mid_middle"
    assert analytics.court.shot_geometry((3.0, 3.0), (6.1 - outside_x, 10.0), None, None)["landing_zone"] == "mid_forehand"


@pytest.mark.parametrize(
    "landing_x, left_handed, expected_side",
    [
        (2.54, False, "forehand"),
        (2.55, False, "middle"),
        (3.05, True, "middle"),
        (3.55, False, "middle"),
        (3.56, False, "backhand"),
        (3.56, True, "forehand"),
        (2.54, True, "backhand"),
    ],
)
def test_landing_zone_edges_hold_on_the_receivers_half(landing_x, left_handed, expected_side):
    assert landing_zone((landing_x, 10.0), receiver_left_handed=left_handed) == f"mid_{expected_side}"


def test_far_hitter_on_a_middle_edge_is_middle_after_normalization():
    assert shot_geometry((3.55, 10.0), (3.0, 3.0), (3.55, 10.0), None)["hit_zone"] == "mid_middle"
    assert shot_geometry((3.5501, 10.0), (3.0, 3.0), (3.5501, 10.0), None)["hit_zone"] == "mid_backhand"


def test_is_cross_ignores_the_middle_column():
    assert is_cross_court((2.6, 2.0), (3.5, 12.0)) is False
    assert is_cross_court((2.5, 2.0), (3.5, 12.0)) is True
    assert is_cross_court((3.0, 2.0), (3.04, 12.0)) is False


def test_old_six_zone_shots_table_raises_schema_mismatch_naming_constraints(tmp_path):
    database_path = tmp_path / "old_zones.duckdb"
    old_schema = (PROJECT_ROOT / "analytics" / "schema.sql").read_text()
    for six_zone_side in ("'front_middle', ", "'mid_middle', ", "'rear_middle', "):
        old_schema = old_schema.replace(six_zone_side, "")
    connection = duckdb.connect(str(database_path))
    connection.execute(old_schema)
    connection.close()
    with pytest.raises(ShotsSchemaMismatchError) as raised:
        build_shots(database_path)
    assert raised.value.existing_columns == raised.value.expected_columns
    assert raised.value.existing_constraints != raised.value.expected_constraints
    assert "drop the shots table" in str(raised.value)
