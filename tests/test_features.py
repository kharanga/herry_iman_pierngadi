import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from analytics.build_shots import SCHEMA_PATH, schema_columns
from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M, shot_geometry
from vision.court import homography_from_corners
import cv2

from vision.features import (
    FEET_MAX_FRAME_GAP,
    TILE_HEIGHT,
    TILE_WIDTH,
    build_own_match,
    not_over_net_breakdown,
    rally_count,
    sample_tally,
    spot_check,
    spot_check_sample,
    unnormalised_point,
    compare_with_shuttleset,
    comparison_summary,
    feet_at,
    feet_by_frame,
    floor_point_at_rest,
    main,
    matched_shot_pairs,
    own_shot_rows,
    parse_range,
    schema_frame,
    shot_row,
    shuttleset_player_side,
    with_labelled_sides,
    write_own_shots,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
REAL_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "features.md"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
TRACKED_RANGES = [(14201, 14947, 1), (16551, 17343, 1), (81360, 82116, 2), (107560, 108103, 3), (114673, 116641, 3)]
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
BROADCAST_CORNERS = [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]]
SCHEMA_COLUMN_NAMES = [name for name, _ in schema_columns()]


def positions_table(rows: list[tuple[int, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "player", "x_m", "y_m"]).assign(track_id=1, confidence=0.9)


def resting_shuttle(first_frame: int, x: int = 640, y: int = 500, count: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"frame": range(first_frame, first_frame + count), "x": x, "y": y, "visibility": 1, "inpainted": 0})


def rotated(point: tuple[float, float]) -> tuple[float, float]:
    return (COURT_WIDTH_M - point[0], COURT_LENGTH_M - point[1])


def test_shot_row_has_exactly_the_schema_columns_and_only_me_or_opp_hitters():
    row = shot_row("club", 1, 1, 1, 100, 30.0, "me", (3.0, 2.0), (2.5, 10.0), (3.0, 11.0), frozenset())
    assert list(row) == SCHEMA_COLUMN_NAMES and row["source"] == "own" and row["player"] == "me"
    assert all(row[column] is None for column in ("type_en", "type_group", "type_pred", "type_conf", "backhand", "roundscore_me", "roundscore_opp", "getpoint_player", "lose_reason"))
    assert (row["player_x"], row["player_y"]) == (row["hit_x"], row["hit_y"])
    frame = schema_frame([row])
    assert list(frame.columns) == SCHEMA_COLUMN_NAMES and frame.shape == (1, len(SCHEMA_COLUMN_NAMES))
    expected_dtypes = {"INTEGER": "Int64", "DOUBLE": "float64", "BOOLEAN": "boolean", "VARCHAR": "object"}
    assert {name: str(frame[name].dtype) for name, _ in schema_columns()} == {name: expected_dtypes[duckdb_type] for name, duckdb_type in schema_columns()}


@pytest.mark.parametrize("hitter, hit, landing, opponent", [
    ("me", (3.0, 2.0), (2.5, 10.0), (3.0, 11.0)),
    ("opp", (2.5, 10.0), (2.0, 4.0), (2.0, 4.0)),
    ("opp", (1.0, 12.0), None, None),
    ("me", None, (2.0, 9.0), (2.0, 9.0)),
])
def test_zones_and_cross_come_from_the_shared_geometry_function(hitter, hit, landing, opponent):
    for left_handed in (frozenset(), frozenset({"me"}), frozenset({"opp"})):
        row = shot_row("club", 1, 1, 1, 100, 30.0, hitter, hit, landing, opponent, left_handed)
        geometry = shot_geometry(hit, landing, hit, opponent, hitter_left_handed=hitter in left_handed, receiver_left_handed=hitter not in left_handed and bool(left_handed))
        assert (row["hit_zone"], row["landing_zone"], row["is_cross"]) == (geometry["hit_zone"], geometry["landing_zone"], geometry["is_cross"])
        for name in ("hit", "landing", "player", "opponent"):
            expected = geometry[name]
            assert (row[f"{name}_x"], row[f"{name}_y"]) == ((None, None) if expected is None else tuple(expected))


def test_far_hitter_is_rotated_to_the_near_half():
    row = shot_row("club", 1, 1, 2, 130, 30.0, "opp", (2.5, 10.0), (2.0, 4.0), (2.0, 4.0), frozenset())
    assert (row["hit_x"], row["hit_y"]) == pytest.approx(rotated((2.5, 10.0)))
    assert (row["landing_x"], row["landing_y"]) == pytest.approx(rotated((2.0, 4.0)))
    assert row["hit_y"] < COURT_LENGTH_M / 2 < row["landing_y"]


def test_feet_lookup_uses_the_nearest_frame_within_the_gap():
    feet = feet_by_frame(positions_table([(100, "me", 3.0, 2.0), (104, "me", 3.4, 2.4), (100, "opp", 3.0, 11.0)]))
    assert feet_at(feet, "me", 100) == (3.0, 2.0) and feet_at(feet, "me", 101) == (3.0, 2.0) and feet_at(feet, "me", 103) == (3.4, 2.4)
    assert feet_at(feet, "me", 100 + FEET_MAX_FRAME_GAP + 1) == (3.4, 2.4) and feet_at(feet, "opp", 100 + FEET_MAX_FRAME_GAP + 1) is None
    assert feet_at(feet, "nobody", 100) is None


def test_landing_is_the_receivers_feet_at_the_next_hit_and_the_rest_point_on_the_last_shot():
    homography = homography_from_corners(BROADCAST_CORNERS)
    positions = positions_table([(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0), (130, "opp", 2.5, 10.0), (130, "me", 2.0, 4.0), (160, "me", 1.0, 3.0), (160, "opp", 4.0, 9.0)])
    hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [100, 130, 160], "hitter": ["me", "opp", "me"], "score": 0.9, "reach": 0.0})
    rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [180], "hits": [3]})
    rows = own_shot_rows(hits, rallies, positions, resting_shuttle(180, y=400), homography, "club", 2, 30.0, first_rally=4)
    assert [(row["set_no"], row["rally"], row["ball_round"], row["player"], row["frame_num"]) for row in rows] == [(2, 4, 1, "me", 100), (2, 4, 2, "opp", 130), (2, 4, 3, "me", 160)]
    assert (rows[0]["landing_x"], rows[0]["landing_y"]) == (2.5, 10.0)
    assert (rows[1]["landing_x"], rows[1]["landing_y"]) == pytest.approx(rotated((1.0, 3.0)))
    assert (rows[1]["hit_x"], rows[1]["hit_y"]) == pytest.approx(rotated((2.5, 10.0)))
    rest = floor_point_at_rest(resting_shuttle(180, y=400), 180, homography)
    assert (rows[2]["landing_x"], rows[2]["landing_y"]) == pytest.approx(rest) and rest[1] > COURT_LENGTH_M / 2 and rows[2]["landing_zone"] != "not_over_net"
    assert (rows[0]["opponent_x"], rows[0]["opponent_y"]) == (3.0, 11.0)


def test_last_landing_is_null_when_the_shuttle_never_rests_or_the_track_ends():
    homography = homography_from_corners(BROADCAST_CORNERS)
    assert floor_point_at_rest(resting_shuttle(180).iloc[:19], 180, homography) is None
    moving = resting_shuttle(180).assign(x=lambda frame: 640 + 30 * np.arange(20))
    assert floor_point_at_rest(moving, 180, homography) is None
    positions = positions_table([(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0)])
    hits = pd.DataFrame({"rally": [1], "frame": [100], "hitter": ["me"], "score": 0.9, "reach": 0.0})
    rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [180], "hits": [1]})
    rows = own_shot_rows(hits, rallies, positions, moving, homography, "club", 1, 30.0)
    assert rows[0]["landing_x"] is None and rows[0]["landing_zone"] is None and rows[0]["is_cross"] is None


def test_same_player_twice_in_a_row_lands_on_the_hitters_own_half():
    positions = positions_table([(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0), (120, "me", 2.0, 3.0), (120, "opp", 3.0, 11.0)])
    hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 120], "hitter": ["me", "me"], "score": 0.9, "reach": 0.0})
    rallies = pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [180], "hits": [2]})
    rows = own_shot_rows(hits, rallies, positions, resting_shuttle(180).iloc[:0], homography_from_corners(BROADCAST_CORNERS), "club", 1, 30.0)
    assert rows[0]["landing_zone"] == "not_over_net" and (rows[0]["landing_x"], rows[0]["landing_y"]) == (2.0, 3.0)


def shots_rows(source: str, match_id: str, player: str, count: int) -> list[dict]:
    return [shot_row(match_id, 1, 1, ball_round, 100 + ball_round, 30.0, player, (3.0, 2.0), (2.5, 10.0), (3.0, 11.0), frozenset()) | {"source": source}
            for ball_round in range(1, count + 1)]


def row_counts(database_path: Path) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return dict(connection.execute("SELECT shots.source || ':' || shots.match_id, COUNT(*) FROM shots GROUP BY ALL").fetchall())
    finally:
        connection.close()


def test_write_replaces_only_that_matchs_own_rows_in_one_transaction(tmp_path):
    database_path = tmp_path / "scout.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    connection.register("seed", schema_frame(shots_rows("shuttleset", "pro", "A", 4) + shots_rows("own", "other_game", "me", 2)))
    connection.execute("INSERT INTO shots SELECT * FROM seed")
    connection.close()
    assert write_own_shots(database_path, "club", schema_frame(shots_rows("own", "club", "me", 3))) == 3
    assert row_counts(database_path) == {"shuttleset:pro": 4, "own:other_game": 2, "own:club": 3}
    assert write_own_shots(database_path, "club", schema_frame(shots_rows("own", "club", "opp", 5))) == 5
    assert row_counts(database_path) == {"shuttleset:pro": 4, "own:other_game": 2, "own:club": 5}
    with pytest.raises(duckdb.Error):
        write_own_shots(database_path, "club", schema_frame([shots_rows("own", "club", "me", 1)[0] | {"player": "X"}]))
    assert row_counts(database_path) == {"shuttleset:pro": 4, "own:other_game": 2, "own:club": 5}
    connection = duckdb.connect(str(database_path), read_only=True)
    columns = connection.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'shots' ORDER BY ordinal_position").fetchall()
    players = {row[0] for row in connection.execute("SELECT DISTINCT shots.player FROM shots WHERE shots.source = 'own'").fetchall()}
    connection.close()
    assert [tuple(column) for column in columns] == [tuple(column) for column in schema_columns()] and players <= {"me", "opp"}


def test_shuttleset_sides_follow_downcourt_set_changes_and_the_eleven_point_switch():
    assert shuttleset_player_side("A", 1, 0, 0, 0) == "me" and shuttleset_player_side("B", 1, 0, 0, 0) == "opp"
    assert shuttleset_player_side("A", 1, 0, 0, 1) == "opp" and shuttleset_player_side("A", 2, 0, 0, 1) == "me"
    assert shuttleset_player_side("A", 3, 10, 9, 1) == "opp" and shuttleset_player_side("A", 3, 11, 9, 1) == "me" and shuttleset_player_side("B", 3, 9, 11, 1) == "opp"
    assert shuttleset_player_side("A", 3, None, None, 0) == "me"


def test_matching_is_one_to_one_closest_first_within_the_tolerance_and_per_set():
    own = pd.DataFrame({"set_no": [1, 1, 2], "frame_num": [100, 104, 100], "player": ["me", "opp", "me"], "type_pred": ["net", "lift", "net"], "type_conf": 0.9,
                        "hit_x": 3.0, "hit_y": 2.0, "landing_x": 2.5, "landing_y": 10.0, "player_x": 3.0, "player_y": 2.0, "hit_zone": "rear_middle", "landing_zone": "mid_middle", "is_cross": False})
    labelled = pd.DataFrame({"set_no": [1, 1, 1, 2], "frame_num": [102, 103, 120, 106], "player": ["B", "A", "B", "A"], "roundscore_me": 0, "roundscore_opp": 0,
                             "type_group": ["net", "drive", "net", "net"], "hit_x": 3.0, "hit_y": 2.0, "landing_x": 2.5, "landing_y": 10.0, "player_x": 3.0, "player_y": 2.0,
                             "hit_zone": "rear_middle", "landing_zone": "mid_middle", "is_cross": False})
    pairs = matched_shot_pairs(own, with_labelled_sides(labelled, downcourt=1))
    assert sorted(zip(pairs["own_frame_num"], pairs["label_frame_num"])) == [(100, 102), (104, 103)]
    summary = comparison_summary(pairs, len(own), len(labelled))
    assert (summary["matched_pairs"], summary["hitter_agreement"], summary["type_pred_accuracy"], summary["type_pred_judged"]) == (2, 1.0, 0.5, 2)
    empty = comparison_summary(matched_shot_pairs(own.iloc[:0], with_labelled_sides(labelled, 1)), 0, len(labelled))
    assert empty["matched_pairs"] == 0 and empty["type_pred_accuracy"] is None and empty["hit_point_median_error_m"] is None


def test_range_parsing_and_command_errors(tmp_path):
    assert parse_range(["1", "2", "3"]) == (1, 2, 3)
    with pytest.raises(ValueError):
        parse_range(["1", "2", "4"])
    result = subprocess.run([sys.executable, "-m", "vision.features", str(tmp_path / "missing.mp4"), "--match-id", "x", "--range", "1", "2", "1", "--database", str(tmp_path / "db.duckdb")], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr.startswith("error: ") and not (tmp_path / "db.duckdb").exists()
    assert main([str(tmp_path / "missing.mp4"), "--match-id", "x", "--range", "1", "2", "4", "--database", str(tmp_path / "db.duckdb")]) == 1
    assert subprocess.run([sys.executable, "-m", "vision.features", str(tmp_path / "missing.mp4"), "--match-id", "x"], cwd=PROJECT_ROOT, capture_output=True).returncode == 2


real_data_required = pytest.mark.skipif(
    not (REAL_DATABASE_PATH.exists() and REAL_VIDEO_PATH.exists() and all((PROJECT_ROOT / "data" / "hits" / "match" / f"hits_{start}_{end}.csv").exists() for start, end, _ in TRACKED_RANGES)),
    reason="needs data/scout.duckdb, data/match.mp4 and the saved tracks of the five hit-detection ranges")


@pytest.fixture(scope="module")
def shipped_own_rows():
    if not REAL_DATABASE_PATH.exists():
        pytest.skip("data/scout.duckdb is not built")
    connection = duckdb.connect(str(REAL_DATABASE_PATH), read_only=True)
    try:
        rows = connection.execute("SELECT * FROM shots WHERE shots.source = 'own' AND shots.match_id = 'match_tracked' ORDER BY shots.set_no, shots.rally, shots.ball_round").df()
    finally:
        connection.close()
    if rows.empty:
        pytest.skip("no own rows for match_tracked in data/scout.duckdb")
    return rows


@real_data_required
def test_rebuilding_from_the_saved_tracks_reproduces_the_shipped_rows(shipped_own_rows, tmp_path, capsys):
    database_path = tmp_path / "scout.duckdb"
    assert main([str(REAL_VIDEO_PATH), "--match-id", "match_tracked", *[word for start, end, set_no in TRACKED_RANGES for word in ("--range", str(start), str(end), str(set_no))], "--database", str(database_path)]) == 0
    printed = capsys.readouterr().out
    assert "wrote 126 own shots for match 'match_tracked' (8 rallies, 121 with a landing point)" in printed
    rebuilt = build_own_match(REAL_VIDEO_PATH, "match_tracked", TRACKED_RANGES, database_path).sort_values(["set_no", "rally", "ball_round"]).reset_index(drop=True)
    shipped = shipped_own_rows.drop(columns=["type_pred", "type_conf"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt.drop(columns=["type_pred", "type_conf"]).astype(shipped.dtypes.to_dict()), shipped, check_dtype=False)
    assert row_counts(database_path) == {"own:match_tracked": 126}


def test_shipped_rows_keep_the_contract(shipped_own_rows):
    rows = shipped_own_rows
    assert len(rows) == 126 and set(rows["player"]) == {"me", "opp"} and set(rows["fps"]) == {30.0}
    assert rows["player_y"].max() <= COURT_LENGTH_M / 2 and rows["opponent_y"].min() >= COURT_LENGTH_M / 2
    assert rows[["type_en", "type_group", "backhand", "roundscore_me", "roundscore_opp", "getpoint_player", "lose_reason"]].isna().all().all()
    assert rows["landing_x"].isna().sum() == 5 and rows["type_pred"].notna().sum() == 121 and rows["hit_x"].notna().all()
    assert rows.groupby(["set_no", "rally"]).ngroups == 8
    pairs = 0
    for _, rally_rows in rows.groupby(["set_no", "rally"]):
        rally_rows = rally_rows.sort_values("ball_round").reset_index(drop=True)
        for index in range(len(rally_rows) - 1):
            this_shot, next_shot = rally_rows.iloc[index], rally_rows.iloc[index + 1]
            next_feet = (next_shot["hit_x"], next_shot["hit_y"])
            expected = next_feet if this_shot["player"] == next_shot["player"] else rotated(next_feet)
            assert (this_shot["landing_x"], this_shot["landing_y"]) == pytest.approx(expected, abs=1e-6)
            pairs += 1
    assert pairs == 118
    for row in rows.itertuples():
        geometry = shot_geometry((row.hit_x, row.hit_y), None if pd.isna(row.landing_x) else (row.landing_x, row.landing_y), (row.player_x, row.player_y), (row.opponent_x, row.opponent_y))
        assert (row.hit_zone, None if pd.isna(row.landing_zone) else row.landing_zone, None if pd.isna(row.is_cross) else bool(row.is_cross)) == (geometry["hit_zone"], geometry["landing_zone"], geometry["is_cross"])


def test_comparison_with_shuttleset_reproduces_the_documented_numbers(shipped_own_rows):
    summary, pairs = compare_with_shuttleset(REAL_DATABASE_PATH, "match_tracked", AXELSEN_GINTING)
    assert (summary["own_shots"], summary["matched_pairs"], summary["type_pred_judged"]) == (126, 114, 110)
    assert summary["hitter_agreement"] == pytest.approx(107 / 114) and summary["type_pred_accuracy"] == pytest.approx(73 / 110)
    assert round(summary["hit_point_median_error_m"], 2) == 1.14 and round(summary["player_point_median_error_m"], 2) == 0.61 and round(summary["landing_point_median_error_m"], 2) == 1.23
    assert (round(summary["hit_zone_agreement"], 3), round(summary["landing_zone_agreement"], 3), round(summary["is_cross_agreement"], 3)) == (0.596, 0.526, 0.882)
    judged = pairs[pairs["own_type_pred"].notna() & (pairs["label_type_group"] != "unknown")]
    correct = judged[judged["own_type_pred"] == judged["label_type_group"]]["label_type_group"].value_counts()
    totals = judged["label_type_group"].value_counts()
    assert {group: (int(correct.get(group, 0)), int(totals[group])) for group in totals.index} == {"net": (29, 36), "lift": (12, 19), "drive": (5, 18), "drop": (8, 15), "smash": (12, 14), "clear": (7, 7), "serve": (0, 1)}


def test_results_doc_row_counts_match_the_shipped_rows(shipped_own_rows):
    doc = RESULTS_PATH.read_text()
    not_over_net = int((shipped_own_rows["landing_zone"] == "not_over_net").sum())
    documented = re.search(r"(\d+) of the 126 rows have\s+`not_over_net`", doc)
    assert documented and int(documented.group(1)) == not_over_net
    assert "73 of 110 = 66.4%" in doc and "107 of 114" in doc and "median 0.88 m" in doc and "median 0.61 m" in doc


def test_results_doc_spot_check_tally_matches_the_sampled_rows(shipped_own_rows):
    doc = RESULTS_PATH.read_text()
    sampled_frames = [int(frame) for frame in re.findall(r"\bf(\d{5,6})\b", doc)]
    tally = re.search(r"Predictions in the sample: (\d+) right, (\d+) wrong", doc)
    assert tally
    connection = duckdb.connect(str(REAL_DATABASE_PATH), read_only=True)
    labelled = connection.execute("SELECT shots.set_no, shots.frame_num, shots.type_group FROM shots WHERE shots.source = 'shuttleset' AND shots.match_id = ?", [AXELSEN_GINTING]).df()
    connection.close()
    frames = [14347, 14532, 14675, 16681, 16749, 16757, 16853, 17215, 17248, 81472, 81862, 107722, 107782, 108071, 114857, 114901, 114969, 115032, 116547, 116621]
    right = wrong = 0
    for frame in frames:
        own = shipped_own_rows[shipped_own_rows["frame_num"] == frame].iloc[0]
        near = labelled[(labelled["set_no"] == own["set_no"]) & ((labelled["frame_num"] - frame).abs() <= 5)]
        if pd.isna(own["type_pred"]) or near.empty:
            continue
        right += int(near["type_group"].iloc[0] == own["type_pred"])
        wrong += int(near["type_group"].iloc[0] != own["type_pred"])
    assert (int(tally.group(1)), int(tally.group(2))) == (right, wrong), (right, wrong)


def test_rally_count_counts_set_and_rally_pairs():
    assert rally_count(pd.DataFrame({"set_no": [1, 1, 2, 2, 3], "rally": [1, 1, 1, 2, 1]})) == 4
    assert rally_count(pd.DataFrame({"set_no": [], "rally": []})) == 0


def test_not_over_net_breakdown_separates_same_player_and_rally_end_cases():
    own = pd.DataFrame({
        "set_no": [1, 1, 1, 1, 2, 2],
        "rally": [1, 1, 1, 1, 1, 1],
        "ball_round": [1, 2, 3, 4, 1, 2],
        "player": ["me", "me", "opp", "me", "opp", "me"],
        "landing_zone": ["not_over_net", "front_middle", "mid_middle", "not_over_net", "not_over_net", None],
    })
    assert not_over_net_breakdown(own.sample(frac=1, random_state=3)) == {"rows": 6, "not_over_net": 3, "followed_by_same_player": 1, "rally_end_rest_point": 1, "other": 1}


def test_sample_tally_counts_right_wrong_unlabelled_and_unpredicted():
    sample = pd.DataFrame({"type_pred": ["net", "net", None, "lift", None], "label_type_group": ["net", "drop", "net", None, None]})
    assert sample_tally(sample) == {"rows": 5, "right": 1, "wrong": 1, "without_label": 1, "without_prediction": 2}


def test_unnormalised_point_undoes_the_far_side_rotation_only():
    assert unnormalised_point(3.6, 3.4, "opp") == pytest.approx((2.5, 10.0))
    assert unnormalised_point(3.0, 2.0, "me") == (3.0, 2.0)
    assert unnormalised_point(None, 2.0, "me") is None and unnormalised_point(float("nan"), 2.0, "opp") is None


def test_spot_check_sample_is_deterministic_without_replacement_and_in_row_order():
    own = pd.DataFrame({"frame_num": range(100, 130)})
    first = spot_check_sample(own, row_count=20, seed=1)["frame_num"].tolist()
    assert first == spot_check_sample(own, row_count=20, seed=1)["frame_num"].tolist() and first == sorted(set(first)) and len(first) == 20
    assert first != spot_check_sample(own, row_count=20, seed=2)["frame_num"].tolist()
    assert len(spot_check_sample(own.iloc[:5])) == 5


def test_command_requires_a_range_or_spot_check(tmp_path):
    result = subprocess.run([sys.executable, "-m", "vision.features", str(tmp_path / "missing.mp4"), "--match-id", "x"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--range" in result.stderr and "--spot-check" in result.stderr
    assert main([str(tmp_path / "missing.mp4"), "--match-id", "x", "--spot-check", "--database", str(tmp_path / "db.duckdb")]) == 1


@real_data_required
def test_spot_check_regenerates_the_shipped_images_and_the_documented_counts(shipped_own_rows, tmp_path):
    result = spot_check(REAL_VIDEO_PATH, "match_tracked", REAL_DATABASE_PATH, AXELSEN_GINTING, checks_dir=tmp_path / "checks")
    assert result["not_over_net"] == {"rows": 126, "not_over_net": 25, "followed_by_same_player": 23, "rally_end_rest_point": 2, "other": 0}
    assert result["sample"] == {"rows": 20, "right": 9, "wrong": 7, "without_label": 2, "without_prediction": 2}
    doc = RESULTS_PATH.read_text()
    assert ", ".join(str(frame) for frame in result["sample_frames"]) in re.sub(r"\s+", " ", doc)
    assert [path.name for path in result["sheets"]] == [f"match_tracked_spot_check_{index}.jpg" for index in range(4)] + ["match_tracked_label_vs_feet.jpg"]
    for path in result["sheets"]:
        image = cv2.imread(str(path))
        assert image is not None and image.shape[1] in (2 * TILE_WIDTH, 3 * TILE_WIDTH) and image.shape[0] % TILE_HEIGHT == 0
        shipped = PROJECT_ROOT / "data" / "checks" / "features" / path.name
        if shipped.exists():
            assert path.read_bytes() == shipped.read_bytes()


def test_check_images_can_be_regenerated_from_the_repo():
    source = (PROJECT_ROOT / "vision" / "features.py").read_text()
    assert "def spot_check(" in source and 'CHECKS_DIR = Path("data/checks/features")' in source
    assert "--spot-check" in RESULTS_PATH.read_text()


def test_features_source_has_no_comments_or_docstrings_and_docs_have_no_task_ids():
    source = (PROJECT_ROOT / "vision" / "features.py").read_text()
    assert all(block.lstrip().startswith("SELECT") for block in source.split('"""')[1::2]) and "'''" not in source
    assert not re.search(r"(?m)^\s*#", source)
    for doc in ("docs/vision/features.md", "docs/results/features.md"):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())
    assert "Examples need:" not in (PROJECT_ROOT / "docs" / "vision" / "features.md").read_text()
