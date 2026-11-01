import ast
import re
import subprocess
import sys
from pathlib import Path

import shutil

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
    HIT_POINT_SOURCES,
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
    load_range_tracks,
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
from vision.hits import output_paths as hit_output_paths
from shots.predict import predict_shot_types

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


RALLY_POSITIONS = [(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0), (130, "opp", 2.5, 10.0), (130, "me", 2.0, 4.0), (160, "me", 1.0, 3.0), (160, "opp", 4.0, 9.0)]
RALLY_HITS = pd.DataFrame({"rally": [1, 1, 1], "frame": [100, 130, 160], "hitter": ["me", "opp", "me"], "score": 0.9, "reach": np.nan})


def test_landing_is_the_receivers_feet_at_the_next_hit_and_null_on_the_last_shot():
    rows = own_shot_rows(RALLY_HITS, positions_table(RALLY_POSITIONS), "club", 2, 30.0, first_rally=4)
    assert [(row["set_no"], row["rally"], row["ball_round"], row["player"], row["frame_num"]) for row in rows] == [(2, 4, 1, "me", 100), (2, 4, 2, "opp", 130), (2, 4, 3, "me", 160)]
    assert (rows[0]["hit_x"], rows[0]["hit_y"]) == (3.0, 2.0) == (rows[0]["player_x"], rows[0]["player_y"])
    assert (rows[0]["landing_x"], rows[0]["landing_y"]) == (2.5, 10.0)
    assert (rows[1]["landing_x"], rows[1]["landing_y"]) == pytest.approx(rotated((1.0, 3.0)))
    assert (rows[1]["hit_x"], rows[1]["hit_y"]) == pytest.approx(rotated((2.5, 10.0)))
    assert rows[2]["landing_x"] is None and rows[2]["landing_y"] is None and rows[2]["landing_zone"] is None and rows[2]["is_cross"] is None
    assert (rows[2]["hit_x"], rows[2]["hit_y"]) == (1.0, 3.0)
    assert (rows[0]["opponent_x"], rows[0]["opponent_y"]) == (3.0, 11.0)
    assert [(row["hit_x"], row["hit_y"]) for row in rows] == [(row["player_x"], row["player_y"]) for row in rows]


def test_a_lone_hit_and_a_hit_without_a_position_give_null_landing_and_hit_point():
    positions = positions_table([(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0)])
    hits = pd.DataFrame({"rally": [1], "frame": [100], "hitter": ["me"], "score": 0.9, "reach": np.nan})
    rows = own_shot_rows(hits, positions, "club", 1, 30.0)
    assert len(rows) == 1 and rows[0]["landing_x"] is None and rows[0]["landing_zone"] is None and rows[0]["is_cross"] is None
    far_away = pd.DataFrame({"rally": [1, 1], "frame": [100, 200], "hitter": ["me", "opp"], "score": 0.9, "reach": np.nan})
    rows = own_shot_rows(far_away, positions, "club", 1, 30.0)
    assert rows[0]["landing_x"] is None and rows[1]["hit_x"] is None and rows[1]["player_x"] is None and rows[1]["hit_zone"] is None


def test_an_inferred_serve_row_is_a_hit_like_any_other():
    serve = pd.DataFrame({"rally": [1], "frame": [77], "hitter": ["opp"], "score": [np.nan], "reach": [np.nan]})
    positions = positions_table([(77, "opp", 3.0, 11.0), (77, "me", 3.0, 2.0)] + RALLY_POSITIONS)
    rows = own_shot_rows(pd.concat([serve, RALLY_HITS]), positions, "club", 1, 30.0)
    assert [(row["ball_round"], row["player"], row["frame_num"]) for row in rows] == [(1, "opp", 77), (2, "me", 100), (3, "opp", 130), (4, "me", 160)]
    assert (rows[0]["hit_x"], rows[0]["hit_y"]) == pytest.approx(rotated((3.0, 11.0))) and (rows[0]["landing_x"], rows[0]["landing_y"]) == pytest.approx(rotated((3.0, 2.0)))
    assert rows[0]["hit_zone"] is not None and rows[0]["landing_zone"] is not None
    without_position = own_shot_rows(pd.concat([serve, RALLY_HITS]), positions_table(RALLY_POSITIONS), "club", 1, 30.0)
    assert without_position[0]["hit_x"] is None and without_position[0]["player_x"] is None and without_position[0]["hit_zone"] is None
    assert (without_position[0]["landing_x"], without_position[0]["landing_y"]) == pytest.approx(rotated((3.0, 2.0)))
    assert without_position[1]["ball_round"] == 2 and (without_position[1]["hit_x"], without_position[1]["hit_y"]) == (3.0, 2.0)
    no_score_column = own_shot_rows(pd.concat([serve, RALLY_HITS]).drop(columns=["score"]), positions, "club", 1, 30.0)
    assert [row["frame_num"] for row in no_score_column] == [77, 100, 130, 160]


def test_feet_gap_scales_with_the_frame_rate():
    positions = positions_table([(94, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0)])
    hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 130], "hitter": ["me", "opp"], "score": 0.9, "reach": np.nan})
    at_30 = own_shot_rows(hits, positions, "club", 1, 30.0)
    at_60 = own_shot_rows(hits, positions, "club", 1, 60.0)
    assert at_30[0]["hit_x"] is None and at_30[0]["player_x"] is None
    assert (at_60[0]["hit_x"], at_60[0]["hit_y"]) == (3.0, 2.0) and at_60[0]["fps"] == 60.0


def test_same_player_twice_in_a_row_lands_on_the_hitters_own_half():
    positions = positions_table([(100, "me", 3.0, 2.0), (100, "opp", 3.0, 11.0), (120, "me", 2.0, 3.0), (120, "opp", 3.0, 11.0)])
    hits = pd.DataFrame({"rally": [1, 1], "frame": [100, 120], "hitter": ["me", "me"], "score": 0.9, "reach": np.nan})
    rows = own_shot_rows(hits, positions, "club", 1, 30.0)
    assert rows[0]["landing_zone"] == "not_over_net" and (rows[0]["landing_x"], rows[0]["landing_y"]) == (2.0, 3.0)


def test_features_module_reads_no_shuttle_camera_or_contact_data(tmp_path, monkeypatch):
    module = ast.parse((PROJECT_ROOT / "vision" / "features.py").read_text())
    imported = {node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)} | {alias.name for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(name.startswith(("vision.shuttle", "vision.camera", "vision.contact")) for name in imported if name)
    import vision.features

    monkeypatch.setattr(vision.features, "output_paths", lambda video, start, end: (tmp_path / "hits.csv", tmp_path / "rallies.csv"))
    monkeypatch.setattr(vision.features, "player_positions_path", lambda video, start, end: tmp_path / "positions.csv")
    with pytest.raises(FileNotFoundError, match=r"python -m vision.hits clip.mp4 --start 0 --end 9"):
        load_range_tracks(Path("clip.mp4"), 0, 9)
    RALLY_HITS.to_csv(tmp_path / "hits.csv", index=False)
    pd.DataFrame({"rally": [1], "start_frame": [100], "end_frame": [180], "hits": [3]}).to_csv(tmp_path / "rallies.csv", index=False)
    with pytest.raises(FileNotFoundError, match=r"python -m vision.players clip.mp4 --start 0 --end 9"):
        load_range_tracks(Path("clip.mp4"), 0, 9)
    positions_table(RALLY_POSITIONS).to_csv(tmp_path / "positions.csv", index=False)
    hits, rallies, positions = load_range_tracks(Path("clip.mp4"), 0, 9)
    assert len(hits) == 3 and len(rallies) == 1 and len(positions) == 6


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
                        "hit_x": 3.0, "hit_y": 2.0, "landing_x": 2.5, "landing_y": 10.0, "player_x": 3.0, "player_y": 2.0, "hit_zone": "rear_backhand", "landing_zone": "mid_backhand", "is_cross": False})
    labelled = pd.DataFrame({"set_no": [1, 1, 1, 2], "frame_num": [102, 103, 120, 106], "player": ["B", "A", "B", "A"], "roundscore_me": 0, "roundscore_opp": 0,
                             "type_group": ["net", "drive", "net", "net"], "hit_x": 3.0, "hit_y": 2.0, "landing_x": 2.5, "landing_y": 10.0, "player_x": 3.0, "player_y": 2.0,
                             "hit_zone": "rear_backhand", "landing_zone": "mid_backhand", "is_cross": False})
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


def tracks_are_saved() -> bool:
    return all(
        (PROJECT_ROOT / path).is_file()
        for start, end, _ in TRACKED_RANGES
        for path in (*hit_output_paths(Path("data/match.mp4"), start, end), Path("data/players/match") / f"positions_{start}_{end}.csv")
    )


def saved_hits_have_serve_rows() -> bool:
    return tracks_are_saved() and all(pd.read_csv(hit_output_paths(Path("data/match.mp4"), start, end)[0])["score"].isna().any() for start, end, _ in TRACKED_RANGES)


real_data_required = pytest.mark.skipif(
    not (REAL_DATABASE_PATH.exists() and REAL_VIDEO_PATH.exists() and saved_hits_have_serve_rows()),
    reason="needs data/scout.duckdb, data/match.mp4 and the movement detector's hits (with serve rows) and player tracks of the five ranges (python -m vision.hits)")

spot_check_data_required = pytest.mark.skipif(
    not (REAL_DATABASE_PATH.exists() and REAL_VIDEO_PATH.exists() and (PROJECT_ROOT / "data" / "calibration" / "match" / "court.json").exists()),
    reason="needs data/scout.duckdb, data/match.mp4 and its court calibration")


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
def test_rebuilding_from_the_saved_tracks_gives_the_documented_player_only_rows(tmp_path, capsys):
    database_path = tmp_path / "scout.duckdb"
    assert main([str(REAL_VIDEO_PATH), "--match-id", "match_movement", *[word for start, end, set_no in TRACKED_RANGES for word in ("--range", str(start), str(end), str(set_no))], "--database", str(database_path)]) == 0
    printed = capsys.readouterr().out
    assert "wrote 132 own shots for match 'match_movement' (7 rallies, 125 with a landing point)" in printed
    assert "hit points:" not in printed
    rebuilt = build_own_match(REAL_VIDEO_PATH, "match_movement", TRACKED_RANGES, tmp_path / "again.duckdb").sort_values(["set_no", "rally", "ball_round"]).reset_index(drop=True)
    connection = duckdb.connect(str(database_path), read_only=True)
    written = connection.execute("SELECT * FROM shots WHERE shots.source = 'own' ORDER BY shots.set_no, shots.rally, shots.ball_round").df()
    connection.close()
    pd.testing.assert_frame_equal(rebuilt.astype(written.dtypes.to_dict()), written, check_dtype=False)
    assert row_counts(database_path) == {"own:match_movement": 132}
    last_shots = written.groupby(["set_no", "rally"])["ball_round"].transform("max") == written["ball_round"]
    assert written.loc[last_shots, "landing_x"].isna().all() and int(last_shots.sum()) == 7
    serves = written[written["ball_round"] == 1]
    assert len(serves) == 7 and serves["player"].tolist() == ["me", "me", "opp", "me", "opp", "opp", "me"]
    returns = written[written["ball_round"] == 2]
    assert (returns["player"].to_numpy() != serves["player"].to_numpy()).all()
    assert ((written["hit_x"] == written["player_x"]) | written["hit_x"].isna()).all() and (written["hit_x"].isna() == written["player_x"].isna()).all()


def test_shipped_rows_keep_the_contract(shipped_own_rows):
    rows = shipped_own_rows
    assert len(rows) == 132 and set(rows["player"]) == {"me", "opp"} and set(rows["fps"]) == {30.0}
    assert rows["player_y"].max() <= COURT_LENGTH_M / 2 and rows["opponent_y"].min() >= COURT_LENGTH_M / 2
    assert rows[["type_en", "type_group", "backhand", "roundscore_me", "roundscore_opp", "getpoint_player", "lose_reason"]].isna().all().all()
    assert rows["landing_x"].isna().sum() == 7 and rows["type_pred"].notna().sum() == 119 and rows["hit_x"].isna().sum() == 6
    assert (rows["hit_x"].isna() == rows["player_x"].isna()).all() and rows[rows["hit_x"].isna()]["ball_round"].tolist() == [1] * 6
    assert rows.groupby(["set_no", "rally"]).ngroups == 7 and (rows[rows["ball_round"] == 1]["frame_num"] + 23 == rows[rows["ball_round"] == 2]["frame_num"].to_numpy()).all()
    pairs = 0
    for _, rally_rows in rows.groupby(["set_no", "rally"]):
        rally_rows = rally_rows.sort_values("ball_round").reset_index(drop=True)
        for index in range(len(rally_rows) - 1):
            this_shot, next_shot = rally_rows.iloc[index], rally_rows.iloc[index + 1]
            next_feet = (next_shot["hit_x"], next_shot["hit_y"])
            expected = next_feet if this_shot["player"] == next_shot["player"] else rotated(next_feet)
            assert this_shot["player"] != next_shot["player"]
            assert (this_shot["landing_x"], this_shot["landing_y"]) == pytest.approx(expected, abs=1e-6)
            pairs += 1
    assert pairs == 125
    for row in rows.itertuples():
        hit = None if pd.isna(row.hit_x) else (row.hit_x, row.hit_y)
        geometry = shot_geometry(hit, None if pd.isna(row.landing_x) else (row.landing_x, row.landing_y), hit, None if pd.isna(row.opponent_x) else (row.opponent_x, row.opponent_y))
        assert (None if pd.isna(row.hit_zone) else row.hit_zone, None if pd.isna(row.landing_zone) else row.landing_zone, None if pd.isna(row.is_cross) else bool(row.is_cross)) == (geometry["hit_zone"], geometry["landing_zone"], geometry["is_cross"])


PLAYER_ONLY_VERSION = {
    "accuracy": (92, 116), "hit": 1.05, "landing": 1.06, "zones": (0.699, 0.700, 0.922),
    "groups": {"net": (36, 38), "smash": (14, 15), "clear": (6, 7), "lift": (13, 19), "drive": (11, 19), "drop": (12, 16), "serve": (0, 2)},
    "by_set": {"sets 1-2": (60, 74), "set 3": (32, 42)},
}


def judged_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs[pairs["own_type_pred"].notna() & pairs["label_type_group"].notna() & (pairs["label_type_group"] != "unknown")]


def correct_and_total(judged: pd.DataFrame) -> tuple[int, int]:
    return int((judged["own_type_pred"] == judged["label_type_group"]).sum()), len(judged)


def assert_comparison_matches(database_path: Path, expected: dict) -> None:
    summary, pairs = compare_with_shuttleset(database_path, "match_tracked", AXELSEN_GINTING)
    assert (summary["own_shots"], summary["matched_pairs"], summary["type_pred_judged"]) == (132, 123, 116)
    assert summary["hitter_agreement"] == 1.0 and round(summary["player_point_median_error_m"], 2) == 0.57
    assert summary["type_pred_accuracy"] == pytest.approx(expected["accuracy"][0] / expected["accuracy"][1])
    assert round(summary["hit_point_median_error_m"], 2) == expected["hit"] and round(summary["landing_point_median_error_m"], 2) == expected["landing"]
    assert (round(summary["hit_zone_agreement"], 3), round(summary["landing_zone_agreement"], 3), round(summary["is_cross_agreement"], 3)) == expected["zones"]
    judged = judged_pairs(pairs)
    assert {group: correct_and_total(part) for group, part in judged.groupby("label_type_group")} == expected["groups"]
    assert {"sets 1-2": correct_and_total(judged[judged["own_set_no"] < 3]), "set 3": correct_and_total(judged[judged["own_set_no"] == 3])} == expected["by_set"]


def test_comparison_with_shuttleset_reproduces_the_documented_numbers(shipped_own_rows):
    assert_comparison_matches(REAL_DATABASE_PATH, PLAYER_ONLY_VERSION)


@real_data_required
def test_player_only_rows_after_prediction_match_the_no_shuttle_doc(tmp_path):
    database_path = tmp_path / "scout.duckdb"
    shutil.copy(REAL_DATABASE_PATH, database_path)
    build_own_match(REAL_VIDEO_PATH, "match_movement", TRACKED_RANGES, database_path)
    predict_shot_types(database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute("SELECT * FROM shots WHERE shots.source = 'own' AND shots.match_id = 'match_movement' ORDER BY shots.set_no, shots.rally, shots.ball_round").df()
    connection.close()
    last_shots = rows.groupby(["set_no", "rally"])["ball_round"].transform("max") == rows["ball_round"]
    assert rows.loc[last_shots, "type_pred"].isna().all() and rows.loc[last_shots, "landing_x"].isna().all()
    assert rows[rows["ball_round"] == 1]["type_pred"].fillna("none").value_counts().to_dict() == {"none": 6, "serve": 1}
    assert rows[rows["ball_round"] == 2]["type_pred"].value_counts().to_dict() == {"net": 3, "drive": 3, "clear": 1}
    assert rows["type_pred"].fillna("none").value_counts().to_dict() == {"net": 41, "lift": 18, "smash": 17, "drive": 14, "clear": 14, "drop": 14, "none": 13, "serve": 1}
    assert int(rows[rows["ball_round"] == 1]["hit_x"].isna().sum()) == 6
    doc = re.sub(r"\s+", " ", (PROJECT_ROOT / "docs" / "results" / "no_shuttle.md").read_text())
    assert "| Own rows on the 5 ranges | 126 (8 rallies, one a pickup) | 125 (7 rallies) | 132 (7 rallies, 7 serves) |" in doc
    assert "net 41, lift 18, smash 17, drive 14, clear 14, drop 14, serve 1, none 13" in doc


def test_results_doc_states_the_player_only_numbers(shipped_own_rows):
    doc = re.sub(r"\s+", " ", RESULTS_PATH.read_text())
    right, total = PLAYER_ONLY_VERSION["accuracy"]
    assert f"{right} of {total} = {100 * right / total:.1f}%" in doc
    assert "| **Feet-only classifier on tracked positions, this match** | **92 of 116 = 79.3%** |" in doc
    for group, (correct, count) in PLAYER_ONLY_VERSION["groups"].items():
        assert f"| {group} | {correct} of {count} |" in doc, group
    assert "By set: sets 1-2 60 of 74, set 3 32 of 42" in doc
    assert "hit point is a median 1.05 m off (mean 1.10), the landing point 1.06 m (mean 1.14)" in doc
    assert "`hit_zone` 69.9%, `landing_zone` 70.0%, `is_cross` 92.2%" in doc
    assert "Hitter agreement with the labels is 123 of 123" in doc and "a median 0.57 m off (mean 0.68)" in doc
    assert "66 entries for `me`, 68 for `opp`" in doc and "--hit-point" not in doc and "contact estimate (0 fallbacks" not in doc


def test_only_the_feet_hit_point_is_accepted_and_the_option_is_gone(tmp_path):
    assert HIT_POINT_SOURCES == ("feet",)
    for other in ("contact", "shoulder"):
        with pytest.raises(ValueError, match="hit point must be one of"):
            build_own_match(tmp_path / "missing.mp4", "club", [(1, 2, 1)], tmp_path / "scout.duckdb", hit_point=other)
    result = subprocess.run([sys.executable, "-m", "vision.features", str(tmp_path / "missing.mp4"), "--match-id", "club", "--range", "1", "2", "1", "--hit-point", "feet"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--hit-point" in result.stderr and not (tmp_path / "scout.duckdb").exists()


def test_results_doc_row_counts_match_the_shipped_rows(shipped_own_rows):
    doc = RESULTS_PATH.read_text()
    not_over_net = int((shipped_own_rows["landing_zone"] == "not_over_net").sum())
    documented = re.search(r"(\d+) of the (\d+) rows have\s+`not_over_net`", doc)
    assert documented and (int(documented.group(1)), int(documented.group(2))) == (not_over_net, len(shipped_own_rows)) == (0, 132)
    assert "92 of 116 = 79.3%" in doc and "119 rows" in doc and "123 of 123" in doc and "median 0.57 m" in doc


def test_results_doc_spot_check_tally_matches_the_sampled_rows(shipped_own_rows):
    doc = RESULTS_PATH.read_text()
    tally = re.search(r"Predictions in the sample: (\d+) right, (\d+) wrong", doc)
    assert tally
    connection = duckdb.connect(str(REAL_DATABASE_PATH), read_only=True)
    labelled = connection.execute("SELECT shots.set_no, shots.frame_num, shots.type_group FROM shots WHERE shots.source = 'shuttleset' AND shots.match_id = ?", [AXELSEN_GINTING]).df()
    connection.close()
    frames = [14285, 14327, 14492, 14676, 16683, 16750, 16811, 16894, 17314, 17335, 81488, 81806, 107647, 107724, 108042, 114811, 114871, 114979, 115082, 116515]
    assert ", ".join(str(frame) for frame in frames) in re.sub(r"\s+", " ", doc)
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
        "landing_zone": ["not_over_net", "front_backhand", "mid_backhand", "not_over_net", "not_over_net", None],
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


@spot_check_data_required
def test_spot_check_regenerates_the_shipped_images_and_the_documented_counts(shipped_own_rows, tmp_path):
    result = spot_check(REAL_VIDEO_PATH, "match_tracked", REAL_DATABASE_PATH, AXELSEN_GINTING, checks_dir=tmp_path / "checks")
    assert result["not_over_net"] == {"rows": 132, "not_over_net": 0, "followed_by_same_player": 0, "rally_end_rest_point": 0, "other": 0}
    assert result["sample"] == {"rows": 20, "right": 15, "wrong": 4, "without_label": 0, "without_prediction": 1}
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
    for doc in ("docs/vision/features.md", "docs/results/features.md", "docs/vision/court.md"):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())
    assert "Examples need:" not in (PROJECT_ROOT / "docs" / "vision" / "features.md").read_text()


def test_keep_other_sets_replaces_only_the_written_sets(tmp_path):
    database_path = tmp_path / "scout.duckdb"
    rows = lambda set_no, count, match_id="club": schema_frame([shot_row(match_id, set_no, 1, ball_round, 100 + ball_round, 30.0, "me", (3.0, 2.0), None, None, frozenset()) for ball_round in range(1, count + 1)])
    write_own_shots(database_path, "club", rows(1, 3))
    write_own_shots(database_path, "club", rows(2, 2), replace_sets=[2])
    write_own_shots(database_path, "other", rows(1, 1, "other"))
    write_own_shots(database_path, "club", rows(2, 4), replace_sets=[2])
    connection = duckdb.connect(str(database_path), read_only=True)
    counts = connection.execute("SELECT shots.match_id, shots.set_no, COUNT(*) FROM shots GROUP BY shots.match_id, shots.set_no ORDER BY shots.match_id, shots.set_no").fetchall()
    connection.close()
    assert counts == [("club", 1, 3), ("club", 2, 4), ("other", 1, 1)]
    write_own_shots(database_path, "club", rows(3, 1))
    connection = duckdb.connect(str(database_path), read_only=True)
    counts = connection.execute("SELECT shots.match_id, shots.set_no, COUNT(*) FROM shots GROUP BY shots.match_id, shots.set_no ORDER BY shots.match_id, shots.set_no").fetchall()
    connection.close()
    assert counts == [("club", 3, 1), ("other", 1, 1)]


def own_rows_for_sheets(frame_count: int) -> pd.DataFrame:
    rows = []
    for rally, serve_frame in ((1, 2), (2, 20)):
        for ball_round, (player, frame) in enumerate((("opp", serve_frame), ("me", serve_frame + 5), ("opp", serve_frame + 10)), start=1):
            row = shot_row("club", 1, rally, ball_round, frame, 30.0, player, (3.0, 2.0), None if ball_round == 3 else (2.5, 10.0), (3.0, 11.0), frozenset())
            rows.append({**row, "type_pred": "net", "type_conf": 0.9, "label_type_group": None, "label_player": None})
    rows[0]["hit_x"] = rows[0]["hit_y"] = rows[0]["player_x"] = rows[0]["player_y"] = None
    return pd.DataFrame(rows)


def test_first_rows_and_direction_word_and_sheet_path():
    from vision.features import ORDERED_SHEET_ROWS, direction_word, first_rows, frame_centre, ordered_sheet_path

    own = pd.DataFrame({"set_no": [2, 1, 1, 1], "rally": [1, 2, 1, 1], "ball_round": [1, 1, 2, 1]})
    assert first_rows(own, 3)[["set_no", "rally", "ball_round"]].values.tolist() == [[1, 1, 1], [1, 1, 2], [1, 2, 1]]
    assert len(first_rows(own, 0)) == 0 and len(first_rows(own, 10)) == 4
    assert (direction_word(True), direction_word(False), direction_word(None), direction_word(np.nan)) == ("cross", "straight", "no direction", "no direction")
    assert frame_centre(np.zeros((480, 640, 3), dtype=np.uint8)) == (320, 240)
    assert ordered_sheet_path("club", 12, 1, Path("out")).as_posix() == "out/club_first12_1.jpg" and ORDERED_SHEET_ROWS == 10


def test_inferred_serve_frames_come_from_every_hits_file_with_a_score_column(tmp_path):
    from vision.features import inferred_serve_frames

    folder = tmp_path / "hits" / "club"
    folder.mkdir(parents=True)
    pd.DataFrame({"rally": [1, 1], "frame": [77, 100], "hitter": ["opp", "me"], "score": [np.nan, 0.9], "reach": np.nan}).to_csv(folder / "hits_0_199.csv", index=False)
    pd.DataFrame({"rally": [1, 1], "frame": [300, 330], "hitter": ["me", "opp"], "score": [np.nan, 0.8], "reach": np.nan}).to_csv(folder / "hits_200_399.csv", index=False)
    pd.DataFrame({"rally": [1], "frame": [500], "hitter": ["me"]}).to_csv(folder / "hits_400_599.csv", index=False)
    pd.DataFrame({"rally": [1], "start_frame": [77], "end_frame": [150], "hits": [2]}).to_csv(folder / "rallies_0_199.csv", index=False)
    assert inferred_serve_frames(Path("club.mp4"), tmp_path / "hits") == {77, 300}
    assert inferred_serve_frames(Path("other.mp4"), tmp_path / "hits") == set()


def test_ordered_sheets_hold_the_first_rows_in_order_ten_per_sheet(tmp_path):
    from vision.features import write_ordered_spot_check_sheets

    video_path = tmp_path / "club.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    for _ in range(40):
        writer.write(np.full((720, 1280, 3), 100, dtype=np.uint8))
    writer.release()
    homography = homography_from_corners(BROADCAST_CORNERS)
    own = own_rows_for_sheets(40).sample(frac=1, random_state=2)
    sheets = write_ordered_spot_check_sheets(video_path, homography, own, 5, "club", {2, 20}, tmp_path / "checks", rows_per_sheet=2)
    assert [path.name for path in sheets] == ["club_first5_0.jpg", "club_first5_1.jpg", "club_first5_2.jpg"]
    heights = [cv2.imread(str(path)).shape for path in sheets]
    assert heights == [(2 * TILE_HEIGHT, 2 * TILE_WIDTH, 3), (2 * TILE_HEIGHT, 2 * TILE_WIDTH, 3), (TILE_HEIGHT, 2 * TILE_WIDTH, 3)]
    last_shot_tile = cv2.imread(str(sheets[1]))[:TILE_HEIGHT, TILE_WIDTH:]
    assert last_shot_tile[40:, :].max() == 0 and cv2.imread(str(sheets[2]))[40:, TILE_WIDTH:].max() > 0
    assert write_ordered_spot_check_sheets(video_path, homography, own, 0, "club", set(), tmp_path / "empty") == []
    assert sorted(path.name for path in (tmp_path / "checks").iterdir()) == ["club_first5_0.jpg", "club_first5_1.jpg", "club_first5_2.jpg"]


def test_spot_check_first_command_draws_without_writing_rows(tmp_path, monkeypatch, capsys):
    import vision.features

    received = {}

    def fake_ordered_spot_check(video_path, match_id, row_count, database_path, shuttleset_match_id, **keywords):
        received.update(video_path=video_path, match_id=match_id, row_count=row_count, database_path=database_path, shuttleset_match_id=shuttleset_match_id, keywords=keywords)
        return [tmp_path / "club_first12_0.jpg"]

    monkeypatch.setattr(vision.features, "ordered_spot_check", fake_ordered_spot_check)
    video_path = tmp_path / "club.mp4"
    video_path.write_bytes(b"")
    assert main([str(video_path), "--match-id", "club", "--spot-check-first", "12", "--database", str(tmp_path / "db.duckdb"), "--compare", AXELSEN_GINTING]) == 0
    assert received == {"video_path": video_path, "match_id": "club", "row_count": 12, "database_path": tmp_path / "db.duckdb", "shuttleset_match_id": AXELSEN_GINTING, "keywords": {"hits_dir": Path("data/hits")}}
    assert "club_first12_0.jpg" in capsys.readouterr().out and not (tmp_path / "db.duckdb").exists()
    result = subprocess.run([sys.executable, "-m", "vision.features", str(video_path), "--match-id", "club"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--spot-check-first" in result.stderr


@spot_check_data_required
def test_ordered_spot_check_on_the_broadcast_match_writes_two_sheets_for_twelve_rows(tmp_path):
    from vision.features import ordered_spot_check

    sheets = ordered_spot_check(REAL_VIDEO_PATH, "match_tracked", 12, REAL_DATABASE_PATH, AXELSEN_GINTING, checks_dir=tmp_path / "checks")
    assert [path.name for path in sheets] == ["match_tracked_first12_0.jpg", "match_tracked_first12_1.jpg"]
    assert cv2.imread(str(sheets[0])).shape == (10 * TILE_HEIGHT, 2 * TILE_WIDTH, 3) and cv2.imread(str(sheets[1])).shape == (2 * TILE_HEIGHT, 2 * TILE_WIDTH, 3)
    with pytest.raises(ValueError, match="no own rows"):
        ordered_spot_check(REAL_VIDEO_PATH, "nobody", 5, REAL_DATABASE_PATH, None, checks_dir=tmp_path / "none")
