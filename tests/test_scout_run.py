import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd
import pytest

import scout.run as scout_run
from analytics.build_shots import SCHEMA_PATH
from scout.run import (
    CalibrationInputs,
    SetVideo,
    StepFailedError,
    StepLog,
    TrackingOptions,
    batch_frames,
    main,
    parse_arguments,
    players_arguments,
    parse_forced_steps,
    run_match,
    run_video_steps,
    set_videos,
    step_arguments,
    step_outputs,
    tracking_options,
    write_match_rows,
)
from vision import court, features, hits, players, shuttle
from vision.features import build_own_match
from tests.results_pages import docs_on_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
BROADCAST_VIDEO = Path("data/match.mp4")
TEST_RANGE = (14201, 14947)
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"


def write_video(video_path: Path, brightness: int, frame_count: int = 12) -> Path:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(frame_count):
        writer.write(np.full((48, 64, 3), brightness, dtype=np.uint8))
    writer.release()
    return video_path


@pytest.fixture
def outputs_in(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setattr(court, "calibration_path", lambda video, *rest: root / "calibration" / Path(video).stem / "court.json")
    monkeypatch.setattr(players, "positions_path", lambda video, start, end, *rest: root / "players" / Path(video).stem / f"positions_{start}_{end}.csv")
    monkeypatch.setattr(players, "boxes_path", lambda video, start, end, *rest: root / "players" / Path(video).stem / f"boxes_{start}_{end}.csv")
    monkeypatch.setattr(hits, "output_paths", lambda video, start, end, *rest: (root / "hits" / Path(video).stem / f"hits_{start}_{end}.csv", root / "hits" / Path(video).stem / f"rallies_{start}_{end}.csv"))
    calls = []

    def fake_command(step):
        def run(arguments):
            calls.append((step, Path(arguments[0]).resolve(), list(arguments)))
            video = Path(arguments[0])
            numbers = [int(value) for value in arguments[2:5:2]] if "--start" in arguments else [0, 0]
            fake_video = SetVideo(video, 1, *numbers)
            for path in step_outputs(step, fake_video):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(video.resolve()))
            return 0
        return run

    monkeypatch.setattr(scout_run, "STEP_COMMANDS", {step: fake_command(step) for step in scout_run.VIDEO_STEPS})
    return calls


def test_set_numbers_follow_argument_order_and_bad_inputs_are_rejected(tmp_path):
    first, second, third = (write_video(tmp_path / name, 10 * index) for index, name in enumerate(("b_first.avi", "a_second.avi", "c_third.avi")))
    videos = set_videos([first, second, third])
    assert [(video.video_path.name, video.set_no, video.start_frame, video.end_frame) for video in videos] == [
        ("b_first.avi", 1, 0, 11), ("a_second.avi", 2, 0, 11), ("c_third.avi", 3, 0, 11)]
    assert [video.set_no for video in set_videos([third, first])] == [1, 2]
    with pytest.raises(ValueError, match="1 to 3"):
        set_videos([first, second, third, first])
    with pytest.raises(ValueError, match="1 to 3"):
        set_videos([])
    with pytest.raises(ValueError, match="different file names"):
        set_videos([first, write_video(tmp_path / "other" / "b_first.avi", 99)])
    with pytest.raises(FileNotFoundError):
        set_videos([tmp_path / "missing.avi"], (0, 5))
    with pytest.raises(ValueError, match="bad frame range"):
        set_videos([first], (5, 2))


def test_existing_outputs_are_reused_and_force_reruns_only_the_named_steps(tmp_path, outputs_in):
    video = set_videos([write_video(tmp_path / "club" / "set1.avi", 50)], (0, 9))[0]
    run_video_steps(video, frozenset(), StepLog())
    assert [step for step, _, _ in outputs_in] == ["court", "players", "hits"] == list(scout_run.VIDEO_STEPS)
    outputs_in.clear()
    log = StepLog()
    run_video_steps(video, frozenset(), log)
    assert outputs_in == [] and all("reused" in line for line in log.lines)
    assert [line.split(":")[0] for line in log.lines] == ["set 1 court", "set 1 players", "set 1 hits"]
    run_video_steps(video, frozenset({"hits"}), StepLog())
    assert [step for step, _, _ in outputs_in] == ["hits"]
    outputs_in.clear()
    run_video_steps(video, frozenset({"all"}), StepLog())
    assert [step for step, _, _ in outputs_in] == ["court", "players", "hits"]


def test_tracking_options_reach_the_players_command_only(tmp_path, outputs_in):
    video = set_videos([write_video(tmp_path / "club" / "set1.avi", 50)], (0, 9))[0]
    run_video_steps(video, frozenset(), StepLog())
    arguments_by_step = {step: arguments for step, _, arguments in outputs_in}
    assert arguments_by_step["players"][1:] == ["--start", "0", "--end", "9", "--batch-frames", "8"]
    assert "--device" not in arguments_by_step["court"] + arguments_by_step["hits"] and "--batch-frames" not in arguments_by_step["court"] + arguments_by_step["hits"]
    outputs_in.clear()
    run_video_steps(video, frozenset({"all"}), StepLog(), tracking=TrackingOptions("cpu", 1))
    arguments_by_step = {step: arguments for step, _, arguments in outputs_in}
    assert arguments_by_step["players"][1:] == ["--start", "0", "--end", "9", "--device", "cpu", "--batch-frames", "1"]
    assert arguments_by_step["hits"][1:] == ["--start", "0", "--end", "9"]


def test_features_are_built_with_the_feet_hit_point_by_default(tmp_path, monkeypatch):
    video = SetVideo(Path("club/set2.mp4"), 2, 0, 899)
    received = {}

    def fake_build_own_match(video_path, match_id, ranges, database_path, left_handed=frozenset(), **keywords):
        received.update(video_path=video_path, match_id=match_id, ranges=ranges, left_handed=left_handed, keywords=keywords)
        return pd.DataFrame([own_row(2, 1, 1)])

    monkeypatch.setattr(features, "build_own_match", fake_build_own_match)
    rows = scout_run.set_rows(video, "club", frozenset({"opp"}))
    assert len(rows) == 1 and received == {"video_path": video.video_path, "match_id": "club", "ranges": [(0, 899, 2)], "left_handed": frozenset({"opp"}), "keywords": {}}
    assert inspect.signature(build_own_match).parameters["hit_point"].default == "feet" and features.HIT_POINT_SOURCES == ("feet",)
    assert not hasattr(scout_run, "HIT_POINT") and "hit_point" not in (PROJECT_ROOT / "scout" / "run.py").read_text()


def test_a_failing_step_stops_the_run_and_names_the_step(tmp_path, outputs_in, monkeypatch):
    video = set_videos([write_video(tmp_path / "set1.avi", 50)], (0, 9))[0]
    monkeypatch.setitem(scout_run.STEP_COMMANDS, "players", lambda arguments: 3)
    with pytest.raises(StepFailedError) as error:
        run_video_steps(video, frozenset(), StepLog())
    assert (error.value.step, error.value.exit_code) == ("players", 3) and [step for step, _, _ in outputs_in] == ["court"]


def test_a_different_video_with_the_same_file_name_does_not_reuse_another_videos_outputs(tmp_path, outputs_in):
    first_match = set_videos([write_video(tmp_path / "2026_11_02" / "club" / "set1.avi", 40)], (0, 9))[0]
    next_match = set_videos([write_video(tmp_path / "2026_11_09" / "club" / "set1.avi", 200)], (0, 9))[0]
    run_video_steps(first_match, frozenset(), StepLog())
    outputs_in.clear()
    try:
        run_video_steps(next_match, frozenset(), StepLog())
    except (ValueError, RuntimeError):
        return
    reused_from_first_match = [path for step in scout_run.VIDEO_STEPS for path in step_outputs(step, next_match) if path.read_text() == str(first_match.video_path.resolve())]
    assert reused_from_first_match == [], f"the next match's set1 silently reused the first match's outputs: {[path.as_posix() for path in reused_from_first_match]}"


def test_duplicate_file_names_in_one_run_are_rejected_with_a_frame_range_too(tmp_path):
    first = write_video(tmp_path / "a" / "club" / "set1.avi", 40)
    second = write_video(tmp_path / "b" / "club" / "set1.avi", 200)
    with pytest.raises(ValueError, match="different file names"):
        set_videos([first, second], (0, 9))


def test_command_lines_for_every_step_and_calibration_option():
    video = SetVideo(Path("club/set2.mp4"), 2, 0, 899)
    calibration = CalibrationInputs(120, True, {2: ("a=1,2", "b=3,4", "c=5,6", "d=7,8")})
    assert step_arguments("court", video, calibration) == ["club/set2.mp4", "--frame", "120", "--points", "a=1,2", "b=3,4", "c=5,6", "d=7,8"]
    assert step_arguments("court", SetVideo(Path("club/set1.mp4"), 1, 0, 899), calibration) == ["club/set1.mp4", "--frame", "120", "--pick-points"]
    assert step_arguments("court", video) == ["club/set2.mp4", "--frame", "0"]
    assert step_arguments("players", video) == ["club/set2.mp4", "--start", "0", "--end", "899", "--batch-frames", str(players.DETECTION_BATCH_FRAMES)]
    assert step_arguments("players", video, tracking=TrackingOptions("mps", 4)) == ["club/set2.mp4", "--start", "0", "--end", "899", "--device", "mps", "--batch-frames", "4"]
    assert players_arguments(video, TrackingOptions("cpu", 1)) == ["club/set2.mp4", "--start", "0", "--end", "899", "--device", "cpu", "--batch-frames", "1"]
    assert step_arguments("hits", video) == ["club/set2.mp4", "--start", "0", "--end", "899"]
    assert step_arguments("hits", video, calibration, TrackingOptions("cpu", 1)) == ["club/set2.mp4", "--start", "0", "--end", "899"]
    for gone_step in ("camera", "shuttle", "features"):
        with pytest.raises(ValueError, match=f"unknown step: '{gone_step}'"):
            step_arguments(gone_step, video)
        with pytest.raises(ValueError, match=f"unknown step: '{gone_step}'"):
            step_outputs(gone_step, video)
    assert sorted(parse_forced_steps(["players,hits", "court"])) == ["court", "hits", "players"]
    assert scout_run.FORCEABLE_STEPS == ("court", "players", "hits", "features", "all")
    for gone_step in ("shuttle", "camera"):
        with pytest.raises(scout_run.argparse.ArgumentTypeError, match="unknown step"):
            parse_forced_steps([gone_step])
    assert scout_run.DEVICES == ("mps", "cpu") and TrackingOptions() == TrackingOptions(None, players.DETECTION_BATCH_FRAMES)


def test_tracking_options_are_parsed_and_validated():
    assert tracking_options(parse_arguments(["set1.mp4", "--match-id", "club"])) == TrackingOptions(None, players.DETECTION_BATCH_FRAMES)
    assert tracking_options(parse_arguments(["set1.mp4", "--match-id", "club", "--device", "cpu", "--batch-frames", "3"])) == TrackingOptions("cpu", 3)
    assert batch_frames("30") == 30 and batch_frames("1") == 1
    for bad in ("0", "-2", "two"):
        with pytest.raises(scout_run.argparse.ArgumentTypeError):
            batch_frames(bad)
    for bad_arguments in (["--batch-frames", "0"], ["--device", "cuda"], ["--shuttle-ensemble", "nonoverlap"], ["--hit-point", "feet"], ["--net-posts", "1", "2,3", "4,5"]):
        with pytest.raises(SystemExit) as exit_info:
            parse_arguments(["set1.mp4", "--match-id", "club", *bad_arguments])
        assert exit_info.value.code == 2


def test_command_errors_exit_one(tmp_path, capsys):
    assert main([str(tmp_path / "missing.mp4"), "--match-id", "x", "--no-serve", "--database", str(tmp_path / "db.duckdb")]) == 1
    assert "error: video not found" in capsys.readouterr().err
    video = write_video(tmp_path / "set1.avi", 50)
    assert main([str(video), "--match-id", "x", "--no-serve", "--force", "tracking", "--database", str(tmp_path / "db.duckdb")]) == 1
    assert main([str(video), "--match-id", "x", "--no-serve", "--court-points", "4", "a=1,2", "b=3,4", "c=5,6", "d=7,8", "--database", str(tmp_path / "db.duckdb")]) == 1
    assert main([str(video), "--match-id", "x", "--no-serve", "--court-points", "1", "a=1,2", "--database", str(tmp_path / "db.duckdb")]) == 1
    with pytest.raises(ValueError, match="calibration given for set"):
        run_match([video], "x", tmp_path / "db.duckdb", (0, 5), calibration=CalibrationInputs(court_points_by_set={2: ("a=1,2", "b=3,4", "c=5,6", "d=7,8")}))
    assert main([str(video), "--match-id", "x", "--no-serve", "--force", "shuttle", "--database", str(tmp_path / "db.duckdb")]) == 1
    assert "unknown step(s) to force: ['shuttle']" in capsys.readouterr().err
    assert subprocess.run([sys.executable, "-m", "scout.run"], cwd=PROJECT_ROOT, capture_output=True).returncode == 2


def own_row(set_no: int, rally: int, ball_round: int) -> dict:
    return {"source": "own", "match_id": "club", "set_no": set_no, "rally": rally, "ball_round": ball_round, "frame_num": 100 * rally + ball_round, "player": "me" if ball_round % 2 else "opp"}


def test_writing_the_match_keeps_every_set_and_replaces_only_that_matchs_rows(tmp_path):
    database_path = tmp_path / "scout.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    connection.execute("INSERT INTO shots (source, match_id, set_no, rally, ball_round, frame_num, player) VALUES ('own', 'other', 1, 1, 1, 1, 'me'), ('own', 'club', 3, 9, 9, 9, 'me')")
    connection.close()
    columns = [name for name in pd.DataFrame([own_row(1, 1, 1)]).columns]
    rows_by_set = [pd.DataFrame([own_row(1, 1, 1), own_row(1, 1, 2)]), pd.DataFrame([own_row(2, 1, 1)])]
    assert write_match_rows(database_path, "club", [frame.reindex(columns=columns) for frame in rows_by_set]) == 3
    connection = duckdb.connect(str(database_path), read_only=True)
    kept = connection.execute("SELECT shots.match_id, shots.set_no, COUNT(*) FROM shots GROUP BY shots.match_id, shots.set_no ORDER BY shots.match_id, shots.set_no").fetchall()
    connection.close()
    assert kept == [("club", 1, 2), ("club", 2, 1), ("other", 1, 1)]


def broadcast_outputs_ready() -> bool:
    video = SetVideo(BROADCAST_VIDEO, 1, *TEST_RANGE)
    return (PROJECT_ROOT / BROADCAST_VIDEO).exists() and REAL_DATABASE_PATH.exists() and all(
        (PROJECT_ROOT / path).is_file() for step in scout_run.VIDEO_STEPS for path in step_outputs(step, video)
    ) and (PROJECT_ROOT / shuttle.positions_path(BROADCAST_VIDEO, *TEST_RANGE)).is_file()


@pytest.mark.skipif(not broadcast_outputs_ready(), reason="needs data/match.mp4, data/scout.duckdb and every tracked output of frames 14201-14947")
def test_a_run_reusing_every_tracked_output_is_repeatable_and_writes_only_where_told(tmp_path, monkeypatch):
    monkeypatch.setattr(scout_run, "report_path_for", lambda match_id: tmp_path / "reports" / match_id / "index.html")
    original_write_report = scout_run.write_report
    monkeypatch.setattr(scout_run, "write_report", lambda match_id, me, database_path: original_write_report(match_id, me, tmp_path / "reports" / match_id / "index.html", database_path))
    results = []
    for attempt in range(2):
        database_path = tmp_path / f"scout_{attempt}.duckdb"
        shutil.copy(REAL_DATABASE_PATH, database_path)
        report_path, log = run_match([BROADCAST_VIDEO], "int_review", database_path, TEST_RANGE)
        assert all("reused" in line for line in log.lines if re.match(r"set 1 (court|players|hits):", line))
        assert [line.split(":")[0] for line in log.lines[:4]] == ["set 1 court", "set 1 players", "set 1 hits", "set 1 features"]
        assert not any(re.match(r"set 1 (camera|shuttle)", line) for line in log.lines)
        connection = duckdb.connect(str(database_path), read_only=True)
        rows = connection.execute("SELECT * EXCLUDE (type_conf) FROM shots WHERE shots.match_id = 'int_review' ORDER BY shots.set_no, shots.rally, shots.ball_round").df()
        counts = connection.execute("SELECT shots.source, COUNT(*) FROM shots GROUP BY shots.source ORDER BY shots.source").fetchall()
        connection.close()
        results.append((rows, report_path.read_text(encoding="utf-8"), (tmp_path / "reports" / "int_review" / "tendencies_me.json").read_text(), counts))
    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]
    rows = results[0][0]
    assert len(rows) > 0 and set(rows["set_no"]) == {1} and set(rows["player"]) <= {"me", "opp"}
    with_scratch = tmp_path / "direct.duckdb"
    direct = build_own_match(BROADCAST_VIDEO, "int_review", [(*TEST_RANGE, 1)], with_scratch, hit_point="feet").sort_values(["set_no", "rally", "ball_round"]).reset_index(drop=True)
    predicted_columns = ["type_detail_pred", "type_pred"]
    pd.testing.assert_frame_equal(rows.drop(columns=predicted_columns).reset_index(drop=True), direct.drop(columns=predicted_columns + ["type_conf"]).astype(rows.drop(columns=predicted_columns).dtypes.to_dict()), check_dtype=False)
    assert ((rows["hit_x"] == rows["player_x"]) | rows["hit_x"].isna()).all() and (rows["hit_x"].isna() == rows["player_x"].isna()).all()
    assert rows.groupby(["set_no", "rally"])["landing_x"].apply(lambda landing: landing.isna().sum() == 1).all()
    assert (rows[rows["ball_round"] == 1]["frame_num"] + 23 == rows[rows["ball_round"] == 2]["frame_num"].to_numpy()).all()
    entries = json.loads((tmp_path / "reports" / "int_review" / "tendencies_me.json").read_text())
    assert entries and all(entry["match_id"] == "int_review" and entry["player"] == "me" for entry in entries)
    assert not (PROJECT_ROOT / "data" / "reports" / "int_review").exists()


def test_scout_source_and_docs_have_no_comments_or_task_ids():
    source = (PROJECT_ROOT / "scout" / "run.py").read_text()
    assert '"""' not in source and "'''" not in source and not re.search(r"(?m)^\s*#", source)
    for doc in docs_on_disk(("docs/analytics/run.md", "docs/results/end_to_end.md")):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())


def write_unrecorded_outputs(video: SetVideo, steps=scout_run.VIDEO_STEPS) -> None:
    for step in steps:
        for path in step_outputs(step, video):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"video": str(video.video_path)}) if step == "court" else "made before source records")


def test_records_from_another_video_force_a_rerun_and_the_log_names_that_video(tmp_path, outputs_in):
    first = set_videos([write_video(tmp_path / "2026_11_02" / "club" / "set1.avi", 40)], (0, 9))[0]
    second = set_videos([write_video(tmp_path / "2026_11_09" / "club" / "set1.avi", 200)], (0, 9))[0]
    run_video_steps(first, frozenset(), StepLog())
    outputs_in.clear()
    log = StepLog()
    run_video_steps(second, frozenset(), log)
    assert [step for step, _, _ in outputs_in] == list(scout_run.VIDEO_STEPS)
    assert all(f"made from another video ({first.video_path.resolve()})" in line for line in log.lines)
    for step in scout_run.VIDEO_STEPS:
        for path in step_outputs(step, second):
            record = json.loads(scout_run.source_record_path(path).read_text())
            assert record["video"] == str(second.video_path.resolve()) and record == {"video": record["video"], **scout_run.video_fingerprint(second.video_path)}
    outputs_in.clear()
    run_video_steps(second, frozenset(), StepLog())
    assert outputs_in == []


def test_outputs_made_before_records_are_adopted_only_for_the_video_their_calibration_names(tmp_path, outputs_in):
    video = set_videos([write_video(tmp_path / "club" / "set1.avi", 40)], (0, 9))[0]
    write_unrecorded_outputs(video)
    log = StepLog()
    run_video_steps(video, frozenset(), log)
    assert outputs_in == [] and all("no source record yet" in line for line in log.lines)
    assert all(scout_run.source_record_path(path).is_file() for step in scout_run.VIDEO_STEPS for path in step_outputs(step, video))
    other = set_videos([write_video(tmp_path / "elsewhere" / "club" / "set1.avi", 90)], (0, 9))[0]
    for step in scout_run.VIDEO_STEPS:
        for path in step_outputs(step, other):
            scout_run.source_record_path(path).unlink()
    run_video_steps(other, frozenset(), StepLog())
    assert [step for step, _, _ in outputs_in] == list(scout_run.VIDEO_STEPS)


def test_a_rerun_court_step_stops_unrecorded_tracks_from_being_adopted(tmp_path, outputs_in):
    video = set_videos([write_video(tmp_path / "club" / "set1.avi", 40)], (0, 9))[0]
    write_unrecorded_outputs(video)
    run_video_steps(video, frozenset({"court"}), StepLog())
    assert [step for step, _, _ in outputs_in] == list(scout_run.VIDEO_STEPS)


def test_a_different_recording_saved_at_the_same_path_does_not_adopt_the_old_unrecorded_tracks(tmp_path, outputs_in):
    video = set_videos([write_video(tmp_path / "recordings" / "set1.avi", 40)], (0, 9))[0]
    write_unrecorded_outputs(video)
    write_video(video.video_path, 200, frame_count=20)
    run_video_steps(video, frozenset(), StepLog())
    assert [step for step, _, _ in outputs_in] == list(scout_run.VIDEO_STEPS), "tracks made before source records were adopted for a different recording saved at the same path"
