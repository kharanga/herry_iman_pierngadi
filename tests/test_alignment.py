import ast
import doctest
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vision.alignment import (
    SHUTTLESET_SET_DIR,
    check_image_dir,
    check_image_name,
    draw_hit_marker,
    label_scale,
    load_set_shots,
    main,
    open_video,
    pick_spread_shots,
    read_frame,
    scaled_point,
    set_csv_path,
    shot_caption,
    shots_with_hit_point,
    spread_positions,
    write_alignment_checks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIGNMENT_SOURCE_PATH = PROJECT_ROOT / "vision" / "alignment.py"
ALIGNMENT_RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "alignment.md"
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
REAL_SET_DIR = PROJECT_ROOT / SHUTTLESET_SET_DIR
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
SYNTHETIC_MATCH = "Synthetic_Match"
SYNTHETIC_FRAME_COUNT = 10

requires_match_video = pytest.mark.skipif(
    not MATCH_VIDEO_PATH.exists() or not REAL_SET_DIR.exists(),
    reason="needs data/match.mp4 and ShuttleSet (run ingest/fetch_shuttleset.sh)",
)


def write_synthetic_video(video_path: Path, width: int, height: int, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))
    for frame_index in range(frame_count):
        writer.write(np.full((height, width, 3), frame_index * 20, dtype=np.uint8))
    writer.release()


def write_set_csv(set_dir: Path, match_folder: str, set_no: int, rows: list[dict]) -> Path:
    csv_path = set_dir / match_folder / f"set{set_no}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["rally", "ball_round", "frame_num", "type", "hit_x", "hit_y"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def synthetic_shot(rally: int, ball_round: int, frame_num: int, hit_x, hit_y) -> dict:
    return {"rally": rally, "ball_round": ball_round, "frame_num": frame_num, "type": "lob", "hit_x": hit_x, "hit_y": hit_y}


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    video_path = tmp_path / "synthetic.avi"
    write_synthetic_video(video_path, 640, 360, SYNTHETIC_FRAME_COUNT)
    return video_path


@pytest.fixture
def synthetic_set_dir(tmp_path) -> Path:
    set_dir = tmp_path / "set"
    write_set_csv(
        set_dir,
        SYNTHETIC_MATCH,
        1,
        [
            synthetic_shot(1, 1, 0, None, None),
            synthetic_shot(1, 2, 2, 640.0, 360.0),
            synthetic_shot(1, 3, 4, 400.0, 200.0),
            synthetic_shot(2, 1, 6, 800.0, 500.0),
            synthetic_shot(2, 2, 8, 1000.0, 600.0),
        ],
    )
    return set_dir


def test_set_csv_path_and_check_image_dir_follow_spec_layout():
    assert set_csv_path(Path("set"), "M", 3) == Path("set/M/set3.csv")
    assert check_image_dir(Path("data/checks"), "M", 2) == Path("data/checks/M/set2")
    assert check_image_name(1, 2, 300) == "rally1_shot2_frame300.jpg"


def test_label_scale_maps_1280_wide_labels_to_video_width():
    assert label_scale(1280.0) == 1.0
    assert label_scale(640.0) == 0.5
    assert label_scale(1920.0) == 1.5


def test_scaled_point_returns_rounded_ints():
    scaled_x, scaled_y = scaled_point(100.4, 200.6, 1.0)
    assert (scaled_x, scaled_y) == (100, 201)
    assert isinstance(scaled_x, int) and isinstance(scaled_y, int)
    assert scaled_point(640.0, 360.0, 0.5) == (320, 180)


def test_shots_with_hit_point_drops_rows_missing_either_coordinate():
    shots = pd.DataFrame({"ball_round": [1, 2, 3, 4], "hit_x": [None, 1.0, 2.0, None], "hit_y": [None, None, 3.0, 4.0]})
    kept_shots = shots_with_hit_point(shots)
    assert kept_shots["ball_round"].tolist() == [3]
    assert kept_shots.index.tolist() == [0]


def test_shots_with_hit_point_missing_column_raises_key_error():
    with pytest.raises(KeyError):
        shots_with_hit_point(pd.DataFrame({"hit_x": [1.0]}))


@pytest.mark.parametrize(
    "row_count, pick_count, expected_positions",
    [
        (0, 4, []),
        (1, 4, [0]),
        (4, 4, [0, 1, 2, 3]),
        (5, 4, [0, 1, 3, 4]),
        (10, 4, [0, 3, 6, 9]),
        (100, 4, [0, 33, 66, 99]),
    ],
)
def test_spread_positions_includes_first_and_last_without_duplicates(row_count, pick_count, expected_positions):
    assert spread_positions(row_count, pick_count) == expected_positions


def test_spread_positions_with_pick_count_one_returns_one_position():
    assert len(spread_positions(10, 1)) == 1


def test_pick_spread_shots_with_pick_count_one_returns_first_shot_with_hit_point():
    shots = pd.DataFrame({"ball_round": range(1, 6), "hit_x": [None] + [1.0] * 4, "hit_y": [None] + [1.0] * 4})
    assert pick_spread_shots(shots, pick_count=1)["ball_round"].tolist() == [2]


@pytest.mark.parametrize(
    "row_count, pick_count",
    [(10, 0), (10, -1), (10, -5), (0, 0), (-3, 4), (-3, -3), (1, 0)],
)
def test_spread_positions_with_zero_or_negative_counts_returns_empty(row_count, pick_count):
    assert spread_positions(row_count, pick_count) == []


@pytest.mark.parametrize("pick_count", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("row_count", [1, 2, 5, 13, 100])
def test_spread_positions_are_sorted_unique_in_range_and_at_most_pick_count(row_count, pick_count):
    positions = spread_positions(row_count, pick_count)
    assert positions == sorted(set(positions))
    assert all(0 <= position < row_count for position in positions)
    assert len(positions) == min(row_count, pick_count)
    assert positions[0] == 0
    if pick_count > 1:
        assert positions[-1] == row_count - 1


@pytest.mark.parametrize("pick_count", [0, -2])
def test_pick_spread_shots_with_zero_or_negative_pick_count_returns_no_rows(pick_count):
    shots = pd.DataFrame({"ball_round": range(1, 6), "hit_x": [1.0] * 5, "hit_y": [1.0] * 5})
    assert len(pick_spread_shots(shots, pick_count=pick_count)) == 0


def test_pick_spread_shots_on_empty_and_all_serve_sets_returns_no_rows():
    no_hit_points = pd.DataFrame({"ball_round": [1, 2], "hit_x": [None, None], "hit_y": [None, None]})
    assert len(pick_spread_shots(no_hit_points)) == 0
    assert len(pick_spread_shots(no_hit_points.iloc[0:0])) == 0


def test_shot_caption_text():
    assert shot_caption(3, 7, "smash") == "rally 3 shot 7 smash"


def test_draw_hit_marker_draws_red_circle_and_leaves_input_unchanged():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    marked_frame = draw_hit_marker(frame, (150, 120), "caption")
    assert frame.sum() == 0
    assert marked_frame is not frame
    assert marked_frame[120, 162].tolist() == [0, 0, 255]
    assert marked_frame[120, 150].tolist() == [0, 0, 0]


def test_draw_hit_marker_point_outside_frame_does_not_raise():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert draw_hit_marker(frame, (500, 500), "").shape == (50, 50, 3)


def test_load_set_shots_reads_csv_and_reports_missing_path(synthetic_set_dir):
    assert load_set_shots(synthetic_set_dir, SYNTHETIC_MATCH, 1)["frame_num"].tolist() == [0, 2, 4, 6, 8]
    with pytest.raises(FileNotFoundError, match=re.escape(str(synthetic_set_dir / SYNTHETIC_MATCH / "set2.csv"))):
        load_set_shots(synthetic_set_dir, SYNTHETIC_MATCH, 2)


def test_load_set_shots_on_directory_instead_of_csv_raises_file_not_found(tmp_path):
    (tmp_path / "M" / "set1.csv").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_set_shots(tmp_path, "M", 1)


def test_open_video_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        open_video(tmp_path / "missing.mp4")


def test_open_video_unreadable_file_raises_os_error(tmp_path):
    garbage_path = tmp_path / "garbage.mp4"
    garbage_path.write_bytes(b"not a video")
    with pytest.raises(OSError, match="could not open video"):
        open_video(garbage_path)


def test_read_frame_returns_requested_frame_and_none_outside_video(synthetic_video):
    capture = open_video(synthetic_video)
    try:
        assert int(read_frame(capture, 5).mean()) == pytest.approx(100, abs=3)
        assert int(read_frame(capture, 0).mean()) == pytest.approx(0, abs=3)
        assert read_frame(capture, -1) is None
        assert read_frame(capture, SYNTHETIC_FRAME_COUNT) is None
        assert int(read_frame(capture, 2).mean()) == pytest.approx(40, abs=3)
    finally:
        capture.release()


def test_write_alignment_checks_writes_into_checks_dir_with_scaled_marker(
    synthetic_set_dir, synthetic_video, tmp_path, monkeypatch
):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    checks_dir = tmp_path / "checks"
    written_paths = write_alignment_checks(
        SYNTHETIC_MATCH, 1, synthetic_video, set_dir=synthetic_set_dir, checks_dir=checks_dir
    )
    expected_dir = checks_dir / SYNTHETIC_MATCH / "set1"
    assert written_paths == [
        expected_dir / "rally1_shot2_frame2.jpg",
        expected_dir / "rally1_shot3_frame4.jpg",
        expected_dir / "rally2_shot1_frame6.jpg",
        expected_dir / "rally2_shot2_frame8.jpg",
    ]
    assert all(path.is_file() for path in written_paths)
    assert list(working_dir.iterdir()) == []
    first_image = cv2.imread(str(written_paths[0]))
    assert first_image.shape == (360, 640, 3)
    marker_ring_pixel = first_image[180, 320 + 12]
    assert marker_ring_pixel[2] > 200 and marker_ring_pixel[0] < 80


def test_write_alignment_checks_offset_skips_frames_past_the_end(synthetic_set_dir, synthetic_video, tmp_path):
    written_paths = write_alignment_checks(
        SYNTHETIC_MATCH, 1, synthetic_video, offset=3, set_dir=synthetic_set_dir, checks_dir=tmp_path / "checks"
    )
    assert [path.name for path in written_paths] == [
        "rally1_shot2_frame2.jpg",
        "rally1_shot3_frame4.jpg",
        "rally2_shot1_frame6.jpg",
    ]


def test_write_alignment_checks_negative_offset_skips_negative_frames(synthetic_set_dir, synthetic_video, tmp_path):
    written_paths = write_alignment_checks(
        SYNTHETIC_MATCH, 1, synthetic_video, offset=-3, set_dir=synthetic_set_dir, checks_dir=tmp_path / "checks"
    )
    assert [path.name for path in written_paths] == [
        "rally1_shot3_frame4.jpg",
        "rally2_shot1_frame6.jpg",
        "rally2_shot2_frame8.jpg",
    ]


def test_write_alignment_checks_checks_csv_before_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="set CSV not found"):
        write_alignment_checks("No_Match", 1, tmp_path / "missing.mp4", set_dir=tmp_path, checks_dir=tmp_path / "checks")
    assert not (tmp_path / "checks").exists()


def test_write_alignment_checks_set_without_hit_points_writes_nothing(synthetic_video, tmp_path):
    set_dir = tmp_path / "set"
    write_set_csv(set_dir, SYNTHETIC_MATCH, 2, [synthetic_shot(1, 1, 0, None, None)])
    assert write_alignment_checks(SYNTHETIC_MATCH, 2, synthetic_video, set_dir=set_dir, checks_dir=tmp_path / "checks") == []


@pytest.fixture
def project_like_working_dir(tmp_path, monkeypatch, synthetic_set_dir) -> Path:
    working_dir = tmp_path / "project"
    (working_dir / SHUTTLESET_SET_DIR).parent.mkdir(parents=True)
    (working_dir / SHUTTLESET_SET_DIR).symlink_to(synthetic_set_dir, target_is_directory=True)
    monkeypatch.chdir(working_dir)
    return working_dir


def test_main_success_writes_to_data_checks_and_returns_zero(project_like_working_dir, synthetic_video):
    assert main([SYNTHETIC_MATCH, "1", str(synthetic_video)]) == 0
    written_images = sorted(project_like_working_dir.rglob("*.jpg"))
    assert [path.relative_to(project_like_working_dir).parent.as_posix() for path in written_images] == [
        f"data/checks/{SYNTHETIC_MATCH}/set1"
    ] * 4


def test_main_missing_set_csv_prints_error_and_returns_one(project_like_working_dir, synthetic_video, capsys):
    assert main(["No_Such_Match", "1", str(synthetic_video)]) == 1
    assert "error: set CSV not found" in capsys.readouterr().err
    assert list(project_like_working_dir.rglob("*.jpg")) == []


def test_main_missing_video_prints_error_and_returns_one(project_like_working_dir, tmp_path, capsys):
    assert main([SYNTHETIC_MATCH, "1", str(tmp_path / "nope.mp4")]) == 1
    assert "error: video not found" in capsys.readouterr().err


def test_main_unopenable_video_returns_one(project_like_working_dir, tmp_path, capsys):
    garbage_path = tmp_path / "garbage.mp4"
    garbage_path.write_bytes(b"not a video")
    assert main([SYNTHETIC_MATCH, "1", str(garbage_path)]) == 1
    assert "error: could not open video" in capsys.readouterr().err


def test_main_set_without_hit_points_returns_one_without_creating_folder(
    project_like_working_dir, synthetic_set_dir, synthetic_video, capsys
):
    write_set_csv(synthetic_set_dir, SYNTHETIC_MATCH, 2, [synthetic_shot(1, 1, 0, None, None)])
    assert main([SYNTHETIC_MATCH, "2", str(synthetic_video)]) == 1
    error_output = capsys.readouterr().err
    assert "error: the set has no shots with hit_x and hit_y to check" in error_output
    assert "no frames could be read" not in error_output
    assert not (project_like_working_dir / "data" / "checks").exists()


def test_main_empty_set_csv_returns_one_without_creating_folder(
    project_like_working_dir, synthetic_set_dir, synthetic_video, capsys
):
    write_set_csv(synthetic_set_dir, SYNTHETIC_MATCH, 3, [])
    assert main([SYNTHETIC_MATCH, "3", str(synthetic_video)]) == 1
    assert "error: the set has no shots with hit_x and hit_y to check" in capsys.readouterr().err
    assert not (project_like_working_dir / "data" / "checks").exists()


def test_main_offset_past_end_returns_one(project_like_working_dir, synthetic_video, capsys):
    assert main([SYNTHETIC_MATCH, "1", str(synthetic_video), "--offset", "999999"]) == 1
    assert "error: no frames could be read" in capsys.readouterr().err


@pytest.mark.parametrize("bad_arguments", [[SYNTHETIC_MATCH, "4", "video.mp4"], [SYNTHETIC_MATCH, "x", "video.mp4"], []])
def test_main_invalid_arguments_exit_two(project_like_working_dir, bad_arguments):
    with pytest.raises(SystemExit) as raised:
        main(bad_arguments)
    assert raised.value.code == 2


def temp_entries() -> set[str]:
    return {path.name for path in Path(tempfile.gettempdir()).iterdir() if path.name.startswith("tmp")}


def test_alignment_doc_examples_pass_and_remove_their_temp_folder():
    entries_before = temp_entries()
    failure_count, attempted_count = doctest.testfile(
        str(PROJECT_ROOT / "docs" / "vision" / "alignment.md"),
        module_relative=False,
        optionflags=doctest.ELLIPSIS,
        verbose=False,
    )
    assert attempted_count > 0
    assert failure_count == 0
    assert temp_entries() - entries_before == set()


def test_alignment_source_has_no_comments_or_docstrings():
    source_text = ALIGNMENT_SOURCE_PATH.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


def test_root_check_alignment_script_removed_and_no_root_images():
    assert not (PROJECT_ROOT / "check_alignment.py").exists()
    assert list(PROJECT_ROOT.glob("*.jpg")) == []


@requires_match_video
def test_seeked_frame_equals_sequentially_decoded_frame():
    sequential_capture = cv2.VideoCapture(str(MATCH_VIDEO_PATH))
    for _ in range(601):
        _, sequential_frame = sequential_capture.read()
    sequential_capture.release()
    capture = open_video(MATCH_VIDEO_PATH)
    try:
        seeked_frame = read_frame(capture, 600)
    finally:
        capture.release()
    assert np.array_equal(seeked_frame, sequential_frame)


def documented_frames_by_set() -> dict[int, list[str]]:
    rows = re.findall(r"^\| (\d) \| ([^|]+) \|", ALIGNMENT_RESULTS_PATH.read_text(), re.MULTILINE)
    return {int(set_no): [shot.strip() for shot in shots.split(",")] for set_no, shots in rows}


@requires_match_video
@pytest.mark.parametrize("set_no", [1, 2, 3])
def test_results_table_matches_rerun_on_axelsen_ginting(set_no, tmp_path):
    written_paths = write_alignment_checks(
        AXELSEN_GINTING, set_no, MATCH_VIDEO_PATH, set_dir=REAL_SET_DIR, checks_dir=tmp_path / "checks"
    )
    rerun_shots = [
        "{}/{} @ {}".format(*re.fullmatch(r"rally(\d+)_shot(\d+)_frame(\d+)\.jpg", path.name).groups())
        for path in written_paths
    ]
    assert rerun_shots == documented_frames_by_set()[set_no]
    assert all(cv2.imread(str(path)).shape == (720, 1280, 3) for path in written_paths)
