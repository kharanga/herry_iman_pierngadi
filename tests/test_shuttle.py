import argparse
import ast
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import vision.shuttle
from vision.shuttle import (
    HEATMAP_THRESHOLD,
    INVISIBLE_COORDINATE_THRESHOLD,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    POSITION_COLUMNS,
    HeatmapEnsemble,
    check_ensemble_mode,
    coordinates_below_threshold_to_zero,
    draw_shuttle_trail,
    ensemble_weights,
    heatmap_position,
    inpaint_mask,
    main,
    median_background,
    model_input_image,
    overlay_path,
    padded_window_frame_indices,
    positions_path,
    sampled_frame_indices,
    track_shuttle,
    tracknet_window_input,
    visible_trail,
    weighted_frame_average,
    window_starts,
)
from vision.tracknet import InpaintNet, TrackNet, load_inpaintnet, load_tracknet, read_checkpoint, tracknet_input_channels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISION_DIR = PROJECT_ROOT / "vision"
MATCH_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
TRACKNET_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "tracknetv3" / "TrackNet_best.pt"
INPAINTNET_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "tracknetv3" / "InpaintNet_best.pt"
BUILDER_POSITIONS_PATH = PROJECT_ROOT / "data" / "shuttle" / "match" / "shuttle_14201_14947.csv"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"

requires_checkpoints = pytest.mark.skipif(
    not TRACKNET_CHECKPOINT.exists() or not INPAINTNET_CHECKPOINT.exists(), reason="run ./vision/fetch_tracknetv3.sh first"
)
requires_checkpoints_and_video = pytest.mark.skipif(
    not TRACKNET_CHECKPOINT.exists() or not INPAINTNET_CHECKPOINT.exists() or not MATCH_VIDEO_PATH.exists(),
    reason="needs the TrackNetV3 checkpoints and data/match.mp4",
)


def write_video(video_path: Path, frames: list[np.ndarray]) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return video_path


def test_model_input_image_is_rgb_channels_first_at_model_size():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = 255
    image = model_input_image(frame)
    assert image.shape == (3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH) and image.dtype == np.uint8
    assert image[2].min() == 255 and image[0].max() == 0


def test_median_background_ignores_a_moving_object():
    frames = [np.full((72, 128, 3), 40, dtype=np.uint8) for _ in range(5)]
    frames[2][10:20, 10:20] = 255
    assert int(median_background(frames).max()) == 40


def test_tracknet_window_input_stacks_background_and_frames_scaled_to_unit_range():
    background = np.full((3, 4, 5), 255, dtype=np.uint8)
    window = tracknet_window_input(background, [np.zeros((3, 4, 5), dtype=np.uint8)] * 8)
    assert window.shape == (27, 4, 5) and window.dtype == np.float32
    assert window[:3].max() == 1.0 and window[3:].max() == 0.0
    assert tracknet_input_channels(8, "concat") == 27
    with pytest.raises(ValueError, match="only 'concat'"):
        tracknet_input_channels(8, "subtract")


@pytest.mark.parametrize("sequence_length", [1, 8, 16])
def test_ensemble_weights_are_symmetric_and_sum_to_one(sequence_length):
    weights = ensemble_weights(sequence_length)
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, weights[::-1])
    assert weights.argmax() in (sequence_length // 2, (sequence_length - 1) // 2)


@pytest.mark.parametrize(
    "frame_count, mode, expected_starts",
    [
        (10, "weighted", [0, 1, 2]),
        (8, "weighted", [0]),
        (3, "weighted", [0]),
        (20, "nonoverlap", [0, 8, 12]),
        (16, "nonoverlap", [0, 8]),
        (3, "nonoverlap", [0]),
    ],
)
def test_window_starts_cover_every_frame(frame_count, mode, expected_starts):
    starts = window_starts(frame_count, 8, mode)
    assert starts == expected_starts
    covered = {min(start + offset, frame_count - 1) for start in starts for offset in range(8)}
    assert covered == set(range(frame_count))


def test_unknown_ensemble_mode_raises():
    with pytest.raises(ValueError, match="ensemble mode"):
        check_ensemble_mode("median")
    with pytest.raises(ValueError):
        window_starts(10, 8, "median")


def test_padded_window_repeats_the_last_frame():
    assert padded_window_frame_indices(0, 8, 3) == [0, 1, 2, 2, 2, 2, 2, 2]
    assert padded_window_frame_indices(4, 4, 10) == [4, 5, 6, 7]


def test_heatmap_position_blank_largest_blob_and_scaling():
    blank = np.zeros((288, 512), dtype=np.float32)
    assert heatmap_position(blank, 1280, 720) == (0, 0, 0)
    below_threshold = blank.copy()
    below_threshold[100:110, 200:210] = HEATMAP_THRESHOLD
    assert heatmap_position(below_threshold, 1280, 720) == (0, 0, 0)
    two_blobs = blank.copy()
    two_blobs[10:12, 10:12] = 0.9
    two_blobs[100:110, 200:210] = 0.9
    x, y, visibility = heatmap_position(two_blobs, 1280, 720)
    assert visibility == 1
    assert x == int(205 * 1280 / 512) and y == int(105 * 720 / 288)


def test_heatmap_position_at_the_origin_is_reported_invisible():
    corner = np.zeros((288, 512), dtype=np.float32)
    corner[0:1, 0:1] = 0.9
    assert heatmap_position(corner, 1280, 720) == (0, 0, 0)


def test_inpaint_mask_fills_gaps_between_visible_points_but_not_exits_at_the_top():
    visible_gap = inpaint_mask([300, 310, 0, 0, 320, 330], [1, 1, 0, 0, 1, 1], 36)
    assert visible_gap.tolist() == [0, 0, 1, 1, 0, 0]
    left_through_top = inpaint_mask([300, 20, 0, 0, 20, 300], [1, 1, 0, 0, 1, 1], 36)
    assert left_through_top.tolist() == [0, 0, 0, 0, 0, 0]
    assert inpaint_mask([], [], 36).tolist() == []
    assert inpaint_mask([300, 310], [1, 1], 36).tolist() == [0, 0]


def test_coordinates_below_threshold_become_zero_and_others_stay():
    coordinates = np.array([[0.01, 0.02], [0.5, 0.01], [0.3, 0.4]], dtype=np.float32)
    cleaned = coordinates_below_threshold_to_zero(coordinates)
    np.testing.assert_allclose(cleaned, [[0.0, 0.0], [0.5, 0.01], [0.3, 0.4]])
    assert INVISIBLE_COORDINATE_THRESHOLD == pytest.approx(50 / np.hypot(288, 512))
    np.testing.assert_allclose(coordinates, [[0.01, 0.02], [0.5, 0.01], [0.3, 0.4]])


def test_streaming_heatmap_ensemble_equals_the_batch_weighted_average():
    random = np.random.default_rng(0)
    frame_count, sequence_length = 13, 8
    starts = window_starts(frame_count, sequence_length, "weighted")
    window_heatmaps = random.random((len(starts), sequence_length, 4, 6)).astype(np.float32)
    batch_average = weighted_frame_average(window_heatmaps, starts, frame_count, ensemble_weights(sequence_length))
    ensemble = HeatmapEnsemble(frame_count, sequence_length, "weighted")
    for start, heatmaps in zip(starts, window_heatmaps):
        ensemble.add_window(start, heatmaps)
    streamed = dict(ensemble.pop_frames_before(frame_count))
    assert sorted(streamed) == list(range(frame_count))
    for frame in range(frame_count):
        np.testing.assert_allclose(streamed[frame], batch_average[frame], rtol=1e-5)
    assert ensemble.pop_frames_before(frame_count) == []


def test_sampled_frame_indices_are_sorted_unique_and_inside_the_range():
    assert sampled_frame_indices(10, 12, 300) == [10, 11, 12]
    indices = sampled_frame_indices(0, 10_000, 300)
    assert len(indices) == 300 and indices[0] == 0 and indices[-1] == 10_000


def test_visible_trail_skips_invisible_frames_and_respects_length():
    positions = pd.DataFrame({"frame": [1, 2, 3, 4], "x": [10, 0, 30, 40], "y": [11, 0, 31, 41], "visibility": [1, 0, 1, 1]})
    assert visible_trail(positions, 4, 8) == [(10, 11), (30, 31), (40, 41)]
    assert visible_trail(positions, 4, 2) == [(30, 31), (40, 41)]
    assert visible_trail(positions, 0, 8) == []


def test_draw_shuttle_trail_marks_the_current_position_and_keeps_the_input():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    drawn = draw_shuttle_trail(frame, [(10, 10), (30, 30)])
    assert drawn.sum() > 0 and frame.sum() == 0
    assert draw_shuttle_trail(frame, []).sum() == 0


def test_output_paths_follow_the_video_stem():
    assert positions_path(Path("videos/game.mp4"), 5, 9).as_posix() == "data/shuttle/game/shuttle_5_9.csv"
    assert overlay_path(Path("videos/game.mp4"), 5, 9).as_posix() == "data/checks/shuttle/game_shuttle_5_9.mp4"


def test_networks_produce_the_expected_shapes_on_small_inputs():
    torch.manual_seed(0)
    with torch.no_grad():
        heatmaps = TrackNet(27, 8).eval()(torch.zeros(1, 27, 32, 64))
        coordinates = InpaintNet().eval()(torch.zeros(2, 16, 2), torch.zeros(2, 16, 1))
    assert heatmaps.shape == (1, 8, 32, 64) and float(heatmaps.min()) >= 0 and float(heatmaps.max()) <= 1
    assert coordinates.shape == (2, 16, 2)


def test_missing_checkpoint_raises_with_the_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_tracknetv3.sh"):
        read_checkpoint(tmp_path / "missing.pt")


@requires_checkpoints
def test_published_checkpoints_load_strictly_with_weights_only():
    tracknet, tracknet_sequence_length = load_tracknet(TRACKNET_CHECKPOINT)
    inpaintnet, inpaintnet_sequence_length = load_inpaintnet(INPAINTNET_CHECKPOINT)
    assert (tracknet_sequence_length, inpaintnet_sequence_length) == (8, 16)
    assert not tracknet.training and not inpaintnet.training


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["missing.mp4", "--start", "-1", "--end", "20"], "start frame must be >= 0"),
        (["missing.mp4", "--start", "20", "--end", "10"], "before start frame"),
        (["missing.mp4", "--start", "0", "--end", "20"], "video not found"),
        (["missing.mp4"], "give --start and --end"),
    ],
)
def test_main_errors_exit_one_and_write_nothing(arguments, message, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(arguments) == 1
    assert message in capsys.readouterr().err
    assert list(tmp_path.rglob("*")) == []


def test_main_missing_checkpoint_exits_one(tmp_path, monkeypatch, capsys):
    video_path = write_video(tmp_path / "clip.avi", [np.zeros((72, 128, 3), dtype=np.uint8)] * 20)
    monkeypatch.chdir(tmp_path)
    assert main([str(video_path), "--start", "0", "--end", "19"]) == 1
    assert "checkpoint not found" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("arguments", [["video.mp4", "--ensemble", "median", "--start", "0", "--end", "5"], []])
def test_main_invalid_arguments_exit_two(arguments):
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


def test_shuttleset_rally_padding_is_clamped_at_frame_zero(tmp_path, monkeypatch):
    set_dir = tmp_path / vision.shuttle.SHUTTLESET_SET_DIR / "Early_Match"
    set_dir.mkdir(parents=True)
    pd.DataFrame({"rally": [1, 1], "frame_num": [10, 40]}).to_csv(set_dir / "set1.csv", index=False)
    monkeypatch.chdir(tmp_path)
    parsed = argparse.Namespace(shuttleset_rally=["Early_Match", "1", "1"], start=None, end=None, padding=30)
    assert vision.shuttle.frames_to_track(parsed) == (0, 70)


@requires_checkpoints_and_video
def test_short_run_on_the_match_video_finds_the_shuttle_where_the_builder_did():
    positions = track_shuttle(MATCH_VIDEO_PATH, 14600, 14615, "nonoverlap", TRACKNET_CHECKPOINT, INPAINTNET_CHECKPOINT)
    assert list(positions.columns) == POSITION_COLUMNS
    assert positions["frame"].tolist() == list(range(14600, 14616))
    assert set(positions["visibility"]) <= {0, 1} and positions["visibility"].sum() >= 14
    if BUILDER_POSITIONS_PATH.exists():
        builder = pd.read_csv(BUILDER_POSITIONS_PATH).set_index("frame").loc[14604:14611]
        mine = positions.set_index("frame").loc[14604:14611]
        both_visible = (builder["visibility"] == 1) & (mine["visibility"] == 1)
        distances = np.hypot(builder["x"] - mine["x"], builder["y"] - mine["y"])[both_visible]
        assert both_visible.sum() >= 6 and distances.max() <= 20


@pytest.mark.parametrize("source_name", ["shuttle.py", "tracknet.py"])
def test_shuttle_sources_have_no_comments_docstrings_or_third_party_imports(source_name):
    source_text = (VISION_DIR / source_name).read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []
    assert "third_party" not in source_text and "sys.path" not in source_text
    assert "torch.load" not in source_text or "weights_only=True" in source_text


def test_fetch_script_writes_only_under_data_and_verifies_checksums():
    script_text = (VISION_DIR / "fetch_tracknetv3.sh").read_text()
    assert [line for line in script_text.splitlines()[1:] if line.lstrip().startswith("#")] == []
    assert re.search(r'repo_dir="data/', script_text) and re.search(r'checkpoint_dir="data/', script_text)
    assert script_text.count("shasum -a 256") == 1 and "checksum mismatch" in script_text
    assert "pip install" not in script_text and "requirements.txt" not in script_text
    assert "unzip -tq" in script_text and "HTML page" in script_text
    assert 'mktemp -d "$checkpoint_dir/' in script_text and "trap 'rm -rf" in script_text


def test_third_party_notices_credit_tracknetv3_for_the_derived_network_file():
    notices_text = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    upstream_notice = "Copyright (c) 2024 qaz812345"
    assert upstream_notice in notices_text and "MIT License" in notices_text
    tracknet_section = notices_text.split("## TrackNetV3")[1].split("\n## ")[0]
    assert "`vision/tracknet.py`" in tracknet_section and upstream_notice in tracknet_section


@pytest.mark.parametrize("doc_name", ["vision/shuttle.md", "results/shuttle.md"])
def test_shuttle_docs_have_no_task_ids(doc_name):
    assert re.findall(TASK_ID_PATTERN, (PROJECT_ROOT / "docs" / doc_name).read_text()) == []
