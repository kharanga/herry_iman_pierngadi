import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vision.players
from vision.court import NET_Y_M, court_to_pixels, homography_from_corners
from vision.players import (
    CPU_DEVICE,
    DETECTION_BATCH_FRAMES,
    GPU_DEVICE,
    MAX_DETECTION_WIDTH_PX,
    PERSON_CLASS_ID,
    PREFETCHED_CHUNKS,
    chunked,
    default_device,
    detections_of_result,
    empty_detections,
    frames_with_view,
    main,
    prefetched,
    track_people,
    track_players,
    warm_up_tracker,
)
from tests.test_players import (
    MATCH_CALIBRATION_PATH,
    MATCH_CORNERS,
    MATCH_VIDEO_PATH,
    MODEL_PATH,
    court_drawing,
    requires_yolo_and_match_video,
    view_video_and_calibration,
)

MATCH_HOMOGRAPHY = homography_from_corners(MATCH_CORNERS)


def result_with_boxes(boxes_xyxy, track_ids, confidences):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor(boxes_xyxy, dtype=torch.float32),
            id=torch.tensor(track_ids, dtype=torch.float32),
            conf=torch.tensor(confidences, dtype=torch.float32),
        )
    )


class FakeTracker:
    def __init__(self, boxes_per_frame=None):
        self.track_calls = []
        self.predict_calls = []
        self.boxes_per_frame = boxes_per_frame

    def track(self, frames, **keywords):
        self.track_calls.append((list(frames), keywords))
        return [self.result_for(frame) for frame in frames]

    def predict(self, frame, **keywords):
        self.predict_calls.append((frame, keywords))
        return [self.result_for(frame)]

    def result_for(self, frame):
        if self.boxes_per_frame is None:
            return SimpleNamespace(boxes=None)
        return result_with_boxes(*self.boxes_per_frame)


def two_players_boxes():
    feet = court_to_pixels(MATCH_HOMOGRAPHY, [[3.0, 2.0], [3.0, 11.0]])
    boxes = [[foot_x - 20, foot_y - 120, foot_x + 20, foot_y] for foot_x, foot_y in feet]
    return boxes, [7, 9], [0.9, 0.8]


def test_default_device_is_mps_only_when_torch_sees_it(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == GPU_DEVICE == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == CPU_DEVICE == "cpu"
    assert DETECTION_BATCH_FRAMES == 8 and PREFETCHED_CHUNKS == 2


def test_chunked_groups_lazily_with_a_short_last_chunk_and_rejects_bad_sizes():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]
    assert list(chunked([], 3)) == [] and list(chunked(range(2), 5)) == [[0, 1]]
    consumed = []

    def counting():
        for item in range(100):
            consumed.append(item)
            yield item

    chunks = chunked(counting(), 4)
    assert next(chunks) == [0, 1, 2, 3] and consumed == [0, 1, 2, 3]
    for bad_size in (0, -1):
        with pytest.raises(ValueError, match="chunk size must be >= 1"):
            list(chunked(range(3), bad_size))


def test_prefetched_keeps_order_and_reads_at_most_depth_ahead():
    produced = []

    def items():
        for item in range(10):
            produced.append(item)
            yield item

    stream = prefetched(items(), depth=2)
    assert next(stream) == 0
    time.sleep(0.05)
    assert len(produced) <= 4
    assert list(stream) == list(range(1, 10))


def test_prefetched_raises_the_producer_error_after_the_items_before_it():
    def failing():
        yield "a"
        yield "b"
        raise ValueError("decoder failed")

    stream = prefetched(failing())
    assert next(stream) == "a" and next(stream) == "b"
    with pytest.raises(ValueError, match="decoder failed"):
        next(stream)


def test_prefetched_stops_and_joins_the_producer_when_the_consumer_stops_early():
    released = threading.Event()
    produced = []

    def items():
        try:
            for item in range(1000):
                produced.append(item)
                yield item
        finally:
            released.set()

    threads_before = threading.active_count()
    stream = prefetched(items(), depth=3)
    assert next(stream) == 0
    stream.close()
    assert released.is_set() and len(produced) < 1000
    assert threading.active_count() == threads_before


def test_prefetched_consumer_exception_also_stops_the_producer():
    released = threading.Event()

    def items():
        try:
            yield from range(1000)
        finally:
            released.set()

    with pytest.raises(RuntimeError):
        for item in prefetched(items()):
            raise RuntimeError("consumer failed")
    assert released.is_set()


def test_frames_with_view_flags_each_frame_and_keeps_index_and_frame():
    court_frame = court_drawing()
    blank_frame = np.full(court_frame.shape, 90, dtype=np.uint8)
    flagged = list(frames_with_view([(5, court_frame), (6, blank_frame)], MATCH_HOMOGRAPHY, 0.6))
    assert [(index, in_view) for index, _, in_view in flagged] == [(5, True), (6, False)]
    assert flagged[0][1] is court_frame and flagged[1][1] is blank_frame
    assert list(frames_with_view([], MATCH_HOMOGRAPHY, 0.6)) == []


def test_detections_of_result_converts_tensors_and_handles_missing_boxes_or_ids():
    detections = detections_of_result(result_with_boxes([[1, 2, 3, 4], [5, 6, 7, 8]], [3, 1], [0.5, 0.25]))
    np.testing.assert_array_equal(detections.boxes_xyxy, [[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_array_equal(detections.track_ids, [3, 1]) and detections.track_ids.dtype == int
    np.testing.assert_allclose(detections.confidences, [0.5, 0.25])
    for empty in (SimpleNamespace(boxes=None), SimpleNamespace(boxes=SimpleNamespace(id=None, xyxy=None, conf=None))):
        assert detections_of_result(empty).boxes_xyxy.shape == empty_detections().boxes_xyxy.shape == (0, 4)
        assert len(detections_of_result(empty).track_ids) == 0


def test_track_people_sends_the_list_as_one_call_and_returns_one_result_per_frame():
    tracker = FakeTracker(two_players_boxes())
    frames = [np.zeros((720, 1920, 3), dtype=np.uint8) for _ in range(3)]
    detections = track_people(tracker, frames, GPU_DEVICE)
    assert len(detections) == 3 and all(list(each.track_ids) == [7, 9] for each in detections)
    assert len(tracker.track_calls) == 1
    sent_frames, keywords = tracker.track_calls[0]
    assert len(sent_frames) == 3 and all(sent is frame for sent, frame in zip(sent_frames, frames))
    assert keywords["device"] == GPU_DEVICE and keywords["persist"] is True and keywords["classes"] == [PERSON_CLASS_ID]
    assert keywords["imgsz"] == MAX_DETECTION_WIDTH_PX and keywords["tracker"] == "bytetrack.yaml"
    assert track_people(tracker, [np.zeros((360, 640, 3), dtype=np.uint8)])[0].track_ids.tolist() == [7, 9]
    assert tracker.track_calls[-1][1]["imgsz"] == 640 and tracker.track_calls[-1][1]["device"] == CPU_DEVICE


def test_track_people_with_no_frames_does_not_call_the_model():
    tracker = FakeTracker(two_players_boxes())
    assert track_people(tracker, [], GPU_DEVICE) == []
    assert tracker.track_calls == []


def test_warm_up_runs_predict_once_on_a_black_frame_without_touching_the_tracker_state():
    tracker = FakeTracker()
    warm_up_tracker(tracker, (1080, 1920, 3), GPU_DEVICE)
    assert tracker.track_calls == [] and len(tracker.predict_calls) == 1
    frame, keywords = tracker.predict_calls[0]
    assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8 and not frame.any()
    assert keywords["imgsz"] == MAX_DETECTION_WIDTH_PX and keywords["device"] == GPU_DEVICE and keywords["classes"] == [PERSON_CLASS_ID]
    assert "persist" not in keywords and "tracker" not in keywords


def fake_tracked_run(monkeypatch, video_path, calibration_file, batch_frames, device=CPU_DEVICE):
    tracker = FakeTracker(two_players_boxes())
    monkeypatch.setattr(vision.players, "load_person_tracker", lambda model_path: tracker)
    run = track_players(video_path, 0, 7, calibration_file, MODEL_PATH, device=device, batch_frames=batch_frames)
    return tracker, run


@pytest.mark.parametrize("batch_frames, expected_call_sizes", [(1, [1, 1, 1, 1, 1]), (2, [2, 1, 2]), (3, [1, 2, 2]), (8, [5]), (30, [5])])
def test_track_players_drops_cutaways_before_each_call_and_never_sends_an_empty_chunk(view_video_and_calibration, monkeypatch, batch_frames, expected_call_sizes):
    video_path, calibration_file, frame_is_main_view = view_video_and_calibration
    tracker, run = fake_tracked_run(monkeypatch, video_path, calibration_file, batch_frames)
    main_view = [index for index, is_main in enumerate(frame_is_main_view) if is_main]
    assert run.main_view_frames == main_view and run.skipped_frames == [index for index, is_main in enumerate(frame_is_main_view) if not is_main]
    assert [len(frames) for frames, _ in tracker.track_calls] == expected_call_sizes
    assert len(tracker.predict_calls) == 1
    assert all(keywords["device"] == CPU_DEVICE and keywords["imgsz"] == 1280 for _, keywords in tracker.track_calls)
    assert sorted(run.positions["frame"].unique().tolist()) == main_view
    assert run.positions.groupby("frame")["player"].apply(lambda players: sorted(players)).tolist() == [["me", "opp"]] * len(main_view)


def test_track_players_gives_the_same_rows_for_every_batch_size(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    _, per_frame = fake_tracked_run(monkeypatch, video_path, calibration_file, 1)
    for batch_frames in (3, 8):
        _, batched = fake_tracked_run(monkeypatch, video_path, calibration_file, batch_frames)
        assert batched.positions.to_dict("records") == per_frame.positions.to_dict("records")
        assert batched.boxes.to_dict("records") == per_frame.boxes.to_dict("records")
        assert batched.main_view_frames == per_frame.main_view_frames


def test_track_players_warms_up_before_the_first_chunk_on_the_chosen_device(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    order = []
    tracker = FakeTracker(two_players_boxes())
    original_track, original_predict = tracker.track, tracker.predict
    tracker.track = lambda frames, **keywords: (order.append("track"), original_track(frames, **keywords))[1]
    tracker.predict = lambda frame, **keywords: (order.append("predict"), original_predict(frame, **keywords))[1]
    monkeypatch.setattr(vision.players, "load_person_tracker", lambda model_path: tracker)
    track_players(video_path, 0, 7, calibration_file, MODEL_PATH, device=GPU_DEVICE, batch_frames=4)
    assert order[0] == "predict" and order.count("predict") == 1 and order.count("track") == 2
    assert tracker.predict_calls[0][1]["device"] == GPU_DEVICE and tracker.predict_calls[0][0].shape == (720, 1280, 3)
    assert all(keywords["device"] == GPU_DEVICE for _, keywords in tracker.track_calls)


def test_track_players_without_a_device_uses_the_default_device(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    monkeypatch.setattr(vision.players, "default_device", lambda: "cpu")
    tracker, _ = fake_tracked_run(monkeypatch, video_path, calibration_file, 2, device=None)
    assert all(keywords["device"] == "cpu" for _, keywords in tracker.track_calls)


def test_track_players_rejects_a_batch_below_one_and_releases_the_producer(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    threads_before = threading.active_count()
    with pytest.raises(ValueError, match="chunk size must be >= 1"):
        fake_tracked_run(monkeypatch, video_path, calibration_file, 0)
    assert threading.active_count() == threads_before


def test_track_players_reports_a_decode_error_from_the_background_thread(view_video_and_calibration, monkeypatch):
    video_path, calibration_file, _ = view_video_and_calibration
    tracker = FakeTracker(two_players_boxes())
    monkeypatch.setattr(vision.players, "load_person_tracker", lambda model_path: tracker)
    threads_before = threading.active_count()
    with pytest.raises(ValueError, match="no frames could be read"):
        track_players(video_path, 20, 25, calibration_file, MODEL_PATH, device=CPU_DEVICE, batch_frames=2)
    assert tracker.track_calls == [] and threading.active_count() == threads_before


def test_main_passes_device_and_batch_through_and_rejects_bad_values(view_video_and_calibration, monkeypatch, capsys, tmp_path):
    video_path, calibration_file, _ = view_video_and_calibration
    received = {}

    def fake_track_players(*arguments, **keywords):
        received.update(keywords)
        frames = [2, 3, 4, 6, 7]
        return vision.players.TrackingRun(vision.players.pd.DataFrame(columns=vision.players.POSITION_COLUMNS), frames, [0, 1, 5], [])

    monkeypatch.setattr(vision.players, "track_players", fake_track_players)
    monkeypatch.setattr(vision.players, "positions_path", lambda video, start, end: tmp_path / "positions.csv")
    monkeypatch.setattr(vision.players, "boxes_path", lambda video, start, end: tmp_path / "boxes.csv", raising=False)
    base = [str(video_path), "--start", "0", "--end", "7", "--calibration", str(calibration_file), "--model", str(MODEL_PATH)]
    assert main([*base, "--device", "cpu", "--batch-frames", "3"]) == 0
    assert received["device"] == "cpu" and received["batch_frames"] == 3
    received.clear()
    assert main(base) == 0
    assert received["device"] is None and received["batch_frames"] == DETECTION_BATCH_FRAMES
    with pytest.raises(SystemExit) as exit_info:
        main([*base, "--device", "cuda"])
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as exit_info:
        main([*base, "--batch-frames", "two"])
    assert exit_info.value.code == 2


def test_main_batch_below_one_is_reported_as_an_error(view_video_and_calibration, monkeypatch, capsys):
    video_path, calibration_file, _ = view_video_and_calibration
    monkeypatch.setattr(vision.players, "load_person_tracker", lambda model_path: FakeTracker(two_players_boxes()))
    assert main([str(video_path), "--start", "0", "--end", "7", "--calibration", str(calibration_file), "--model", str(MODEL_PATH), "--device", "cpu", "--batch-frames", "0"]) == 1
    assert "error: chunk size must be >= 1" in capsys.readouterr().err


@requires_yolo_and_match_video
def test_gpu_batches_match_the_cpu_frame_by_frame_path_on_the_broadcast_rally(tmp_path):
    batched = track_players(MATCH_VIDEO_PATH, 14700, 14723, MATCH_CALIBRATION_PATH, MODEL_PATH, device=default_device(), batch_frames=8)
    per_frame = track_players(MATCH_VIDEO_PATH, 14700, 14723, MATCH_CALIBRATION_PATH, MODEL_PATH, device=CPU_DEVICE, batch_frames=1)
    assert batched.main_view_frames == per_frame.main_view_frames == list(range(14700, 14724))
    keys = ["frame", "player"]
    merged = batched.positions.merge(per_frame.positions, on=keys, suffixes=("_gpu", "_cpu"))
    assert len(merged) == len(batched.positions) == len(per_frame.positions) == 2 * 24
    assert np.hypot(merged["x_m_gpu"] - merged["x_m_cpu"], merged["y_m_gpu"] - merged["y_m_cpu"]).max() < 0.05
    boxes = batched.boxes.merge(per_frame.boxes, on=keys, suffixes=("_gpu", "_cpu"))
    box_columns = [column for column in batched.boxes.columns if column not in keys and column != "track_id"]
    assert len(boxes) == len(batched.boxes)
    assert max(float((boxes[f"{column}_gpu"] - boxes[f"{column}_cpu"]).abs().max()) for column in box_columns) < 2.0
    assert (merged["y_m_gpu"] < NET_Y_M).tolist() == (merged["player"] == "me").tolist()
