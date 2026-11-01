import http.client
import json
import shutil
import threading
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pandas as pd
import pytest

import vision.label_ui as label_ui
from vision.hits import output_paths
from vision.label import handle_key, initial_state, save_labels
from vision.label_ui import (
    ACTION_KEYS,
    LabelSession,
    apply_action,
    clip_command,
    delete_label_at,
    go_to_rally_index,
    jump_to_frame,
    make_handler,
    parse_byte_range,
    progress,
    serve,
    state_json,
)

FRAME_COUNT = 240
FPS = 30.0
requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on the PATH")


def write_test_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for index in range(FRAME_COUNT):
        writer.write(np.full((48, 64, 3), index % 256, dtype=np.uint8))
    writer.release()
    return path


def detector_tables():
    hits = pd.DataFrame({"rally": [1, 1, 1, 2, 2], "frame": [20, 40, 61, 150, 171], "hitter": ["opp", "me", "opp", "me", "opp"], "score": [float("nan"), 0.9, 0.8, float("nan"), 0.7], "reach": float("nan")})
    rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [20, 150], "end_frame": [90, 200], "hits": [3, 2]})
    return hits, rallies


@pytest.fixture
def labelling(tmp_path):
    video = write_test_video(tmp_path / "clip_source.avi")
    hits_dir = tmp_path / "hits"
    hits_file, rallies_file = output_paths(video, 0, FRAME_COUNT - 1, hits_dir)
    hits_file.parent.mkdir(parents=True)
    hits, rallies = detector_tables()
    hits.to_csv(hits_file, index=False)
    rallies.to_csv(rallies_file, index=False)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    session = LabelSession(video, labels_dir / "ui.csv", labels_dir / "ui_rallies.csv", tmp_path / "players", hits_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reference = initial_state(hits, rallies, FRAME_COUNT - 1, session.state.fps)
    yield {"session": session, "port": server.server_address[1], "reference": reference, "labels_dir": labels_dir, "hits": hits, "rallies": rallies}
    server.shutdown()
    server.server_close()
    session.close()


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def act(port, action):
    status, _, body = request(port, "POST", "/api/action", json.dumps(action).encode())
    return status, json.loads(body)


def test_the_state_endpoint_is_the_state_of_vision_label_on_the_same_files(labelling):
    status, headers, body = request(labelling["port"], "GET", "/api/state")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(body) == state_json(labelling["reference"])


def test_every_key_through_http_gives_the_state_and_the_csvs_of_handle_key(labelling):
    reference, labels_dir = labelling["reference"], labelling["labels_dir"]
    keys = ["a", "d", ".", ".", "h", "m", "c", "h", "o", "k", ",", "x", "u", "]", "a", "?", ">", "h", "esc", "r", "[", "-", "u", "+", "<", "h", "o", "v", "]"]
    for key in keys:
        status, state = act(labelling["port"], {"type": "key", "key": key})
        reference = handle_key(reference, key)
        if reference.dirty:
            reference = save_labels(reference, labels_dir / "ref.csv", labels_dir / "ref_rallies.csv")
        assert status == 200
        assert state == state_json(reference), key
    assert (labels_dir / "ui.csv").read_bytes() == (labels_dir / "ref.csv").read_bytes()
    assert (labels_dir / "ui_rallies.csv").read_bytes() == (labels_dir / "ref_rallies.csv").read_bytes()


def test_editing_the_same_frame_twice_autosaves_one_row_with_the_last_values(labelling):
    port, labels_dir = labelling["port"], labelling["labels_dir"]
    act(port, {"type": "jump", "frame": 50})
    for keys in (["h", "m", "c"], ["h", "o", "k"]):
        for key in keys:
            act(port, {"type": "key", "key": key})
        saved = pd.read_csv(labels_dir / "ui.csv")
        assert saved["frame_num"].tolist() == [50]
    assert saved[["player", "type_group", "source"]].values.tolist() == [["opp", "smash", "hand"]]
    status, state = act(port, {"type": "key", "key": "u"})
    assert [(label["player"], label["type_group"]) for label in state["labels"]] == [("me", "clear")]
    assert pd.read_csv(labels_dir / "ui.csv")[["frame_num", "player", "type_group"]].values.tolist() == [[50, "me", "clear"]]


def test_labels_saved_by_the_ui_come_back_after_a_restart_and_in_the_opencv_tool(labelling, tmp_path):
    port, labels_dir, session = labelling["port"], labelling["labels_dir"], labelling["session"]
    for action in ({"type": "key", "key": "a"}, {"type": "key", "key": "l"}, {"type": "rally", "index": 1}, {"type": "key", "key": "a"}, {"type": "key", "key": "n"}):
        act(port, action)
    restarted = LabelSession(session.video_path, labels_dir / "ui.csv", labels_dir / "ui_rallies.csv", tmp_path / "players", tmp_path / "hits")
    try:
        assert state_json(restarted.state)["progress"] == {"rallies_labelled": 2, "rallies": 2, "hits_labelled": 2, "serves_without_length": 0, "unchecked_rallies": 2}
        assert restarted.state.labels == session.state.labels
    finally:
        restarted.close()
    in_opencv_tool = initial_state(labelling["hits"], labelling["rallies"], FRAME_COUNT - 1, session.state.fps, pd.read_csv(labels_dir / "ui.csv"), pd.read_csv(labels_dir / "ui_rallies.csv"))
    assert in_opencv_tool.labels == session.state.labels and in_opencv_tool.rallies == session.state.rallies


def test_jump_rally_and_delete_actions(labelling):
    port = labelling["port"]
    status, state = act(port, {"type": "jump", "frame": 160})
    assert (state["frame"], state["rally_index"]) == (160, 1)
    status, state = act(port, {"type": "jump", "frame": 10 ** 6})
    assert state["frame"] == FRAME_COUNT - 1
    status, state = act(port, {"type": "rally", "index": -3})
    assert (state["frame"], state["rally_index"]) == (20, 0)
    for key in ("a", "c"):
        act(port, {"type": "key", "key": key})
    status, state = act(port, {"type": "delete", "frame": 40})
    assert state["labels"] == [] and state["can_undo"]
    status, state = act(port, {"type": "delete", "frame": 41})
    assert status == 200 and state["labels"] == []


@pytest.mark.parametrize("action", [{"type": "key", "key": "w"}, {"type": "key", "key": "q"}, {"type": "key"}, {"type": "bogus"}, {"type": "jump", "frame": "abc"}, {}])
def test_actions_the_page_never_sends_are_refused_with_400_and_change_nothing(labelling, action):
    status, body = act(labelling["port"], action)
    assert status == 400 and "error" in body
    assert request(labelling["port"], "GET", "/api/state")[2] and json.loads(request(labelling["port"], "GET", "/api/state")[2]) == state_json(labelling["reference"])


@pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b'{"type": "rally", "index": null}', b'"key"', b'{"type": "key", "key": ["a"]}', b'{"type": "key", "key": {}}', b'{"type": "jump", "frame": true}'])
def test_a_malformed_action_body_gets_a_400_answer(labelling, body):
    status, _, answer = request(labelling["port"], "POST", "/api/action", body)
    assert status == 400 and "error" in json.loads(answer)


def test_frames_are_jpeg_and_bad_frame_requests_are_400_or_404(labelling):
    port = labelling["port"]
    status, headers, body = request(port, "GET", "/frame.jpg?f=100")
    assert status == 200 and headers["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8"
    for path in ("/frame.jpg?f=abc", "/frame.jpg", f"/frame.jpg?f={FRAME_COUNT}"):
        assert request(port, "GET", path)[0] == 400
    assert request(port, "GET", "/missing")[0] == 404
    assert request(port, "POST", "/api/other", b"{}")[0] == 404


def test_the_page_lists_the_action_keys_and_no_write_or_quit_key(labelling):
    status, headers, body = request(labelling["port"], "GET", "/")
    page = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert json.dumps(sorted(ACTION_KEYS)) in page
    assert "w" not in ACTION_KEYS and "q" not in ACTION_KEYS


@requires_ffmpeg
def test_the_rally_clip_has_every_rally_frame_and_answers_range_requests(labelling):
    port = labelling["port"]
    status, headers, clip = request(port, "GET", "/clip.mp4?start=20&end=90")
    assert status == 200 and headers["Content-Type"] == "video/mp4" and headers["Accept-Ranges"] == "bytes"
    capture = cv2.VideoCapture(str(labelling["session"].clip_path(20, 90)))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 71
    finally:
        capture.release()
    status, headers, part = request(port, "GET", "/clip.mp4?start=20&end=90", headers={"Range": "bytes=10-19"})
    assert status == 206 and headers["Content-Range"] == f"bytes 10-19/{len(clip)}" and part == clip[10:20]
    status, headers, whole = request(port, "GET", "/clip.mp4?start=20&end=90", headers={"Range": f"bytes={len(clip)}-"})
    assert status == 200 and whole == clip


@requires_ffmpeg
def test_only_the_current_rallys_clip_is_kept(labelling):
    session = labelling["session"]
    first = session.clip_path(20, 90)
    second = session.clip_path(150, 200)
    assert not first.exists() and second.exists()
    assert sorted(path.name for path in session.clip_dir.iterdir()) == ["clip_150_200.mp4"]


@requires_ffmpeg
def test_a_clip_with_its_end_before_its_start_is_a_400_and_the_server_keeps_answering(labelling):
    status, _, body = request(labelling["port"], "GET", "/clip.mp4?start=90&end=20")
    assert status == 400 and "error" in json.loads(body)
    assert request(labelling["port"], "GET", "/api/state")[0] == 200


def test_serve_binds_to_localhost_only_and_removes_the_clip_folder_when_stopped(labelling, monkeypatch, capsys):
    addresses = []

    class StopAtOnce(ThreadingHTTPServer):
        def __init__(self, address, handler):
            addresses.append(address)
            super().__init__(("127.0.0.1", 0), handler)

        def serve_forever(self, poll_interval=0.5):
            raise KeyboardInterrupt

    monkeypatch.setattr(label_ui, "ThreadingHTTPServer", StopAtOnce)
    session = labelling["session"]
    serve(session, port=8766, open_browser=False)
    assert addresses == [("127.0.0.1", 8766)]
    assert not session.clip_dir.exists()
    assert "http://127.0.0.1:" in capsys.readouterr().out


def test_parse_byte_range_cases():
    assert parse_byte_range("bytes=0-99", 1000) == (0, 99)
    assert parse_byte_range("bytes=900-", 1000) == (900, 999)
    assert parse_byte_range("bytes=-100", 1000) == (900, 999)
    assert parse_byte_range("bytes=-5000", 1000) == (0, 999)
    assert parse_byte_range("bytes=990-2000", 1000) == (990, 999)
    for header in (None, "", "bytes=1000-", "bytes=50-10", "items=0-1", "bytes=a-b"):
        assert parse_byte_range(header, 1000) is None
    assert parse_byte_range("bytes=0-1", 0) is None


def test_clip_command_cuts_the_rally_in_seconds_without_audio(tmp_path):
    command = clip_command(tmp_path / "v.mp4", 120, 179, 60.0, tmp_path / "c.mp4", width=320)
    assert command[command.index("-ss") + 1] == "2.000" and command[command.index("-t") + 1] == "1.000"
    assert "-an" in command and "scale=320:-2" in command and command[-1] == str(tmp_path / "c.mp4")


def test_pure_navigation_and_progress():
    hits, rallies = detector_tables()
    state = initial_state(hits, rallies, FRAME_COUNT - 1, FPS)
    assert (jump_to_frame(state, 170).frame, jump_to_frame(state, 170).rally_index) == (170, 1)
    assert jump_to_frame(go_to_rally_index(state, 1), 120).rally_index == 1
    assert jump_to_frame(state, -5).frame == 0
    assert go_to_rally_index(state, 9).frame == 150
    labelled = apply_action(apply_action(state, {"type": "key", "key": "a"}), {"type": "key", "key": "c"})
    assert progress(labelled) == {"rallies_labelled": 1, "rallies": 2, "hits_labelled": 1, "serves_without_length": 0, "unchecked_rallies": 2}
    assert delete_label_at(labelled, 39) is labelled
    assert progress(delete_label_at(labelled, 40))["hits_labelled"] == 0
    with pytest.raises(ValueError):
        apply_action(state, {"type": "key", "key": "zz"})


class GoneClient:
    def write(self, data):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def handler_with_gone_client(session, method, path, body=b""):
    import io

    from email.message import Message

    handler = make_handler(session).__new__(make_handler(session))
    headers = Message()
    headers["Content-Length"] = str(len(body))
    handler.command, handler.path, handler.request_version = method, path, "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.headers, handler.rfile, handler.wfile = headers, io.BytesIO(body), GoneClient()
    handler.close_connection = False
    handler.log_message = lambda *arguments: None
    return handler


@pytest.mark.parametrize("path", ["/", "/api/state", "/frame.jpg?f=3", "/frame.jpg?f=x", "/nowhere"])
def test_a_client_gone_before_a_get_answer_is_dropped_quietly(labelling, path):
    handler = handler_with_gone_client(labelling["session"], "GET", path)
    handler.do_GET()
    assert handler.close_connection is True


def test_a_client_gone_before_a_500_answer_is_dropped_quietly(labelling, monkeypatch):
    def broken_frame(frame_index):
        raise RuntimeError("decoder broke")

    monkeypatch.setattr(labelling["session"], "frame_jpeg", broken_frame)
    handler = handler_with_gone_client(labelling["session"], "GET", "/frame.jpg?f=3")
    handler.do_GET()
    assert handler.close_connection is True


def test_a_client_gone_before_a_post_answer_keeps_the_saved_action(labelling):
    port, labels_dir = labelling["port"], labelling["labels_dir"]
    act(port, {"type": "jump", "frame": 50})
    for body in [json.dumps({"type": "key", "key": key}).encode() for key in ("h", "m", "n")] + [b"{not json", b""]:
        handler = handler_with_gone_client(labelling["session"], "POST", "/api/action", body)
        handler.do_POST()
        assert handler.close_connection is True
    handler = handler_with_gone_client(labelling["session"], "POST", "/elsewhere")
    handler.do_POST()
    assert handler.close_connection is True
    assert pd.read_csv(labels_dir / "ui.csv")[["frame_num", "player", "type_group"]].values.tolist() == [[50, "me", "net"]]
    status, _, body = request(port, "GET", "/api/state")
    assert status == 200 and [(label["player"], label["type_group"]) for label in json.loads(body)["labels"]] == [("me", "net")]


def test_the_server_keeps_answering_after_clients_hang_up_mid_answer(labelling):
    import socket

    for _ in range(5):
        with socket.create_connection(("127.0.0.1", labelling["port"])) as hung_up:
            hung_up.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            hung_up.sendall(b"GET /frame.jpg?f=5 HTTP/1.1\r\nHost: x\r\n\r\n")
    assert request(labelling["port"], "GET", "/api/state")[0] == 200


def test_the_label_keys_through_http_give_the_state_and_the_csvs_of_handle_key(labelling):
    reference, labels_dir = labelling["reference"], labelling["labels_dir"]
    keys = ["h", "s", "h", "m", "s", "c", "k", "o", "l", ".", ".", "h", "o", "s", "esc", "h", "o", "n", ">", "h", "m", "v", "]", "h", "o", "s", "s", ">", "h", "m", "l"]
    for key in keys:
        status, state = act(labelling["port"], {"type": "key", "key": key})
        reference = handle_key(reference, key)
        if reference.dirty:
            reference = save_labels(reference, labels_dir / "ref.csv", labels_dir / "ref_rallies.csv")
        assert status == 200 and state == state_json(reference), key
    saved = pd.read_csv(labels_dir / "ui.csv")
    assert (labels_dir / "ui.csv").read_bytes() == (labels_dir / "ref.csv").read_bytes()
    assert saved[["type_group", "type_detail"]].values.tolist() == [["serve", "serve_long"], ["net", "net"], ["drive", "drive"], ["serve", "serve_short"], ["lift", "lift"]]


def test_the_serve_prompt_shows_in_the_state_and_the_page_has_a_button_per_label(labelling):
    port = labelling["port"]
    for key in ("h", "m", "s"):
        status, state = act(port, {"type": "key", "key": key})
    assert state["pending"]["choosing_serve_length"] is True and state["labels"] == []
    status, state = act(port, {"type": "key", "key": "l"})
    assert state["pending"] is None and [(label["type_group"], label["type_detail"]) for label in state["labels"]] == [("serve", "serve_long")]
    page = request(port, "GET", "/")[2].decode()
    assert "data-keys='[\"s\",\"s\"]'" in page and "data-keys='[\"s\",\"l\"]'" in page
    for key in ("c", "d", "k", "n", "l", "v"):
        assert f'data-key="{key}"' in page, key
    for key in ("t", "p"):
        assert f'data-key="{key}"' not in page, key
    assert "SERVE, press s for SHORT or l for LONG (esc cancels)" in page
