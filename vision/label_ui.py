import argparse
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

from vision.alignment import open_video, read_frame
from vision.hits import HITS_DIR, output_paths
from vision.label import (
    KEY_HELP,
    LEFT_PANEL_WIDTH,
    PLAYER_KEYS,
    STEP_KEYS,
    TYPE_KEYS,
    LabelState,
    handle_key,
    initial_state,
    label_paths,
    load_tracks,
    rally_containing,
    remembered,
    save_labels,
    scaled,
    video_last_frame_and_fps,
)
from vision.overlay import draw_tracking_overlay, frame_state
from vision.players import PLAYERS_DIR

DEFAULT_PORT = 8766
CLIP_WIDTH = 480
JPEG_QUALITY = 85
CLIENT_GONE_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
ACTION_KEYS = set(STEP_KEYS) | set(TYPE_KEYS) | set(PLAYER_KEYS) | {"[", "]", "h", "a", "x", "u", "r", "+", "-", "esc"}
BROWSER_KEY_HELP = (KEY_HELP[0].replace("space play/pause left", "space play/pause the clip"),) + KEY_HELP[1:-1] + ("+ add rally here   - delete rally   1/2/3 loop speed   esc cancel   every change is saved at once",)


def jump_to_frame(state: LabelState, frame: int) -> LabelState:
    frame = max(0, min(int(frame), state.last_frame))
    index = rally_containing(state.rallies, frame)
    return replace(state, frame=frame, rally_index=state.rally_index if index is None else index, pending=None)


def go_to_rally_index(state: LabelState, index: int) -> LabelState:
    if not state.rallies:
        return state
    index = max(0, min(int(index), len(state.rallies) - 1))
    return replace(state, rally_index=index, frame=state.rallies[index].start_frame, pending=None)


def delete_label_at(state: LabelState, frame: int) -> LabelState:
    kept = tuple(label for label in state.labels if label.frame != int(frame))
    if len(kept) == len(state.labels):
        return state
    return replace(remembered(state), labels=kept, dirty=True)


ACTION_FIELDS = {"key": "key", "jump": "frame", "rally": "index", "delete": "frame"}


def whole_number(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
        raise ValueError(f"not a whole number: {value!r}")
    return int(value)


def validated_action(action) -> tuple[str, str | int]:
    if not isinstance(action, dict):
        raise ValueError(f"an action is a JSON object, not {action!r}")
    kind = action.get("type")
    if not isinstance(kind, str) or kind not in ACTION_FIELDS:
        raise ValueError(f"unknown action: {action}")
    field = ACTION_FIELDS[kind]
    if field not in action:
        raise ValueError(f"a {kind} action needs {field!r}: {action}")
    if set(action) - {"type", field}:
        raise ValueError(f"unexpected fields in {action}")
    value = action[field]
    if kind == "key":
        if not isinstance(value, str) or value not in ACTION_KEYS:
            raise ValueError(f"unknown action: {action}")
        return kind, value
    return kind, whole_number(value)


def apply_action(state: LabelState, action) -> LabelState:
    kind, value = validated_action(action)
    if kind == "key":
        return handle_key(state, value)
    if kind == "jump":
        return jump_to_frame(state, value)
    if kind == "rally":
        return go_to_rally_index(state, value)
    return delete_label_at(state, value)


def progress(state: LabelState) -> dict:
    labelled_rallies = {label.rally for label in state.labels}
    return {"rallies_labelled": len(labelled_rallies), "rallies": len(state.rallies), "hits_labelled": len(state.labels)}


def state_json(state: LabelState) -> dict:
    span = state.rally
    in_rally = (lambda frame: span is not None and span.start_frame <= frame <= span.end_frame)
    return {
        "frame": state.frame,
        "last_frame": state.last_frame,
        "fps": state.fps,
        "rally_index": state.rally_index,
        "rally": None if span is None else {"rally": span.rally, "start_frame": span.start_frame, "end_frame": span.end_frame},
        "rally_count": len(state.rallies),
        "labels": [{"rally": label.rally, "frame": label.frame, "player": label.player, "type_group": label.type_group, "source": label.source} for label in state.labels if in_rally(label.frame)],
        "proposals": [{"frame": proposal.frame, "hitter": proposal.hitter} for proposal in state.proposals if in_rally(proposal.frame)],
        "pending": None if state.pending is None else {"frame": state.pending.frame, "source": state.pending.source, "player": state.pending.player},
        "dirty": state.dirty,
        "can_undo": bool(state.history),
        "progress": progress(state),
    }


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes=") or size == 0:
        return None
    first, _, last = header[len("bytes="):].split(",")[0].strip().partition("-")
    try:
        if first == "":
            length = int(last)
            return max(size - length, 0), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None
    if start >= size or start > end:
        return None
    return start, min(end, size - 1)


def checked_clip_span(start_frame: int, end_frame: int, last_frame: int) -> tuple[int, int]:
    if start_frame < 0 or start_frame > end_frame or start_frame > last_frame:
        raise ValueError(f"no clip for frames {start_frame}-{end_frame}: need 0 <= start <= end and start <= {last_frame}")
    return start_frame, min(end_frame, last_frame)


def clip_command(video_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path, width: int = CLIP_WIDTH) -> list[str]:
    return [
        shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_frame / fps:.3f}", "-i", str(video_path), "-t", f"{(end_frame - start_frame + 1) / fps:.3f}",
        "-an", "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]


class LabelSession:
    def __init__(self, video_path: Path, labels_file: Path, rallies_file: Path, players_dir: Path = PLAYERS_DIR, hits_dir: Path = HITS_DIR):
        self.video_path = Path(video_path)
        self.labels_file, self.rallies_file = Path(labels_file), Path(rallies_file)
        last_frame, fps = video_last_frame_and_fps(self.video_path)
        hits_file, rallies_source = output_paths(self.video_path, 0, last_frame, hits_dir)
        if not hits_file.is_file() or not rallies_source.is_file():
            raise FileNotFoundError(f"hit files not found: {hits_file}, {rallies_source} (run: python -m vision.hits {video_path} --start 0 --end {last_frame})")
        self.hits_table, self.rallies_table = pd.read_csv(hits_file), pd.read_csv(rallies_source)
        labels = pd.read_csv(self.labels_file) if self.labels_file.is_file() else None
        label_rallies = pd.read_csv(self.rallies_file) if self.rallies_file.is_file() else None
        self.state = initial_state(self.hits_table, self.rallies_table, last_frame, fps, labels, label_rallies)
        self.tracks = load_tracks(self.video_path, 0, last_frame, players_dir)
        self.lock = threading.Lock()
        self.clip_dir = Path(tempfile.mkdtemp(prefix="label_ui_clips_"))
        self.clip_key: tuple[int, int] | None = None

    def act(self, action: dict) -> dict:
        with self.lock:
            self.state = apply_action(self.state, action)
            if self.state.dirty:
                self.state = save_labels(self.state, self.labels_file, self.rallies_file)
            return state_json(self.state)

    def snapshot(self) -> dict:
        with self.lock:
            return state_json(self.state)

    def frame_jpeg(self, frame_index: int) -> bytes:
        capture = open_video(self.video_path)
        try:
            frame = read_frame(capture, int(frame_index))
        finally:
            capture.release()
        if frame is None:
            raise ValueError(f"could not read frame {frame_index}")
        if self.tracks is not None:
            frame = draw_tracking_overlay(frame, frame_state(int(frame_index), self.tracks[0], self.tracks[1], self.hits_table, self.rallies_table))
        ok, encoded = cv2.imencode(".jpg", scaled(frame, LEFT_PANEL_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return encoded.tobytes()

    def clip_path(self, start_frame: int, end_frame: int) -> Path:
        start_frame, end_frame = checked_clip_span(start_frame, end_frame, self.state.last_frame)
        with self.lock:
            path = self.clip_dir / f"clip_{start_frame}_{end_frame}.mp4"
            if self.clip_key != (start_frame, end_frame) or not path.is_file():
                for old in self.clip_dir.glob("clip_*.mp4"):
                    old.unlink()
                subprocess.run(clip_command(self.video_path, start_frame, end_frame, self.state.fps, path), check=True)
                self.clip_key = (start_frame, end_frame)
            return path

    def close(self) -> None:
        shutil.rmtree(self.clip_dir, ignore_errors=True)


def make_handler(session: LabelSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *arguments):
            return

        def send_body(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload, status: int = HTTPStatus.OK):
            self.send_body(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            try:
                self.answer_get()
            except CLIENT_GONE_ERRORS:
                self.close_connection = True

        def answer_get(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.send_body(HTTPStatus.OK, PAGE_HTML.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self.send_json(session.snapshot())
                elif url.path == "/frame.jpg":
                    self.send_body(HTTPStatus.OK, session.frame_jpeg(int(query["f"][0])), "image/jpeg")
                elif url.path == "/clip.mp4":
                    self.send_clip(int(query["start"][0]), int(query["end"][0]))
                else:
                    self.send_json({"error": f"not found: {url.path}"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except CLIENT_GONE_ERRORS:
                raise
            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                self.send_json({"error": f"could not cut the clip with ffmpeg: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception as error:
                self.send_json({"error": f"server error: {error!r}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def send_clip(self, start_frame: int, end_frame: int):
            path = session.clip_path(start_frame, end_frame)
            data = path.read_bytes()
            byte_range = parse_byte_range(self.headers.get("Range"), len(data))
            if byte_range is None:
                self.send_body(HTTPStatus.OK, data, "video/mp4", {"Accept-Ranges": "bytes"})
                return
            first, last = byte_range
            self.send_body(HTTPStatus.PARTIAL_CONTENT, data[first : last + 1], "video/mp4", {"Accept-Ranges": "bytes", "Content-Range": f"bytes {first}-{last}/{len(data)}"})

        def do_POST(self):
            try:
                self.answer_post()
            except CLIENT_GONE_ERRORS:
                self.close_connection = True

        def answer_post(self):
            if urlparse(self.path).path != "/api/action":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                action = json.loads(self.rfile.read(length) or b"{}")
                self.send_json(session.act(action))
            except CLIENT_GONE_ERRORS:
                raise
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self.send_json({"error": f"server error: {error!r}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def stop_on_terminate(signal_number, frame):
    raise KeyboardInterrupt


def serve(session: LabelSession, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"labelling {session.video_path} at {url} (ctrl-c to stop; every change is saved to {session.labels_file})", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label hits</title>
<style>
body{margin:0;font:14px -apple-system,Helvetica,Arial,sans-serif;background:#111;color:#eee}
header{padding:6px 10px;background:#222;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
main{display:flex;gap:10px;padding:10px;flex-wrap:wrap}
#left{flex:2 1 600px;min-width:320px}#right{flex:1 1 320px;min-width:280px}
#frame{width:100%;display:block;background:#000}#clip{width:100%;display:block;background:#000}
#timeline{position:relative;height:44px;background:#2a2a2a;margin-top:6px;cursor:pointer}
.tick{position:absolute;top:12px;width:2px;height:20px;background:#fc0}
.lab{position:absolute;top:2px;width:3px;height:40px;background:#3c3}
.cur{position:absolute;top:0;width:2px;height:44px;background:#fff}
.playhead{position:absolute;bottom:0;width:6px;height:6px;margin-left:-3px;border-radius:3px;background:#0cf}
#cliptrack{position:relative;height:10px;background:#2a2a2a;margin-top:4px}
button{background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:5px 8px;margin:2px;cursor:pointer}
button:hover{background:#444}.pending{color:#f6f;font-weight:bold}
table{width:100%;border-collapse:collapse}td{padding:3px 4px;border-bottom:1px solid #333}tr.row:hover{background:#222;cursor:pointer}
.small{color:#aaa;font-size:12px}
</style></head><body>
<header><b>Label hits</b><span id="status"></span><span id="progress" class="small"></span><span id="pending" class="pending"></span></header>
<main>
<section id="left"><img id="frame" alt="current frame">
<div id="timeline" title="click to jump"></div>
<div id="buttons">
<div><button data-key="<">-10</button><button data-key=",">-1</button><button data-key=".">+1</button><button data-key=">">+10</button>
<button data-key="[">prev rally</button><button data-key="]">next rally</button></div>
<div><button data-key="h">hit (h)</button><button data-key="a">accept proposal (a)</button><button data-key="m">me (m)</button><button data-key="o">opp (o)</button>
<button data-key="s">serve</button><button data-key="c">clear</button><button data-key="d">drop</button><button data-key="k">smash</button>
<button data-key="n">net</button><button data-key="l">lift</button><button data-key="v">drive</button><button data-key="?">unknown</button></div>
<div><button data-key="x">delete nearest (x)</button><button data-key="u">undo (u)</button><button data-key="r">rally start/end here (r)</button>
<button data-key="+">add rally (+)</button><button data-key="-">remove rally (-)</button><button data-key="esc">cancel (esc)</button></div>
<div class="small" id="help"></div></div></section>
<section id="right"><video id="clip" muted loop playsinline autoplay></video><div id="cliptrack"></div>
<div>loop speed <button data-speed="0.25">0.25x</button><button data-speed="0.5">0.5x</button><button data-speed="1">1x</button></div>
<h4>Hits in this rally</h4><table id="hits"></table></section>
</main>
<script>
let state=null, speed=0.5, clipKey="";
const $=id=>document.getElementById(id);
const ACTION_KEYS=ACTION_KEY_LIST;
$("help").textContent=KEY_HELP_TEXT;
async function post(action){const r=await fetch("/api/action",{method:"POST",body:JSON.stringify(action)});render(await r.json());}
function x(frame,width){const s=state.rally;return s?Math.round((frame-s.start_frame)/Math.max(s.end_frame-s.start_frame,1)*(width-2)):0;}
function render(s){if(s.error){alert(s.error);return;}state=s;
 $("frame").src="/frame.jpg?f="+s.frame;
 const r=s.rally;$("status").textContent="f"+s.frame+"  "+(r?"rally "+r.rally+" of "+s.rally_count+" ["+r.start_frame+"-"+r.end_frame+"]":"no rally")+(s.dirty?" (unsaved)":" (saved)");
 $("progress").textContent=s.progress.rallies_labelled+" of "+s.progress.rallies+" rallies labelled, "+s.progress.hits_labelled+" hits";
 $("pending").textContent=s.pending?("HIT at f"+s.pending.frame+(s.pending.player?" by "+s.pending.player+": pick a type":": pick me or opp")):"";
 const t=$("timeline"),w=t.clientWidth;t.innerHTML="";
 s.proposals.forEach(p=>{const d=document.createElement("div");d.className="tick";d.style.left=x(p.frame,w)+"px";d.title="proposal f"+p.frame+" "+p.hitter;t.appendChild(d);});
 s.labels.forEach(l=>{const d=document.createElement("div");d.className="lab";d.style.left=x(l.frame,w)+"px";d.title=l.frame+" "+l.player+" "+l.type_group;t.appendChild(d);});
 if(r&&s.frame>=r.start_frame&&s.frame<=r.end_frame){const c=document.createElement("div");c.className="cur";c.style.left=x(s.frame,w)+"px";t.appendChild(c);}
 const rows=s.labels.map(l=>"<tr class='row' data-frame='"+l.frame+"'><td>f"+l.frame+"</td><td>"+l.player+"</td><td>"+l.type_group+"</td><td class='small'>"+l.source+"</td><td><button data-delete='"+l.frame+"'>delete</button></td></tr>");
 $("hits").innerHTML=rows.join("")||"<tr><td class='small'>none yet</td></tr>";
 if(r){const key=r.start_frame+"-"+r.end_frame;if(key!==clipKey){clipKey=key;$("clip").src="/clip.mp4?start="+r.start_frame+"&end="+r.end_frame;$("clip").playbackRate=speed;}}
}
$("timeline").addEventListener("click",e=>{if(!state||!state.rally)return;const r=state.rally,w=e.currentTarget.clientWidth;post({type:"jump",frame:Math.round(r.start_frame+e.offsetX/(w-2)*(r.end_frame-r.start_frame))});});
document.addEventListener("click",e=>{const b=e.target;if(b.dataset.key)post({type:"key",key:b.dataset.key});
 if(b.dataset.speed){speed=parseFloat(b.dataset.speed);$("clip").playbackRate=speed;}
 if(b.dataset.delete){e.stopPropagation();post({type:"delete",frame:parseInt(b.dataset.delete)});return;}
 const row=b.closest&&b.closest("tr.row");if(row&&!b.dataset.delete)post({type:"jump",frame:parseInt(row.dataset.frame)});});
document.addEventListener("keydown",e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;let k=e.key;if(k==="ArrowLeft")k=e.shiftKey?"<":",";else if(k==="ArrowRight")k=e.shiftKey?">":".";else if(k==="Escape")k="esc";
 else if(k===" "){const v=$("clip");v.paused?v.play():v.pause();e.preventDefault();return;}
 else if(k==="1"||k==="2"||k==="3"){speed={"1":0.25,"2":0.5,"3":1}[k];$("clip").playbackRate=speed;return;}
 if(ACTION_KEYS.includes(k)){if([",",".","<",">"].includes(k))e.preventDefault();post({type:"key",key:k});}});
setInterval(()=>{const v=$("clip");if(!state||!state.rally||!v.duration)return;const tr=$("cliptrack");const f=v.currentTime*state.fps;const span=state.rally.end_frame-state.rally.start_frame;
 tr.innerHTML="<div class='playhead' style='left:"+Math.round(f/Math.max(span,1)*(tr.clientWidth-2))+"px'></div><div class='cur' style='height:10px;left:"+Math.round((state.frame-state.rally.start_frame)/Math.max(span,1)*(tr.clientWidth-2))+"px'></div>";},100);
fetch("/api/state").then(r=>r.json()).then(render);
</script></body></html>""".replace("ACTION_KEY_LIST", json.dumps(sorted(ACTION_KEYS))).replace("KEY_HELP_TEXT", json.dumps(" | ".join(BROWSER_KEY_HELP)))


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.label_ui", description="Label hits in the browser, rally by rally, starting from the detector's proposals.")
    parser.add_argument("video", type=Path, help="path to the set video")
    parser.add_argument("--match-id", required=True, help="recording id (printed on start)")
    parser.add_argument("--labels", type=Path, help="labels CSV (default data/labels/<video id>.csv; the rallies file sits next to it)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port (default 8766; 0 picks a free one)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the tracking files for the overlay (default data/players)")
    parser.add_argument("--hits-dir", type=Path, default=HITS_DIR, help="root of the hits/rallies files (default data/hits)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    labels_file, rallies_file = label_paths(parsed.video)
    if parsed.labels is not None:
        labels_file = Path(parsed.labels)
        rallies_file = labels_file.with_name(labels_file.stem + "_rallies.csv")
    try:
        session = LabelSession(parsed.video, labels_file, rallies_file, parsed.players_dir, parsed.hits_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"match {parsed.match_id}")
    signal.signal(signal.SIGTERM, stop_on_terminate)
    serve(session, parsed.port, not parsed.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
