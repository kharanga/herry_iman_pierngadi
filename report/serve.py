import argparse
import shutil
import socket
import sys
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from analytics.tendencies import DATABASE_PATH, parse_sets
from report.build import UnknownMatchError, UnknownPlayerError, build_report
from report.page import render_page

REPORTS_DIR = Path("data/reports")
DEFAULT_PORT = 8765


def report_path_for(match_id: str, reports_dir: Path = REPORTS_DIR) -> Path:
    return reports_dir / match_id / "index.html"


def write_report(
    match_id: str,
    me: str | None = None,
    output_path: Path | None = None,
    database_path: Path = DATABASE_PATH,
    sets: tuple[int, ...] | None = None,
) -> Path:
    output_path = report_path_for(match_id) if output_path is None else Path(output_path)
    page = render_page(build_report(match_id, me, database_path, sets))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def local_network_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *arguments):
        return


def serve_directory(directory: Path, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), partial(QuietHandler, directory=str(directory)))
    print(f"Open on your phone: http://{local_network_ip()}:{port}/   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_page(page_path: Path, port: int = DEFAULT_PORT) -> None:
    with tempfile.TemporaryDirectory(prefix="badminton_report_") as served_dir:
        shutil.copyfile(page_path, Path(served_dir) / "index.html")
        serve_directory(Path(served_dir), port)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m report.serve",
        description="Build the scouting report for a match and serve it on the local network.",
    )
    parser.add_argument("match_id", help="match folder name (ShuttleSet) or own recording id")
    parser.add_argument("--me", help="which player is me: A or B for ShuttleSet (default A, the winner); me for own footage")
    parser.add_argument("--sets", help="comma-separated set numbers, e.g. 1 or 1,2 (default: whole match)")
    parser.add_argument("--out", type=Path, help="HTML file to write (default data/reports/<match_id>/index.html)")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to serve on (default {DEFAULT_PORT})")
    parser.add_argument("--no-serve", action="store_true", help="only write the HTML file")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        output_path = write_report(parsed.match_id, parsed.me, parsed.out, parsed.database, parse_sets(parsed.sets))
    except (UnknownMatchError, UnknownPlayerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Report written: {output_path}")
    if not parsed.no_serve:
        serve_page(output_path, parsed.port)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
