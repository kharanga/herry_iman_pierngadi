import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PHONE_WIDTH_PX = 390
PREVIEW_HEIGHT_PX = 1500
WINDOW_PADDING_PX = 40
PREVIEW_DIR = Path("data/checks/report")


def harness_html(page_path: Path, width_px: int, height_px: int) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>body{margin:0;background:#888}"
        f"iframe{{display:block;width:{width_px}px;height:{height_px}px;border:0;background:#fff}}</style></head>"
        f'<body><iframe src="{page_path.resolve().as_uri()}"></iframe></body></html>'
    )


def screenshot_at_width(
    page_path: Path,
    output_png: Path,
    width_px: int = PHONE_WIDTH_PX,
    height_px: int = PREVIEW_HEIGHT_PX,
    chrome_path: Path = CHROME_PATH,
) -> Path:
    if not Path(chrome_path).is_file():
        raise FileNotFoundError(f"Chrome not found at {chrome_path}; the screenshot check needs a Chrome binary")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as harness_dir:
        harness_path = Path(harness_dir) / "harness.html"
        harness_path.write_text(harness_html(Path(page_path), width_px, height_px), encoding="utf-8")
        subprocess.run(
            [
                str(chrome_path),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--window-size={width_px + WINDOW_PADDING_PX},{height_px}",
                f"--screenshot={output_png.resolve()}",
                harness_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    return output_png


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m report.preview",
        description=f"Screenshot a report page at phone width ({PHONE_WIDTH_PX} px) with headless Chrome.",
    )
    parser.add_argument("page_path", type=Path, help="report HTML file")
    parser.add_argument(
        "output_png", nargs="?", type=Path, default=None,
        help=f"PNG to write (default {PREVIEW_DIR}/<report folder>_{PHONE_WIDTH_PX}px.png)",
    )
    return parser.parse_args(arguments)


def default_preview_path(page_path: Path) -> Path:
    return PREVIEW_DIR / f"{page_path.parent.name}_{PHONE_WIDTH_PX}px.png"


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    output_png = parsed.output_png or default_preview_path(parsed.page_path)
    print(f"saved {screenshot_at_width(parsed.page_path, output_png)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
