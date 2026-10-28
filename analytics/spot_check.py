import sys
from pathlib import Path

import cv2
import duckdb
import numpy as np

from analytics.court import CENTRE_X_M, COURT_LENGTH_M, COURT_WIDTH_M, DEPTH_BAND_M, NET_Y_M

DATABASE_PATH = Path("data/scout.duckdb")
SPOT_CHECK_PATH = Path("data/checks/shots/random_shots.png")
SPOT_CHECK_SEED = 7
SPOT_CHECK_SHOT_COUNT = 10
PIXELS_PER_METER = 60
MARGIN_PIXELS = 50
LEGEND_WIDTH_PIXELS = 720
SINGLES_SIDELINE_INSET_M = 0.46
SHORT_SERVICE_LINE_FROM_NET_M = 1.98
DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M = 0.76
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COURT_GREEN = (70, 120, 60)
ZONE_GRAY = (140, 160, 140)
HIT_COLOR = (0, 200, 255)
LANDING_COLOR = (0, 0, 230)
OPPONENT_COLOR = (230, 120, 0)

SPOT_CHECK_CANDIDATES_SQL = """
SELECT
    shots.match_id,
    shots.set_no,
    shots.rally,
    shots.ball_round,
    shots.player,
    shots.type_en,
    shots.type_group,
    shots.hit_x,
    shots.hit_y,
    shots.landing_x,
    shots.landing_y,
    shots.opponent_x,
    shots.opponent_y,
    shots.hit_zone,
    shots.landing_zone,
    shots.is_cross
FROM shots
WHERE shots.source = 'shuttleset'
    AND shots.hit_x IS NOT NULL
    AND shots.landing_x IS NOT NULL
ORDER BY shots.match_id, shots.set_no, shots.rally, shots.ball_round
"""


def court_to_image(point_m) -> tuple[int, int]:
    x_m, y_m = point_m
    return (
        int(round(MARGIN_PIXELS + x_m * PIXELS_PER_METER)),
        int(round(MARGIN_PIXELS + (COURT_LENGTH_M - y_m) * PIXELS_PER_METER)),
    )


def court_lines_m() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    near_short_service_y = NET_Y_M - SHORT_SERVICE_LINE_FROM_NET_M
    far_short_service_y = NET_Y_M + SHORT_SERVICE_LINE_FROM_NET_M
    far_long_service_y = COURT_LENGTH_M - DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M
    right_singles_x = COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M
    return [
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),
        ((0.0, 0.0), (0.0, COURT_LENGTH_M)),
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),
        ((SINGLES_SIDELINE_INSET_M, 0.0), (SINGLES_SIDELINE_INSET_M, COURT_LENGTH_M)),
        ((right_singles_x, 0.0), (right_singles_x, COURT_LENGTH_M)),
        ((0.0, near_short_service_y), (COURT_WIDTH_M, near_short_service_y)),
        ((0.0, far_short_service_y), (COURT_WIDTH_M, far_short_service_y)),
        ((0.0, DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M), (COURT_WIDTH_M, DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M)),
        ((0.0, far_long_service_y), (COURT_WIDTH_M, far_long_service_y)),
        ((CENTRE_X_M, 0.0), (CENTRE_X_M, near_short_service_y)),
        ((CENTRE_X_M, far_short_service_y), (CENTRE_X_M, COURT_LENGTH_M)),
    ]


def zone_boundaries_m() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    depth_lines = [
        ((0.0, NET_Y_M + direction * band * DEPTH_BAND_M), (COURT_WIDTH_M, NET_Y_M + direction * band * DEPTH_BAND_M))
        for direction in (-1, 1)
        for band in (1, 2)
    ]
    return depth_lines + [((CENTRE_X_M, 0.0), (CENTRE_X_M, COURT_LENGTH_M))]


def blank_court_image() -> np.ndarray:
    court_width_pixels = int(COURT_WIDTH_M * PIXELS_PER_METER) + 2 * MARGIN_PIXELS
    height_pixels = int(COURT_LENGTH_M * PIXELS_PER_METER) + 2 * MARGIN_PIXELS
    image = np.full((height_pixels, court_width_pixels + LEGEND_WIDTH_PIXELS, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (court_width_pixels, height_pixels), COURT_GREEN, -1)
    for start, end in zone_boundaries_m():
        cv2.line(image, court_to_image(start), court_to_image(end), ZONE_GRAY, 1, cv2.LINE_AA)
    for start, end in court_lines_m():
        cv2.line(image, court_to_image(start), court_to_image(end), WHITE, 2, cv2.LINE_AA)
    cv2.line(image, court_to_image((-0.3, NET_Y_M)), court_to_image((COURT_WIDTH_M + 0.3, NET_Y_M)), BLACK, 3, cv2.LINE_AA)
    cv2.putText(image, "hitter's half (near)", (MARGIN_PIXELS, height_pixels - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(image, "receiver's half (far)", (MARGIN_PIXELS, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return image


def shot_legend_line(shot_number: int, shot: dict) -> str:
    direction = "cross" if shot["is_cross"] else "straight"
    return (
        f"{shot_number}. {shot['type_en']} ({shot['type_group']}) by {shot['player']}: "
        f"{shot['hit_zone']} -> {shot['landing_zone']}, {direction}"
    )


def shot_source_line(shot: dict) -> str:
    return f"    {shot['match_id'][:60]} set {shot['set_no']} rally {shot['rally']} shot {shot['ball_round']}"


def draw_shot(image: np.ndarray, shot_number: int, shot: dict) -> None:
    hit_pixel = court_to_image((shot["hit_x"], shot["hit_y"]))
    landing_pixel = court_to_image((shot["landing_x"], shot["landing_y"]))
    if shot["opponent_x"] is not None:
        cv2.circle(image, court_to_image((shot["opponent_x"], shot["opponent_y"])), 4, OPPONENT_COLOR, -1, cv2.LINE_AA)
    cv2.arrowedLine(image, hit_pixel, landing_pixel, LANDING_COLOR, 2, cv2.LINE_AA, tipLength=0.04)
    cv2.circle(image, hit_pixel, 6, HIT_COLOR, -1, cv2.LINE_AA)
    cv2.putText(image, str(shot_number), (hit_pixel[0] + 7, hit_pixel[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


def draw_legend(image: np.ndarray, shots: list[dict]) -> None:
    legend_x = int(COURT_WIDTH_M * PIXELS_PER_METER) + 2 * MARGIN_PIXELS + 15
    lines = [
        "10 random ShuttleSet shots, court meters, normalized",
        "so the hitter is on the near (bottom) half.",
        "yellow = hit (hitter's feet), red arrow -> landing",
        "(where the receiver reached it), blue = receiver",
        "at the hit. Gray lines = zone bands and centre.",
        "",
    ]
    for shot_number, shot in enumerate(shots, start=1):
        lines += [shot_legend_line(shot_number, shot), shot_source_line(shot)]
    for line_index, text in enumerate(lines):
        cv2.putText(image, text, (legend_x, 40 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)


def random_shots(connection: duckdb.DuckDBPyConnection, shot_count: int, seed: int) -> list[dict]:
    cursor = connection.execute(SPOT_CHECK_CANDIDATES_SQL)
    column_names = [description[0] for description in cursor.description]
    candidates = [dict(zip(column_names, row)) for row in cursor.fetchall()]
    chosen_indexes = np.random.default_rng(seed).choice(len(candidates), size=shot_count, replace=False)
    return [candidates[index] for index in sorted(chosen_indexes)]


def draw_spot_check(
    database_path: Path,
    output_path: Path = SPOT_CHECK_PATH,
    shot_count: int = SPOT_CHECK_SHOT_COUNT,
    seed: int = SPOT_CHECK_SEED,
) -> list[dict]:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        shots = random_shots(connection, shot_count, seed)
    finally:
        connection.close()
    image = blank_court_image()
    for shot_number, shot in enumerate(shots, start=1):
        draw_shot(image, shot_number, shot)
    draw_legend(image, shots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return shots


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SPOT_CHECK_PATH
    shots = draw_spot_check(database_path, output_path)
    for shot_number, shot in enumerate(shots, start=1):
        print(shot_legend_line(shot_number, shot))
    print(f"saved {output_path}")
