import math

from analytics.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DEPTH_BAND_M,
    DEPTH_NAMES,
    MIDDLE_COLUMN_LEFT_EDGE_X_M,
    MIDDLE_COLUMN_RIGHT_EDGE_X_M,
    NET_Y_M,
    NOT_OVER_NET_ZONE,
)
from analytics.spot_check import court_lines_m
from analytics.tendencies import split_response_condition

Rect = tuple[float, float, float, float]

SVG_UNITS_PER_METER = 10
SVG_MARGIN = 3
SVG_WIDTH_PX = 54
DIAGRAM_FAMILIES = ("zone_mix", "serve_placement")


def zone_rect_in_near_half(zone: str, left_handed: bool = False) -> Rect:
    depth, side = zone.split("_")
    band_index = DEPTH_NAMES.index(depth)
    y_top = NET_Y_M - band_index * DEPTH_BAND_M
    y_bottom = 0.0 if depth == "rear" else y_top - DEPTH_BAND_M
    if side == "middle":
        return (MIDDLE_COLUMN_LEFT_EDGE_X_M, y_bottom, MIDDLE_COLUMN_RIGHT_EDGE_X_M, y_top)
    is_right_side = (side == "forehand") != left_handed
    if is_right_side:
        return (MIDDLE_COLUMN_RIGHT_EDGE_X_M, y_bottom, COURT_WIDTH_M, y_top)
    return (0.0, y_bottom, MIDDLE_COLUMN_LEFT_EDGE_X_M, y_top)


def rotate_rect_half_turn(rect: Rect) -> Rect:
    x0, y0, x1, y1 = rect
    return (COURT_WIDTH_M - x1, COURT_LENGTH_M - y1, COURT_WIDTH_M - x0, COURT_LENGTH_M - y0)


def bullet_zone_rect(
    hitter_is_me: bool, family: str, zone: str, me_left_handed: bool = False, opponent_left_handed: bool = False
) -> Rect | None:
    if family not in DIAGRAM_FAMILIES or zone == NOT_OVER_NET_ZONE:
        return None
    zone_owner_is_me = hitter_is_me if family == "zone_mix" else not hitter_is_me
    rect = zone_rect_in_near_half(zone, me_left_handed if zone_owner_is_me else opponent_left_handed)
    return rect if zone_owner_is_me else rotate_rect_half_turn(rect)


def reply_zone_rects(
    hitter_is_me: bool, condition: str, me_left_handed: bool = False, opponent_left_handed: bool = False
) -> tuple[Rect, Rect]:
    _, previous_hit_zone, previous_landing_zone = split_response_condition(condition)
    previous_hitter_is_me = not hitter_is_me
    previous_hitter_left_handed = me_left_handed if previous_hitter_is_me else opponent_left_handed
    replier_left_handed = me_left_handed if hitter_is_me else opponent_left_handed
    from_rect = zone_rect_in_near_half(previous_hit_zone, previous_hitter_left_handed)
    to_rect = zone_rect_in_near_half(previous_landing_zone, replier_left_handed)
    return (
        from_rect if previous_hitter_is_me else rotate_rect_half_turn(from_rect),
        to_rect if hitter_is_me else rotate_rect_half_turn(to_rect),
    )


def rect_centre(rect: Rect) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def svg_x(x_m: float) -> float:
    return round(x_m * SVG_UNITS_PER_METER, 1)


def svg_y(y_m: float) -> float:
    return round((COURT_LENGTH_M - y_m) * SVG_UNITS_PER_METER, 1)


def rect_svg(rect: Rect, css_class: str) -> str:
    x0, y0, x1, y1 = rect
    return (
        f'<rect x="{svg_x(x0):g}" y="{svg_y(y1):g}" width="{round(svg_x(x1) - svg_x(x0), 1):g}" '
        f'height="{round(svg_y(y0) - svg_y(y1), 1):g}" class="{css_class}"/>'
    )


ARROWHEAD_LENGTH = 9.0
ARROWHEAD_HALF_WIDTH = 4.5


def arrowhead_points(from_point: tuple[float, float], to_point: tuple[float, float]) -> list[tuple[float, float]]:
    (from_x, from_y), (to_x, to_y) = from_point, to_point
    length = math.hypot(to_x - from_x, to_y - from_y)
    if length == 0:
        return []
    unit_x, unit_y = (to_x - from_x) / length, (to_y - from_y) / length
    base_x, base_y = to_x - unit_x * ARROWHEAD_LENGTH, to_y - unit_y * ARROWHEAD_LENGTH
    return [
        (to_x, to_y),
        (base_x - unit_y * ARROWHEAD_HALF_WIDTH, base_y + unit_x * ARROWHEAD_HALF_WIDTH),
        (base_x + unit_y * ARROWHEAD_HALF_WIDTH, base_y - unit_x * ARROWHEAD_HALF_WIDTH),
    ]


def arrow_svg(from_rect: Rect, to_rect: Rect) -> str:
    (from_x_m, from_y_m), (to_x_m, to_y_m) = rect_centre(from_rect), rect_centre(to_rect)
    from_point, to_point = (svg_x(from_x_m), svg_y(from_y_m)), (svg_x(to_x_m), svg_y(to_y_m))
    head = arrowhead_points(from_point, to_point)
    if not head:
        return ""
    points = " ".join(f"{round(x, 1):g},{round(y, 1):g}" for x, y in head)
    return (
        f'<line x1="{from_point[0]:g}" y1="{from_point[1]:g}" x2="{to_point[0]:g}" y2="{to_point[1]:g}" class="arrow"/>'
        f'<polygon points="{points}" class="arrowhead"/>'
    )


def court_svg(shaded_rect: Rect | None, label: str, reply_rects: tuple[Rect, Rect] | None = None) -> str:
    view_width = COURT_WIDTH_M * SVG_UNITS_PER_METER + 2 * SVG_MARGIN
    view_height = COURT_LENGTH_M * SVG_UNITS_PER_METER + 2 * SVG_MARGIN
    height_px = round(SVG_WIDTH_PX * view_height / view_width)
    parts = [
        f'<svg class="court" viewBox="{-SVG_MARGIN} {-SVG_MARGIN} {view_width:g} {view_height:g}" '
        f'width="{SVG_WIDTH_PX}" height="{height_px}" role="img" aria-label="{label}">',
        f'<rect x="0" y="0" width="{svg_x(COURT_WIDTH_M):g}" height="{svg_y(0):g}" class="floor"/>',
    ]
    if reply_rects is not None:
        parts.append(rect_svg(reply_rects[0], "zone from"))
        parts.append(rect_svg(reply_rects[1], "zone"))
    if shaded_rect is not None:
        x0, y0, x1, y1 = shaded_rect
        parts.append(
            f'<rect x="{svg_x(x0):g}" y="{svg_y(y1):g}" width="{svg_x(x1) - svg_x(x0):g}" '
            f'height="{svg_y(y0) - svg_y(y1):g}" class="zone"/>'
        )
    for (start_x, start_y), (end_x, end_y) in court_lines_m():
        parts.append(f'<line x1="{svg_x(start_x):g}" y1="{svg_y(start_y):g}" x2="{svg_x(end_x):g}" y2="{svg_y(end_y):g}"/>')
    parts.append(f'<line x1="{-SVG_MARGIN}" y1="{svg_y(NET_Y_M):g}" x2="{svg_x(COURT_WIDTH_M) + SVG_MARGIN:g}" y2="{svg_y(NET_Y_M):g}" class="net"/>')
    if reply_rects is not None:
        parts.append(arrow_svg(*reply_rects))
    parts.append("</svg>")
    return "".join(parts)
