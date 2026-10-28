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


def svg_x(x_m: float) -> float:
    return round(x_m * SVG_UNITS_PER_METER, 1)


def svg_y(y_m: float) -> float:
    return round((COURT_LENGTH_M - y_m) * SVG_UNITS_PER_METER, 1)


def court_svg(shaded_rect: Rect | None, label: str) -> str:
    view_width = COURT_WIDTH_M * SVG_UNITS_PER_METER + 2 * SVG_MARGIN
    view_height = COURT_LENGTH_M * SVG_UNITS_PER_METER + 2 * SVG_MARGIN
    height_px = round(SVG_WIDTH_PX * view_height / view_width)
    parts = [
        f'<svg class="court" viewBox="{-SVG_MARGIN} {-SVG_MARGIN} {view_width:g} {view_height:g}" '
        f'width="{SVG_WIDTH_PX}" height="{height_px}" role="img" aria-label="{label}">',
        f'<rect x="0" y="0" width="{svg_x(COURT_WIDTH_M):g}" height="{svg_y(0):g}" class="floor"/>',
    ]
    if shaded_rect is not None:
        x0, y0, x1, y1 = shaded_rect
        parts.append(
            f'<rect x="{svg_x(x0):g}" y="{svg_y(y1):g}" width="{svg_x(x1) - svg_x(x0):g}" '
            f'height="{svg_y(y0) - svg_y(y1):g}" class="zone"/>'
        )
    for (start_x, start_y), (end_x, end_y) in court_lines_m():
        parts.append(f'<line x1="{svg_x(start_x):g}" y1="{svg_y(start_y):g}" x2="{svg_x(end_x):g}" y2="{svg_y(end_y):g}"/>')
    parts.append(f'<line x1="{-SVG_MARGIN}" y1="{svg_y(NET_Y_M):g}" x2="{svg_x(COURT_WIDTH_M) + SVG_MARGIN:g}" y2="{svg_y(NET_Y_M):g}" class="net"/>')
    parts.append("</svg>")
    return "".join(parts)
