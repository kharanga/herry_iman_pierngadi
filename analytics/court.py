import math

import numpy as np

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = COURT_LENGTH_M / 2
CENTRE_X_M = COURT_WIDTH_M / 2
DEPTH_BAND_M = NET_Y_M / 3
MIDDLE_COLUMN_WIDTH_M = 1.0
MIDDLE_COLUMN_LEFT_EDGE_X_M = CENTRE_X_M - MIDDLE_COLUMN_WIDTH_M / 2
MIDDLE_COLUMN_RIGHT_EDGE_X_M = CENTRE_X_M + MIDDLE_COLUMN_WIDTH_M / 2
CROSS_COURT_MIN_LATERAL_M = 1.0
DEPTH_NAMES = ("front", "mid", "rear")
SIDE_NAMES = ("forehand", "middle", "backhand")
NOT_OVER_NET_ZONE = "not_over_net"
ZONE_NAMES = tuple(f"{depth}_{side}" for depth in DEPTH_NAMES for side in SIDE_NAMES)
LANDING_ZONE_NAMES = ZONE_NAMES + (NOT_OVER_NET_ZONE,)

Point = tuple[float, float]


def apply_homography(homography, points) -> np.ndarray:
    point_array = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([point_array, np.ones(len(point_array))]) @ np.asarray(homography, dtype=float).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def diagram_to_court_homography(diagram_near_left: Point, diagram_far_right: Point) -> np.ndarray:
    (near_left_x, near_left_y), (far_right_x, far_right_y) = diagram_near_left, diagram_far_right
    meters_per_unit_across = COURT_WIDTH_M / (far_right_x - near_left_x)
    meters_per_unit_along = COURT_LENGTH_M / (near_left_y - far_right_y)
    return np.array(
        [
            [meters_per_unit_across, 0.0, -near_left_x * meters_per_unit_across],
            [0.0, -meters_per_unit_along, near_left_y * meters_per_unit_along],
            [0.0, 0.0, 1.0],
        ]
    )


def shuttleset_pixel_to_court_homography(shuttleset_homography, near_left_pixel: Point, far_right_pixel: Point) -> np.ndarray:
    diagram_near_left, diagram_far_right = apply_homography(shuttleset_homography, [near_left_pixel, far_right_pixel])
    return diagram_to_court_homography(tuple(diagram_near_left), tuple(diagram_far_right)) @ np.asarray(
        shuttleset_homography, dtype=float
    )


def is_missing(point: Point | None) -> bool:
    return point is None or any(value is None or math.isnan(value) for value in point)


def present_or_none(point: Point | None) -> Point | None:
    return None if is_missing(point) else (float(point[0]), float(point[1]))


def is_on_far_half(point: Point) -> bool:
    return point[1] > NET_Y_M


def rotate_half_turn(point: Point | None) -> Point | None:
    if is_missing(point):
        return None
    return (COURT_WIDTH_M - point[0], COURT_LENGTH_M - point[1])


def hitter_is_on_far_half(
    player: Point | None, hit: Point | None, opponent: Point | None, landing: Point | None
) -> bool | None:
    for hitter_point in (player, hit):
        if not is_missing(hitter_point):
            return is_on_far_half(hitter_point)
    for receiver_point in (opponent, landing):
        if not is_missing(receiver_point):
            return not is_on_far_half(receiver_point)
    return None


def depth_name(distance_from_net_m: float) -> str:
    band_index = min(int(max(distance_from_net_m, 0.0) // DEPTH_BAND_M), len(DEPTH_NAMES) - 1)
    return DEPTH_NAMES[band_index]


def is_in_middle_column(x_m: float) -> bool:
    return MIDDLE_COLUMN_LEFT_EDGE_X_M <= x_m <= MIDDLE_COLUMN_RIGHT_EDGE_X_M


def side_name(x_m: float, left_handed: bool = False) -> str:
    if is_in_middle_column(x_m):
        return "middle"
    is_right_of_centre = x_m > CENTRE_X_M
    return "forehand" if is_right_of_centre != left_handed else "backhand"


def zone_in_near_half(point: Point, left_handed: bool = False) -> str:
    return f"{depth_name(NET_Y_M - point[1])}_{side_name(point[0], left_handed)}"


def hit_zone(hit: Point | None, hitter_left_handed: bool = False) -> str | None:
    if is_missing(hit):
        return None
    return zone_in_near_half(hit, hitter_left_handed)


def landing_zone(landing: Point | None, receiver_left_handed: bool = False) -> str | None:
    if is_missing(landing):
        return None
    if not is_on_far_half(landing):
        return NOT_OVER_NET_ZONE
    return zone_in_near_half(rotate_half_turn(landing), receiver_left_handed)


def is_cross_court(hit: Point | None, landing: Point | None) -> bool | None:
    if is_missing(hit) or is_missing(landing):
        return None
    on_opposite_sides_of_centre = (hit[0] > CENTRE_X_M) != (landing[0] > CENTRE_X_M)
    return on_opposite_sides_of_centre and abs(landing[0] - hit[0]) >= CROSS_COURT_MIN_LATERAL_M


def hitter_point_for_zone(hit: Point | None, player: Point | None) -> Point | None:
    if hit is not None and not is_on_far_half(hit):
        return hit
    return player if player is not None else hit


def shot_geometry(
    hit: Point | None,
    landing: Point | None,
    player: Point | None,
    opponent: Point | None,
    hitter_left_handed: bool = False,
    receiver_left_handed: bool = False,
) -> dict:
    hit, landing, player, opponent = (present_or_none(point) for point in (hit, landing, player, opponent))
    if hitter_is_on_far_half(player, hit, opponent, landing):
        hit, landing, player, opponent = (rotate_half_turn(point) for point in (hit, landing, player, opponent))
    hitter_point = hitter_point_for_zone(hit, player)
    return {
        "hit": hit,
        "landing": landing,
        "player": player,
        "opponent": opponent,
        "hit_zone": hit_zone(hitter_point, hitter_left_handed),
        "landing_zone": landing_zone(landing, receiver_left_handed),
        "is_cross": is_cross_court(hitter_point, landing),
    }
