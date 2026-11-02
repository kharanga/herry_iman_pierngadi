import numpy as np

from analytics.court import DEPTH_EDGES_M, NET_Y_M
from vision.court import court_to_pixels, draw_zones, homography_from_corners, text_scale_for_width, zone_boundary_segments

CORNERS = [[100.0, 300.0], [400.0, 300.0], [350.0, 100.0], [150.0, 100.0]]


def test_text_scales_with_the_frame_width_and_never_gets_thinner_than_one_pixel():
    assert text_scale_for_width(1280, 0.5) == (0.5, 1)
    scale, thickness = text_scale_for_width(1920, 0.5)
    assert scale == 0.75 and thickness == 1
    assert text_scale_for_width(3840, 0.5) == (1.5, 2)
    assert text_scale_for_width(320, 0.4)[1] == 1


def test_the_zone_lines_sit_at_the_new_depth_edges():
    segments = zone_boundary_segments()
    assert DEPTH_EDGES_M == (2.75, 4.00)
    assert segments["near_depth_1"][0][1] == NET_Y_M - 2.75 and segments["near_depth_2"][0][1] == NET_Y_M - 4.00
    assert segments["far_depth_1"][0][1] == NET_Y_M + 2.75 and segments["far_depth_2"][0][1] == NET_Y_M + 4.00


def test_the_caption_fits_inside_narrow_and_wide_frames():
    for width, height in ((480, 360), (1920, 1080)):
        scale = width / 480
        corners = [[x * scale, y * scale] for x, y in CORNERS]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        drawn = draw_zones(frame, homography_from_corners(corners))
        bottom = drawn[height - 40 :, :]
        written_columns = np.flatnonzero(bottom.any(axis=(0, 2)))
        assert len(written_columns) and written_columns.min() >= 0 and written_columns.max() <= width - 6
        assert not frame.any()
