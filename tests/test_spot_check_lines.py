import pytest

from analytics.court import DEPTH_EDGES_M, NET_Y_M, SHORT_SERVICE_LINE_FROM_NET_M
from analytics.spot_check import court_lines_m


def horizontal_line_ys(lines):
    return sorted({start[1] for start, end in lines if start[1] == end[1]})


def test_the_painted_short_service_lines_stay_at_1_98_m_and_no_zone_edge_is_painted():
    ys = horizontal_line_ys(court_lines_m())
    assert SHORT_SERVICE_LINE_FROM_NET_M == 1.98
    assert any(abs(y - (NET_Y_M - 1.98)) < 1e-9 for y in ys)
    assert any(abs(y - (NET_Y_M + 1.98)) < 1e-9 for y in ys)
    for edge in DEPTH_EDGES_M:
        assert not any(abs(y - (NET_Y_M - edge)) < 1e-9 or abs(y - (NET_Y_M + edge)) < 1e-9 for y in ys), edge


def test_the_centre_line_runs_from_the_baseline_to_the_short_service_line():
    centre = [(start, end) for start, end in court_lines_m() if start[0] == end[0] == 3.05 and start[1] == 0.0]
    assert centre and centre[0][1][1] == pytest.approx(NET_Y_M - 1.98)
