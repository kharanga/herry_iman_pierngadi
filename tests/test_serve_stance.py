import numpy as np
import pandas as pd

from analytics.court import CENTRE_X_M, NET_Y_M, SHORT_SERVICE_LINE_FROM_NET_M
from vision.movement import MovementSettings
from vision.serve_stance import ServeGuess, diagonal, first_mover, in_service_court, later_of_close_guesses, runs_of, serve_for_stretch, serve_hit_near

SETTINGS = MovementSettings(fps=30.0)


def test_the_service_court_is_behind_the_short_service_line_on_either_half():
    y = np.array([NET_Y_M - SHORT_SERVICE_LINE_FROM_NET_M - 0.1, NET_Y_M - 1.0, NET_Y_M + 3.0, -0.4, -0.6, NET_Y_M - SHORT_SERVICE_LINE_FROM_NET_M + 0.4])
    x = np.full(len(y), 2.0)
    assert in_service_court(x, y).tolist() == [True, False, True, True, False, True]
    assert in_service_court(np.array([-0.6, 6.6, 7.0]), np.full(3, 2.0)).tolist() == [False, True, False]


def test_diagonal_means_opposite_sides_of_the_centre_line():
    assert diagonal(np.array([CENTRE_X_M + 1, CENTRE_X_M - 1, CENTRE_X_M + 1]), np.array([CENTRE_X_M - 1, CENTRE_X_M + 1, CENTRE_X_M + 0.5])).tolist() == [True, True, False]


def test_runs_bridge_short_gaps_and_drop_short_runs():
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert runs_of(mask, 3, 1) == [(0, 4), (8, 13)]
    assert runs_of(mask, 1, 0) == [(0, 1), (3, 4), (8, 11), (13, 13)]
    assert runs_of(np.zeros(4, dtype=bool), 1, 0) == []


def test_the_first_player_to_move_after_the_stance_serves():
    speeds = {"me": np.array([0, 0, 0, 0.2, 0.3, 1.5, 1.5]), "opp": np.array([0, 0, 0, 1.0, 1.0, 1.0, 1.0])}
    assert first_mover(speeds, 2, 4, 0.8) == "opp"
    assert first_mover(speeds, 2, 4, 1.2) == "me"
    assert first_mover({"me": np.zeros(5), "opp": np.zeros(5)}, 1, 3, 0.8) is None


def test_the_serve_frame_is_the_servers_hit_nearest_the_stance_end_within_the_window():
    hits = pd.DataFrame({"frame": [90, 100, 112, 130], "hitter": ["me", "opp", "me", "me"], "score": 0.9})
    assert serve_hit_near(hits, "me", 95, 125, 105) == 112
    assert serve_hit_near(hits, "opp", 95, 125, 105) == 100
    assert serve_hit_near(hits, "me", 113, 125, 105) is None
    assert serve_hit_near(hits.iloc[:0], "me", 0, 999, 5) is None


def guess(frame):
    return ServeGuess(frame, "me", frame - 30, frame, True)


def test_close_guesses_keep_the_later_and_a_stretch_takes_the_guess_nearest_its_start():
    kept = later_of_close_guesses([guess(100), guess(150), guess(300), guess(500)], 90)
    assert [g.frame for g in kept] == [150, 300, 500]
    guesses = [guess(100), guess(200), guess(400)]
    assert serve_for_stretch(guesses, 210, SETTINGS).frame == 200
    assert serve_for_stretch(guesses, 1000, SETTINGS) is None
    assert serve_for_stretch(guesses, 400 - 30, SETTINGS).frame == 400 and serve_for_stretch(guesses, 400 - 31, SETTINGS) is None
    assert serve_for_stretch([guess(100)], 100 + 60 + 1, SETTINGS) is None
