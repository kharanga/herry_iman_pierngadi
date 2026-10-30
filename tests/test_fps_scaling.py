from pathlib import Path

import numpy as np
import pytest

from vision.contact import ContactSettings, contact_settings_for_video
from vision.hits import (
    REFERENCE_FPS,
    REFERENCE_FRAME_WIDTH_PX,
    FrameTracks,
    HitDetectionSettings,
    detect_hits,
    frames_for_fps,
    load_frame_tracks,
    settings_for_video,
    video_fps_and_width,
)
from vision.players import MAX_DETECTION_WIDTH_PX

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BROADCAST_VIDEO = PROJECT_ROOT / "data" / "match.mp4"
TRACKED_RANGES = [(14201, 14947), (16551, 17343), (81360, 82116), (107560, 108103), (114673, 116641)]


def test_reference_video_settings_are_the_tuned_defaults():
    assert (REFERENCE_FPS, REFERENCE_FRAME_WIDTH_PX, MAX_DETECTION_WIDTH_PX) == (30.0, 1280, 1280)
    assert settings_for_video(30.0, 1280) == HitDetectionSettings() and contact_settings_for_video(30.0) == ContactSettings()


def test_frame_counts_scale_with_fps_and_speeds_and_radii_with_width():
    base = HitDetectionSettings()
    sixty = settings_for_video(60.0, 1920)
    assert (sixty.velocity_window_frames, sixty.min_hit_gap_frames, sixty.rally_gap_frames, sixty.rest_frame_count) == (
        2 * base.velocity_window_frames, 2 * base.min_hit_gap_frames, 2 * base.rally_gap_frames, 2 * base.rest_frame_count)
    assert sixty.rest_radius_px == pytest.approx(1.5 * base.rest_radius_px)
    assert sixty.speed_floor_px_per_frame == pytest.approx(base.speed_floor_px_per_frame * 1.5 / 2)
    assert sixty.speed_after_min_px_per_frame == pytest.approx(base.speed_after_min_px_per_frame * 1.5 / 2)
    assert (sixty.hit_score_min, sixty.reach_max_body_heights) == (base.hit_score_min, base.reach_max_body_heights)
    assert settings_for_video(25.0, 1280).min_hit_gap_frames == 7 and frames_for_fps(1, 5.0) == 1 and frames_for_fps(20, 59.94) == 40
    contact = contact_settings_for_video(60.0)
    assert (contact.shuttle_max_frame_gap, contact.feet_max_frame_gap) == (4, 6)
    assert (contact.reach_radius_m, contact.intersection_choice) == (ContactSettings().reach_radius_m, ContactSettings().intersection_choice)


def at_double_rate(tracks: FrameTracks, pixel_scale: float) -> FrameTracks:
    count = len(tracks.frames)
    fine = np.arange(2 * count - 1) / 2.0
    lower, upper = np.floor(fine).astype(int), np.ceil(fine).astype(int)

    def interpolated(values):
        return np.column_stack([np.interp(fine, np.arange(count), values[:, column]) for column in range(values.shape[1])])

    def valid_at_both_ends(mask):
        mask = np.asarray(mask, dtype=bool)
        return mask[lower] & mask[upper]

    return FrameTracks(
        frames=(2 * tracks.frames[0] + np.arange(2 * count - 1)).astype(int),
        shuttle_points=interpolated(tracks.shuttle_points) * pixel_scale,
        shuttle_valid=valid_at_both_ends(tracks.shuttle_valid),
        near_boxes=interpolated(tracks.near_boxes) * pixel_scale,
        far_boxes=interpolated(tracks.far_boxes) * pixel_scale,
        main_view=valid_at_both_ends(tracks.main_view),
        shuttle_inpainted=valid_at_both_ends(tracks.shuttle_inpainted),
    )


def hits_within_one_frame(reference_frames: np.ndarray, other_frames: np.ndarray) -> int:
    return int(sum(len(other_frames) and np.min(np.abs(other_frames - frame)) <= 1 for frame in reference_frames))


@pytest.mark.skipif(not all((PROJECT_ROOT / "data" / "hits" / "match" / f"hits_{start}_{end}.csv").exists() for start, end in TRACKED_RANGES) or not BROADCAST_VIDEO.exists(),
                    reason="needs data/match.mp4 and the saved tracks of the five hit-detection ranges")
def test_a_sixty_fps_wider_copy_of_the_broadcast_tracks_gives_the_thirty_fps_hits_with_scaled_settings():
    assert video_fps_and_width(BROADCAST_VIDEO) == (30.0, 1280.0)
    reference = scaled_matches = unscaled_count = scaled_count = 0
    for start, end in TRACKED_RANGES:
        tracks = load_frame_tracks(BROADCAST_VIDEO, start, end)
        thirty = detect_hits(tracks, settings_for_video(30.0, 1280))["frame"].to_numpy()
        sixty_tracks = at_double_rate(tracks, 1920 / 1280)
        back_to_thirty = lambda frames: (frames - 2 * tracks.frames[0]) / 2 + tracks.frames[0]
        scaled = back_to_thirty(detect_hits(sixty_tracks, settings_for_video(60.0, 1920))["frame"].to_numpy())
        unscaled = detect_hits(sixty_tracks, HitDetectionSettings())["frame"].to_numpy()
        assert abs(len(scaled) - len(thirty)) <= 1
        reference += len(thirty)
        scaled_count += len(scaled)
        unscaled_count += len(unscaled)
        scaled_matches += hits_within_one_frame(thirty, scaled)
    assert scaled_matches >= 0.97 * reference and abs(scaled_count - reference) <= 2
    assert unscaled_count > 1.5 * reference
