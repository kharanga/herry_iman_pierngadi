import numpy as np
import pandas as pd

JITTER_OFFSETS_FRAMES = (5, 10)
JITTER_SEED = 0
RALLY_KEY_COLUMNS = ["source", "match_id", "set_no", "rally"]


def jitter_label(max_offset_frames: int) -> str:
    return f"hit frames +/- {max_offset_frames}"


def uniform_frame_offsets(shot_count: int, max_offset_frames: int, random_seed: int = JITTER_SEED) -> np.ndarray:
    return np.random.default_rng(random_seed).integers(-max_offset_frames, max_offset_frames + 1, size=shot_count)


def rally_order_index(shot_rows: pd.DataFrame) -> pd.Index:
    return shot_rows.sort_values(RALLY_KEY_COLUMNS + ["ball_round", "frame_num"], kind="stable").index


def jitter_hit_frames(shot_rows: pd.DataFrame, max_offset_frames: int, random_seed: int = JITTER_SEED) -> pd.DataFrame:
    jittered = shot_rows.copy()
    jittered["frame_num"] = shot_rows["frame_num"].astype(float) + uniform_frame_offsets(len(jittered), max_offset_frames, random_seed)
    ordered_frames = jittered.loc[rally_order_index(shot_rows), RALLY_KEY_COLUMNS + ["frame_num"]]
    rallies = ordered_frames.groupby(RALLY_KEY_COLUMNS, sort=False, dropna=False)["frame_num"]
    jittered["previous_frame_num"] = rallies.shift(1).reindex(shot_rows.index)
    jittered["next_frame_num"] = rallies.shift(-1).reindex(shot_rows.index)
    return jittered
