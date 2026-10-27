from collections.abc import Sequence

import numpy as np

from src.preprocess.transforms import velo_to_world


# stack filtered velodyne scans into one world frame xyz cloud. offline reference map only.
def accumulate_world_points(scans_velo: Sequence[np.ndarray], t_world_velo: np.ndarray) -> np.ndarray:
    if len(scans_velo) != len(t_world_velo):
        raise ValueError(f"{len(scans_velo)} scans but {len(t_world_velo)} poses")
    if len(scans_velo) == 0:
        return np.empty((0, 3), dtype=np.float32)
    total = sum(len(s) for s in scans_velo)
    out = np.empty((total, 3), dtype=np.float32)
    at = 0
    for scan, pose in zip(scans_velo, t_world_velo):
        out[at : at + len(scan)] = velo_to_world(scan[:, :3], pose)
        at += len(scan)
    return out
