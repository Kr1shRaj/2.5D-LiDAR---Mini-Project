from pathlib import Path

import numpy as np

# read one velodyne scan. rows = x, y, z, reflectance. velodyne frame.
def load_scan(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path}: float count {raw.size} not multiple of 4")
    return raw.reshape(-1, 4)


# read semantickitti labels. low 16 bit = semantic class, high 16 bit = instance id
def load_labels(path: str | Path, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint32)
    if n_points is not None and raw.size != n_points:
        raise ValueError(f"{path}: {raw.size} labels but scan has {n_points} points")
    semantic = (raw & 0xFFFF).astype(np.uint16)
    instance = (raw >> 16).astype(np.uint16)
    return semantic, instance


# read calib.txt into named 4x4 matrices (P0..P3, Tr)
def load_calib(path: str | Path) -> dict[str, np.ndarray]:
    calib: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        nums = np.array(values.split(), dtype=np.float64)
        if nums.size != 12:
            raise ValueError(f"{path}: entry {key!r} has {nums.size} values, want 12")
        mat = np.eye(4)
        mat[:3, :4] = nums.reshape(3, 4)
        calib[key.strip()] = mat
    if "Tr" not in calib:
        raise ValueError(f"{path}: missing Tr (velodyne -> camera) entry")
    return calib


# read poses.txt into (M, 4, 4) camera poses. camera frame, not velodyne.
def load_poses(path: str | Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.shape[1] != 12:
        raise ValueError(f"{path}: pose rows have {rows.shape[1]} values, want 12")
    poses = np.tile(np.eye(4), (rows.shape[0], 1, 1))
    poses[:, :3, :4] = rows.reshape(-1, 3, 4)
    return poses
