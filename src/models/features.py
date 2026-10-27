from collections.abc import Callable

import numpy as np

from src.mapping.uniform_grid import GridLayers, point_to_cell

STEP_D_M = 0.2  # probe offset for height step: ground change across 40 cm
SLOPE_D_M = 0.5  # probe offset for slope: ground change across 1 m

FEATURES = [
    "rel_mean",  # z mean over local terrain
    "rel_max",  # z max over local terrain = obstacle height
    "rel_ground",  # cell ground over local terrain
    "roughness",  # z std inside cell
    "relief",  # z max - z min inside cell
    "step",  # biggest ground change across 2 * STEP_D_M, x or y
    "slope",  # ground gradient over 2 * SLOPE_D_M
    "range",  # distance to sensor
    "resolution",  # cell size at this spot
    "count",  # observations (fused: scans, reference: points)
    "weight",  # decayed confidence
    "age",  # seconds since last seen
    "logodds",  # obstacle evidence
    "inherited",  # 1 = placeholder from coarse parent
]
LAYERS = ["z_mean", "z_max", "z_min", "ground", "z_var", "count", "weight", "last_seen", "logodds", "inherited",
          "resolution"]

Sampler = Callable[[np.ndarray, list[str]], dict[str, np.ndarray]]


# sampler over plain uniform grid. layers grid lacks (weight, logodds, ...) come back NaN.
def grid_sampler(layers: GridLayers) -> Sampler:
    def sample_many(xy: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        ix, iy, valid = point_to_cell(xy, layers.spec)
        out = {}
        for n in names:
            a = np.full(len(xy), np.nan)
            if n == "resolution":
                a[valid] = layers.spec.resolution
            elif hasattr(layers, n):
                a[valid] = getattr(layers, n)[ix[valid], iy[valid]]
            out[n] = a
        return out

    return sample_many


# per-spot feature table from any map sampler. terrain_z = local terrain under each spot.
def cell_features(sample_many: Sampler, xy: np.ndarray, terrain_z: np.ndarray, sensor_xy: np.ndarray,
                  t_now: float) -> dict[str, np.ndarray]:
    c = sample_many(xy, LAYERS)
    # NaN-safe: count 0 cells have NaN layers already, missing count = 0
    count = np.nan_to_num(c["count"], nan=0.0)
    offsets = {"x+": (STEP_D_M, 0), "x-": (-STEP_D_M, 0), "y+": (0, STEP_D_M), "y-": (0, -STEP_D_M),
               "X+": (SLOPE_D_M, 0), "X-": (-SLOPE_D_M, 0), "Y+": (0, SLOPE_D_M), "Y-": (0, -SLOPE_D_M)}
    # all 8 neighbour probes in one batched lookup
    probes = np.concatenate([xy + np.array(o) for o in offsets.values()])
    ground = sample_many(probes, ["ground"])["ground"].reshape(len(offsets), len(xy))
    g = dict(zip(offsets, ground))
    step_x, step_y = np.abs(g["x+"] - g["x-"]), np.abs(g["y+"] - g["y-"])
    both_nan = np.isnan(step_x) & np.isnan(step_y)
    step = np.where(both_nan, np.nan, np.fmax(step_x, step_y))
    slope = np.hypot(g["X+"] - g["X-"], g["Y+"] - g["Y-"]) / (2 * SLOPE_D_M)
    return {
        "rel_mean": c["z_mean"] - terrain_z,
        "rel_max": c["z_max"] - terrain_z,
        "rel_ground": c["ground"] - terrain_z,
        "roughness": np.sqrt(c["z_var"]),
        "relief": c["z_max"] - c["z_min"],
        "step": step,
        "slope": slope,
        "range": np.hypot(xy[:, 0] - sensor_xy[0], xy[:, 1] - sensor_xy[1]),
        "resolution": c["resolution"],
        "count": count,
        "weight": c["weight"],
        "age": t_now - c["last_seen"],
        "logodds": c["logodds"],
        "inherited": c["inherited"],
    }


# dict of feature arrays -> (N, F) matrix in FEATURES order
def feature_matrix(f: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([f[n] for n in FEATURES])
