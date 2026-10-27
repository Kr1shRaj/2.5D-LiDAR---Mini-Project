from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter

from src.mapping.uniform_grid import GridSpec, build_uniform_grid, spec_from_bounds


# coarse ground under one scan. big cells so car roofs do not count as ground.
@dataclass
class Terrain:
    spec: GridSpec
    z: np.ndarray

    # terrain height at xy. NaN where no terrain cell nearby.
    def sample(self, xy: np.ndarray) -> np.ndarray:
        ix = np.floor((xy[:, 0] - self.spec.x_min) / self.spec.resolution).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.spec.y_min) / self.spec.resolution).astype(np.int64)
        ok = (ix >= 0) & (ix < self.spec.nx) & (iy >= 0) & (iy < self.spec.ny)
        out = np.full(len(xy), np.nan)
        out[ok] = self.z[ix[ok], iy[ok]]
        return out


# percentile ground per big cell, then 3x3 min so cell under car borrows road from neighbours
def local_terrain(points: np.ndarray, frame: str, cell: float, ground_percentile: float) -> Terrain:
    if len(points) == 0:
        raise ValueError("no points for terrain")
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    spec = spec_from_bounds(lo[0], hi[0] + 1e-6, lo[1], hi[1] + 1e-6, cell, frame=frame)
    g = build_uniform_grid(points, spec, points_frame=frame, ground_percentile=ground_percentile).ground
    low = minimum_filter(np.where(np.isnan(g), np.inf, g), size=3, mode="constant", cval=np.inf)
    return Terrain(spec=spec, z=np.where(np.isinf(low), np.nan, low))


