from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter

from src.mapping.uniform_grid import GridSpec, cell_stats_sparse, point_to_cell, spec_from_bounds


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


# percentile ground per big cell, then 3x3 min so cell under car borrows road from neighbours.
# bounds = known xy box (x_min, x_max, y_min, y_max); else taken from points.
def local_terrain(
    points: np.ndarray,
    frame: str,
    cell: float,
    ground_percentile: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> Terrain:
    if len(points) == 0:
        raise ValueError("no points for terrain")
    if bounds is None:
        lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
        bounds = (lo[0], hi[0] + 1e-6, lo[1], hi[1] + 1e-6)
    spec = spec_from_bounds(*bounds, cell, frame=frame)
    if spec.frame != frame:
        raise ValueError(f"points in frame {frame!r} but grid in frame {spec.frame!r}")
    # sparse kernel: only occupied cells sorted, same percentile as build_uniform_grid ground layer
    ix, iy, valid = point_to_cell(points[:, :2], spec)
    ids, _, _, _, _, _, ground = cell_stats_sparse(ix[valid] * spec.ny + iy[valid], points[valid, 2].astype(np.float64),
                                                   ground_percentile)
    g = np.full(spec.nx * spec.ny, np.nan)
    g[ids] = ground
    g = g.reshape(spec.shape)
    low = minimum_filter(np.where(np.isnan(g), np.inf, g), size=3, mode="constant", cval=np.inf)
    return Terrain(spec=spec, z=np.where(np.isinf(low), np.nan, low))


