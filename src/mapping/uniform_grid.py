from dataclasses import dataclass

import numpy as np

# Layer arrays are indexed [ix, iy]: ix along +x, iy along +y.
# Cell (ix, iy) covers x in [x_min + ix*res, x_min + (ix+1)*res), same for y.


# fixed grid layout. frame tag stops mixing velodyne and world points.
@dataclass(frozen=True)
class GridSpec:
    x_min: float
    y_min: float
    resolution: float
    nx: int
    ny: int
    frame: str

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"grid shape must be positive, got {(self.nx, self.ny)}")
        if not self.frame:
            raise ValueError("grid frame must be named (e.g. 'velo', 'world')")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.resolution

    @property
    def y_max(self) -> float:
        return self.y_min + self.ny * self.resolution


# make square grid of side `extent` metres centred on (cx, cy)
def centered_spec(extent: float, resolution: float, frame: str, cx: float = 0.0, cy: float = 0.0) -> GridSpec:
    n = int(round(extent / resolution))
    if not np.isclose(n * resolution, extent):
        raise ValueError(f"extent {extent} not a whole number of {resolution} m cells")
    half = n * resolution / 2.0
    return GridSpec(x_min=cx - half, y_min=cy - half, resolution=resolution, nx=n, ny=n, frame=frame)


# smallest grid snapped to whole cells that covers box [x_min, x_max] x [y_min, y_max]
def spec_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float, resolution: float, frame: str) -> GridSpec:
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(f"bad bounds x[{x_min}, {x_max}] y[{y_min}, {y_max}]")
    x0 = np.floor(x_min / resolution) * resolution
    y0 = np.floor(y_min / resolution) * resolution
    # float division can round up one cell. step back so min point stays inside.
    x0 = x0 - resolution if x0 > x_min else x0
    y0 = y0 - resolution if y0 > y_min else y0
    nx = int(np.floor((x_max - x0) / resolution)) + 1
    ny = int(np.floor((y_max - y0) / resolution)) + 1
    return GridSpec(x_min=float(x0), y_min=float(y0), resolution=resolution, nx=nx, ny=ny, frame=frame)


# turn lidar xy into grid cell index. valid = point lands inside grid.
def point_to_cell(xy: np.ndarray, spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.floor((xy[:, 0] - spec.x_min) / spec.resolution).astype(np.int64)
    iy = np.floor((xy[:, 1] - spec.y_min) / spec.resolution).astype(np.int64)
    valid = (ix >= 0) & (ix < spec.nx) & (iy >= 0) & (iy < spec.ny)
    return ix, iy, valid


# centre xy of every cell, shape (nx, ny, 2)
def cell_centers(spec: GridSpec) -> np.ndarray:
    xs = spec.x_min + (np.arange(spec.nx) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(spec.ny) + 0.5) * spec.resolution
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy], axis=-1)


# per cell height layers. count == 0 means unobserved, every float layer is NaN there.
@dataclass
class GridLayers:
    spec: GridSpec
    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_var: np.ndarray
    ground: np.ndarray
    ground_percentile: float

    @property
    def observed(self) -> np.ndarray:
        return self.count > 0

    @property
    def roughness(self) -> np.ndarray:
        return np.sqrt(self.z_var)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.count, self.z_min, self.z_max, self.z_mean, self.z_var, self.ground))


# per cell percentile of z, linear interp like np.percentile.
def _cell_percentile(flat: np.ndarray, z: np.ndarray, count: np.ndarray, percentile: float) -> np.ndarray:
    order = np.lexsort((z, flat))
    z_sorted = z[order]
    starts = np.concatenate(([0], np.cumsum(count)[:-1]))
    out = np.full(count.shape, np.nan)
    hit = count > 0
    pos = (percentile / 100.0) * (count[hit] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    frac = pos - lo
    base = starts[hit]
    out[hit] = z_sorted[base + lo] * (1.0 - frac) + z_sorted[base + hi] * frac
    return out


# bin points into uniform 2.5d grid. points must be in spec.frame. outside points dropped.
def build_uniform_grid(
    points: np.ndarray,
    spec: GridSpec,
    points_frame: str,
    ground_percentile: float,
) -> GridLayers:
    if points_frame != spec.frame:
        raise ValueError(f"points in frame {points_frame!r} but grid in frame {spec.frame!r}")
    if not 0.0 <= ground_percentile <= 100.0:
        raise ValueError(f"ground_percentile must be in [0, 100], got {ground_percentile}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must be (N, 3+), got {points.shape}")

    pts = points[np.all(np.isfinite(points[:, :3]), axis=1)]
    ix, iy, valid = point_to_cell(pts[:, :2], spec)
    flat = ix[valid] * spec.ny + iy[valid]
    z = pts[valid, 2].astype(np.float64)
    n_cells = spec.nx * spec.ny

    count = np.bincount(flat, minlength=n_cells)
    hit = count > 0

    z_sum = np.bincount(flat, weights=z, minlength=n_cells)
    z_mean = np.full(n_cells, np.nan)
    z_mean[hit] = z_sum[hit] / count[hit]

    # two pass variance. avoid sum-of-squares cancellation at -1.73 m offset.
    sq_dev = np.bincount(flat, weights=(z - z_mean[flat]) ** 2, minlength=n_cells)
    z_var = np.full(n_cells, np.nan)
    z_var[hit] = sq_dev[hit] / count[hit]

    z_min = np.full(n_cells, np.inf)
    np.minimum.at(z_min, flat, z)
    z_max = np.full(n_cells, -np.inf)
    np.maximum.at(z_max, flat, z)
    z_min[~hit] = np.nan
    z_max[~hit] = np.nan

    ground = _cell_percentile(flat, z, count, ground_percentile)

    shape = spec.shape
    return GridLayers(
        spec=spec,
        count=count.reshape(shape),
        z_min=z_min.reshape(shape),
        z_max=z_max.reshape(shape),
        z_mean=z_mean.reshape(shape),
        z_var=z_var.reshape(shape),
        ground=ground.reshape(shape),
        ground_percentile=ground_percentile,
    )
