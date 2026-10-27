from dataclasses import dataclass, field

import numpy as np

from src.mapping.uniform_grid import GridSpec, build_uniform_grid, centered_spec, point_to_cell

# Sensor-centred multi-resolution grid. Each band = annulus r in [r_min, r_max) with own cell size.
# A point goes to exactly one band, picked by its own horizontal range r = sqrt(x^2 + y^2).
# A band stores only cells whose square touches its annulus, as flat 1D layers (no dense square).
# Cells on a band edge are partial: each band only sees points from its own side of the edge.


# one distance ring: range [r_min, r_max) metres, square cells of `resolution` metres
@dataclass(frozen=True)
class Band:
    r_min: float
    r_max: float
    resolution: float


# bands must be ascending, touching, no gap, no overlap. lattice must fit band square.
def validate_bands(bands: tuple[Band, ...] | list[Band]) -> None:
    if not bands:
        raise ValueError("need at least one band")
    for k, b in enumerate(bands):
        if not (0.0 <= b.r_min < b.r_max) or b.resolution <= 0:
            raise ValueError(f"bad band {b}")
        n = 2.0 * b.r_max / b.resolution
        if not np.isclose(n, round(n)):
            raise ValueError(f"band {b}: 2*r_max not whole number of cells")
        if k > 0 and not np.isclose(bands[k - 1].r_max, b.r_min):
            raise ValueError(f"band {k - 1} ends at {bands[k - 1].r_max} but band {k} starts at {b.r_min}")


# mask of spec cells whose square touches annulus [r_min, r_max) around center
def annulus_mask(spec: GridSpec, band: Band, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    x0 = spec.x_min + np.arange(spec.nx) * spec.resolution - center[0]
    y0 = spec.y_min + np.arange(spec.ny) * spec.resolution - center[1]
    x1, y1 = x0 + spec.resolution, y0 + spec.resolution
    # nearest and farthest distance from center to each cell square
    near_x = np.maximum.reduce([x0, -x1, np.zeros_like(x0)])
    near_y = np.maximum.reduce([y0, -y1, np.zeros_like(y0)])
    far_x = np.maximum(np.abs(x0), np.abs(x1))
    far_y = np.maximum(np.abs(y0), np.abs(y1))
    d_near = np.hypot(near_x[:, None], near_y[None, :])
    d_far = np.hypot(far_x[:, None], far_y[None, :])
    return (d_near < band.r_max) & (d_far > band.r_min)


# square lattice for band centred on sensor + mask of cells that touch annulus [r_min, r_max)
def band_lattice(band: Band, frame: str) -> tuple[GridSpec, np.ndarray]:
    spec = centered_spec(2.0 * band.r_max, band.resolution, frame=frame)
    return spec, annulus_mask(spec, band)


# one band worth of compact cell layers. flat arrays line up with `flat_ids`.
@dataclass
class BandLayers:
    band: Band
    spec: GridSpec
    flat_ids: np.ndarray
    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_var: np.ndarray
    ground: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # optional layers, e.g. fused confidence

    @property
    def observed(self) -> np.ndarray:
        return self.count > 0

    # named layer: core field or extra
    def layer(self, name: str) -> np.ndarray:
        return self.extra[name] if name in self.extra else getattr(self, name)

    @property
    def n_cells(self) -> int:
        return int(self.flat_ids.size)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.count, self.z_min, self.z_max, self.z_mean, self.z_var, self.ground))

    # dense [ix, iy] copy of one layer for plotting. cells outside band = NaN.
    def dense(self, name: str) -> np.ndarray:
        out = np.full(self.spec.nx * self.spec.ny, np.nan)
        out[self.flat_ids] = self.layer(name)
        return out.reshape(self.spec.shape)


# full foveated map: bands near to far, rings around `center` (sensor xy in map frame)
@dataclass
class FoveatedLayers:
    bands: list[BandLayers]
    frame: str
    ground_percentile: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.bands)

    @property
    def n_cells(self) -> int:
        return sum(b.n_cells for b in self.bands)

    # find (band index, compact cell index) for query xy. -1 = outside every band.
    def lookup(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.hypot(xy[:, 0] - self.center[0], xy[:, 1] - self.center[1])
        band_idx = np.full(len(xy), -1, dtype=np.int64)
        cell_idx = np.full(len(xy), -1, dtype=np.int64)
        for k, bl in enumerate(self.bands):
            sel = np.flatnonzero((r >= bl.band.r_min) & (r < bl.band.r_max))
            ix, iy, valid = point_to_cell(xy[sel], bl.spec)
            flat = ix * bl.spec.ny + iy
            pos = np.searchsorted(bl.flat_ids, flat)
            pos_c = np.minimum(pos, bl.n_cells - 1)
            hit = valid & (bl.flat_ids[pos_c] == flat)
            band_idx[sel[hit]] = k
            cell_idx[sel[hit]] = pos_c[hit]
        return band_idx, cell_idx

    # read one layer at query xy. NaN where outside bands or unobserved.
    def sample(self, xy: np.ndarray, name: str) -> np.ndarray:
        return self.sample_many(xy, [name])[name]

    # read many layers with one lookup. "resolution" = cell size of band hit. NaN outside bands.
    def sample_many(self, xy: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        band_idx, cell_idx = self.lookup(xy)
        out = {n: np.full(len(xy), np.nan) for n in names}
        for k, bl in enumerate(self.bands):
            sel = band_idx == k
            for n in names:
                out[n][sel] = bl.band.resolution if n == "resolution" else bl.layer(n)[cell_idx[sel]]
        return out


# bin points into sensor-centred foveated grid. points must be in `frame`, centred on sensor.
def build_foveated_grid(
    points: np.ndarray,
    bands: tuple[Band, ...] | list[Band],
    frame: str,
    points_frame: str,
    ground_percentile: float,
) -> FoveatedLayers:
    if points_frame != frame:
        raise ValueError(f"points in frame {points_frame!r} but grid in frame {frame!r}")
    validate_bands(bands)
    r = np.hypot(points[:, 0], points[:, 1])
    out = []
    for band in bands:
        spec, owned = band_lattice(band, frame)
        in_band = points[(r >= band.r_min) & (r < band.r_max)]
        dense = build_uniform_grid(in_band, spec, points_frame=frame, ground_percentile=ground_percentile)
        flat_ids = np.flatnonzero(owned.ravel())
        if dense.count.ravel()[~owned.ravel()].any():
            raise AssertionError(f"band {band}: point landed in cell outside annulus")
        out.append(
            BandLayers(
                band=band,
                spec=spec,
                flat_ids=flat_ids,
                count=dense.count.ravel()[flat_ids],
                z_min=dense.z_min.ravel()[flat_ids],
                z_max=dense.z_max.ravel()[flat_ids],
                z_mean=dense.z_mean.ravel()[flat_ids],
                z_var=dense.z_var.ravel()[flat_ids],
                ground=dense.ground.ravel()[flat_ids],
            )
        )
    return FoveatedLayers(bands=out, frame=frame, ground_percentile=ground_percentile)
