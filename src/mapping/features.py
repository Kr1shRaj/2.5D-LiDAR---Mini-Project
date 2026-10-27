import numpy as np

from src.mapping.uniform_grid import GridLayers, cell_centers


# find curb-like ground steps. central diff over 2 cells, step size in [min_step, max_step] metres.
# returns edge cell centre xy, unit normal pointing low -> high, step size along normal.
def ground_step_edges(
    layers: GridLayers,
    min_step: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < min_step < max_step:
        raise ValueError(f"need 0 < min_step < max_step, got {min_step}, {max_step}")
    g = layers.ground
    gx = np.full(g.shape, np.nan)
    gy = np.full(g.shape, np.nan)
    gx[1:-1, :] = g[2:, :] - g[:-2, :]
    gy[:, 1:-1] = g[:, 2:] - g[:, :-2]
    mag = np.hypot(gx, gy)
    edge = np.isfinite(mag) & (mag >= min_step) & (mag <= max_step)
    xy = cell_centers(layers.spec)[edge]
    normals = np.column_stack([gx[edge], gy[edge]]) / mag[edge][:, None]
    return xy, normals, mag[edge]


# height change across edge: layer value at xy + d*n minus value at xy - d*n
def step_across(sample_fn, xy: np.ndarray, normals: np.ndarray, d: float) -> np.ndarray:
    return sample_fn(xy + d * normals) - sample_fn(xy - d * normals)
