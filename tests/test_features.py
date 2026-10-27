import numpy as np
import pytest

from src.mapping.features import ground_step_edges, step_across
from src.mapping.foveated_grid import Band, build_foveated_grid
from src.mapping.uniform_grid import build_uniform_grid, centered_spec, point_to_cell

GROUND_Z = -1.73
CURB_H = 0.15
CURB_X = 5.05  # mid 0.2 m cell [5.0, 5.2) and mid 0.1 m cell [5.0, 5.1)


# seeded dense road with 15 cm curb: x >= CURB_X is sidewalk
@pytest.fixture
def curb_scene() -> np.ndarray:
    rng = np.random.default_rng(seed=11)
    n = 200000
    x = rng.uniform(2.0, 8.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = np.where(x >= CURB_X, GROUND_Z + CURB_H, GROUND_Z)
    return np.column_stack([x, y, z])


def test_step_edges_found_on_curb_line(curb_scene: np.ndarray) -> None:
    layers = build_uniform_grid(curb_scene, centered_spec(20.0, 0.1, "velo"), "velo", ground_percentile=10.0)
    xy, normals, size = ground_step_edges(layers, min_step=0.08, max_step=0.3)
    assert len(xy) > 0
    assert np.all(np.abs(xy[:, 0] - CURB_X) < 0.15)  # all edges hug curb line
    assert np.allclose(normals, [1.0, 0.0], atol=1e-6)  # low road -> high sidewalk
    assert np.allclose(size, CURB_H)


def test_flat_ground_no_edges() -> None:
    rng = np.random.default_rng(seed=12)
    pts = np.column_stack([rng.uniform(-5, 5, 50000), rng.uniform(-5, 5, 50000), np.full(50000, GROUND_Z)])
    layers = build_uniform_grid(pts, centered_spec(10.0, 0.1, "velo"), "velo", ground_percentile=10.0)
    xy, _, _ = ground_step_edges(layers, min_step=0.08, max_step=0.3)
    assert len(xy) == 0


def test_fine_cells_keep_curb_coarse_cells_blur(curb_scene: np.ndarray) -> None:
    probe = np.column_stack([np.full(20, CURB_X), np.linspace(-2.5, 2.5, 20)])
    normal = np.tile([1.0, 0.0], (20, 1))
    fine = build_foveated_grid(curb_scene, (Band(0.0, 10.0, 0.1),), "velo", "velo", ground_percentile=10.0)
    coarse = build_foveated_grid(curb_scene, (Band(0.0, 10.0, 0.2),), "velo", "velo", ground_percentile=10.0)
    fine_step = step_across(lambda q: fine.sample(q, "ground"), probe, normal, d=0.1)
    coarse_step = step_across(lambda q: coarse.sample(q, "ground"), probe, normal, d=0.1)
    # probes at x 4.95 and 5.15: 0.1 m cells split them, 0.2 m cell [5.0, 5.2) straddles curb
    assert np.allclose(fine_step, CURB_H)
    assert np.allclose(coarse_step, 0.0)
    # coarse cell still knows step is there: z max sees sidewalk, ground sees road
    ix, iy, _ = point_to_cell(np.array([[5.1, 0.0]]), coarse.bands[0].spec)
    flat = ix[0] * coarse.bands[0].spec.ny + iy[0]
    k = int(np.searchsorted(coarse.bands[0].flat_ids, flat))
    assert coarse.bands[0].z_max[k] - coarse.bands[0].ground[k] == pytest.approx(CURB_H)
