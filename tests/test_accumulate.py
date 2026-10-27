import numpy as np
import pytest

from src.mapping.uniform_grid import build_uniform_grid, point_to_cell, spec_from_bounds
from src.temporal.accumulate import accumulate_world_points

GROUND_Z = -1.73


# rigid 4x4: yaw about z, then shift
def yaw_pose(yaw: float, tx: float, ty: float, tz: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = [tx, ty, tz]
    return t


# world scene: flat road + 1 m post at (3.1, 1.1)
@pytest.fixture
def world_scene() -> np.ndarray:
    rng = np.random.default_rng(seed=3)
    ground = np.column_stack([rng.uniform(-4, 8, 3000), rng.uniform(-4, 4, 3000), np.full(3000, GROUND_Z)])
    post = np.column_stack([np.full(50, 3.1), np.full(50, 1.1), np.linspace(GROUND_Z, GROUND_Z + 1.0, 50)])
    return np.vstack([ground, post])


def test_two_views_land_on_same_world_spot(world_scene: np.ndarray) -> None:
    poses = np.stack([np.eye(4), yaw_pose(0.4, 2.5, -0.7, 0.1)])
    # each scan sees same world points in its own velodyne frame: p_velo = T^-1 p_world
    scans = []
    for t in poses:
        inv = np.linalg.inv(t)
        scans.append(world_scene @ inv[:3, :3].T + inv[:3, 3])
    acc = accumulate_world_points(scans, poses)
    assert acc.dtype == np.float32
    assert acc.shape == (2 * len(world_scene), 3)
    assert np.allclose(acc[: len(world_scene)], world_scene, atol=1e-5)
    assert np.allclose(acc[len(world_scene) :], world_scene, atol=1e-5)

    spec = spec_from_bounds(-4, 8, -4, 4, 0.5, frame="world")
    ref = build_uniform_grid(acc, spec, points_frame="world", ground_percentile=10.0)
    single = build_uniform_grid(world_scene, spec, points_frame="world", ground_percentile=10.0)
    assert np.array_equal(ref.count, 2 * single.count)
    assert np.allclose(ref.z_max, single.z_max, atol=1e-5, equal_nan=True)
    ix, iy, _ = point_to_cell(np.array([[3.1, 1.1]]), spec)
    assert ref.z_max[ix[0], iy[0]] == pytest.approx(GROUND_Z + 1.0, abs=1e-5)


def test_scan_pose_count_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        accumulate_world_points([np.zeros((3, 4))], np.stack([np.eye(4), np.eye(4)]))


def test_empty_accumulation() -> None:
    assert accumulate_world_points([], np.empty((0, 4, 4))).shape == (0, 3)


def test_spec_from_bounds_covers_extremes() -> None:
    spec = spec_from_bounds(0.6, 10.3, -2.2, 3.0, 0.2, frame="world")
    pts = np.array([[0.6, -2.2], [10.3, 3.0], [0.6, 3.0], [10.3, -2.2]])
    _, _, valid = point_to_cell(pts, spec)
    assert valid.all()
    assert spec.x_min <= 0.6 and spec.x_max > 10.3
    assert np.isclose(spec.x_min / 0.2, round(spec.x_min / 0.2))  # snapped to cell lattice
