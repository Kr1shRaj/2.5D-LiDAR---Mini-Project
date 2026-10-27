import numpy as np
import pytest

from src.mapping.uniform_grid import (
    GridSpec,
    build_uniform_grid,
    cell_centers,
    centered_spec,
    point_to_cell,
)
from src.viz.grid_viz import plot_elevation_and_roughness

GROUND_Z = -1.73  # kitti hdl-64e road height in velodyne frame
BOX_TOP_Z = -0.73  # 1 m tall box
RES = 0.5
STEP = 0.1  # 25 points per 0.5 m cell


# 10 m x 10 m velodyne frame grid, 0.5 m cells, centred on sensor
@pytest.fixture
def spec() -> GridSpec:
    return centered_spec(extent=10.0, resolution=RES, frame="velo")


# regular xy lattice at cell-safe offsets inside [lo, hi)
def lattice(x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> np.ndarray:
    xs = np.arange(x_lo + STEP / 2, x_hi, STEP)
    ys = np.arange(y_lo + STEP / 2, y_hi, STEP)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


# flat road, 1 m box at x[1,2) y[-1,0), wall face at x=2, empty hole at x[-4,-3) y[3,4)
@pytest.fixture
def scene() -> np.ndarray:
    ground_xy = lattice(-5.0, 5.0, -5.0, 5.0)
    in_box = (ground_xy[:, 0] >= 1.0) & (ground_xy[:, 0] < 2.0) & (ground_xy[:, 1] >= -1.0) & (ground_xy[:, 1] < 0.0)
    in_hole = (ground_xy[:, 0] >= -4.0) & (ground_xy[:, 0] < -3.0) & (ground_xy[:, 1] >= 3.0) & (ground_xy[:, 1] < 4.0)
    ground_xy = ground_xy[~in_box & ~in_hole]
    ground = np.column_stack([ground_xy, np.full(len(ground_xy), GROUND_Z)])

    top_xy = lattice(1.0, 2.0, -1.0, 0.0)
    top = np.column_stack([top_xy, np.full(len(top_xy), BOX_TOP_Z)])

    wall_y = np.arange(-1.0 + STEP / 2, 0.0, STEP)
    wall_z = np.linspace(GROUND_Z, BOX_TOP_Z, 11)
    wy, wz = np.meshgrid(wall_y, wall_z, indexing="ij")
    wall = np.column_stack([np.full(wy.size, 2.0), wy.ravel(), wz.ravel()])

    return np.vstack([ground, top, wall])


# --- binning and coordinate mapping ---


def test_spec_bounds_centered(spec: GridSpec) -> None:
    assert spec.shape == (20, 20)
    assert spec.x_min == pytest.approx(-5.0)
    assert spec.x_max == pytest.approx(5.0)
    assert spec.y_min == pytest.approx(-5.0)
    assert spec.y_max == pytest.approx(5.0)


def test_point_to_cell_known_points(spec: GridSpec) -> None:
    xy = np.array(
        [
            [-5.0, -5.0],  # lower corner -> first cell
            [-4.99, 4.99],  # top left
            [0.0, 0.0],  # sensor origin -> cell (10, 10)
            [-0.01, -0.01],  # just below origin -> (9, 9)
            [1.25, -0.75],  # inside box cell
            [4.999, 4.999],  # last cell
        ]
    )
    ix, iy, valid = point_to_cell(xy, spec)
    assert valid.all()
    assert list(zip(ix.tolist(), iy.tolist())) == [(0, 0), (0, 19), (10, 10), (9, 9), (12, 8), (19, 19)]


def test_point_to_cell_out_of_bounds(spec: GridSpec) -> None:
    xy = np.array([[5.0, 0.0], [0.0, 5.0], [-5.01, 0.0], [0.0, -100.0]])
    _, _, valid = point_to_cell(xy, spec)
    assert not valid.any()


def test_cell_centers_roundtrip(spec: GridSpec) -> None:
    centers = cell_centers(spec)
    ix, iy, valid = point_to_cell(centers.reshape(-1, 2), spec)
    assert valid.all()
    expect_ix, expect_iy = np.meshgrid(np.arange(spec.nx), np.arange(spec.ny), indexing="ij")
    assert np.array_equal(ix, expect_ix.ravel())
    assert np.array_equal(iy, expect_iy.ravel())
    assert centers[10, 10] == pytest.approx([0.25, 0.25])


def test_layer_axis_order_x_then_y(spec: GridSpec) -> None:
    pts = np.array([[4.75, -4.75, 0.5]])  # far +x, far -y
    layers = build_uniform_grid(pts, spec, points_frame="velo", ground_percentile=10.0)
    assert layers.count[19, 0] == 1
    assert layers.count.sum() == 1


def test_counts_conserve_points(spec: GridSpec, scene: np.ndarray) -> None:
    outside = np.array([[50.0, 0.0, GROUND_Z], [0.0, -6.0, GROUND_Z], [np.nan, 0.0, 0.0]])
    layers = build_uniform_grid(np.vstack([scene, outside]), spec, points_frame="velo", ground_percentile=10.0)
    assert layers.count.sum() == len(scene)


# --- height metrics ---


def test_flat_ground_cells(spec: GridSpec, scene: np.ndarray) -> None:
    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    ground_cell = (layers.count[0, 0], layers.z_mean[0, 0], layers.z_var[0, 0])
    assert ground_cell[0] == 25
    assert ground_cell[1] == pytest.approx(GROUND_Z)
    assert ground_cell[2] == pytest.approx(0.0, abs=1e-12)
    # every pure ground cell sits at sensor ground, not z = 0
    pure_ground = np.isclose(layers.z_max, GROUND_Z)
    assert pure_ground.sum() == 400 - 4 - 4 - 2  # minus hole, box, wall cells
    assert np.allclose(layers.ground[pure_ground], GROUND_Z)


def test_box_top_cells(spec: GridSpec, scene: np.ndarray) -> None:
    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    box = (slice(12, 14), slice(8, 10))
    assert np.all(layers.count[box] == 25)
    assert np.allclose(layers.z_max[box], BOX_TOP_Z)
    assert np.allclose(layers.z_min[box], BOX_TOP_Z)
    assert np.allclose(layers.z_var[box], 0.0)
    assert np.allclose(layers.ground[box], BOX_TOP_Z)


def test_wall_edge_cells_show_step(spec: GridSpec, scene: np.ndarray) -> None:
    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    edge = (14, slice(8, 10))  # cells x[2.0, 2.5): ground + wall face
    assert np.allclose(layers.z_min[edge], GROUND_Z)
    assert np.allclose(layers.z_max[edge], BOX_TOP_Z)
    assert np.all(layers.z_var[edge] > 0.01)
    assert np.allclose(layers.ground[edge], GROUND_Z)
    assert np.all(layers.z_max[edge] - layers.z_min[edge] == pytest.approx(1.0))


def test_metrics_match_numpy_per_cell(spec: GridSpec) -> None:
    rng = np.random.default_rng(seed=0)
    pts = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, 4000),
            rng.uniform(-5.0, 5.0, 4000),
            GROUND_Z + rng.normal(0.0, 0.3, 4000),
        ]
    )
    pct = 7.5
    layers = build_uniform_grid(pts, spec, points_frame="velo", ground_percentile=pct)
    ix, iy, _ = point_to_cell(pts[:, :2], spec)
    for cx, cy in [(0, 0), (10, 10), (19, 3), (7, 15)]:
        z = pts[(ix == cx) & (iy == cy), 2]
        assert layers.count[cx, cy] == len(z)
        assert layers.z_mean[cx, cy] == pytest.approx(z.mean())
        assert layers.z_var[cx, cy] == pytest.approx(z.var())
        assert layers.z_min[cx, cy] == pytest.approx(z.min())
        assert layers.z_max[cx, cy] == pytest.approx(z.max())
        assert layers.ground[cx, cy] == pytest.approx(np.percentile(z, pct))


# --- unobserved space ---


def test_unobserved_cells_are_nan_not_zero(spec: GridSpec, scene: np.ndarray) -> None:
    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    hole = (slice(2, 4), slice(16, 18))
    assert np.all(layers.count[hole] == 0)
    assert not layers.observed[hole].any()
    for layer in (layers.z_min, layers.z_max, layers.z_mean, layers.z_var, layers.ground):
        assert np.all(np.isnan(layer[hole]))
    assert layers.observed.sum() == 400 - 4


def test_empty_cloud_all_unknown(spec: GridSpec) -> None:
    layers = build_uniform_grid(np.empty((0, 3)), spec, points_frame="velo", ground_percentile=10.0)
    assert layers.count.sum() == 0
    assert np.all(np.isnan(layers.z_mean))
    assert np.all(np.isnan(layers.ground))


# --- guards ---


def test_frame_mismatch_rejected(spec: GridSpec, scene: np.ndarray) -> None:
    with pytest.raises(ValueError, match="frame"):
        build_uniform_grid(scene, spec, points_frame="world", ground_percentile=10.0)


def test_bad_percentile_rejected(spec: GridSpec, scene: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=150.0)


def test_extent_must_fit_cells() -> None:
    with pytest.raises(ValueError):
        centered_spec(extent=10.0, resolution=0.3, frame="velo")


# --- viz smoke ---


def test_viz_writes_png(spec: GridSpec, scene: np.ndarray, tmp_path) -> None:
    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    out = tmp_path / "grid.png"
    fig = plot_elevation_and_roughness(layers, out)
    assert out.stat().st_size > 0
    assert len(fig.axes) >= 2


def test_sample_layer_reads_cells_and_nan_outside(spec: GridSpec, scene: np.ndarray) -> None:
    from src.mapping.uniform_grid import sample_layer

    layers = build_uniform_grid(scene, spec, points_frame="velo", ground_percentile=10.0)
    got = sample_layer(layers, np.array([[1.25, -0.75], [0.1, 0.1], [-3.75, 3.25], [50.0, 0.0]]), "z_max")
    assert got[:2] == pytest.approx([BOX_TOP_Z, GROUND_Z])
    assert np.isnan(got[2]) and np.isnan(got[3])  # hole + outside grid
