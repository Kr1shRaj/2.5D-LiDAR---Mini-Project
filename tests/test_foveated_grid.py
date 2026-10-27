import numpy as np
import pytest

from src.mapping.foveated_grid import Band, band_lattice, build_foveated_grid, validate_bands
from src.mapping.uniform_grid import build_uniform_grid, centered_spec

GROUND_Z = -1.73
BANDS = (Band(0.0, 4.0, 0.1), Band(4.0, 10.0, 0.25), Band(10.0, 20.0, 0.5))


# seeded flat road disk out to 20 m + two 1 m posts: one near, one far
@pytest.fixture
def scene() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    n = 60000
    r = 20.0 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, GROUND_Z)])
    near_post = np.column_stack([np.full(30, 2.05), np.full(30, 1.05), np.linspace(GROUND_Z, GROUND_Z + 1.0, 30)])
    far_post = np.column_stack([np.full(30, 15.2), np.full(30, -3.3), np.linspace(GROUND_Z, GROUND_Z + 1.0, 30)])
    return np.vstack([ground, near_post, far_post])


def test_validate_bands_rejects_gap_and_bad_lattice() -> None:
    validate_bands(BANDS)
    with pytest.raises(ValueError):
        validate_bands((Band(0, 4, 0.1), Band(5, 10, 0.25)))  # gap
    with pytest.raises(ValueError):
        validate_bands((Band(0, 4, 0.3),))  # 8 m / 0.3 not whole
    with pytest.raises(ValueError):
        validate_bands(())


def test_band_lattice_owns_annulus_only() -> None:
    spec, owned = band_lattice(Band(4.0, 10.0, 0.5), frame="velo")
    assert spec.shape == (40, 40)
    assert not owned[19, 19] and not owned[20, 20]  # cells at sensor sit inside inner hole
    assert owned[0, 20]  # cell at x -10..-9.5 touches outer ring
    assert not owned[0, 0]  # far corner is past 10 m
    # owned count lies between annulus area / cell area and that plus edge cells
    ring_cells = np.pi * (10.0**2 - 4.0**2) / 0.25
    assert ring_cells < owned.sum() < ring_cells * 1.3


def test_every_point_counted_once(scene: np.ndarray) -> None:
    fov = build_foveated_grid(scene, BANDS, frame="velo", points_frame="velo", ground_percentile=10.0)
    assert sum(int(b.count.sum()) for b in fov.bands) == len(scene)


def test_point_on_band_edge_goes_outward() -> None:
    pts = np.array([[4.0, 0.0, GROUND_Z], [3.999, 0.0, GROUND_Z], [20.0, 0.0, GROUND_Z]])
    fov = build_foveated_grid(pts, BANDS, frame="velo", points_frame="velo", ground_percentile=10.0)
    assert [int(b.count.sum()) for b in fov.bands] == [1, 1, 0]  # r = 20 is outside last band


def test_single_band_matches_uniform_grid(scene: np.ndarray) -> None:
    fov = build_foveated_grid(scene, (Band(0.0, 20.0, 0.5),), frame="velo", points_frame="velo", ground_percentile=10.0)
    uni = build_uniform_grid(scene, centered_spec(40.0, 0.5, frame="velo"), points_frame="velo", ground_percentile=10.0)
    for name in ("count", "z_min", "z_max", "z_mean", "z_var", "ground"):
        assert np.allclose(fov.bands[0].dense(name)[uni.observed], getattr(uni, name)[uni.observed])


def test_flat_ground_and_posts_kept(scene: np.ndarray) -> None:
    fov = build_foveated_grid(scene, BANDS, frame="velo", points_frame="velo", ground_percentile=10.0)
    for b in fov.bands:
        # fine post cell may hold post points only, so skip cells with height
        flat = b.observed & np.isclose(b.z_max, GROUND_Z)
        assert flat.sum() > 0.9 * b.observed.sum()
        assert np.allclose(b.ground[flat], GROUND_Z)
    tops = fov.sample(np.array([[2.05, 1.05], [15.2, -3.3]]), "z_max")
    assert tops == pytest.approx([GROUND_Z + 1.0, GROUND_Z + 1.0])
    road = fov.sample(np.array([[5.0, -5.0], [7.0, 7.0], [-12.0, -12.0]]), "z_max")  # 0.1 m cells too sparse to probe
    assert road == pytest.approx([GROUND_Z] * 3)


def test_unobserved_cells_nan_and_outside_minus_one() -> None:
    pts = np.array([[1.0, 1.0, GROUND_Z]])
    fov = build_foveated_grid(pts, BANDS, frame="velo", points_frame="velo", ground_percentile=10.0)
    for b in fov.bands:
        empty = ~b.observed
        assert np.all(np.isnan(b.ground[empty])) and np.all(np.isnan(b.z_max[empty]))
    band_idx, cell_idx = fov.lookup(np.array([[30.0, 0.0], [1.0, 1.0]]))
    assert band_idx.tolist()[0] == -1 and cell_idx.tolist()[0] == -1
    assert band_idx.tolist()[1] == 0
    assert np.isnan(fov.sample(np.array([[30.0, 0.0], [5.0, 5.0]]), "ground")).all()


def test_foveated_stores_fewer_cells_than_fine_uniform(scene: np.ndarray) -> None:
    fov = build_foveated_grid(scene, BANDS, frame="velo", points_frame="velo", ground_percentile=10.0)
    fine = build_uniform_grid(scene, centered_spec(40.0, 0.1, frame="velo"), points_frame="velo", ground_percentile=10.0)
    assert fov.n_cells < fine.count.size / 10
    assert fov.nbytes < fine.nbytes / 10


def test_frame_mismatch_rejected(scene: np.ndarray) -> None:
    with pytest.raises(ValueError, match="frame"):
        build_foveated_grid(scene, BANDS, frame="velo", points_frame="world", ground_percentile=10.0)
