import numpy as np
import pytest

from src.mapping.uniform_grid import build_uniform_grid, centered_spec
from src.models.features import FEATURES, cell_features, feature_matrix, grid_sampler
from src.models.targets import HAZARDOUS, OBSTACLE, TRAVERSABLE, UNKNOWN, Thresholds, geometric_classes

GROUND_Z = -1.73
TH = Thresholds(obstacle_h=0.3, step=0.08, slope=0.15, rough=0.05)


# seeded test yard: flat road, 15 cm curb at x >= 4, 25 % ramp at y >= 4, 1 m box, empty patch
@pytest.fixture
def yard():
    rng = np.random.default_rng(seed=31)
    n = 400000
    x, y = rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)
    z = np.full(n, GROUND_Z)
    z[(x >= 4.0) & (y < 4.0)] += 0.15  # sidewalk
    ramp = (y >= 4.0) & (x < 4.0)
    z[ramp] += 0.25 * (y[ramp] - 4.0)
    keep = ~((x > -6) & (x < -4) & (y > -6) & (y < -4))  # empty patch, nothing seen
    pts = np.column_stack([x, y, z])[keep]
    box = np.column_stack([rng.uniform(-3, -2, 3000), rng.uniform(-3, -2, 3000), rng.uniform(GROUND_Z, GROUND_Z + 1, 3000)])
    pts = np.vstack([pts, box])
    return build_uniform_grid(pts, centered_spec(16.0, 0.1, "velo"), "velo", ground_percentile=10.0)


# true terrain of yard: flat, ramp follows slope (sidewalk counts as raised, not terrain)
def terrain(xy: np.ndarray) -> np.ndarray:
    return GROUND_Z + np.where((xy[:, 1] >= 4.0) & (xy[:, 0] < 4.0), 0.25 * (xy[:, 1] - 4.0), 0.0)


def classify(layers, xy: np.ndarray) -> np.ndarray:
    f = cell_features(grid_sampler(layers), xy, terrain(xy), np.zeros(2), 0.0)
    return geometric_classes(f, np.nan_to_num(f["count"]) > 0, TH)


def test_rule_labels_each_feature(yard) -> None:
    xy = np.array([
        [0.05, 0.05],  # flat road
        [4.05, 0.05],  # curb edge
        [6.05, 0.05],  # flat sidewalk, only 15 cm up
        [0.05, 6.05],  # steep ramp
        [-2.45, -2.45],  # box
        [-4.95, -4.95],  # empty patch
    ])
    assert classify(yard, xy).tolist() == [TRAVERSABLE, HAZARDOUS, TRAVERSABLE, HAZARDOUS, OBSTACLE, UNKNOWN]


def test_features_values(yard) -> None:
    xy = np.array([[4.05, 0.05], [0.05, 6.05], [-2.45, -2.45]])
    f = cell_features(grid_sampler(yard), xy, np.full(3, GROUND_Z), np.array([0.0, 0.0]), 0.0)
    assert f["step"][0] == pytest.approx(0.15, abs=0.01)
    assert f["slope"][1] == pytest.approx(0.25, abs=0.02)
    assert f["rel_max"][2] == pytest.approx(1.0, abs=0.1)  # random top point in cell
    assert f["range"][2] == pytest.approx(np.hypot(2.45, 2.45))
    assert np.isnan(f["weight"]).all()  # plain grid has no fused confidence
    assert feature_matrix(f).shape == (3, len(FEATURES))


def test_missing_terrain_never_obstacle(yard) -> None:
    xy = np.array([[-2.45, -2.45]])
    f = cell_features(grid_sampler(yard), xy, np.array([np.nan]), np.zeros(2), 0.0)
    assert geometric_classes(f, np.array([True]), TH).tolist() != [OBSTACLE]
